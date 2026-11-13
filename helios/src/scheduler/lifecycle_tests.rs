/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use std::collections::HashSet;

type Projection = [u64; 14];
type Action = [u64; 3];

const MAX_SUBMISSIONS: u64 = 4;
const MAX_DEPTH: usize = 12;

struct Fixture {
    scheduler: RequestScheduler,
    fence_origin: u64,
    last_plan: Option<ScheduledPlan>,
}

fn request_spec(slot: usize, generation: u64) -> RequestSpec {
    let mut spec = tests::spec(63, 32);
    spec.prompt[0] = 100 + slot as u32 + generation as u32 * 10;
    spec
}

fn completion_records(scheduler: &RequestScheduler, fence: Fence) -> Vec<CompletionRecord> {
    let submission = &scheduler.submissions[&fence];
    let mut records = vec![CompletionRecord::default(); 2];
    for (row, lane) in submission.lanes.iter().enumerate() {
        let Ok(request) = scheduler.request(lane.request) else {
            continue;
        };
        let kind = submission.lane_kind(row);
        let transfer = matches!(
            kind,
            GraphKind::Init | GraphKind::Restore | GraphKind::Snapshot
        );
        assert!(!matches!(
            kind,
            GraphKind::Speculative | GraphKind::Mixed | GraphKind::Vision
        ));
        let consumed = if transfer {
            request.consumed_tokens
        } else {
            lane.start_position + lane.tokens as u64
        };
        let emits = !transfer && (!kind.is_prefill() || consumed >= request.prompt_tokens as u64);
        records[lane.request.slot as usize] = CompletionRecord {
            generation: lane.request.generation as i64,
            submission_sequence: fence.0 as i64,
            status: 1,
            consumed_tokens: consumed as i64,
            generated_tokens: i64::from(request.generated_tokens) + i64::from(emits),
            token_id: if emits { 99 } else { -1 },
            ..Default::default()
        };
    }
    records
}

impl Fixture {
    fn new() -> Self {
        let mut scheduler = tests::scheduler(2, 64, 16);
        for slot in 0..2 {
            assert_eq!(
                scheduler.admit(request_spec(slot, 1)).unwrap().slot as usize,
                slot
            );
        }
        for _ in 0..16 {
            if scheduler
                .slots
                .iter()
                .all(|slot| slot.request.as_ref().unwrap().phase == RequestPhase::Decode)
            {
                let fence_origin = scheduler.submitted.0;
                assert_eq!(scheduler.completed.0, fence_origin);
                return Self {
                    scheduler,
                    fence_origin,
                    last_plan: None,
                };
            }
            let plan = scheduler
                .plan()
                .expect("warmup must reach decode through real scheduler plans");
            let fence = Fence(scheduler.submitted.0 + 1);
            scheduler.commit(&plan, fence).unwrap();
            let records = completion_records(&scheduler, fence);
            scheduler
                .complete(fence, &records, Default::default())
                .unwrap();
        }
        panic!("scheduler warmup did not reach two decode requests");
    }

    fn logical_fence(&self, fence: Fence) -> u64 {
        fence.0.saturating_sub(self.fence_origin)
    }

    fn projection(&self) -> Projection {
        let s = &self.scheduler;
        let mut result = [0; 14];
        result[0] = self.logical_fence(s.submitted);
        result[1] = self.logical_fence(s.completed);
        for (index, slot) in s.slots.iter().enumerate() {
            result[2 + index] = if slot.request.is_some() {
                1
            } else if slot.retiring {
                2
            } else {
                0
            };
            result[4 + index] = slot.generation;
            result[6 + index] = slot
                .request
                .as_ref()
                .map(|request| request.last_use)
                .or_else(|| {
                    s.retired
                        .iter()
                        .find(|retired| retired.slot == index)
                        .map(|retired| retired.through)
                })
                .map_or(0, |fence| self.logical_fence(fence));
        }
        for (index, (fence, submission)) in s.submissions.iter().enumerate() {
            assert!(index < 2);
            result[8 + index] = self.logical_fence(*fence);
            result[10 + index] = submission
                .lanes
                .iter()
                .fold(0, |mask, lane| mask | (1 << lane.request.slot));
            result[12 + index] = u64::from(submission.requires_feedback);
        }
        result
    }

    fn state_key(&self) -> String {
        let s = &self.scheduler;
        let live = s
            .slots
            .iter()
            .map(|slot| {
                slot.request.as_ref().map(|request| {
                    (
                        request.handle,
                        request.phase,
                        request.epoch,
                        request.consumed_tokens,
                        request.generated_tokens,
                        request.scheduled_consumed,
                        request.pending,
                        (
                            request.private_pages.clone(),
                            request.reserved_pages.clone(),
                            request.terminal_snapshot,
                            request.prompt_snapshot,
                            request.anchor_guard,
                            request.checkpoint_capture,
                            request.checkpoint_copy_ns,
                            request.checkpoint_accounting(),
                            request.prefix,
                            request.admission_order,
                        ),
                    )
                })
            })
            .collect::<Vec<_>>();
        let submitted = s
            .submissions
            .iter()
            .map(|(fence, submission)| (*fence, &submission.step))
            .collect::<Vec<_>>();
        let retired = s
            .retired
            .iter()
            .map(|retired| {
                (
                    retired.slot,
                    retired.through,
                    &retired.pages,
                    retired.snapshots,
                )
            })
            .collect::<Vec<_>>();
        format!(
            "{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}",
            self.projection(),
            live,
            submitted,
            retired,
            s.stats(),
            s.free_pages,
            s.free_snapshots
        )
    }

    fn assert_ownership(&self) {
        let s = &self.scheduler;
        for (fence, submission) in &s.submissions {
            for lane in &submission.lanes {
                let slot = &s.slots[lane.request.slot as usize];
                assert_eq!(
                    slot.generation, lane.request.generation,
                    "queued request generation was reused before completion"
                );
                assert!(
                    slot.request
                        .as_ref()
                        .is_some_and(|request| request.handle == lane.request)
                        || s.retired.iter().any(|retired| {
                            retired.slot == lane.request.slot as usize && retired.through >= *fence
                        }),
                    "queued request lost ownership before completion"
                );
            }
        }
        let mut pages = s.free_pages.iter().copied().collect::<HashSet<_>>();
        let mut snapshots = s.free_snapshots.iter().copied().collect::<HashSet<_>>();
        for (index, slot) in s.slots.iter().enumerate() {
            assert!(!(slot.retiring && slot.request.is_some()));
            assert_eq!(
                slot.retiring,
                s.retired.iter().any(|retired| retired.slot == index)
            );
            if let Some(request) = &slot.request {
                assert_eq!(request.handle.generation, slot.generation);
                assert_eq!(
                    request.pending,
                    s.submissions
                        .values()
                        .filter(|submission| submission
                            .lanes
                            .iter()
                            .any(|lane| lane.request == request.handle))
                        .count()
                );
                for page in request.private_pages.iter().chain(&request.reserved_pages) {
                    assert!(
                        pages.insert(*page),
                        "live private page aliases free or owned storage"
                    );
                }
                for snapshot in [
                    request.anchor_guard,
                    request.terminal_snapshot,
                    request.message_snapshot,
                    request.prompt_snapshot,
                    request.window_snapshot,
                    request.checkpoint_capture.map(|capture| capture.snapshot),
                ]
                .into_iter()
                .flatten()
                {
                    assert!(
                        snapshots.insert(snapshot),
                        "live snapshot aliases free or owned storage"
                    );
                }
            }
        }
        for retired in &s.retired {
            assert!(
                retired.through > s.completed,
                "completed retirement still reserves a slot"
            );
            for page in &retired.pages {
                assert!(
                    pages.insert(*page),
                    "retired page became reusable before its fence"
                );
            }
            for snapshot in retired.snapshots.into_iter().flatten() {
                assert!(
                    snapshots.insert(snapshot),
                    "retired snapshot became reusable before its fence"
                );
            }
        }
    }

    fn actions(&self) -> Vec<Action> {
        let s = &self.scheduler;
        let mut actions = vec![[5, 0, 0]];
        if let Some(slot) = s
            .slots
            .iter()
            .position(|slot| slot.request.is_none() && !slot.retiring)
            && s.slots[slot].generation < 2
        {
            actions.push([1, slot as u64, 0]);
        }
        if self.projection()[0] < MAX_SUBMISSIONS
            && let Some(plan) = s.plan()
        {
            actions.push([
                2,
                plan.lanes
                    .iter()
                    .fold(0, |mask, lane| mask | (1 << lane.request.slot)),
                u64::from(s.plan_requires_feedback(&plan)),
            ]);
        }
        if !s.submissions.is_empty() {
            actions.push([3, 0, 0]);
        }
        for (index, slot) in s.slots.iter().enumerate() {
            if slot.request.is_some() {
                actions.push([4, index as u64, 0]);
            }
        }
        if s.slots
            .iter()
            .any(|slot| slot.request.is_some() || slot.retiring)
            || !s.submissions.is_empty()
        {
            actions.push([6, 0, 0]);
        }
        actions
    }

    fn apply(&mut self, action: Action) {
        let before = self.state_key();
        let s = &mut self.scheduler;
        match action[0] {
            1 => {
                let slot = action[1] as usize;
                let handle = s
                    .admit(request_spec(slot, s.slots[slot].generation + 1))
                    .unwrap();
                assert_eq!(handle.slot as usize, slot);
            }
            2 => {
                let plan = s.plan().unwrap();
                assert_eq!(
                    action[1],
                    plan.lanes
                        .iter()
                        .fold(0, |mask, lane| mask | (1 << lane.request.slot))
                );
                assert_eq!(action[2], u64::from(s.plan_requires_feedback(&plan)));
                s.commit(&plan, Fence(s.submitted.0 + 1)).unwrap();
                self.last_plan = Some(plan);
            }
            3 => {
                let fence = *s.submissions.first_key_value().unwrap().0;
                let records = completion_records(s, fence);
                let protected = s
                    .retired
                    .iter()
                    .filter(|retired| retired.through > fence)
                    .map(|retired| {
                        (
                            retired.slot,
                            retired.through,
                            retired.pages.clone(),
                            retired.snapshots,
                        )
                    })
                    .collect::<Vec<_>>();
                s.complete(fence, &records, Default::default()).unwrap();
                for (slot, through, pages, snapshots) in protected {
                    let retired = s
                        .retired
                        .iter()
                        .find(|retired| retired.slot == slot)
                        .expect(
                            "first completion released a slot still referenced by a later frame",
                        );
                    assert_eq!(
                        (retired.through, &retired.pages, retired.snapshots),
                        (through, &pages, snapshots)
                    );
                    assert!(
                        pages.iter().all(|page| !s.free_pages.contains(page)),
                        "first completion released pages referenced by a later frame"
                    );
                    assert!(
                        snapshots
                            .into_iter()
                            .flatten()
                            .all(|snapshot| !s.free_snapshots.contains(&snapshot)),
                        "first completion released snapshots referenced by a later frame"
                    );
                }
            }
            4 => {
                let slot = action[1] as usize;
                let request = s.slots[slot].request.as_ref().unwrap();
                let handle = request.handle;
                let through = request.last_use;
                let pages = request
                    .private_pages
                    .iter()
                    .chain(&request.reserved_pages)
                    .copied()
                    .collect::<Vec<_>>();
                let snapshots = [
                    request.terminal_snapshot,
                    request.message_snapshot,
                    request.prompt_snapshot,
                    request.window_snapshot,
                    request.checkpoint_capture.map(|capture| capture.snapshot),
                ];
                s.cancel(handle).unwrap();
                if through > s.completed {
                    let retired = s
                        .retired
                        .iter()
                        .find(|retired| retired.slot == slot)
                        .expect("cancel released a slot referenced by queued work");
                    assert_eq!(
                        retired.through, through,
                        "cancel shortened the last-use fence"
                    );
                    assert_eq!(retired.pages, pages, "cancel lost private page ownership");
                    assert_eq!(
                        retired.snapshots, snapshots,
                        "cancel lost snapshot ownership"
                    );
                    assert!(
                        pages.iter().all(|page| !s.free_pages.contains(page)),
                        "cancel released pages referenced by queued work"
                    );
                    assert!(
                        snapshots
                            .into_iter()
                            .flatten()
                            .all(|snapshot| !s.free_snapshots.contains(&snapshot)),
                        "cancel released snapshots referenced by queued work"
                    );
                }
            }
            5 => {
                assert!(
                    s.release_after_shutdown(&Shutdown {
                        drained: false,
                        error: Some("uncertain stream drain".into())
                    })
                    .is_err(),
                    "unconfirmed drain accepted resource release"
                );
                assert_eq!(
                    self.state_key(),
                    before,
                    "unconfirmed drain released ownership"
                );
            }
            6 => {
                let mut pages = Vec::new();
                let mut snapshots = Vec::new();
                for request in s.slots.iter().filter_map(|slot| slot.request.as_ref()) {
                    pages.extend(
                        request
                            .private_pages
                            .iter()
                            .chain(&request.reserved_pages)
                            .copied(),
                    );
                    snapshots.extend(
                        [
                            request.terminal_snapshot,
                            request.message_snapshot,
                            request.prompt_snapshot,
                            request.window_snapshot,
                            request.checkpoint_capture.map(|capture| capture.snapshot),
                            request.anchor_guard,
                        ]
                        .into_iter()
                        .flatten(),
                    );
                }
                for retired in &s.retired {
                    pages.extend(retired.pages.iter().copied());
                    snapshots.extend(retired.snapshots.into_iter().flatten());
                }
                s.release_after_shutdown(&Shutdown {
                    drained: true,
                    error: Some("executor failure after drain".into()),
                })
                .unwrap();
                assert!(
                    pages.iter().all(|page| s.free_pages.contains(page)),
                    "confirmed drain lost private pages"
                );
                assert!(
                    snapshots
                        .iter()
                        .all(|snapshot| s.free_snapshots.contains(snapshot)),
                    "confirmed drain lost private snapshots"
                );
            }
            _ => unreachable!(),
        }
        self.assert_ownership();
    }
}

#[derive(Default, Debug)]
struct Coverage {
    two_pending: bool,
    cancel_two_pending: bool,
    first_completion_retains_slot: bool,
    stale_handle: bool,
    stale_plan: bool,
    reused_generation_plan: bool,
    stale_completion: bool,
    partially_valid_completion: bool,
    duplicate_completion: bool,
    out_of_order: bool,
    uncertain_drain: bool,
    feedback_barrier: bool,
    full_window_rejection: bool,
    feedback_rejection: bool,
}

fn check_rejections(fixture: &mut Fixture, coverage: &mut Coverage) {
    let before = fixture.state_key();
    let s = &mut fixture.scheduler;
    if let Some(plan) = &fixture.last_plan {
        let full = s.submissions.len() == 2;
        let feedback = s
            .submissions
            .values()
            .any(|submission| submission.requires_feedback);
        if full || feedback {
            let error = s
                .commit(plan, Fence(s.submitted.0 + 1))
                .expect_err("commit accepted work beyond the window or feedback barrier");
            assert_eq!(
                error.to_string(),
                "submission window or feedback barrier is full",
                "commit bypassed the submission window or feedback barrier"
            );
            coverage.full_window_rejection |= full;
            coverage.feedback_rejection |= feedback;
        }
    }
    if s.submissions.len() == 2 {
        coverage.two_pending = true;
        let fence = *s.submissions.last_key_value().unwrap().0;
        let records = completion_records(s, fence);
        assert!(
            s.complete(fence, &records, Default::default()).is_err(),
            "out-of-order completion accepted"
        );
        coverage.out_of_order = true;
    }
    if let Some((fence, _)) = s.submissions.first_key_value() {
        let fence = *fence;
        let mut records = completion_records(s, fence);
        coverage.partially_valid_completion |=
            records.iter().filter(|record| record.status != 0).count() > 1;
        if let Some(record) = records.iter_mut().rfind(|record| record.status != 0) {
            record.generation += 1;
            assert!(
                s.complete(fence, &records, Default::default()).is_err(),
                "stale generation completion accepted"
            );
            coverage.stale_completion = true;
        }
    }
    if s.completed.0 > fixture.fence_origin {
        assert!(
            s.complete(
                s.completed,
                &[CompletionRecord::default(), CompletionRecord::default()],
                Default::default()
            )
            .is_err(),
            "duplicate completion accepted"
        );
        coverage.duplicate_completion = true;
    }
    let stale_handles = s
        .slots
        .iter()
        .enumerate()
        .filter(|(_, slot)| slot.generation > 1)
        .map(|(index, slot)| RequestHandle {
            slot: index as u32,
            generation: slot.generation - 1,
        })
        .collect::<Vec<_>>();
    for stale in stale_handles {
        assert!(
            s.request(stale).is_err(),
            "reused slot accepted an old request handle"
        );
        assert!(
            s.cancel(stale).is_err(),
            "old request handle cancelled a reused slot"
        );
        coverage.stale_handle = true;
    }
    assert_eq!(
        fixture.state_key(),
        before,
        "rejected completion mutated scheduler state"
    );
}

#[test]
fn scheduler_matches_lifecycle_model() {
    let mut pending = VecDeque::from([Vec::<Action>::new()]);
    let mut visited = HashSet::new();
    let mut edges = BTreeSet::new();
    let mut coverage = Coverage::default();
    let mut reused = Fixture::new();
    let stale_plan = reused.scheduler.plan().unwrap();
    let handle = stale_plan.lanes[0].request;
    reused.apply([4, u64::from(handle.slot), 0]);
    reused.apply([1, u64::from(handle.slot), 0]);
    let replacement = reused.scheduler.slots[handle.slot as usize]
        .request
        .as_ref()
        .unwrap()
        .handle;
    assert!(replacement.generation > handle.generation);
    let before = reused.state_key();
    assert!(
        reused
            .scheduler
            .commit(&stale_plan, Fence(reused.scheduler.submitted.0 + 1))
            .is_err(),
        "old generation execution plan accepted after slot reuse"
    );
    assert_eq!(
        reused.state_key(),
        before,
        "stale plan changed a reused slot"
    );
    coverage.reused_generation_plan = true;
    while let Some(path) = pending.pop_front() {
        let mut fixture = Fixture::new();
        for action in &path {
            fixture.apply(*action);
        }
        if !visited.insert(fixture.state_key()) {
            continue;
        }
        fixture.assert_ownership();
        check_rejections(&mut fixture, &mut coverage);
        if path.len() == MAX_DEPTH {
            continue;
        }
        let before = fixture.projection();
        for action in fixture.actions() {
            let mut candidate = Fixture::new();
            for previous in &path {
                candidate.apply(*previous);
            }
            let stale_plan = (action[0] == 4)
                .then(|| candidate.scheduler.plan())
                .flatten()
                .filter(|plan| {
                    plan.lanes
                        .iter()
                        .any(|lane| u64::from(lane.request.slot) == action[1])
                });
            if action[0] == 4 && candidate.scheduler.submissions.len() == 2 {
                coverage.cancel_two_pending = true;
            }
            candidate.apply(action);
            let after = candidate.projection();
            if let Some(plan) = stale_plan {
                let state = candidate.state_key();
                let fence = Fence(candidate.scheduler.submitted.0 + 1);
                assert!(
                    candidate.scheduler.commit(&plan, fence).is_err(),
                    "cancelled request plan accepted"
                );
                assert_eq!(candidate.state_key(), state);
                coverage.stale_plan = true;
            }
            if action[0] == 3 && before[2..4].contains(&2) && after[2..4].contains(&2) {
                coverage.first_completion_retains_slot = true;
            }
            coverage.uncertain_drain |= action[0] == 5 && before[8] != 0;
            coverage.feedback_barrier |= action[0] == 2 && action[2] == 1;
            edges.insert((before, action, after));
            let mut successor = path.clone();
            successor.push(action);
            pending.push_back(successor);
        }
    }
    assert!(
        coverage.two_pending
            && coverage.cancel_two_pending
            && coverage.first_completion_retains_slot,
        "missing deferred-retirement coverage: {coverage:?}"
    );
    assert!(
        coverage.stale_handle
            && coverage.stale_plan
            && coverage.reused_generation_plan
            && coverage.stale_completion
            && coverage.partially_valid_completion
            && coverage.duplicate_completion
            && coverage.out_of_order,
        "missing rejection coverage: {coverage:?}"
    );
    assert!(
        coverage.uncertain_drain
            && coverage.feedback_barrier
            && coverage.full_window_rejection
            && coverage.feedback_rejection,
        "missing drain or barrier coverage: {coverage:?}"
    );
    for (before, action, after) in &edges {
        println!("LIFECYCLE_EDGE [{before:?}, {action:?}, {after:?}]");
    }
    println!(
        "LIFECYCLE_COVERAGE states={} edges={} depth={MAX_DEPTH} {coverage:?}",
        visited.len(),
        edges.len()
    );
}
