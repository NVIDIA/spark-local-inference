/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;

struct Observation {
    origin: u64,
    owners: [u64; 3],
    uses: [u64; 3],
    published: Vec<(CheckpointId, u64)>,
}

impl Observation {
    fn project(&mut self, scheduler: &RequestScheduler) -> [u64; 15] {
        let mut result = [0; 15];
        result[0] = scheduler.submitted.0 - self.origin;
        result[1] = scheduler.completed.0 - self.origin;
        for slot in &scheduler.slots {
            if let Some(request) = &slot.request {
                result[2] |= 1 << request.handle.slot;
                if let Some(prefix) = request.prefix {
                    for id in scheduler.cache.chain(prefix).unwrap() {
                        if !self.published.iter().any(|(known, _)| *known == id) {
                            self.published.push((id, request.handle.slot as u64 + 1));
                        }
                    }
                }
                let armed = request.checkpoint_capture.map(|capture| capture.snapshot);
                let copying = request.boundary_capture.map(|capture| capture.snapshot);
                for (snapshot, phase) in request
                    .anchor_guard
                    .map(|id| (id, 1))
                    .into_iter()
                    .chain(armed.map(|id| (id, 2)))
                    .chain(copying.map(|id| (id, if request.pending > 0 { 3 } else { 2 })))
                {
                    let row = snapshot.0 as usize;
                    self.owners[row] = request.handle.slot as u64 + 1;
                    let fence = if phase == 1 {
                        0
                    } else {
                        request.last_use.0 - self.origin
                    };
                    if phase != 1 {
                        self.uses[row] = self.uses[row].max(fence);
                    }
                    result[3 + row * 4..7 + row * 4].copy_from_slice(&[
                        phase,
                        self.owners[row],
                        fence,
                        self.uses[row],
                    ]);
                }
            }
        }
        for (id, owner) in &self.published {
            if let Ok(checkpoint) = scheduler.cache.checkpoint(*id) {
                let row = checkpoint.snapshot().0 as usize;
                result[3 + row * 4..7 + row * 4].copy_from_slice(&[
                    4,
                    *owner,
                    self.uses[row],
                    self.uses[row],
                ]);
            }
        }
        for retirement in &scheduler.retired {
            for snapshot in retirement.snapshots.into_iter().flatten() {
                let row = snapshot.0 as usize;
                result[3 + row * 4..7 + row * 4].copy_from_slice(&[
                    5,
                    self.owners[row],
                    retirement.through.0 - self.origin,
                    self.uses[row],
                ]);
            }
        }
        for row in 0..3 {
            result[6 + row * 4] = self.uses[row];
            if self.uses[row] > result[1] {
                assert!(
                    matches!(result[3 + row * 4], 2 | 3 | 5),
                    "outstanding checkpoint use lost destination ownership"
                );
                assert!(!scheduler.free_snapshots.contains(&SnapshotId(row as u32)));
            }
        }
        result
    }
}

fn complete_records(
    scheduler: &RequestScheduler,
    plan: &ScheduledPlan,
    fence: Fence,
) -> Vec<CompletionRecord> {
    let mut records = vec![CompletionRecord::default(); 2];
    for (row, lane) in plan.lanes.iter().enumerate() {
        let request = scheduler.request(lane.request).unwrap();
        let consumed = if plan.lane_kind(row).is_prefill() {
            lane.start_position + lane.tokens as u64
        } else {
            request.consumed_tokens
        };
        records[lane.request.slot as usize] = CompletionRecord {
            generation: lane.request.generation as i64,
            submission_sequence: fence.0 as i64,
            status: 1,
            consumed_tokens: consumed as i64,
            generated_tokens: request.generated_tokens as i64,
            token_id: -1,
            ..Default::default()
        };
    }
    records
}

fn edge(before: [u64; 15], action: [u64; 3], after: [u64; 15]) {
    println!(
        "CHECKPOINT_RESERVATION_EDGE {}",
        format_args!("[{before:?},{action:?},{after:?}]")
    );
}

#[test]
fn checkpoint_reservations_match_lifecycle_model() {
    for cancellation in 0..5 {
        let mut scheduler = tests::scheduler(2, 64, 3);
        let handles = [0, 1].map(|slot| {
            let mut spec = tests::spec(64, 16);
            spec.prompt[0] += slot;
            spec.message_boundaries = vec![32];
            scheduler.admit(spec).unwrap()
        });
        let init = scheduler.plan().unwrap();
        assert_eq!(init.kind, GraphKind::Init);
        scheduler.commit(&init, Fence(1)).unwrap();
        let records = complete_records(&scheduler, &init, Fence(1));
        scheduler
            .complete(Fence(1), &records, Default::default())
            .unwrap();
        let mut observation = Observation {
            origin: 1,
            owners: [0; 3],
            uses: [0; 3],
            published: vec![],
        };
        let mut before = observation.project(&scheduler);
        let produce = scheduler.plan().unwrap();
        assert!(produce.kind.is_prefill());
        assert_eq!(produce.lanes.len(), 1);
        let handle = produce.lanes[0].request;
        assert!(handles.contains(&handle));
        scheduler.commit(&produce, Fence(2)).unwrap();
        let snapshot = scheduler
            .request(handle)
            .unwrap()
            .checkpoint_capture
            .unwrap()
            .snapshot;
        assert!(!scheduler.revoke_unused_anchor_guards());
        assert!(!scheduler.free_snapshots.contains(&snapshot));
        let request = handle.slot as u64 + 1;
        let row = snapshot.0 as u64 + 1;
        let mut after = observation.project(&scheduler);
        edge(before, [13, request, row], after);
        before = after;
        if cancellation == 0 {
            let records = complete_records(&scheduler, &produce, Fence(2));
            scheduler.cancel(handle).unwrap();
            after = observation.project(&scheduler);
            edge(before, [15, request, 0], after);
            scheduler
                .complete(Fence(2), &records, Default::default())
                .unwrap();
            edge(after, [16, 0, 0], observation.project(&scheduler));
            assert!(scheduler.free_snapshots.contains(&snapshot));
            continue;
        }
        let records = complete_records(&scheduler, &produce, Fence(2));
        scheduler
            .complete(Fence(2), &records, Default::default())
            .unwrap();
        after = observation.project(&scheduler);
        edge(before, [5, 0, 0], after);
        before = after;
        if cancellation == 1 {
            scheduler.cancel(handle).unwrap();
            edge(before, [15, request, 0], observation.project(&scheduler));
            continue;
        }
        let copy = scheduler.plan().unwrap();
        assert_eq!(copy.kind, GraphKind::Snapshot);
        scheduler.commit(&copy, Fence(3)).unwrap();
        after = observation.project(&scheduler);
        edge(before, [4, request, row], after);
        before = after;
        if cancellation == 2 {
            let records = complete_records(&scheduler, &copy, Fence(3));
            scheduler.cancel(handle).unwrap();
            after = observation.project(&scheduler);
            edge(before, [15, request, 0], after);
            scheduler
                .complete(Fence(3), &records, Default::default())
                .unwrap();
            edge(after, [16, 0, 0], observation.project(&scheduler));
            assert!(scheduler.free_snapshots.contains(&snapshot));
            continue;
        }
        let records = complete_records(&scheduler, &copy, Fence(3));
        scheduler
            .complete(Fence(3), &records, Default::default())
            .unwrap();
        after = observation.project(&scheduler);
        edge(before, [14, request, row], after);
        before = after;
        if cancellation == 3 {
            scheduler.cancel(handle).unwrap();
            edge(before, [15, request, 0], observation.project(&scheduler));
        }
        assert!(
            scheduler
                .cache
                .checkpoint(observation.published[0].0)
                .is_ok()
        );
    }
}

#[test]
fn unused_anchor_guards_revoke_oldest_first_without_cancelling_requests() {
    let mut scheduler = tests::scheduler(2, 64, 3);
    scheduler
        .share_snapshot_pages(&[40..44, 42..46, 48..52])
        .unwrap();
    let first = scheduler.admit(tests::spec(64, 16)).unwrap();
    let second = scheduler.admit(tests::spec(64, 16)).unwrap();
    let mut observation = Observation {
        origin: 0,
        owners: [0; 3],
        uses: [0; 3],
        published: vec![],
    };
    let mut before = observation.project(&scheduler);
    let one = scheduler.reserve_snapshot(0).unwrap();
    scheduler.request_mut(first).unwrap().anchor_guard = Some(one);
    let mut after = observation.project(&scheduler);
    edge(before, [1, first.slot as u64 + 1, one.0 as u64 + 1], after);
    before = after;
    let two = scheduler.reserve_snapshot(0).unwrap();
    scheduler.request_mut(second).unwrap().anchor_guard = Some(two);
    after = observation.project(&scheduler);
    edge(before, [1, second.slot as u64 + 1, two.0 as u64 + 1], after);
    before = after;
    assert!(scheduler.revoke_unused_anchor_guards());
    after = observation.project(&scheduler);
    edge(before, [17, first.slot as u64 + 1, one.0 as u64 + 1], after);
    assert!(scheduler.request(first).unwrap().anchor_guard.is_none());
    assert_eq!(scheduler.request(second).unwrap().anchor_guard, Some(two));
    assert!(scheduler.free_snapshots.contains(&one));
    assert!(!scheduler.free_snapshots.contains(&two));
    before = after;
    assert!(scheduler.revoke_unused_anchor_guards());
    edge(
        before,
        [17, second.slot as u64 + 1, two.0 as u64 + 1],
        observation.project(&scheduler),
    );
    assert!(scheduler.request(second).unwrap().anchor_guard.is_none());
    assert!(scheduler.free_snapshots.contains(&two));
    assert_eq!(scheduler.stats().snapshot_backing_pages, 0);
    assert!(!scheduler.revoke_unused_anchor_guards());
}
