/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use crate::execution::{GraphDescriptor, PrefixAllocation, SamplingMode, TokenVector};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug)]
struct Geometry {
    batch: usize,
    drafts: usize,
    width: usize,
    packed: usize,
    page: usize,
}

impl Geometry {
    fn completion_token_capacity(self) -> usize {
        if self.width <= 8 { 8 } else { 16 }
    }
}

fn ready(geometry: Geometry, prompt: usize, budget: u32) -> (RequestScheduler, Vec<RequestHandle>) {
    let mut scheduler = RequestScheduler::new(
        [7; 32],
        SchedulerLimits {
            request_slots: geometry.batch,
            max_active: geometry.batch,
            max_sequence_tokens: (prompt + budget as usize).max(64),
            prefill_chunk_size: 64,
            pages: 128,
            snapshots: 16,
        },
        SchedulerCapabilities {
            completion_token_capacity: geometry.completion_token_capacity(),
            vocabulary: 1024,
            page_tokens: geometry.page,
            vision: crate::model::VisionCapabilities::disabled(),
            draft_pages_per_request: 2,
            completion_inputs: vec![],
            speculative_allocations: vec![(
                geometry.batch,
                geometry.drafts,
                geometry.width,
                geometry.packed,
                PrefixAllocation::Gpu,
            )],
            state_rows_per_request: None,
        },
    )
    .unwrap();
    scheduler
        .enable_speculation(&(1..=geometry.width).collect::<Vec<_>>())
        .unwrap();
    scheduler
        .fixed_speculative_width(Some(geometry.width))
        .unwrap();
    let mut graphs = tests::duration_graphs(geometry.batch);
    graphs.retain(|graph| graph.kind != GraphKind::Speculative);
    for graph in &mut graphs {
        graph.estimated_duration_ns = 1_000_000;
    }
    graphs.push(GraphDescriptor {
        name: "bounded-speculative-accounting".into(),
        kind: GraphKind::Speculative,
        sampling_mode: SamplingMode::General,
        confidence_width: geometry.drafts,
        draft_tokens: geometry.drafts,
        handle: 1,
        batch_capacity: geometry.batch,
        token_capacity: geometry.width,
        packed_token_capacity: geometry.packed,
        prefill_capacity: 0,
        estimated_duration_ns: 1_000,
        input_bindings: vec![],
        tuned_bindings: vec![],
    });
    scheduler.measured_graph_durations(&graphs).unwrap();
    let handles: Vec<_> = (0..geometry.batch)
        .map(|_| scheduler.admit(tests::spec(prompt, budget)).unwrap())
        .collect();
    for _ in 0..16 {
        if handles
            .iter()
            .all(|handle| scheduler.request(*handle).unwrap().phase == RequestPhase::Decode)
        {
            return (scheduler, handles);
        }
        let kind = if handles
            .iter()
            .any(|handle| scheduler.request(*handle).unwrap().phase == RequestPhase::Initialize)
        {
            GraphKind::Init
        } else if handles
            .iter()
            .any(|handle| scheduler.request(*handle).unwrap().phase == RequestPhase::Prefill)
        {
            GraphKind::Prefill
        } else {
            GraphKind::Snapshot
        };
        let plan = scheduler.phase_plan(kind, None);
        assert_ne!(plan.kind, GraphKind::Speculative);
        let fence = Fence(scheduler.submitted.0 + 1);
        let mut records = vec![CompletionRecord::default(); geometry.batch];
        for lane in &plan.lanes {
            let request = scheduler.request(lane.request).unwrap();
            let prefill = plan.kind.is_prefill();
            let samples = prefill && lane.start_position as usize + lane.tokens == prompt;
            records[lane.request.slot as usize] = CompletionRecord {
                generation: lane.request.generation as i64,
                submission_sequence: fence.0 as i64,
                consumed_tokens: if prefill {
                    (lane.start_position + lane.tokens as u64) as i64
                } else {
                    request.consumed_tokens as i64
                },
                generated_tokens: if samples {
                    1
                } else {
                    request.generated_tokens as i64
                },
                token_id: if samples { 100 } else { -1 },
                status: 1,
                ..Default::default()
            };
        }
        scheduler.commit(&plan, fence).unwrap();
        scheduler
            .complete(fence, &records, Default::default())
            .unwrap();
    }
    panic!("preparation did not reach decode: {geometry:?}");
}

fn completion(
    scheduler: &RequestScheduler,
    plan: &ScheduledPlan,
    fence: Fence,
    counts: [usize; 2],
    offers: [usize; 2],
    terminal: [bool; 2],
) -> Vec<CompletionRecord> {
    let mut records = vec![CompletionRecord::default(); scheduler.limits.request_capacity()];
    for lane in &plan.lanes {
        let slot = lane.request.slot as usize;
        let Ok(request) = scheduler.request(lane.request) else {
            continue;
        };
        let count = counts[slot];
        let mut vector = TokenVector {
            emitted_count: count as i64,
            offered_drafts: offers[slot].saturating_sub(1) as i64,
            accepted_drafts: count.saturating_sub(1) as i64,
            draft_cache_consumed: if count == 0 {
                -1
            } else {
                request.consumed_tokens as i64 + count as i64 - 1
            },
            verification_width: plan.token_capacity as i64,
            tokens: vec![-1; scheduler.capabilities.completion_token_capacity],
            ..Default::default()
        };
        for index in 0..count {
            vector.tokens[index] = 200 + index as i64;
        }
        records[slot] = CompletionRecord {
            generation: lane.request.generation as i64,
            submission_sequence: fence.0 as i64,
            consumed_tokens: request.consumed_tokens as i64 + count as i64,
            generated_tokens: request.generated_tokens as i64 + count as i64,
            token_id: if count == 0 {
                -1
            } else {
                vector.tokens[count - 1]
            },
            status: if terminal[slot] { 2 } else { 1 },
            vector: Some(vector),
            ..Default::default()
        };
    }
    records
}

fn complete(
    scheduler: &mut RequestScheduler,
    plan: &ScheduledPlan,
    fence: Fence,
    records: &[CompletionRecord],
) -> Result<Vec<GeneratedToken>> {
    scheduler
        .advance_to(scheduler.submissions[&fence].expected_finish_ns)
        .unwrap();
    if plan.draft_tokens == 0 {
        scheduler.complete(fence, records, Default::default())
    } else {
        let confidence = vec![
            vec![0.5; scheduler.capabilities.completion_token_capacity - 1];
            scheduler.limits.request_capacity()
        ];
        scheduler.complete_packed_round(fence, records, Default::default(), &confidence)
    }
}

fn assert_accounting(scheduler: &RequestScheduler, handles: &[RequestHandle]) {
    for handle in handles {
        let request = scheduler.request(*handle).unwrap();
        let pending: u64 = scheduler
            .submissions
            .values()
            .map(|submission| {
                submission
                    .lanes
                    .iter()
                    .filter(|lane| lane.request == *handle)
                    .map(|lane| lane.tokens.min(submission.packed_token_capacity) as u64)
                    .sum::<u64>()
            })
            .sum();
        assert_eq!(
            request.scheduled_consumed,
            request.consumed_tokens + pending,
            "pending speculative advance must use physical capacity"
        );
        assert_eq!(
            request.pending,
            scheduler
                .submissions
                .values()
                .filter(|submission| submission.lanes.iter().any(|lane| lane.request == *handle))
                .count()
        );
        for submission in scheduler.submissions.values() {
            if let Some(lane) = submission.lanes.iter().find(|lane| lane.request == *handle) {
                let touched = lane.start_position as usize + lane.tokens;
                let pages =
                    touched.div_ceil(scheduler.capabilities.page_tokens) + scheduler.draft_pages();
                assert!(
                    request.pages().count() >= pages,
                    "queued lane must retain full target and draft page reservation"
                );
                assert!(touched <= scheduler.limits.max_sequence_tokens);
            }
        }
        assert!(request.generated_tokens as u64 + pending <= u64::from(request.max_new_tokens));
    }
}

fn projection(
    scheduler: &RequestScheduler,
    handles: &[RequestHandle],
    geometry: Geometry,
    base: u64,
) -> Vec<Vec<u64>> {
    let first = scheduler.request(handles[0]).unwrap();
    let mut result = vec![vec![
        geometry.page as u64,
        scheduler.limits.max_sequence_tokens as u64,
        first.max_new_tokens as u64,
        geometry.batch as u64,
        geometry.drafts as u64,
        geometry.width as u64,
        geometry.packed as u64,
        scheduler.draft_pages() as u64,
    ]];
    for slot in 0..2 {
        result.push(handles.get(slot).map_or(vec![0; 4], |handle| {
            let request = scheduler.request(*handle).unwrap();
            vec![
                request.consumed_tokens,
                request.generated_tokens as u64,
                request.scheduled_consumed,
                request.pages().count() as u64,
            ]
        }));
    }
    for index in 0..2 {
        result.push(
            scheduler
                .submissions
                .values()
                .nth(index)
                .map_or(vec![0; 2], |submission| {
                    (0..2)
                        .map(|slot| {
                            submission
                                .lanes
                                .iter()
                                .find(|lane| lane.request.slot as usize == slot)
                                .map_or(0, |lane| lane.tokens as u64)
                        })
                        .collect()
                }),
        );
    }
    result.push(vec![
        scheduler.submitted.0 - base,
        scheduler.completed.0 - base,
    ]);
    result
}

fn observe(
    edges: &mut BTreeSet<String>,
    before: Vec<Vec<u64>>,
    action: [usize; 3],
    scheduler: &RequestScheduler,
    handles: &[RequestHandle],
    geometry: Geometry,
    base: u64,
) {
    edges.insert(format!(
        "[{before:?},{action:?},{:?}]",
        projection(scheduler, handles, geometry, base)
    ));
}

#[test]
fn scheduler_matches_speculative_accounting_model() {
    let mut edges = BTreeSet::new();
    let mut cases = 0;
    for page in [64, 128, 256] {
        for width in (1..=8).chain([9, 16]) {
            for packed in
                (1..=width).filter(|packed| width <= 8 || [1, 8, width - 1, width].contains(packed))
            {
                for distance in [0, 1, 2, 7] {
                    for budget in [2 * width + 2, width + 1] {
                        for count in 1..=packed {
                            let geometry = Geometry {
                                batch: 1,
                                drafts: width - 1,
                                width,
                                packed,
                                page,
                            };
                            let (mut scheduler, handles) =
                                ready(geometry, page - distance, budget as u32);
                            let base = scheduler.completed.0;
                            let first = scheduler.plan().unwrap();
                            assert_eq!(
                                (
                                    first.kind,
                                    first.batch_capacity,
                                    first.draft_tokens,
                                    first.token_capacity,
                                    first.packed_token_capacity
                                ),
                                (GraphKind::Speculative, 1, width - 1, width, packed)
                            );
                            let before = projection(&scheduler, &handles, geometry, base);
                            let first_fence = Fence(base + 1);
                            scheduler.commit(&first, first_fence).unwrap();
                            assert_accounting(&scheduler, &handles);
                            observe(
                                &mut edges,
                                before,
                                [1, first.lanes[0].tokens, 0],
                                &scheduler,
                                &handles,
                                geometry,
                                base,
                            );
                            let second = scheduler.plan();
                            if let Some(second) = &second {
                                assert_eq!(second.kind, GraphKind::Speculative);
                                let before = projection(&scheduler, &handles, geometry, base);
                                scheduler.commit(second, Fence(base + 2)).unwrap();
                                assert_accounting(&scheduler, &handles);
                                observe(
                                    &mut edges,
                                    before,
                                    [1, second.lanes[0].tokens, 0],
                                    &scheduler,
                                    &handles,
                                    geometry,
                                    base,
                                );
                                assert!(
                                    scheduler.plan().is_none(),
                                    "two completion frames must bound submission"
                                );
                            }
                            let first_count = count.min(first.lanes[0].tokens);
                            let mut total = first_count;
                            let mut expected_tokens = vec![100];
                            expected_tokens
                                .extend((0..first_count).map(|index| 200 + index as u32));
                            let records = completion(
                                &scheduler,
                                &first,
                                first_fence,
                                [first_count, 0],
                                [first.lanes[0].tokens.min(packed), 0],
                                [false; 2],
                            );
                            let before = projection(&scheduler, &handles, geometry, base);
                            let emitted =
                                complete(&mut scheduler, &first, first_fence, &records).unwrap();
                            assert_eq!(emitted.len(), first_count);
                            assert_accounting(&scheduler, &handles);
                            observe(
                                &mut edges,
                                before,
                                [2, first_count, 0],
                                &scheduler,
                                &handles,
                                geometry,
                                base,
                            );
                            if let Some(second) = &second {
                                let second_count = second.lanes[0].tokens.min(packed) + 1
                                    - count.min(second.lanes[0].tokens.min(packed));
                                total += second_count;
                                expected_tokens
                                    .extend((0..second_count).map(|index| 200 + index as u32));
                                let records = completion(
                                    &scheduler,
                                    second,
                                    Fence(base + 2),
                                    [second_count, 0],
                                    [second.lanes[0].tokens.min(packed), 0],
                                    [false; 2],
                                );
                                let before = projection(&scheduler, &handles, geometry, base);
                                let emitted =
                                    complete(&mut scheduler, second, Fence(base + 2), &records)
                                        .unwrap();
                                assert_eq!(emitted.len(), second_count);
                                assert_accounting(&scheduler, &handles);
                                observe(
                                    &mut edges,
                                    before,
                                    [2, second_count, 0],
                                    &scheduler,
                                    &handles,
                                    geometry,
                                    base,
                                );
                            }
                            let request = scheduler.request(handles[0]).unwrap();
                            assert_eq!(request.generated_tokens as usize, 1 + total);
                            assert_eq!(request.consumed_tokens as usize, page - distance + total);
                            assert_eq!(request.tokens().len(), page - distance + 1 + total);
                            assert_eq!(&request.tokens()[page - distance..], expected_tokens);
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    for page in [64, 128, 256] {
        for width in (1..=8).chain([9, 16]) {
            for drafts in BTreeSet::from([0, width - 1]) {
                for packed in (2..=2 * width).filter(|packed| {
                    width <= 8
                        || [2, width - 1, width, width + 1, 2 * width - 1, 2 * width]
                            .contains(packed)
                }) {
                    for first_count in 1..=width.min(drafts + 1).min(packed - 1) {
                        let second_count = width.min(drafts + 1).min(packed - first_count);
                        let geometry = Geometry {
                            batch: 2,
                            drafts,
                            width,
                            packed,
                            page,
                        };
                        let (mut scheduler, handles) =
                            ready(geometry, page - 1, (4 * width + 4) as u32);
                        let base = scheduler.completed.0;
                        let mut plans = Vec::new();
                        for frame in 1..=2 {
                            let plan = scheduler.plan().unwrap();
                            assert_eq!(plan.lanes.len(), 2);
                            let before = projection(&scheduler, &handles, geometry, base);
                            scheduler.commit(&plan, Fence(base + frame)).unwrap();
                            assert_accounting(&scheduler, &handles);
                            let limits: [usize; 2] = std::array::from_fn(|slot| {
                                plan.lanes
                                    .iter()
                                    .find(|lane| lane.request.slot as usize == slot)
                                    .unwrap()
                                    .tokens
                            });
                            observe(
                                &mut edges,
                                before,
                                [1, limits[0], limits[1]],
                                &scheduler,
                                &handles,
                                geometry,
                                base,
                            );
                            plans.push(plan);
                        }
                        for (index, plan) in plans.iter().enumerate() {
                            let counts = if index == 0 {
                                [first_count, second_count]
                            } else {
                                [second_count, first_count]
                            };
                            let fence = Fence(base + index as u64 + 1);
                            let records =
                                completion(&scheduler, plan, fence, counts, counts, [false; 2]);
                            let before = projection(&scheduler, &handles, geometry, base);
                            let output = complete(&mut scheduler, plan, fence, &records).unwrap();
                            assert_eq!(output.len(), counts.iter().sum::<usize>());
                            assert_accounting(&scheduler, &handles);
                            observe(
                                &mut edges,
                                before,
                                [2, counts[0], counts[1]],
                                &scheduler,
                                &handles,
                                geometry,
                                base,
                            );
                        }
                        cases += 1;
                    }
                }
            }
        }
    }
    for edge in &edges {
        println!("SPECULATION_EDGE {edge}");
    }
    println!(
        "speculative accounting covered {cases} schedules and {} distinct transitions",
        edges.len()
    );
}

#[test]
fn packed_two_lane_completions_validate_atomically_for_bounded_geometry() {
    let mut cases = 0;
    for page in [64, 128, 256] {
        for width in (1..=8).chain([9, 16]) {
            for drafts in 0..width {
                let capacities: BTreeSet<_> = [2, 3.min(2 * width), 2 * width - 1, 2 * width]
                    .into_iter()
                    .filter(|packed| *packed >= 2)
                    .collect();
                for packed in capacities {
                    let geometry = Geometry {
                        batch: 2,
                        drafts,
                        width,
                        packed,
                        page,
                    };
                    let (mut scheduler, handles) =
                        ready(geometry, page - 1, (4 * width + 4) as u32);
                    let base = scheduler.completed.0;
                    let first = scheduler.plan().unwrap();
                    assert_eq!(
                        (
                            first.kind,
                            first.lanes.len(),
                            first.batch_capacity,
                            first.draft_tokens,
                            first.token_capacity,
                            first.packed_token_capacity
                        ),
                        (GraphKind::Speculative, 2, 2, drafts, width, packed)
                    );
                    scheduler.commit(&first, Fence(base + 1)).unwrap();
                    let second = scheduler.plan().unwrap();
                    scheduler.commit(&second, Fence(base + 2)).unwrap();
                    assert_accounting(&scheduler, &handles);
                    let valid = completion(
                        &scheduler,
                        &first,
                        Fence(base + 1),
                        [1, 1],
                        [1, 1],
                        [false; 2],
                    );
                    for fault in 0..5 {
                        let mut bad = valid.clone();
                        let record = &mut bad[handles[1].slot as usize];
                        match fault {
                            0 => record.generation += 1,
                            1 => record.submission_sequence += 1,
                            2 => record.consumed_tokens += 1,
                            3 => {
                                record.vector.as_mut().unwrap().tokens[0] = -1;
                                record.token_id = -1;
                            }
                            4 => record.vector.as_mut().unwrap().emitted_count = 0,
                            _ => unreachable!(),
                        }
                        let before = projection(&scheduler, &handles, geometry, base);
                        let tokens: Vec<_> = handles
                            .iter()
                            .map(|handle| scheduler.request(*handle).unwrap().tokens().to_vec())
                            .collect();
                        let population = scheduler.forecasts.population.clone();
                        let free = scheduler.free_page_count();
                        assert!(
                            complete(&mut scheduler, &first, Fence(base + 1), &bad).is_err(),
                            "malformed second lane accepted: {geometry:?}, fault={fault}"
                        );
                        assert_eq!(
                            projection(&scheduler, &handles, geometry, base),
                            before,
                            "completion validation must not commit the first lane before checking the second"
                        );
                        assert_eq!(scheduler.forecasts.population, population);
                        assert_eq!(scheduler.free_page_count(), free);
                        for (handle, expected) in handles.iter().zip(&tokens) {
                            assert_eq!(scheduler.request(*handle).unwrap().tokens(), expected);
                        }
                        cases += 1;
                    }
                    let limits: [usize; 2] = std::array::from_fn(|slot| {
                        first
                            .lanes
                            .iter()
                            .find(|lane| lane.request.slot as usize == slot)
                            .unwrap()
                            .tokens
                    });
                    if limits.iter().sum::<usize>() > packed {
                        let bad = completion(
                            &scheduler,
                            &first,
                            Fence(base + 1),
                            [1, 1],
                            limits,
                            [false; 2],
                        );
                        let before = projection(&scheduler, &handles, geometry, base);
                        let error =
                            complete(&mut scheduler, &first, Fence(base + 1), &bad).unwrap_err();
                        assert!(error.to_string().contains("physical row budget"), "{error}");
                        assert_eq!(projection(&scheduler, &handles, geometry, base), before);
                        cases += 1;
                    }
                    complete(&mut scheduler, &first, Fence(base + 1), &valid).unwrap();
                    assert_accounting(&scheduler, &handles);
                    let valid = completion(
                        &scheduler,
                        &second,
                        Fence(base + 2),
                        [1, 1],
                        [1, 1],
                        [false; 2],
                    );
                    complete(&mut scheduler, &second, Fence(base + 2), &valid).unwrap();
                    assert_accounting(&scheduler, &handles);
                }
            }
        }
    }
    println!("checked {cases} transactional two-lane rejection cases");
}

#[test]
fn terminal_noop_and_cancel_preserve_queued_speculative_storage() {
    let mut cases = 0;
    for page in [64, 128, 256] {
        for width in (1..=8).chain([9, 16]) {
            for cancel in [false, true] {
                let geometry = Geometry {
                    batch: 2,
                    drafts: width - 1,
                    width,
                    packed: 3.min(2 * width),
                    page,
                };
                let (mut scheduler, handles) = ready(geometry, page - 2, (4 * width + 4) as u32);
                let first = scheduler.plan().unwrap();
                let base = scheduler.completed.0;
                scheduler.commit(&first, Fence(base + 1)).unwrap();
                let second = scheduler.plan().unwrap();
                scheduler.commit(&second, Fence(base + 2)).unwrap();
                let first_records = completion(
                    &scheduler,
                    &first,
                    Fence(base + 1),
                    [1, 1],
                    [1, 1],
                    [true, false],
                );
                let held = {
                    let request = scheduler.request(handles[0]).unwrap();
                    request.private_pages.len() + request.reserved_pages.len()
                };
                let free = scheduler.free_page_count();
                if cancel {
                    scheduler.cancel(handles[0]).unwrap();
                    assert_eq!(scheduler.free_page_count(), free);
                    assert!(scheduler.admit(tests::spec(page - 3, 2)).is_err());
                }
                complete(&mut scheduler, &first, Fence(base + 1), &first_records).unwrap();
                if cancel {
                    assert_eq!(scheduler.free_page_count(), free);
                    assert!(scheduler.admit(tests::spec(page - 3, 2)).is_err());
                }
                let progressing = second
                    .lanes
                    .iter()
                    .find(|lane| lane.request == handles[1])
                    .unwrap()
                    .tokens
                    .min(geometry.packed);
                let mut second_records = completion(
                    &scheduler,
                    &second,
                    Fence(base + 2),
                    [0, progressing],
                    [0, progressing],
                    [true, false],
                );
                if cancel {
                    second_records[handles[0].slot as usize] = CompletionRecord {
                        generation: handles[0].generation as i64,
                        submission_sequence: (base + 2) as i64,
                        consumed_tokens: (page - 1) as i64,
                        generated_tokens: 2,
                        token_id: -1,
                        status: 2,
                        vector: Some(TokenVector {
                            verification_width: width as i64,
                            draft_cache_consumed: -1,
                            tokens: vec![-1; geometry.completion_token_capacity()],
                            ..Default::default()
                        }),
                        ..Default::default()
                    };
                }
                let population = scheduler.forecasts.population.clone();
                if !cancel && width > 1 {
                    let before = projection(&scheduler, &handles, geometry, base);
                    let mut invalid = second_records.clone();
                    invalid[handles[0].slot as usize]
                        .vector
                        .as_mut()
                        .unwrap()
                        .offered_drafts = 1;
                    let error =
                        complete(&mut scheduler, &second, Fence(base + 2), &invalid).unwrap_err();
                    assert!(
                        error
                            .to_string()
                            .contains("finished speculative no-op cannot offer draft positions"),
                        "{error}"
                    );
                    assert_eq!(projection(&scheduler, &handles, geometry, base), before);
                }
                let emitted =
                    complete(&mut scheduler, &second, Fence(base + 2), &second_records).unwrap();
                assert_eq!(emitted.len(), progressing);
                assert!(emitted.iter().all(|event| event.request == handles[1]));
                if cancel {
                    assert_eq!(scheduler.free_page_count(), free + held);
                    let next = scheduler.admit(tests::spec(page - 3, 2)).unwrap();
                    assert_eq!(next.slot, handles[0].slot);
                    assert_eq!(next.generation, handles[0].generation + 1);
                    assert!(scheduler.request(handles[0]).is_err());
                    let init = scheduler.phase_plan(GraphKind::Init, Some(next));
                    let fence = Fence(scheduler.submitted.0 + 1);
                    scheduler.commit(&init, fence).unwrap();
                    let mut records = vec![CompletionRecord::default(); 2];
                    records[next.slot as usize] = CompletionRecord {
                        generation: handles[0].generation as i64,
                        submission_sequence: fence.0 as i64,
                        token_id: -1,
                        status: 1,
                        ..Default::default()
                    };
                    assert!(
                        scheduler
                            .complete(fence, &records, Default::default())
                            .is_err(),
                        "retired generation must not complete work for the reused slot"
                    );
                    assert_eq!(scheduler.request(next).unwrap().pending, 1);
                    records[next.slot as usize].generation = next.generation as i64;
                    scheduler
                        .complete(fence, &records, Default::default())
                        .unwrap();
                } else {
                    let request = scheduler.request(handles[0]).unwrap();
                    assert_eq!(
                        (
                            request.consumed_tokens,
                            request.generated_tokens,
                            request.pending
                        ),
                        ((page - 1) as u64, 2, 0)
                    );
                    if width == 1 {
                        assert_eq!(scheduler.forecasts.population, population);
                    }
                }
                cases += 1;
            }
        }
    }
    println!("checked {cases} queued terminal/cancellation schedules");
}
