/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use super::{policy::Candidate, utility::ValueRate};
use crate::execution::{ExecutionTiming, SamplingMode, TokenVector};

const MS: u64 = 1_000_000;

#[path = "planning_tests.rs"]
mod planning_tests;

#[test]
fn padding_pruning_keeps_three_requests_on_batch_four_and_two_requests_on_batch_two() {
    let mut s = engine(8, false);
    let handles = (0..3).map(|id| add(&mut s, id, 64, 4096)).collect::<Vec<_>>();
    for handle in handles {
        let request = s.request_mut(handle).unwrap();
        request.phase = RequestPhase::Decode;
        request.consumed_tokens = 64;
        request.scheduled_consumed = 64;
        request.generated_tokens = 1;
    }
    let (values, age, _) = s.lane_values(s.window_remaining().unwrap());
    let mut candidates = Vec::new();
    s.candidates(&mut candidates, &values, age);
    assert!(candidates.iter().any(|c| c.kind == GraphKind::Decode && c.batch == 8 && c.mask.count_ones() == 3));
    let count = s.prune_padding(&mut candidates);
    candidates.truncate(count);
    assert!(candidates.iter().any(|c| c.kind == GraphKind::Decode && c.batch == 4 && c.mask.count_ones() == 3));
    assert!(candidates.iter().any(|c| c.kind == GraphKind::Decode && c.batch == 2 && c.mask.count_ones() == 2));
    assert!(!candidates.iter().any(|c| c.kind == GraphKind::Decode && c.batch == 8));
}

#[test]
fn faster_padded_estimates_do_not_displace_equivalent_unpadded_work() {
    let mut s = engine(8, false);
    let h = add(&mut s, 1, 64, 128);
    ready(&mut s, &[h]);
    set_cost(&mut s, GraphKind::Decode, 1, 1, 10 * MS);
    for batch in [2, 4, 8] {
        set_cost(&mut s, GraphKind::Decode, batch, 1, MS);
    }
    let plan = s.plan().unwrap();
    assert_eq!((plan.kind, plan.batch_capacity, plan.lanes.len()), (GraphKind::Decode, 1, 1));
}

#[test]
fn equal_counts_from_different_cohorts_and_different_logical_prefixes_are_not_pruned() {
    let mut s = engine(4, true);
    let handles = (0..2).map(|id| add(&mut s, id, 64, 128)).collect::<Vec<_>>();
    ready(&mut s, &handles);
    let candidate = Candidate { host_lengths: None, draft_tokens: 0, kind: GraphKind::Speculative, mask: 1 << handles[0].slot,
        batch: 1, capacity: 1, packed_token_capacity: 1, prefill_capacity: 0, prefill_slot: None,
        work: 0, duration_ns: MS, rate: ValueRate::new(MS, 1.0, 0.0), feedback: true,
        };
    assert!(!s.dominates_padding(candidate, Candidate { batch: 4, capacity: 4, draft_tokens: 3, packed_token_capacity: 16, ..candidate }));
    assert!(!s.dominates_padding(candidate, Candidate { batch: 4, packed_token_capacity: 4, mask: 1 << handles[1].slot, ..candidate }));
}

fn engine(active: usize, speculative: bool) -> RequestScheduler {
    engine_with_geometry(active, speculative, 64, None)
}

fn engine_with_geometry(
    active: usize, speculative: bool, page_tokens: usize, state_rows_per_request: Option<usize>,
) -> RequestScheduler {
    let mut s = RequestScheduler::new(
        [7; 32],
        SchedulerLimits {
            max_active: active,
            request_slots: active.next_power_of_two(),
            max_sequence_tokens: 8192,
            prefill_chunk_size: 512,
            pages: 1024,
            snapshots: 256,
        },
        SchedulerCapabilities {
            completion_token_capacity: 16,
            vocabulary: 1024,
            page_tokens,
            vision: crate::model::VisionCapabilities::disabled(),
            draft_pages_per_request: 0,
            completion_inputs: vec![],
            speculative_allocations: vec![],
            state_rows_per_request,
        },
    )
    .unwrap();
    let mut graphs = super::tests::duration_graphs(active);
    for g in &mut graphs {
        g.estimated_duration_ns = match g.kind {
            GraphKind::Prefill => (g.token_capacity as u64 / 64 + 1) * MS,
            GraphKind::Decode => MS + (g.batch_capacity as u64 - 1) * MS / 4,
            GraphKind::Speculative => {
                MS + (g.batch_capacity as u64 - 1) * MS / 4 + (g.token_capacity as u64 - 1) * MS / 2
            }
            _ => MS / 10,
        };
    }
    s.measured_graph_durations(&graphs).unwrap();
    if speculative {
        s.enable_speculation(&[1, 2, 4, 8]).unwrap();
    }
    s
}

fn add(s: &mut RequestScheduler, id: u32, prompt: usize, output: u32) -> RequestHandle {
    s.admit(RequestSpec {
        prompt: vec![id; prompt],
        message_boundaries: vec![],
        conditioning: vec![],
        max_new_tokens: output,
        grammar_constrained: false,
        host_stops: false,
        prompt_logprobs: false,
        growable_output: false,
    })
    .unwrap()
}

fn complete(
    s: &mut RequestScheduler,
    plan: &ScheduledPlan,
    fence: Fence,
    accepted: usize,
    publish: bool,
) {
    complete_timed(s, plan, fence, accepted, publish, Default::default());
}

fn completion_records(
    s: &RequestScheduler,
    plan: &ScheduledPlan,
    fence: Fence,
    accepted: usize,
) -> Vec<CompletionRecord> {
    let mut records = vec![CompletionRecord::default(); s.limits.request_capacity()];
    for (row, lane) in plan.lanes.iter().enumerate() {
        let kind = plan.lane_kind(row);
        let Ok(r) = s.request(lane.request) else {
            continue;
        };
        let count = match kind {
            GraphKind::Prefill => {
                usize::from(r.consumed_tokens + lane.tokens as u64 == r.prompt_tokens as u64)
            }
            GraphKind::Resume | GraphKind::Decode => 1,
            GraphKind::Speculative => lane.tokens.min(accepted + 1),
            _ => 0,
        };
        let consumed = if kind == GraphKind::Speculative {
            r.consumed_tokens + count as u64
        } else {
            lane.start_position + lane.tokens as u64
        };
        let generated = r.generated_tokens + count as u32;
        let mut record = CompletionRecord {
            generation: lane.request.generation as i64,
            submission_sequence: fence.0 as i64,
            consumed_tokens: consumed as i64,
            generated_tokens: generated as i64,
            token_id: if count == 0 { -1 } else { 99 },
            status: if count > 0 && generated == r.max_new_tokens {
                2
            } else {
                1
            },
            ..Default::default()
        };
        if kind == GraphKind::Speculative {
            let mut tokens = vec![-1; 16];
            tokens[..count].fill(99);
            record.vector = Some(TokenVector {
                emitted_count: count as i64,
                offered_drafts: (plan.token_capacity - 1) as i64,
                accepted_drafts: (count - 1) as i64,
                verification_width: plan.token_capacity as i64,
                draft_cache_consumed: -1,
                tokens,
                ..Default::default()
            });
        }
        records[lane.request.slot as usize] = record;
    }
    records
}

fn complete_timed(
    s: &mut RequestScheduler,
    plan: &ScheduledPlan,
    fence: Fence,
    accepted: usize,
    publish: bool,
    timing: ExecutionTiming,
) {
    let records = completion_records(s, plan, fence, accepted);
    let output = s.complete(fence, &records, timing).unwrap();
    if publish {
        for token in output {
            s.published_output(token.request, fence).unwrap();
        }
    }
    let finished: Vec<_> = s
        .slots
        .iter()
        .filter_map(|slot| slot.request.as_ref())
        .filter(|r| r.phase == RequestPhase::Finishing && r.pending == 0)
        .map(|r| r.handle)
        .collect();
    for h in finished {
        s.finish(h).unwrap();
    }
}

fn step(s: &mut RequestScheduler, accepted: usize) -> ScheduledPlan {
    let plan = s.plan().expect("runnable trace must make progress");
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    s.advance_to(s.now_ns + plan.estimated_duration_ns())
        .unwrap();
    complete_timed(
        s,
        &plan,
        fence,
        accepted,
        true,
        gpu_time(plan.estimated_duration_ns()),
    );
    plan
}

fn ready(s: &mut RequestScheduler, handles: &[RequestHandle]) {
    for _ in 0..200 {
        if handles
            .iter()
            .all(|h| s.request(*h).unwrap().phase == RequestPhase::Decode)
        {
            return;
        }
        step(s, 0);
    }
    panic!("requests did not reach decode");
}

#[test]
fn a_speculative_burst_counts_one_published_delivery() {
    let mut s = engine(1, true);
    s.fixed_speculative_width(Some(8)).unwrap();
    let h = add(&mut s, 1, 65, 64);
    ready(&mut s, &[h]);
    let deliveries = s.request(h).unwrap().accounting.published_deliveries;
    let output = s.request(h).unwrap().generated_tokens;
    step(&mut s, 7);
    let request = s.request(h).unwrap();
    assert_eq!(request.generated_tokens, output + 8);
    assert_eq!(request.accounting.published_deliveries, deliveries + 1);
    assert_eq!(request.accounting.last_output_fence, Some(s.completed));
}

#[test]
fn publication_requires_a_matching_output_completion() {
    let mut s = engine(1, true);
    let h = add(&mut s, 1, 65, 64);
    assert!(s.published_output(h, Fence(0)).is_err());
    step(&mut s, 0);
    assert!(s.published_output(h, s.completed).is_err());
    ready(&mut s, &[h]);
    let earlier = s.request(h).unwrap().accounting.last_output_fence.unwrap();
    step(&mut s, 1);
    let before = *s.request(h).unwrap().accounting();
    assert!(s.published_output(h, earlier).is_err());
    assert!(s.published_output(h, Fence(s.submitted.0 + 1)).is_err());
    assert_eq!(
        s.request(h).unwrap().accounting.last_output_ns,
        before.last_output_ns
    );
    assert_eq!(
        s.request(h).unwrap().accounting.published_deliveries,
        before.published_deliveries
    );
}

#[test]
fn large_cohort_search_covers_high_slots_without_growing_scratch() {
    let mut s = engine(64, true);
    let handles: Vec<_> = (1..=64).map(|id| add(&mut s, id, 1, 64)).collect();
    for h in &handles {
        let r = s.request_mut(*h).unwrap();
        r.phase = RequestPhase::Decode;
        r.consumed_tokens = 1;
        r.scheduled_consumed = 1;
        r.generated_tokens = 1;
    }
    s.advance_to(100 * MS).unwrap();
    for h in &handles[..63] {
        s.request_mut(*h).unwrap().accounting.last_output_ns = Some(s.now_ns);
    }
    let capacity = s.candidates.borrow().capacity();
    let plan = s.plan().unwrap();
    assert!(plan.lanes.iter().any(|l| l.request == handles[63]));
    assert!(plan.selection_value_rate().is_finite());
    assert_eq!(s.candidates.borrow().capacity(), capacity);
    assert!(plan.lanes.iter().all(|l| l.tokens <= plan.token_capacity));
}

#[test]
fn prefix_restore_earns_no_credit_for_reused_prompt_tokens() {
    let mut s = engine(1, false);
    let seed = add(&mut s, 1, 65, 1);
    while s.request(seed).is_ok() {
        step(&mut s, 0);
    }
    let h = add(&mut s, 1, 65, 32);
    assert_eq!(s.request(h).unwrap().reused_prompt_tokens(), 65);
    assert_eq!(s.request(h).unwrap().accounting.committed_prefill_tokens, 0);
    let restore = step(&mut s, 0);
    assert_eq!(restore.kind, GraphKind::Restore);
    assert_eq!(s.request(h).unwrap().accounting.committed_output_tokens, 0);
    let resume = step(&mut s, 0);
    assert_eq!(resume.kind, GraphKind::Resume);
    let r = s.request(h).unwrap();
    assert_eq!(r.accounting.committed_output_tokens, 1);
}

#[test]
fn host_delays_do_not_change_speculative_graph_selection() {
    for queued in [false, true] {
        for stale in [false, true] {
            let mut s = engine(1, true);
            s.configure_policy(PolicyConfig::throughput()).unwrap();
            let handle = add(&mut s, 1, 64, 4096);
            ready(&mut s, &[handle]);
            if stale {
                s.advance_to(s.now_ns + s.policy.estimate_ttl_ns).unwrap();
            }
            let plan = s.plan().unwrap();
            let key = cost::Key::from_step(&plan);
            let gpu_ns = plan.estimated_duration_ns();
            let mut timing = gpu_time(gpu_ns);
            timing.wall_ns = 100 * gpu_ns;
            timing.queued = queued;
            s.observe_execution(&plan, timing);
            assert_eq!(s.estimated_duration_ns(&plan), Some(gpu_ns));
            assert_eq!(cost::Key::from_step(&s.plan().unwrap()), key);
        }
    }
}

#[test]
fn a_plan_keeps_its_duration_when_the_graph_estimate_changes() {
    let mut s = engine(1, false);
    add(&mut s, 1, 1, 32);
    let plan = s.plan().unwrap();
    let estimate = plan.estimated_duration_ns();
    let selected = plan.selected_cost;
    s.advance_to(1).unwrap();
    s.observe_execution(
        &plan,
        ExecutionTiming {
            gpu_ns: [
                Some(80 * MS),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            ..Default::default()
        },
    );
    assert_ne!(s.estimated_duration_ns(&plan), Some(estimate));
    assert_eq!(plan.estimated_duration_ns(), estimate);
    s.commit(&plan, Fence(1)).unwrap();
    let timing = s.submitted_timing(Fence(1)).unwrap();
    assert_eq!(timing.retained_ns, selected.retained_ns);
    assert_eq!(timing.observed_at_ns, selected.observed_at_ns);
    assert_eq!(timing.freshness, selected.freshness);
    assert_eq!(timing.selected_at_ns, 0);
    assert_eq!(timing.commit_ns, 1);
    assert_ne!(timing.retained_ns, s.estimated_duration_ns(&plan).unwrap());
    assert_eq!(s.window_remaining(), Some(estimate));
}

#[test]
fn atomic_prefill_respects_spans_without_a_duration_cap() {
    let mut s = engine(1, false);
    let h = add(&mut s, 1, 256, 32);
    s.set_prefill_spans(h, &[0..256]).unwrap();
    step(&mut s, 0);
    let plan = s.plan().unwrap();
    assert_eq!(
        (plan.kind, plan.token_capacity, plan.lanes[0].tokens),
        (GraphKind::Prefill, 256, 256)
    );
    step(&mut s, 0);
    assert_eq!(s.request(h).unwrap().consumed_tokens, 256);
}

#[test]
fn short_prefill_does_not_explore_padding_after_the_smallest_shape_is_measured() {
    let mut s = cold_engine(1, false);
    s.prefill_graphs(&[64, 128, 512]).unwrap();
    let h = add(&mut s, 1, 130, 32);
    s.request_mut(h).unwrap().message_boundaries = vec![17];
    assert_eq!(measured_step(&mut s, MS / 10, 0).kind, GraphKind::Init);
    let first = measured_step(&mut s, MS, 0);
    assert_eq!((first.token_capacity, first.lanes[0].tokens), (64, 17));
    assert_eq!(measured_step(&mut s, MS / 10, 0).kind, GraphKind::Snapshot);
    let second = measured_step(&mut s, 2 * MS, 0);
    assert_eq!((second.token_capacity, second.lanes[0].tokens), (128, 113));
    s.cancel(h).unwrap();
    let h = add(&mut s, 2, 48, 32);
    assert_eq!(measured_step(&mut s, MS / 10, 0).kind, GraphKind::Init);
    let tail = s.plan().unwrap();
    assert_eq!((tail.token_capacity, tail.lanes[0].tokens), (64, 48));
    s.cancel(h).unwrap();
}

#[test]
fn indeterminate_and_equal_costs_naturally_choose_full_useful_prefill() {
    for known in [false, true] {
        let mut s = cold_engine(1, false);
        if known {
            let mut graphs = super::tests::duration_graphs(1);
            for graph in &mut graphs {
                graph.estimated_duration_ns = MS;
            }
            s.measured_graph_durations(&graphs).unwrap();
        }
        let h = add(&mut s, 1, 4096, 64);
        measured_step(&mut s, MS / 10, 0);
        let plan = measured_step(&mut s, 5 * MS, 0);
        assert_eq!(plan.kind, GraphKind::Prefill);
        assert_eq!((plan.token_capacity, plan.lanes[0].tokens), (512, 512));
        let credit = s.request(h).unwrap().accounting();
        assert_eq!(credit.committed_prefill_tokens, 512);
        assert_eq!(credit.committed_output_tokens, 0);
    }
}

#[test]
fn indeterminate_and_equal_costs_naturally_choose_full_useful_speculation() {
    for known in [false, true] {
        for (limit, width) in [(64, 8), (3, 2)] {
            let mut s = cold_engine(1, true);
            if known {
                let mut graphs = super::tests::duration_graphs(1);
                for graph in &mut graphs {
                    graph.estimated_duration_ns = MS;
                }
                s.measured_graph_durations(&graphs).unwrap();
            }
            add(&mut s, 1, 1, limit);
            for kind in [GraphKind::Init, GraphKind::Prefill, GraphKind::Snapshot] {
                assert_eq!(measured_step(&mut s, MS / 10, 0).kind, kind);
            }
            let plan = s.plan().unwrap();
            assert_eq!(plan.kind, GraphKind::Speculative);
            assert_eq!((plan.token_capacity, plan.lanes[0].tokens), (width, width));
        }
    }
}

#[test]
fn indeterminate_execution_occupies_the_submission_window_until_completion() {
    let mut s = cold_engine(2, false);
    add(&mut s, 1, 1, 64);
    let plan = s.plan().unwrap();
    assert_eq!(plan.estimated_duration_ns(), 0);
    assert!(plan.requires_feedback);
    s.commit(&plan, Fence(1)).unwrap();
    add(&mut s, 2, 1, 64);
    assert!(!s.submission_window_available(0));
    assert!(!s.submission_window_available(MS));
    assert!(s.plan().is_none());
    s.advance_to(MS).unwrap();
    complete(&mut s, &plan, Fence(1), 0, true);
    s.observe_execution(
        &plan,
        ExecutionTiming {
            gpu_ns: [
                Some(MS),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            ..Default::default()
        },
    );
    assert_eq!(s.estimated_duration_ns(&plan), Some(MS));
    assert!(s.plan().is_some());
}

#[test]
fn stale_graph_cost_retains_its_last_valid_observation_and_resets_the_median() {
    let mut s = cold_engine(1, false);
    add(&mut s, 1, 1, 64);
    let plan = s.plan().unwrap();
    let ttl = s.policy.estimate_ttl_ns;
    s.advance_to(10 * MS).unwrap();
    s.observe_execution(
        &plan,
        ExecutionTiming {
            gpu_ns: [
                Some(MS),
                Some(2 * MS),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            ..Default::default()
        },
    );
    assert_eq!(s.estimated_duration_ns(&plan), Some(2 * MS));
    s.advance_to(10 * MS + ttl - 1).unwrap();
    assert_eq!(s.estimated_duration_ns(&plan), Some(2 * MS));
    s.observe_execution(
        &plan,
        ExecutionTiming {
            gpu_ns: [
                Some(0),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            ..Default::default()
        },
    );
    s.advance_to(10 * MS + ttl).unwrap();
    assert_eq!(s.estimated_duration_ns(&plan), Some(2 * MS));
    assert!(!s.costs.is_fresh(cost::Key::from_step(&plan)));
    s.observe_execution(
        &plan,
        ExecutionTiming {
            wall_ns: MS,
            ..Default::default()
        },
    );
    assert_eq!(s.estimated_duration_ns(&plan), Some(2 * MS));
    s.observe_execution(
        &plan,
        ExecutionTiming {
            gpu_ns: [
                Some(8 * MS),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            ..Default::default()
        },
    );
    assert_eq!(s.estimated_duration_ns(&plan), Some(8 * MS));
    s.observe_execution(
        &plan,
        ExecutionTiming {
            gpu_ns: [
                Some(16 * MS),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            ..Default::default()
        },
    );
    assert_eq!(s.estimated_duration_ns(&plan), Some(12 * MS));
}

#[test]
fn funded_stale_graph_recheck_recovers_a_poisoned_observation() {
    let mut s = engine(1, true);
    let h = add(&mut s, 1, 1, 64);
    ready(&mut s, &[h]);
    let mut graphs = super::tests::duration_graphs(1);
    for graph in &mut graphs {
        graph.estimated_duration_ns = if graph.token_capacity == 8 {
            2 * MS
        } else {
            4 * MS
        };
    }
    s.costs = super::tests::measured_costs(&graphs);
    let plan = s.plan().unwrap();
    assert_eq!(plan.token_capacity, 8);
    s.observe_execution(
        &plan,
        ExecutionTiming {
            gpu_ns: [
                Some(1_000 * MS),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            ..Default::default()
        },
    );
    assert_eq!(s.estimated_duration_ns(&plan), Some(501_000_000));
    assert_ne!(s.plan().unwrap().token_capacity, 8);
    let ttl = s.policy.estimate_ttl_ns;
    s.advance_to(ttl / 2).unwrap();
    for width in [1, 2, 4] {
        let mut other = (*plan).clone();
        other.token_capacity = width;
        other.draft_tokens = width - 1;
        other.packed_token_capacity = other.batch_capacity * width;
        s.observe_execution(
            &other,
            ExecutionTiming {
                gpu_ns: [
                    Some(4 * MS),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
                ..Default::default()
            },
        );
    }
    s.advance_to(ttl).unwrap();
    assert_ne!(s.plan().unwrap().token_capacity, 8);
    s.refresh.credit_ns = refresh::CREDIT_CAP_NS;
    let retry = measured_step(&mut s, 2 * MS, 7);
    assert_eq!(
        (retry.token_capacity, retry.estimated_duration_ns()),
        (8, 501_000_000)
    );
    assert!(matches!(retry.origin, PlanOrigin::RefreshStart(_)));
    assert_eq!(s.estimated_duration_ns(&retry), Some(2 * MS));
}

#[test]
fn neglected_stale_shapes_require_funding_while_the_winner_stays_fresh() {
    let mut s = cold_engine(1, true);
    s.configure_policy(PolicyConfig {
        estimate_ttl_ns: 20 * MS,
        ..s.policy
    })
    .unwrap();
    let h = add(&mut s, 1, 1, 1024);
    for _ in 0..3 {
        measured_step(&mut s, MS / 10, 0);
    }
    let mut widths = Vec::new();
    for _ in 0..4 {
        widths.push(measured_step(&mut s, MS, 7).token_capacity);
    }
    assert_eq!(widths, [8, 4, 2, 1]);
    for _ in 0..15 {
        assert_eq!(measured_step(&mut s, MS, 7).token_capacity, 8);
    }
    for _ in 0..10 {
        let plan = measured_step(&mut s, MS, 7);
        assert_eq!(plan.token_capacity, 8);
        assert_eq!(plan.origin, PlanOrigin::Ordinary);
    }
    let mut retried = Vec::new();
    for _ in 0..3 {
        s.refresh.credit_ns = refresh::CREDIT_CAP_NS;
        let plan = measured_step(&mut s, MS, 7);
        if matches!(plan.origin, PlanOrigin::RefreshStart(_)) {
            retried.push(plan.token_capacity);
        }
    }
    assert!(retried.contains(&4) && retried.contains(&2) && retried.contains(&1));
    assert!(!retried.contains(&8));
    assert!(s.request(h).unwrap().accounting.committed_output_tokens > 0);
}

fn gpu_time(ns: u64) -> ExecutionTiming {
    let mut gpu_ns = [None; 16];
    gpu_ns[0] = Some(ns);
    ExecutionTiming {
        gpu_ns,
        ..Default::default()
    }
}

fn price_close(actual: utility::ServicePrice, expected: utility::ServicePrice) {
    assert!((actual.log_price(1.0) - expected.log_price(1.0)).abs() < 1e-12);
}

#[test]
fn service_uses_validated_progress_and_serial_execution_time_once() {
    let mut s = engine(1, false);
    let h = add(&mut s, 1, 4096, 32);
    step(&mut s, 0);
    let plan = s.plan().unwrap();
    assert_eq!(plan.kind, GraphKind::Prefill);
    let rate = s.service_rates(s.request(h).unwrap()).0.unwrap();
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    let mut expected = s.request(h).unwrap().accounting.service.tokens.unwrap();
    expected.observe(
        500.0,
        plan.lanes[0].tokens as f64,
        rate,
        s.policy.adaptation_ms,
    );
    let mut timing = gpu_time(2 * MS);
    timing.gpu_ns[1] = Some(3 * MS);
    timing.wall_ns = 500 * MS;
    s.advance_to(s.now_ns + timing.wall_ns).unwrap();
    complete_timed(&mut s, &plan, fence, 0, true, timing);
    price_close(
        s.request(h).unwrap().accounting.service.tokens.unwrap(),
        expected,
    );
    s.advance_to(s.now_ns + 100 * MS).unwrap();
    price_close(
        s.request(h).unwrap().accounting.service.tokens.unwrap(),
        expected,
    );
}

#[test]
fn pipelined_frames_use_ordered_completion_deltas_and_one_delivery_per_frame() {
    let mut s = engine(1, false);
    let h = add(&mut s, 1, 65, 32);
    ready(&mut s, &[h]);
    let (token_rate, delivery_rate) = s.service_rates(s.request(h).unwrap());
    let first = s.plan().unwrap();
    let a = Fence(s.submitted.0 + 1);
    s.commit(&first, a).unwrap();
    let second = s.plan().unwrap();
    let b = Fence(a.0 + 1);
    s.commit(&second, b).unwrap();
    let before = s.request(h).unwrap().accounting;
    let mut tokens = before.service.tokens.unwrap();
    let mut deliveries = before.service.deliveries.unwrap();
    for (plan, fence, ms) in [(&first, a, 2), (&second, b, 3)] {
        tokens.observe(ms as f64, 1.0, token_rate.unwrap(), s.policy.adaptation_ms);
        deliveries.observe(
            ms as f64,
            1.0,
            delivery_rate.unwrap(),
            s.policy.adaptation_ms,
        );
        complete_timed(&mut s, plan, fence, 0, true, gpu_time(ms * MS));
        let after = s.request(h).unwrap().accounting;
        price_close(after.service.tokens.unwrap(), tokens);
        price_close(after.service.deliveries.unwrap(), deliveries);
    }
    assert_eq!(
        s.request(h).unwrap().accounting.published_deliveries,
        before.published_deliveries + 2
    );
}

#[test]
fn speculative_token_volume_and_visible_burst_service_have_separate_prices() {
    let mut s = engine(1, true);
    s.fixed_speculative_width(Some(8)).unwrap();
    let h = add(&mut s, 1, 65, 64);
    ready(&mut s, &[h]);
    let (token_rate, delivery_rate) = s.service_rates(s.request(h).unwrap());
    let plan = s.plan().unwrap();
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    let before = s.request(h).unwrap().accounting.service;
    let mut token_price = before.tokens.unwrap();
    token_price.observe(30.0, 8.0, token_rate.unwrap(), s.policy.adaptation_ms);
    let mut hidden_price = before.deliveries.unwrap();
    hidden_price.observe(30.0, 0.0, delivery_rate.unwrap(), s.policy.adaptation_ms);
    complete_timed(&mut s, &plan, fence, 7, false, gpu_time(30 * MS));
    price_close(
        s.request(h).unwrap().accounting.service.tokens.unwrap(),
        token_price,
    );
    price_close(
        s.request(h).unwrap().accounting.service.deliveries.unwrap(),
        hidden_price,
    );
    let mut published_price = before.deliveries.unwrap();
    published_price.observe(30.0, 1.0, delivery_rate.unwrap(), s.policy.adaptation_ms);
    for _ in 0..8 {
        s.published_output(h, fence).unwrap();
        price_close(
            s.request(h).unwrap().accounting.service.deliveries.unwrap(),
            published_price,
        );
    }
}

#[test]
fn invalid_completion_changes_neither_prices_nor_cost_estimates() {
    let mut s = engine(1, false);
    let h = add(&mut s, 1, 65, 32);
    ready(&mut s, &[h]);
    let plan = s.plan().unwrap();
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    let before = s.request(h).unwrap().accounting.service;
    let cost = s.estimated_duration_ns(&plan);
    let mut records = completion_records(&s, &plan, fence, 0);
    records[h.slot as usize].generation += 1;
    assert!(s.complete(fence, &records, gpu_time(100 * MS)).is_err());
    price_close(
        s.request(h).unwrap().accounting.service.tokens.unwrap(),
        before.tokens.unwrap(),
    );
    price_close(
        s.request(h).unwrap().accounting.service.deliveries.unwrap(),
        before.deliveries.unwrap(),
    );
    assert_eq!(s.estimated_duration_ns(&plan), cost);
    assert_eq!(s.submitted_step(fence).unwrap(), &plan.step);
}

#[test]
fn missing_device_timing_does_not_substitute_host_or_estimated_time() {
    for gpu_ns in [None, Some(0)] {
        let mut s = engine(1, false);
        let h = add(&mut s, 1, 65, 32);
        ready(&mut s, &[h]);
        let plan = s.plan().unwrap();
        let fence = Fence(s.submitted.0 + 1);
        s.commit(&plan, fence).unwrap();
        let before = s.request(h).unwrap().accounting.service;
        let mut timing = gpu_time(0);
        timing.gpu_ns[0] = gpu_ns;
        timing.wall_ns = 500 * MS;
        complete_timed(&mut s, &plan, fence, 0, true, timing);
        price_close(
            s.request(h).unwrap().accounting.service.tokens.unwrap(),
            before.tokens.unwrap(),
        );
        price_close(
            s.request(h).unwrap().accounting.service.deliveries.unwrap(),
            before.deliveries.unwrap(),
        );
    }
}

#[test]
fn an_arrival_during_a_graph_is_not_charged_for_unavailable_service() {
    let mut s = engine(2, false);
    let first = add(&mut s, 1, 65, 32);
    ready(&mut s, &[first]);
    let plan = s.plan().unwrap();
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    let arrival = add(&mut s, 2, 65, 32);
    complete_timed(&mut s, &plan, fence, 0, true, gpu_time(10 * MS));
    assert!(
        s.request(arrival)
            .unwrap()
            .accounting
            .service
            .tokens
            .is_none()
    );
}

#[test]
fn service_snapshot_does_not_update_a_reused_request_slot() {
    let mut s = engine(2, false);
    let first = add(&mut s, 1, 65, 32);
    ready(&mut s, &[first]);
    let plan = s.plan().unwrap();
    let cancelled = add(&mut s, 2, 65, 32);
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    assert!(
        s.request(cancelled)
            .unwrap()
            .accounting
            .service
            .tokens
            .is_some()
    );
    s.cancel(cancelled).unwrap();
    let replacement = add(&mut s, 3, 65, 32);
    assert_eq!(replacement.slot, cancelled.slot);
    assert_ne!(replacement.generation, cancelled.generation);
    complete_timed(&mut s, &plan, fence, 0, true, gpu_time(10 * MS));
    assert!(
        s.request(replacement)
            .unwrap()
            .accounting
            .service
            .tokens
            .is_none()
    );
}

#[test]
fn stale_costs_remain_available_as_service_normalizers() {
    let mut s = engine(1, true);
    let h = add(&mut s, 1, 65, 32);
    ready(&mut s, &[h]);
    assert!(s.service_rates(s.request(h).unwrap()).0.is_some());
    let rates = s.service_rates(s.request(h).unwrap());
    s.advance_to(s.now_ns + s.policy.estimate_ttl_ns).unwrap();
    assert_eq!(s.service_rates(s.request(h).unwrap()), rates);
}

#[test]
fn unserved_prices_grow_with_device_time_and_inbound_waiters_are_excluded() {
    let mut s = engine(3, false);
    let first = add(&mut s, 1, 65, 64);
    ready(&mut s, &[first]);
    let plan = s.plan().unwrap();
    let waiting = add(&mut s, 2, 4096, 64);
    let inbound = add(&mut s, 3, 4096, 64);
    s.request_mut(inbound).unwrap().phase = RequestPhase::Inbound;
    let rate = s.service_rates(s.request(waiting).unwrap()).0.unwrap();
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    let mut expected = s
        .request(waiting)
        .unwrap()
        .accounting
        .service
        .tokens
        .unwrap();
    expected.observe(50.0, 0.0, rate, s.policy.adaptation_ms);
    complete_timed(&mut s, &plan, fence, 0, true, gpu_time(50 * MS));
    price_close(
        s.request(waiting)
            .unwrap()
            .accounting
            .service
            .tokens
            .unwrap(),
        expected,
    );
    assert!(
        s.request(inbound)
            .unwrap()
            .accounting
            .service
            .tokens
            .is_none()
    );
    assert!(
        s.request(inbound)
            .unwrap()
            .accounting
            .service
            .deliveries
            .is_none()
    );
}

#[test]
fn price_updates_preserve_an_unrelated_immutable_plan() {
    let mut s = engine(2, false);
    let first = add(&mut s, 1, 65, 64);
    ready(&mut s, &[first]);
    let plan = s.plan().unwrap();
    let second = add(&mut s, 2, 4096, 64);
    s.request_mut(second).unwrap().phase = RequestPhase::Prefill;
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    let other = s.phase_plan(GraphKind::Prefill, Some(second));
    assert_eq!(other.lanes.len(), 1);
    assert_eq!(other.lanes[0].request, second);
    complete_timed(&mut s, &plan, fence, 0, true, gpu_time(MS));
    s.commit(&other, Fence(fence.0 + 1)).unwrap();
}

#[test]
fn late_publication_cannot_rewind_prices_past_another_completion() {
    let mut s = engine(1, false);
    let h = add(&mut s, 1, 65, 64);
    ready(&mut s, &[h]);
    let first = s.plan().unwrap();
    let a = Fence(s.submitted.0 + 1);
    s.commit(&first, a).unwrap();
    complete_timed(&mut s, &first, a, 0, false, gpu_time(MS));
    let second = s.plan().unwrap();
    let b = Fence(a.0 + 1);
    s.commit(&second, b).unwrap();
    complete_timed(&mut s, &second, b, 0, false, gpu_time(MS));
    let before = s.request(h).unwrap().accounting.service.deliveries.unwrap();
    assert!(s.published_output(h, a).is_err());
    price_close(
        s.request(h).unwrap().accounting.service.deliveries.unwrap(),
        before,
    );
    s.published_output(h, b).unwrap();
}

#[test]
fn first_published_output_initializes_delivery_history_and_preserves_token_price() {
    let mut s = engine(1, false);
    let h = add(&mut s, 1, 65, 64);
    step(&mut s, 0);
    let prefill = s.plan().unwrap();
    let a = Fence(s.submitted.0 + 1);
    s.commit(&prefill, a).unwrap();
    let mut tokens = s.request(h).unwrap().accounting.service.tokens.unwrap();
    let rate = s.service_rates(s.request(h).unwrap()).0.unwrap();
    tokens.observe(3.0, 66.0, rate, s.policy.adaptation_ms);
    complete_timed(&mut s, &prefill, a, 0, true, gpu_time(3 * MS));
    let service = s.request(h).unwrap().accounting.service;
    price_close(service.tokens.unwrap(), tokens);
    let deliveries = utility::ServicePrice::initial(1.0 / 3.0);
    price_close(service.deliveries.unwrap(), deliveries);
    let snapshot = s.plan().unwrap();
    assert_eq!(snapshot.kind, GraphKind::Snapshot);
    s.commit(&snapshot, Fence(a.0 + 1)).unwrap();
    let service = s.request(h).unwrap().accounting.service;
    price_close(service.tokens.unwrap(), tokens);
    price_close(service.deliveries.unwrap(), deliveries);
}

fn cold_engine(active: usize, speculative: bool) -> RequestScheduler {
    let mut s = engine(active, speculative);
    s.graph_durations(&super::tests::duration_graphs(active))
        .unwrap();
    s
}

fn measured_step(s: &mut RequestScheduler, duration_ns: u64, accepted: usize) -> ScheduledPlan {
    let plan = s.plan().expect("runnable request must make progress");
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    s.advance_to(s.now_ns + duration_ns).unwrap();
    complete_timed(s, &plan, fence, accepted, true, gpu_time(duration_ns));
    plan
}

fn interactive_engine(active: usize, speculative: bool) -> RequestScheduler {
    let mut s = RequestScheduler::new(
        [17; 32],
        SchedulerLimits {
            max_active: active,
            request_slots: active.next_power_of_two(),
            max_sequence_tokens: 65536,
            prefill_chunk_size: 4096,
            pages: 32768,
            snapshots: 256,
        },
        SchedulerCapabilities {
            completion_token_capacity: 16,
            vocabulary: 1024,
            page_tokens: 64,
            vision: crate::model::VisionCapabilities::disabled(),
            draft_pages_per_request: 0,
            completion_inputs: vec![],
            speculative_allocations: vec![],
            state_rows_per_request: None,
        },
    )
    .unwrap();
    s.prefill_graphs(&[64, 128, 256, 512, 1024, 2048, 4096])
        .unwrap();
    if speculative {
        s.enable_speculation(&[1, 3, 8]).unwrap();
    }
    let mut graphs = super::tests::duration_graphs(active);
    for g in &mut graphs {
        g.estimated_duration_ns = match g.kind {
            GraphKind::Prefill => 2 * MS + g.token_capacity as u64 * MS / 64,
            GraphKind::Decode => (15 + 2 * (g.batch_capacity as u64 - 1)) * MS,
            GraphKind::Speculative => {
                let solo = match g.token_capacity {
                    1 => 15 * MS,
                    3 => 17_300_000,
                    8 => 30 * MS,
                    n => (15 + 2 * n as u64) * MS,
                };
                solo + 2 * (g.batch_capacity as u64 - 1) * MS
            }
            _ => MS / 10,
        };
    }
    s.measured_graph_durations(&graphs).unwrap();
    s
}

fn set_cost(s: &mut RequestScheduler, kind: GraphKind, batch: usize, capacity: usize, nanos: u64) {
    let step = ScheduledStep {
        draft_tokens: if kind == GraphKind::Speculative { capacity - 1 } else { 0 },
        packed_token_capacity: if kind == GraphKind::Speculative { batch * capacity } else { 0 },
        kind,
        batch_capacity: batch,
        token_capacity: capacity,
        mixed: None,
        lanes: Vec::new(),
        requires_feedback: false,
    };
    s.costs.set_measured(&step, nanos);
}

#[test]
fn long_conversations_use_work_budget_and_cross_unselected_messages() {
    let mut s = interactive_engine(32, false);
    let mut handles = Vec::new();
    for id in 1..=4 {
        handles.push(s.admit(RequestSpec {
            prompt: vec![id; 32768],
            message_boundaries: (32..32768).step_by(32).collect(),
            conditioning: vec![],
            max_new_tokens: 2,
            grammar_constrained: false,
            host_stops: false,
            prompt_logprobs: false,
            growable_output: false,
        }).unwrap());
    }
    let mut full_chunks = 0;
    let mut prefill_calls = 0;
    for _ in 0..10000 {
        if handles.iter().all(|&h| s.request(h).is_err()) {
            break;
        }
        let plan = step(&mut s, 0);
        if plan.kind == GraphKind::Prefill {
            prefill_calls += 1;
            full_chunks += usize::from(plan.lanes[0].tokens == 4096);
        }
    }
    assert!(handles.iter().all(|&h| s.request(h).is_err()));
    assert!(s.stats().message_checkpoints >= 4);
    assert!(s.stats().checkpoint_extra_work_ns <= 4 * (8 * 66 + 15) * MS / 20);
    assert!(full_chunks >= 4 * 6);
    assert!(prefill_calls <= 4 * 24, "{prefill_calls} prefill calls, {full_chunks} full chunks");
    assert_eq!(s.stats().snapshot_evictions, 0);
}

#[test]
fn one_two_and_sixteen_long_prefills_use_full_efficient_chunks() {
    for count in [1, 2, 16] {
        let mut s = interactive_engine(count, true);
        let handles: Vec<_> = (1..=count as u32)
            .map(|id| add(&mut s, id, 32768, 128))
            .collect();
        let mut chunks = 0;
        while chunks < count * 3 {
            let plan = step(&mut s, 0);
            if plan.kind == GraphKind::Prefill {
                assert_eq!((plan.token_capacity, plan.lanes[0].tokens), (4096, 4096));
                chunks += 1;
            }
            assert!(
                handles
                    .iter()
                    .all(|h| s.request(*h).unwrap().generated_tokens == 0)
            );
        }
        assert!(
            handles
                .iter()
                .all(|h| s.request(*h).unwrap().consumed_tokens > 0)
        );
    }
}

#[test]
fn small_prefill_tails_choose_the_best_effective_value_rate() {
    for (prompt, capacity) in [(64, 64), (511, 512), (4095, 4096)] {
        let mut s = interactive_engine(1, false);
        add(&mut s, 1, prompt, 128);
        assert_eq!(step(&mut s, 0).kind, GraphKind::Init);
        let plan = s.plan().unwrap();
        assert_eq!(plan.kind, GraphKind::Prefill);
        assert_eq!(
            (plan.token_capacity, plan.lanes[0].tokens),
            (capacity, prompt)
        );
    }
}

#[test]
fn equivalent_prefill_tails_use_the_smaller_graph_despite_a_faster_padded_estimate() {
    let mut s = interactive_engine(1, false);
    set_cost(&mut s, GraphKind::Prefill, 1, 4096, MS);
    add(&mut s, 1, 64, 128);
    step(&mut s, 0);
    let plan = s.plan().unwrap();
    assert_eq!((plan.token_capacity, plan.lanes[0].tokens), (64, 64));
}

#[test]
fn efficient_thirty_millisecond_speculation_beats_seventeen_millisecond_rounds() {
    let mut s = interactive_engine(1, true);
    let h = add(&mut s, 1, 64, 2048);
    ready(&mut s, &[h]);
    for _ in 0..128 {
        s.request_mut(h).unwrap().acceptance.observe(speculation::ProposalSource::Mtp, 0, 7, 7, true, None);
    }
    let plan = s.plan().unwrap();
    assert_eq!(
        (plan.kind, plan.token_capacity, plan.estimated_duration_ns()),
        (GraphKind::Speculative, 8, 30 * MS)
    );
    for _ in 0..256 {
        s.request_mut(h).unwrap().acceptance.observe(speculation::ProposalSource::Mtp, 0, 7, 0, true, None);
    }
    assert!(s.plan().unwrap().token_capacity < 8);
}

#[test]
fn equivalent_decode_work_uses_the_smaller_batch_despite_a_faster_padded_estimate() {
    let mut s = interactive_engine(4, false);
    let h = add(&mut s, 1, 64, 1024);
    ready(&mut s, &[h]);
    set_cost(&mut s, GraphKind::Decode, 4, 1, 10 * MS);
    let plan = s.plan().unwrap();
    assert_eq!((plan.batch_capacity, plan.lanes.len()), (1, 1));
    assert_eq!(plan.lanes[0].request, h);
}

#[test]
fn speculative_cohorts_are_selected_by_expected_lane_value() {
    let mut s = interactive_engine(8, true);
    let handles: Vec<_> = (1..=8).map(|id| add(&mut s, id, 64, 2048)).collect();
    for (index, h) in handles.iter().enumerate() {
        let r = s.request_mut(*h).unwrap();
        r.phase = RequestPhase::Decode;
        r.consumed_tokens = 64;
        r.scheduled_consumed = 64;
        r.generated_tokens = 1;
        for _ in 0..128 {
            r.acceptance
                .observe(speculation::ProposalSource::Mtp, 0, 7, if index >= 6 { 7 } else { 0 }, true, None);
        }
    }
    for batch in [1, 2, 4, 8] {
        for width in [1, 3, 8] {
            set_cost(&mut s, GraphKind::Speculative, batch, width, 200 * MS);
        }
    }
    set_cost(&mut s, GraphKind::Speculative, 2, 8, 30 * MS);
    let plan = s.plan().unwrap();
    assert_eq!((plan.batch_capacity, plan.token_capacity), (2, 8));
    assert_eq!(
        plan.lanes.iter().map(|l| l.request).collect::<Vec<_>>(),
        handles[6..]
    );
}

#[test]
fn two_long_decodes_keep_both_streams_flowing() {
    for accepted in [0, 7] {
        let mut s = interactive_engine(2, true);
        let handles = [add(&mut s, 1, 64, 2048), add(&mut s, 2, 64, 2048)];
        let mut last = [None; 2];
        let mut gaps = [0; 2];
        for _ in 0..200 {
            let plan = step(&mut s, accepted);
            if plan.kind == GraphKind::Speculative {
                for lane in &plan.lanes {
                    let i = handles.iter().position(|h| *h == lane.request).unwrap();
                    if let Some(t) = last[i] {
                        gaps[i] = gaps[i].max(s.now_ns - t);
                    }
                    last[i] = Some(s.now_ns);
                }
            }
        }
        assert!(
            handles
                .iter()
                .all(|h| s.request(*h).unwrap().generated_tokens > 100)
        );
        assert!(gaps.iter().all(|g| *g <= 2 * 32 * MS), "gaps={gaps:?}");
    }
}

#[test]
fn long_prefill_and_low_acceptance_decode_both_make_progress() {
    let mut s = interactive_engine(2, true);
    let decode = add(&mut s, 1, 64, 4096);
    ready(&mut s, &[decode]);
    for _ in 0..128 {
        s.request_mut(decode)
            .unwrap()
            .acceptance
            .observe(speculation::ProposalSource::Mtp, 0, 7, 0, true, None);
    }
    let prefill = add(&mut s, 2, 32768, 128);
    let mut last = s.now_ns;
    let mut gap = 0;
    let mut prefill_chunks = 0;
    for _ in 0..400 {
        let plan = step(&mut s, 0);
        if plan.lanes.iter().any(|l| l.request == decode) && plan.kind == GraphKind::Speculative {
            gap = gap.max(s.now_ns - last);
            last = s.now_ns;
        }
        if plan.kind == GraphKind::Prefill {
            prefill_chunks += 1;
        }
        if s.request(prefill).is_ok_and(|r| r.consumed_tokens >= 32768) {
            break;
        }
    }
    assert!(s.request(prefill).unwrap().consumed_tokens >= 32768);
    assert!(s.request(decode).unwrap().generated_tokens > 1);
    gap = gap.max(s.now_ns - last);
    assert!(prefill_chunks >= 8);
    assert!(gap < 300 * MS, "output gap={} ms", gap / MS);
}

#[test]
fn prefill_frames_drain_even_when_the_next_request_is_independent() {
    let mut s = interactive_engine(2, false);
    add(&mut s, 1, 32768, 128);
    add(&mut s, 2, 32768, 128);
    step(&mut s, 0);
    let first = s.plan().unwrap();
    assert_eq!((first.kind, first.token_capacity), (GraphKind::Prefill, 4096));
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&first, fence).unwrap();
    assert!(s.plan().is_none());
    assert!(first.estimated_duration_ns() > 10 * MS);
    complete(&mut s, &first, fence, 0, false);
    let second = s.plan().unwrap();
    assert_eq!((second.kind, second.token_capacity), (GraphKind::Prefill, 4096));
    s.commit(&second, Fence(fence.0 + 1)).unwrap();
    assert!(s.plan().is_none());
}

#[test]
fn decode_waiting_selects_a_smaller_prefill_at_the_efficiency_knee() {
    let mut s = interactive_engine(5, true);
    s.prefill_graphs(&[64, 512, 4096]).unwrap();
    s.configure_policy(PolicyConfig {
        output_token_value: 16.0,
        token_fairness: 0.0,
        delivery_fairness: 0.0,
        decode_age_price: 0.04,
        ..s.policy
    })
    .unwrap();
    set_cost(&mut s, GraphKind::Prefill, 1, 512, 8 * MS);
    set_cost(&mut s, GraphKind::Prefill, 1, 4096, 62 * MS);
    let prefill = add(&mut s, 1, 32768, 128);
    step(&mut s, 0);
    let decodes: Vec<_> = (2..=5).map(|id| add(&mut s, id, 64, 2048)).collect();
    for h in &decodes {
        let now = s.now_ns;
        let r = s.request_mut(*h).unwrap();
        r.phase = RequestPhase::Decode;
        r.consumed_tokens = 64;
        r.scheduled_consumed = 64;
        r.generated_tokens = 1;
        r.accounting.last_output_ns = Some(now);
        for _ in 0..128 {
            r.acceptance.observe(speculation::ProposalSource::Mtp, 0, 7, 7, true, None);
        }
    }
    let smaller = s.plan().unwrap();
    assert_eq!(
        (smaller.kind, smaller.token_capacity),
        (GraphKind::Prefill, 512)
    );
    assert_eq!(smaller.lanes[0].request, prefill);
    s.advance_to(s.now_ns + 1000 * MS).unwrap();
    assert_eq!(s.plan().unwrap().kind, GraphKind::Speculative);
    for h in decodes {
        s.cancel(h).unwrap();
    }
    let larger = s.plan().unwrap();
    assert_eq!(
        (larger.kind, larger.token_capacity),
        (GraphKind::Prefill, 4096)
    );
}

#[test]
fn terminal_snapshot_completes_while_another_long_prompt_is_still_prefilling() {
    let mut s = interactive_engine(2, false);
    let short = add(&mut s, 1, 64, 1);
    step(&mut s, 0);
    step(&mut s, 0);
    assert_eq!(s.request(short).unwrap().phase, RequestPhase::Snapshot);
    let long = add(&mut s, 2, 32768, 128);
    let start = s.now_ns;
    while s.request(short).is_ok() && s.now_ns - start < 1000 * MS {
        step(&mut s, 0);
    }
    assert!(s.request(short).is_err());
    assert!(s.request(long).unwrap().consumed_tokens < 32768);
}

#[test]
fn five_and_sixteen_streams_receive_repeated_output_service() {
    for (count, accepted, known_acceptance) in [5, 16].into_iter().flat_map(|n| {
        [0, 3, 7]
            .into_iter()
            .flat_map(move |a| [false, true].map(|k| (n, a, k)))
    }) {
        let mut s = interactive_engine(count, true);
        let handles: Vec<_> = (1..=count as u32)
            .map(|id| add(&mut s, id, 64, 4096))
            .collect();
        if known_acceptance {
            for &h in &handles {
                for _ in 0..128 {
                    s.request_mut(h)
                        .unwrap()
                        .acceptance
                        .observe(speculation::ProposalSource::Mtp, 0, 7, accepted, true, None);
                }
            }
        }
        let mut first = vec![None; count];
        let mut last = vec![None; count];
        let mut gap = vec![0; count];
        for _ in 0..150 {
            let plan = step(&mut s, accepted);
            if matches!(plan.kind, GraphKind::Prefill | GraphKind::Speculative) {
                for lane in &plan.lanes {
                    let i = handles.iter().position(|h| *h == lane.request).unwrap();
                    first[i].get_or_insert(s.now_ns);
                    if let Some(at) = last[i] {
                        gap[i] = gap[i].max(s.now_ns - at);
                    }
                    last[i] = Some(s.now_ns);
                }
            }
        }
        for (i, at) in last.iter().enumerate() {
            gap[i] = gap[i].max(s.now_ns - at.unwrap_or(0));
        }
        assert!(
            first.iter().all(|t| t.is_some_and(|t| t < 100 * MS)),
            "count={count} accepted={accepted} known={known_acceptance} first={first:?}"
        );
        assert!(
            gap.iter().all(|g| *g < 150 * MS),
            "count={count} accepted={accepted} known={known_acceptance} gaps={gap:?}"
        );
        let before: Vec<_> = handles
            .iter()
            .map(|h| s.request(*h).unwrap().generated_tokens)
            .collect();
        for _ in 0..100 {
            step(&mut s, accepted);
        }
        let progress: Vec<_> = handles
            .iter()
            .enumerate()
            .map(|(i, h)| s.request(*h).unwrap().generated_tokens - before[i])
            .collect();
        assert!(
            *progress.iter().min().unwrap() >= 50,
            "progress={progress:?}"
        );
        assert!(
            progress.iter().max().unwrap() - progress.iter().min().unwrap() <= 8,
            "count={count} accepted={accepted} known={known_acceptance} progress={progress:?}"
        );
    }
}

#[test]
fn delivery_reference_uses_batch_throughput_and_excludes_prefill_only_requests() {
    let mut s = interactive_engine(16, true);
    let handles: Vec<_> = (1..=2).map(|id| add(&mut s, id, 64, 4096)).collect();
    ready(&mut s, &handles);
    let references = |s: &RequestScheduler| {
        let rates = std::array::from_fn(|slot| {
            s.slots
                .get(slot)
                .and_then(|s| s.request.as_ref())
                .and_then(|r| s.service_rates(r).1)
        });
        s.delivery_reference_intervals(&rates)
    };
    assert_eq!(references(&s)[handles[0].slot as usize], Some(17.0));
    for id in 3..=16 {
        add(&mut s, id, 32768, 128);
    }
    assert_eq!(references(&s)[handles[0].slot as usize], Some(17.0));
    set_cost(&mut s, GraphKind::Speculative, 16, 1, MS);
    assert_eq!(references(&s)[handles[0].slot as usize], Some(1.0));
    let retained = references(&s);
    s.advance_to(s.now_ns + s.policy.estimate_ttl_ns).unwrap();
    assert_eq!(references(&s), retained);
}

#[test]
fn snapshot_enabling_a_shared_decode_batch_competes_by_the_complete_sequence_cost() {
    let mut s = interactive_engine(16, true);
    let handles: Vec<_> = (1..=16).map(|id| add(&mut s, id, 64, 4096)).collect();
    for &h in &handles {
        for _ in 0..128 {
            s.request_mut(h).unwrap().acceptance.observe(speculation::ProposalSource::Mtp, 0, 7, 7, true, None);
        }
    }
    for _ in 0..200 {
        let decoding = handles
            .iter()
            .filter(|h| s.request(**h).unwrap().phase == RequestPhase::Decode)
            .count();
        let copying = handles
            .iter()
            .filter(|h| s.request(**h).unwrap().phase == RequestPhase::Snapshot)
            .count();
        if decoding == 15 && copying == 1 {
            break;
        }
        step(&mut s, 7);
    }
    let plan = s.plan().unwrap();
    assert_eq!(plan.kind, GraphKind::Snapshot);
    assert_eq!(plan.lanes.len(), 1);
    assert_eq!(plan.estimated_duration_ns(), MS / 10);
    let handle = plan.lanes[0].request;
    let tokens = s.request(handle).unwrap().generated_tokens;
    for batch in crate::execution::BATCH_CAPACITIES
        .into_iter()
        .take_while(|b| *b <= 16)
    {
        set_cost(&mut s, GraphKind::Snapshot, batch, 1, 1000 * MS);
    }
    assert_eq!(s.plan().unwrap().kind, GraphKind::Speculative);
    for batch in crate::execution::BATCH_CAPACITIES
        .into_iter()
        .take_while(|b| *b <= 16)
    {
        set_cost(&mut s, GraphKind::Snapshot, batch, 1, MS / 10);
    }
    assert_eq!(step(&mut s, 7).kind, GraphKind::Snapshot);
    assert_eq!(s.request(handle).unwrap().generated_tokens, tokens);
    let batch = s.plan().unwrap();
    assert_eq!(batch.kind, GraphKind::Speculative);
    assert_eq!(batch.lanes.len(), 16);
}

#[test]
fn equally_costly_snapshot_can_finish_one_request_and_enable_another() {
    let mut s = interactive_engine(2, true);
    let terminal = add(&mut s, 1, 64, 1);
    let continuing = add(&mut s, 2, 64, 4096);
    step(&mut s, 7);
    for handle in [terminal, continuing] {
        let plan = s.phase_plan(GraphKind::Prefill, Some(handle));
        let fence = Fence(s.submitted.0 + 1);
        s.commit(&plan, fence).unwrap();
        s.advance_to(s.now_ns + plan.estimated_duration_ns())
            .unwrap();
        complete_timed(
            &mut s,
            &plan,
            fence,
            7,
            true,
            gpu_time(plan.estimated_duration_ns()),
        );
    }
    let plan = s.plan().unwrap();
    assert_eq!(plan.kind, GraphKind::Snapshot);
    assert_eq!(plan.lanes.len(), 2);
    step(&mut s, 7);
    assert!(s.request(terminal).is_err());
    assert_eq!(s.request(continuing).unwrap().phase, RequestPhase::Decode);
    assert_eq!(s.request(continuing).unwrap().generated_tokens, 1);
}

#[test]
fn prerequisite_cohort_accounts_for_ready_batch_mates() {
    let mut s = interactive_engine(16, true);
    s.configure_policy(PolicyConfig {
        token_fairness: 0.0,
        delivery_fairness: 0.0,
        decode_age_price: 0.0,
        ..Default::default()
    })
    .unwrap();
    let execute = |s: &mut RequestScheduler, kind, handle| {
        let plan = s.phase_plan(kind, Some(handle));
        let fence = Fence(s.submitted.0 + 1);
        s.commit(&plan, fence).unwrap();
        s.advance_to(s.now_ns + plan.estimated_duration_ns())
            .unwrap();
        complete_timed(
            s,
            &plan,
            fence,
            7,
            true,
            gpu_time(plan.estimated_duration_ns()),
        );
    };
    let ready_handles: Vec<_> = (1..=14).map(|id| add(&mut s, id, 64, 4096)).collect();
    for &handle in &ready_handles {
        for kind in [GraphKind::Init, GraphKind::Prefill, GraphKind::Snapshot] {
            execute(&mut s, kind, handle);
        }
    }
    for &handle in &ready_handles {
        for _ in 0..128 {
            s.request_mut(handle)
                .unwrap()
                .acceptance
                .observe(speculation::ProposalSource::Mtp, 0, 7, 7, true, None);
        }
    }
    let decode = add(&mut s, 15, 64, 4096);
    let prefill = s
        .admit(RequestSpec {
            prompt: vec![16; 32768],
            message_boundaries: vec![512],
            conditioning: vec![],
            max_new_tokens: 128,
            grammar_constrained: false,
            host_stops: false,
            prompt_logprobs: false,
            growable_output: false,
        })
        .unwrap();
    for handle in [decode, prefill] {
        for kind in [GraphKind::Init, GraphKind::Prefill] {
            execute(&mut s, kind, handle);
        }
    }
    for batch in [2, 4, 8, 16] {
        set_cost(&mut s, GraphKind::Snapshot, batch, 1, 1000 * MS);
    }
    let plan = s.plan().unwrap();
    assert_eq!(plan.kind, GraphKind::Snapshot);
    assert_eq!(plan.batch_capacity, 1);
    assert_eq!(plan.lanes[0].request, decode);
}

#[test]
fn invalid_utility_configuration_is_rejected_before_admission() {
    for fraction in [-0.1, 1.1, f64::INFINITY, f64::NAN] {
        assert!(engine(1, false).configure_policy(PolicyConfig {
            checkpoint_extra_work_fraction: fraction, ..Default::default()
        }).is_err());
    }
    for policy in [
        PolicyConfig {
            output_token_value: f64::INFINITY,
            ..Default::default()
        },
        PolicyConfig {
            token_fairness: -1.0,
            ..Default::default()
        },
        PolicyConfig {
            delivery_fairness: f64::NAN,
            ..Default::default()
        },
        PolicyConfig {
            decode_age_price: -1.0,
            ..Default::default()
        },
        PolicyConfig {
            terminal_value: -1.0,
            ..Default::default()
        },
        PolicyConfig {
            terminal_age_price: f64::NAN,
            ..Default::default()
        },
        PolicyConfig {
            terminal_value: f64::INFINITY,
            ..Default::default()
        },
        PolicyConfig {
            adaptation_ms: 0.0,
            ..Default::default()
        },
        PolicyConfig {
            estimate_ttl_ns: 0,
            ..Default::default()
        },
    ] {
        assert!(engine(1, false).configure_policy(policy).is_err());
    }
}

fn mixed_engine(
    active: usize,
    speculative: bool,
    shapes: &[(usize, usize, usize, u64)],
) -> RequestScheduler {
    let mut s = interactive_engine(active, speculative);
    let mut graphs = super::tests::duration_graphs(active);
    for g in &mut graphs {
        g.estimated_duration_ns = s
            .costs
            .duration(g.kind, g.batch_capacity, g.token_capacity)
            .unwrap();
    }
    for &(batch, width, prefill, nanos) in shapes {
        graphs.push(crate::execution::GraphDescriptor { tuned_bindings: vec![], confidence_width: 0, draft_tokens: 0, packed_token_capacity: 0,
            name: format!("mixed-{batch}-{prefill}-{width}"),
            kind: GraphKind::Mixed,
            sampling_mode: SamplingMode::General,
            handle: graphs.len() + 1,
            batch_capacity: batch,
            token_capacity: width,
            prefill_capacity: prefill,
            estimated_duration_ns: nanos,
            input_bindings: vec![],
        });
    }
    s.measured_graph_durations(&graphs).unwrap();
    s.mixed_graphs(
        &shapes
            .iter()
            .map(|&(b, w, p, _)| (b, w, p))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    s
}

fn throughput_only(s: &mut RequestScheduler) {
    s.configure_policy(PolicyConfig {
        token_fairness: 0.0,
        delivery_fairness: 0.0,
        decode_age_price: 0.0,
        ..Default::default()
    })
    .unwrap();
}

fn initialize(s: &mut RequestScheduler, handle: RequestHandle) {
    let plan = s.phase_plan(GraphKind::Init, Some(handle));
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    complete(s, &plan, fence, 0, true);
}

#[test]
fn mixed_prefill_exploration_requires_additional_useful_tokens() {
    for prompt_tokens in [0, 17, 512] {
        let mut s = mixed_engine(2, false, &[(2, 1, 64, MS), (2, 1, 512, 0)]);
        throughput_only(&mut s);
        let decoder = add(&mut s, 1, 64, 1024);
        ready(&mut s, &[decoder]);
        if prompt_tokens > 0 {
            let prompt = add(&mut s, 2, prompt_tokens, 128);
            initialize(&mut s, prompt);
        }
        let plan = s.plan().unwrap();
        assert_eq!(plan.kind, GraphKind::Mixed);
        assert_eq!(plan.prefill_capacity(), if prompt_tokens > 64 { 512 } else { 64 });
    }
}

#[test]
fn mixed_value_selects_an_efficiency_knee_and_a_fitting_prompt_tail() {
    for prompt_tokens in [64, 8192] {
        let mut s = mixed_engine(
            2,
            false,
            &[
                (2, 1, 64, 2 * MS),
                (2, 1, 512, 8 * MS),
                (2, 1, 4096, 80 * MS),
            ],
        );
        throughput_only(&mut s);
        let d = add(&mut s, 1, 64, 1024);
        ready(&mut s, &[d]);
        let p = add(&mut s, 2, prompt_tokens, 128);
        initialize(&mut s, p);
        let plan = s.plan().unwrap();
        assert_eq!(plan.kind, GraphKind::Mixed);
        assert_eq!(
            plan.prefill_capacity(),
            if prompt_tokens == 64 { 64 } else { 512 }
        );
        assert_eq!(plan.mixed.as_ref().unwrap().prefill_row, Some(1));
        assert_eq!(plan.lanes[0].request, d);
        assert_eq!(plan.lanes[0].tokens, 1);
        assert_eq!(plan.lanes[1].request, p);
        assert_eq!(plan.lanes[1].tokens, plan.prefill_capacity());
        let expected =
            (plan.lanes[1].tokens as f64 + 64.0 + if prompt_tokens == 64 { 64.0 } else { 0.0 })
                / (plan.estimated_duration_ns() as f64 / 1e6);
        assert!(
            (plan.selection_value_rate() * plan.selection_log_scale().exp() - expected).abs()
                < 1e-10
        );
    }
}

#[test]
fn empty_mixed_catalog_completes_overlapping_prompt_and_generation() {
    for speculative in [false, true] {
        let mut s = mixed_engine(2, speculative, &[]);
        let decoder = add(&mut s, 1, 64, 128);
        ready(&mut s, &[decoder]);
        let prompt = add(&mut s, 2, 8192, 128);
        initialize(&mut s, prompt);
        let mut prefill = false;
        let mut generation = false;
        for _ in 0..1000 {
            if s.slots.iter().all(|slot| slot.request.is_none()) {
                break;
            }
            let plan = step(&mut s, if speculative { 3 } else { 0 });
            assert_ne!(plan.kind, GraphKind::Mixed);
            prefill |= plan.kind == GraphKind::Prefill;
            generation |= matches!(plan.kind, GraphKind::Decode | GraphKind::Speculative);
        }
        assert!(prefill && generation);
        assert!(s.slots.iter().all(|slot| slot.request.is_none()));
    }
}

#[test]
fn mixed_shapes_can_win_with_either_phase_unoccupied() {
    let mut s = mixed_engine(2, false, &[(2, 1, 4096, MS)]);
    throughput_only(&mut s);
    let p = add(&mut s, 1, 8192, 1024);
    initialize(&mut s, p);
    let plan = s.plan().unwrap();
    assert_eq!(plan.kind, GraphKind::Mixed);
    assert_eq!(plan.lanes.len(), 1);
    assert_eq!(plan.lanes[0].tokens, 4096);
    assert_eq!(plan.mixed.as_ref().unwrap().prefill_row, Some(0));
    ready(&mut s, &[p]);
    let plan = s.plan().unwrap();
    assert_eq!(plan.kind, GraphKind::Mixed);
    assert_eq!(plan.lanes.len(), 1);
    assert_eq!(plan.mixed.as_ref().unwrap().prefill_row, None);
    assert_eq!(plan.lane_kind(0), GraphKind::Decode);
}

#[test]
fn mixed_prefill_and_verification_commit_distinct_progress_and_reject_bad_records_atomically() {
    let mut s = mixed_engine(2, true, &[(2, 8, 512, MS)]);
    let d = add(&mut s, 1, 64, 1024);
    ready(&mut s, &[d]);
    let p = add(&mut s, 2, 8192, 128);
    initialize(&mut s, p);
    let plan = s.plan().unwrap();
    assert_eq!(plan.kind, GraphKind::Mixed);
    assert_eq!(plan.token_capacity, 8);
    let consumed = s.request(d).unwrap().consumed_tokens;
    let output = s.request(d).unwrap().generated_tokens;
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    assert!(s.plan().is_none());
    let records = completion_records(&s, &plan, fence, 0);
    let mut broken = records.clone();
    broken[p.slot as usize].consumed_tokens -= 1;
    assert!(s.complete(fence, &broken, gpu_time(200 * MS)).is_err());
    assert_eq!(s.request(d).unwrap().consumed_tokens, consumed);
    assert_eq!(s.request(p).unwrap().consumed_tokens, 0);
    assert_eq!(s.estimated_duration_ns(&plan), Some(MS));
    broken = records;
    broken[d.slot as usize]
        .vector
        .as_mut()
        .unwrap()
        .verification_width = 1;
    assert!(s.complete(fence, &broken, gpu_time(200 * MS)).is_err());
    complete_timed(&mut s, &plan, fence, 0, true, gpu_time(2 * MS));
    assert_eq!(s.request(p).unwrap().consumed_tokens, 512);
    assert_eq!(
        s.request(p).unwrap().accounting.committed_prefill_tokens,
        512
    );
    assert_eq!(s.request(p).unwrap().generated_tokens, 0);
    assert_eq!(s.request(d).unwrap().consumed_tokens, consumed + 1);
    assert_eq!(s.request(d).unwrap().generated_tokens, output + 1);
    assert_eq!(
        s.request(d).unwrap().accounting.committed_prefill_tokens,
        64
    );
    assert_eq!(s.request(d).unwrap().pending, 0);
}

#[test]
fn mixed_prompt_tail_publishes_first_output_and_preserves_its_snapshot_boundary() {
    let mut s = mixed_engine(2, true, &[(2, 8, 64, MS)]);
    let d = add(&mut s, 1, 64, 1024);
    ready(&mut s, &[d]);
    let p = add(&mut s, 2, 64, 128);
    initialize(&mut s, p);
    let output = s.request(d).unwrap().generated_tokens;
    let plan = step(&mut s, 3);
    assert_eq!(plan.kind, GraphKind::Mixed);
    let request = s.request(p).unwrap();
    assert_eq!(request.generated_tokens, 1);
    assert_eq!(request.accounting.published_deliveries, 1);
    assert_eq!(request.phase, RequestPhase::Snapshot);
    assert_eq!(
        request.boundary_capture.unwrap().resume,
        RequestPhase::Decode
    );
    assert_eq!(s.request(d).unwrap().generated_tokens, output + 4);
    ready(&mut s, &[p, d]);
    assert!(s.plan().unwrap().lanes.iter().all(|l| l.tokens <= 8));
}

#[test]
fn mixed_cost_observations_and_expiry_are_specific_to_both_token_capacities() {
    let mut s = mixed_engine(
        2,
        true,
        &[(2, 1, 64, MS), (2, 8, 64, 2 * MS), (2, 8, 512, 3 * MS)],
    );
    let make = |width, prefill| ScheduledStep { draft_tokens: 0, packed_token_capacity: 0,
        kind: GraphKind::Mixed,
        batch_capacity: 2,
        token_capacity: width,
        mixed: Some(MixedStep {
            prefill_capacity: prefill,
            prefill_row: None,
            speculative: true,
        }),
        lanes: vec![],
        requires_feedback: true,
    };
    s.advance_to(10 * MS).unwrap();
    s.observe_execution(&make(8, 64), gpu_time(10 * MS));
    assert_eq!(s.estimated_duration_ns(&make(1, 64)), Some(MS));
    assert_eq!(s.estimated_duration_ns(&make(8, 64)), Some(6 * MS));
    assert_eq!(s.estimated_duration_ns(&make(8, 512)), Some(3 * MS));
    s.advance_to(s.policy.estimate_ttl_ns).unwrap();
    assert_eq!(s.estimated_duration_ns(&make(8, 512)), Some(3 * MS));
    assert!(!s.costs.is_fresh(cost::Key::from_step(&make(8, 512))));
    assert_eq!(s.estimated_duration_ns(&make(8, 64)), Some(6 * MS));
    s.advance_to(s.policy.estimate_ttl_ns + 10 * MS).unwrap();
    assert_eq!(s.estimated_duration_ns(&make(8, 64)), Some(6 * MS));
    s.observe_execution(&make(8, 64), gpu_time(MS));
    assert_eq!(s.estimated_duration_ns(&make(8, 64)), Some(MS));
}

#[test]
fn mixed_and_separate_selection_matches_exhaustive_useful_work_per_time() {
    #[derive(Clone, Copy)]
    struct Alternative { kind: u8, mask: u64, prompt: usize, batch: usize, prefill: usize, rate: f64 }
    for seed in 0..16usize {
        let shapes: Vec<_> = [2, 4, 8]
            .into_iter()
            .flat_map(|batch| {
                [64, 512, 4096].into_iter().map(move |p| {
                    (
                        batch,
                        1,
                        p,
                        (3 + (seed * 7 + batch * 3 + p / 64) % 79) as u64 * MS,
                    )
                })
            })
            .collect();
        let mut s = mixed_engine(6, false, &shapes);
        throughput_only(&mut s);
        for slot in 0..6 {
            let prompt = if slot < 2 {
                [64, 129, 320, 900][(seed + slot) % 4]
            } else {
                64
            };
            let h = add(&mut s, slot as u32 + 1, prompt, 1024);
            let r = s.request_mut(h).unwrap();
            if slot < 2 {
                r.phase = RequestPhase::Prefill;
            } else {
                r.phase = RequestPhase::Decode;
                r.consumed_tokens = 64;
                r.scheduled_consumed = 64;
                r.generated_tokens = 1;
            }
        }
        let mut alternatives = Vec::<Alternative>::new();
        let mut consider = |batch: usize, pcap: usize, ns: u64, allow_p: bool, allow_d: bool| {
            let mut best = [None::<Alternative>; 2];
            for mask in 1u64..64 {
                let ps = (mask & 3).count_ones() as usize;
                let ds = (mask & !3).count_ones() as usize;
                if ps > 1 || ps + ds > batch || (!allow_p && ps > 0) || (!allow_d && ds > 0) {
                    continue;
                }
                let mut value = 64.0 * ds as f64;
                let mut prompt_work = 0;
                for slot in 0..2 {
                    if mask & (1 << slot) == 0 {
                        continue;
                    }
                    let prompt = s.slots[slot].request.as_ref().unwrap().prompt_tokens;
                    prompt_work = prompt.min(pcap);
                    value += prompt.min(pcap) as f64 + if prompt <= pcap { 64.0 } else { 0.0 };
                }
                let alternative = Alternative { kind: if allow_p && allow_d { 2 } else { u8::from(allow_p) },
                    mask, prompt: prompt_work, batch, prefill: pcap, rate: value * 1e6 / ns as f64 };
                let entry = &mut best[usize::from(ps != 0)];
                if entry.is_none_or(|old| alternative.rate > old.rate
                    || (alternative.rate == old.rate && (mask.count_ones() > old.mask.count_ones()
                        || (mask.count_ones() == old.mask.count_ones() && mask < old.mask)))) {
                    *entry = Some(alternative);
                }
            }
            alternatives.extend(best.into_iter().flatten());
        };
        for &p in &s.prefill_catalog {
            consider(
                1,
                p,
                s.costs.duration(GraphKind::Prefill, 1, p).unwrap(),
                true,
                false,
            );
        }
        for b in [1, 2, 4, 8] {
            consider(
                b,
                0,
                s.costs.duration(GraphKind::Decode, b, 1).unwrap(),
                false,
                true,
            );
        }
        for &(b, _, p, t) in &shapes {
            consider(b, p, t, true, true);
        }
        let expected = alternatives.iter().filter(|candidate| !alternatives.iter().any(|other| {
            other.kind == candidate.kind && other.mask == candidate.mask && other.prompt == candidate.prompt
                && other.batch <= candidate.batch && other.prefill <= candidate.prefill
                && (other.batch < candidate.batch || other.prefill < candidate.prefill)
        })).map(|candidate| candidate.rate).fold(0.0f64, f64::max);
        let plan = s.plan().unwrap();
        let actual = plan.selection_value_rate() * plan.selection_log_scale().exp();
        assert!(
            (actual - expected).abs() < 1e-9,
            "seed {seed}: {actual} != {expected}; {plan:?}"
        );
        assert_eq!(
            plan.lanes
                .iter()
                .enumerate()
                .filter(|(row, _)| plan.lane_kind(*row) == GraphKind::Prefill)
                .count(),
            usize::from(
                plan.kind == GraphKind::Prefill
                    || plan.mixed.as_ref().is_some_and(|m| m.prefill_row.is_some())
            )
        );
    }
}

#[test]
fn mixed_catalog_completes_common_concurrent_workloads_without_decode_starvation() {
    for (prefills, decoders, accepted) in [
        (2, 0, 0),
        (1, 1, 0),
        (1, 1, 7),
        (1, 4, 3),
        (1, 5, 3),
        (0, 2, 0),
        (0, 2, 7),
        (0, 16, 3),
    ] {
        let active: usize = prefills + decoders;
        let mut shapes = Vec::new();
        for b in crate::execution::BATCH_CAPACITIES
            .into_iter()
            .filter(|b| *b >= 2 && *b <= active.next_power_of_two())
        {
            for p in [64, 512, 4096] {
                for w in [1, 3, 8] {
                    let decode = match w {
                        1 => 15 * MS,
                        3 => 17_300_000,
                        _ => 30 * MS,
                    } + 2 * (b as u64 - 2) * MS;
                    shapes.push((b, w, p, (2 * MS + p as u64 * MS / 64 + decode) * 4 / 5));
                }
            }
        }
        let mut s = mixed_engine(active, true, &shapes);
        let handles: Vec<_> = (0..active)
            .map(|i| {
                add(
                    &mut s,
                    i as u32 + 1,
                    if i < prefills { 32768 } else { 64 },
                    128,
                )
            })
            .collect();
        let mut maximum_gap = 0;
        let mut launches = 0;
        while s.slots.iter().any(|slot| slot.request.is_some()) && launches < 20_000 {
            step(&mut s, accepted);
            launches += 1;
            for &h in &handles {
                if let Ok(r) = s.request(h) {
                    if r.generated_tokens < r.max_new_tokens {
                        if let Some(last) = r.accounting.last_output_ns {
                            maximum_gap = maximum_gap.max(s.now_ns - last);
                        }
                    }
                }
            }
        }
        assert_eq!(
            s.slots.iter().filter(|slot| slot.request.is_some()).count(),
            0,
            "prefills={prefills}, decoders={decoders}, accepted={accepted}"
        );
        assert!(
            maximum_gap < 1000 * MS,
            "prefills={prefills}, decoders={decoders}, accepted={accepted}, gap={maximum_gap}"
        );
    }
}

#[test]
fn mixed_prerequisite_values_a_new_prompt_with_ready_decoders() {
    let mut s = mixed_engine(2, false, &[(2, 1, 512, MS)]);
    throughput_only(&mut s);
    let d = add(&mut s, 1, 64, 1024);
    ready(&mut s, &[d]);
    let p = add(&mut s, 2, 8192, 128);
    let plan = s.plan().unwrap();
    assert_eq!(plan.kind, GraphKind::Init);
    assert_eq!(plan.lanes[0].request, p);
    let expected = (512.0 + 64.0) / 1.1;
    assert!(
        (plan.selection_value_rate() * plan.selection_log_scale().exp() - expected).abs() < 1e-9
    );
    assert_eq!(plan.estimated_duration_ns(), MS / 10);
    step(&mut s, 0);
    let plan = s.plan().unwrap();
    assert_eq!(plan.kind, GraphKind::Mixed);
    assert_eq!(plan.lanes.len(), 2);
}

#[test]
fn mixed_verification_width_reserves_state_even_when_separate_graphs_are_narrower() {
    let mut s = interactive_engine(2, true);
    s.speculative_graphs(&[(1, 1), (2, 1)]).unwrap();
    assert_eq!(s.execution_width(), 1);
    s.mixed_graphs(&[(2, 8, 512)]).unwrap();
    assert_eq!(s.execution_width(), 8);
    assert_eq!(s.execution_rows(), 8);
    assert!(s.fixed_speculative_width(Some(1)).is_err());
    assert!(s.enable_speculation(&[]).is_err());
    let first = add(&mut s, 1, 65, 128);
    let second = add(&mut s, 2, 65, 128);
    assert_eq!(
        s.request(second).unwrap().state_slot.0 - s.request(first).unwrap().state_slot.0,
        8
    );
}

#[test]
fn stale_mixed_catalog_preserves_retained_useful_value_selection() {
    let mut s = mixed_engine(
        4,
        true,
        &[(4, 1, 64, MS), (4, 3, 512, MS), (4, 8, 4096, MS)],
    );
    let decoders: Vec<_> = (1..=3).map(|id| add(&mut s, id, 64, 1024)).collect();
    ready(&mut s, &decoders);
    let p = add(&mut s, 4, 8192, 128);
    initialize(&mut s, p);
    s.advance_to(s.now_ns + s.policy.estimate_ttl_ns).unwrap();
    let plan = s.plan().unwrap();
    assert_eq!(plan.kind, GraphKind::Mixed);
    assert_eq!(
        (
            plan.batch_capacity,
            plan.prefill_capacity(),
            plan.token_capacity
        ),
        (4, 4096, 8)
    );
    assert_eq!(plan.estimated_duration_ns(), MS);
    assert_eq!(plan.origin, PlanOrigin::Ordinary);
    assert_eq!(plan.lanes.len(), 4);
    assert_eq!(plan.lanes.last().unwrap().tokens, 4096);
}

#[test]
fn mixed_cancellation_keeps_other_rows_live_and_retires_pages_only_after_completion() {
    let mut s = mixed_engine(2, true, &[(2, 8, 512, MS)]);
    let d = add(&mut s, 1, 64, 1024);
    ready(&mut s, &[d]);
    let p = add(&mut s, 2, 8192, 128);
    initialize(&mut s, p);
    let plan = s.plan().unwrap();
    assert_eq!(plan.kind, GraphKind::Mixed);
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    let records = completion_records(&s, &plan, fence, 3);
    let free = s.free_pages.len();
    let output = s.request(d).unwrap().generated_tokens;
    s.cancel(p).unwrap();
    assert_eq!(s.free_pages.len(), free);
    assert!(
        s.admit(RequestSpec {
            prompt: vec![3; 64],
            message_boundaries: vec![],
            conditioning: vec![],
            max_new_tokens: 128,
            grammar_constrained: false,
            host_stops: false,
            prompt_logprobs: false,
            growable_output: false,
        })
        .is_err()
    );
    let generated = s.complete(fence, &records, gpu_time(MS)).unwrap();
    assert_eq!(generated.len(), 4);
    assert!(generated.iter().all(|t| t.request == d));
    assert_eq!(s.request(d).unwrap().generated_tokens, output + 4);
    assert!(s.free_pages.len() > free);
    let replacement = add(&mut s, 3, 64, 128);
    assert_eq!(replacement.slot, p.slot);
    assert_ne!(replacement.generation, p.generation);
    assert_eq!(
        s.request(replacement)
            .unwrap()
            .accounting
            .committed_prefill_tokens,
        0
    );
}

#[test]
fn refresh_skips_infeasible_ages_and_resolves_the_oldest_feasible_utility_group() {
    let mut s = engine(1, true);
    let h = add(&mut s, 1, 64, 4096);
    ready(&mut s, &[h]);
    let key = |width| cost::Key { drafts: width - 1, kind: GraphKind::Speculative, batch: 1,
        tokens: width, prefill: 0, packed: width };
    let infeasible = cost::Key { drafts: 0, kind: GraphKind::Prefill, batch: 1,
        tokens: 64, prefill: 0, packed: 0 };
    let base = s.now_ns;
    s.advance_to(base + MS).unwrap();
    for width in [2, 4] { set_cost(&mut s, GraphKind::Speculative, 1, width, 100 * MS); }
    s.advance_to(base + 2 * MS).unwrap();
    set_cost(&mut s, GraphKind::Speculative, 1, 1, MS);
    s.advance_to(base + s.policy.estimate_ttl_ns + 2 * MS).unwrap();
    let others: Vec<_> = s.costs.stale_keys()
        .filter(|candidate| ![infeasible, key(1), key(2), key(4)].contains(candidate))
        .map(|candidate| (candidate, s.costs.duration_key(candidate).unwrap())).collect();
    for (candidate, nanos) in others {
        set_cost(&mut s, candidate.kind, candidate.batch, candidate.tokens, nanos);
    }
    set_cost(&mut s, GraphKind::Speculative, 1, 8, MS / 10);
    assert_eq!(s.plan().unwrap().token_capacity, 8);
    assert!(s.costs.refresh_order(infeasible) < s.costs.refresh_order(key(2)));
    assert!(s.costs.refresh_order(key(2)) < s.costs.refresh_order(key(1)));
    s.refresh.credit_ns = refresh::CREDIT_CAP_NS;
    assert_eq!(s.plan().unwrap().origin, PlanOrigin::RefreshStart(key(4)));
    s.costs.record_attempt(key(4));
    assert_eq!(s.plan().unwrap().origin, PlanOrigin::RefreshStart(key(2)));
}

#[test]
fn refresh_commit_rechecks_credit_and_freezes_physical_cost_context() {
    let mut s = engine(2, false);
    let a = add(&mut s, 1, 64, 4096);
    let b = add(&mut s, 2, 64, 4096);
    ready(&mut s, &[a, b]);
    s.advance_to(s.now_ns + s.policy.estimate_ttl_ns).unwrap();
    let mut first = s.phase_plan(GraphKind::Decode, Some(a));
    let key = cost::Key::from_step(&first);
    first.origin = PlanOrigin::RefreshStart(key);
    let mut saved = s.phase_plan(GraphKind::Decode, Some(b));
    saved.origin = PlanOrigin::RefreshStart(key);
    s.refresh.credit_ns = refresh::CREDIT_CAP_NS;
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&first, fence).unwrap();
    let context = s.submitted_timing(fence).unwrap();
    assert_eq!(context.retained_ns, MS);
    assert_eq!(context.freshness, CostFreshness::Stale);
    assert_eq!(context.origin, TimingOrigin::Refresh);
    assert_eq!(context.credit_ns, refresh::CREDIT_CAP_NS);
    assert!(s.submitted_step(fence).unwrap().requires_feedback);
    let mut timing = gpu_time(8 * MS);
    timing.gpu_ns[15] = Some(9 * MS);
    timing.wall_ns = 12 * MS;
    s.advance_to(s.now_ns + timing.wall_ns).unwrap();
    complete_timed(&mut s, &first, fence, 0, false, timing);
    assert_eq!(s.refresh_credit_ns(), -7_000_000.0);
    assert!(!s.refresh.active());
    let order = s.costs.refresh_order(key);
    let pending = s.request(b).unwrap().pending;
    assert!(s.commit(&saved, Fence(fence.0 + 1)).is_err());
    assert_eq!(s.costs.refresh_order(key), order);
    assert_eq!(s.request(b).unwrap().pending, pending);
    assert_eq!(s.refresh_credit_ns(), -7_000_000.0);
    assert_eq!(s.submitted, fence);
}

#[test]
fn invalid_refresh_timing_pays_isolated_wall_time_once_without_refreshing_cost() {
    let mut s = engine(1, false);
    let h = add(&mut s, 1, 1, 4096);
    ready(&mut s, &[h]);
    s.advance_to(s.now_ns + s.policy.estimate_ttl_ns).unwrap();
    let mut plan = s.phase_plan(GraphKind::Decode, Some(h));
    let key = cost::Key::from_step(&plan);
    plan.origin = PlanOrigin::RefreshStart(key);
    s.refresh.credit_ns = refresh::CREDIT_CAP_NS;
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    let records = completion_records(&s, &plan, fence, 0);
    let before = s.costs.observation(key);
    let timing = ExecutionTiming { wall_ns: 100 * MS, queued: true, ..Default::default() };
    assert!(s.complete(fence, &records, timing).is_err());
    assert_eq!(s.refresh_credit_ns(), refresh::CREDIT_CAP_NS);
    assert!(s.submitted_step(fence).is_ok());
    s.advance_to(s.now_ns + timing.wall_ns).unwrap();
    s.complete(fence, &records, ExecutionTiming { queued: false, ..timing }).unwrap();
    assert_eq!(s.refresh_credit_ns(), -95_000_000.0);
    assert_eq!(s.costs.observation(key), before);
    assert!(s.complete(fence, &records, timing).is_err());
    assert_eq!(s.refresh_credit_ns(), -95_000_000.0);
    s.idle().unwrap();
    assert_eq!(s.refresh_credit_ns(), -95_000_000.0);
}

#[test]
fn ordinary_income_excludes_queued_wall_wait_and_idle_discards_only_positive_credit() {
    let mut s = engine(2, false);
    let a = add(&mut s, 1, 64, 4096);
    let b = add(&mut s, 2, 64, 4096);
    ready(&mut s, &[a, b]);
    s.refresh.credit_ns = 0.0;
    let first = s.phase_plan(GraphKind::Decode, Some(a));
    let second = s.phase_plan(GraphKind::Decode, Some(b));
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&first, fence).unwrap();
    s.commit(&second, Fence(fence.0 + 1)).unwrap();
    assert!(s.idle().is_err());
    let frozen = s.submitted_timing(Fence(fence.0 + 1)).unwrap();
    let mut timing = gpu_time(10 * MS);
    timing.wall_ns = 50 * MS;
    complete_timed(&mut s, &first, fence, 0, false, timing);
    assert_eq!(s.submitted_timing(Fence(fence.0 + 1)).unwrap().retained_ns, frozen.retained_ns);
    timing.gpu_ns[15] = Some(20 * MS);
    timing.wall_ns = 200 * MS;
    timing.queued = true;
    complete_timed(&mut s, &second, Fence(fence.0 + 1), 0, false, timing);
    assert!((s.refresh_credit_ns() - 70_000_000.0 * 0.002 / 0.998).abs() < 1e-8);
    s.idle().unwrap();
    assert_eq!(s.refresh_credit_ns(), 0.0);
}

#[test]
fn saved_fresh_plans_gain_a_physical_staleness_feedback_barrier_at_commit() {
    let mut s = engine(2, false);
    let a = add(&mut s, 1, 64, 4096);
    let b = add(&mut s, 2, 64, 4096);
    ready(&mut s, &[a, b]);
    let first = s.phase_plan(GraphKind::Decode, Some(a));
    let second = s.phase_plan(GraphKind::Decode, Some(b));
    assert!(!first.requires_feedback);
    s.advance_to(s.now_ns + s.policy.estimate_ttl_ns).unwrap();
    assert!(s.plan_requires_feedback(&first));
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&first, fence).unwrap();
    assert!(s.submitted_step(fence).unwrap().requires_feedback);
    assert_eq!(s.submitted_timing(fence).unwrap().freshness, CostFreshness::Fresh);
    let order = s.costs.refresh_order(cost::Key::from_step(&second));
    let pending = s.request(b).unwrap().pending;
    assert!(s.commit(&second, Fence(fence.0 + 1)).is_err());
    assert_eq!(s.request(b).unwrap().pending, pending);
    assert_eq!(s.costs.refresh_order(cost::Key::from_step(&second)), order);
}

#[test]
fn initial_unknown_graph_completions_neither_earn_nor_spend_refresh_credit() {
    let mut s = cold_engine(1, false);
    add(&mut s, 1, 64, 64);
    for _ in 0..3 {
        let plan = measured_step(&mut s, 10 * MS, 0);
        assert_eq!(plan.origin, PlanOrigin::Discovery);
        assert_eq!(s.refresh_credit_ns(), 0.0);
        assert!(!s.refresh.active());
    }
}

#[test]
fn refresh_funding_requires_earned_credit_and_accounts_for_repeated_attempts_and_idle() {
    let mut s = engine(2, false);
    let h = add(&mut s, 1, 64, 4096);
    ready(&mut s, &[h]);
    set_cost(&mut s, GraphKind::Decode, 1, 1, 10 * MS);
    set_cost(&mut s, GraphKind::Decode, 2, 1, 100 * MS);
    s.idle().unwrap();
    s.advance_to(s.now_ns + s.policy.estimate_ttl_ns).unwrap();
    let ordinary = |s: &mut RequestScheduler, duration| {
        let plan = measured_step(s, duration, 0);
        assert_eq!(plan.origin, PlanOrigin::Ordinary);
        assert_eq!((plan.kind, plan.batch_capacity), (GraphKind::Decode, 1));
    };
    ordinary(&mut s, 10 * MS);
    assert!(s.refresh_credit_ns() > 0.0);
    s.idle().unwrap();
    assert_eq!(s.refresh_credit_ns(), 0.0);

    let mut ordinary_ns = 10 * MS;
    for _ in 0..249 {
        ordinary(&mut s, 10 * MS);
        ordinary_ns += 10 * MS;
    }
    ordinary(&mut s, 4 * MS);
    ordinary_ns += 4 * MS;
    assert!(s.refresh_credit_ns() < refresh::CREDIT_CAP_NS);
    assert_eq!(s.plan().unwrap().origin, PlanOrigin::Ordinary);
    ordinary(&mut s, MS);
    ordinary_ns += MS;
    assert_eq!(s.refresh_credit_ns(), refresh::CREDIT_CAP_NS);

    let first = s.plan().unwrap();
    let key = cost::Key::from_step(&first);
    assert_eq!(first.origin, PlanOrigin::RefreshStart(key));
    assert_eq!((first.kind, first.batch_capacity), (GraphKind::Decode, 2));
    let before = s.costs.observation(key);
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&first, fence).unwrap();
    s.advance_to(s.now_ns + 8 * MS).unwrap();
    complete_timed(&mut s, &first, fence, 0, false,
        ExecutionTiming { wall_ns: 8 * MS, ..Default::default() });
    assert_eq!(s.costs.observation(key), before);
    assert_eq!(s.refresh_credit_ns(), -3_000_000.0);
    assert!(!s.refresh.active());
    let mut refresh_ns = 8 * MS;
    assert!(refresh_ns as f64 <= ordinary_ns as f64 * 0.002 / 0.998 + (8 * MS) as f64);
    s.idle().unwrap();
    s.advance_to(s.now_ns + s.policy.estimate_ttl_ns).unwrap();
    assert_eq!(s.refresh_credit_ns(), -3_000_000.0);

    for _ in 0..399 {
        ordinary(&mut s, 10 * MS);
        ordinary_ns += 10 * MS;
    }
    assert!(s.refresh_credit_ns() < refresh::CREDIT_CAP_NS);
    assert_eq!(s.plan().unwrap().origin, PlanOrigin::Ordinary);
    ordinary(&mut s, 2 * MS);
    ordinary_ns += 2 * MS;
    assert_eq!(s.refresh_credit_ns(), refresh::CREDIT_CAP_NS);
    let second = measured_step(&mut s, 20 * MS, 0);
    assert_eq!(second.origin, PlanOrigin::RefreshStart(key));
    refresh_ns += 20 * MS;
    assert_eq!(s.refresh_credit_ns(), -15_000_000.0);
    assert!(!s.refresh.active());
    assert!(refresh_ns as f64 <= ordinary_ns as f64 * 0.002 / 0.998 + (20 * MS) as f64);
    assert!((s.refresh_credit_ns()
        - ((ordinary_ns - 10 * MS) as f64 * 0.002 / 0.998 - refresh_ns as f64)).abs() < 1e-6);
    s.idle().unwrap();
    assert_eq!(s.refresh_credit_ns(), -15_000_000.0);
}

#[test]
fn acceptance_learning_escapes_width_one_and_pays_for_complete_rounds() {
    let mut s = engine(1, true);
    let handle = add(&mut s, 1, 64, 4096);
    ready(&mut s, &[handle]);
    for width in 1..=8 { set_cost(&mut s, GraphKind::Speculative, 1, width, (9 + width as u64) * MS); }
    let generated = s.request(handle).unwrap().generated_tokens;
    for _ in 0..256 {
        s.request_mut(handle).unwrap().acceptance.observe(speculation::ProposalSource::Mtp,
            generated, 7, 0, true, None);
    }
    s.refresh.acceptance_credit_ns = 0.0;
    assert_eq!(s.plan().unwrap().token_capacity, 1);
    let mut ordinary_ns = 0;
    let mut probe_ns = 0;
    let mut recovered = false;
    for _ in 0..1800 {
        let plan = s.plan().unwrap();
        let duration = plan.estimated_duration_ns();
        if plan.origin.is_acceptance() { probe_ns += duration; }
        else { ordinary_ns += duration; }
        let accepted = if s.request(handle).unwrap().generated_tokens < generated + 512 { 0 } else { 7 };
        if accepted == 7 && plan.origin == PlanOrigin::Ordinary && plan.token_capacity >= 6 {
            recovered = true;
        }
        step(&mut s, accepted);
        assert!(probe_ns as f64 <= ordinary_ns as f64 * 0.01 / 0.99 + 17.0 * MS as f64 + 1.0);
        if recovered { break; }
    }
    assert!(recovered, "an improved request must escape width one");
    assert!(probe_ns > 0);
}

#[test]
fn survival_feedback_is_transactional_and_replaces_binary_acceptance() {
    let mut s = engine(1, true);
    s.fixed_speculative_width(Some(4)).unwrap();
    let handle = add(&mut s, 1, 64, 128);
    ready(&mut s, &[handle]);
    let plan = s.plan().unwrap();
    assert_eq!(plan.token_capacity, 4);
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    let records = completion_records(&s, &plan, fence, 0);
    let before = s.request(handle).unwrap().generated_tokens;
    for row in [[0.75, -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.],
        [0.75, 0.8, 0.25, -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.], [f32::NAN; 15]] {
        assert!(s.complete_with_survival(fence, &records, gpu_time(MS), Some(&[row.to_vec()])).is_err());
        assert_eq!(s.request(handle).unwrap().generated_tokens, before);
        assert_eq!(s.request(handle).unwrap().acceptance.rounds(), 0);
        assert!(s.submissions.contains_key(&fence));
    }
    let row = [0.75, 0.5, 0.25, -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.];
    s.complete_with_survival(fence, &records, gpu_time(MS), Some(&[row.to_vec()])).unwrap();
    let request = s.request(handle).unwrap();
    assert_eq!(request.generated_tokens, before + 1);
    assert_eq!(request.acceptance.rounds(), 1);
    assert!(request.acceptance.prefix_probability(speculation::ProposalSource::Mtp, 0).unwrap() > 0.5);
    assert!(s.complete_with_survival(fence, &records, gpu_time(MS), Some(&[row.to_vec()])).is_err());
    assert_eq!(s.request(handle).unwrap().acceptance.rounds(), 1);
}

fn terminal_snapshot_progress_with_ready_decode(policy: PolicyConfig) {
    let mut s = interactive_engine(2, false);
    s.configure_policy(policy).unwrap();
    let continuing = add(&mut s, 1, 64, 4096);
    let terminal = add(&mut s, 2, 64, 1);
    let execute = |s: &mut RequestScheduler, kind, handle| {
        let plan = s.phase_plan(kind, Some(handle));
        let fence = Fence(s.submitted.0 + 1);
        s.commit(&plan, fence).unwrap();
        s.advance_to(s.now_ns + plan.estimated_duration_ns()).unwrap();
        complete_timed(s, &plan, fence, 0, true, gpu_time(plan.estimated_duration_ns()));
    };
    for handle in [continuing, terminal] {
        execute(&mut s, GraphKind::Init, handle);
    }
    execute(&mut s, GraphKind::Prefill, continuing);
    execute(&mut s, GraphKind::Snapshot, continuing);
    execute(&mut s, GraphKind::Prefill, terminal);
    assert_eq!(s.request(continuing).unwrap().phase, RequestPhase::Decode);
    assert_eq!(s.request(terminal).unwrap().phase, RequestPhase::Snapshot);
    assert_eq!(s.request(terminal).unwrap().generated_tokens, 1);
    assert_eq!(s.request(terminal).unwrap().pending, 0);
    assert_eq!(s.costs.duration(GraphKind::Snapshot, 1, 1), Some(MS / 10));
    assert_eq!(s.costs.duration(GraphKind::Decode, 1, 1), Some(15 * MS));

    // Both requests reached these phases through ordinary completion records.
    // Only the policy chooses subsequent work; the helper retires requests
    // after Finishing, which is also the runtime's terminal-publication gate.
    let start = s.now_ns;
    let mut decode_steps = 0;
    let mut snapshot_steps = 0;
    while s.request(terminal).is_ok() && s.now_ns - start < 1000 * MS {
        match step(&mut s, 0).kind {
            GraphKind::Decode => decode_steps += 1,
            GraphKind::Snapshot => snapshot_steps += 1,
            kind => panic!("unexpected operation while terminal snapshot competes with decode: {kind:?}"),
        }
    }
    eprintln!(
        "terminal snapshot: elapsed_ns={}, decode_steps={decode_steps}, snapshot_steps={snapshot_steps}, terminal_phase={:?}, continuing_generated={}",
        s.now_ns - start,
        s.request(terminal).ok().map(|r| r.phase),
        s.request(continuing).unwrap().generated_tokens,
    );
    assert!(s.request(continuing).unwrap().generated_tokens < 4096);
    assert!(
        s.request(terminal).is_err(),
        "terminal snapshot did not complete within one simulated second while another request continued decoding"
    );
}

#[test]
fn terminal_snapshot_progresses_with_ready_decode_under_balanced_policy() {
    terminal_snapshot_progress_with_ready_decode(PolicyConfig::default());
}

#[test]
fn terminal_snapshot_progresses_with_ready_decode_under_throughput_policy() {
    terminal_snapshot_progress_with_ready_decode(PolicyConfig::throughput());
}

fn execute_phase(s: &mut RequestScheduler, kind: GraphKind, handle: RequestHandle) {
    let plan = s.phase_plan(kind, Some(handle));
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    s.advance_to(s.now_ns + plan.estimated_duration_ns()).unwrap();
    complete_timed(s, &plan, fence, 0, true, gpu_time(plan.estimated_duration_ns()));
}

#[test]
fn terminal_snapshot_drains_queued_decode_before_another_submission() {
    let mut s = engine(2, false);
    s.configure_policy(PolicyConfig::throughput()).unwrap();
    let terminal = add(&mut s, 1, 64, 2);
    let continuing = add(&mut s, 2, 64, 4096);
    for handle in [terminal, continuing] {
        for kind in [GraphKind::Init, GraphKind::Prefill, GraphKind::Snapshot] {
            execute_phase(&mut s, kind, handle);
        }
    }
    let finish = s.phase_plan(GraphKind::Decode, Some(terminal));
    assert!(!s.plan_requires_feedback(&finish));
    let a = Fence(s.submitted.0 + 1);
    s.commit(&finish, a).unwrap();
    let other = s.phase_plan(GraphKind::Decode, Some(continuing));
    assert!(!s.plan_requires_feedback(&other));
    let b = Fence(a.0 + 1);
    s.commit(&other, b).unwrap();
    complete_timed(&mut s, &finish, a, 0, true, gpu_time(MS));
    assert_eq!(s.request(terminal).unwrap().phase, RequestPhase::Snapshot);
    assert_eq!(s.submissions.len(), 1);
    assert!(s.plan().is_none(), "terminal cleanup must stop refilling the pipeline");
    complete_timed(&mut s, &other, b, 0, true, gpu_time(MS));
    let publications = s.stats().snapshot_publications;
    let plan = s.plan().unwrap();
    assert_eq!(plan.kind, GraphKind::Snapshot);
    assert_eq!(plan.lanes[0].request, terminal);
    let c = Fence(b.0 + 1);
    s.commit(&plan, c).unwrap();
    assert_eq!(s.request(terminal).unwrap().phase, RequestPhase::Snapshot);
    assert_eq!(s.stats().snapshot_publications, publications);
    complete_timed(&mut s, &plan, c, 0, true, gpu_time(MS / 10));
    assert!(s.request(terminal).is_err());
    assert_eq!(s.stats().snapshot_publications, publications + 1);
    assert_eq!(s.plan().unwrap().kind, GraphKind::Decode);
}

#[test]
fn terminal_snapshot_batch_includes_other_ready_checkpoints() {
    let mut s = engine(4, false);
    s.configure_policy(PolicyConfig::throughput()).unwrap();
    let handles = [add(&mut s, 1, 64, 1), add(&mut s, 2, 64, 1),
        add(&mut s, 3, 64, 1), add(&mut s, 4, 64, 4096)];
    for handle in handles {
        execute_phase(&mut s, GraphKind::Init, handle);
        execute_phase(&mut s, GraphKind::Prefill, handle);
    }
    let plan = s.plan().unwrap();
    assert_eq!(plan.kind, GraphKind::Snapshot);
    assert_eq!(plan.batch_capacity, 4);
    assert_eq!(plan.lanes.len(), 4);
    step(&mut s, 0);
    assert!(handles[..3].iter().all(|h| s.request(*h).is_err()));
    assert_eq!(s.request(handles[3]).unwrap().phase, RequestPhase::Decode);
}

#[test]
fn terminal_snapshot_publishes_exact_partial_page_for_shared_cache_geometries() {
    for (page_tokens, state_rows) in [(64, None), (128, Some(1)), (256, Some(3))] {
        let mut s = engine_with_geometry(4, false, page_tokens, state_rows);
        s.configure_policy(PolicyConfig::throughput()).unwrap();
        let terminal = add(&mut s, 1, page_tokens + 1, 1);
        let continuing = add(&mut s, 2, 64, 4096);
        for handle in [terminal, continuing] {
            execute_phase(&mut s, GraphKind::Init, handle);
            while s.request(handle).unwrap().phase == RequestPhase::Prefill {
                execute_phase(&mut s, GraphKind::Prefill, handle);
            }
        }
        assert_eq!(s.request(terminal).unwrap().phase, RequestPhase::Snapshot);
        let plan = s.plan().unwrap();
        assert_eq!(plan.kind, GraphKind::Snapshot);
        let publications = s.stats().snapshot_publications;
        let fence = Fence(s.submitted.0 + 1);
        s.commit(&plan, fence).unwrap();
        let spec = RequestSpec { prompt: vec![1; page_tokens + 1], message_boundaries: vec![],
            conditioning: vec![], max_new_tokens: 1, grammar_constrained: false,
            host_stops: false, prompt_logprobs: false, growable_output: false };
        assert_eq!(s.plan_admission(&spec).unwrap().resident_tokens(), 0);
        assert_eq!(s.stats().snapshot_publications, publications);
        complete_timed(&mut s, &plan, fence, 0, true, gpu_time(MS / 10));
        assert!(s.request(terminal).is_err());
        assert_eq!(s.plan_admission(&spec).unwrap().resident_tokens(), (page_tokens + 1) as u64);
    }
}

#[test]
fn terminal_snapshot_uses_loaded_shape_with_known_or_unknown_timing() {
    for nanos in [0, MS / 10] {
        let mut s = engine(2, false);
        let mut graphs = super::tests::duration_graphs(2);
        graphs.retain(|g| g.kind != GraphKind::Snapshot || g.batch_capacity == 2);
        for graph in &mut graphs {
            if graph.kind == GraphKind::Snapshot {
                graph.estimated_duration_ns = nanos;
            }
        }
        s.measured_graph_durations(&graphs).unwrap();
        s.configure_policy(PolicyConfig::throughput()).unwrap();
        let terminal = add(&mut s, 1, 64, 1);
        execute_phase(&mut s, GraphKind::Init, terminal);
        execute_phase(&mut s, GraphKind::Prefill, terminal);
        assert_eq!(s.costs.duration(GraphKind::Snapshot, 1, 1), None);
        assert_eq!(s.costs.duration(GraphKind::Snapshot, 2, 1), Some(nanos));
        let plan = s.plan().unwrap();
        assert_eq!(plan.kind, GraphKind::Snapshot);
        assert_eq!(plan.batch_capacity, 2);
        assert_eq!(plan.lanes.len(), 1);
        assert_eq!(plan.origin, if nanos == 0 { PlanOrigin::Discovery } else { PlanOrigin::Ordinary });
        step(&mut s, 0);
        assert!(s.request(terminal).is_err());
    }
}


fn feedback_window_engine(rows: Option<usize>, policy: PolicyConfig) -> RequestScheduler {
    let mut s = RequestScheduler::new(
        [7; 32],
        SchedulerLimits {
            max_active: 4,
            request_slots: 4,
            max_sequence_tokens: 8192,
            prefill_chunk_size: 512,
            pages: 1024,
            snapshots: 256,
        },
        SchedulerCapabilities {
            completion_token_capacity: 16,
            vocabulary: 1024,
            page_tokens: 64,
            vision: crate::model::VisionCapabilities::disabled(),
            draft_pages_per_request: 0,
            completion_inputs: vec![],
            speculative_allocations: vec![],
            state_rows_per_request: rows,
        },
    )
    .unwrap();
    let mut graphs = super::tests::duration_graphs(4);
    graphs.retain(|g| g.kind != GraphKind::Speculative);
    s.measured_graph_durations(&graphs).unwrap();
    s.configure_policy(policy).unwrap();
    s
}

#[test]
fn feedback_prefill_tails_progress_before_existing_generations_finish() {
    for rows in [None, Some(1), Some(2)] {
        for policy in [PolicyConfig::throughput(), PolicyConfig::default()] {
            let mut s = feedback_window_engine(rows, policy);
            for capacity in [64, 128, 256, 512] {
                set_cost(&mut s, GraphKind::Prefill, 1, capacity, 31 * MS);
            }
            for batch in [1, 2, 4] {
                set_cost(
                    &mut s,
                    GraphKind::Decode,
                    batch,
                    1,
                    (22 + batch as u64 / 2) * MS,
                );
            }
            let handles = (0..4)
                .map(|id| {
                    s.admit(RequestSpec {
                        prompt: vec![id; 87],
                        message_boundaries: vec![42, 82, 87],
                        conditioning: vec![],
                        max_new_tokens: 256,
                        grammar_constrained: false,
                        host_stops: false,
                        prompt_logprobs: false,
                        growable_output: true,
                    })
                    .unwrap()
                })
                .collect::<Vec<_>>();
            let mut running = std::collections::VecDeque::new();
            let mut first_output = [false; 4];
            let started = s.now_ns;
            let mut launches = 0;
            let mut drains = 0;
            let mut draining = false;
            while s.slots.iter().any(|slot| slot.request.is_some()) && launches < 4096 {
                while running.len() < 2 {
                    let Some(plan) = s.plan() else {
                        if !first_output.iter().all(|v| *v)
                            && !draining
                            && !s.submissions.is_empty()
                            && !s.submissions.values().any(|v| v.requires_feedback)
                        {
                            drains += 1;
                            draining = true;
                        }
                        break;
                    };
                    draining = false;
                    let fence = Fence(s.submitted.0 + 1);
                    s.commit(&plan, fence).unwrap();
                    running.push_back((plan, fence));
                    launches += 1;
                }
                let (plan, fence) = running
                    .pop_front()
                    .expect("live requests must make progress");
                let finish = s.submissions[&fence].expected_finish_ns.max(s.now_ns);
                s.advance_to(finish).unwrap();
                complete_timed(
                    &mut s,
                    &plan,
                    fence,
                    0,
                    true,
                    gpu_time(plan.estimated_duration_ns()),
                );
                if plan.kind == GraphKind::Prefill {
                    for lane in &plan.lanes {
                        if lane.start_position + lane.tokens as u64 == 87 {
                            let index = handles.iter().position(|h| *h == lane.request).unwrap();
                            first_output[index] = true;
                            // Four three-chunk prompts cost 372ms of prefill.
                            // Allow lifecycle work and drain latency, but never
                            // a generation-sized wait for an eligible tail.
                            assert!(s.now_ns - started <= 1_000 * MS);
                        }
                    }
                }
                if handles
                    .iter()
                    .any(|h| s.request(*h).map_or(true, |r| r.generated_tokens >= 256))
                {
                    assert!(
                        first_output.iter().all(|v| *v),
                        "a feedback-only prompt tail was hidden behind an entire generation"
                    );
                }
            }
            assert!(first_output.iter().all(|v| *v));
            assert!(
                drains <= 4,
                "feedback drain/refill oscillated without serving the prompt tails"
            );
            assert!(!s.slots.iter().any(|slot| slot.request.is_some()));
        }
    }
}

#[test]
fn higher_value_feedback_work_drains_but_never_crosses_an_inflight_barrier() {
    let mut s = feedback_window_engine(None, PolicyConfig::throughput());
    set_cost(&mut s, GraphKind::Decode, 1, 1, 10 * MS);
    let existing = add(&mut s, 1, 64, 128);
    ready(&mut s, &[existing]);
    let plan = s.plan().unwrap();
    assert_eq!(plan.kind, GraphKind::Decode);
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    for capacity in [64, 128, 256, 512] {
        set_cost(&mut s, GraphKind::Prefill, 1, capacity, MS);
    }
    let waiting = add(&mut s, 2, 54, 128);
    assert!(
        s.plan().is_none(),
        "refilling decode hides higher-value feedback work"
    );
    let finish = s.submissions[&fence].expected_finish_ns;
    s.advance_to(finish).unwrap();
    complete_timed(
        &mut s,
        &plan,
        fence,
        0,
        true,
        gpu_time(plan.estimated_duration_ns()),
    );
    let feedback = s.plan().unwrap();
    // The arriving request first needs its feedback-only initialization.
    assert_eq!(feedback.kind, GraphKind::Init);
    assert!(feedback.requires_feedback && feedback.lanes.iter().any(|l| l.request == waiting));
    let next = Fence(s.submitted.0 + 1);
    s.commit(&feedback, next).unwrap();
    assert!(s.plan().is_none(), "feedback still owns the entire window");
}

#[test]
fn stale_feedback_winner_drains_once_then_rejoins_the_pipeline() {
    let mut policy = PolicyConfig::throughput();
    policy.estimate_ttl_ns = 3 * MS;
    let mut s = feedback_window_engine(None, policy);
    let handles = (0..2)
        .map(|id| add(&mut s, id + 1, 64, 128))
        .collect::<Vec<_>>();
    ready(&mut s, &handles);
    for batch in [1, 2] {
        set_cost(&mut s, GraphKind::Decode, batch, 1, 10 * MS);
    }
    set_cost(&mut s, GraphKind::Decode, 4, 1, 100 * MS);
    let first = s.phase_plan(GraphKind::Decode, Some(handles[0]));
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&first, fence).unwrap();
    s.advance_to(s.now_ns + 4 * MS).unwrap();
    set_cost(&mut s, GraphKind::Decode, 1, 1, 10 * MS);
    assert!(
        s.plan().is_none(),
        "a stale higher-value cohort must drain rather than be hidden by a fresh alternative"
    );
    s.advance_to(s.submissions[&fence].expected_finish_ns)
        .unwrap();
    complete_timed(
        &mut s,
        &first,
        fence,
        0,
        true,
        gpu_time(first.estimated_duration_ns()),
    );
    let probe = s.plan().unwrap();
    assert_eq!(probe.batch_capacity, 2);
    assert!(s.plan_requires_feedback(&probe));
    let next = Fence(s.submitted.0 + 1);
    s.commit(&probe, next).unwrap();
    s.advance_to(s.submissions[&next].expected_finish_ns)
        .unwrap();
    complete_timed(
        &mut s,
        &probe,
        next,
        0,
        true,
        gpu_time(probe.estimated_duration_ns()),
    );
    for _ in 0..2 {
        let plan = s.plan().expect("refreshed work must not repeatedly drain");
        assert!(!s.plan_requires_feedback(&plan));
        let fence = Fence(s.submitted.0 + 1);
        s.commit(&plan, fence).unwrap();
    }
    assert_eq!(s.submissions.len(), 2);
}

fn terminal_competition(policy: PolicyConfig) -> (RequestScheduler, RequestHandle, RequestHandle) {
    let mut s = engine(2, false);
    s.configure_policy(policy).unwrap();
    let continuing = add(&mut s, 1, 64, 4096);
    let terminal = add(&mut s, 2, 64, 1);
    for kind in [GraphKind::Init, GraphKind::Prefill, GraphKind::Snapshot] {
        execute_phase(&mut s, kind, continuing);
    }
    for kind in [GraphKind::Init, GraphKind::Prefill] {
        execute_phase(&mut s, kind, terminal);
    }
    for batch in [1, 2] {
        set_cost(&mut s, GraphKind::Snapshot, batch, 1, 10 * MS);
    }
    (s, terminal, continuing)
}

#[test]
fn terminal_value_competes_with_decode_without_an_unconditional_priority() {
    for (value, expected) in [(1.0, GraphKind::Decode), (1024.0, GraphKind::Snapshot)] {
        let (mut s, terminal, _) = terminal_competition(PolicyConfig {
            token_fairness: 0.0, terminal_value: value, terminal_age_price: 0.0,
            ..PolicyConfig::throughput()
        });
        assert_eq!(s.plan().unwrap().kind, expected);
        let since = s.request(terminal).unwrap().accounting.terminal_since_ns.unwrap();
        s.advance_to(since + 1000 * MS).unwrap();
        assert_eq!(s.plan().unwrap().kind, expected);
    }
}

#[test]
fn terminal_age_changes_the_winner_independently_of_streaming_delivery() {
    let (mut s, terminal, _) = terminal_competition(PolicyConfig {
        token_fairness: 0.0, terminal_value: 1.0, terminal_age_price: 1.0,
        ..PolicyConfig::throughput()
    });
    assert_eq!(s.plan().unwrap().kind, GraphKind::Decode);
    let since = s.request(terminal).unwrap().accounting.terminal_since_ns.unwrap();
    s.advance_to(since + 30 * MS).unwrap();
    s.published_output(terminal, s.completed).unwrap();
    let (values, _, scale) = s.lane_values(0);
    let value = values[terminal.slot as usize];
    assert_eq!(value.age_ms, 30.0);
    assert_eq!(value.delivery, 0.0);
    assert!((value.terminal * scale.exp() - 1.0).abs() < 1e-12);
    assert_eq!(s.plan().unwrap().kind, GraphKind::Snapshot);
}

#[test]
fn terminal_age_starts_at_eos_and_survives_an_already_queued_decode() {
    let mut s = engine(1, false);
    let handle = add(&mut s, 1, 64, 128);
    for kind in [GraphKind::Init, GraphKind::Prefill, GraphKind::Snapshot] {
        execute_phase(&mut s, kind, handle);
    }
    let a = s.phase_plan(GraphKind::Decode, Some(handle));
    let first = Fence(s.submitted.0 + 1);
    s.commit(&a, first).unwrap();
    let b = s.phase_plan(GraphKind::Decode, Some(handle));
    let second = Fence(first.0 + 1);
    s.commit(&b, second).unwrap();
    s.advance_to(s.now_ns + MS).unwrap();
    let mut records = completion_records(&s, &a, first, 0);
    records[handle.slot as usize].status = 2;
    s.complete(first, &records, gpu_time(MS)).unwrap();
    let since = s.now_ns;
    assert_eq!(s.request(handle).unwrap().accounting.terminal_since_ns, Some(since));
    assert!(s.plan().is_none());
    s.advance_to(s.now_ns + 5 * MS).unwrap();
    records[handle.slot as usize].submission_sequence = second.0 as i64;
    s.complete(second, &records, gpu_time(MS)).unwrap();
    assert_eq!(s.request(handle).unwrap().accounting.terminal_since_ns, Some(since));
    assert_eq!(s.lane_values(0).0[handle.slot as usize].age_ms, 5.0);
    assert_eq!(s.plan().unwrap().kind, GraphKind::Snapshot);
}

#[test]
fn terminal_cohort_matches_exhaustive_common_utility_scoring() {
    let mut s = engine(4, false);
    s.configure_policy(PolicyConfig::throughput()).unwrap();
    let handles: Vec<_> = (1..=4).map(|id| add(&mut s, id, 64, 1)).collect();
    for &handle in handles.iter().rev() {
        execute_phase(&mut s, GraphKind::Init, handle);
        execute_phase(&mut s, GraphKind::Prefill, handle);
    }
    for (batch, duration) in [(1, 4 * MS), (2, 5 * MS), (4, 30 * MS)] {
        set_cost(&mut s, GraphKind::Snapshot, batch, 1, duration);
    }
    let (values, age, _) = s.lane_values(0);
    let mut best = f64::NEG_INFINITY;
    for mask in 1u64..16 {
        let batch = (mask.count_ones() as usize).next_power_of_two();
        let ns = s.costs.duration(GraphKind::Snapshot, batch, 1).unwrap();
        let ms = ns as f64 / 1e6;
        let work = policy::bits(mask).map(|slot| values[slot].finish(ms, ms)).sum();
        best = best.max(ValueRate::new(ns, work, age).score);
    }
    let plan = s.plan().unwrap();
    assert!((plan.selection_value_rate() - best).abs() < 1e-12);
    assert_eq!(plan.batch_capacity, 2);
    assert!(!plan.lanes.iter().any(|lane| lane.request == handles[0]));
}

#[test]
fn speculative_lookahead_reuses_request_and_reconciles_variable_acceptance() {
    let mut scheduler = engine(1, true);
    let handle = add(&mut scheduler, 5, 64, 32);
    ready(&mut scheduler, &[handle]);
    for width in [1, 2, 4, 8] { set_cost(&mut scheduler, GraphKind::Speculative, 1, width, MS); }
    let first = scheduler.plan().unwrap();
    assert_eq!(first.kind, GraphKind::Speculative);
    assert!(first.token_capacity > 1);
    let first_fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&first, first_fence).unwrap();
    let second = scheduler.plan().unwrap();
    assert_eq!(second.lanes[0].request, handle);
    assert_eq!(second.lanes[0].start_position, first.lanes[0].start_position + first.lanes[0].tokens as u64);
    let second_fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&second, second_fence).unwrap();
    assert!(scheduler.plan().is_none());
    let records = completion_records(&scheduler, &first, first_fence, 0);
    scheduler.complete(first_fence, &records, gpu_time(MS)).unwrap();
    assert_eq!(scheduler.request(handle).unwrap().scheduled_consumed,
        scheduler.request(handle).unwrap().consumed_tokens + second.lanes[0].tokens as u64);
    let records = completion_records(&scheduler, &second, second_fence, 0);
    scheduler.complete(second_fence, &records, gpu_time(MS)).unwrap();
    let request = scheduler.request(handle).unwrap();
    assert_eq!(request.pending, 0);
    assert_eq!(request.scheduled_consumed, request.consumed_tokens);
}

#[test]
fn speculative_lookahead_bounds_output_by_all_pending_lane_ceilings() {
    let mut scheduler = engine(1, true);
    let handle = add(&mut scheduler, 5, 64, 10);
    ready(&mut scheduler, &[handle]);
    for width in [1, 2, 4, 8] { set_cost(&mut scheduler, GraphKind::Speculative, 1, width, MS); }
    let before = scheduler.request(handle).unwrap().generated_tokens as usize;
    let first = scheduler.plan().unwrap();
    let fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&first, fence).unwrap();
    let second = scheduler.plan().unwrap();
    assert!(before + first.lanes[0].tokens + second.lanes[0].tokens <= 10);
}

#[test]
fn projected_speculative_service_does_not_commit_output_or_publication() {
    let mut scheduler = engine(1, true);
    let handle = add(&mut scheduler, 5, 64, 32);
    ready(&mut scheduler, &[handle]);
    for width in [1, 2, 4, 8] { set_cost(&mut scheduler, GraphKind::Speculative, 1, width, MS); }
    let plan = scheduler.plan().unwrap();
    let before = scheduler.request(handle).unwrap().accounting;
    let fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&plan, fence).unwrap();
    let request = scheduler.request(handle).unwrap();
    let projected = scheduler.projected_service(request);
    assert!(projected.tokens.is_some());
    let mut unserved = request.accounting.service.tokens.unwrap();
    unserved.observe(plan.estimated_duration_ns() as f64 / 1e6, 0.0,
        scheduler.service_rates(request).0.unwrap(), scheduler.policy.adaptation_ms);
    assert!(projected.tokens.unwrap().log_price(1.0) < unserved.log_price(1.0));
    assert_eq!(request.accounting.committed_output_tokens, before.committed_output_tokens);
    assert_eq!(request.accounting.published_deliveries, before.published_deliveries);
    assert!(scheduler.submissions[&fence].expected_tokens[handle.slot as usize] > 1.0);
}

#[test]
fn projected_delivery_age_uses_the_expected_delivery_endpoint() {
    let mut scheduler = engine(2, false);
    let handles = [add(&mut scheduler, 1, 64, 32), add(&mut scheduler, 2, 64, 32)];
    ready(&mut scheduler, &handles);
    let first = scheduler.plan().unwrap();
    let fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&first, fence).unwrap();
    let finish = scheduler.submissions[&fence].expected_finish_ns;
    let (values, _, _) = scheduler.lane_values(finish - scheduler.now_ns + 3 * MS);
    assert_eq!(values[handles[0].slot as usize].age_ms, 3.0);
}

#[test]
fn host_prefix_allocator_honors_budget_and_gpu_allocator_preserves_lane_ceilings() {
    let mut scheduler = engine(2, true);
    let handles = [add(&mut scheduler, 1, 64, 32), add(&mut scheduler, 2, 64, 32)];
    ready(&mut scheduler, &handles);
    let mask = handles.iter().fold(0, |mask, handle| mask | 1 << handle.slot);
    let exact = scheduler.round_lengths(mask, 2, 7, 8, 5);
    assert_eq!(exact.iter().sum::<usize>(), 5);
    assert!(handles.iter().all(|handle| exact[handle.slot as usize] >= 1));
    scheduler.capabilities.speculative_allocations.push((2, 7, 8, 5, crate::execution::PrefixAllocation::Gpu));
    let ceilings = scheduler.round_lengths(mask, 2, 7, 8, 5);
    assert_eq!(ceilings[handles[0].slot as usize], 8);
    assert_eq!(ceilings[handles[1].slot as usize], 8);
}

#[test]
fn acceptance_probe_charges_one_round_preserves_timing_credit_and_carries_debt() {
    let mut state = refresh::RefreshState::default();
    state.complete(PlanOrigin::Ordinary, gpu_time(495 * MS));
    assert!(state.acceptance_ready());
    let timing_credit = state.credit_ns;
    let round = ScheduledStep { draft_tokens: 7, kind: GraphKind::Speculative,
        batch_capacity: 1, token_capacity: 8, packed_token_capacity: 8,
        mixed: None, requires_feedback: true,
        lanes: vec![ScheduledLane { request: RequestHandle { slot: 0, generation: 1 }, start_position: 65, tokens: 8 }] };
    state.validate(PlanOrigin::AcceptanceStart).unwrap();
    state.commit(PlanOrigin::AcceptanceStart, Fence(7), &round);
    assert!(!state.acceptance_ready());
    assert!(state.validate_completion(PlanOrigin::AcceptanceStart,
        ExecutionTiming { queued: true, ..gpu_time(10 * MS) }).is_err());
    state.complete(PlanOrigin::AcceptanceStart, gpu_time(10 * MS));
    assert_eq!(state.acceptance_credit_ns, -5.0 * MS as f64);
    assert_eq!(state.credit_ns, timing_credit);
    assert!(!state.active());
    state.idle();
    assert_eq!(state.acceptance_credit_ns, -5.0 * MS as f64);
    assert_eq!(state.credit_ns, 0.0);
}

fn joined_round_engine(allocation: crate::execution::PrefixAllocation) -> RequestScheduler {
    let mut scheduler = engine(2, true);
    let mut graphs = super::tests::duration_graphs(2);
    graphs.iter_mut().for_each(|graph| graph.estimated_duration_ns = 10 * MS);
    let graph = crate::execution::GraphDescriptor { tuned_bindings: vec![],
        confidence_width: if allocation == crate::execution::PrefixAllocation::Gpu { 7 } else { 0 },
        name: "speculative-b2-d7-w8-p5".into(),
        kind: GraphKind::Speculative, sampling_mode: SamplingMode::General,
        draft_tokens: 7, handle: 1, batch_capacity: 2, token_capacity: 8,
        packed_token_capacity: 5, prefill_capacity: 0, estimated_duration_ns: MS, input_bindings: vec![] };
    graphs.push(graph);
    scheduler.measured_graph_durations(&graphs).unwrap();
    scheduler.capabilities.speculative_allocations = vec![(2, 7, 8, 5, allocation)];
    scheduler
}

#[test]
fn device_prefix_completion_validates_aggregate_budget_atomically() {
    let mut scheduler = joined_round_engine(crate::execution::PrefixAllocation::Gpu);
    let handles = [add(&mut scheduler, 1, 64, 32), add(&mut scheduler, 2, 64, 32)];
    ready(&mut scheduler, &handles);
    let plan = scheduler.plan().unwrap();
    assert_eq!(plan.packed_token_capacity, 5);
    assert!(plan.lanes.iter().all(|lane| lane.tokens == 8));
    let fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&plan, fence).unwrap();
    let mut records = completion_records(&scheduler, &plan, fence, 0);
    for handle in handles { records[handle.slot as usize].vector.as_mut().unwrap().offered_drafts = 2; }
    let before = scheduler.request(handles[0]).unwrap().generated_tokens;
    assert!(scheduler.complete_packed_round(fence, &records, gpu_time(MS), &vec![vec![0.5; 15]; 2]).is_err());
    assert_eq!(scheduler.request(handles[0]).unwrap().generated_tokens, before);
    assert!(scheduler.submissions.contains_key(&fence));
    records[handles[1].slot as usize].vector.as_mut().unwrap().offered_drafts = 1;
    let output = scheduler.complete_packed_round(fence, &records, gpu_time(MS), &vec![vec![0.5; 15]; 2]).unwrap();
    assert_eq!(output.len(), 2);
}

#[test]
fn host_prefix_completion_uses_exact_lengths_without_confidence_sidecar() {
    let mut scheduler = joined_round_engine(crate::execution::PrefixAllocation::Host);
    let handles = [add(&mut scheduler, 1, 64, 32), add(&mut scheduler, 2, 64, 32)];
    ready(&mut scheduler, &handles);
    let plan = scheduler.plan().unwrap();
    assert_eq!(plan.packed_token_capacity, 5);
    assert_eq!(plan.lanes.iter().map(|lane| lane.tokens).sum::<usize>(), 5);
    let fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&plan, fence).unwrap();
    let mut records = completion_records(&scheduler, &plan, fence, 0);
    for lane in &plan.lanes { records[lane.request.slot as usize].vector.as_mut().unwrap().offered_drafts = lane.tokens as i64 - 1; }
    let output = scheduler.complete(fence, &records, gpu_time(MS)).unwrap();
    assert_eq!(output.len(), 2);
}

#[test]
fn queued_completion_progress_is_bounded_by_lane_extent_not_reservation_cursor() {
    let mut scheduler = engine(1, true);
    let handle = add(&mut scheduler, 5, 64, 32);
    ready(&mut scheduler, &[handle]);
    for width in [1, 2, 4, 8] { set_cost(&mut scheduler, GraphKind::Speculative, 1, width, MS); }
    let first = scheduler.plan().unwrap();
    let first_fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&first, first_fence).unwrap();
    let mut second = scheduler.plan().unwrap();
    second.step.lanes[0].tokens = 1;
    let second_fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&second, second_fence).unwrap();
    let records = completion_records(&scheduler, &first, first_fence, 0);
    scheduler.complete(first_fence, &records, gpu_time(MS)).unwrap();
    let mut records = completion_records(&scheduler, &second, second_fence, 0);
    let record = &mut records[handle.slot as usize];
    record.consumed_tokens += 1;
    record.generated_tokens += 1;
    let vector = record.vector.as_mut().unwrap();
    vector.emitted_count = 2;
    vector.accepted_drafts = 1;
    vector.tokens[1] = 99;
    let before = scheduler.request(handle).unwrap().generated_tokens;
    assert!(scheduler.complete(second_fence, &records, gpu_time(MS)).is_err());
    assert_eq!(scheduler.request(handle).unwrap().generated_tokens, before);
}

#[test]
fn pending_advance_ceilings_follow_queued_rounds_and_retirement() {
    let mut scheduler = engine(1, true);
    let handle = add(&mut scheduler, 5, 64, 32);
    ready(&mut scheduler, &[handle]);
    for width in [1, 2, 4, 8] { set_cost(&mut scheduler, GraphKind::Speculative, 1, width, MS); }
    assert_eq!(scheduler.pending_token_ceiling(handle), 0);
    let first = scheduler.plan().unwrap();
    let first_fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&first, first_fence).unwrap();
    let second = scheduler.plan().unwrap();
    let second_fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&second, second_fence).unwrap();
    let ceilings = [first.lanes[0].tokens, second.lanes[0].tokens];
    assert_eq!(scheduler.pending_token_ceiling(handle), ceilings[0] + ceilings[1]);
    assert_eq!(scheduler.pending_ceiling(scheduler.request(handle).unwrap()), ceilings[0] + ceilings[1]);
    let records = completion_records(&scheduler, &first, first_fence, 0);
    scheduler.complete(first_fence, &records, gpu_time(MS)).unwrap();
    assert_eq!(scheduler.pending_token_ceiling(handle), ceilings[1]);
    assert_eq!(scheduler.pending_ceiling(scheduler.request(handle).unwrap()), ceilings[1]);
    let records = completion_records(&scheduler, &second, second_fence, 0);
    scheduler.cancel(handle).unwrap();
    // A retired handle still reports the lanes its queued round retains.
    assert_eq!(scheduler.pending_token_ceiling(handle), ceilings[1]);
    scheduler.complete(second_fence, &records, gpu_time(MS)).unwrap();
    assert_eq!(scheduler.pending_token_ceiling(handle), 0);
}

#[test]
fn queued_generation_acknowledges_a_finished_predecessor_without_more_output() {
    let mut scheduler = engine(1, true);
    let handle = add(&mut scheduler, 5, 64, 32);
    ready(&mut scheduler, &[handle]);
    for width in [1, 2, 4, 8] { set_cost(&mut scheduler, GraphKind::Speculative, 1, width, MS); }
    let first = scheduler.plan().unwrap();
    let first_fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&first, first_fence).unwrap();
    let second = scheduler.plan().unwrap();
    let second_fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&second, second_fence).unwrap();
    let mut records = completion_records(&scheduler, &first, first_fence, 0);
    records[handle.slot as usize].status = 2;
    let output = scheduler.complete(first_fence, &records, gpu_time(MS)).unwrap();
    assert_eq!(output.len(), 1);
    assert!(scheduler.plan().is_none());
    let request = scheduler.request(handle).unwrap();
    let mut records = vec![CompletionRecord::default(); scheduler.limits.request_capacity()];
    records[handle.slot as usize] = CompletionRecord {
        generation: handle.generation as i64, submission_sequence: second_fence.0 as i64,
        consumed_tokens: request.consumed_tokens as i64, generated_tokens: request.generated_tokens as i64,
        token_id: -1, status: 2,
        vector: Some(TokenVector { verification_width: second.token_capacity as i64,
            draft_cache_consumed: -1, ..Default::default() }), ..Default::default()
    };
    let committed = request.accounting.committed_output_tokens;
    assert!(scheduler.complete(second_fence, &records, gpu_time(MS)).unwrap().is_empty());
    let request = scheduler.request(handle).unwrap();
    assert_eq!(request.pending, 0);
    assert_eq!(request.accounting.committed_output_tokens, committed);
    assert_eq!(scheduler.plan().unwrap().kind, GraphKind::Snapshot);
}

#[test]
fn funded_acceptance_probe_compares_offered_prefixes_across_equal_width_budgets() {
    let mut scheduler = engine(2, true);
    scheduler.configure_policy(PolicyConfig::throughput()).unwrap();
    scheduler.enable_speculation(&[1, 8]).unwrap();
    let mut graphs = super::tests::duration_graphs(2);
    graphs.iter_mut().for_each(|graph| graph.estimated_duration_ns = 10 * MS);
    for (packed, duration) in [(4, MS), (7, 13 * MS / 10)] {
        graphs.push(crate::execution::GraphDescriptor { tuned_bindings: vec![], confidence_width: 0,
            name: format!("speculative-b2-d7-w8-p{packed}"), kind: GraphKind::Speculative,
            sampling_mode: SamplingMode::General, draft_tokens: 7, handle: 1,
            batch_capacity: 2, token_capacity: 8, packed_token_capacity: packed,
            prefill_capacity: 0, estimated_duration_ns: duration, input_bindings: vec![] });
    }
    scheduler.measured_graph_durations(&graphs).unwrap();
    let handles = [add(&mut scheduler, 1, 64, 32), add(&mut scheduler, 2, 64, 32)];
    ready(&mut scheduler, &handles);
    scheduler.refresh.acceptance_credit_ns = 0.0;
    scheduler.refresh.credit_ns = 0.0;
    let ordinary = scheduler.plan().unwrap();
    assert_eq!((ordinary.token_capacity, ordinary.packed_token_capacity), (8, 4), "origin {:?}, duration {}", ordinary.origin, ordinary.estimated_duration_ns());
    let fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&ordinary, fence).unwrap();
    scheduler.refresh.acceptance_credit_ns = refresh::CREDIT_CAP_NS;
    assert!(scheduler.plan().is_none());
    let mut records = completion_records(&scheduler, &ordinary, fence, 0);
    for lane in &ordinary.lanes {
        records[lane.request.slot as usize].vector.as_mut().unwrap().offered_drafts = lane.tokens as i64 - 1;
    }
    scheduler.complete(fence, &records, gpu_time(MS)).unwrap();
    let probe = scheduler.plan().unwrap();
    assert_eq!(probe.origin, PlanOrigin::AcceptanceStart);
    assert_eq!((probe.token_capacity, probe.packed_token_capacity), (8, 7));
    assert_eq!(probe.lanes.iter().map(|lane| lane.tokens).sum::<usize>(), 7);
}

#[test]
fn funded_refresh_winner_drains_a_busy_generation_queue() {
    let mut scheduler = engine(1, true);
    let handle = add(&mut scheduler, 5, 64, 32);
    ready(&mut scheduler, &[handle]);
    for width in [2, 4, 8] { set_cost(&mut scheduler, GraphKind::Speculative, 1, width, 100 * MS); }
    scheduler.advance_to(scheduler.now_ns + scheduler.policy.estimate_ttl_ns).unwrap();
    set_cost(&mut scheduler, GraphKind::Speculative, 1, 1, MS);
    scheduler.refresh.credit_ns = 0.0;
    scheduler.refresh.acceptance_credit_ns = 0.0;
    let ordinary = scheduler.plan().unwrap();
    assert_eq!(ordinary.token_capacity, 1);
    let fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&ordinary, fence).unwrap();
    scheduler.refresh.credit_ns = refresh::CREDIT_CAP_NS;
    assert!(scheduler.plan().is_none());
    let records = completion_records(&scheduler, &ordinary, fence, 0);
    scheduler.complete(fence, &records, gpu_time(MS)).unwrap();
    let refresh = scheduler.plan().unwrap();
    assert!(matches!(refresh.origin, PlanOrigin::RefreshStart(_)));
    assert!(refresh.token_capacity > 1);
}

#[test]
fn finished_queued_round_does_not_observe_uncomputed_confidence() {
    let mut scheduler = joined_round_engine(crate::execution::PrefixAllocation::Gpu);
    let handle = add(&mut scheduler, 1, 64, 32);
    ready(&mut scheduler, &[handle]);
    let first = scheduler.plan().unwrap();
    assert_eq!(first.packed_token_capacity, 5);
    let first_fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&first, first_fence).unwrap();
    let second = scheduler.plan().unwrap();
    let second_fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&second, second_fence).unwrap();
    let mut records = completion_records(&scheduler, &first, first_fence, 0);
    records[handle.slot as usize].status = 2;
    records[handle.slot as usize].vector.as_mut().unwrap().offered_drafts = 4;
    scheduler.complete_packed_round(first_fence, &records, gpu_time(MS), &vec![vec![0.9; 15]; 2]).unwrap();
    let population = scheduler.forecasts.population.clone();
    let request = scheduler.request(handle).unwrap();
    let mut records = vec![CompletionRecord::default(); scheduler.limits.request_capacity()];
    records[handle.slot as usize] = CompletionRecord {
        generation: handle.generation as i64, submission_sequence: second_fence.0 as i64,
        consumed_tokens: request.consumed_tokens as i64, generated_tokens: request.generated_tokens as i64,
        token_id: -1, status: 2,
        vector: Some(TokenVector { verification_width: second.token_capacity as i64,
            draft_cache_consumed: -1, ..Default::default() }), ..Default::default()
    };
    scheduler.complete_packed_round(second_fence, &records, gpu_time(MS), &vec![vec![0.0; 15]; 2]).unwrap();
    assert_eq!(scheduler.forecasts.population, population);
    assert_eq!(scheduler.request(handle).unwrap().pending, 0);
}

#[test]
fn host_prefix_allocator_prices_the_configured_acceptance_transform() {
    let mut scheduler = engine(2, true);
    scheduler.configure_policy(PolicyConfig { adaptive_verification_alpha: 0.5, ..PolicyConfig::default() }).unwrap();
    let handles = [add(&mut scheduler, 1, 64, 32), add(&mut scheduler, 2, 64, 32)];
    ready(&mut scheduler, &handles);
    for (index, handle) in handles.into_iter().enumerate() {
        let profile = &mut scheduler.request_mut(handle).unwrap().acceptance;
        *profile = Default::default();
        for _ in 0..8 {
            if index == 0 { profile.observe(speculation::ProposalSource::Mtp, 0, 1, 1, false, None); }
            else { profile.observe(speculation::ProposalSource::Mtp, 0, 1, 0, true, Some(&[0.3; 15])); }
        }
    }
    let mask = handles.iter().fold(0, |mask, handle| mask | 1 << handle.slot);
    let mut values = [utility::LaneValue { output_token: 1.0, ..Default::default() }; REQUEST_LANES];
    values[handles[1].slot as usize].output_token = 2.0;
    let lengths = scheduler.prefix_lengths_with_values(mask, 2, 1, 2, 3, false, &values, false);
    assert_eq!(lengths[handles[0].slot as usize], 1);
    assert_eq!(lengths[handles[1].slot as usize], 2);
}

#[test]
fn terminal_decode_records_preserve_the_consumed_output_delta() {
    for (consumed, emitted) in [(0, 1), (1, 0)] {
        let mut scheduler = engine(1, false);
        let handle = add(&mut scheduler, 1, 64, 32);
        ready(&mut scheduler, &[handle]);
        let plan = scheduler.plan().unwrap();
        let fence = Fence(scheduler.submitted.0 + 1);
        scheduler.commit(&plan, fence).unwrap();
        let request = scheduler.request(handle).unwrap();
        let before = (request.consumed_tokens, request.generated_tokens);
        let mut records = vec![CompletionRecord::default(); scheduler.limits.request_capacity()];
        records[handle.slot as usize] = CompletionRecord {
            generation: handle.generation as i64, submission_sequence: fence.0 as i64,
            consumed_tokens: before.0 as i64 + consumed, generated_tokens: before.1 as i64 + emitted,
            token_id: if emitted == 0 { -1 } else { 99 }, status: 2, ..Default::default()
        };
        assert!(scheduler.complete(fence, &records, gpu_time(MS)).is_err());
        let request = scheduler.request(handle).unwrap();
        assert_eq!((request.consumed_tokens, request.generated_tokens), before);
        assert!(scheduler.submissions.contains_key(&fence));
    }
}

#[test]
fn registered_ragged_graphs_preserve_physical_width_above_average_row_budget() {
    use crate::execution::PrefixAllocation;
    for (batch, packed) in [(4, 5), (32, 128)] {
        for allocation in [PrefixAllocation::Host, PrefixAllocation::Gpu] {
            let mut scheduler = engine(batch, true);
            scheduler.enable_speculation(&[1, 8]).unwrap();
            let mut graphs = super::tests::duration_graphs(batch);
            graphs.retain(|graph| graph.kind != GraphKind::Speculative || graph.token_capacity == 1);
            for graph in &mut graphs { graph.estimated_duration_ns = 10 * MS; }
            graphs.push(crate::execution::GraphDescriptor { tuned_bindings: vec![],
                name: format!("speculative-b{batch}-d7-w8-p{packed}"),
                kind: GraphKind::Speculative, sampling_mode: SamplingMode::General,
                confidence_width: if allocation == PrefixAllocation::Gpu { 7 } else { 0 },
                draft_tokens: 7, handle: 1, batch_capacity: batch, token_capacity: 8,
                packed_token_capacity: packed, prefill_capacity: 0,
                estimated_duration_ns: MS, input_bindings: vec![],
            });
            scheduler.measured_graph_durations(&graphs).unwrap();
            scheduler.capabilities.speculative_allocations = vec![(batch, 7, 8, packed, allocation)];
            let registered: Vec<_> = crate::execution::BATCH_CAPACITIES.into_iter()
                .take_while(|&capacity| capacity <= batch).map(|capacity| (capacity, 1))
                .chain(std::iter::once((batch, 8))).collect();
            scheduler.speculative_graphs(&registered).unwrap();
            for id in 0..batch {
                let handle = add(&mut scheduler, id as u32, 64, 32);
                let request = scheduler.request_mut(handle).unwrap();
                request.phase = RequestPhase::Decode;
                request.consumed_tokens = 64;
                request.scheduled_consumed = 64;
                request.generated_tokens = 1;
            }
            let plan = scheduler.plan().unwrap();
            assert_eq!((plan.batch_capacity, plan.draft_tokens, plan.token_capacity,
                plan.packed_token_capacity), (batch, 7, 8, packed));
            assert_eq!(plan.lanes.len(), batch);
            plan.validate_mixed().unwrap();
            match allocation {
                PrefixAllocation::Host => assert_eq!(plan.lanes.iter().map(|lane| lane.tokens).sum::<usize>(), packed),
                PrefixAllocation::Gpu => assert!(plan.lanes.iter().all(|lane| lane.tokens == 8)),
            }
        }
    }
}

#[test]
fn width_one_gpu_prefix_round_completes_without_a_confidence_sidecar() {
    let mut scheduler = engine(1, true);
    scheduler.fixed_speculative_width(Some(1)).unwrap();
    scheduler.capabilities.speculative_allocations = vec![(1, 0, 1, 1, crate::execution::PrefixAllocation::Gpu)];
    let handle = add(&mut scheduler, 5, 64, 32);
    ready(&mut scheduler, &[handle]);
    let plan = scheduler.plan().unwrap();
    assert_eq!((plan.draft_tokens, plan.token_capacity, plan.packed_token_capacity), (0, 1, 1));
    let fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&plan, fence).unwrap();
    let records = completion_records(&scheduler, &plan, fence, 0);
    assert_eq!(scheduler.complete(fence, &records, gpu_time(MS)).unwrap().len(), 1);
}

fn fixed_host_round_engine(batch: usize, packed: usize) -> RequestScheduler {
    let mut scheduler = engine(4, true);
    scheduler.fixed_speculative_width(Some(8)).unwrap();
    let mut graphs = super::tests::duration_graphs(4);
    for graph in &mut graphs { graph.estimated_duration_ns = 10 * MS; }
    graphs.push(crate::execution::GraphDescriptor { tuned_bindings: vec![], confidence_width: 0,
        name: format!("speculative-b{batch}-d7-w8-p{packed}"), kind: GraphKind::Speculative,
        sampling_mode: SamplingMode::General, draft_tokens: 7, handle: 1,
        batch_capacity: batch, token_capacity: 8, packed_token_capacity: packed,
        prefill_capacity: 0, estimated_duration_ns: MS, input_bindings: vec![] });
    scheduler.measured_graph_durations(&graphs).unwrap();
    scheduler
}

fn fixed_host_requests(scheduler: &mut RequestScheduler, tails: &[u32]) -> Vec<RequestHandle> {
    tails.iter().enumerate().map(|(id, &tail)| {
        let handle = add(scheduler, id as u32 + 1, 64, tail + 1);
        let request = scheduler.request_mut(handle).unwrap();
        request.phase = RequestPhase::Decode;
        request.consumed_tokens = 64;
        request.scheduled_consumed = 64;
        request.generated_tokens = 1;
        handle
    }).collect()
}

#[test]
fn fixed_host_round_does_not_shorten_prefixes_to_enter_a_smaller_physical_graph() {
    let mut scheduler = fixed_host_round_engine(4, 5);
    fixed_host_requests(&mut scheduler, &[8, 8]);
    let plan = scheduler.plan().unwrap();
    assert_eq!(plan.token_capacity, 8);
    assert_eq!(plan.lanes.len(), 2);
    assert!(plan.lanes.iter().all(|lane| lane.tokens == 8));
    assert!(plan.packed_token_capacity >= 16);
    let (values, age, _) = scheduler.lane_values(0);
    let mut candidates = Vec::new();
    scheduler.packed_round_candidates(&mut candidates, &values, age, false);
    assert!(!candidates.iter().any(|candidate| candidate.batch == 4 && candidate.packed_token_capacity == 5));
}

#[test]
fn fixed_host_round_selects_complete_request_tails_that_fit_the_physical_budget() {
    let mut scheduler = fixed_host_round_engine(4, 5);
    let handles = fixed_host_requests(&mut scheduler, &[8, 2, 3]);
    assert_eq!(scheduler.speculative_duration(scheduler.request(handles[1]).unwrap(), 4, 8), Some(MS));
    let plan = scheduler.plan().unwrap();
    assert_eq!((plan.token_capacity, plan.packed_token_capacity), (8, 5));
    assert_eq!(plan.lanes.iter().map(|lane| (lane.request, lane.tokens)).collect::<Vec<_>>(),
        vec![(handles[1], 2), (handles[2], 3)]);
    scheduler.commit(&plan, Fence(1)).unwrap();
    assert_eq!(scheduler.pending_token_ceiling(handles[0]), 0);
    assert_eq!(scheduler.pending_token_ceiling(handles[1]), 2);
    assert_eq!(scheduler.pending_token_ceiling(handles[2]), 3);
}

#[test]
fn fixed_host_cohort_competes_by_full_prefix_utility_within_the_row_budget() {
    let mut scheduler = fixed_host_round_engine(4, 8);
    let handles = fixed_host_requests(&mut scheduler, &[8, 4, 3, 1]);
    let mut values = [utility::LaneValue::default(); REQUEST_LANES];
    for (handle, value) in handles.iter().zip([11.0, 6.0, 5.0, 1.0]) {
        let request = scheduler.request(*handle).unwrap();
        let length = (request.max_new_tokens - request.generated_tokens) as usize;
        values[handle.slot as usize].output_token = value / scheduler.expected_speculative_tokens(request, length);
    }
    let mut candidates = Vec::new();
    scheduler.packed_round_candidates(&mut candidates, &values, 0.0, false);
    let candidate = candidates.iter().find(|candidate|
        candidate.batch == 4 && candidate.packed_token_capacity == 8).unwrap();
    let expected = handles[1..].iter().fold(0, |mask, handle| mask | 1 << handle.slot);
    assert_eq!(candidate.mask, expected);
    let lengths = scheduler.round_lengths(candidate.mask, candidate.batch, candidate.draft_tokens,
        candidate.capacity, candidate.packed_token_capacity);
    assert_eq!(handles.iter().map(|handle| lengths[handle.slot as usize]).collect::<Vec<_>>(), [0, 4, 3, 1]);
}

#[test]
#[ignore = "manual CPU scheduler planning measurement"]
fn mixed_tail_planning_cpu_measurement() {
    use std::time::Instant;

    for (active, shapes) in [(32, vec![(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 11), (2, 13), (2, 16), (4, 5), (4, 6), (4, 7), (4, 8), (4, 10), (4, 11), (4, 13), (4, 16), (4, 20), (4, 25), (4, 28), (4, 32), (8, 9), (8, 10), (8, 12), (8, 15), (8, 19), (8, 24), (8, 27), (8, 31), (8, 39), (8, 44), (8, 50), (8, 64), (16, 17), (16, 19), (16, 22), (16, 28), (16, 32), (16, 36), (16, 41), (16, 47), (16, 54), (16, 61), (16, 78), (16, 88), (16, 100), (16, 128), (32, 33), (32, 39), (32, 46), (32, 55), (32, 65), (32, 77), (32, 91), (32, 108), (32, 128)]), (64, vec![(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 9), (2, 11), (2, 13), (2, 16), (4, 5), (4, 6), (4, 7), (4, 8), (4, 10), (4, 11), (4, 13), (4, 16), (4, 20), (4, 25), (4, 32), (8, 9), (8, 10), (8, 12), (8, 15), (8, 19), (8, 24), (8, 27), (8, 31), (8, 39), (8, 50), (8, 64), (16, 17), (16, 19), (16, 22), (16, 28), (16, 36), (16, 41), (16, 47), (16, 54), (16, 61), (16, 78), (16, 100), (16, 128), (32, 33), (32, 39), (32, 46), (32, 55), (32, 65), (32, 77), (32, 91), (32, 108), (32, 128), (64, 65), (64, 77), (64, 91), (64, 108), (64, 128)])] {
        for mode in ["fixed_host", "adaptive_host", "adaptive_gpu"] {
        let mut scheduler = engine(active, true);
        scheduler.enable_speculation(&[1, 8]).unwrap();
        if mode == "fixed_host" { scheduler.fixed_speculative_width(Some(8)).unwrap(); }
        let mut graphs = super::tests::duration_graphs(active);
        graphs.retain(|graph| graph.kind != GraphKind::Speculative || graph.token_capacity == 1);
        for &(batch, packed) in &shapes {
            graphs.push(crate::execution::GraphDescriptor { tuned_bindings: vec![], confidence_width: if mode == "adaptive_gpu" { 7 } else { 0 },
                name: format!("speculative-b{batch}-d7-w8-p{packed}"), kind: GraphKind::Speculative,
                sampling_mode: SamplingMode::General, draft_tokens: 7, handle: 1,
                batch_capacity: batch, token_capacity: 8, packed_token_capacity: packed,
                prefill_capacity: 0, estimated_duration_ns: MS + packed as u64 * 1_000,
                input_bindings: vec![] });
        }
        scheduler.measured_graph_durations(&graphs).unwrap();
        if mode == "adaptive_gpu" {
            scheduler.capabilities.speculative_allocations = shapes.iter()
                .map(|&(batch, packed)| (batch, 7, 8, packed, crate::execution::PrefixAllocation::Gpu)).collect();
        }
        let tails: Vec<_> = (0..active).map(|slot| 1 + (slot % 8) as u32).collect();
        fixed_host_requests(&mut scheduler, &tails);
        for _ in 0..3 { std::hint::black_box(scheduler.plan().unwrap()); }
        let mut durations = Vec::new();
        for _ in 0..30 {
            let start = Instant::now();
            std::hint::black_box(scheduler.plan().unwrap());
            durations.push(start.elapsed().as_nanos() as u64);
        }
        durations.sort_unstable();
        println!("{mode}_plan C={active} W=8 max_P=128 wider_graphs=64 cyclic_tails=1..8 iterations=30 ns min={} median={} p90={} max={}",
            durations[0], durations[15], durations[27], durations[29]);
        }
    }
}


#[test]
fn fixed_host_cohort_enforces_lane_capacity_when_three_short_tails_would_be_more_valuable() {
    let mut scheduler = fixed_host_round_engine(2, 8);
    let handles = fixed_host_requests(&mut scheduler, &[4, 3, 1, 1]);
    let mut values = [utility::LaneValue::default(); REQUEST_LANES];
    for (handle, value) in handles.iter().zip([6.0, 5.0, 4.0, 4.0]) {
        let request = scheduler.request(*handle).unwrap();
        let length = (request.max_new_tokens - request.generated_tokens) as usize;
        values[handle.slot as usize].output_token = value / scheduler.expected_speculative_tokens(request, length);
    }
    let mut candidates = Vec::new();
    scheduler.packed_round_candidates(&mut candidates, &values, 0.0, false);
    let candidate = candidates.iter().find(|candidate|
        candidate.batch == 2 && candidate.packed_token_capacity == 8).unwrap();
    assert_eq!(candidate.mask, (1 << handles[0].slot) | (1 << handles[1].slot));
}


#[test]
fn gpu_round_reassigns_a_finished_lanes_physical_row_to_a_progressing_request() {
    let mut scheduler = joined_round_engine(crate::execution::PrefixAllocation::Gpu);
    let handles = [add(&mut scheduler, 1, 64, 32), add(&mut scheduler, 2, 64, 32)];
    ready(&mut scheduler, &handles);
    let first = scheduler.plan().unwrap();
    let first_fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&first, first_fence).unwrap();
    let second = scheduler.plan().unwrap();
    assert_eq!(second.packed_token_capacity, 5);
    assert_eq!(second.lanes.len(), 2);
    let second_fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&second, second_fence).unwrap();
    let mut records = completion_records(&scheduler, &first, first_fence, 0);
    records[handles[0].slot as usize].status = 2;
    records[handles[0].slot as usize].vector.as_mut().unwrap().offered_drafts = 1;
    records[handles[1].slot as usize].vector.as_mut().unwrap().offered_drafts = 2;
    scheduler.complete_packed_round(first_fence, &records, gpu_time(MS), &vec![vec![0.5; 15]; 2]).unwrap();
    let request = scheduler.request(handles[0]).unwrap();
    let finished = CompletionRecord {
        generation: handles[0].generation as i64, submission_sequence: second_fence.0 as i64,
        consumed_tokens: request.consumed_tokens as i64, generated_tokens: request.generated_tokens as i64,
        token_id: -1, status: 2,
        vector: Some(TokenVector { verification_width: 8, draft_cache_consumed: -1,
            ..Default::default() }), ..Default::default()
    };
    let mut records = completion_records(&scheduler, &second, second_fence, 0);
    records[handles[0].slot as usize] = finished;
    records[handles[1].slot as usize].vector.as_mut().unwrap().offered_drafts = 4;
    let output = scheduler.complete_packed_round(second_fence, &records, gpu_time(MS), &[vec![0.0; 15], vec![0.5; 15]]).unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].request, handles[1]);
    assert_eq!(scheduler.request(handles[0]).unwrap().pending, 0);
}

#[test]
fn cached_host_prefix_ranks_match_greedy_allocation_for_every_small_cohort_order() {
    for alpha in [0.5, 1.0, 2.0] {
        for tails in [[8, 4, 3, 1], [2, 8, 1, 6], [1, 1, 1, 1]] {
            let mut scheduler = engine(4, true);
            scheduler.configure_policy(PolicyConfig { adaptive_verification_alpha: alpha, ..PolicyConfig::default() }).unwrap();
            let handles = fixed_host_requests(&mut scheduler, &tails);
            for (index, handle) in handles.iter().enumerate() {
                let profile = &mut scheduler.request_mut(*handle).unwrap().acceptance;
                let probability = [0.1, 0.9, 0.5, 0.5][index];
                let survival: [f32; 15] = std::array::from_fn(|depth| (probability as f32).powi(depth as i32 + 1));
                for _ in 0..8 { profile.observe(speculation::ProposalSource::Mtp, 1, 7, 0, true, Some(&survival)); }
            }
            let available: Vec<_> = handles.iter().map(|handle| handle.slot as usize).collect();
            for weights in [[1.0; 4], [1.0, 2.0, 0.0, 3.0]] {
                let mut values = [utility::LaneValue::default(); REQUEST_LANES];
                for (&slot, weight) in available.iter().zip(weights) { values[slot].output_token = weight; }
                for optimistic in [false, true] {
                    let ranking = scheduler.host_prefix_ranking(&available, 7, 8, &values, optimistic);
                    for a in 0..4 { for b in 0..4 { for c in 0..4 { for d in 0..4 {
                        let order = [a, b, c, d];
                        if order.iter().fold(0, |mask, index| mask | 1 << index) != 15 { continue; }
                        for budget in [4, 5, 8, 13, 20, 32] {
                            let mut frontier = std::collections::BinaryHeap::new();
                            let mut lengths = [0; REQUEST_LANES];
                            let mut mask = 0;
                            for (count, index) in order.into_iter().enumerate() {
                                let slot = available[index];
                                mask |= 1 << slot;
                                ranking.extend_cohort(slot, &mut lengths, &mut frontier, budget - count - 1);
                                let reference = scheduler.prefix_lengths_with_values(mask, 4, 7, 8, budget,
                                    false, &values, optimistic);
                                assert_eq!(lengths, reference, "alpha={alpha} tails={tails:?} weights={weights:?} order={order:?} budget={budget} optimistic={optimistic}");
                            }
                        }
                    } } } }
                }
            }
        }
    }
}

fn single_lane_packed_engine(page_tokens: usize) -> RequestScheduler {
    let mut scheduler = engine_with_geometry(1, true, page_tokens, None);
    scheduler.enable_speculation(&[1, 8]).unwrap();
    let mut graphs = super::tests::duration_graphs(1);
    for graph in &mut graphs { graph.estimated_duration_ns = 10 * MS; }
    graphs.push(crate::execution::GraphDescriptor { tuned_bindings: vec![], confidence_width: 7,
        name: "speculative-b1-d7-w8-p3".into(), kind: GraphKind::Speculative,
        sampling_mode: SamplingMode::General, draft_tokens: 7, handle: 1,
        batch_capacity: 1, token_capacity: 8, packed_token_capacity: 3,
        prefill_capacity: 0, estimated_duration_ns: MS, input_bindings: vec![] });
    scheduler.measured_graph_durations(&graphs).unwrap();
    scheduler.capabilities.speculative_allocations = vec![(1, 7, 8, 3, crate::execution::PrefixAllocation::Gpu)];
    scheduler
}

fn packed_completion_records(scheduler: &RequestScheduler, plan: &ScheduledPlan,
    fence: Fence, accepted: usize) -> Vec<CompletionRecord> {
    let mut records = completion_records(scheduler, plan, fence, accepted);
    for lane in &plan.lanes {
        records[lane.request.slot as usize].vector.as_mut().unwrap().offered_drafts =
            lane.tokens.min(plan.packed_token_capacity).saturating_sub(1) as i64;
    }
    records
}

#[test]
fn pending_packed_output_bound_preserves_wider_rounds_with_nine_tokens_remaining() {
    let mut scheduler = single_lane_packed_engine(64);
    let handle = add(&mut scheduler, 5, 64, 10);
    ready(&mut scheduler, &[handle]);
    let start = scheduler.request(handle).unwrap().consumed_tokens;
    assert_eq!(scheduler.request(handle).unwrap().generated_tokens, 1);
    let first = scheduler.plan().unwrap();
    assert_eq!((first.token_capacity, first.packed_token_capacity, first.lanes[0].tokens), (8, 3, 8));
    let first_fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&first, first_fence).unwrap();
    assert_eq!(scheduler.pending_token_ceiling(handle), 3);
    assert_eq!(scheduler.request(handle).unwrap().scheduled_consumed, start + 3);
    let second = scheduler.plan().unwrap();
    assert_eq!((second.token_capacity, second.packed_token_capacity, second.lanes[0].tokens), (8, 3, 6));
    assert_eq!(second.lanes[0].start_position, start + 3);
    let second_fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&second, second_fence).unwrap();
    assert_eq!(scheduler.pending_token_ceiling(handle), 6);
    assert_eq!(scheduler.request(handle).unwrap().scheduled_consumed, start + 6);
    for (plan, fence) in [(&first, first_fence), (&second, second_fence)] {
        scheduler.advance_to(scheduler.submissions[&fence].expected_finish_ns).unwrap();
        let records = packed_completion_records(&scheduler, plan, fence, 2);
        scheduler.complete_packed_round(fence, &records, gpu_time(MS), &[vec![0.5; 15]]).unwrap();
    }
    let tail = scheduler.plan().unwrap();
    assert_eq!((tail.token_capacity, tail.lanes[0].tokens), (8, 3));
    let fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&tail, fence).unwrap();
    assert!(scheduler.plan().is_none());
    let records = packed_completion_records(&scheduler, &tail, fence, 2);
    scheduler.complete_packed_round(fence, &records, gpu_time(MS), &[vec![0.5; 15]]).unwrap();
    let request = scheduler.request(handle).unwrap();
    assert_eq!(request.generated_tokens, request.max_new_tokens);
    assert_eq!(request.pending, 0);
}

#[test]
fn pending_packed_context_bound_keeps_full_execution_prefix_inside_the_context() {
    let mut scheduler = single_lane_packed_engine(4);
    scheduler.limits.max_sequence_tokens = 576;
    scheduler.capabilities.draft_pages_per_request = 2;
    let handle = add(&mut scheduler, 5, 568, 8);
    ready(&mut scheduler, &[handle]);
    let first = scheduler.plan().unwrap();
    assert_eq!((first.token_capacity, first.lanes[0].tokens), (8, 7));
    let first_fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&first, first_fence).unwrap();
    assert_eq!(scheduler.request(handle).unwrap().scheduled_consumed, 571);
    let second = scheduler.plan().unwrap();
    assert_eq!((second.token_capacity, second.lanes[0].start_position, second.lanes[0].tokens), (8, 571, 4));
    assert!(second.lanes[0].start_position + second.lanes[0].tokens as u64 <= 576);
    scheduler.commit(&second, Fence(first_fence.0 + 1)).unwrap();
    assert_eq!(scheduler.request(handle).unwrap().scheduled_consumed, 574);
}

#[test]
fn pending_packed_advance_keeps_full_page_and_draft_storage_fenced() {
    let mut scheduler = single_lane_packed_engine(4);
    scheduler.capabilities.draft_pages_per_request = 2;
    let handle = add(&mut scheduler, 5, 64, 10);
    ready(&mut scheduler, &[handle]);
    let first = scheduler.plan().unwrap();
    let first_fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&first, first_fence).unwrap();
    let live = scheduler.request(handle).unwrap();
    let prefix_pages = live.prefix_tokens as usize / 4;
    let first_touch = first.lanes[0].start_position as usize + first.lanes[0].tokens;
    assert_eq!(live.private_pages.len(), first_touch.div_ceil(4) - prefix_pages + 2);
    assert!(live.scheduled_consumed < first_touch as u64);
    let second = scheduler.plan().unwrap();
    let second_fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&second, second_fence).unwrap();
    let second_touch = second.lanes[0].start_position as usize + second.lanes[0].tokens;
    let expected_private = second_touch.div_ceil(4) - prefix_pages + 2;
    assert_eq!(scheduler.request(handle).unwrap().private_pages.len(), expected_private);
    let records = packed_completion_records(&scheduler, &first, first_fence, 0);
    scheduler.complete_packed_round(first_fence, &records, gpu_time(MS), &[vec![0.5; 15]]).unwrap();
    let live = scheduler.request(handle).unwrap();
    assert_eq!(live.scheduled_consumed, live.consumed_tokens + 3);
    assert_eq!(live.private_pages.len(), expected_private);
    let records = packed_completion_records(&scheduler, &second, second_fence, 2);
    let held = live.private_pages.len() + live.reserved_pages.len();
    let free = scheduler.free_pages.len();
    scheduler.cancel(handle).unwrap();
    assert_eq!(scheduler.free_pages.len(), free);
    scheduler.complete_packed_round(second_fence, &records, gpu_time(MS), &[vec![0.5; 15]]).unwrap();
    assert_eq!(scheduler.free_pages.len(), free + held);
}

#[test]
fn sixteen_wide_round_commits_fifteen_drafts_and_respects_output_limit() {
    for remaining in [1usize, 8, 16] {
        let mut s = engine(1, false);
        s.enable_speculation(&[1, 8, 16]).unwrap();
        s.fixed_speculative_width(Some(16)).unwrap();
        let mut graph = super::tests::duration_graphs(1).into_iter()
            .find(|g| g.kind == GraphKind::Speculative && g.token_capacity == 8).unwrap();
        graph.token_capacity = 16;
        graph.draft_tokens = 15;
        graph.packed_token_capacity = 16;
        let mut graphs = super::tests::duration_graphs(1);
        graphs.push(graph);
        s.measured_graph_durations(&graphs).unwrap();
        let handle = add(&mut s, 1, 64, (remaining + 1) as u32);
        ready(&mut s, &[handle]);
        let plan = s.plan().unwrap();
        assert_eq!(plan.token_capacity, 16);
        assert_eq!(plan.lanes[0].tokens, remaining);
        let fence = Fence(s.submitted.0 + 1);
        s.commit(&plan, fence).unwrap();
        let mut records = completion_records(&s, &plan, fence, 15);
        records[handle.slot as usize].vector.as_mut().unwrap().offered_drafts = (remaining - 1) as i64;
        s.complete(fence, &records, gpu_time(MS)).unwrap();
        assert_eq!(s.request(handle).unwrap().generated_tokens, (remaining + 1) as u32);
    }
}

#[test]
fn fixed_host_prefixes_preserve_lengths_above_255() {
    let fixture = engine(1, false);
    let mut capabilities = fixture.capabilities.clone();
    capabilities.completion_token_capacity = 257;
    let mut scheduler = RequestScheduler::new([0; 32], fixture.limits.clone(), capabilities).unwrap();
    scheduler.enable_speculation(&[1, 257]).unwrap();
    scheduler.fixed_speculative_width(Some(257)).unwrap();
    let graphs = vec![crate::execution::GraphDescriptor {
        tuned_bindings: vec![], confidence_width: 0, name: "speculative-b1-d256-w257-p257".into(),
        kind: GraphKind::Speculative, sampling_mode: SamplingMode::General, draft_tokens: 256,
        handle: 1, batch_capacity: 1, token_capacity: 257, packed_token_capacity: 257,
        prefill_capacity: 0, estimated_duration_ns: MS, input_bindings: vec![],
    }];
    scheduler.measured_graph_durations(&graphs).unwrap();
    let handles = fixed_host_requests(&mut scheduler, &[257]);
    let plan = scheduler.plan().unwrap();
    assert_eq!(plan.kind, GraphKind::Speculative);
    assert_eq!(plan.lanes[0].request, handles[0]);
    assert_eq!(plan.lanes[0].tokens, 257);
    assert_eq!(plan.token_capacity, 257);
}
