/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use crate::execution::PrefixAllocation;

fn planning_scheduler(active: usize, allocation: PrefixAllocation) -> RequestScheduler {
    let mut scheduler = engine(active, true);
    let mut graphs = super::super::tests::duration_graphs(active);
    let rectangular = graphs.clone();
    for graph in rectangular.iter().filter(|g| g.kind == GraphKind::Speculative) {
        for rows in [1, 2, 4] {
            let packed = graph.batch_capacity * rows;
            if packed >= graph.packed_token_capacity { continue; }
            let mut graph = graph.clone();
            graph.name = format!("{}-p{packed}", graph.name);
            graph.packed_token_capacity = packed;
            graphs.push(graph);
        }
    }
    for graph in &mut graphs {
        graph.estimated_duration_ns = if graph.kind == GraphKind::Speculative {
            100_000 + graph.batch_capacity as u64 * 10_000
                + graph.draft_tokens as u64 * 15_000 + graph.packed_token_capacity as u64 * 2_000
        } else { MS };
        if graph.kind == GraphKind::Speculative {
            graph.confidence_width = if allocation == PrefixAllocation::Gpu { graph.draft_tokens } else { 0 };
            scheduler.capabilities.speculative_allocations.push((graph.batch_capacity,
                graph.draft_tokens, graph.token_capacity, graph.packed_token_capacity, allocation));
        }
    }
    scheduler.measured_graph_durations(&graphs).unwrap();
    for id in 0..active {
        let handle = add(&mut scheduler, id as u32, 64, 256);
        let request = scheduler.request_mut(handle).unwrap();
        request.phase = RequestPhase::Decode;
        request.consumed_tokens = 80;
        request.scheduled_consumed = 80;
        request.generated_tokens = 17;
        for sample in 0..8 {
            let accepted = (id + sample) % 8;
            request.acceptance.observe(speculation::ProposalSource::Mtp, 17,
                7, accepted, accepted < 7, None);
        }
    }
    scheduler
}

fn greedy_lengths(scheduler: &RequestScheduler, mask: u64, batch: usize, drafts: usize,
    width: usize, packed: usize, preserve: bool, values: &[utility::LaneValue; REQUEST_LANES],
    optimistic: bool) -> [usize; REQUEST_LANES] {
    let ceilings: [_; REQUEST_LANES] = std::array::from_fn(|slot| {
        if mask & (1 << slot) == 0 { 0 }
        else { scheduler.speculative_tokens(scheduler.slots[slot].request.as_ref().unwrap(), width.min(drafts + 1)) }
    });
    let gpu = scheduler.gpu_prefix_allocation(batch, drafts, width, packed);
    if (gpu && preserve) || (!gpu && scheduler.speculative_fixed_width.is_some())
        || ceilings.iter().sum::<usize>() <= packed { return ceilings; }
    let probabilities: [[f64; 7]; REQUEST_LANES] = std::array::from_fn(|slot| {
        let Some(request) = scheduler.slots.get(slot).and_then(|slot| slot.request.as_ref()) else { return [0.0; 7]; };
        if gpu { scheduler.speculative_curve(request, optimistic) }
        else { request.acceptance.curve(scheduler.proposal_source, request.generated_tokens, optimistic)
            .map(|p| p.powf(scheduler.policy.adaptive_verification_alpha) * values[slot].output_token) }
    });
    let mut lengths = [0; REQUEST_LANES];
    for slot in policy::bits(mask) { lengths[slot] = 1; }
    for _ in 0..packed.saturating_sub(mask.count_ones() as usize) {
        let Some(slot) = policy::bits(mask).filter(|&slot| lengths[slot] < ceilings[slot])
            .max_by(|&a, &b| probabilities[a][lengths[a] - 1].total_cmp(&probabilities[b][lengths[b] - 1])
                .then_with(|| lengths[b].cmp(&lengths[a])).then_with(|| b.cmp(&a))) else { break; };
        lengths[slot] += 1;
    }
    lengths
}

#[test]
fn ranked_prefix_allocation_matches_greedy_frontiers() {
    for active in [1, 3, 8, 32, 64] {
        for allocation in [PrefixAllocation::Host, PrefixAllocation::Gpu] {
            for source in [speculation::ProposalSource::Mtp, speculation::ProposalSource::Copy] {
                let mut scheduler = planning_scheduler(active, allocation);
                scheduler.proposal_source = source;
                let round = ScheduledStep {
                    kind: GraphKind::Speculative, batch_capacity: active.next_power_of_two(),
                    draft_tokens: 7, token_capacity: 8, packed_token_capacity: active.next_power_of_two() * 8,
                    mixed: None, requires_feedback: false,
                    lanes: scheduler.slots.iter().filter_map(|slot| slot.request.as_ref()).map(|request|
                        ScheduledLane { request: request.handle, start_position: request.consumed_tokens, tokens: 8 }).collect(),
                };
                let mut records = vec![CompletionRecord::default(); scheduler.limits.request_capacity()];
                let confidence: Vec<_> = round.lanes.iter().enumerate().map(|(row, lane)| {
                    records[lane.request.slot as usize] = CompletionRecord {
                        status: 1, generated_tokens: 18, vector: Some(TokenVector {
                            emitted_count: 1, offered_drafts: 7, verification_width: 8, ..Default::default()
                        }), ..Default::default()
                    };
                    std::array::from_fn(|depth| [0.0, 0.25, 0.75, 1.0][(row + depth) % 4])
                }).collect();
                scheduler.observe_packed_forecasts(&round, &records, &confidence);
                for seed in 0..4 {
                    scheduler.policy.adaptive_verification_alpha = [0.5, 1.0, 2.0, 4.0][seed];
                    let mut mask = 0u64;
                    let mut values = [utility::LaneValue::default(); REQUEST_LANES];
                    for (slot, entry) in scheduler.slots.iter_mut().enumerate() {
                        let Some(request) = entry.request.as_mut() else { continue; };
                        if (slot + seed) % 3 != 1 { mask |= 1 << slot; }
                        request.max_new_tokens = request.generated_tokens + 1 + ((slot + seed) % 8) as u32;
                        request.host_stops = seed == 2 && slot % 7 == 0;
                        request.grammar_constrained = seed == 3 && slot % 11 == 0;
                        request.copy_proposal = Some(speculation::CopyProposal {
                            consumed: request.consumed_tokens, tokens: [9; 7], count: (slot + seed) % 8,
                            grammar_masks: vec![],
                        });
                        if seed == 0 { request.acceptance = Default::default(); }
                        else { request.acceptance.observe(source, request.generated_tokens,
                            7, slot % 8, slot % 8 < 7, None); }
                        values[slot].output_token = [0.0, 1.0, 1.0, 4.0][(slot + seed) % 4];
                    }
                    scheduler.limits.max_sequence_tokens = if seed == 3 { 83 } else { 8192 };
                    scheduler.forecasts.population = [0.9, 0.9, 0.5, 0.5, 0.25, 0.0, 0.0];
                    let lanes = mask.count_ones() as usize;
                    for width in [1, 2, 4, 8] {
                        for drafts in [width - 1, 7] {
                            for packed in lanes..=lanes * width {
                                scheduler.capabilities.speculative_allocations = vec![
                                    (active.next_power_of_two(), drafts, width, packed, allocation)];
                                for (fixed, preserve, optimistic) in [
                                    (false, false, false), (false, false, true),
                                    (false, true, false), (true, false, false),
                                ] {
                                    scheduler.speculative_fixed_width = fixed.then_some(width);
                                    let expected = greedy_lengths(&scheduler, mask, active.next_power_of_two(),
                                        drafts, width, packed, preserve, &values, optimistic);
                                    let actual = scheduler.prefix_lengths_with_values(mask, active.next_power_of_two(),
                                        drafts, width, packed, preserve, &values, optimistic);
                                    assert_eq!(actual, expected,
                                        "active={active} allocation={allocation:?} source={source:?} seed={seed} width={width} drafts={drafts} packed={packed} fixed={fixed} preserve={preserve} optimistic={optimistic}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn host_cohort_ranking_matches_greedy_reallocation() {
    for active in [3, 8, 32, 64] {
        let scheduler = planning_scheduler(active, PrefixAllocation::Host);
        let (values, _, _) = scheduler.lane_values(scheduler.window_remaining().unwrap());
        let available: Vec<_> = (0..active).rev().collect();
        for optimistic in [false, true] {
            let ranking = scheduler.host_prefix_ranking(&available, 7, 8, &values, optimistic);
            for packed in [active, active * 2, active * 4] {
                let mut lengths = [0; REQUEST_LANES];
                let mut frontier = std::collections::BinaryHeap::new();
                let mut mask = 0;
                for (count, &slot) in available.iter().enumerate() {
                    mask |= 1 << slot;
                    ranking.extend_cohort(slot, &mut lengths, &mut frontier, packed - count - 1);
                    assert_eq!(lengths, greedy_lengths(&scheduler, mask, active.next_power_of_two(),
                        7, 8, packed, false, &values, optimistic));
                }
            }
        }
    }
}

#[test]
#[ignore = "CPU planning benchmark; run in release mode with --ignored --nocapture --test-threads=1"]
fn benchmark_packed_planning() {
    assert!(!cfg!(debug_assertions), "planning measurements require a release build");
    for active in [8, 16, 32, 64] {
        for allocation in [PrefixAllocation::Host, PrefixAllocation::Gpu] {
            let mut scheduler = planning_scheduler(active, allocation);
            let first = scheduler.plan().unwrap();
            assert!(!first.requires_feedback);
            scheduler.commit(&first, Fence(scheduler.submitted.0 + 1)).unwrap();
            for probe in [false, true] {
                scheduler.refresh.acceptance_credit_ns = if probe { refresh::CREDIT_CAP_NS } else { 0.0 };
                for _ in 0..10 { std::hint::black_box(scheduler.plan()); }
                let samples: Vec<_> = (0..9).map(|_| {
                    let start = std::time::Instant::now();
                    for _ in 0..50 { std::hint::black_box(scheduler.plan()); }
                    start.elapsed().as_nanos() / 50
                }).collect();
                println!("planning requests={active} allocation={allocation:?} queued=1 probe={probe} ns={samples:?}");
            }
        }
    }
}
