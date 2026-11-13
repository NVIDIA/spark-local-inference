/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;

#[test]
fn unknown_checkpoint_cost_keeps_final_message_anchor_and_all_semantic_marks() {
    let mut s = scheduler(2, false);
    s.graph_durations(&super::tests::duration_graphs(4)).unwrap();
    let mut request = spec(None, 700, 3);
    request.message_boundaries = (1..700).collect();
    let handle = s.admit(request).unwrap();
    assert_eq!(s.request(handle).unwrap().message_boundaries.len(), 699);
    run_step(&mut s);
    assert_eq!(s.plan().unwrap().lanes[0].tokens, 512);
    run_step(&mut s);
    assert_eq!((s.plan().unwrap().kind, s.plan().unwrap().lanes[0].tokens, s.request(handle).unwrap().consumed_tokens), (GraphKind::Prefill, 187, 512));
    run_step(&mut s);
    assert_eq!(s.request(handle).unwrap().snapshot_consumed(), 699);
    assert_eq!(s.plan().unwrap().kind, GraphKind::Snapshot);
}

#[test]
fn atomic_spans_filter_capture_cuts_without_discarding_message_marks() {
    let mut s = scheduler(2, false);
    let mut request = spec(None, 700, 3);
    request.message_boundaries = vec![300, 450];
    let handle = s.admit(request).unwrap();
    s.set_prefill_spans(handle, &[400..500]).unwrap();
    assert_eq!(s.request(handle).unwrap().message_boundaries, [300, 450]);
    stop_at_capture(&mut s, handle, 300);
}

fn scheduler(snapshots: u32, mtp: bool) -> RequestScheduler {
    scheduler_with_pages(snapshots, mtp, 64)
}

fn scheduler_with_pages(snapshots: u32, mtp: bool, page_tokens: usize) -> RequestScheduler {
    scheduler_with_concurrency(snapshots, mtp, page_tokens, 4)
}

fn scheduler_with_concurrency(snapshots: u32, mtp: bool, page_tokens: usize, active: usize) -> RequestScheduler {
    let mut s = RequestScheduler::new(
        [9; 32],
        SchedulerLimits {
            request_slots: 4,
            max_active: active,
            max_sequence_tokens: 8192,
            prefill_chunk_size: 512,
            pages: 256,
            snapshots,
        },
        SchedulerCapabilities {
            completion_token_capacity: 16,
            vocabulary: 248320,
            page_tokens,
            vision: crate::model::VisionCapabilities {
                feature_slots: 4,
                features_per_request: 16384,
                max_images: 4,
                patches_per_feature: 4,
                patch_capacities: vec![256, 1024, 4096, 16384],
            },
            draft_pages_per_request: 1,
            completion_inputs: vec![],
            speculative_allocations: vec![],
            state_rows_per_request: None,
        },
    )
    .unwrap();
    s.measured_graph_durations(&super::tests::duration_graphs(4))
        .unwrap();
    if mtp {
        s.enable_speculation(&[1, 2, 4, 8]).unwrap();
    }
    s
}
fn spec(system: Option<usize>, prompt: usize, output: u32) -> RequestSpec {
    RequestSpec {
        message_boundaries: system.into_iter().collect(),
        prompt: (0..prompt as u32).collect(),
        conditioning: vec![],
        max_new_tokens: output,
        grammar_constrained: false,
        host_stops: false,
        prompt_logprobs: false,
        growable_output: false,
    }
}
fn run_step(s: &mut RequestScheduler) -> ScheduledPlan {
    let step = s.plan().unwrap();
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&step, fence).unwrap();
    let mut records = vec![CompletionRecord::default(); s.limits.request_capacity()];
    for lane in &step.lanes {
        let r = s.request(lane.request).unwrap();
        let end = lane.start_position + lane.tokens as u64;
        let emitted = match step.kind {
            GraphKind::Prefill => usize::from(end == r.prompt_tokens as u64),
            GraphKind::Decode | GraphKind::Resume => 1,
            GraphKind::Speculative => step
                .token_capacity
                .min((r.max_new_tokens - r.generated_tokens) as usize),
            _ => 0,
        };
        let consumed = if step.kind == GraphKind::Speculative {
            r.consumed_tokens + emitted as u64
        } else {
            end
        };
        let generated = r.generated_tokens + emitted as u32;
        let token = if emitted > 0 { 900 } else { -1 };
        let mut record = CompletionRecord {
            generation: lane.request.generation as i64,
            submission_sequence: fence.0 as i64,
            consumed_tokens: consumed as i64,
            generated_tokens: generated as i64,
            token_id: token,
            status: if emitted > 0 && generated == r.max_new_tokens {
                2
            } else {
                1
            },
            ..Default::default()
        };
        if step.kind == GraphKind::Speculative {
            let mut vector = crate::execution::TokenVector {
                emitted_count: emitted as i64,
                offered_drafts: (step.token_capacity - 1) as i64,
                accepted_drafts: emitted.min(step.token_capacity - 1) as i64,
                verification_width: step.token_capacity as i64,
                draft_cache_consumed: consumed as i64 - 1,
                ..Default::default()
            };
            vector.tokens[..emitted].fill(token);
            record.vector = Some(vector);
        }
        records[lane.request.slot as usize] = record;
    }
    s.complete(fence, &records, Default::default()).unwrap();
    step
}
fn stop_at_capture(s: &mut RequestScheduler, handle: RequestHandle, endpoint: u64) {
    for _ in 0..200 {
        let r = s.request(handle).unwrap();
        if r.phase == RequestPhase::Snapshot && r.snapshot_consumed() == endpoint {
            return;
        }
        run_step(s);
    }
    panic!("endpoint not scheduled");
}
fn finish(s: &mut RequestScheduler, h: RequestHandle) -> CheckpointId {
    for _ in 0..250 {
        if s.request(h).unwrap().phase == RequestPhase::Finishing {
            return s.finish(h).unwrap().unwrap();
        }
        run_step(s);
    }
    panic!("request failed to finish");
}

fn linear_checkpoint_costs(s: &mut RequestScheduler, fraction: f64) {
    let mut graphs = super::tests::duration_graphs(s.limits.max_active);
    for graph in &mut graphs {
        if graph.kind == GraphKind::Prefill { graph.estimated_duration_ns = graph.token_capacity as u64 * 1000; }
    }
    s.measured_graph_durations(&graphs).unwrap();
    s.configure_policy(PolicyConfig { checkpoint_extra_work_fraction: fraction, ..Default::default() }).unwrap();
}

#[test]
fn measured_work_can_fund_more_than_eight_checkpoint_cuts() {
    let template = scheduler_with_concurrency(32, false, 64, 1);
    let mut s = RequestScheduler::new([9; 32], SchedulerLimits {
        max_active: 32, request_slots: 32, ..template.limits
    }, template.capabilities.clone()).unwrap();
    linear_checkpoint_costs(&mut s, 0.05);
    let mut request = spec(None, 2048, 3);
    request.message_boundaries = (128..2048).step_by(128).collect();
    let handle = s.admit(request).unwrap();
    finish(&mut s, handle);
    assert_eq!(s.stats().message_checkpoints, 15);
    assert_eq!(s.stats().prompt_checkpoints, 1);
    assert!(s.request(handle).is_err());
    assert!(s.stats().checkpoint_extra_work_ns <= 2048 * 1000 / 20);
}

#[test]
fn snapshot_capacity_does_not_disable_prompt_fallback_anchor() {
    for snapshots in [0, 1, 2, 4, 8, 12] {
        for output in [1, 3] {
            let mut s = scheduler(snapshots, false);
            let handle = s.admit(spec(None, 700, output)).unwrap();
            for _ in 0..50 {
                if s.request(handle).unwrap().phase == RequestPhase::Finishing { break; }
                run_step(&mut s);
            }
            assert_eq!(s.request(handle).unwrap().phase, RequestPhase::Finishing);
            assert_eq!(s.request(handle).unwrap().generated_tokens, output);
            s.finish(handle).unwrap();
            assert_eq!(s.stats().checkpoint_anchor_captures, u64::from(snapshots > 0));
            assert_eq!(s.stats().prompt_checkpoints, u64::from(snapshots > 0));
            assert!(s.stats().cached_checkpoints <= snapshots as usize);
        }
    }
}

#[test]
fn duplicate_finished_boundary_needs_no_terminal_snapshot() {
    let mut s = scheduler(2, false);
    let a = s.admit(spec(Some(70), 70, 1)).unwrap();
    let b = s.admit(spec(Some(70), 70, 1)).unwrap();
    let mut pending = vec![a, b];
    for _ in 0..20 {
        run_step(&mut s);
        let finished: Vec<_> = pending
            .iter()
            .copied()
            .filter(|&h| s.request(h).unwrap().phase == RequestPhase::Finishing)
            .collect();
        for h in finished {
            s.finish(h).unwrap();
            pending.retain(|&other| other != h);
        }
        if pending.is_empty() {
            break;
        }
    }
    assert!(pending.is_empty());
    assert_eq!(s.stats().snapshot_publications, 1);
    assert_eq!(s.free_snapshot_count(), 1);
    assert_eq!(s.free_page_count(), 255);
}

#[test]
fn requests_finish_when_no_checkpoint_storage_is_available() {
    let mut s = scheduler(0, false);
    let handle = s.admit(spec(Some(10), 70, 3)).unwrap();
    for _ in 0..20 {
        if s.request(handle).unwrap().phase == RequestPhase::Finishing {
            break;
        }
        assert_ne!(run_step(&mut s).kind, GraphKind::Snapshot);
    }
    assert_eq!(s.request(handle).unwrap().generated_tokens, 3);
    assert_eq!(s.finish(handle).unwrap(), None);
    assert_eq!(s.free_page_count(), 256);
}

#[test]
fn exact_system_prompt_terminal_endpoints_and_partial_restore_pages() {
    for page_tokens in [64, 256] {
        for system in [1, 2, 3, 10, 15, 63, 64, 65, 66, 67, 127, 255, 256, 257, 511] {
            for mtp in [false, true] {
                let mut s = scheduler_with_concurrency(8, mtp, page_tokens, 2);
                let prompt = system + 67;
                let h = s.admit(spec(Some(system), prompt, 3)).unwrap();
                stop_at_capture(&mut s, h, system as u64);
                let prior = s.request(h).unwrap().pages().collect::<Vec<_>>();
                assert_eq!(s.request(h).unwrap().generated_tokens, 0);
                assert_eq!(
                    s.request(h).unwrap().snapshot_destination(),
                    s.request(h).unwrap().message_snapshot.unwrap()
                );
                run_step(&mut s);
                let r = s.request(h).unwrap();
                assert_eq!(r.prefix_tokens, system as u64);
                assert_eq!(r.shared_pages.len(), system / page_tokens);
                assert_eq!(r.pages().collect::<Vec<_>>(), prior);
                assert_eq!(r.reused_prompt_tokens(), 0);
                let continuation = s.plan().unwrap();
                assert_eq!(continuation.kind, GraphKind::Prefill);
                assert_eq!(continuation.lanes[0].start_position, system as u64);
                assert_eq!(continuation.lanes[0].tokens, 67);
                stop_at_capture(&mut s, h, prompt as u64);
                run_step(&mut s);
                let terminal = finish(&mut s, h);
                assert_eq!(
                    s.cache
                        .checkpoint(terminal)
                        .unwrap()
                        .prefix()
                        .consumed_tokens,
                    (prompt + 2) as u64
                );
                assert_eq!(
                    s.cache.page_chain(terminal).unwrap().len(),
                    (prompt + 2) / page_tokens
                );
                assert_eq!(
                    s.cache.lookup(&history(system)).unwrap().consumed_tokens,
                    system as u64
                );
                assert_eq!(
                    s.cache.lookup(&history(prompt)).unwrap().consumed_tokens,
                    prompt as u64
                );
                let mut branch = spec(Some(system), prompt, 3);
                branch.prompt[system] = 777;
                let a = s.admit(branch.clone()).unwrap();
                let b = s.admit(branch).unwrap();
                assert_eq!(s.request(a).unwrap().consumed_tokens, system as u64);
                run_step(&mut s);
                let a = s.request(a).unwrap();
                let b = s.request(b).unwrap();
                assert_eq!(a.phase, RequestPhase::Prefill);
                assert_eq!(a.shared_pages, b.shared_pages);
                if system % page_tokens != 0 || mtp {
                    assert_ne!(a.private_pages[0], b.private_pages[0]);
                }
                assert_eq!(
                    a.pages().count(),
                    system.div_ceil(page_tokens) + usize::from(mtp)
                );
                if mtp && system % page_tokens >= 2 {
                    let target_full = system / page_tokens;
                    let a_pages = a.pages().collect::<Vec<_>>();
                    let b_pages = b.pages().collect::<Vec<_>>();
                    assert_ne!(a_pages[target_full], b_pages[target_full]);
                    assert_ne!(a_pages[target_full + 1], b_pages[target_full + 1]);
                }
            }
        }
    }
}

#[test]
fn prompt_6459_retains_exact_prompt_and_terminal_with_immutable_admission_reuse() {
    let mut s = scheduler(8, false);
    let h = s.admit(spec(Some(10), 6459, 144)).unwrap();
    let terminal = finish(&mut s, h);
    assert_eq!(
        s.cache
            .checkpoint(terminal)
            .unwrap()
            .prefix()
            .consumed_tokens,
        6602
    );
    assert_eq!(
        s.cache.lookup(&history(6459)).unwrap().consumed_tokens,
        6459
    );
    let h = s.admit(spec(Some(10), 6459, 144)).unwrap();
    assert_eq!(s.request(h).unwrap().reused_prompt_tokens(), 6459);
    assert_eq!(run_step(&mut s).kind, GraphKind::Restore);
    assert_eq!(s.plan().unwrap().kind, GraphKind::Resume);
    run_step(&mut s);
    assert_eq!(s.request(h).unwrap().reused_prompt_tokens(), 6459);
}

#[test]
fn boundary_snapshots_are_optional_and_cancel_behind_fence() {
    let mut s = scheduler(2, false);
    let h = s.admit(spec(Some(10), 70, 3)).unwrap();
    assert_eq!(s.free_snapshot_count(), 2);
    assert_eq!(s.stats().active_requests, 1);
    finish(&mut s, h);
    let mut s = scheduler_with_concurrency(3, false, 64, 1);
    let h = s.admit(spec(Some(10), 70, 3)).unwrap();
    assert_eq!(s.free_snapshot_count(), 3);
    stop_at_capture(&mut s, h, 10);
    let step = s.plan().unwrap();
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&step, fence).unwrap();
    s.cancel(h).unwrap();
    assert_eq!(s.free_snapshot_count(), 2);
    let mut record = vec![CompletionRecord::default(); 4];
    record[h.slot as usize] = CompletionRecord {
        generation: h.generation as i64,
        submission_sequence: fence.0 as i64,
        consumed_tokens: 10,
        token_id: -1,
        status: 1,
        ..Default::default()
    };
    s.complete(fence, &record, Default::default()).unwrap();
    assert_eq!(s.free_snapshot_count(), 3);
    assert_eq!(s.free_page_count(), 256);
    assert_eq!(s.cache.stats().published_checkpoints, 0);
}

#[test]
fn duplicate_boundary_publication_does_not_remap_live_partial_pages() {
    let mut s = scheduler_with_concurrency(6, false, 64, 2);
    let a = s.admit(spec(Some(10), 70, 2)).unwrap();
    let b = s.admit(spec(Some(10), 70, 2)).unwrap();
    stop_at_capture(&mut s, a, 10);
    run_step(&mut s);
    stop_at_capture(&mut s, b, 10);
    let pages = s.request(b).unwrap().pages().collect::<Vec<_>>();
    run_step(&mut s);
    assert_eq!(s.request(b).unwrap().pages().collect::<Vec<_>>(), pages);
    assert_eq!(s.request(b).unwrap().prefix_tokens, 0);
    assert!(s.request(a).unwrap().prefix_tokens >= 10);
    assert_eq!(s.cache.lookup(&history(10)).unwrap().consumed_tokens, 10);
    assert!(s.request(b).unwrap().message_snapshot.is_none());
    assert_eq!(s.free_snapshot_count(), 3);
}

#[test]
fn failed_boundary_copy_is_not_published_and_all_reservations_retire() {
    let mut s = scheduler_with_concurrency(3, false, 64, 1);
    let h = s.admit(spec(Some(3), 70, 3)).unwrap();
    stop_at_capture(&mut s, h, 3);
    let step = s.plan().unwrap();
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&step, fence).unwrap();
    let mut records = vec![CompletionRecord::default(); 4];
    records[h.slot as usize] = CompletionRecord {
        generation: h.generation as i64,
        submission_sequence: fence.0 as i64,
        consumed_tokens: 3,
        token_id: -1,
        status: 3,
        error_code: 9,
        ..Default::default()
    };
    s.complete(fence, &records, Default::default()).unwrap();
    assert_eq!(s.finish(h).unwrap(), None);
    assert_eq!(s.free_snapshot_count(), 3);
    assert_eq!(s.free_page_count(), 256);
}

#[test]
fn identical_prompt_and_terminal_endpoint_requires_one_reservation() {
    let mut s = scheduler(1, false);
    let h = s.admit(spec(None, 3, 1)).unwrap();
    let id = finish(&mut s, h);
    assert_eq!(s.cache.checkpoint(id).unwrap().prefix().consumed_tokens, 3);
    assert!(s.cache.page_chain(id).unwrap().is_empty());
}

fn history(n: usize) -> PrefixHistory {
    let mut h = PrefixHistory::new([9; 32]);
    h.append(&(0..n as u32).collect::<Vec<_>>(), &[]).unwrap();
    h
}

#[test]
fn runtime_concurrency_and_loaded_width_assign_fixed_execution_rows() {
    let template = scheduler(256, false);
    for active in [1usize, 4, 32] {
        for width in [1usize, 3, 6] {
            let mut s = RequestScheduler::new(
                [9; 32],
                SchedulerLimits {
                    max_active: active,
                    request_slots: 32,
                    ..template.limits
                },
                template.capabilities.clone(),
            )
            .unwrap();
            s.enable_speculation(&[1, 2, 3, 4, 5, 6]).unwrap();
            let shapes: Vec<_> = crate::execution::BATCH_CAPACITIES
                .into_iter()
                .filter(|batch| *batch <= active.next_power_of_two())
                .flat_map(|batch| (1..=width).map(move |width| (batch, width)))
                .collect();
            s.speculative_graphs(&shapes).unwrap();
            let mut handles = Vec::new();
            let mut owned = BTreeSet::new();
            for _ in 0..active {
                let handle = s.admit(spec(None, 64, 1)).unwrap();
                let request = s.request(handle).unwrap();
                assert_eq!(
                    request.state_slot.0 as usize,
                    1 + handle.slot as usize * width
                );
                for column in 0..width {
                    assert!(owned.insert(request.state_slot.0 + column as u32));
                }
                assert!(request.terminal_snapshot.is_none());
                handles.push(handle);
            }
            assert_eq!(s.free_snapshot_count(), 256);
            for handle in handles {
                s.cancel(handle).unwrap();
            }
            assert_eq!(s.free_snapshot_count(), 256);
        }
    }
}

#[test]
fn atomic_prefill_spans_keep_bidirectional_inputs_together() {
    let mut s = scheduler(24, false);
    let h = s.admit(spec(None, 1537, 1)).unwrap();
    s.set_prefill_spans(h, &[400..780, 810..1190]).unwrap();
    let mut chunks = Vec::new();
    while s.request(h).unwrap().phase != RequestPhase::Finishing {
        let step = run_step(&mut s);
        if step.kind == GraphKind::Prefill {
            let lane = &step.lanes[0];
            chunks.push((lane.start_position, lane.tokens));
        }
    }
    assert_eq!(chunks, [(0, 400), (400, 410), (810, 512), (1322, 215)]);
    s.finish(h).unwrap();
    let reused = s.admit(spec(None, 1537, 1)).unwrap();
    s.set_prefill_spans(reused, &[400..780, 810..1190]).unwrap();
    assert!(s.request(reused).unwrap().reused_prompt_tokens() >= 1537);
}

#[test]
fn atomic_prefill_spans_reject_invalid_geometry_before_execution() {
    for spans in [
        vec![10..10],
        vec![20..19],
        vec![0..513],
        vec![900..1100],
        vec![20..100, 90..110],
        vec![200..300, 10..20],
    ] {
        let mut s = scheduler(8, false);
        let h = s.admit(spec(None, 1024, 1)).unwrap();
        assert!(s.set_prefill_spans(h, &spans).is_err());
        assert_eq!(s.plan().unwrap().kind, GraphKind::Init);
    }
    let mut s = scheduler(8, false);
    let h = s.admit(spec(Some(512), 1024, 1)).unwrap();
    s.set_prefill_spans(h, &[400..700]).unwrap();
    run_step(&mut s);
    assert!(s.set_prefill_spans(h, &[800..900]).is_err());
}

#[test]
fn empty_atomic_spans_preserve_text_chunking() {
    let mut s = scheduler(16, false);
    let h = s.admit(spec(None, 1100, 1)).unwrap();
    s.set_prefill_spans(h, &[]).unwrap();
    let mut chunks = Vec::new();
    while s.request(h).unwrap().phase != RequestPhase::Finishing {
        let step = run_step(&mut s);
        if step.kind == GraphKind::Prefill {
            chunks.push(step.lanes[0].tokens);
        }
    }
    assert_eq!(chunks, [512, 512, 76]);
}

#[test]
fn snapshot_pressure_retains_complete_conversations_until_endpoint_eviction() {
    for capacity in [2usize, 3, 16, 17] {
        let mut s = scheduler_with_concurrency(capacity as u32, false, 64, 1);
        let turns = capacity + 3;
        let mut lengths = Vec::new();
        for turn in 0..turns {
            let length = 160 + 8 * turn;
            let mut request = spec(Some(length - 10), length, 1);
            request.prompt[0] = (turn + 1) as u32;
            let handle = s.admit(request).unwrap();
            while s.request(handle).unwrap().phase != RequestPhase::Finishing {
                run_step(&mut s);
            }
            s.finish(handle).unwrap();
            lengths.push(length);
        }
        assert!(s.stats().cached_checkpoints <= capacity);
        let latest = *lengths.last().unwrap();
        let mut latest_history = PrefixHistory::new([9; 32]);
        let mut tokens: Vec<_> = (0..latest as u32).collect();
        tokens[0] = turns as u32;
        latest_history.append(&tokens, &[]).unwrap();
        assert!(s.cache.lookup(&latest_history).unwrap().consumed_tokens >= (latest - 10) as u64);
        assert!(s.stats().snapshot_publications >= turns as u64);
    }
}

#[test]
fn every_message_endpoint_can_restore_after_a_changed_following_turn() {
    let endpoints = vec![1, 7, 63, 64, 65, 129, 300, 514, 700];
    for page_tokens in [64, 128, 256, 2048] {
        for mtp in [false, true] {
            for endpoints in endpoints.chunks(6) {
                let mut s = scheduler_with_concurrency(16, mtp, page_tokens, 1);
                linear_checkpoint_costs(&mut s, 1.0);
                let mut request = spec(None, 901, 3);
                request.message_boundaries = endpoints.to_vec();
                let handle = s.admit(request).unwrap();
                for &end in endpoints {
                    stop_at_capture(&mut s, handle, end as u64);
                    assert_eq!(s.request(handle).unwrap().generated_tokens, 0);
                    run_step(&mut s);
                    assert_eq!(s.request(handle).unwrap().prefix_tokens, end as u64);
                }
                finish(&mut s, handle);
                assert_eq!(s.stats().message_checkpoints, endpoints.len() as u64);
                for &end in endpoints {
                    let mut branch = spec(None, end + 10, 1);
                    branch.prompt[end..].fill(1000);
                    branch.message_boundaries = endpoints.iter().copied().filter(|&n| n <= end).collect();
                    let branch = s.admit(branch).unwrap();
                    assert_eq!(s.request(branch).unwrap().reused_prompt_tokens(), end as u64);
                    assert_eq!(run_step(&mut s).kind, GraphKind::Restore);
                    let step = s.plan().unwrap();
                    assert_eq!(step.kind, GraphKind::Prefill);
                    assert_eq!(step.lanes[0].start_position, end as u64);
                    assert_eq!(step.lanes[0].tokens, 10);
                    s.cancel(branch).unwrap();
                }
            }
        }
    }
}

#[test]
fn message_checkpoint_churn_preserves_idle_conversation_endpoints() {
    for page_tokens in [64, 128, 256, 2048] {
        let mut s = scheduler_with_pages(256, false, page_tokens);
        let mut conversations = Vec::new();
        for conversation in 0..4 {
            let mut request = spec(None, 1024, 1);
            request.prompt[1..].fill(1000 + conversation);
            request.message_boundaries = (1..1024).step_by(4).collect();
            conversations.push(request.clone());
            let handle = s.admit(request).unwrap();
            for _ in 0..600 {
                if s.request(handle).unwrap().phase == RequestPhase::Finishing {
                    break;
                }
                run_step(&mut s);
            }
            assert_eq!(s.request(handle).unwrap().phase, RequestPhase::Finishing);
            s.finish(handle).unwrap();
        }
        assert_eq!(s.stats().snapshot_evictions, 0);
        assert!(s.stats().snapshot_publications <= 256);
        assert!(s.free_page_count() > 128);
        for request in conversations {
            let handle = s.admit(request).unwrap();
            assert_eq!(s.request(handle).unwrap().reused_prompt_tokens(), 1024);
            s.cancel(handle).unwrap();
        }
    }
}

#[test]
fn branch_saturation_reclaims_old_endpoints_without_stalling() {
    let mut s = scheduler(256, false);
    let mut idle = spec(None, 1024, 2);
    idle.prompt[1..].fill(1000);
    let handle = s.admit(idle.clone()).unwrap();
    finish(&mut s, handle);
    for turn in 0..180 {
        let mut request = spec(None, 1024 + turn * 8, 2);
        request.prompt[1..].fill(1001);
        let handle = s.admit(request).unwrap();
        finish(&mut s, handle);
    }
    assert!(s.free_page_count() > 128);
    let handle = s.admit(idle).unwrap();
    assert_eq!(s.request(handle).unwrap().reused_prompt_tokens(), 0);
    s.cancel(handle).unwrap();
    let mut recent = spec(None, 1024 + 179 * 8, 2);
    recent.prompt[1..].fill(1001);
    let expected = recent.prompt.len() as u64;
    let handle = s.admit(recent).unwrap();
    assert_eq!(s.request(handle).unwrap().reused_prompt_tokens(), expected);
    s.cancel(handle).unwrap();
}

#[test]
fn many_message_boundaries_make_progress_with_bounded_snapshot_storage() {
    let mut s = scheduler(2, false);
    let mut request = spec(None, 100, 3);
    request.message_boundaries = (1..100).collect();
    let handle = s.admit(request).unwrap();
    for _ in 0..250 {
        if s.request(handle).unwrap().phase == RequestPhase::Finishing {
            break;
        }
        run_step(&mut s);
    }
    assert_eq!(s.request(handle).unwrap().phase, RequestPhase::Finishing);
    assert_eq!(s.request(handle).unwrap().generated_tokens, 3);
    s.finish(handle).unwrap();
    assert!(s.stats().cached_checkpoints <= 2);
}

#[test]
fn message_boundaries_must_be_ordered_unique_and_within_the_prompt() {
    for endpoints in [vec![0], vec![2, 1], vec![1, 1], vec![101]] {
        let mut s = scheduler(4, false);
        let mut request = spec(None, 100, 3);
        request.message_boundaries = endpoints;
        assert!(s.admit(request).is_err());
        assert_eq!(s.free_page_count(), 256);
    }
}

#[test]
fn completed_message_anchor_is_offered_immediately_near_prompt() {
    for gap in [1, 63, 64] {
        for output in [1, 3] {
            let mut s = scheduler_with_concurrency(16, false, 64, 1);
            s.enable_publications();
            let message = 191;
            let handle = s.admit(spec(Some(message), message + gap, output)).unwrap();
            stop_at_capture(&mut s, handle, message as u64);
            run_step(&mut s);
            let message_id = s.request(handle).unwrap().prefix.unwrap();
            let offered = s.take_publications();
            assert_eq!(offered.len(), 1);
            assert_eq!(offered[0].id, message_id);
            assert_eq!(offered[0].key.consumed_tokens, message as u64);
            finish(&mut s, handle);
            assert_eq!(s.cache.checkpoint(message_id).unwrap().retention_rank().role, RetentionRole::Anchor);
        }
    }
}

#[test]
fn completed_message_is_offered_once_and_reuse_preserves_exact_endpoint() {
    let mut s = scheduler_with_concurrency(16, false, 64, 1);
    s.enable_publications();
    let handle = s.admit(spec(Some(127), 129, 3)).unwrap();
    stop_at_capture(&mut s, handle, 127);
    run_step(&mut s);
    let message = s.request(handle).unwrap().prefix.unwrap();
    assert_eq!(s.take_publications()[0].id, message);
    finish(&mut s, handle);
    s.take_publications();
    let handle = s.admit(spec(None, 127, 1)).unwrap();
    assert_eq!(s.request(handle).unwrap().prefix, Some(message));
    assert_eq!(s.request(handle).unwrap().reused_prompt_tokens(), 127);
    assert!(s.take_publications().is_empty());
    s.cancel(handle).unwrap();
}

#[test]
fn completed_message_remains_offered_when_prompt_copy_storage_is_unavailable() {
    let mut s = scheduler_with_concurrency(1, false, 64, 1);
    s.enable_publications();
    let handle = s.admit(spec(Some(127), 129, 3)).unwrap();
    stop_at_capture(&mut s, handle, 127);
    run_step(&mut s);
    let message = s.request(handle).unwrap().prefix.unwrap();
    assert_eq!(s.take_publications()[0].id, message);
    assert_eq!(finish(&mut s, handle), message);
    assert_eq!(s.stats().prompt_checkpoints, 0);
    assert!(s.take_publications().is_empty());
}

#[test]
fn supplementary_copy_work_is_spent_only_when_snapshot_is_committed() {
    for submit_copy in [false, true] {
        let mut s = scheduler_with_concurrency(8, false, 64, 1);
        linear_checkpoint_costs(&mut s, 0.05);
        let mut request = spec(None, 1024, 3);
        request.message_boundaries = vec![128, 896];
        let handle = s.admit(request).unwrap();
        stop_at_capture(&mut s, handle, 128);
        assert_eq!(s.request(handle).unwrap().checkpoint_accounting().extra_spent_ns, 0);
        if submit_copy {
            let copy = s.plan().unwrap();
            let fence = Fence(s.submitted.0 + 1);
            s.commit(&copy, fence).unwrap();
            assert_eq!(s.request(handle).unwrap().checkpoint_accounting().extra_spent_ns, 1);
            s.cancel(handle).unwrap();
            assert!(s.free_snapshot_count() < 8);
            let records = vec![CompletionRecord::default(); s.limits.request_capacity()];
            s.complete(fence, &records, Default::default()).unwrap();
        } else { s.cancel(handle).unwrap(); }
        assert_eq!(s.stats().checkpoint_extra_work_ns, u64::from(submit_copy));
        assert_eq!(s.free_snapshot_count(), 8);
    }
}

#[test]
fn stale_supplementary_quote_keeps_cut_and_reserves_anchor_without_copy() {
    let mut s = scheduler_with_concurrency(3, false, 64, 1);
    linear_checkpoint_costs(&mut s, 0.05);
    let mut request = spec(None, 1024, 3);
    request.message_boundaries = vec![128, 896];
    let handle = s.admit(request).unwrap();
    run_step(&mut s);
    let plan = s.plan().unwrap();
    assert_eq!(plan.lanes[0].tokens, 128);
    let held = s.reserve_snapshot(0).unwrap();
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    assert!(s.request(handle).unwrap().checkpoint_capture.is_none());
    assert!(s.request(handle).unwrap().anchor_guard.is_some());
    let mut records = vec![CompletionRecord::default(); s.limits.request_capacity()];
    records[handle.slot as usize] = CompletionRecord {
        generation: handle.generation as i64, submission_sequence: fence.0 as i64,
        consumed_tokens: 128, token_id: -1, status: 1, ..Default::default()
    };
    s.complete(fence, &records, Default::default()).unwrap();
    assert_eq!(s.request(handle).unwrap().consumed_tokens, 128);
    assert_eq!(s.request(handle).unwrap().phase, RequestPhase::Prefill);
    assert_eq!(s.stats().checkpoint_skip_storage, 1);
    assert_eq!(s.stats().checkpoint_extra_work_ns, 0);
    s.release_snapshot(held);
    stop_at_capture(&mut s, handle, 896);
    run_step(&mut s);
    assert_eq!(s.cache.lookup(&history(896)).unwrap().consumed_tokens, 896);
    s.cancel(handle).unwrap();
}

#[test]
fn exhausted_storage_keeps_the_nearest_message_supplementaries() {
    for snapshots in [2, 3] {
        let mut s = scheduler_with_concurrency(snapshots, false, 64, 1);
        linear_checkpoint_costs(&mut s, 0.05);
        let mut request = spec(None, 1024, 3);
        request.message_boundaries = (128..1024).step_by(128).collect();
        let handle = s.admit(request).unwrap();
        finish(&mut s, handle);
        let latest_extra = 768;
        assert_eq!(s.cache.lookup(&history(latest_extra)).unwrap().consumed_tokens, latest_extra as u64);
        assert_eq!(s.cache.lookup(&history(896)).unwrap().consumed_tokens, 896);
        assert_eq!(s.stats().snapshot_evictions, 0);
        assert_eq!(s.stats().message_checkpoints, snapshots as u64);
    }
}
