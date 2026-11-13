/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use crate::execution::ExecutionTiming;

fn scheduler(page_tokens: usize, rows: usize) -> RequestScheduler {
    scheduler_with_active(page_tokens, rows, 1)
}

fn scheduler_with_active(page_tokens: usize, rows: usize, active: usize) -> RequestScheduler {
    scheduler_with_resources(page_tokens, rows, active, 16)
}

fn scheduler_with_resources(
    page_tokens: usize,
    rows: usize,
    active: usize,
    snapshots: u32,
) -> RequestScheduler {
    let mut scheduler = RequestScheduler::new(
        [19; 32],
        SchedulerLimits {
            request_slots: active.next_power_of_two(),
            max_active: active,
            max_sequence_tokens: 65_536,
            prefill_chunk_size: 4096,
            pages: (65_536 / page_tokens * active) as u32,
            snapshots,
        },
        SchedulerCapabilities {
            completion_token_capacity: 16,
            vocabulary: 248_320,
            page_tokens,
            vision: crate::model::VisionCapabilities {
                feature_slots: active,
                features_per_request: 16_384,
                max_images: 4,
                patches_per_feature: 4,
                patch_capacities: vec![256, 1024, 4096, 16_384],
            },
            draft_pages_per_request: 1,
            completion_inputs: vec![],
            speculative_allocations: vec![],
            state_rows_per_request: Some(rows),
        },
    )
    .unwrap();
    scheduler
        .measured_graph_durations(&tests::duration_graphs(active))
        .unwrap();
    scheduler
}

fn inputs(count: usize) -> Vec<ConditionedInput> {
    (0..count)
        .map(|image| ConditionedInput {
            token_span: 17 + image * 4096..17 + (image + 1) * 4096,
            feature_range: image * 4096..(image + 1) * 4096,
            patches: 16_384,
        })
        .collect()
}

fn specification(inputs: &[ConditionedInput]) -> RequestSpec {
    let mut spec = tests::spec(inputs.last().unwrap().token_span.end + 23, 1);
    spec.conditioning = inputs
        .iter()
        .enumerate()
        .map(|(image, input)| ConditioningEvent {
            position: input.token_span.start as u64,
            identity: [image as u8; 32],
        })
        .collect();
    spec
}

fn admit(scheduler: &mut RequestScheduler, inputs: &[ConditionedInput]) -> RequestHandle {
    let handle = scheduler.admit(specification(inputs)).unwrap();
    scheduler.set_conditioned_inputs(handle, inputs).unwrap();
    handle
}

fn commit_step(
    scheduler: &mut RequestScheduler,
    handle: RequestHandle,
) -> (ScheduledPlan, Fence, Vec<CompletionRecord>) {
    if scheduler.request(handle).unwrap().phase == RequestPhase::Vision {
        scheduler.conditioned_inputs_ready(handle, true).unwrap();
    }
    let step = scheduler
        .plan()
        .expect("conditioned request has executable work");
    let fence = Fence(scheduler.submitted.0 + 1);
    scheduler.commit(&step, fence).unwrap();
    let mut records = vec![CompletionRecord::default(); scheduler.limits.request_slots];
    for lane in &step.lanes {
        let request = scheduler.request(lane.request).unwrap();
        let prefill_end = lane.start_position + lane.tokens as u64;
        let emitted = usize::from(
            (step.kind == GraphKind::Prefill && prefill_end == request.prompt_tokens as u64)
                || matches!(step.kind, GraphKind::Decode | GraphKind::Resume),
        );
        records[lane.request.slot as usize] = CompletionRecord {
            generation: lane.request.generation as i64,
            submission_sequence: fence.0 as i64,
            consumed_tokens: match step.kind {
                GraphKind::Prefill => prefill_end,
                GraphKind::Decode => request.consumed_tokens + 1,
                _ => request.consumed_tokens,
            } as i64,
            generated_tokens: request.generated_tokens as i64 + emitted as i64,
            token_id: if emitted == 1 { 900 } else { -1 },
            status: if emitted == 1 && request.generated_tokens + 1 == request.max_new_tokens {
                2
            } else {
                1
            },
            ..Default::default()
        };
    }
    (step, fence, records)
}

fn step(scheduler: &mut RequestScheduler, handle: RequestHandle) -> ScheduledPlan {
    let (step, fence, records) = commit_step(scheduler, handle);
    scheduler
        .advance_to(scheduler.now_ns + step.estimated_duration_ns())
        .unwrap();
    let mut timing = ExecutionTiming::default();
    timing.gpu_ns[0] = Some(step.estimated_duration_ns());
    timing.wall_ns = step.estimated_duration_ns();
    let generated = scheduler.complete(fence, &records, timing).unwrap();
    for token in generated {
        scheduler.published_output(token.request, fence).unwrap();
    }
    step
}

fn advance_to_window_snapshot(scheduler: &mut RequestScheduler, handle: RequestHandle) {
    for _ in 0..100 {
        let request = scheduler.request(handle).unwrap();
        if request.phase == RequestPhase::Snapshot
            && request.consumed_tokens < request.prompt_tokens as u64
        {
            return;
        }
        step(scheduler, handle);
    }
    panic!("private window checkpoint was not reached");
}

#[test]
fn cold_history_exceeding_two_arenas_keeps_pages_state_and_user_reuse_across_windows() {
    for (page_tokens, rows) in [(64, 1), (128, 4), (1024, 8)] {
        let mut scheduler = scheduler(page_tokens, rows);
        let images = inputs(12);
        let mut handle = admit(&mut scheduler, &images);
        let state = scheduler.request(handle).unwrap().state_slot;
        let image_slot = scheduler.request(handle).unwrap().image_slot;
        let initial_lookups = scheduler.stats().cache_lookups;
        let mut renewals = 0;
        let mut vision_jobs = 0;
        let mut sampled = 0;
        let mut published_prompt = false;
        for _ in 0..200 {
            let request = scheduler.request(handle).unwrap();
            if request.phase == RequestPhase::Finishing {
                break;
            }
            if request.conditioned_window_ready() {
                let old_handle = handle;
                let consumed = request.consumed_tokens;
                assert_ne!(
                    consumed % page_tokens as u64,
                    0,
                    "checkpoint must exercise a partial page"
                );
                let pages: Vec<_> = request.pages().collect();
                let reserved = request.reserved_pages.clone();
                let key = request.history.key_at(consumed).unwrap();
                let snapshot = request.window_snapshot.unwrap();
                let free = scheduler.free_snapshot_count();
                assert!(
                    scheduler.plan().is_none(),
                    "private checkpoint waits for explicit epoch renewal"
                );
                handle = scheduler.resume_conditioned_window(handle).unwrap();
                renewals += 1;
                assert_eq!(handle.slot, old_handle.slot);
                assert_eq!(handle.generation, old_handle.generation + 1);
                assert!(scheduler.request(old_handle).is_err());
                assert!(
                    scheduler
                        .conditioned_inputs_ready(old_handle, true)
                        .is_err()
                );
                let request = scheduler.request(handle).unwrap();
                assert_eq!(request.state_slot, state);
                assert_eq!(request.image_slot, image_slot);
                assert_eq!(request.pages().collect::<Vec<_>>(), pages);
                assert_eq!(request.reserved_pages, reserved);
                assert_eq!(request.history.key_at(consumed).unwrap(), key);
                assert_eq!(request.prefix_snapshot(), Some(snapshot));
                assert_eq!(request.generated_tokens, 0);
                assert_eq!(scheduler.free_snapshot_count(), free);
                assert_eq!(
                    request.feature_index((renewals * 16_384) as i32).unwrap(),
                    0
                );
                assert!(
                    request
                        .feature_index(((renewals - 1) * 16_384) as i32)
                        .is_err()
                );
            }
            let before = scheduler.request(handle).unwrap().generated_tokens;
            let executed = step(&mut scheduler, handle);
            vision_jobs += usize::from(executed.kind == GraphKind::Vision);
            sampled += (scheduler.request(handle).unwrap().generated_tokens - before) as usize;
            assert_eq!(scheduler.stats().cache_lookups, initial_lookups);
            assert_eq!(scheduler.stats().reused_tokens, 0);
            let request = scheduler.request(handle).unwrap();
            if executed.kind == GraphKind::Snapshot
                && request.consumed_tokens == request.prompt_tokens as u64
            {
                published_prompt = true;
            }
            assert_eq!(
                scheduler.stats().snapshot_publications,
                u64::from(published_prompt)
            );
        }
        assert_eq!(renewals, 2);
        assert_eq!(vision_jobs, images.len());
        assert_eq!(
            sampled, 1,
            "window endpoints cannot produce an output token"
        );
        assert_eq!(
            scheduler.request(handle).unwrap().phase,
            RequestPhase::Finishing
        );
        assert_eq!(scheduler.request(handle).unwrap().window_snapshot, None);
        assert!(published_prompt);
        assert_eq!(scheduler.stats().snapshot_publications, 1);
        assert!(scheduler.finish(handle).unwrap().is_some());
        assert_eq!(scheduler.stats().snapshot_publications, 1);
    }
}

#[test]
fn private_snapshot_cancellation_retires_its_row_pages_and_arena_only_after_the_fence() {
    let mut scheduler = scheduler(64, 4);
    let initial_snapshots = scheduler.free_snapshot_count();
    let initial_pages = scheduler.free_page_count();
    let handle = admit(&mut scheduler, &inputs(5));
    advance_to_window_snapshot(&mut scheduler, handle);
    let (_, fence, records) = commit_step(&mut scheduler, handle);
    let retained = scheduler.free_snapshot_count();
    assert!(retained < initial_snapshots);
    assert!(scheduler.resume_conditioned_window(handle).is_err());
    scheduler.cancel(handle).unwrap();
    assert_eq!(scheduler.free_snapshot_count(), retained);
    assert_eq!(scheduler.free_image_slots.len(), 0);
    assert_eq!(scheduler.stats().retiring_slots, 1);
    scheduler
        .complete(fence, &records, Default::default())
        .unwrap();
    assert_eq!(scheduler.free_snapshot_count(), initial_snapshots);
    assert_eq!(scheduler.free_page_count(), initial_pages);
    assert_eq!(scheduler.free_image_slots.len(), 1);
    assert_eq!(scheduler.stats().retiring_slots, 0);
    assert_eq!(scheduler.stats().snapshot_publications, 0);
}

#[test]
fn restore_rejects_a_completion_from_the_previous_execution_epoch() {
    let mut scheduler = scheduler(64, 1);
    let handle = admit(&mut scheduler, &inputs(5));
    advance_to_window_snapshot(&mut scheduler, handle);
    step(&mut scheduler, handle);
    let next = scheduler.resume_conditioned_window(handle).unwrap();
    let (executed, fence, mut records) = commit_step(&mut scheduler, next);
    assert_eq!(executed.kind, GraphKind::Restore);
    records[next.slot as usize].generation = handle.generation as i64;
    assert!(
        scheduler
            .complete(fence, &records, Default::default())
            .is_err()
    );
    assert_eq!(
        scheduler.request(next).unwrap().phase,
        RequestPhase::Restore
    );
    records[next.slot as usize].generation = next.generation as i64;
    scheduler
        .complete(fence, &records, Default::default())
        .unwrap();
    assert_eq!(scheduler.request(next).unwrap().phase, RequestPhase::Vision);
}

#[test]
fn cancellation_during_private_restore_keeps_the_snapshot_until_its_fence() {
    let mut scheduler = scheduler(128, 4);
    let free_snapshots = scheduler.free_snapshot_count();
    let free_pages = scheduler.free_page_count();
    let handle = admit(&mut scheduler, &inputs(5));
    advance_to_window_snapshot(&mut scheduler, handle);
    step(&mut scheduler, handle);
    let next = scheduler.resume_conditioned_window(handle).unwrap();
    let (_, fence, records) = commit_step(&mut scheduler, next);
    let retained = scheduler.free_snapshot_count();
    assert!(scheduler.cancel(handle).is_err());
    assert!(scheduler.request(next).is_ok());
    scheduler.cancel(next).unwrap();
    assert_eq!(scheduler.free_snapshot_count(), retained);
    assert_eq!(scheduler.free_image_slots.len(), 0);
    scheduler
        .complete(fence, &records, Default::default())
        .unwrap();
    assert_eq!(scheduler.free_snapshot_count(), free_snapshots);
    assert_eq!(scheduler.free_page_count(), free_pages);
    assert_eq!(scheduler.free_image_slots.len(), 1);
    assert_eq!(scheduler.stats().snapshot_publications, 0);
}

#[test]
fn window_boundaries_keep_bidirectional_image_spans_inside_one_prefill() {
    let mut scheduler = scheduler(64, 1);
    let mut images = inputs(9);
    for (index, input) in images.iter_mut().enumerate() {
        input.token_span.start += 5 * index;
        input.token_span.end += 5 * index;
    }
    let mut handle = admit(&mut scheduler, &images);
    let spans: Vec<_> = images
        .iter()
        .map(|input| input.token_span.clone())
        .collect();
    scheduler.set_prefill_spans(handle, &spans).unwrap();
    let mut renewals = 0;
    for _ in 0..200 {
        if scheduler.request(handle).unwrap().phase == RequestPhase::Finishing {
            break;
        }
        if scheduler
            .request(handle)
            .unwrap()
            .conditioned_window_ready()
        {
            let consumed = scheduler.request(handle).unwrap().consumed_tokens as usize;
            assert!(
                spans
                    .iter()
                    .all(|span| !(span.start < consumed && consumed < span.end))
            );
            handle = scheduler.resume_conditioned_window(handle).unwrap();
            renewals += 1;
        }
        let executed = step(&mut scheduler, handle);
        if executed.kind == GraphKind::Prefill {
            let lane = &executed.lanes[0];
            let start = lane.start_position as usize;
            let end = start + lane.tokens;
            for span in spans
                .iter()
                .filter(|span| span.start < end && span.end > start)
            {
                assert!(
                    start <= span.start && span.end <= end,
                    "prefill {start}..{end} splits image {span:?}"
                );
            }
        }
    }
    assert_eq!(renewals, 2);
    assert_eq!(
        scheduler.request(handle).unwrap().phase,
        RequestPhase::Finishing
    );
}

#[test]
fn partial_image_prefix_reuse_preserves_canonical_indices_and_only_counts_initial_reuse() {
    let mut scheduler = scheduler(64, 4);
    let images = inputs(12);
    let endpoint = images[4].token_span.start + 512;
    let mut specification = specification(&images);
    specification.message_boundaries.push(endpoint);
    let mut handle = scheduler.admit(specification.clone()).unwrap();
    scheduler.set_conditioned_inputs(handle, &images).unwrap();
    for _ in 0..150 {
        if scheduler.request(handle).unwrap().prefix_tokens == endpoint as u64 {
            break;
        }
        if scheduler
            .request(handle)
            .unwrap()
            .conditioned_window_ready()
        {
            handle = scheduler.resume_conditioned_window(handle).unwrap();
        }
        step(&mut scheduler, handle);
    }
    assert_eq!(
        scheduler.request(handle).unwrap().prefix_tokens,
        endpoint as u64
    );
    scheduler.cancel(handle).unwrap();
    handle = scheduler.admit(specification).unwrap();
    assert_eq!(
        scheduler.request(handle).unwrap().consumed_tokens,
        endpoint as u64
    );
    scheduler
        .set_conditioned_inputs(handle, &images[4..])
        .unwrap();
    assert_eq!(
        scheduler
            .request(handle)
            .unwrap()
            .feature_index(16_384 + 512)
            .unwrap(),
        512
    );
    assert!(scheduler.request(handle).unwrap().feature_index(0).is_err());
    let expected_reuse = scheduler.stats().reused_tokens;
    assert_eq!(expected_reuse, endpoint as u64);
    let mut renewals = 0;
    for _ in 0..200 {
        if scheduler.request(handle).unwrap().phase == RequestPhase::Finishing {
            break;
        }
        if scheduler
            .request(handle)
            .unwrap()
            .conditioned_window_ready()
        {
            handle = scheduler.resume_conditioned_window(handle).unwrap();
            renewals += 1;
        }
        step(&mut scheduler, handle);
        assert_eq!(scheduler.stats().reused_tokens, expected_reuse);
        assert_eq!(
            scheduler.request(handle).unwrap().reused_prompt_tokens(),
            endpoint as u64
        );
    }
    assert_eq!(renewals, 1);
    assert_eq!(
        scheduler.request(handle).unwrap().phase,
        RequestPhase::Finishing
    );
}

#[test]
fn vision_windows_make_progress_while_another_request_keeps_decoding() {
    let mut scheduler = scheduler_with_active(64, 1, 2);
    let mut durations = tests::duration_graphs(2);
    for graph in &mut durations {
        graph.estimated_duration_ns = 1_000_000;
    }
    scheduler.measured_graph_durations(&durations).unwrap();
    let decoder = scheduler.admit(tests::spec(32, 512)).unwrap();
    for _ in 0..10 {
        if scheduler.request(decoder).unwrap().phase == RequestPhase::Decode {
            break;
        }
        step(&mut scheduler, decoder);
    }
    assert_eq!(
        scheduler.request(decoder).unwrap().phase,
        RequestPhase::Decode
    );
    let mut visual = admit(&mut scheduler, &inputs(5));
    let mut vision_jobs = 0;
    let mut renewals = 0;
    for _ in 0..511 {
        if scheduler.request(visual).unwrap().generated_tokens == 1 {
            break;
        }
        if scheduler
            .request(visual)
            .unwrap()
            .conditioned_window_ready()
        {
            visual = scheduler.resume_conditioned_window(visual).unwrap();
            renewals += 1;
        }
        let selected = step(&mut scheduler, visual);
        if selected.kind == GraphKind::Vision {
            assert!(
                selected.selection_value_rate() > 0.0,
                "vision must retain the value of its following prefill"
            );
            vision_jobs += 1;
        }
        if selected.kind == GraphKind::Snapshot
            && scheduler.request(visual).unwrap().consumed_tokens
                < scheduler.request(visual).unwrap().prompt_tokens as u64
        {
            assert!(selected.selection_value_rate() > 0.0);
        }
        assert_eq!(
            scheduler.request(decoder).unwrap().phase,
            RequestPhase::Decode,
            "visual phase {:?}, consumed {}, vision jobs {}, renewals {}, elapsed {}ns",
            scheduler.request(visual).unwrap().phase,
            scheduler.request(visual).unwrap().consumed_tokens,
            vision_jobs,
            renewals,
            scheduler.now_ns
        );
    }
    assert_eq!(
        vision_jobs,
        5,
        "visual phase {:?}, consumed {}, decoder output {}, elapsed {}ns",
        scheduler.request(visual).unwrap().phase,
        scheduler.request(visual).unwrap().consumed_tokens,
        scheduler.request(decoder).unwrap().generated_tokens,
        scheduler.now_ns
    );
    assert_eq!(renewals, 1);
    let completed = scheduler.request(visual).unwrap();
    assert_eq!(completed.consumed_tokens, completed.prompt_tokens as u64);
    assert_eq!(completed.generated_tokens, 1);
    assert!(matches!(
        completed.phase,
        RequestPhase::Snapshot | RequestPhase::Finishing
    ));
}

#[test]
fn unavailable_cold_window_row_is_rejected_without_leaking_page_reservations() {
    let mut scheduler = scheduler_with_resources(64, 1, 1, 0);
    let available_pages = scheduler.free_page_count();
    let images = inputs(5);
    let handle = scheduler.admit(specification(&images)).unwrap();
    let reserved = scheduler.request(handle).unwrap().reserved_pages.len();
    let error = scheduler
        .set_conditioned_inputs(handle, &images)
        .unwrap_err();
    assert!(matches!(
        error,
        crate::Error::Admission(crate::AdmissionError::Context(_))
    ));
    assert_eq!(
        scheduler.request(handle).unwrap().reserved_pages.len(),
        reserved
    );
    assert_eq!(scheduler.request(handle).unwrap().last_use(), Fence(0));
    scheduler.cancel(handle).unwrap();
    assert_eq!(scheduler.free_page_count(), available_pages);
    assert_eq!(scheduler.free_image_slots.len(), 1);
}

#[test]
fn an_external_checkpoint_pin_keeps_window_storage_failure_retryable() {
    let mut scheduler = scheduler_with_resources(64, 1, 1, 1);
    let text = scheduler.admit(tests::spec(32, 1)).unwrap();
    for _ in 0..10 {
        if scheduler.request(text).unwrap().phase == RequestPhase::Finishing {
            break;
        }
        step(&mut scheduler, text);
    }
    let checkpoint = scheduler.finish(text).unwrap().unwrap();
    scheduler.cache.pin(checkpoint).unwrap();
    let images = inputs(5);
    let handle = scheduler.admit(specification(&images)).unwrap();
    let error = scheduler
        .set_conditioned_inputs(handle, &images)
        .unwrap_err();
    assert!(matches!(
        error,
        crate::Error::Admission(crate::AdmissionError::Capacity(_))
    ));
    scheduler.cancel(handle).unwrap();
    assert!(scheduler.cache.checkpoint(checkpoint).is_ok());
    scheduler.cache.unpin(checkpoint).unwrap();
    let handle = scheduler.admit(specification(&images)).unwrap();
    scheduler.set_conditioned_inputs(handle, &images).unwrap();
    assert!(scheduler.request(handle).unwrap().window_snapshot.is_some());
    assert!(scheduler.cache.checkpoint(checkpoint).is_err());
    scheduler.cancel(handle).unwrap();
}
