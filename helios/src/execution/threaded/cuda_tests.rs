/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use crate::runtime_package::{NativeResources, driver::Driver, test_support};

fn uploads(limits: [i32; 4], confidence: f32, reset: bool, terminal: bool, slots: [i32; 4]) -> Vec<OwnedUpload> {
    let mut values = vec![
        ("speculative_prefix_limits", limits.into_iter().flat_map(i32::to_le_bytes).collect()),
        ("test_confidence", [confidence; 12].into_iter().flat_map(f32::to_le_bytes).collect()),
        ("live_rows", 2i32.to_le_bytes().to_vec()),
        ("reset_state", i32::from(reset).to_le_bytes().to_vec()),
        ("terminal_after_round", i32::from(terminal).to_le_bytes().to_vec()),
        ("row_slot_ids", slots.into_iter().flat_map(i32::to_le_bytes).collect()),
        ("row_generations", slots.into_iter().flat_map(|slot| (11 + i64::from(slot)).to_le_bytes()).collect()),
    ];
    values.drain(..).map(|(buffer, bytes)| OwnedUpload { buffer: buffer.into(), offset: 0, bytes }).collect()
}

fn borrowed(owned: &[OwnedUpload]) -> Vec<Upload<'_>> {
    owned.iter().map(|upload| Upload { buffer: &upload.buffer, offset: upload.offset, bytes: &upload.bytes }).collect()
}

#[test]
#[ignore = "requires four assigned CUDA devices and FLARE_THREADED_TEST_PACKAGE"]
fn four_cuda_rank_threads_preserve_two_frames_and_device_continuations() {
    let root = std::path::PathBuf::from(std::env::var_os("FLARE_THREADED_TEST_PACKAGE").expect("fixture directory"));
    let owners: Vec<_> = (0..4).map(|rank| unsafe {
        NativeResources::load(test_support::authorize(root.join(format!("rank-{rank}/package"))), rank).unwrap()
    }).collect();
    let mut manifests: Vec<_> = owners.iter().map(|owner| owner.execution_manifest().unwrap()).collect();
    let root_manifest = manifests.remove(0);
    let pending_buffer = root_manifest.buffer("pending_inputs").unwrap().clone();
    let context = root_manifest.context;
    let completion_capacity = root_manifest.completion_token_capacity;
    let mut executor = unsafe { ThreadedExecutor::new(root_manifest, manifests).unwrap() };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        for (same_requests, terminal) in [(true, false), (false, false), (true, true)] {
            let slots = [[3, 0, 2, 1], if same_requests { [3, 0, 2, 1] } else { [2, 1, 3, 0] }];
            let limits = [[2, 1, 0, 0], [1, 3, 0, 0]];
            let confidence = [0.75, 0.25];
            let owned = [uploads(limits[0], confidence[0], true, terminal, slots[0]),
                         uploads(limits[1], confidence[1], false, false, slots[1])];
            let first = executor.submit(0, &borrowed(&owned[0]), &[], false).unwrap().unwrap();
            assert_eq!(executor.outstanding(), 1);
            assert!(!executor.can_submit(false));
            let deadline = Instant::now() + Duration::from_secs(30);
            let second = loop {
                assert!(Instant::now() < deadline, "first frame preparation timed out");
                if let Some(fence) = executor.submit(0, &borrowed(&owned[1]), &[], false).unwrap() { break fence; }
                assert!(executor.poll().unwrap().is_none(), "first completion was consumed before the second reservation");
                thread::yield_now();
            };
            assert_eq!(second.0, first.0 + 1);
            assert_eq!(executor.outstanding(), 2);
            assert!(!executor.can_submit(false));
            assert!(executor.submit(0, &borrowed(&owned[1]), &[], false).unwrap().is_none());
            for (round, fence) in [first, second].into_iter().enumerate() {
                let completion = loop {
                    assert!(Instant::now() < deadline, "rank completion timed out");
                    if let Some(completion) = executor.poll().unwrap() { break completion; }
                    thread::yield_now();
                };
                assert_eq!(completion.fence, fence);
                assert_eq!(executor.outstanding(), 1 - round);
                assert!(completion.timing.gpu_ns[..4].iter().all(Option::is_some));
                assert_eq!(completion.timing.queued, round != 0);
                let reported = completion.proposal_confidence.unwrap();
                for row in 0..4 {
                    let expected = if row < 2 && !(terminal && round == 1) { confidence[round] } else { 0.0 };
                    assert_eq!(&reported[row][..3], &[expected; 3]);
                    assert_eq!(reported[row].len(), completion_capacity - 1);
                    assert!(reported[row][3..].iter().all(|value| *value == 0.0));
                }
                for row in 0..2 {
                    let record = &completion.records[slots[round][row] as usize];
                    let emitted = if terminal { usize::from(round == 0) } else { limits[round][row] as usize };
                    let predecessor = if round == 1 && same_requests { if terminal { 1 } else { limits[0][row] as usize } } else { 0 };
                    assert_eq!(record.submission_sequence, fence.0 as i64);
                    assert_eq!(record.consumed_tokens, (10 + predecessor + emitted) as i64);
                    assert_eq!(record.generated_tokens, (3 + predecessor + emitted) as i64);
                    assert_eq!(record.vector.as_ref().unwrap().emitted_count, emitted as i64);
                    if terminal { assert_eq!(record.status, 2); }
                }
                for slot in &slots[round][2..] { assert_eq!(completion.records[*slot as usize].status, 0); }
            }
            if !terminal {
                let driver = Driver::load().unwrap();
                let mut bytes = vec![0u8; pending_buffer.nbytes];
                unsafe {
                    driver.check("cuCtxSetCurrent", (driver.context_set)(context as *mut _)).unwrap();
                    driver.check("cuCtxSynchronize", (driver.context_sync)()).unwrap();
                    driver.check("cuMemcpyDtoH_v2", (driver.download)(bytes.as_mut_ptr().cast(), pending_buffer.address as u64, bytes.len())).unwrap();
                }
                let values: Vec<_> = bytes.chunks_exact(8).map(|value| i64::from_le_bytes(value.try_into().unwrap())).collect();
                assert_eq!([values[0], values[8]], if same_requests { [6, 5] } else { [4, 4] });
            }
        }
        let owned = uploads([2, 1, 0, 0], 0.5, true, false, [3, 0, 2, 1]);
        executor.submit(0, &borrowed(&owned), &[], false).unwrap().unwrap();
        let deadline = Instant::now() + Duration::from_secs(30);
        while executor.submit(0, &borrowed(&owned), &[], false).unwrap().is_none() {
            assert!(Instant::now() < deadline, "preparation before shutdown timed out");
            assert!(executor.poll().unwrap().is_none());
            thread::yield_now();
        }
        assert_eq!(executor.outstanding(), 2);
    }));
    let shutdown = executor.close();
    if !shutdown.drained {
        std::mem::forget(owners);
        panic!("rank resources retained after failed drain: {:?}", shutdown.error);
    }
    assert!(shutdown.error.is_none(), "{:?}", shutdown.error);
    result.unwrap();
}
