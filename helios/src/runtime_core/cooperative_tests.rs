/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use crate::ReplayWorker;
use std::time::Duration;

#[test]
fn cooperative_worker_grids_clamp_along_each_axis() {
    for axis in 0..3 {
        let mut captured = [1; 3];
        let mut expected = [1; 3];
        captured[axis] = 48;
        expected[axis] = 40;
        assert_eq!(clamp_cooperative_grid(captured, 40).unwrap(), expected);
        assert_eq!(clamp_cooperative_grid(expected, 40).unwrap(), expected);
        assert_eq!(clamp_cooperative_grid(captured, 80).unwrap(), captured);
    }
}

#[test]
fn cooperative_grid_clamping_preserves_other_dimensions() {
    for (captured, expected) in [
        ([48, 2, 3], [6, 2, 3]),
        ([2, 48, 3], [2, 6, 3]),
        ([2, 3, 48], [2, 3, 6]),
    ] {
        assert_eq!(clamp_cooperative_grid(captured, 40).unwrap(), expected);
    }
}

#[test]
fn cooperative_grid_clamping_rejects_planes_above_capacity() {
    for (grid, capacity) in [([1, 1, 48], 0), ([8, 8, 8], 40), ([u32::MAX; 3], 40)] {
        assert!(clamp_cooperative_grid(grid, capacity).is_err());
    }
}

#[test]
#[ignore = "requires FLARE_COOPERATIVE_TEST_CORE exported by cooperative_core_fixture.py on the assigned GPU"]
fn oversized_cooperative_grids_replay_at_device_capacity() {
    let root =
        std::path::PathBuf::from(std::env::var_os("FLARE_COOPERATIVE_TEST_CORE").unwrap());
    let owner =
        unsafe { NativeResources::load(test_support::authorize(root.join("core")), 0).unwrap() };
    let expected: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("expected.json")).unwrap()).unwrap();
    let length = expected["length"].as_u64().unwrap() as usize;
    let worker = unsafe { ReplayWorker::new(owner.graph_bundle(), 1).unwrap() };
    let poison = vec![0xff; (length * 2 + 3) * 4];
    for _ in 0..4 {
        for (index, name) in owner.graph_names().iter().enumerate() {
            owner.upload(name, &poison).unwrap();
            worker.submit(index, 3).unwrap();
            let result = worker.poll(Duration::from_secs(10)).unwrap().unwrap();
            assert!(result.error.is_none(), "{:?}", result.error);
            let bytes = owner.download(name).unwrap();
            let values: Vec<i32> = bytes
                .chunks_exact(4)
                .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            for i in 0..length {
                assert_eq!(values[i], i as i32 * 3 + 7, "{name}: element {i}");
                assert_eq!(
                    values[length + i],
                    if name == "ordinary" {
                        -1
                    } else {
                        (length - 1 - i) as i32 * 3 + 18
                    },
                    "{name}: post-barrier element {i}",
                );
            }
            let grid: Vec<i32> = serde_json::from_value(expected["grids"][name].clone()).unwrap();
            assert_eq!(values[length * 2..], grid, "{name}: replayed grid");
        }
    }
    assert!(worker.close().drained);
}
