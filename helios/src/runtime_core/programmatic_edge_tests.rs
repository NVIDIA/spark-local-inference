/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use crate::ReplayWorker;
use std::time::Duration;

#[test]
#[ignore = "requires FLARE_PROGRAMMATIC_EDGE_TEST_CORE exported on the assigned GPU"]
fn native_core_preserves_programmatic_overlap() {
    let root =
        std::path::PathBuf::from(std::env::var_os("FLARE_PROGRAMMATIC_EDGE_TEST_CORE").unwrap());
    let owner =
        unsafe { NativeResources::load(test_support::authorize(root.join("core")), 0).unwrap() };
    let worker = unsafe { ReplayWorker::new(owner.graph_bundle(), 1).unwrap() };
    for _ in 0..3 {
        owner.upload("state", &[0; 64]).unwrap();
        worker.submit(0, 2).unwrap();
        let completed = worker.poll(Duration::from_secs(10)).unwrap().unwrap();
        assert!(completed.error.is_none(), "{:?}", completed.error);
        let state = owner.download("state").unwrap();
        let times: Vec<_> = (0..7)
            .map(|index| u64::from_le_bytes(state[index * 8..index * 8 + 8].try_into().unwrap()))
            .collect();
        let result = i32::from_le_bytes(state[60..64].try_into().unwrap());
        assert_eq!(result, 7);
        assert_eq!(times[6], 5);
        assert!(
            times[4] < times[5]
                && times[5] < times[0]
                && times[0] < times[2]
                && times[2] < times[1]
                && times[1] < times[3],
            "graph dependencies lost ordinary completion or programmatic overlap: {times:?}"
        );
    }
    assert!(worker.close().drained);
}
