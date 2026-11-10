/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cross-process CUDA graph replay with every allocation verified to relocate.

mod allocation_watch;

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use flare_helios::{ReplayWorker, runtime_core::NativeResources};
use serde::Deserialize;

#[derive(Deserialize)]
struct Expected {
    addresses: HashMap<String, usize>,
    outputs: HashMap<String, String>,
    modified_outputs: HashMap<String, String>,
    pointers: Vec<Reference>,
}

#[derive(Deserialize)]
struct Reference {
    allocation: String,
    offset: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("expected fixture directory")?,
    );
    let expected: Expected = serde_json::from_slice(&std::fs::read(root.join("expected.json"))?)?;
    let allocations = allocation_watch::AllocationWatch::new()?;
    let owner = unsafe {
        NativeResources::load(
            flare_helios::runtime_core::test_support::authorize(root.join("core")),
            0,
        )?
    };
    assert!(
        allocations.count() > 0,
        "CUDA allocation callbacks were not recorded"
    );
    let payload = owner.download("payload")?;
    assert_eq!(payload.len(), (20 << 20) + 17);
    assert!(
        payload
            .iter()
            .enumerate()
            .all(|(index, value)| *value == (index % 251) as u8)
    );
    for (name, original) in &expected.addresses {
        assert_ne!(
            owner.allocation(name)?.address,
            *original,
            "allocation {name} did not relocate"
        );
    }
    let pointers = owner.download("pointers")?;
    for (entry, reference) in pointers.chunks_exact(8).zip(&expected.pointers) {
        assert_eq!(
            u64::from_le_bytes(entry.try_into()?),
            (owner.allocation(&reference.allocation)?.address + reference.offset) as u64
        );
    }
    let worker = unsafe { ReplayWorker::new(owner.graph_bundle(), 1)? };
    let poison = vec![0xff; owner.allocation("scratch")?.nbytes];
    let mut baseline = None;
    for iteration in 0..32 {
        if iteration == 16 {
            owner.upload("input", &std::fs::read(root.join("modified-input.bin"))?)?;
        }
        let outputs = if iteration < 16 {
            &expected.outputs
        } else {
            &expected.modified_outputs
        };
        for (index, name) in owner.graph_names().iter().enumerate() {
            owner.upload("scratch", &poison)?;
            let ticket = worker.submit(index, 3)?;
            let result = worker
                .poll(Duration::from_secs(30))?
                .ok_or("replay timed out")?;
            assert_eq!(result.ticket, ticket);
            if let Some(error) = result.error {
                return Err(error.into());
            }
            let reference = std::fs::read(root.join(&outputs[name]))?;
            let output = owner.download(name)?;
            assert_eq!(output, reference, "output differs for {name}");
        }
        if iteration == 0 {
            baseline = Some(allocations.count());
        }
    }
    assert_eq!(
        Some(allocations.count()),
        baseline,
        "CUDA allocation occurred during repeated replay"
    );
    let shutdown = worker.close();
    assert!(shutdown.drained, "native replay did not drain");
    if let Some(error) = shutdown.error {
        return Err(error.into());
    }
    let maps = std::fs::read_to_string("/proc/self/maps")?;
    assert!(
        !maps.contains("libpython")
            && !maps.contains("libtorch")
            && !maps.contains("site-cores")
    );
    drop(owner);
    println!(
        "native replay passed: relocated allocations, pointer tables, 64 alternating submissions, 192 graph launches, changed inputs, poisoned scratch, no CUDA allocation calls during replay; no Python or Torch mappings"
    );
    Ok(())
}
