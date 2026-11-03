/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use crate::{ReplayWorker, cuda::Stream};
use std::{
    sync::atomic::{AtomicU32, Ordering},
    time::Duration,
};

#[test]
#[ignore = "requires FLARE_EVENT_TEST_PACKAGE exported by fixed_package_fixture.py on the assigned GPU"]
fn named_external_wait_survives_export_and_native_replay() {
    let root = std::path::PathBuf::from(std::env::var_os("FLARE_EVENT_TEST_PACKAGE").unwrap());
    let owner =
        unsafe { NativeResources::load(test_support::authorize(root.join("package")), 0).unwrap() };
    let stream = Stream::open(owner.context as usize, 0).unwrap();
    let flag = stream.mapped_host_buffer(4).unwrap();
    let address = stream.mapped_address(&flag).unwrap();
    let ready = unsafe { &*(flag.address() as *const AtomicU32) };
    let worker = unsafe { ReplayWorker::new(owner.graph_bundle(), 1).unwrap() };
    let expected: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("expected.json")).unwrap()).unwrap();
    let manifest = owner.execution_manifest().unwrap();
    manifest.validate().unwrap();
    assert_eq!(manifest.completion_token_capacity, expected["completion_token_capacity"].as_u64().unwrap() as usize);
    assert_eq!(manifest.buffer("completion_records").unwrap().shape[2], 16 + manifest.completion_token_capacity);
    for generation in 1..=3 {
        unsafe {
            stream.wait_value(address, generation).unwrap();
            stream
                .record_borrowed(owner.events["input_ready"] as usize)
                .unwrap();
        }
        worker.submit(0, 1).unwrap();
        let early = worker.poll(Duration::from_millis(20));
        ready.store(generation, Ordering::Release);
        assert!(early.unwrap().is_none());
        let result = worker.poll(Duration::from_secs(10)).unwrap().unwrap();
        assert!(result.error.is_none(), "{:?}", result.error);
        let output = owner
            .download(expected["reference"]["allocation"].as_str().unwrap())
            .unwrap();
        let offset = expected["reference"]["offset"].as_u64().unwrap() as usize;
        let values: Vec<u8> = serde_json::from_value(expected["bytes"].clone()).unwrap();
        assert_eq!(&output[offset..offset + values.len()], values);
    }
    assert!(worker.close().drained);
    stream.synchronize().unwrap();
}

#[test]
#[ignore = "requires FLARE_ENGRAM_TEST_PACKAGE exported by engram_package_fixture.py on the assigned GPU"]
fn disk_cached_and_resident_tables_feed_native_b12x_graphs() {
    use super::{payload::Files, protection::AuthorizedKind};
    use crate::{
        engram::{Binding, Shared, Storage},
        execution::{GraphExecutor, Upload},
    };
    let root = std::path::PathBuf::from(std::env::var_os("FLARE_ENGRAM_TEST_PACKAGE").unwrap());
    let table_rank: usize = std::env::var("FLARE_ENGRAM_TEST_RANK")
        .map(|rank| rank.parse().unwrap())
        .unwrap_or(0);
    let reference_rows: Vec<usize> =
        serde_json::from_slice(&std::fs::read(root.join("reference_rows.json")).unwrap()).unwrap();
    for (storage, cache_bytes, fail_read, resident_scales) in [
        (Storage::Disk, 0, false, false),
        (Storage::Ram, 0, false, false),
        (Storage::Disk, 1 << 20, false, false),
        (Storage::Disk, 0, true, false),
        (Storage::Disk, 1 << 20, true, false),
        (Storage::Disk, 0, false, true),
        (Storage::Disk, 1 << 20, false, true),
        (Storage::Disk, 1 << 20, true, true),
    ] {
        let authorized = test_support::authorize(root.join("package"));
        let AuthorizedKind::Payload(payload) = authorized.kind else {
            panic!("Engram fixture must use a payload package");
        };
        let index = payload.descriptor().unwrap();
        let (archive, mut packages) = payload.into_parts();
        let package = packages.remove(0);
        package.validate().unwrap();
        archive.validate(&package).unwrap();
        let catalog = index.engram.unwrap();
        let layout = &catalog.tables[0];
        assert_eq!(reference_rows.len(), layout.rows as usize);
        let weight_source = archive
            .asset_path(layout.weights.as_ref().unwrap())
            .unwrap();
        let (first_row, rows, world) = if layout.shards.is_empty() {
            (0, layout.rows, 1)
        } else {
            let shard = &layout.shards[table_rank];
            (shard.first_row, shard.rows, layout.shards.len())
        };
        let tables = super::engram::decode(
            catalog,
            &archive,
            std::slice::from_ref(&package),
            table_rank,
            world,
        )
        .unwrap();
        let progress = StartupProgress::silent();
        let phase = progress.phase("Load test model", None);
        let mut owner = unsafe { NativeResources::initialize(&package, 0).unwrap() };
        unsafe {
            owner.reserve(&package).unwrap();
            owner.allocate(&package, &phase).unwrap();
            owner.load_payload(&archive, &package, &phase, 1).unwrap();
            owner.prepare_kernels(&archive, &package, &phase).unwrap();
            owner.prepare_graphs(&archive, &package, &phase).unwrap();
        }
        let failed_source = std::env::temp_dir().join(format!(
            "helios-engram-truncated-{}.safetensors",
            std::process::id()
        ));
        let mut sources = archive
            .direct_files()
            .collect::<crate::Result<Vec<_>>>()
            .unwrap();
        if fail_read {
            let file = tables.tables[0].planes[0].file;
            std::fs::copy(&weight_source, &failed_source).unwrap();
            // Corrupt the private copy; snapshot payloads themselves remain read-only.
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&failed_source, std::fs::Permissions::from_mode(0o600))
                    .unwrap();
            }
            #[cfg(windows)]
            {
                let mut permissions = std::fs::metadata(&failed_source).unwrap().permissions();
                permissions.set_readonly(false);
                std::fs::set_permissions(&failed_source, permissions).unwrap();
            }
            sources[file] =
                std::sync::Arc::new(crate::os::direct_file(&failed_source, false).unwrap());
        }
        let shared = Shared::new(
            tables,
            &sources,
            1,
            4,
            owner.context as usize,
            0,
            storage,
            cache_bytes,
            1,
            resident_scales,
        )
        .unwrap();
        assert!(shared.cache_bytes() <= cache_bytes);
        let mut manifest = owner.execution_manifest().unwrap();
        let output = manifest.buffer("output").unwrap().clone();
        manifest.engram = Some(Binding {
            shared,
            rank: 0,
            events: vec![owner.events["engram_1_ready"] as usize],
        });
        let stream = Stream::open(manifest.context, 0).unwrap();
        let mut downloaded = stream.host_buffer(output.nbytes).unwrap();
        let transfer = vec![0x5a; manifest.buffer("transfer_source").unwrap().nbytes];
        let mut executor = unsafe { GraphExecutor::new(manifest).unwrap() };
        if fail_read {
            std::fs::OpenOptions::new()
                .write(true)
                .open(&failed_source)
                .unwrap()
                .set_len(0)
                .unwrap();
            let bytes = vec![0; 96 * 8];
            executor
                .submit(
                    0,
                    &[
                        Upload {
                            buffer: "engram_1_hashes",
                            offset: 0,
                            bytes: &bytes,
                        },
                        Upload {
                            buffer: "transfer_source",
                            offset: 0,
                            bytes: &transfer,
                        },
                    ],
                    false,
                )
                .unwrap()
                .unwrap();
            // A second reader job may already be queued when the first disk read fails.
            // Either preflight observes that error or both queued frames drain explicitly.
            match executor.submit(
                0,
                &[
                    Upload {
                        buffer: "engram_1_hashes",
                        offset: 0,
                        bytes: &bytes,
                    },
                    Upload {
                        buffer: "transfer_source",
                        offset: 0,
                        bytes: &transfer,
                    },
                ],
                false,
            ) {
                Ok(Some(_)) => {}
                Ok(None) => panic!("second Engram frame was unavailable"),
                Err(error) => assert!(error.to_string().contains("Engram read")),
            }
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            loop {
                match executor.poll() {
                    Err(error) => {
                        assert!(error.to_string().contains("Engram read"));
                        break;
                    }
                    Ok(Some(_)) => panic!("failed table read delivered a completion"),
                    Ok(None) => {}
                }
                assert!(std::time::Instant::now() < deadline);
                std::thread::yield_now();
            }
            let shutdown = executor.close();
            assert!(shutdown.drained && shutdown.error.is_some());
            std::fs::remove_file(failed_source).unwrap();
            continue;
        }
        // Distinct requests exercise both pinned frames before either completion is retired.
        for generations in [[1, 2], [2, 3], [3, 1]] {
            let mut pending = Vec::new();
            for generation in generations {
                let ids: Vec<i64> = (0..96)
                    .map(|row| {
                        if generation == 3 && row >= 48 {
                            -1
                        } else {
                            (row * 19 + generation) % rows as i64
                        }
                    })
                    .collect();
                let checksum: i64 = ids
                    .iter()
                    .filter(|id| **id >= 0)
                    .map(|id| {
                        let source_row = reference_rows[first_row as usize + *id as usize];
                        (0..256)
                            .map(|column| {
                                let value = ((source_row * 256 + column) % 15) as i64 - 7;
                                value * value
                            })
                            .sum::<i64>()
                    })
                    .sum();
                let bytes: Vec<u8> = ids.iter().flat_map(|n| n.to_le_bytes()).collect();
                let fence = executor
                    .submit(
                        0,
                        &[
                            Upload {
                                buffer: "engram_1_hashes",
                                offset: 0,
                                bytes: &bytes,
                            },
                            Upload {
                                buffer: "transfer_source",
                                offset: 0,
                                bytes: &transfer,
                            },
                        ],
                        false,
                    )
                    .unwrap()
                    .unwrap();
                pending.push((fence, checksum, ids));
            }
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            for (fence, checksum, _) in &pending {
                loop {
                    if let Some(completion) = executor.poll().unwrap() {
                        assert_eq!(completion.fence, *fence);
                        assert_eq!(
                            completion.records[0].consumed_tokens, *checksum,
                            "{storage:?}: captured row checksum for {fence:?}"
                        );
                        break;
                    }
                    assert!(std::time::Instant::now() < deadline);
                    std::thread::yield_now();
                }
            }
            let generation = generations[1];
            let ids = &pending[1].2;
            unsafe {
                stream
                    .download(&downloaded, output.address, output.nbytes)
                    .unwrap();
            }
            stream.synchronize().unwrap();
            for (row, id) in ids.iter().enumerate() {
                for column in 0..256 {
                    let value = if *id < 0 {
                        0.0
                    } else {
                        let source_row = reference_rows[first_row as usize + *id as usize];
                        ((source_row * 256 + column) % 15) as f32 - 7.0
                    };
                    let expected = (value.to_bits() >> 16) as u16;
                    let offset = (row * 256 + column) * 2;
                    let actual = u16::from_le_bytes(
                        downloaded.bytes()[offset..offset + 2].try_into().unwrap(),
                    );
                    assert_eq!(
                        actual, expected,
                        "{storage:?}: generation {generation}, row {row}, column {column}"
                    );
                }
            }
            downloaded.bytes_mut().fill(0xA5);
        }
        assert!(executor.close().drained);
    }
}
