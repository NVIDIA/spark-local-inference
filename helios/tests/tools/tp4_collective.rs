/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Reload core-owned TP4 transports after the Python preparation process exits.

#[path = "allocation_watch.rs"]
mod allocation_watch;

use flare_helios::{
    execution::{BufferDescriptor, ExecutionManifest, ExecutionWorker},
    runtime_core::{KvCacheLimit, LoadOptions, NativeModel, test_support},
    startup::StartupProgress,
};
use serde::Deserialize;
use std::{ffi::c_void, path::PathBuf, thread, time::{Duration, Instant}};

#[derive(Deserialize)]
struct Expected {
    ranks: Vec<Rank>, preparation_processes: Vec<Process>,
    #[serde(default)] startup_tuning: bool,
}
#[derive(Deserialize)]
struct Process { pid: u32, start_ticks: u64 }
#[derive(Deserialize)]
struct Rank { cases: Vec<Case>, #[serde(default)] profiles: Vec<Profile> }
#[derive(Deserialize)]
struct Profile { buffer: String, status: String, allowed: Vec<i32> }
#[derive(Deserialize)]
struct Case { graph: String, source: String, output: String, input: String, expected: String, step: usize }

type Context = unsafe extern "C" fn(*mut c_void) -> i32;
type Upload = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
type Download = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
type Synchronize = unsafe extern "C" fn() -> i32;

struct Driver {
    _library: libloading::Library,
    context: Context,
    upload: Upload,
    download: Download,
    synchronize: Synchronize,
}
impl Driver {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        unsafe {
            let library = libloading::Library::new("libcuda.so.1")?;
            Ok(Self {
                context: *library.get(b"cuCtxSetCurrent\0")?,
                upload: *library.get(b"cuMemcpyHtoD_v2\0")?,
                download: *library.get(b"cuMemcpyDtoH_v2\0")?,
                synchronize: *library.get(b"cuCtxSynchronize\0")?,
                _library: library,
            })
        }
    }
    fn check(status: i32) -> Result<(), Box<dyn std::error::Error>> {
        if status == 0 { Ok(()) } else { Err(format!("CUDA fixture call failed: {status}").into()) }
    }
    fn select(&self, manifest: &ExecutionManifest) -> Result<(), Box<dyn std::error::Error>> {
        unsafe { Self::check((self.context)(manifest.context as *mut c_void)) }
    }
    fn write(&self, buffer: &BufferDescriptor, bytes: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(buffer.nbytes, bytes.len());
        unsafe { Self::check((self.upload)(buffer.address as u64, bytes.as_ptr().cast(), bytes.len())) }
    }
    fn read(&self, buffer: &BufferDescriptor) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let mut bytes = vec![0; buffer.nbytes];
        unsafe { Self::check((self.download)(bytes.as_mut_ptr().cast(), buffer.address as u64, bytes.len()))?; }
        Ok(bytes)
    }
}

fn buffer<'a>(manifest: &'a ExecutionManifest, name: &str) -> &'a BufferDescriptor {
    manifest.buffers.iter().find(|buffer| buffer.name == name).expect("fixture buffer missing")
}

fn scalar(driver: &Driver, manifest: &ExecutionManifest, name: &str) -> Result<i32, Box<dyn std::error::Error>> {
    driver.select(manifest)?;
    Ok(i32::from_ne_bytes(driver.read(buffer(manifest, name))?.try_into().unwrap()))
}

fn check_startup(driver: &Driver, manifests: &[ExecutionManifest]) -> Result<(), Box<dyn std::error::Error>> {
    for manifest in manifests {
        assert_eq!(scalar(driver, manifest, "startup-count")?, 1, "shared producer must run once");
        assert_eq!(scalar(driver, manifest, "startup-selector")?, 3);
        assert_eq!(scalar(driver, manifest, "startup-bias")?, 7);
        assert_eq!(scalar(driver, manifest, "unused-count")?, 0, "unreferenced producer ran");
    }
    Ok(())
}

pub fn run(root: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let expected: Expected = serde_json::from_slice(&std::fs::read(root.join("collective-expected.json"))?)?;
    assert_eq!(expected.ranks.len(), 4);
    for process in &expected.preparation_processes {
        if let Ok(stat) = std::fs::read_to_string(format!("/proc/{}/stat", process.pid)) {
            let ticks = stat.rsplit_once(") ").unwrap().1.split_whitespace().nth(19).unwrap().parse::<u64>()?;
            assert_ne!(ticks, process.start_ticks, "Python preparation process is still alive");
        }
    }
    let allocations = allocation_watch::AllocationWatch::new()?;
    // Bound the toy cache to its declared 128-token context. Memory mode's
    // 8192-token minimum is intended for model-serving memory allocation.
    let options = LoadOptions { kv_cache_limit: KvCacheLimit::Context, ..LoadOptions::default() };
    let owner = unsafe { NativeModel::load(test_support::authorize(root.join("core")),
        &[0, 1, 2, 3], &options, &StartupProgress::silent())? };
    let manifests = owner.manifests()?;
    assert_eq!(manifests.len(), 4);
    let driver = Driver::new()?;
    let fail_startup = std::env::args().any(|argument| argument == "--startup-failure");
    let collective_profiles = !expected.ranks[0].profiles.is_empty();
    let workers = if collective_profiles {
        assert!(!fail_startup);
        for (rank, manifest) in manifests.iter().enumerate() {
            for profile in &expected.ranks[rank].profiles {
                assert_eq!(scalar(&driver, manifest, &profile.buffer)?, -1);
                assert_eq!(scalar(&driver, manifest, &profile.status)?, -1);
            }
        }
        vec![unsafe { ExecutionWorker::new_threaded(manifests[0].clone(), manifests[1..].to_vec())? }]
    } else if expected.startup_tuning {
        for manifest in &manifests {
            assert_eq!(scalar(&driver, manifest, "startup-count")?, 0);
            assert_eq!(scalar(&driver, manifest, "startup-selector")?, 0);
            assert_eq!(scalar(&driver, manifest, "startup-bias")?, 0);
            assert_eq!(scalar(&driver, manifest, "startup-observed")?, 0);
        }
        if fail_startup {
            driver.select(&manifests[1])?;
            driver.write(buffer(&manifests[1], "startup-failure"), &17i32.to_ne_bytes())?;
        }
        let opened = unsafe { ExecutionWorker::new_threaded(manifests[0].clone(), manifests[1..].to_vec()) };
        if fail_startup {
            let error = match opened {
                Ok(worker) => {
                    assert!(worker.close().drained);
                    return Err("startup accepted a failing rank".into());
                }
                Err(error) => error,
            };
            assert!(error.to_string().contains("status 17"), "{error}");
            check_startup(&driver, &manifests)?;
            for manifest in &manifests {
                assert_eq!(scalar(&driver, manifest, "startup-observed")?, 0, "request graph ran before rejection");
            }
            println!("native startup tuning rejected failing rank before request readiness: {error}");
            return Ok(());
        }
        let worker = opened?;
        check_startup(&driver, &manifests)?;
        vec![worker]
    } else {
        assert!(!fail_startup, "startup failure requires the startup fixture");
        manifests.iter().map(|manifest| unsafe { ExecutionWorker::new(manifest.clone()) })
            .collect::<Result<Vec<_>, _>>()?
    };
    let mut selected = Vec::new();
    for (rank, manifest) in manifests.iter().enumerate() {
        let mut values = Vec::new();
        for profile in &expected.ranks[rank].profiles {
            let value = scalar(&driver, manifest, &profile.buffer)?;
            assert!(profile.allowed.contains(&value), "startup selected an unlisted profile: {value}");
            assert_eq!(scalar(&driver, manifest, &profile.status)?, 0);
            values.push(value);
        }
        selected.push(values);
    }
    assert!(selected.iter().all(|values| values == &selected[0]), "startup profiles differ across ranks");
    let baseline = allocations.count();
    for case in &expected.ranks[0].cases {
        let mut inputs = Vec::new();
        let mut graph_indices = Vec::new();
        for (rank, manifest) in manifests.iter().enumerate() {
            let local = expected.ranks[rank].cases.iter()
                .find(|entry| entry.graph == case.graph && entry.step == case.step).unwrap();
            let input = std::fs::read(root.join(&local.input))?;
            driver.select(manifest)?;
            driver.write(buffer(manifest, &local.source), &input)?;
            let output = buffer(manifest, &local.output);
            driver.write(output, &vec![0xff; output.nbytes])?;
            if expected.startup_tuning && matches!(case.graph.as_str(), "init" | "decode") {
                driver.write(buffer(manifest, "startup-observed"), &(-99i32).to_ne_bytes())?;
            }
            unsafe { Driver::check((driver.synchronize)())?; }
            inputs.push(input);
            graph_indices.push(manifest.graphs.iter().position(|graph| graph.name == local.graph).unwrap());
        }
        let tickets = workers.iter().zip(&graph_indices)
            .map(|(worker, &graph)| worker.submit(graph, vec![], false)?.ok_or("native fixture queue full".into()))
            .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
        for (rank, worker) in workers.iter().enumerate() {
            let deadline = Instant::now() + Duration::from_secs(30);
            loop {
                if let Some(completion) = worker.poll()? {
                    assert_eq!(completion.fence, tickets[rank]);
                    assert_eq!(completion.records.len(), 1);
                    assert_eq!(completion.records[0].fields(), vec![
                        0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 1,
                        -1, -1, -1, -1, -1, -1, -1, -1,
                    ]);
                    break;
                }
                if Instant::now() > deadline { return Err("TP4 collective replay timed out".into()); }
                thread::sleep(Duration::from_millis(1));
            }
        }
        for (rank, manifest) in manifests.iter().enumerate() {
            let local = expected.ranks[rank].cases.iter()
                .find(|entry| entry.graph == case.graph && entry.step == case.step).unwrap();
            driver.select(manifest)?;
            assert_eq!(driver.read(buffer(manifest, &local.source))?, inputs[rank], "source mutated");
            assert_eq!(driver.read(buffer(manifest, &local.output))?, std::fs::read(root.join(&local.expected))?,
                "rank {rank} output differs for {} step {}", local.graph, local.step);
            if expected.startup_tuning && matches!(case.graph.as_str(), "init" | "decode") {
                assert_eq!(scalar(&driver, manifest, "startup-observed")?, 40,
                    "compiled consumer did not read both persistent startup outputs");
            }
        }
        if expected.startup_tuning { check_startup(&driver, &manifests)?; }
        for (rank, manifest) in manifests.iter().enumerate() {
            for (profile, value) in expected.ranks[rank].profiles.iter().zip(&selected[rank]) {
                assert_eq!(scalar(&driver, manifest, &profile.buffer)?, *value, "serving mutated its profile");
            }
        }
    }
    assert_eq!(allocations.count(), baseline, "CUDA allocation during native collective replay");
    for worker in &workers {
        let shutdown = worker.close();
        assert!(shutdown.drained, "native collective worker did not drain");
        if let Some(error) = shutdown.error { return Err(error.into()); }
    }
    let maps = std::fs::read_to_string("/proc/self/maps")?;
    assert!(!maps.contains("libpython") && !maps.contains("libtorch") && !maps.contains("site-cores"));
    drop(workers);
    drop(owner);
    println!("native TP4 collective core replay passed after Python exit: changed inputs, poisoned outputs, exact payload bytes, source preservation, allocation stability, all-rank drain");
    Ok(())
}
