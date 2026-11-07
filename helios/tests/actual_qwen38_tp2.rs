/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Actual Qwen selected-TP2 local cache acquisition from an externally combined
//! TP1/TP2 package. The combined package is produced by the public HPP combine
//! path; this test only reads exports and drives the public Helios cache
//! acquisition/materializer.

#![cfg(feature = "runtime-package")]

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    time::{Instant, SystemTime},
};

use flare_helios::package::{
    cache, derivation, naming, prepared_weights, progress, protection, reader, schema::wire,
};
use flare_helios::runtime_package::Publisher;
use prost::Message as _;

const CHUNK: usize = 8 << 20;

fn fixture_root() -> PathBuf {
    std::env::var_os("FLARE_QWEN38_TP_QUALIFICATION")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/data/flare-qwen38-tp-qualification"))
}

fn package_path(root: &Path, name: &str) -> PathBuf {
    let direct = root.join(name);
    if direct.is_file() {
        return direct;
    }
    direct.join("model.safetensors")
}

fn load_profiles(path: &Path) -> Vec<Publisher> {
    serde_json::from_slice(&fs::read(path).unwrap())
        .unwrap_or_else(|error| panic!("publisher profile {}: {error}", path.display()))
}


fn tp_selection(
    metadata: &reader::Metadata,
    publishers: &[Publisher],
    tensor_parallel: u32,
) -> protection::Selection {
    let envelope = protection::envelope(&metadata.envelope).unwrap();
    let mut matches = Vec::new();
    for (index, signed) in envelope.deployments.iter().enumerate() {
        let body = wire::DeploymentBody::decode(signed.body.as_slice()).unwrap();
        if body.tensor_parallel == tensor_parallel {
            matches.push(index);
        }
    }
    assert_eq!(
        matches.len(),
        1,
        "package {} must contain exactly one TP{tensor_parallel} deployment",
        metadata.path.display()
    );
    let selection = protection::Selection::Deployment {
        deployment: matches[0],
        rank: None,
    };
    protection::verify(&metadata.envelope, publishers, selection).unwrap();
    selection
}

fn payload_ids(verified: &protection::VerifiedPackage) -> BTreeSet<Vec<u8>> {
    let mut ids = BTreeSet::new();
    for rank in verified.ranks() {
        for initializer in &rank.initializers {
            if let Some(slice) = initializer
                .source
                .as_ref()
                .and_then(|source| source.slice.as_ref())
            {
                ids.insert(slice.payload_id.clone());
            }
        }
        for module in &rank.modules {
            for global in &module.globals {
                if let Some(slice) = global
                    .contents
                    .as_ref()
                    .and_then(|contents| contents.slice.as_ref())
                {
                    ids.insert(slice.payload_id.clone());
                }
            }
        }
    }
    ids
}

fn open_payloads(
    metadata: &reader::Metadata,
    verified: &protection::VerifiedPackage,
) -> reader::PayloadFiles {
    let ids = payload_ids(verified);
    metadata
        .payloads(verified.catalog(), ids.iter().map(Vec::as_slice))
        .unwrap_or_else(|error| panic!("open selected payloads from {}: {error}", metadata.path.display()))
}

fn compare_range(
    expected: &reader::PayloadFiles,
    expected_id: &[u8],
    expected_offset: u64,
    expected_len: u64,
    actual: &reader::PayloadFiles,
    actual_id: &[u8],
    actual_offset: u64,
    actual_len: u64,
    expected_buffer: &mut [u8],
    actual_buffer: &mut [u8],
) {
    assert_eq!(expected_len, actual_len, "payload range length differs");
    let expected_slice = expected
        .get(expected_id)
        .unwrap_or_else(|error| panic!("expected payload {expected_id:?}: {error}"));
    let actual_slice = actual
        .get(actual_id)
        .unwrap_or_else(|error| panic!("actual payload {actual_id:?}: {error}"));
    let expected_end = expected_offset
        .checked_add(expected_len)
        .expect("expected payload range overflow");
    let actual_end = actual_offset
        .checked_add(actual_len)
        .expect("actual payload range overflow");
    assert!(
        expected_end <= expected_slice.len(),
        "expected payload range exceeds payload"
    );
    assert!(actual_end <= actual_slice.len(), "actual payload range exceeds payload");

    let mut position = 0u64;
    while position < expected_len {
        let count = (expected_len - position).min(CHUNK as u64) as usize;
        let expected_position = expected_offset
            .checked_add(position)
            .expect("expected payload read offset overflow");
        let actual_position = actual_offset
            .checked_add(position)
            .expect("actual payload read offset overflow");
        expected_slice
            .read_at(expected_position, &mut expected_buffer[..count])
            .unwrap();
        actual_slice
            .read_at(actual_position, &mut actual_buffer[..count])
            .unwrap();
        assert_eq!(
            expected_buffer[..count],
            actual_buffer[..count],
            "payload bytes differ at source offset {position}"
        );
        position += count as u64;
    }
}

#[derive(Clone)]
struct InitializerRange {
    destination: u64,
    nbytes: u64,
    payload_id: Vec<u8>,
    source_offset: u64,
    role: i32,
}

fn initializer_ranges(
    verified: &protection::VerifiedPackage,
) -> BTreeMap<(u32, String), Vec<InitializerRange>> {
    let mut ranges: BTreeMap<(u32, String), Vec<InitializerRange>> = BTreeMap::new();
    for rank in verified.ranks() {
        for initializer in &rank.initializers {
            let Some(slice) = initializer
                .source
                .as_ref()
                .and_then(|source| source.slice.as_ref())
            else {
                continue;
            };
            ranges
                .entry((rank.rank, initializer.allocation.clone()))
                .or_default()
                .push(InitializerRange {
                    destination: initializer.destination_offset,
                    nbytes: slice.nbytes,
                    payload_id: slice.payload_id.clone(),
                    source_offset: slice.offset,
                    role: initializer
                        .source
                        .as_ref()
                        .map(|source| source.role)
                        .unwrap_or_default(),
                });
        }
    }
    for ranges in ranges.values_mut() {
        ranges.sort_by_key(|range| range.destination);
        for pair in ranges.windows(2) {
            let end = pair[0]
                .destination
                .checked_add(pair[0].nbytes)
                .expect("initializer destination overflow");
            assert!(
                end <= pair[1].destination,
                "initializer destination ranges overlap"
            );
        }
    }
    ranges
}

fn source_less_initializers(
    verified: &protection::VerifiedPackage,
) -> BTreeMap<(u32, String, u64), usize> {
    let mut result = BTreeMap::new();
    for rank in verified.ranks() {
        for initializer in &rank.initializers {
            if initializer.source.as_ref().and_then(|source| source.slice.as_ref()).is_none() {
                *result
                    .entry((
                        rank.rank,
                        initializer.allocation.clone(),
                        initializer.destination_offset,
                    ))
                    .or_default() += 1;
            }
        }
    }
    result
}

fn coverage(ranges: &[InitializerRange]) -> Vec<(u64, u64, i32)> {
    let mut result: Vec<(u64, u64, i32)> = Vec::new();
    for range in ranges {
        let end = range
            .destination
            .checked_add(range.nbytes)
            .expect("initializer destination overflow");
        match result.last_mut() {
            Some(last) if last.1 == range.destination && last.2 == range.role => last.1 = end,
            Some(last) if last.1 <= range.destination => {
                result.push((range.destination, end, range.role))
            }
            Some(_) => panic!("initializer destination ranges overlap"),
            None => result.push((range.destination, end, range.role)),
        }
    }
    result
}

fn compare_initializers(
    expected_verified: &protection::VerifiedPackage,
    expected_files: &reader::PayloadFiles,
    actual_verified: &protection::VerifiedPackage,
    actual_files: &reader::PayloadFiles,
    expected_buffer: &mut [u8],
    actual_buffer: &mut [u8],
) -> (usize, usize, u64) {
    let expected_ranges = initializer_ranges(expected_verified);
    let actual_ranges = initializer_ranges(actual_verified);
    assert_eq!(
        expected_ranges.keys().collect::<Vec<_>>(),
        actual_ranges.keys().collect::<Vec<_>>(),
        "initializer allocation coverage differs"
    );
    assert_eq!(
        source_less_initializers(expected_verified),
        source_less_initializers(actual_verified),
        "source-less initializer coverage differs"
    );
    let mut expected_count = 0usize;
    let mut actual_count = 0usize;
    let mut bytes = 0u64;
    for (key, expected) in &expected_ranges {
        let actual = actual_ranges
            .get(key)
            .unwrap_or_else(|| panic!("actual initializers omit allocation {key:?}"));
        assert_eq!(
            coverage(expected),
            coverage(actual),
            "initializer destination coverage differs for {key:?}"
        );
        expected_count += expected.len();
        actual_count += actual.len();
        let mut actual_index = 0usize;
        for expected_range in expected {
            let end = expected_range
                .destination
                .checked_add(expected_range.nbytes)
                .expect("initializer destination overflow");
            let mut position = expected_range.destination;
            while position < end {
                while actual_index < actual.len() {
                    let end = actual[actual_index]
                        .destination
                        .checked_add(actual[actual_index].nbytes)
                        .expect("actual initializer destination overflow");
                    if end > position {
                        break;
                    }
                    actual_index += 1;
                }
                let actual_range = actual
                    .get(actual_index)
                    .unwrap_or_else(|| panic!("actual initializer coverage ends early for {key:?}"));
                let actual_end = actual_range
                    .destination
                    .checked_add(actual_range.nbytes)
                    .expect("actual initializer destination overflow");
                assert!(
                    actual_range.destination <= position && position < actual_end,
                    "actual initializer coverage has a gap for {key:?}"
                );
                let stop = end.min(actual_end);
                let expected_source_offset = expected_range
                    .source_offset
                    .checked_add(position - expected_range.destination)
                    .expect("expected initializer source offset overflow");
                let actual_source_offset = actual_range
                    .source_offset
                    .checked_add(position - actual_range.destination)
                    .expect("actual initializer source offset overflow");
                compare_range(
                    expected_files,
                    &expected_range.payload_id,
                    expected_source_offset,
                    stop - position,
                    actual_files,
                    &actual_range.payload_id,
                    actual_source_offset,
                    stop - position,
                    expected_buffer,
                    actual_buffer,
                );
                bytes = bytes.saturating_add(stop - position);
                position = stop;
            }
        }
    }
    (expected_count, actual_count, bytes)
}

fn compare_rank_bytes(
    expected_files: &reader::PayloadFiles,
    expected_verified: &protection::VerifiedPackage,
    actual_files: &reader::PayloadFiles,
    actual_verified: &protection::VerifiedPackage,
) -> (usize, usize, u64, usize, u64) {
    let mut expected_buffer = vec![0u8; CHUNK];
    let mut actual_buffer = vec![0u8; CHUNK];
    let (expected_initializers, actual_initializers, initializer_bytes) = compare_initializers(
        expected_verified,
        expected_files,
        actual_verified,
        actual_files,
        &mut expected_buffer,
        &mut actual_buffer,
    );
    let actual_ranks: BTreeMap<u32, &wire::RankBody> = actual_verified
        .ranks()
        .iter()
        .map(|rank| (rank.rank, rank))
        .collect();
    let mut global_count = 0usize;
    let mut global_bytes = 0u64;
    for expected_rank in expected_verified.ranks() {
        let actual_rank = actual_ranks
            .get(&expected_rank.rank)
            .unwrap_or_else(|| panic!("installed rank {} is missing", expected_rank.rank));
        assert_eq!(
            expected_rank.modules.len(),
            actual_rank.modules.len(),
            "rank {} module count differs",
            expected_rank.rank
        );
        for (module_index, expected_module) in expected_rank.modules.iter().enumerate() {
            let actual_module = &actual_rank.modules[module_index];
            assert_eq!(
                expected_module.globals.len(),
                actual_module.globals.len(),
                "rank {} module {} global count differs",
                expected_rank.rank, module_index
            );
            for (index, expected_global) in expected_module.globals.iter().enumerate() {
                let actual_global = &actual_module.globals[index];
                assert_eq!(
                    expected_global.symbol, actual_global.symbol,
                    "rank {} module {} global {} symbol differs",
                    expected_rank.rank, module_index, index
                );
                assert_eq!(
                    expected_global.nbytes, actual_global.nbytes,
                    "rank {} module {} global {} length differs",
                    expected_rank.rank, module_index, index
                );
                let expected_source = expected_global
                    .contents
                    .as_ref()
                    .and_then(|contents| contents.slice.as_ref());
                let actual_source = actual_global
                    .contents
                    .as_ref()
                    .and_then(|contents| contents.slice.as_ref());
                assert_eq!(
                    expected_source.is_some(),
                    actual_source.is_some(),
                    "rank {} module {} global {} source presence differs",
                    expected_rank.rank, module_index, index
                );
                if let (Some(expected_source), Some(actual_source)) =
                    (expected_source, actual_source)
                {
                    assert_eq!(
                        expected_global.contents.as_ref().unwrap().role,
                        actual_global.contents.as_ref().unwrap().role,
                        "rank {} module {} global {} role differs",
                        expected_rank.rank, module_index, index
                    );
                    assert_eq!(
                        expected_source.nbytes, actual_source.nbytes,
                        "rank {} module {} global {} slice length differs",
                        expected_rank.rank, module_index, index
                    );
                    compare_range(
                        &expected_files,
                        &expected_source.payload_id,
                        expected_source.offset,
                        expected_source.nbytes,
                        &actual_files,
                        &actual_source.payload_id,
                        actual_source.offset,
                        actual_source.nbytes,
                        &mut expected_buffer,
                        &mut actual_buffer,
                    );
                    global_count += 1;
                    global_bytes = global_bytes.saturating_add(actual_source.nbytes);
                }
            }
        }
    }
    (
        expected_initializers,
        actual_initializers,
        initializer_bytes,
        global_count,
        global_bytes,
    )
}

#[derive(Default)]
struct Usage {
    logical_bytes: u64,
    physical_bytes: u64,
    shared_objects: usize,
    files: usize,
}

#[cfg(unix)]
fn usage(directory: &Path) -> Usage {
    use std::os::unix::fs::MetadataExt;
    let mut seen: BTreeMap<(u64, u64), (u64, u64, u64)> = BTreeMap::new();
    let mut stack = vec![directory.to_path_buf()];
    while let Some(path) = stack.pop() {
        for entry in fs::read_dir(&path).unwrap() {
            let entry = entry.unwrap();
            let metadata = fs::symlink_metadata(entry.path()).unwrap();
            if metadata.is_dir() {
                stack.push(entry.path());
            } else if metadata.is_file() {
                seen.insert(
                    (metadata.dev(), metadata.ino()),
                    (metadata.len(), metadata.blocks() * 512, metadata.nlink()),
                );
            }
        }
    }
    let mut usage = Usage::default();
    for (logical, physical, links) in seen.values() {
        usage.logical_bytes += logical;
        usage.physical_bytes += physical;
        usage.files += 1;
        if *links > 1 {
            usage.shared_objects += 1;
        }
    }
    usage
}

#[cfg(not(unix))]
fn usage(_directory: &Path) -> Usage {
    Usage::default()
}

fn files_below(directory: &Path) -> usize {
    let mut count = 0usize;
    let mut stack = vec![directory.to_path_buf()];
    while let Some(path) = stack.pop() {
        let entries = match fs::read_dir(&path) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => panic!("inspect cache path {}: {error}", path.display()),
        };
        for entry in entries {
            let entry = entry.unwrap();
            let metadata = fs::symlink_metadata(entry.path()).unwrap();
            if metadata.is_dir() {
                stack.push(entry.path());
            } else if metadata.is_file() {
                count += 1;
            }
        }
    }
    count
}

fn coverage_test_range(destination: u64, nbytes: u64, role: i32) -> InitializerRange {
    InitializerRange {
        destination,
        nbytes,
        payload_id: Vec::new(),
        source_offset: 0,
        role,
    }
}

#[test]
fn coverage_merges_adjacent_same_role_only() {
    let same_role = vec![
        coverage_test_range(0, 4, 1),
        coverage_test_range(4, 4, 1),
    ];
    assert_eq!(coverage(&same_role), vec![(0, 8, 1)]);

    let different_roles = vec![
        coverage_test_range(0, 4, 1),
        coverage_test_range(4, 4, 2),
    ];
    assert_eq!(
        coverage(&different_roles),
        vec![(0, 4, 1), (4, 8, 2)]
    );
}

#[test]
fn coverage_preserves_gaps_and_rejects_overlaps() {
    let gap = vec![
        coverage_test_range(0, 4, 1),
        coverage_test_range(8, 4, 1),
    ];
    assert_eq!(coverage(&gap), vec![(0, 4, 1), (8, 12, 1)]);

    let overlap = vec![
        coverage_test_range(0, 8, 1),
        coverage_test_range(4, 4, 1),
    ];
    let panic = std::panic::catch_unwind(|| coverage(&overlap)).unwrap_err();
    let message = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .unwrap_or_default();
    assert!(message.contains("overlap"), "{message}");
}

fn peak_rss_kib() -> Option<u64> {
    fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse().ok())
}

#[test]
#[ignore = "requires FLARE_QWEN38_TP_QUALIFICATION with signed TP1/TP2 and an externally combined package"]
fn actual_qwen38_tp2_local_acquisition_matches_independent_export() {
    let root = fixture_root();
    let tp1_path = package_path(&root, "tp1");
    let tp2_path = package_path(&root, "tp2");
    let combined = std::env::var_os("FLARE_QWEN38_COMBINED")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("combined"));
    assert!(tp1_path.is_file(), "missing TP1 package {}", tp1_path.display());
    assert!(tp2_path.is_file(), "missing TP2 package {}", tp2_path.display());
    assert!(
        combined.is_file() || combined.is_dir(),
        "missing externally combined package {}",
        combined.display()
    );
    let profile_path = std::env::var_os("FLARE_QWEN38_PROFILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("development-publisher.json"));
    let cache_root = std::env::var_os("FLARE_QWEN38_TP2_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("tp2-cache"));
    let profiles = load_profiles(&profile_path);

    let tp1_metadata = reader::Metadata::open(&tp1_path).unwrap();
    let tp1_selection = tp_selection(&tp1_metadata, &profiles, 1);
    let tp1_verified = protection::verify(&tp1_metadata.envelope, &profiles, tp1_selection).unwrap();
    let tp2_metadata = reader::Metadata::open(&tp2_path).unwrap();
    let tp2_selection = tp_selection(&tp2_metadata, &profiles, 2);
    let tp2_verified = protection::verify(&tp2_metadata.envelope, &profiles, tp2_selection).unwrap();
    // Pin the independent exports and their selected payload handles before acquisition.
    let expected_payloads = open_payloads(&tp2_metadata, &tp2_verified);
    let _tp1_payloads = open_payloads(&tp1_metadata, &tp1_verified);

    let mut weight_views = 0usize;
    let mut piece_view_bytes = 0u64;
    let mut raw_view_bytes = 0u64;
    let mut empty_source_view_bytes = 0u64;
    let mut unavailable_views = Vec::new();
    for rank in tp2_verified.ranks() {
        if let Some(catalog) = prepared_weights::catalog(rank).unwrap() {
            unavailable_views.extend(catalog.unavailable.iter().cloned());
            for view in &catalog.views {
                weight_views += 1;
                if view.sources.iter().any(|source| !source.pieces.is_empty()) {
                    piece_view_bytes = piece_view_bytes.saturating_add(view.nbytes);
                } else if view.sources.is_empty() {
                    empty_source_view_bytes = empty_source_view_bytes.saturating_add(view.nbytes);
                } else {
                    raw_view_bytes = raw_view_bytes.saturating_add(view.nbytes);
                }
            }
        }
    }

    let combined_metadata = reader::Metadata::open(&combined).unwrap();
    let envelope = protection::envelope(&combined_metadata.envelope).unwrap();
    let deployment_tps: Vec<u32> = envelope
        .deployments
        .iter()
        .map(|signed| {
            wire::DeploymentBody::decode(signed.body.as_slice())
                .unwrap()
                .tensor_parallel
        })
        .collect();
    assert_eq!(deployment_tps.first(), Some(&1), "combined package did not select TP1");
    assert!(deployment_tps.contains(&2), "combined package lacks TP2");
    let combined_selection = tp_selection(&combined_metadata, &profiles, 2);
    let combined_verified =
        protection::verify(&combined_metadata.envelope, &profiles, combined_selection).unwrap();


    eprintln!(
        "actual Qwen38 TP2 LOCAL acquisition begin: tp1={} tp2={} combined={} cache={}",
        tp1_path.display(),
        tp2_path.display(),
        combined.display(),
        cache_root.display()
    );
    assert_eq!(
        files_below(&cache_root),
        0,
        "TP2 qualification cache {} contains preexisting files",
        cache_root.display()
    );
    let store = cache::Root::open(&cache_root).unwrap();
    let model_key = cache::model_key(&combined_verified.package().model_identity).unwrap();
    assert_eq!(
        files_below(&cache_root.join("blobs")),
        0,
        "TP2 qualification cache {} object store is not empty",
        cache_root.display()
    );
    assert!(
        store.versions(&model_key).unwrap_or_default().is_empty(),
        "TP2 qualification cache {} has existing versions",
        cache_root.display()
    );
    let hardware = combined_verified.package().hardware_identity.clone();
    let candidate = naming::deployment_candidate(Some(&hardware), &[2], SystemTime::now())
        .expect("authenticated hardware and TP2 must produce a shared cache label");
    let staging = store.stage_named(&model_key, &candidate).unwrap();
    let source = cache::PackageSource::open(&combined, &profiles, combined_selection).unwrap();
    let progress = progress::AcquisitionProgress::new();
    let plan_started = Instant::now();
    let plan = match store.plan_acquisition_with(
        source,
        &derivation::Limits::default(),
        Some(&progress),
    ) {
        Ok(plan) => plan,
        Err(error) => {
            eprintln!(
                "actual Qwen38 TP2 LOCAL acquisition failed during planning: combined={} cache={} error={error}",
                combined.display(),
                cache_root.display()
            );
            panic!("selected TP2 planning failed: {error}");
        }
    };
    let plan_elapsed = plan_started.elapsed();
    let install_started = Instant::now();
    let entry = match store.install_planned_named(plan, staging) {
        Ok(entry) => entry,
        Err(error) => {
            eprintln!(
                "actual Qwen38 TP2 LOCAL acquisition failed during install: combined={} cache={} error={error}",
                combined.display(),
                cache_root.display()
            );
            panic!("selected TP2 install failed: {error}");
        }
    };
    let install_elapsed = install_started.elapsed();
    store.publish_entry(&entry).unwrap();
    let snapshot = progress.snapshot();
    assert!(
        snapshot.materialized_bytes() > 0,
        "selected TP2 acquisition materialized no bytes"
    );
    let full_verify_started = Instant::now();
    let full_verified = flare_helios::runtime_package::VerifiedPackage::open_tensor_parallel(
        &entry.directory().join("model.safetensors"),
        &profiles,
        2,
        None,
    )
    .unwrap();
    full_verified.verify_stored_data().unwrap();
    let full_verify_elapsed = full_verify_started.elapsed();

    let required = reader::required_payloads(&combined_verified).unwrap();
    let derivations = combined_metadata.derivations.as_ref();
    let all_target_ids: BTreeSet<&[u8]> = derivations
        .map(|set| {
            set.derivations
                .iter()
                .map(|recipe| recipe.target_payload_id.as_slice())
                .collect()
        })
        .unwrap_or_default();
    let target_ids: BTreeSet<&[u8]> = all_target_ids
        .into_iter()
        .filter(|target| required.iter().any(|id| id.as_slice() == *target))
        .collect();
    assert!(!target_ids.is_empty(), "selected TP2 closure produced no derived targets");
    let noncanonical_targets: BTreeSet<&[u8]> = target_ids
        .iter()
        .copied()
        .filter(|target| tp1_verified.catalog().length(target).is_err())
        .collect();
    for target in &noncanonical_targets {
        assert!(
            !combined_metadata
                .locations
                .locations
                .iter()
                .any(|location| location.payload_id == *target),
            "noncanonical derived target was physically installed in the combined source"
        );
    }
    let mut canonical_sources = BTreeSet::new();
    let mut canonical_copy_bytes = 0u64;
    let mut derived_initializer_bytes = 0u64;
    let mut derived_global_bytes = 0u64;
    if let Some(set) = derivations {
        for recipe in &set.derivations {
            if !target_ids.contains(recipe.target_payload_id.as_slice()) {
                continue;
            }
            for operation in &recipe.operations {
                let wire::derivation_operation::Operation::Copy(copy) = operation
                    .operation
                    .as_ref()
                    .unwrap()
                else {
                    continue;
                };
                let mut copied_bytes = copy.contiguous_bytes;
                for extent in &copy.extents {
                    copied_bytes = copied_bytes
                        .checked_mul(*extent)
                        .expect("derivation copy byte count overflow");
                }
                // Count only sources authorized by the independent TP1 canonical
                // selection. Independent TP2 target weight bodies are not evidence.
                if tp1_verified.catalog().length(&copy.source_payload_id).is_ok() {
                    canonical_sources.insert(copy.source_payload_id.clone());
                    canonical_copy_bytes = canonical_copy_bytes.saturating_add(copied_bytes);
                }
            }
        }
    }
    for rank in combined_verified.ranks() {
        for initializer in &rank.initializers {
            if let Some(slice) = initializer
                .source
                .as_ref()
                .and_then(|source| source.slice.as_ref())
                && target_ids.contains(slice.payload_id.as_slice())
            {
                derived_initializer_bytes =
                    derived_initializer_bytes.saturating_add(slice.nbytes);
            }
        }
        for module in &rank.modules {
            for global in &module.globals {
                if let Some(slice) = global
                    .contents
                    .as_ref()
                    .and_then(|contents| contents.slice.as_ref())
                    && target_ids.contains(slice.payload_id.as_slice())
                {
                    derived_global_bytes = derived_global_bytes.saturating_add(slice.nbytes);
                }
            }
        }
    }
    let mut exception_payloads = 0usize;
    let mut exception_bytes = 0u64;
    for location in &combined_metadata.locations.locations {
        if !location.file.contains("exceptions")
            || !required.iter().any(|id| id.as_slice() == location.payload_id.as_slice())
        {
            continue;
        }
        exception_payloads += 1;
        exception_bytes = exception_bytes.saturating_add(
            combined_verified
                .catalog()
                .length(&location.payload_id)
                .unwrap_or(0),
        );
    }
    assert!(
        canonical_copy_bytes > 0,
        "selected TP2 acquisition copied no canonical TP1 source bytes"
    );
    assert!(
        snapshot.materialized_copy_bytes >= canonical_copy_bytes,
        "materialized copy bytes do not cover the selected canonical copies"
    );

    let installed_metadata = reader::Metadata::open(entry.directory()).unwrap();
    let installed_selection = tp_selection(&installed_metadata, &profiles, 2);
    let installed_verified =
        protection::verify(&installed_metadata.envelope, &profiles, installed_selection).unwrap();
    assert_eq!(
        tp2_verified.package().assets,
        installed_verified.package().assets,
        "unchanged frontend assets differ from the independent TP2 package"
    );
    assert_eq!(
        tp2_verified.deployment().and_then(|deployment| deployment.engram.as_ref()),
        installed_verified
            .deployment()
            .and_then(|deployment| deployment.engram.as_ref()),
        "unchanged Engram descriptors differ from the independent TP2 package"
    );
    let (
        expected_initializer_count,
        actual_initializer_count,
        initializer_bytes,
        global_count,
        global_bytes,
    ) = compare_rank_bytes(
        &expected_payloads,
        &tp2_verified,
        &open_payloads(&installed_metadata, &installed_verified),
        &installed_verified,
    );
    assert!(
        expected_initializer_count > 0 && actual_initializer_count > 0,
        "no initializer bytes were compared"
    );
    let mut direct_shared_bytes = 0u64;
    let mut independent_bytes = 0u64;
    let mut zero_initializers = 0usize;
    for rank in installed_verified.ranks() {
        for initializer in &rank.initializers {
            match initializer
                .source
                .as_ref()
                .and_then(|source| source.slice.as_ref())
            {
                None => zero_initializers += 1,
                Some(slice) => {
                    if tp1_verified.catalog().length(&slice.payload_id).is_ok() {
                        direct_shared_bytes = direct_shared_bytes.saturating_add(slice.nbytes);
                    } else if !target_ids.contains(slice.payload_id.as_slice()) {
                        independent_bytes = independent_bytes.saturating_add(slice.nbytes);
                    }
                }
            }
        }
    }
    let mut view_canonical_bytes = 0u64;
    let mut view_derived_bytes = 0u64;
    let mut view_independent_bytes = 0u64;
    let mut view_uninitialized_padding_bytes = 0u64;
    for rank in installed_verified.ranks() {
        let Some(catalog) = prepared_weights::catalog(rank).unwrap() else {
            continue;
        };
        for view in &catalog.views {
            let Some(allocation) = rank.allocations.get(view.allocation as usize) else {
                panic!("weight view names an absent allocation");
            };
            let view_start = view.offset;
            let view_end = view_start
                .checked_add(view.nbytes)
                .expect("weight view range overflow");
            let mut covered = 0u64;
            for initializer in &rank.initializers {
                if initializer.allocation != allocation.name {
                    continue;
                }
                let Some(slice) = initializer
                    .source
                    .as_ref()
                    .and_then(|source| source.slice.as_ref())
                else {
                    continue;
                };
                let initializer_start = initializer.destination_offset;
                let initializer_end = initializer_start
                    .checked_add(slice.nbytes)
                    .expect("initializer range overflow");
                let start = initializer_start.max(view_start);
                let end = initializer_end.min(view_end);
                if start >= end {
                    continue;
                }
                let bytes = end - start;
                covered = covered.saturating_add(bytes);
                if tp1_verified.catalog().length(&slice.payload_id).is_ok() {
                    view_canonical_bytes = view_canonical_bytes.saturating_add(bytes);
                } else if target_ids.contains(slice.payload_id.as_slice()) {
                    view_derived_bytes = view_derived_bytes.saturating_add(bytes);
                } else {
                    view_independent_bytes = view_independent_bytes.saturating_add(bytes);
                }
            }
            assert!(covered <= view.nbytes, "weight view coverage exceeds its length");
            view_uninitialized_padding_bytes =
                view_uninitialized_padding_bytes.saturating_add(view.nbytes.saturating_sub(covered));
        }
    }

    let entry_usage = usage(entry.directory());
    let cache_usage = usage(&cache_root);

    println!(
        "actual Qwen38 TP2 LOCAL acquisition: plan_elapsed_ms={} install_elapsed_ms={} verification_elapsed_ms={} acquisition_elapsed_ms={} initializers={}/{} initializer_bytes={} globals={} global_bytes={} weight_views={} piece_view_bytes={} raw_view_bytes={} empty_source_view_bytes={} unavailable_views={:?} view_canonical_bytes={} view_derived_bytes={} view_independent_bytes={} view_uninitialized_padding_bytes={} canonical_sources={} canonical_copy_bytes={} derived_initializer_bytes={} derived_global_bytes={} noncanonical_targets={} direct_shared_bytes={} independent_bytes={} zero_initializers={} exception_payloads={} exception_bytes={} snapshot={:?} entry_logical_bytes={} entry_physical_bytes={} entry_files={} entry_shared_objects={} cache_logical_bytes={} cache_physical_bytes={} cache_files={} cache_shared_objects={} peak_rss_kib={:?} combined={} cache={}",
        plan_elapsed.as_millis(),
        install_elapsed.as_millis(),
        full_verify_elapsed.as_millis(),
        snapshot.elapsed.as_millis(),
        expected_initializer_count,
        actual_initializer_count,
        initializer_bytes,
        global_count,
        global_bytes,
        weight_views,
        piece_view_bytes,
        raw_view_bytes,
        empty_source_view_bytes,
        unavailable_views,
        view_canonical_bytes,
        view_derived_bytes,
        view_independent_bytes,
        view_uninitialized_padding_bytes,
        canonical_sources.len(),
        canonical_copy_bytes,
        derived_initializer_bytes,
        derived_global_bytes,
        noncanonical_targets.len(),
        direct_shared_bytes,
        independent_bytes,
        zero_initializers,
        exception_payloads,
        exception_bytes,
        snapshot,
        entry_usage.logical_bytes,
        entry_usage.physical_bytes,
        entry_usage.files,
        entry_usage.shared_objects,
        cache_usage.logical_bytes,
        cache_usage.physical_bytes,
        cache_usage.files,
        cache_usage.shared_objects,
        peak_rss_kib(),
        combined.display(),
        entry.directory().display(),
    );

    // Keep the isolated cache entry for native replay and leave the source exports untouched.
    drop(entry);
    drop(store);
}
