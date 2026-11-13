/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Exercise actual Heliopause W4A16 rank exports through Rust combine/cache.
//!
//! The fixture is external and deliberately test-only. It must contain signed
//! TP1 and TP2 cores produced from actual prepared rank metadata and bytes;
//! this test never synthesizes prepared weights or substitutes a weight-only
//! matrix for the real weight/scale owner layout.

#![cfg(feature = "runtime-core")]

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use flare_helios::core::{
    cache, container, conversion, derivation, prepared_weights, protection, reader,
    schema::wire, PayloadId,
};
use flare_helios::runtime_core::Publisher;
use prost::Message as _;
use ring::signature::Ed25519KeyPair;

fn fixture_root() -> Option<PathBuf> {
    std::env::var_os("FLARE_ACTUAL_W4A16_FIXTURE").map(PathBuf::from)
}

fn load_profiles(path: &Path) -> Vec<Publisher> {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn signing_key(path: &Path) -> Ed25519KeyPair {
    // OpenSSL emits PKCS#8 v1 keys without the embedded public key.
    Ed25519KeyPair::from_pkcs8_maybe_unchecked(&fs::read(path).unwrap()).unwrap()
}

fn read_all(slice: &container::Slice) -> Vec<u8> {
    let mut bytes = vec![0; slice.len() as usize];
    slice.read_at(0, &mut bytes).unwrap();
    bytes
}

fn manifest_bytes(metadata: &reader::Metadata) -> Vec<u8> {
    read_all(&metadata.primary().tensor(reader::MANIFEST).unwrap())
}

fn payload_snapshot(metadata: &reader::Metadata) -> BTreeMap<Vec<u8>, Vec<u8>> {
    let mut payloads = BTreeMap::new();
    for location in &metadata.locations.locations {
        let container = if location.file.is_empty() {
            container::Container::open(&metadata.path).unwrap()
        } else {
            container::Container::open(
                &metadata
                    .path
                    .parent()
                    .unwrap()
                    .join(&location.file),
            )
            .unwrap()
        };
        let bytes = read_all(&container.tensor(&location.tensor).unwrap_or_else(|error| {
            panic!(
                "payload snapshot {} file={:?} tensor={:?}: {error}",
                metadata.path.display(),
                location.file,
                location.tensor
            )
        }));
        assert_eq!(
            PayloadId::of(&bytes).as_bytes().to_vec(),
            location.payload_id,
            "payload bytes differ from the signed location"
        );
        payloads.insert(location.payload_id.clone(), bytes);
    }
    payloads
}

fn tp_selection(
    metadata: &reader::Metadata,
    publishers: &[Publisher],
    tensor_parallel: u32,
) -> protection::Selection {
    let envelope = protection::envelope(&metadata.envelope).unwrap();
    for (index, signed) in envelope.deployments.iter().enumerate() {
        let body = wire::DeploymentBody::decode(signed.body.as_slice()).unwrap();
        if body.tensor_parallel == tensor_parallel {
            let selection = protection::Selection::Deployment {
                deployment: index,
                rank: None,
            };
            protection::verify(&metadata.envelope, publishers, selection).unwrap();
            return selection;
        }
    }
    panic!("fixture has no TP{tensor_parallel} deployment");
}

fn read_payload_range(
    metadata: &reader::Metadata,
    verified: &protection::VerifiedCore,
    id: &[u8],
    offset: u64,
    nbytes: u64,
) -> Vec<u8> {
    let files = metadata.payloads(verified.catalog(), [id]).unwrap();
    let payload = files.get(id).unwrap();
    assert!(
        offset
            .checked_add(nbytes)
            .is_some_and(|end| end <= payload.len()),
        "initializer/global slice exceeds its payload"
    );
    let mut bytes = vec![0; nbytes as usize];
    payload.read_at(offset, &mut bytes).unwrap();
    bytes
}

#[derive(Debug)]
struct InitializerBytes {
    rank: u32,
    index: usize,
    allocation: String,
    destination_offset: u64,
    payload_id: Vec<u8>,
    slice_offset: u64,
    nbytes: u64,
    bytes: Vec<u8>,
}

#[derive(Debug)]
struct GlobalBytes {
    rank: u32,
    module: usize,
    index: usize,
    symbol: String,
    nbytes: u64,
    payload_id: Vec<u8>,
    slice_offset: u64,
    bytes: Vec<u8>,
}

fn initializer_snapshot(
    metadata: &reader::Metadata,
    verified: &protection::VerifiedCore,
) -> (Vec<InitializerBytes>, Vec<GlobalBytes>) {
    let mut initializers = Vec::new();
    let mut globals = Vec::new();
    for rank in verified.ranks() {
        for (index, initializer) in rank.initializers.iter().enumerate() {
            let Some(slice) = initializer.source.as_ref().and_then(|use_| use_.slice.as_ref()) else {
                continue;
            };
            initializers.push(InitializerBytes {
                rank: rank.rank,
                index,
                allocation: initializer.allocation.clone(),
                destination_offset: initializer.destination_offset,
                payload_id: slice.payload_id.clone(),
                slice_offset: slice.offset,
                nbytes: slice.nbytes,
                bytes: read_payload_range(
                    metadata,
                    verified,
                    &slice.payload_id,
                    slice.offset,
                    slice.nbytes,
                ),
            });
        }
        for (module, module_body) in rank.modules.iter().enumerate() {
            for (index, global) in module_body.globals.iter().enumerate() {
                let Some(slice) = global.contents.as_ref().and_then(|use_| use_.slice.as_ref()) else {
                    continue;
                };
                globals.push(GlobalBytes {
                    rank: rank.rank,
                    module,
                    index,
                    symbol: global.symbol.clone(),
                    nbytes: global.nbytes,
                    payload_id: slice.payload_id.clone(),
                    slice_offset: slice.offset,
                    bytes: read_payload_range(
                        metadata,
                        verified,
                        &slice.payload_id,
                        slice.offset,
                        slice.nbytes,
                    ),
                });
            }
        }
    }
    (initializers, globals)
}

#[test]
#[ignore = "requires FLARE_ACTUAL_W4A16_FIXTURE with signed TP1/TP2 actual W4A16 cores"]
fn actual_w4a16_combine_preserves_all_initializers_and_globals() {
    let root = fixture_root().expect("FLARE_ACTUAL_W4A16_FIXTURE must point at the fixture");
    let profiles = load_profiles(&root.join("publisher.json"));
    let key = signing_key(&root.join("signing-key.der"));
    let tp1 = root.join("tp1");
    let tp2 = root.join("tp2");

    let tp1_metadata = reader::Metadata::open(&tp1).unwrap();
    let tp1_selection = tp_selection(&tp1_metadata, &profiles, 1);
    let tp1_verified = protection::verify(&tp1_metadata.envelope, &profiles, tp1_selection).unwrap();
    let tp2_metadata = reader::Metadata::open(&tp2).unwrap();
    let tp2_selection = tp_selection(&tp2_metadata, &profiles, 2);
    let tp2_verified =
        protection::verify(&tp2_metadata.envelope, &profiles, tp2_selection).unwrap();
    let mut expected_reusable_weight_bytes = 0u64;
    let mut expected_exception_bytes = 0u64;
    let mut has_piece_metadata = false;
    for rank in tp2_verified.ranks() {
        if let Some(catalog) = prepared_weights::catalog(rank).unwrap() {
            for view in &catalog.views {
                if view.sources.iter().any(|source| !source.pieces.is_empty()) {
                    expected_reusable_weight_bytes = expected_reusable_weight_bytes.saturating_add(view.nbytes);
                } else {
                    expected_exception_bytes = expected_exception_bytes.saturating_add(view.nbytes);
                }
                has_piece_metadata |= view.sources.iter().any(|source|
                    !source.pieces.is_empty() || !source.piece_encoding.is_empty());
            }
        }
    }
    let (expected_initializers, expected_globals) =
        initializer_snapshot(&tp2_metadata, &tp2_verified);
    assert!(
        !expected_initializers.is_empty() || !expected_globals.is_empty(),
        "fixture contains no rank bytes to compare"
    );
    assert!(
        has_piece_metadata,
        "fixture does not contain actual piece metadata"
    );

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let scratch = std::env::temp_dir().join(format!(
        "flare-actual-w4a16-{}-{unique}",
        std::process::id()
    ));
    fs::create_dir_all(&scratch).unwrap();
    let output = scratch.join("combined");
    let accepted = profiles[0].namespace.clone();
    let combine_started = Instant::now();
    conversion::combine(&conversion::CombineRequest {
        sources: &[tp1, tp2],
        output: &output,
        publishers: &profiles,
        accepted: &accepted,
        target_accepted: &accepted,
        target_publisher: None,
        key: &key,
    })
    .unwrap();
    let combine_elapsed = combine_started.elapsed();

    let combined_metadata = reader::Metadata::open(&output).unwrap();
    let combined_selection = tp_selection(&combined_metadata, &profiles, 2);
    let combined_verified =
        protection::verify(&combined_metadata.envelope, &profiles, combined_selection).unwrap();
    for expected in tp2_verified.ranks() {
        let actual = combined_verified
            .ranks()
            .iter()
            .find(|rank| rank.rank == expected.rank)
            .expect("combined TP2 rank missing");
        assert_eq!(
            prepared_weights::catalog(expected).unwrap(),
            prepared_weights::catalog(actual).unwrap(),
            "rank {} prepared weight catalog changed",
            expected.rank
        );
    }
    let combined_payloads = payload_snapshot(&combined_metadata);
    let combined_manifest = manifest_bytes(&combined_metadata);
    let executable_ids: BTreeSet<Vec<u8>> = combined_verified
        .ranks()
        .iter()
        .flat_map(|rank| &rank.resources)
        .filter_map(|resource| {
            resource
                .stored
                .as_ref()
                .map(|slice| slice.payload_id.clone())
        })
        .collect();
    let packed = scratch.join("combined-packed.safetensors");
    let unpacked = scratch.join("combined-unpacked");
    conversion::pack(&output, &packed).unwrap();
    conversion::unpack(&packed, &unpacked).unwrap();
    let packed_metadata = reader::Metadata::open(&packed).unwrap();
    let packed_selection = tp_selection(&packed_metadata, &profiles, 2);
    let _packed_verified =
        protection::verify(&packed_metadata.envelope, &profiles, packed_selection).unwrap();
    let unpacked_metadata = reader::Metadata::open(&unpacked).unwrap();
    let unpacked_selection = tp_selection(&unpacked_metadata, &profiles, 2);
    let unpacked_verified =
        protection::verify(&unpacked_metadata.envelope, &profiles, unpacked_selection).unwrap();
    assert_eq!(combined_metadata.envelope, packed_metadata.envelope);
    assert_eq!(combined_metadata.envelope, unpacked_metadata.envelope);
    let packed_manifest = manifest_bytes(&packed_metadata);
    let unpacked_manifest = manifest_bytes(&unpacked_metadata);
    assert_eq!(combined_manifest, packed_manifest);
    assert_eq!(combined_manifest, unpacked_manifest);
    let packed_payloads = payload_snapshot(&packed_metadata);
    let unpacked_payloads = payload_snapshot(&unpacked_metadata);
    assert_eq!(combined_payloads, packed_payloads);
    assert_eq!(combined_payloads, unpacked_payloads);
    for id in &executable_ids {
        assert_eq!(
            combined_payloads.get(id),
            unpacked_payloads.get(id),
            "compressed executable bytes changed across pack/unpack"
        );
    }
    let combined_derivations = combined_metadata
        .derivations
        .as_ref()
        .map(prost::Message::encode_to_vec);
    assert_eq!(
        combined_derivations,
        packed_metadata
            .derivations
            .as_ref()
            .map(prost::Message::encode_to_vec)
    );
    assert_eq!(
        combined_derivations,
        unpacked_metadata
            .derivations
            .as_ref()
            .map(prost::Message::encode_to_vec)
    );
    let unpacked_recipes = derivation::from_wire(
        unpacked_metadata.derivations.as_ref().unwrap(),
        unpacked_verified.catalog(),
        &derivation::Limits::default(),
    )
    .unwrap();
    let unpacked_targets: BTreeSet<Vec<u8>> = unpacked_recipes
        .iter()
        .map(|recipe| recipe.target.as_bytes().to_vec())
        .collect();
    let unpacked_locations: BTreeSet<Vec<u8>> = unpacked_metadata
        .locations
        .locations
        .iter()
        .map(|location| location.payload_id.clone())
        .collect();
    for recipe in &unpacked_recipes {
        assert!(!unpacked_locations.contains(&recipe.target.as_bytes().to_vec()));
        for operation in &recipe.operations {
            let derivation::Operation::Copy(copy) = operation else {
                continue;
            };
            if !unpacked_targets.contains(&copy.source.as_bytes().to_vec()) {
                assert!(
                    unpacked_locations.contains(&copy.source.as_bytes().to_vec()),
                    "derivation source is missing from the round-tripped closure"
                );
            }
        }
    }
    let combined_metadata = unpacked_metadata;
    let combined_selection = unpacked_selection;
    let combined_verified = unpacked_verified;

    let limits = derivation::Limits::default();
    let recipes = derivation::from_wire(
        combined_metadata.derivations.as_ref().unwrap(),
        combined_verified.catalog(),
        &limits,
    )
    .unwrap();
    let target_ids_owned: BTreeSet<PayloadId> =
        recipes.iter().map(|recipe| recipe.target).collect();
    let source_ids: BTreeSet<PayloadId> = recipes
        .iter()
        .flat_map(|recipe| &recipe.operations)
        .filter_map(|operation| match operation {
            derivation::Operation::Copy(copy) if !target_ids_owned.contains(&copy.source) => {
                Some(copy.source)
            }
            _ => None,
        })
        .collect();
    let source_files = combined_metadata
        .payloads(
            combined_verified.catalog(),
            source_ids.iter().map(|id| id.as_bytes().as_slice()),
        )
        .unwrap();
    let mut materialize_sources: BTreeMap<PayloadId, container::Slice> = source_ids
        .iter()
        .map(|id| (*id, source_files.get(id.as_bytes()).unwrap().clone()))
        .collect();
    let materialize_dir = scratch.join("baseline-materialize");
    fs::create_dir_all(&materialize_dir).unwrap();
    let order = derivation::order(&recipes, &limits).unwrap();
    let mut materialize_runs = 0usize;
    let mut materialize_run_bytes = 0u64;
    let mut materialize_calls = 0usize;
    let mut materialize_targets = 0usize;
    let mut materialize_run_sizes: BTreeMap<u64, usize> = BTreeMap::new();
    let mut coalesced_groups = 0usize;
    let mut verify_targets = 0usize;
    let mut materialize_only = Duration::ZERO;
    let mut verify_only = Duration::ZERO;
    for position in order {
        let recipe = &recipes[position];
        let runs = derivation::destination_runs(recipe, &limits).unwrap();
        materialize_runs += runs.len();
        let mut previous: Option<(u64, Option<(PayloadId, u64)>)> = None;
        for run in &runs {
            materialize_run_bytes = materialize_run_bytes.saturating_add(run.length);
            materialize_calls += (run.length as usize).div_ceil(limits.chunk).max(1);
            *materialize_run_sizes.entry(run.length).or_default() += 1;
            let contiguous = previous.is_some_and(|(destination_end, source_end)| {
                destination_end == run.destination
                    && match (source_end, run.source) {
                        (Some((payload, source_offset)), Some((next_payload, next_offset))) => {
                            payload == next_payload && source_offset == next_offset
                        }
                        (None, None) => true,
                        _ => false,
                    }
            });
            if !contiguous {
                coalesced_groups += 1;
            }
            previous = Some((
                run.destination.saturating_add(run.length),
                run.source
                    .map(|(payload, offset)| (payload, offset.saturating_add(run.length))),
            ));
        }
        let path = materialize_dir.join(format!("target-{position}.bin"));
        let started = Instant::now();
        derivation::materialize(&path, recipe, &materialize_sources, &limits).unwrap();
        materialize_only += started.elapsed();
        assert_eq!(fs::metadata(&path).unwrap().len(), recipe.length);
        let file = Arc::new(fs::File::open(&path).unwrap());
        materialize_sources.insert(
            recipe.target,
            container::Slice::new(file, 0, recipe.length).unwrap(),
        );
        let target = materialize_sources.get(&recipe.target).unwrap().clone();
        let started = Instant::now();
        derivation::verify_output(&target, recipe, &materialize_sources, &limits).unwrap();
        verify_only += started.elapsed();
        verify_targets += 1;
        materialize_targets += 1;
    }
    let materialize_elapsed = materialize_only;
    let verify_elapsed = verify_only;
    println!(
        "actual W4A16 baseline: combine_elapsed_us={} materialize_elapsed_us={} verify_elapsed_us={} targets={}/{} runs={} run_bytes={} read_write_call_pairs={} coalesced_groups={} run_sizes={:?}",
        combine_elapsed.as_micros(),
        materialize_elapsed.as_micros(),
        verify_elapsed.as_micros(),
        materialize_targets,
        verify_targets,
        materialize_runs,
        materialize_run_bytes,
        materialize_calls,
        coalesced_groups,
        materialize_run_sizes
    );
    let _ = fs::remove_dir_all(&materialize_dir);

    let derivations = combined_metadata.derivations.as_ref();
    let target_ids: BTreeSet<&[u8]> = derivations
        .map(|set| {
            set.derivations
                .iter()
                .map(|recipe| recipe.target_payload_id.as_slice())
                .collect()
        })
        .unwrap_or_default();
    assert!(
        !target_ids.is_empty(),
        "combine produced no derived target payloads"
    );
    for target in &target_ids {
        assert!(
            !combined_metadata
                .locations
                .locations
                .iter()
                .any(|location| location.payload_id == *target),
            "derived target payload was physically installed"
        );
    }
    let mut canonical_source_ids = BTreeSet::new();
    let mut canonical_copy_bytes = 0u64;
    if let Some(set) = derivations {
        for recipe in &set.derivations {
            for operation in &recipe.operations {
                let wire::derivation_operation::Operation::Copy(copy) = operation
                    .operation
                    .as_ref()
                    .unwrap()
                else {
                    continue;
                };
                if tp1_verified.catalog().length(&copy.source_payload_id).is_ok() {
                    canonical_source_ids.insert(copy.source_payload_id.clone());
                    canonical_copy_bytes = canonical_copy_bytes.saturating_add(copy.contiguous_bytes);
                }
            }
        }
    }
    assert!(
        !canonical_source_ids.is_empty(),
        "no recipe reused a canonical TP1 payload"
    );

    let mut derived_initializer_bytes = 0u64;
    for rank in combined_verified.ranks() {
        for initializer in &rank.initializers {
            if let Some(slice) = initializer.source.as_ref().and_then(|use_| use_.slice.as_ref())
                && target_ids.contains(slice.payload_id.as_slice())
            {
                derived_initializer_bytes = derived_initializer_bytes.saturating_add(slice.nbytes);
            }
        }
    }
    assert!(
        derived_initializer_bytes > 0,
        "combine did not derive any initializer bytes"
    );
    let mut exception_payloads = 0usize;
    let mut exception_bytes = 0u64;
    let mut exception_sizes = Vec::new();
    for location in &combined_metadata.locations.locations {
        if !location.file.contains("exceptions") {
            continue;
        }
        let files = combined_metadata
            .payloads(combined_verified.catalog(), [location.payload_id.as_slice()])
            .unwrap();
        let payload = files.get(&location.payload_id).unwrap();
        exception_payloads += 1;
        exception_bytes = exception_bytes.saturating_add(payload.len());
        exception_sizes.push(payload.len());
    }
    assert_eq!(
        canonical_copy_bytes, expected_reusable_weight_bytes,
        "not all encoded weight-owner bytes were reused"
    );
    assert_eq!(
        derived_initializer_bytes,
        expected_reusable_weight_bytes.saturating_add(expected_exception_bytes),
        "derived initializer bytes differ from the prepared view total"
    );
    assert_eq!(
        exception_bytes, expected_exception_bytes,
        "exception payloads must contain only non-weight view bytes"
    );
    println!(
        "actual W4A16: targets={} canonical_sources={} canonical_copy_bytes={}/{} derived_initializer_bytes={} exception_payloads={} exception_bytes={}/{} exception_sizes={:?}",
        target_ids.len(),
        canonical_source_ids.len(),
        canonical_copy_bytes,
        expected_reusable_weight_bytes,
        derived_initializer_bytes,
        exception_payloads,
        exception_bytes,
        expected_exception_bytes,
        exception_sizes
    );

    let store = cache::Root::open(scratch.join("cache")).unwrap();
    let core = cache::CoreSource::open(&unpacked, &profiles, combined_selection).unwrap();
    let entry = store.install_core(core, "actual-w4a16", "entry").unwrap();
    let installed_metadata = reader::Metadata::open(entry.directory()).unwrap();
    let installed_selection = tp_selection(&installed_metadata, &profiles, 2);
    let installed_verified =
        protection::verify(&installed_metadata.envelope, &profiles, installed_selection).unwrap();
    let (installed_initializers, installed_globals) =
        initializer_snapshot(&installed_metadata, &installed_verified);
    assert_eq!(
        installed_initializers.len(),
        expected_initializers.len(),
        "combined core initializer count differs"
    );
    assert_eq!(
        installed_globals.len(),
        expected_globals.len(),
        "combined core module-global count differs"
    );

    for expected in expected_initializers {
        let actual = installed_initializers
            .iter()
            .find(|actual| actual.rank == expected.rank && actual.index == expected.index)
            .unwrap_or_else(|| panic!("missing rank {} initializer {}", expected.rank, expected.index));
        assert_eq!(actual.allocation, expected.allocation);
        assert_eq!(actual.destination_offset, expected.destination_offset);
        assert_eq!(actual.payload_id, expected.payload_id);
        assert_eq!(actual.slice_offset, expected.slice_offset);
        assert_eq!(actual.nbytes, expected.nbytes);
        assert_eq!(
            actual.bytes, expected.bytes,
            "rank {} initializer {} differs",
            expected.rank, expected.index
        );
    }
    for expected in expected_globals {
        let actual = installed_globals
            .iter()
            .find(|actual| {
                actual.rank == expected.rank
                    && actual.module == expected.module
                    && actual.index == expected.index
            })
            .unwrap_or_else(|| panic!("missing rank {} global {}", expected.rank, expected.index));
        assert_eq!(actual.symbol, expected.symbol);
        assert_eq!(actual.nbytes, expected.nbytes);
        assert_eq!(actual.payload_id, expected.payload_id);
        assert_eq!(actual.slice_offset, expected.slice_offset);
        assert_eq!(
            actual.bytes, expected.bytes,
            "rank {} module {} global {} differs",
            expected.rank, expected.module, expected.index
        );
    }

    drop(entry);
    let _ = fs::remove_dir_all(scratch);
}
