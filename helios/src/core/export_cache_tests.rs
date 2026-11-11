/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use crate::core::cache::{PayloadObject, Root};

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let index = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("flare-export-cache-{}-{index}", std::process::id()));
        std::fs::create_dir(&path).unwrap();
        std::fs::create_dir(path.join("cache")).unwrap();
        Self(path)
    }

    fn root(&self) -> Root {
        Root::open(&self.0.join("cache")).unwrap()
    }

    fn payload(&self, name: &str, bytes: &[u8], role: wire::PayloadRole) -> PublishPayload {
        let path = self.0.join(name);
        std::fs::write(&path, bytes).unwrap();
        PublishPayload {
            role,
            relative: format!("weights/{name}.safetensors"),
            source: pin_file(&path, bytes.len() as u64).unwrap(),
            id: PayloadId::of(bytes),
            tensor: None,
        }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn corrupt(object: &PayloadObject) {
    let path = object.path();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).unwrap();
    }
    #[cfg(windows)]
    {
        let mut permissions = std::fs::metadata(path).unwrap().permissions();
        permissions.set_readonly(false);
        std::fs::set_permissions(path, permissions).unwrap();
    }
    use std::io::{Seek, SeekFrom};
    let mut file = OpenOptions::new().write(true).open(path).unwrap();
    file.seek(SeekFrom::Start(object.slice().offset())).unwrap();
    file.write_all(&[0xff]).unwrap();
    file.sync_all().unwrap();
}

fn install(
    payloads: &[PublishPayload],
    weights: Option<&[PayloadObject]>,
    root: &Root,
    staging: &mut crate::core::cache::Staging,
    locations: &mut Vec<wire::PayloadLocation>,
    located: &mut BTreeMap<PayloadId, ()>,
    no_weights: bool,
) -> Result<u64, crate::Error> {
    let progress = crate::startup::StartupProgress::silent();
    let phase = progress.phase("Verify test payloads", None);
    install_phase_payloads(
        payloads,
        weights,
        &mut Destination::Cache(root, staging),
        locations,
        located,
        no_weights,
        &phase,
    )
}

#[test]
fn cache_phase_installs_warm_and_cold_payloads_in_input_order_and_deduplicates_ids() {
    let scratch = Scratch::new();
    let root = scratch.root();
    let mut payloads = Vec::new();
    for index in 0..19 {
        let bytes = vec![index as u8; 37 + index];
        payloads.push(scratch.payload(
            &format!("layer-{index}"),
            &bytes,
            wire::PayloadRole::Weights,
        ));
        if index % 2 == 0 {
            root.capture_payload(bytes.len() as u64, bytes.as_slice())
                .unwrap();
        }
    }
    let duplicate = scratch.payload("duplicate", &[0; 37], wire::PayloadRole::Weights);
    payloads.insert(3, duplicate);
    let mut staging = root.stage("model", "mixed").unwrap();
    let mut locations = Vec::new();
    let mut located = BTreeMap::new();
    assert_eq!(
        install(
            &payloads,
            None,
            &root,
            &mut staging,
            &mut locations,
            &mut located,
            false
        )
        .unwrap(),
        0
    );
    assert_eq!(locations.len(), 19);
    assert_eq!(located.len(), 19);
    assert!(
        !staging
            .directory()
            .join("weights/duplicate.safetensors")
            .exists()
    );
    let expected: Vec<_> = payloads
        .iter()
        .filter(|payload| payload.relative != "weights/duplicate.safetensors")
        .map(|payload| payload.relative.clone())
        .collect();
    assert_eq!(
        locations
            .iter()
            .map(|location| location.file.clone())
            .collect::<Vec<_>>(),
        expected
    );
    for payload in &payloads {
        let cached = root.payload(payload.id).unwrap().unwrap();
        assert_eq!(cached.slice().digest().unwrap(), payload.id);
        assert_eq!(cached.length(), payload.nbytes());
    }
    for location in &locations {
        let path = staging.directory().join(&location.file);
        let container = crate::core::container::Container::open(&path).unwrap();
        let slice = container.tensor(container::COMPANION_TENSOR).unwrap();
        assert_eq!(
            slice.digest().unwrap().as_bytes(),
            location.payload_id.as_slice()
        );
    }
}

#[test]
fn every_duplicate_source_is_verified() {
    for warm in [false, true] {
        let scratch = Scratch::new();
        let root = scratch.root();
        let bytes = b"checkpoint bytes";
        let first = scratch.payload("first", bytes, wire::PayloadRole::Weights);
        let mut duplicate =
            scratch.payload("duplicate", b"incorrect bytes!", wire::PayloadRole::Weights);
        duplicate.id = first.id;
        if warm {
            root.capture_payload(bytes.len() as u64, bytes.as_slice())
                .unwrap();
        }
        let mut staging = root.stage("model", "bad-duplicate").unwrap();
        let mut locations = Vec::new();
        let mut located = BTreeMap::new();
        let error = install(
            &[first, duplicate],
            None,
            &root,
            &mut staging,
            &mut locations,
            &mut located,
            false,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("bytes do not match the assigned identity"),
            "{error}"
        );
        assert!(
            !staging
                .directory()
                .join("weights/duplicate.safetensors")
                .exists()
        );
        assert!(
            locations
                .iter()
                .all(|location| location.file == "weights/first.safetensors")
        );
        let staging_path = staging.directory().to_owned();
        drop(staging);
        assert!(!staging_path.exists());
        assert!(root.versions("model").unwrap_or_default().is_empty());
    }
}

#[test]
fn cache_phase_propagates_real_source_read_failures_without_publishing_an_entry() {
    let scratch = Scratch::new();
    let root = scratch.root();
    let good = scratch.payload("good", b"valid payload", wire::PayloadRole::Weights);
    let broken = scratch.payload("truncated", b"complete payload", wire::PayloadRole::Weights);
    OpenOptions::new()
        .write(true)
        .open(scratch.0.join("truncated"))
        .unwrap()
        .set_len(1)
        .unwrap();
    let mut staging = root.stage("model", "truncated").unwrap();
    let mut locations = Vec::new();
    let error = install(
        &[good, broken],
        None,
        &root,
        &mut staging,
        &mut locations,
        &mut BTreeMap::new(),
        false,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("read payload"), "{error}");
    assert!(
        !staging
            .directory()
            .join("weights/truncated.safetensors")
            .exists()
    );
    assert!(
        locations
            .iter()
            .all(|location| location.file == "weights/good.safetensors")
    );
    let staging_path = staging.directory().to_owned();
    drop(staging);
    assert!(!staging_path.exists());
    assert!(root.versions("model").unwrap_or_default().is_empty());
}

#[test]
fn warm_cache_corruption_is_rejected_for_weight_and_state_roles() {
    for (role, no_weights) in [
        (wire::PayloadRole::Weights, false),
        (wire::PayloadRole::Weights, true),
        (wire::PayloadRole::InitialState, true),
    ] {
        let scratch = Scratch::new();
        let root = scratch.root();
        let bytes = b"unchanged source bytes";
        let payload = scratch.payload("source", bytes, role);
        let object = root
            .capture_payload(bytes.len() as u64, bytes.as_slice())
            .unwrap();
        corrupt(&object);
        let mut staging = root.stage("model", "bad-cache").unwrap();
        let mut locations = Vec::new();
        let error = install(
            &[payload],
            None,
            &root,
            &mut staging,
            &mut locations,
            &mut BTreeMap::new(),
            no_weights,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("cached payload"), "{error}");
        assert!(
            error.contains("bytes do not match their identity"),
            "{error}"
        );
        assert!(locations.is_empty());
        assert_eq!(std::fs::read_dir(staging.directory()).unwrap().count(), 0);
        assert!(root.versions("model").unwrap_or_default().is_empty());
    }
}

#[test]
fn no_weights_requires_cached_weights_but_installs_fresh_state() {
    for role in [wire::PayloadRole::Weights, wire::PayloadRole::TunedWeights] {
        let scratch = Scratch::new();
        let root = scratch.root();
        let payload = scratch.payload("missing", b"missing checkpoint", role);
        let id = payload.id;
        let mut staging = root.stage("model", "missing").unwrap();
        let mut locations = Vec::new();
        let error = install(
            &[payload],
            None,
            &root,
            &mut staging,
            &mut locations,
            &mut BTreeMap::new(),
            true,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("no-weights export is missing cached payload"),
            "{error}"
        );
        assert!(root.payload(id).unwrap().is_none());
        assert!(locations.is_empty());
    }
    let scratch = Scratch::new();
    let root = scratch.root();
    let state = scratch.payload(
        "state",
        b"fresh initial state",
        wire::PayloadRole::InitialState,
    );
    let id = state.id;
    let mut staging = root.stage("model", "state").unwrap();
    let mut locations = Vec::new();
    install(
        &[state],
        None,
        &root,
        &mut staging,
        &mut locations,
        &mut BTreeMap::new(),
        true,
    )
    .unwrap();
    assert_eq!(locations.len(), 1);
    assert_eq!(
        root.payload(id).unwrap().unwrap().slice().digest().unwrap(),
        id
    );
    assert!(staging.directory().join(&locations[0].file).is_file());
}

#[test]
fn explicit_weight_reuse_requires_every_occurrence_to_reference_the_exact_lease() {
    let scratch = Scratch::new();
    let root = scratch.root();
    let bytes = b"leased weight bytes";
    let object = root
        .capture_payload(bytes.len() as u64, bytes.as_slice())
        .unwrap();
    let mut correct = scratch.payload("correct", bytes, wire::PayloadRole::Weights);
    correct.source = object.slice().clone();
    let wrong = scratch.payload("wrong-arc", bytes, wire::PayloadRole::Weights);
    let mut staging = root.stage("model", "wrong-lease").unwrap();
    let mut locations = Vec::new();
    let error = install(
        &[correct, wrong],
        Some(std::slice::from_ref(&object)),
        &root,
        &mut staging,
        &mut locations,
        &mut BTreeMap::new(),
        true,
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("does not reference its leased cache object"),
        "{error}"
    );
    assert!(
        !staging
            .directory()
            .join("weights/wrong-arc.safetensors")
            .exists()
    );
    assert!(
        locations
            .iter()
            .all(|location| location.file == "weights/correct.safetensors")
    );
    let staging_path = staging.directory().to_owned();
    drop(staging);
    assert!(!staging_path.exists());
    assert!(root.versions("model").unwrap_or_default().is_empty());
}

#[test]
fn authorized_weight_reuse_keeps_fresh_state_verification_and_byte_accounting() {
    let scratch = Scratch::new();
    let root = scratch.root();
    let bytes = b"reused checkpoint";
    let object = root
        .capture_payload(bytes.len() as u64, bytes.as_slice())
        .unwrap();
    let mut weight = scratch.payload("weight", bytes, wire::PayloadRole::Weights);
    weight.source = object.slice().clone();
    let state = scratch.payload("state", b"valid state", wire::PayloadRole::InitialState);
    let mut staging = root.stage("model", "reuse").unwrap();
    let mut locations = Vec::new();
    let reused = install(
        &[weight, state],
        Some(std::slice::from_ref(&object)),
        &root,
        &mut staging,
        &mut locations,
        &mut BTreeMap::new(),
        true,
    )
    .unwrap();
    assert_eq!(reused, bytes.len() as u64);
    assert_eq!(locations.len(), 2);
    assert_eq!(
        root.payload(object.id())
            .unwrap()
            .unwrap()
            .slice()
            .digest()
            .unwrap(),
        object.id()
    );

    let mut weight = scratch.payload("reused-again", bytes, wire::PayloadRole::Weights);
    weight.source = object.slice().clone();
    let mut state = scratch.payload(
        "corrupt-state",
        b"modified state",
        wire::PayloadRole::InitialState,
    );
    state.id = PayloadId::of(b"original state");
    let mut staging = root.stage("model", "bad-state").unwrap();
    let mut locations = Vec::new();
    let error = install(
        &[weight, state],
        Some(std::slice::from_ref(&object)),
        &root,
        &mut staging,
        &mut locations,
        &mut BTreeMap::new(),
        true,
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("bytes do not match the assigned identity"),
        "{error}"
    );
    assert!(
        !staging
            .directory()
            .join("weights/corrupt-state.safetensors")
            .exists()
    );
    assert!(
        locations
            .iter()
            .all(|location| location.file == "weights/reused-again.safetensors")
    );
    let staging_path = staging.directory().to_owned();
    drop(staging);
    assert!(!staging_path.exists());
    assert!(root.versions("model").unwrap_or_default().is_empty());
}

#[test]
fn duplicate_source_is_verified_when_an_earlier_phase_already_installed_its_identity() {
    let scratch = Scratch::new();
    let root = scratch.root();
    let first = scratch.payload("first", b"valid checkpoint", wire::PayloadRole::Weights);
    let identity = first.id;
    let mut staging = root.stage("model", "duplicate-phase").unwrap();
    let mut locations = Vec::new();
    let mut located = BTreeMap::new();
    install(
        &[first],
        None,
        &root,
        &mut staging,
        &mut locations,
        &mut located,
        false,
    )
    .unwrap();
    let mut duplicate =
        scratch.payload("duplicate", b"wrong checkpoint", wire::PayloadRole::Weights);
    duplicate.id = identity;
    let error = install(
        &[duplicate],
        None,
        &root,
        &mut staging,
        &mut locations,
        &mut located,
        false,
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("bytes do not match the assigned identity"),
        "{error}"
    );
    assert_eq!(locations.len(), 1);
    assert_eq!(located.len(), 1);
    assert!(
        !staging
            .directory()
            .join("weights/duplicate.safetensors")
            .exists()
    );
    assert_eq!(
        root.payload(identity)
            .unwrap()
            .unwrap()
            .slice()
            .digest()
            .unwrap(),
        identity
    );
    let staging_path = staging.directory().to_owned();
    drop(staging);
    assert!(!staging_path.exists());
    assert!(root.versions("model").unwrap_or_default().is_empty());
}

#[test]
#[ignore = "requires a validated TP16 cache entry and exclusive authorization for payload I/O"]
fn verify_actual_tp16_cached_payloads() {
    use crate::core::{protection, reader::Metadata};
    use std::collections::BTreeSet;

    let entry = PathBuf::from(
        std::env::var_os("FLARE_TEST_TP16_CACHE_ENTRY").expect("set FLARE_TEST_TP16_CACHE_ENTRY"),
    );
    let cache = PathBuf::from(
        std::env::var_os("FLARE_TEST_TP16_CACHE_ROOT").expect("set FLARE_TEST_TP16_CACHE_ROOT"),
    );
    let metadata = Metadata::open(&entry).unwrap();
    let envelope = protection::envelope(&metadata.envelope).unwrap();
    let core = wire::CoreBody::decode(envelope.core_body.as_slice()).unwrap();
    let deployments: Vec<_> = envelope
        .deployments
        .iter()
        .filter_map(|signed| {
            let body = wire::DeploymentBody::decode(signed.body.as_slice()).unwrap();
            (body.tensor_parallel == 16).then_some((signed, body))
        })
        .collect();
    assert_eq!(
        deployments.len(),
        1,
        "select a core with one TP16 deployment"
    );
    let (signed_deployment, deployment) = &deployments[0];
    assert_eq!(deployment.ranks.len(), 16);
    let ranks: Vec<_> = envelope
        .ranks
        .iter()
        .filter(|signed| signed.deployment == signed_deployment.index)
        .map(|signed| {
            let body = wire::RankBody::decode(signed.body.as_slice()).unwrap();
            assert_eq!(body.rank, signed.rank);
            body
        })
        .collect();
    assert_eq!(
        ranks.iter().map(|body| body.rank).collect::<Vec<_>>(),
        (0..16).collect::<Vec<_>>()
    );
    let mut catalog = schema::Catalog::new();
    catalog.insert_all(&core.payloads).unwrap();
    catalog.insert_all(&deployment.payloads).unwrap();
    let mut phases = Vec::new();
    let mut selected_ids = BTreeSet::new();
    let mut roles = BTreeMap::new();
    for rank in &ranks {
        catalog.insert_all(&rank.payloads).unwrap();
        let executables: BTreeSet<_> = rank
            .resources
            .iter()
            .map(|resource| resource.stored.as_ref().unwrap().payload_id.as_slice())
            .collect();
        let data: Vec<_> = rank
            .payloads
            .iter()
            .filter(|payload| !executables.contains(payload.id.as_slice()))
            .collect();
        assert!(!data.is_empty(), "rank {} has no data payloads", rank.rank);
        for use_ in rank
            .initializers
            .iter()
            .filter_map(|initializer| initializer.source.as_ref())
            .chain(
                rank.modules
                    .iter()
                    .flat_map(|module| module.globals.iter())
                    .filter_map(|global| global.contents.as_ref()),
            )
        {
            let identity = use_.slice.as_ref().unwrap().payload_id.clone();
            let role = schema::role_of(use_).unwrap();
            if let Some(previous) = roles.insert(identity, role) {
                assert_eq!(previous, role, "one data payload has multiple signed roles");
            }
        }
        for payload in &data {
            selected_ids.insert(payload.id.clone());
        }
        phases.push(data);
    }
    let files = metadata
        .payloads(&catalog, selected_ids.iter().map(Vec::as_slice))
        .unwrap();
    let locations_by_id: BTreeMap<_, _> = metadata
        .locations
        .locations
        .iter()
        .map(|location| (location.payload_id.as_slice(), location))
        .collect();
    let root = Root::open(&cache).unwrap();
    let mut payloads_by_rank = Vec::new();
    for data in phases {
        let mut payloads = Vec::new();
        for descriptor in data {
            let id = PayloadId::from_bytes(descriptor.id.as_slice().try_into().unwrap());
            let object = root
                .payload(id)
                .unwrap()
                .expect("all selected data payloads must already be cached");
            assert_eq!(object.length(), descriptor.nbytes);
            let role = *roles
                .get(&descriptor.id)
                .expect("data payload must have a signed weight or state use");
            assert!(matches!(
                role,
                wire::PayloadRole::Weights
                    | wire::PayloadRole::TunedWeights
                    | wire::PayloadRole::InitialState
            ));
            let location = locations_by_id[descriptor.id.as_slice()];
            assert!(
                !location.file.is_empty(),
                "cache payloads require their stored companion locations"
            );
            payloads.push(PublishPayload {
                role,
                relative: location.file.clone(),
                source: files.get(&descriptor.id).unwrap().clone(),
                id,
                tensor: None,
            });
        }
        payloads_by_rank.push(payloads);
    }
    let unique_cached_bytes: u64 = selected_ids
        .iter()
        .map(|id| catalog.length(id).unwrap())
        .sum();
    let source_bytes: u64 = payloads_by_rank
        .iter()
        .flatten()
        .map(PublishPayload::nbytes)
        .sum();
    let descriptors: usize = payloads_by_rank.iter().map(Vec::len).sum();
    let entry_name = format!(
        "tp16-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let mut staging = root.stage("payload-verification", &entry_name).unwrap();
    let staging_path = staging.directory().to_owned();
    let mut locations = Vec::new();
    let mut located = BTreeMap::new();
    let started = std::time::Instant::now();
    for payloads in &payloads_by_rank {
        assert_eq!(
            install(
                payloads,
                None,
                &root,
                &mut staging,
                &mut locations,
                &mut located,
                false
            )
            .unwrap(),
            0
        );
    }
    let elapsed = started.elapsed().as_secs_f64();
    assert_eq!(locations.len(), selected_ids.len());
    assert_eq!(located.len(), selected_ids.len());
    drop(staging);
    assert!(!staging_path.exists());
    eprintln!(
        "cache_payload_verification ranks=16 phases=16 descriptors={descriptors} source_bytes={source_bytes} cached_copy_bytes={unique_cached_bytes} elapsed_seconds={elapsed:.6} max_workers={} reuse_weights=false no_weights=false",
        std::thread::available_parallelism()
            .map_or(1, |workers| workers.get())
            .min(MAX_PAYLOAD_INSTALL_WORKERS)
    );
}
