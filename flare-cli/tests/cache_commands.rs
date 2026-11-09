/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Integration tests for `flare models` against a real, CPU-prepared core fixture.
//!
//! Every test builds a signed payload-format core with the native export API,
//! installs it into a temporary model cache, and drives the actual `flare` binary.
//! Nothing touches the user's cache or any GPU.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use flare_helios::core::{
    PayloadId, cache, container, export, protection, reader, schema::wire,
};
use flare_helios::runtime_core::Publisher;
use ring::signature::{Ed25519KeyPair, KeyPair};

fn flare(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_flare"))
        .args(arguments)
        .output()
        .unwrap()
}

struct Fixture {
    _scratch: tempfile::TempDir,
    cache: PathBuf,
    model: String,
    publisher: Publisher,
}

/// Publish one deployment core and install it as one cache entry.
fn install(scratch: &Path, core_id: u8, payload: &[u8], entry_suffix: &str) -> Fixture {
    let source = scratch.join(format!("source-{entry_suffix}"));
    std::fs::create_dir_all(&source).unwrap();
    let raw = scratch.join(format!("weights-{entry_suffix}.bin"));
    std::fs::write(&raw, payload).unwrap();
    let graph = scratch.join("graph.bin");
    std::fs::write(&graph, b"graph-bytes").unwrap();

    let id = PayloadId::of(payload);
    let pkcs8 = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new()).unwrap();
    let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
    let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
    let publisher = Publisher {
        name: "Cache Test".into(),
        namespace: "cachetest".into(),
        public_key,
        development: false,
    };
    let runtime = flare_helios::runtime_core::wire::ServingContract {
        model_id: "cache-fixture".into(),
        identity: "cache-fixture".into(),
        max_sequence_tokens: 8,
        prefill_chunk_size: 1,
        pages: 1,
        state_rows: 1,
        page_tokens: 1,
        model_contract: Some(flare_helios::runtime_core::wire::ModelContract {
            vocabulary: 8,
            graphs: vec![flare_helios::runtime_core::wire::GraphContract {
                graph: "decode".into(),
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let rank = wire::RankBody {
        rank: 0,
        deployment_id: "sm120-tp1".into(),
        context: Some(Default::default()),
        allocations: vec![wire::Allocation {
            name: "device-0".into(),
            nbytes: payload.len() as u64,
            alignment: 4096,
            ..Default::default()
        }],
        initializers: vec![wire::Initializer {
            allocation: "device-0".into(),
            destination_offset: 0,
            source: Some(wire::PayloadUse {
                slice: Some(wire::PayloadSlice {
                    payload_id: id.as_bytes().to_vec(),
                    offset: 0,
                    nbytes: payload.len() as u64,
                }),
                role: wire::PayloadRole::Weights as i32,
            }),
        }],
        graphs: vec![wire::Graph {
            name: "decode".into(),
            resource: 0,
            node_count: 1,
            ..Default::default()
        }],
        ..Default::default()
    };
    let request = export::PublishRequest {
        core: wire::CoreBody {
            model_identity: "namespaced/model".into(),
            hardware_identity: "sm120".into(),
            frontend: Some(flare_helios::runtime_core::wire::FrontendSettings::default()),
            sampling_defaults: Some(
                flare_helios::runtime_core::wire::DefaultSamplingParameters::default(),
            ),
            publisher: Some(wire::PublisherMetadata {
                namespace: "cachetest".into(),
                official_source: "https://huggingface.co/cachetest/model".into(),
                ..Default::default()
            }),
            ..Default::default()
        },
        payloads: vec![],
        deployments: vec![export::PublishDeployment {
            payloads: vec![],
            body: wire::DeploymentBody {
                deployment_id: "sm120-tp1".into(),
                architecture: vec![12, 0],
                tensor_parallel: 1,
                contract: Some(runtime),
                ..Default::default()
            },
            ranks: vec![export::PublishRank {
                body: rank,
                payloads: vec![export::PublishPayload {
                    role: wire::PayloadRole::Weights,
                    relative: "weights/layer-000.safetensors".into(),
                    tensor: Some(container::COMPANION_TENSOR.into()),
                    source: container::Slice::new(
                        std::sync::Arc::new(std::fs::File::open(&raw).unwrap()),
                        0,
                        payload.len() as u64,
                    )
                    .unwrap(),
                    id,
                }],
                resources: vec![export::PublishResource {
                    kind: wire::ExecutableKind::Graph,
                    path: graph,
                    nbytes: 11,
                }],
            }],
        }],
        core_id: [core_id; 32],
        public_key,
        accepted: "cachetest".into(),
        no_weights: false,
    };
    export::publish(&request, &key, export::Destination::Directory(&source)).unwrap();

    let cache = scratch.join("cache");
    let metadata = reader::Metadata::open(&source).unwrap();
    let verified = protection::verify(
        &metadata.envelope,
        &[publisher.clone()],
        protection::Selection::Deployment {
            deployment: 0,
            rank: None,
        },
    )
    .unwrap();
    let model = cache::model_key("namespaced/model").unwrap();
    let entry = format!("version-{entry_suffix}");
    let store = cache::Root::open(&cache).unwrap();
    let source = cache::CoreSource::new(metadata, verified).unwrap();
    let entry = store.install_core(source, &model, &entry).unwrap();
    store.publish_entry(&entry).unwrap();
    Fixture {
        _scratch: tempfile::tempdir().unwrap(),
        cache,
        model,
        publisher,
    }
}

fn install_pair() -> (tempfile::TempDir, Fixture, String) {
    let scratch = tempfile::tempdir().unwrap();
    let first = install(scratch.path(), 1, b"first payload bytes", "one");
    let _second = install(scratch.path(), 2, b"second payload bytes", "two");
    let cache = first.cache.clone();
    let model = first.model.clone();
    let fixture = Fixture {
        _scratch: tempfile::tempdir().unwrap(),
        cache,
        model,
        publisher: first.publisher.clone(),
    };
    (scratch, fixture, "version-two".into())
}

#[test]
fn list_reports_default_deployment_and_unverified_metadata() {
    let (scratch, fixture, _) = install_pair();
    let output = flare(&[
        "models",
        "list",
        "--json",
        "--core-cache",
        fixture.cache.to_str().unwrap(),
    ]);
    assert!(
        output.status.success(),
        "{:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    let listing: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let models = listing["models"].as_array().unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0]["model"], "namespaced/model");
    assert_eq!(models[0]["model_key"], fixture.model);
    assert_eq!(models[0]["default"], "version-two");
    let versions = models[0]["versions"].as_array().unwrap();
    assert_eq!(versions.len(), 2);
    for version in versions {
        assert_eq!(version["metadata"], "unverified");
        assert_eq!(version["model"], "namespaced/model");
        assert_eq!(version["form"], "payload");
        assert_eq!(version["deployments"][0]["tensor_parallel"], 1);
        assert_eq!(version["deployments"][0]["id"], "sm120-tp1");
        assert_eq!(version["deployments"][0]["ready"], true);
        // Physical usage is reported separately from logical size, and a shared
        // entry never presents its logical bytes as reclaimable.
        assert!(
            version["disk_physical_bytes"].is_u64() || version["disk_physical_bytes"].is_null()
        );
        assert!(version["logical_bytes"].as_u64().unwrap() > 0);
        if version["shared_objects"].as_u64().unwrap() > 0 {
            assert!(version["shared_logical_bytes"].as_u64().unwrap() > 0);
            // Shared bytes are never presented as reclaimable: the reclaimable
            // figure cannot exceed what is physically held by unshared inodes.
            if let (Some(reclaimable), Some(physical), Some(shared)) = (
                version["reclaimable_bytes"].as_u64(),
                version["disk_physical_bytes"].as_u64(),
                version["shared_physical_bytes"].as_u64(),
            ) {
                assert!(reclaimable <= physical.saturating_sub(shared), "{version}");
            }
        }
    }
    let _ = scratch;
}

#[test]
fn select_changes_default_and_remove_requires_clear_default() {
    let (scratch, fixture, second) = install_pair();
    let cache = fixture.cache.to_str().unwrap();
    let spec = "namespaced/model@version-one".to_string();
    let output = flare(&["models", "select", &spec, "--core-cache", cache]);
    assert!(
        output.status.success(),
        "{:?}",
        String::from_utf8_lossy(&output.stderr)
    );

    let listing = flare(&["models", "list", "--json", "--core-cache", cache]);
    let listing: serde_json::Value = serde_json::from_slice(&listing.stdout).unwrap();
    assert_eq!(listing["models"][0]["default"], "version-one");

    let output = flare(&[
        "models",
        "remove",
        &format!("namespaced/model@{second}"),
        "--core-cache",
        cache,
    ]);
    assert!(
        output.status.success(),
        "{:?}",
        String::from_utf8_lossy(&output.stderr)
    );

    let selected = "namespaced/model@version-one".to_string();
    let output = flare(&["models", "remove", &selected, "--core-cache", cache]);
    assert!(!output.status.success(), "selected default was removed");
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(
        error.contains("default") || error.contains("clear"),
        "{error}"
    );

    let output = flare(&[
        "models",
        "remove",
        &selected,
        "--clear-default",
        "--core-cache",
        cache,
    ]);
    assert!(
        output.status.success(),
        "{:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    let listing = flare(&["models", "list", "--json", "--core-cache", cache]);
    let listing: serde_json::Value = serde_json::from_slice(&listing.stdout).unwrap();
    assert!(
        listing["models"]
            .as_array()
            .unwrap()
            .iter()
            .all(|model| model["versions"].as_array().unwrap().is_empty())
    );
    let _ = scratch;
}

#[test]
fn remove_refuses_a_leased_entry_until_the_lease_ends() {
    let (scratch, fixture, _) = install_pair();
    let cache_path = fixture.cache.to_str().unwrap();
    let store = cache::Root::open(&fixture.cache).unwrap();
    // The default entry is refused for being the default; lease the other one so
    // the refusal under test is the lease, not the default.
    let lease = store.entry(&fixture.model, "version-one").unwrap();
    let spec = "namespaced/model@version-one".to_string();
    let output = flare(&["models", "remove", &spec, "--core-cache", cache_path]);
    assert!(!output.status.success(), "leased entry was removed");
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("leased"), "{error}");
    drop(lease);
    let output = flare(&["models", "remove", &spec, "--core-cache", cache_path]);
    assert!(
        output.status.success(),
        "{:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    let _ = scratch;
}

#[test]
fn shared_objects_survive_gc_and_external_links_are_not_reclaimed() {
    let (scratch, fixture, _) = install_pair();
    let cache_path = fixture.cache.to_str().unwrap();
    // Two entries share one payload object; an unrelated abandoned staging tree exists.
    let abandoned = fixture.cache.join("staging/trees/abandoned-test");
    std::fs::create_dir_all(&abandoned).unwrap();
    std::fs::write(abandoned.join("model.safetensors"), b"partial").unwrap();

    let blob = std::fs::read_dir(fixture.cache.join("blobs"))
        .unwrap()
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .find(|path| path.is_file())
        .expect("a payload object");
    let external = scratch.path().join("external-link.safetensors");
    std::fs::hard_link(&blob, &external).unwrap();

    let output = flare(&["models", "gc", "--json", "--core-cache", cache_path]);
    assert!(
        output.status.success(),
        "{:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["staging_trees"], 1, "{report}");
    assert_eq!(report["payload_objects"], 0, "{report}");
    assert!(!abandoned.exists());
    assert!(blob.exists(), "shared object was reclaimed");
    assert!(external.exists(), "externally linked object was reclaimed");

    std::fs::remove_file(&external).unwrap();
    let listing = flare(&["models", "list", "--json", "--core-cache", cache_path]);
    let listing: serde_json::Value = serde_json::from_slice(&listing.stdout).unwrap();
    let versions = listing["models"][0]["versions"].as_array().unwrap();
    assert!(
        versions
            .iter()
            .all(|version| version["shared_objects"].as_u64().unwrap() >= 1)
    );
    let _ = scratch;
}

#[test]
fn missing_version_removal_fails_without_moving_the_default() {
    let (scratch, fixture, _) = install_pair();
    let cache = fixture.cache.to_str().unwrap();
    let listing = flare(&["models", "list", "--json", "--core-cache", cache]);
    let listing: serde_json::Value = serde_json::from_slice(&listing.stdout).unwrap();
    assert_eq!(listing["models"][0]["default"], "version-two");

    let before = listing["models"][0]["versions"].as_array().unwrap().len();
    let output = flare(&[
        "models",
        "remove",
        "namespaced/model@version-missing",
        "--core-cache",
        cache,
    ]);
    assert!(!output.status.success(), "a missing version was removed");
    let error = String::from_utf8_lossy(&output.stderr).to_lowercase();
    assert!(
        error.contains("missing") || error.contains("not published") || error.contains("version"),
        "{error}"
    );
    let after = flare(&["models", "list", "--json", "--core-cache", cache]);
    let after: serde_json::Value = serde_json::from_slice(&after.stdout).unwrap();
    assert_eq!(after["models"][0]["default"], "version-two");
    assert_eq!(
        after["models"][0]["versions"].as_array().unwrap().len(),
        before
    );
    let _ = scratch;
}

#[test]
fn removing_a_non_default_version_leaves_the_selected_default_in_place() {
    let (scratch, fixture, _) = install_pair();
    let cache = fixture.cache.to_str().unwrap();
    let output = flare(&[
        "models",
        "select",
        "namespaced/model@version-two",
        "--core-cache",
        cache,
    ]);
    assert!(
        output.status.success(),
        "{:?}",
        String::from_utf8_lossy(&output.stderr)
    );

    let output = flare(&[
        "models",
        "remove",
        "namespaced/model@version-one",
        "--core-cache",
        cache,
    ]);
    assert!(
        output.status.success(),
        "{:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    let listing = flare(&["models", "list", "--json", "--core-cache", cache]);
    let listing: serde_json::Value = serde_json::from_slice(&listing.stdout).unwrap();
    assert_eq!(listing["models"][0]["default"], "version-two");
    let versions = listing["models"][0]["versions"].as_array().unwrap();
    assert_eq!(versions.len(), 1);
    assert_eq!(versions[0]["version"], "version-two");
    assert_eq!(versions[0]["default"], true);
    let _ = scratch;
}

#[test]
fn inspect_aliases_accept_versions_and_core_paths() {
    let (scratch, fixture, _) = install_pair();
    let cache = fixture.cache.to_str().unwrap();
    let store = cache::Root::open(&fixture.cache).unwrap();
    let entry = store.entry(&fixture.model, "version-two").unwrap();
    let primary = store.primary(&entry).unwrap();
    let literal = "literal@version.safetensors";
    std::fs::copy(&primary, scratch.path().join(literal)).unwrap();
    let mut documents = Vec::new();
    for nested in [false, true] {
        for target in ["namespaced/model", "namespaced/model@version-two", primary.to_str().unwrap(), literal] {
            let mut arguments = if nested {
                vec!["models", "--core-cache", cache, "inspect"]
            } else {
                vec!["inspect", "--core-cache", cache]
            };
            arguments.extend([target, "--json", "--unverified"]);
            let output = Command::new(env!("CARGO_BIN_EXE_flare"))
                .args(&arguments).current_dir(scratch.path()).output().unwrap();
            assert!(output.status.success(), "{arguments:?}: {}", String::from_utf8_lossy(&output.stderr));
            let mut document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(document["core"]["signatures_verified"], false);
            assert!(document.to_string().contains("sm120-tp1"));
            if target == literal {
                assert_eq!(document["core"]["path"], literal);
            }
            document["core"].as_object_mut().unwrap().remove("path");
            documents.push(document);
        }
    }
    for document in &documents[1..] {
        assert_eq!(document, &documents[0]);
    }
}
