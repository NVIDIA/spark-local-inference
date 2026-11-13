/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Canonical download path for a packed payload core: embedded frontend
//! payloads and located derivation-only dependencies must both resolve.
//!
//! The fixture is a real signed core: the tokenizer lives in the primary, one
//! required weight is derived from a second weight payload that no initializer
//! references, and the core is packed into the first canonical form.

use std::path::{Path, PathBuf};

use flare_helios::core::{
    PayloadId, cache, container, conversion, derivation, export, protection, reader, schema::wire,
};
use flare_helios::runtime_core::Publisher;
use ring::signature::{Ed25519KeyPair, KeyPair};

use ring::digest::{digest, SHA256};

fn sha256(bytes: &[u8]) -> Vec<u8> {
    digest(&SHA256, bytes).as_ref().to_vec()
}

struct Fixture {
    _scratch: tempfile::TempDir,
    packed: PathBuf,
    publisher: Publisher,
    derived_id: PayloadId,
    source_id: PayloadId,
    derived: Vec<u8>,
}

fn publish(scratch: &Path, with_derivation: bool) -> Fixture {
    publish_with_source(scratch, b"canonical weight bytes", with_derivation)
}

fn publish_with_source(scratch: &Path, source: &[u8], with_derivation: bool) -> Fixture {
    let directory = scratch.join("core");
    std::fs::create_dir_all(&directory).unwrap();

    let source = source.repeat(2);
    let derived = source[..source.len() - 5].to_vec();
    let source_id = PayloadId::of(&source);
    let derived_id = PayloadId::of(&derived);
    let tokenizer = b"tokenizer bytes".to_vec();
    let tokenizer_id = PayloadId::of(&tokenizer);

    let source_path = scratch.join("source.bin");
    std::fs::write(&source_path, &source).unwrap();
    let derived_path = scratch.join("derived.bin");
    std::fs::write(&derived_path, &derived).unwrap();
    let tokenizer_path = scratch.join("tokenizer.bin");
    std::fs::write(&tokenizer_path, &tokenizer).unwrap();
    let graph = scratch.join("graph.bin");
    std::fs::write(&graph, b"graph-bytes").unwrap();

    let pkcs8 = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new()).unwrap();
    let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
    let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
    let publisher = Publisher {
        name: "Packed".into(),
        namespace: "packed".into(),
        public_key,
        development: false,
    };
    let payload = |id: PayloadId, path: &Path, nbytes: u64, relative: &str| export::PublishPayload {
        role: wire::PayloadRole::Weights,
        relative: relative.into(),
        tensor: Some(container::COMPANION_TENSOR.into()),
        source: container::Slice::new(
            std::sync::Arc::new(std::fs::File::open(path).unwrap()),
            0,
            nbytes,
        )
        .unwrap(),
        id,
    };
    let runtime = flare_helios::runtime_core::wire::ServingContract {
        model_id: "packed-model".into(),
        identity: "packed-model".into(),
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
    let initializer = |id: PayloadId, allocation: &str, nbytes: u64| wire::Initializer {
        allocation: allocation.into(),
        destination_offset: 0,
        source: Some(wire::PayloadUse {
            slice: Some(wire::PayloadSlice {
                payload_id: id.as_bytes().to_vec(),
                offset: 0,
                nbytes,
            }),
            role: wire::PayloadRole::Weights as i32,
        }),
    };
    let rank = wire::RankBody {
        rank: 0,
        deployment_id: "sm120-tp1".into(),
        context: Some(Default::default()),
        allocations: vec![wire::Allocation {
            name: "device-0".into(),
            nbytes: derived.len() as u64,
            alignment: 4096,
            ..Default::default()
        }],
        // Only the derived target is required. The canonical source weight is
        // located in the core but named by no initializer, and the tokenizer
        // arrives through the core assets, which live in the primary.
        initializers: vec![initializer(derived_id, "device-0", derived.len() as u64)],
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
            model_identity: "packed-model".into(),
            hardware_identity: "sm120".into(),
            frontend: Some(flare_helios::runtime_core::wire::FrontendSettings::default()),
            sampling_defaults: Some(
                flare_helios::runtime_core::wire::DefaultSamplingParameters::default(),
            ),
            publisher: Some(wire::PublisherMetadata {
                namespace: "packed".into(),
                official_source: "https://huggingface.co/packed/model".into(),
                ..Default::default()
            }),
            assets: vec![wire::FrontendAsset {
                name: "tokenizer".into(),
                payload: Some(wire::PayloadUse {
                    slice: Some(wire::PayloadSlice {
                        payload_id: tokenizer_id.as_bytes().to_vec(),
                        offset: 0,
                        nbytes: tokenizer.len() as u64,
                    }),
                    role: wire::PayloadRole::Frontend as i32,
                }),
                sha256: sha256(&tokenizer),
            }],
            ..Default::default()
        },
        // Core-scoped payload: embedded in the primary, no companion path.
        payloads: vec![export::PublishPayload {
            role: wire::PayloadRole::Frontend,
            relative: String::new(),
            tensor: Some("frontend.tokenizer".into()),
            source: container::Slice::new(
                std::sync::Arc::new(std::fs::File::open(&tokenizer_path).unwrap()),
                0,
                tokenizer.len() as u64,
            )
            .unwrap(),
            id: tokenizer_id,
        }],
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
                payloads: vec![
                    payload(source_id, &source_path, source.len() as u64, "weights/source.safetensors"),
                    payload(derived_id, &derived_path, derived.len() as u64, "weights/derived.safetensors"),
                ],
                resources: vec![export::PublishResource {
                    kind: wire::ExecutableKind::Graph,
                    path: graph,
                    nbytes: 11,
                }],
            }],
        }],
        core_id: [41u8; 32],
        public_key,
        accepted: "packed".into(),
        no_weights: false,
    };
    export::publish(&request, &key, export::Destination::Directory(&directory)).unwrap();

    // The derived target is removed from the core and described as a copy of the
    // source payload, which no initializer references.
    if !with_derivation {
        let packed = scratch.join("packed.safetensors");
        conversion::pack(&directory, &packed).unwrap();
        return Fixture {
            _scratch: tempfile::tempdir().unwrap(),
            packed,
            publisher,
            derived_id,
            source_id,
            derived,
        };
    }
    let recipe = derivation::Derivation {
        target: derived_id,
        length: derived.len() as u64,
        operations: vec![derivation::Operation::Copy(derivation::Copy {
            source: source_id,
            source_offset: 0,
            destination_offset: 0,
            contiguous: derived.len() as u64,
            outer: Vec::new(),
        })],
    };
    assert_eq!(
        conversion::emit_derivations(&directory, std::slice::from_ref(&recipe), &derivation::Limits::default())
            .unwrap(),
        1
    );
    let packed = scratch.join("packed.safetensors");
    conversion::pack(&directory, &packed).unwrap();
    Fixture {
        _scratch: tempfile::tempdir().unwrap(),
        packed,
        publisher,
        derived_id,
        source_id,
        derived,
    }
}

fn verify_core(fixture: &Fixture) -> protection::Selection {
    let metadata = reader::Metadata::open(&fixture.packed).unwrap();
    protection::verify(
        &metadata.envelope,
        std::slice::from_ref(&fixture.publisher),
        protection::Selection::Deployment {
            deployment: 0,
            rank: None,
        },
    )
    .unwrap();
    protection::Selection::Deployment {
        deployment: 0,
        rank: None,
    }
}

#[test]
fn packed_download_accepts_embedded_frontend_and_derivation_only_sources() {
    let scratch = tempfile::tempdir().unwrap();
    let fixture = publish(scratch.path(), true);
    let metadata = reader::Metadata::open(&fixture.packed).unwrap();
    let selection = verify_core(&fixture);
    // Only the public install path the download uses: embedded frontend assets and
    // the located derivation source that no initializer names must both resolve.
    let _ = &metadata;

    let cache_root = scratch.path().join("cache");
    let store = cache::Root::open(&cache_root).unwrap();
    let source = cache::CoreSource::open(
        &fixture.packed,
        std::slice::from_ref(&fixture.publisher),
        selection,
    )
    .unwrap();
    let entry = store
        .install_core(source, "packed-model", "entry")
        .expect("packed canonical install");

    // The derived target is materialized from the located source payload.
    let derived_companion = entry
        .directory()
        .join("weights/tuned/sm120-tp1/derived-000.safetensors");
    assert!(derived_companion.is_file(), "{}", derived_companion.display());
    assert_eq!(
        entry
            .payload("weights/tuned/sm120-tp1/derived-000.safetensors")
            .unwrap()
            .len(),
        fixture.derived.len() as u64
    );
    assert!(store.payload(fixture.source_id).unwrap().is_some());
    assert!(store.payload(fixture.derived_id).unwrap().is_some());

    // The frontend stays embedded in the entry primary rather than a companion.
    let primary = container::Container::open(&entry.directory().join("model.safetensors")).unwrap();
    assert!(primary.tensor_names().any(|name| name == "frontend.tokenizer"));
    assert!(!entry.directory().join("frontend.tokenizer").exists());
}

#[test]
fn signed_cache_install_materializes_a_transitive_chain_and_ignores_unselected_hints() {
    let scratch = tempfile::tempdir().unwrap();
    let directory = scratch.path().join("chain-core");
    std::fs::create_dir_all(&directory).unwrap();
    let c = b"transitive chain canonical source".to_vec();
    let b = c[3..19].to_vec();
    let a = b[2..10].to_vec();
    let d = c[6..14].to_vec();
    let c_id = PayloadId::of(&c);
    let b_id = PayloadId::of(&b);
    let a_id = PayloadId::of(&a);
    let d_id = PayloadId::of(&d);
    let paths: Vec<_> = [("c", &c), ("b", &b), ("a", &a), ("d", &d)]
        .iter()
        .map(|(name, bytes)| {
            let path = scratch.path().join(format!("{name}.bin"));
            std::fs::write(&path, bytes).unwrap();
            (path, bytes.len() as u64)
        })
        .collect();
    let graph = scratch.path().join("graph.bin");
    std::fs::write(&graph, b"graph-bytes").unwrap();
    let pkcs8 = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new()).unwrap();
    let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
    let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
    let publisher = Publisher {
        name: "Chain".into(),
        namespace: "chain".into(),
        public_key,
        development: false,
    };
    let payload = |id: PayloadId, path: &Path, nbytes: u64, relative: &str| export::PublishPayload {
        role: wire::PayloadRole::Weights,
        relative: relative.into(),
        tensor: Some(container::COMPANION_TENSOR.into()),
        source: container::Slice::new(
            std::sync::Arc::new(std::fs::File::open(path).unwrap()),
            0,
            nbytes,
        )
        .unwrap(),
        id,
    };
    let contract = flare_helios::runtime_core::wire::ServingContract {
        model_id: "chain-model".into(),
        identity: "chain-model".into(),
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
    let request = export::PublishRequest {
        core: wire::CoreBody {
            model_identity: "chain-model".into(),
            hardware_identity: "sm120".into(),
            frontend: Some(flare_helios::runtime_core::wire::FrontendSettings::default()),
            sampling_defaults: Some(
                flare_helios::runtime_core::wire::DefaultSamplingParameters::default(),
            ),
            publisher: Some(wire::PublisherMetadata {
                namespace: "chain".into(),
                official_source: "https://huggingface.co/chain/model".into(),
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
                contract: Some(contract),
                ..Default::default()
            },
            ranks: vec![export::PublishRank {
                body: wire::RankBody {
                    rank: 0,
                    deployment_id: "sm120-tp1".into(),
                    context: Some(Default::default()),
                    allocations: vec![wire::Allocation {
                        name: "device-0".into(),
                        nbytes: a.len() as u64,
                        alignment: 4096,
                        ..Default::default()
                    }],
                    initializers: vec![wire::Initializer {
                        allocation: "device-0".into(),
                        destination_offset: 0,
                        source: Some(wire::PayloadUse {
                            slice: Some(wire::PayloadSlice {
                                payload_id: a_id.as_bytes().to_vec(),
                                offset: 0,
                                nbytes: a.len() as u64,
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
                },
                payloads: vec![
                    payload(c_id, &paths[0].0, paths[0].1, "weights/c.safetensors"),
                    payload(b_id, &paths[1].0, paths[1].1, "weights/b.safetensors"),
                    payload(a_id, &paths[2].0, paths[2].1, "weights/a.safetensors"),
                    payload(d_id, &paths[3].0, paths[3].1, "weights/d.safetensors"),
                ],
                resources: vec![export::PublishResource {
                    kind: wire::ExecutableKind::Graph,
                    path: graph,
                    nbytes: 11,
                }],
            }],
        }],
        core_id: [51u8; 32],
        public_key,
        accepted: "chain".into(),
        no_weights: false,
    };
    export::publish(&request, &key, export::Destination::Directory(&directory)).unwrap();
    let recipe = |target: PayloadId, source: PayloadId, length: u64, offset: u64| {
        derivation::Derivation {
            target,
            length,
            operations: vec![derivation::Operation::Copy(derivation::Copy {
                source,
                source_offset: offset,
                destination_offset: 0,
                contiguous: length,
                outer: Vec::new(),
            })],
        }
    };
    conversion::emit_derivations(
        &directory,
        &[
            recipe(a_id, b_id, a.len() as u64, 2),
            recipe(b_id, c_id, b.len() as u64, 3),
            recipe(d_id, c_id, d.len() as u64, 6),
        ],
        &derivation::Limits::default(),
    )
    .unwrap();
    let packed = scratch.path().join("chain-packed.safetensors");
    conversion::pack(&directory, &packed).unwrap();
    let metadata = reader::Metadata::open(&packed).unwrap();
    let selection = protection::Selection::Deployment {
        deployment: 0,
        rank: None,
    };
    let _verified = protection::verify(
        &metadata.envelope,
        std::slice::from_ref(&publisher),
        selection,
    )
    .unwrap();
    let store = cache::Root::open(scratch.path().join("chain-cache")).unwrap();
    let source = cache::CoreSource::open(
        &packed,
        std::slice::from_ref(&publisher),
        selection,
    )
    .unwrap();
    let entry = store
        .install_core(source, "chain-model", "entry")
        .unwrap();
    // The initializer identities are signed; the recipes that produce their bytes
    // are unsigned acquisition hints. Read each derived value through the
    // installed payload that the signed initializer binds, not by searching the
    // tuned directory for matching bytes.
    let installed = reader::Metadata::open(entry.directory()).unwrap();
    let verified = protection::verify(
        &installed.envelope,
        std::slice::from_ref(&publisher),
        protection::Selection::Deployment {
            deployment: 0,
            rank: None,
        },
    )
    .unwrap();
    let location = |id: &[u8]| {
        installed
            .locations
            .locations
            .iter()
            .find(|location| location.payload_id == id)
            .map(|location| location.file.clone())
            .expect("installed payload location")
    };
    // The signed initializers bind the derived payload identities; the bytes are
    // then read from the payload the installed entry holds for those identities.
    let signed: std::collections::BTreeSet<Vec<u8>> = verified
        .ranks()
        .iter()
        .flat_map(|rank| rank.initializers.iter())
        .filter_map(|initializer| initializer.source.as_ref()?.slice.as_ref())
        .map(|slice| slice.payload_id.clone())
        .collect();
    assert!(signed.contains(&a_id.as_bytes().to_vec()));
    // `b` is intermediate: no initializer names it, but the unsigned hint for `a`
    // copies from it, so it must be materialized and bound through that hint.
    // The ready entry drops recipes (its closure is materialized); the chain is
    // proven from the signed source core's derivations.
    let chain = metadata.derivations.as_ref().unwrap();
    assert!(chain.derivations.iter().any(|hint| {
        hint.target_payload_id == a_id.as_bytes().to_vec()
            && hint.operations.iter().any(|operation| {
                matches!(&operation.operation,
                    Some(wire::derivation_operation::Operation::Copy(copy))
                        if copy.source_payload_id == b_id.as_bytes().to_vec())
            })
    }));
    let read = |id: PayloadId| -> Vec<u8> {
        let file = location(id.as_bytes());
        let payload = entry.payload(&file).unwrap();
        let mut bytes = vec![0; payload.len() as usize];
        payload.read_at(0, &mut bytes).unwrap();
        bytes
    };
    assert_eq!(read(a_id), a);
    assert_eq!(read(b_id), b);
    assert!(store.payload(a_id).unwrap().is_some());
    assert!(store.payload(d_id).unwrap().is_none());
    assert!(entry.model().contains("chain-model"));
}

#[test]
fn selected_derivation_closure_ignores_unselected_invalid_hints() {
    let scratch = tempfile::tempdir().unwrap();
    let fixture = publish(scratch.path(), true);
    let mut metadata = reader::Metadata::open(&fixture.packed).unwrap();
    let bogus_target = PayloadId::of(b"unselected target");
    let bogus_source = PayloadId::of(b"unselected source");
    let mut set = metadata.derivations.clone().unwrap();
    set.derivations.push(wire::Derivation {
        target_payload_id: bogus_target.as_bytes().to_vec(),
        operations: vec![wire::DerivationOperation {
            operation: Some(wire::derivation_operation::Operation::Copy(
                wire::DerivationCopy {
                    source_payload_id: bogus_source.as_bytes().to_vec(),
                    source_offset: 0,
                    destination_offset: 0,
                    contiguous_bytes: 16,
                    extents: vec![],
                    source_strides: vec![],
                    destination_strides: vec![],
                },
            )),
        }],
    });
    metadata.derivations = Some(set);
    let verified = protection::verify(
        &metadata.envelope,
        std::slice::from_ref(&fixture.publisher),
        protection::Selection::Deployment {
            deployment: 0,
            rank: None,
        },
    )
    .unwrap();
    let closure = cache::selected_derivation_closure(
        &metadata,
        std::slice::from_ref(&verified),
        &derivation::Limits::default(),
    )
    .unwrap();
    assert!(closure.contains(fixture.derived_id.as_bytes().as_slice()));
    assert!(closure.contains(fixture.source_id.as_bytes().as_slice()));
    assert!(!closure.contains(bogus_target.as_bytes().as_slice()));
    assert!(!closure.contains(bogus_source.as_bytes().as_slice()));
}

#[test]
fn packed_download_keeps_embedded_frontend_in_the_primary() {
    let scratch = tempfile::tempdir().unwrap();
    let fixture = publish(scratch.path(), false);
    let selection = verify_core(&fixture);
    let store = cache::Root::open(scratch.path().join("cache")).unwrap();
    let source = cache::CoreSource::open(
        &fixture.packed,
        std::slice::from_ref(&fixture.publisher),
        selection,
    )
    .unwrap();
    let entry = store
        .install_core(source, "packed-model", "entry")
        .expect("packed canonical install");
    let primary = container::Container::open(&entry.directory().join("model.safetensors")).unwrap();
    assert!(primary.tensor_names().any(|name| name == "frontend.tokenizer"));
    assert!(!entry.directory().join("frontend.tokenizer").exists());
    assert!(!entry.directory().join("weights/frontend.tokenizer.safetensors").exists());
}

/// Rewrite the packed primary with unsigned derivations mutated, copying every
/// other tensor verbatim so the signed manifest and payload bytes are unchanged.
fn packed_with_derivations(
    source: &Path,
    destination: &Path,
    mutate: impl FnOnce(&mut wire::DerivationSet),
) -> (Vec<u8>, Vec<u8>) {
    use prost::Message as _;
    let primary = container::Container::open(source).unwrap();
    let metadata = reader::Metadata::open(source).unwrap();
    let mut set = metadata.derivations.clone().unwrap_or_default();
    mutate(&mut set);
    let mut output = std::fs::File::create(destination).unwrap();
    let mut writer = container::Writer::new(&mut output, container::PAYLOAD_FORMAT).unwrap();
    let mut manifest = Vec::new();
    for name in primary.tensor_names() {
        let slice = primary.tensor(name).unwrap();
        if name == reader::DERIVATIONS || name.starts_with(container::ALIGNMENT_PREFIX) {
            continue;
        }
        if name == reader::MANIFEST {
            let mut bytes = vec![0; slice.len() as usize];
            slice.read_at(0, &mut bytes).unwrap();
            manifest = bytes;
        }
        writer
            .add(name, slice.len(), container::Source::Slice(slice))
            .unwrap();
    }
    let encoded = set.encode_to_vec();
    writer
        .add(
            reader::DERIVATIONS,
            encoded.len() as u64,
            container::Source::Bytes(encoded.clone()),
        )
        .unwrap();
    writer.finish().unwrap();
    (manifest, encoded)
}

fn target_recipe<'a>(
    set: &'a mut wire::DerivationSet,
    target: PayloadId,
) -> &'a mut wire::DerivationOperation {
    set.derivations
        .iter_mut()
        .find(|hint| hint.target_payload_id == target.as_bytes().to_vec())
        .expect("recipe for the selected target")
        .operations
        .first_mut()
        .expect("recipe operation")
}

#[test]
fn serialized_bad_recipe_fails_install_and_keeps_the_previous_default() {
    let scratch = tempfile::tempdir().unwrap();
    let fixture = publish(scratch.path(), true);
    let selection = verify_core(&fixture);
    let model = cache::model_key("packed-model").unwrap();
    let store = cache::Root::open(scratch.path().join("cache")).unwrap();

    // Establish a previous signed default with distinct payload bytes, so the
    // requested target is genuinely absent from the cache.
    let previous_scratch = tempfile::tempdir().unwrap();
    let previous = publish_with_source(
        previous_scratch.path(),
        b"previous canonical weight bytes",
        true,
    );
    assert_ne!(previous.derived_id, fixture.derived_id);
    let previous_selection = verify_core(&previous);
    let previous_source = cache::CoreSource::open(
        &previous.packed,
        std::slice::from_ref(&previous.publisher),
        previous_selection,
    )
    .unwrap();
    let first = store
        .install_core(previous_source, &model, "v1")
        .unwrap();
    let previous_source = cache::CoreSource::open(
        &previous.packed,
        std::slice::from_ref(&previous.publisher),
        previous_selection,
    )
    .unwrap();
    store.verify_ready(&first, &previous_source).unwrap();
    assert_eq!(store.ref_entry(&model).unwrap().as_deref(), Some("v1"));
    assert!(store.payload(fixture.derived_id).unwrap().is_none());

    // A serialized hint whose copy offset no longer reproduces the signed target.
    let tampered = scratch.path().join("tampered.safetensors");
    let (tampered_manifest, _) = packed_with_derivations(&fixture.packed, &tampered, |set| {
        if let Some(wire::derivation_operation::Operation::Copy(copy)) =
            target_recipe(set, fixture.derived_id).operation.as_mut()
        {
            copy.source_offset += 1;
        }
    });
    let original_manifest = {
        let primary = container::Container::open(&fixture.packed).unwrap();
        let slice = primary.tensor(reader::MANIFEST).unwrap();
        let mut bytes = vec![0; slice.len() as usize];
        slice.read_at(0, &mut bytes).unwrap();
        bytes
    };
    assert_eq!(
        tampered_manifest, original_manifest,
        "the signed manifest must be unchanged"
    );

    let tampered_source = cache::CoreSource::open(
        &tampered,
        std::slice::from_ref(&fixture.publisher),
        selection,
    )
    .unwrap();
    assert!(
        store.install_core(tampered_source, &model, "v2").is_err(),
        "a recipe that cannot reproduce the signed target must fail"
    );
    assert!(store.entry(&model, "v2").is_err(), "no new entry may appear");
    assert_eq!(
        store.ref_entry(&model).unwrap().as_deref(),
        Some("v1"),
        "the previous default must stay selected"
    );
    assert!(first.directory().is_dir());
    assert!(store.payload(fixture.derived_id).unwrap().is_none());
    let previous_source = cache::CoreSource::open(
        &previous.packed,
        std::slice::from_ref(&previous.publisher),
        previous_selection,
    )
    .unwrap();
    store.verify_ready(&first, &previous_source).unwrap();
    drop(first);
}

#[test]
fn cached_authenticated_target_is_reused_despite_unusable_hint() {
    let scratch = tempfile::tempdir().unwrap();
    let fixture = publish(scratch.path(), true);
    let selection = verify_core(&fixture);
    let model = cache::model_key("packed-model").unwrap();
    let store = cache::Root::open(scratch.path().join("cache")).unwrap();

    let clean = cache::CoreSource::open(
        &fixture.packed,
        std::slice::from_ref(&fixture.publisher),
        selection,
    )
    .unwrap();
    let first = store.install_core(clean, &model, "v1").unwrap();
    assert!(store.payload(fixture.derived_id).unwrap().is_some());

    let tampered = scratch.path().join("tampered-warm.safetensors");
    packed_with_derivations(&fixture.packed, &tampered, |set| {
        if let Some(wire::derivation_operation::Operation::Copy(copy)) =
            target_recipe(set, fixture.derived_id).operation.as_mut()
        {
            copy.source_offset += 1;
        }
    });
    let tampered_source = cache::CoreSource::open(
        &tampered,
        std::slice::from_ref(&fixture.publisher),
        selection,
    )
    .unwrap();
    // The cached target hashes to the signed identity. An unusable unsigned hint
    // must not force materialization and must not override that authoritative
    // object.
    let progress = flare_helios::core::progress::AcquisitionProgress::new();
    let plan = store
        .plan_acquisition_with(
            tampered_source,
            &derivation::Limits::default(),
            Some(&progress),
        )
        .unwrap();
    let second = store
        .install_planned(plan, &model, "v2")
        .expect("a valid authenticated cached target must still be reusable");
    store.publish_entry(&second).unwrap();
    assert_eq!(store.ref_entry(&model).unwrap().as_deref(), Some("v2"));
    let snapshot = progress.snapshot();
    assert_eq!(snapshot.materialized_copy_bytes, 0);
    assert_eq!(snapshot.materialized_fill_bytes, 0);
    let installed = store.entry(&model, "v2").unwrap();
    let metadata = reader::Metadata::open(installed.directory()).unwrap();
    let location = metadata
        .locations
        .locations
        .iter()
        .find(|location| location.payload_id == fixture.derived_id.as_bytes().to_vec())
        .expect("reused target must remain in the entry closure");
    let slice = installed.payload(&location.file).unwrap();
    assert_eq!(slice.digest().unwrap(), fixture.derived_id);
    drop(installed);
    drop(second);
    drop(first);
}

#[test]
fn serialized_unrelated_bogus_hint_is_ignored_by_install() {
    let scratch = tempfile::tempdir().unwrap();
    let fixture = publish(scratch.path(), true);
    let selection = verify_core(&fixture);
    let store = cache::Root::open(scratch.path().join("cache")).unwrap();
    let bogus_target = PayloadId::of(b"unselected target");
    let bogus_source = PayloadId::of(b"unselected source");
    let source = scratch.path().join("bogus.safetensors");
    packed_with_derivations(&fixture.packed, &source, |set| {
        set.derivations.push(wire::Derivation {
            target_payload_id: bogus_target.as_bytes().to_vec(),
            operations: vec![wire::DerivationOperation {
                operation: Some(wire::derivation_operation::Operation::Copy(
                    wire::DerivationCopy {
                        source_payload_id: bogus_source.as_bytes().to_vec(),
                        source_offset: 0,
                        destination_offset: 0,
                        contiguous_bytes: 16,
                        extents: vec![],
                        source_strides: vec![],
                        destination_strides: vec![],
                    },
                )),
            }],
        });
    });
    let core = cache::CoreSource::open(
        &source,
        std::slice::from_ref(&fixture.publisher),
        selection,
    )
    .unwrap();
    let entry = store
        .install_core(core, "bogus-model", "ok")
        .expect("an unrelated unauthenticated hint must not break the install");
    assert!(store.payload(fixture.derived_id).unwrap().is_some());
    assert!(store.payload(bogus_target).unwrap().is_none());
    assert!(store.payload(bogus_source).unwrap().is_none());
    assert!(entry.payload("weights/tuned/sm120-tp1/derived-000.safetensors").is_ok());
    let _ = scratch;
}
