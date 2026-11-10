/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use crate::core::protection::{self, EnvelopeBuilder, Selection};
use crate::runtime_core::{Publisher, wire as runtime};
use ring::rand::SystemRandom;
use ring::signature::{Ed25519KeyPair, KeyPair};

fn weight_catalog() -> wire::PreparedWeightCatalog {
    wire::PreparedWeightCatalog {
        version: 1,
        views: (0..64)
            .map(|index| wire::PreparedWeightView {
                name: format!("decoder.layer{index}.attention.weight"),
                group: "attention".into(),
                dtype: "uint8".into(),
                element_bytes: 1,
                shape: vec![16],
                stride: vec![1],
                allocation: 0,
                offset: index * 16,
                nbytes: 16,
                sources: vec![wire::WeightSource {
                    tensor: format!("model.layers.{index}.self_attn.weight"),
                    ranges: vec![wire::TensorRange { begin: 0, end: 16 }],
                    ..Default::default()
                }],
                ..Default::default()
            })
            .collect(),
        unavailable: vec![],
    }
}

fn rank_body() -> wire::RankBody {
    let (stored, resource) = protection::compress(b"graph resource").unwrap();
    wire::RankBody {
        deployment_id: "sm120-tp".into(),
        context: Some(runtime::Context::default()),
        allocations: vec![wire::Allocation {
            name: "weights".into(),
            nbytes: 4096,
            alignment: 4096,
            address: Some(0x1000_0000),
            memory: runtime::Memory::Device as i32,
            ..Default::default()
        }],
        execution: Some(runtime::Execution::default()),
        payloads: vec![wire::PayloadDescriptor {
            id: vec![1; 32],
            nbytes: stored.len() as u64,
        }],
        resources: vec![wire::ExecutableResource {
            resource_id: 1,
            kind: wire::ExecutableKind::Graph as i32,
            stored: Some(wire::PayloadSlice {
                payload_id: vec![1; 32],
                offset: 0,
                nbytes: stored.len() as u64,
            }),
            decoded_nbytes: 14,
            chunk_nbytes: schema::MAX_CHUNK as u32,
            chunk_sha256: resource.chunk_sha256,
        }],
        graphs: vec![wire::Graph {
            name: "decode".into(),
            node_count: 1,
            ..Default::default()
        }],
        prepared_weights: Some(weight_catalog()),
        ..Default::default()
    }
}

fn compressed_body() -> wire::RankBody {
    let mut body = rank_body();
    compact(&mut body).unwrap();
    assert!(body.prepared_weights.is_none());
    assert!(body.compressed_prepared_weights.is_some());
    body
}

fn invalid_frame(decoded_nbytes: u64) -> wire::RankBody {
    let mut body = rank_body();
    body.prepared_weights = None;
    body.compressed_prepared_weights = Some(wire::CompressedPreparedWeightCatalog {
        zstd: b"invalid zstd frame".to_vec(),
        decoded_nbytes,
    });
    body
}

fn signed(ranks: &[wire::RankBody]) -> (Vec<u8>, Vec<Publisher>) {
    let randomness = SystemRandom::new();
    let pkcs8 = Ed25519KeyPair::generate_pkcs8(&randomness).unwrap();
    let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
    let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
    let core_id = [0x35; 32];
    let encoded_ranks: Vec<_> = ranks
        .iter()
        .enumerate()
        .map(|(index, body)| {
            let mut body = body.clone();
            body.rank = index as u32;
            for resource in &mut body.resources {
                resource.resource_id += index as u64;
            }
            body.encode_to_vec()
        })
        .collect();
    let deployment = wire::DeploymentBody {
        deployment_id: "sm120-tp".into(),
        architecture: vec![12, 0],
        tensor_parallel: ranks.len() as u32,
        contract: Some(runtime::ServingContract::default()),
        ranks: encoded_ranks
            .iter()
            .enumerate()
            .map(|(rank, body)| wire::RankCommitment {
                rank: rank as u32,
                body_digest: protection::sha256(body).to_vec(),
            })
            .collect(),
        ..Default::default()
    }
    .encode_to_vec();
    let core = wire::CoreBody {
        format_version: schema::FORMAT_VERSION,
        core_id: core_id.to_vec(),
        model_identity: "test-model".into(),
        hardware_identity: "sm120".into(),
        publisher: Some(wire::PublisherMetadata {
            namespace: "example".into(),
            official_source: "https://huggingface.co/example/model".into(),
            ..Default::default()
        }),
        frontend: Some(runtime::FrontendSettings::default()),
        sampling_defaults: Some(runtime::DefaultSamplingParameters::default()),
        deployments: vec![wire::DeploymentCommitment {
            deployment_id: "sm120-tp".into(),
            body_digest: protection::sha256(&deployment).to_vec(),
        }],
        ..Default::default()
    };
    let mut builder =
        EnvelopeBuilder::new(core.encode_to_vec(), public_key, core_id).unwrap();
    builder.deployment(0, deployment).unwrap();
    for (rank, bytes) in encoded_ranks.into_iter().enumerate() {
        builder.rank(0, rank as u32, bytes).unwrap();
    }
    (
        builder.sign(&key).unwrap(),
        vec![Publisher {
            name: "Example".into(),
            namespace: "example".into(),
            public_key,
            development: false,
        }],
    )
}

fn selected(rank: Option<u32>) -> Selection {
    Selection::Deployment {
        deployment: 0,
        rank,
    }
}

#[test]
fn inline_catalog_remains_borrowed_and_compressed_catalog_round_trips_exactly() {
    let mut body = rank_body();
    let original = body.clone();
    assert!(matches!(catalog(&body).unwrap(), Some(Cow::Borrowed(_))));
    compact(&mut body).unwrap();
    assert!(body.encoded_len() < original.encoded_len());
    let recovered = catalog(&body).unwrap().unwrap();
    assert!(matches!(recovered, Cow::Owned(_)));
    assert_eq!(*recovered, original.prepared_weights.clone().unwrap());
    assert_eq!(
        reconstructed_rank_nbytes(&body, body.encoded_len()).unwrap(),
        original.encoded_len() as u64
    );
    assert_eq!(
        reconstructed_rank_nbytes(&body, body.encoded_len() + 19).unwrap(),
        original.encoded_len() as u64 + 19
    );
    let encoded = body.encode_to_vec();
    compact(&mut body).unwrap();
    assert_eq!(body.encode_to_vec(), encoded);
}

#[test]
fn absent_and_small_inline_catalogs_preserve_their_representation() {
    let mut body = wire::RankBody::default();
    compact(&mut body).unwrap();
    assert!(catalog(&body).unwrap().is_none());
    body.prepared_weights = Some(wire::PreparedWeightCatalog {
        version: 1,
        ..Default::default()
    });
    let original = body.encode_to_vec();
    compact(&mut body).unwrap();
    assert_eq!(body.encode_to_vec(), original);
    assert!(matches!(catalog(&body).unwrap(), Some(Cow::Borrowed(_))));
}

#[test]
fn compressed_catalog_rejects_corruption_truncation_extra_frames_and_length_mismatch() {
    for fault in [
        "magic",
        "checksum",
        "truncated",
        "trailing",
        "extra_frame",
        "short",
        "long",
    ] {
        let mut body = compressed_body();
        let compressed = body.compressed_prepared_weights.as_mut().unwrap();
        match fault {
            "magic" => compressed.zstd[0] ^= 0x80,
            "checksum" => *compressed.zstd.last_mut().unwrap() ^= 1,
            "truncated" => {
                compressed.zstd.pop();
            }
            "trailing" => compressed.zstd.push(0),
            "extra_frame" => compressed.zstd.extend_from_within(..),
            "short" => compressed.decoded_nbytes -= 1,
            "long" => compressed.decoded_nbytes += 1,
            _ => unreachable!(),
        }
        assert!(catalog(&body).is_err(), "accepted {fault}");
    }
}

#[test]
fn compressed_catalog_requires_valid_protobuf_and_a_bounded_window() {
    let mut body = invalid_frame(1);
    body.compressed_prepared_weights.as_mut().unwrap().zstd =
        zstd::stream::encode_all([0x0f].as_slice(), 3).unwrap();
    let error = catalog(&body).unwrap_err().to_string();
    assert!(error.contains("invalid prepared weight catalog"), "{error}");

    let decoded = weight_catalog().encode_to_vec();
    let mut encoder = zstd::stream::write::Encoder::new(Vec::new(), 3).unwrap();
    encoder.window_log(24).unwrap();
    encoder.include_contentsize(false).unwrap();
    encoder.write_all(&decoded).unwrap();
    let mut oversized_window = encoder.finish().unwrap();
    assert_eq!(oversized_window[4] & 0x20, 0);
    oversized_window[5] = 14 << 3;
    let declared_window = 1u64 << (10 + (oversized_window[5] >> 3));
    assert_eq!(declared_window, 16 << 20);
    let mut body = rank_body();
    body.prepared_weights = None;
    body.compressed_prepared_weights = Some(wire::CompressedPreparedWeightCatalog {
        zstd: oversized_window,
        decoded_nbytes: decoded.len() as u64,
    });
    assert!(catalog(&body).is_err());
}

#[test]
fn compressed_wrapper_and_reconstructed_rank_lengths_are_bounded() {
    let mut body = compressed_body();
    body.prepared_weights = Some(weight_catalog());
    assert!(
        validate_wrapper(&body)
            .unwrap_err()
            .to_string()
            .contains("both inline and compressed")
    );
    body.prepared_weights = None;
    for length in [0, schema::MAX_BODY as u64 + 1, u64::MAX] {
        body.compressed_prepared_weights
            .as_mut()
            .unwrap()
            .decoded_nbytes = length;
        assert!(validate_wrapper(&body).is_err());
    }
    let mut body = invalid_frame(schema::MAX_BODY as u64 - 1024);
    let encoded = body.encoded_len();
    assert!(reconstructed_rank_nbytes(&body, encoded).unwrap() < schema::MAX_BODY as u64);
    assert!(reconstructed_rank_nbytes(&body, encoded + 2048).is_err());
    body.compressed_prepared_weights
        .as_mut()
        .unwrap()
        .zstd
        .clear();
    assert!(validate_wrapper(&body).is_err());
    assert!(reconstructed_rank_nbytes(&compressed_body(), 0).is_err());
}

#[test]
fn authenticated_legacy_and_compressed_catalogs_have_identical_semantics() {
    for body in [rank_body(), compressed_body()] {
        let (bytes, publishers) = signed(&[body]);
        let verified = protection::verify(&bytes, &publishers, selected(None)).unwrap();
        assert_eq!(
            *catalog(verified.rank(0).unwrap()).unwrap().unwrap(),
            weight_catalog()
        );
        assert!(verified.matches_envelope(&bytes));
    }
}

#[test]
fn authenticated_catalogs_retain_semantic_validation() {
    let mut body = rank_body();
    body.prepared_weights.as_mut().unwrap().views[0].allocation = 500;
    compact(&mut body).unwrap();
    let (bytes, publishers) = signed(&[body]);
    let error = protection::verify(&bytes, &publishers, selected(None))
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("weight view names an absent allocation"),
        "{error}"
    );
}

#[test]
fn every_selected_signature_is_verified_before_the_first_catalog_is_decoded() {
    let (bytes, publishers) = signed(&[invalid_frame(128), compressed_body()]);
    let mut envelope = wire::ProtectedCore::decode(bytes.as_slice()).unwrap();
    envelope.ranks[1].signature[0] ^= 0x80;
    let error = protection::verify(&envelope.encode_to_vec(), &publishers, selected(None))
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("signature"), "{error}");
    assert!(!error.contains("prepared weight decompressor"), "{error}");
}

#[test]
fn rank_commitments_are_verified_before_catalog_decoding() {
    let (bytes, publishers) = signed(&[compressed_body()]);
    let mut envelope = wire::ProtectedCore::decode(bytes.as_slice()).unwrap();
    let mut body = wire::RankBody::decode(envelope.ranks[0].body.as_slice()).unwrap();
    body.compressed_prepared_weights.as_mut().unwrap().zstd[0] ^= 0x80;
    envelope.ranks[0].body = body.encode_to_vec();
    let error = protection::verify(&envelope.encode_to_vec(), &publishers, selected(None))
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("rank body differs from its commitment"),
        "{error}"
    );
}

#[test]
fn selected_decoded_metadata_budget_is_checked_before_inflation() {
    let ranks = vec![invalid_frame(schema::MAX_BODY as u64 - 4096); 5];
    let (bytes, publishers) = signed(&ranks);
    assert!(bytes.len() < 4096);
    let error = protection::verify(&bytes, &publishers, selected(None))
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("selected reconstructed rank metadata exceeds its limit"),
        "{error}"
    );
}

#[test]
fn unselected_catalogs_do_not_consume_decoded_budget_or_decode_untrusted_frames() {
    let mut ranks = vec![compressed_body()];
    ranks.extend(vec![invalid_frame(schema::MAX_BODY as u64 - 4096); 5]);
    let (bytes, publishers) = signed(&ranks);
    let verified = protection::verify(&bytes, &publishers, selected(Some(0))).unwrap();
    assert_eq!(verified.ranks().len(), 1);
    assert_eq!(
        *catalog(verified.rank(0).unwrap()).unwrap().unwrap(),
        weight_catalog()
    );
}

#[test]
fn early_initializer_coalescing_preserves_zero_length_rejection() {
    for empty_first in [false, true] {
        let mut body = compressed_body();
        body.payloads.push(wire::PayloadDescriptor {
            id: vec![7; 32],
            nbytes: 16,
        });
        body.initializers = [0, 1]
            .into_iter()
            .map(|index| {
                let nbytes = if (index == 0) == empty_first { 0 } else { 4 };
                let offset = if index == 1 && !empty_first { 4 } else { 0 };
                wire::Initializer {
                    allocation: "weights".into(),
                    destination_offset: offset,
                    source: Some(wire::PayloadUse {
                        role: wire::PayloadRole::Weights as i32,
                        slice: Some(wire::PayloadSlice {
                            payload_id: vec![7; 32],
                            offset,
                            nbytes,
                        }),
                    }),
                }
            })
            .collect();
        crate::core::export::coalesce_initializers(&mut body.initializers).unwrap();
        let (bytes, publishers) = signed(&[body]);
        let error = protection::verify(&bytes, &publishers, selected(None))
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("initializer destination is outside its allocation"),
            "{error}"
        );
    }
}
