use std::path::{Path, PathBuf};
const COMMIT: &str = "0123456789012345678901234567890123456789";
pub(crate) fn signed_payload_fixture(
    parent: &std::path::Path,
) -> (
    std::path::PathBuf,
    std::path::PathBuf,
    flare_helios::runtime_core::Publisher,
    String,
) {
    signed_payload_fixture_with(parent, "gb10-tp1", "sm120")
}

fn signed_payload_fixture_with(
    parent: &std::path::Path,
    deployment_id: &str,
    hardware_identity: &str,
) -> (
    std::path::PathBuf,
    std::path::PathBuf,
    flare_helios::runtime_core::Publisher,
    String,
) {
    signed_payload_fixture_identity(parent, deployment_id, hardware_identity, "owner/offline")
}

fn signed_payload_fixture_identity(
    parent: &Path,
    deployment_id: &str,
    hardware_identity: &str,
    model: &str,
) -> (
    PathBuf,
    PathBuf,
    flare_helios::runtime_core::Publisher,
    String,
) {
    use flare_helios::core::{PayloadId, container, export, schema::wire};
    use flare_helios::runtime_core::Publisher;
    use ring::signature::{Ed25519KeyPair, KeyPair};

    let core = parent.join("core");
    std::fs::create_dir_all(&core).unwrap();
    let weights = b"offline orbit weights".repeat(4);
    let id = PayloadId::of(&weights);
    let raw = parent.join("weights.bin");
    std::fs::write(&raw, &weights).unwrap();
    let graph = parent.join("graph.bin");
    std::fs::write(&graph, b"graph-bytes").unwrap();
    let pkcs8 = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new()).unwrap();
    let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
    let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
    let publisher = Publisher {
        name: "Offline Orbit".into(),
        namespace: "local-development".into(),
        public_key,
        development: true,
    };
    let contract = flare_helios::runtime_core::wire::ServingContract {
        model_id: "orbit-offline".into(),
        identity: "orbit-offline-sm120".into(),
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
        deployment_id: deployment_id.into(),
        context: Some(Default::default()),
        allocations: vec![wire::Allocation {
            name: "device-0".into(),
            nbytes: weights.len() as u64,
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
                    nbytes: weights.len() as u64,
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
            model_identity: model.into(),
            hardware_identity: hardware_identity.into(),
            frontend: Some(flare_helios::runtime_core::wire::FrontendSettings::default()),
            sampling_defaults: Some(
                flare_helios::runtime_core::wire::DefaultSamplingParameters::default(),
            ),
            publisher: Some(wire::PublisherMetadata {
                namespace: "local-development".into(),
                official_source: "local-development".into(),
                development: true,
                ..Default::default()
            }),
            ..Default::default()
        },
        payloads: vec![],
        deployments: vec![export::PublishDeployment {
            payloads: vec![],
            body: wire::DeploymentBody {
                deployment_id: deployment_id.into(),
                architecture: vec![12, 0],
                tensor_parallel: 1,
                contract: Some(contract),
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
                        weights.len() as u64,
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
        core_id: [23u8; 32],
        public_key,
        accepted: "local-development".into(),
        no_weights: false,
    };
    export::publish(&request, &key, export::Destination::Directory(&core)).unwrap();
    let profile = parent.join("publisher.json");
    std::fs::write(
        &profile,
        serde_json::to_vec(&vec![publisher.clone()]).unwrap(),
    )
    .unwrap();
    (core, profile, publisher, COMMIT.to_owned())
}

pub(crate) fn fixture(companions: &[&str]) -> Vec<u8> {
    fn bytes(field: u8, body: &[u8]) -> Vec<u8> {
        let mut out = vec![field * 8 + 2];
        let mut n = body.len();
        while n >= 128 {
            out.push((n as u8 & 127) | 128);
            n >>= 7;
        }
        out.push(n as u8);
        out.extend_from_slice(body);
        out
    }
    // ProtectedIndex -> PublisherMetadata -> ModelIndex, with unsigned discovery metadata.
    let index = [8, if companions.is_empty() { 4 } else { 5 }];
    let metadata = bytes(1, &index);
    let mut envelope = vec![8, 4];
    for (kind, body) in [(1, metadata), (2, vec![]), (3, vec![])] {
        envelope.extend(bytes(4, &[vec![8, kind], bytes(2, &body)].concat()));
    }
    let mut data = Vec::new();
    for name in companions {
        data.extend(bytes(4, &bytes(3, name.as_bytes())));
    }
    let rank = [bytes(2, &[]), bytes(5, &data)].concat();
    envelope.extend(bytes(5, &[vec![8, 4], bytes(2, &rank)].concat()));
    let header = serde_json::json!({
        "__metadata__": {"format": "flare"},
        "flare.payload": {"dtype": "U8", "shape": [0], "data_offsets": [0, 0]},
        "flare.metadata": {"dtype": "U8", "shape": [envelope.len()], "data_offsets": [0, envelope.len()]},
    });
    let mut result = 4088u64.to_le_bytes().to_vec();
    result.extend(serde_json::to_vec(&header).unwrap());
    result.resize(4096, b' ');
    result.extend(envelope);
    result
}
