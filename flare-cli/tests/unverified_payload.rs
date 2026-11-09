/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Unverified payload inspection reads bounded primary metadata only.
//!
//! Each fixture is a real signed payload core built with the native export
//! API. The unverified path is deliberately exercised without publisher
//! profiles, with missing companions, and through both directory and packed
//! forms.

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::Arc,
};

use flare_helios::core::{
    PayloadId, container, conversion, export, protection, reader, schema::wire,
};
use flare_helios::runtime_core::{
    InspectionOptions, Publisher, unverified_metadata, wire as runtime_wire,
};
use ring::signature::{Ed25519KeyPair, KeyPair};

fn flare(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_flare"))
        .args(arguments)
        .output()
        .unwrap()
}

fn options(memory_map: bool, graphs: bool, frontend: bool, all: bool) -> InspectionOptions {
    InspectionOptions {
        memory_map,
        graphs,
        frontend,
        all,
    }
}

fn publish(root: &Path, deployments: &[u32]) -> Publisher {
    let configured: Vec<(u32, u64, u64, Option<u64>, Option<u64>)> = deployments
        .iter()
        .map(|&tensor_parallel| (tensor_parallel, 32, 4, Some(7), Some(7)))
        .collect();
    publish_configured(root, &configured)
}

fn publish_configured(
    root: &Path,
    deployments: &[(u32, u64, u64, Option<u64>, Option<u64>)],
) -> Publisher {
    let weights = b"payload weights".repeat(8);
    let tokenizer = b"tokenizer-json".repeat(4);
    let graph = root.join("graph.bin");
    fs::write(&graph, b"graph-bytes").unwrap();
    let pkcs8 = Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new()).unwrap();
    let key = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
    let public_key: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
    let publisher = Publisher {
        name: "Unverified Test".into(),
        namespace: "unverified-test".into(),
        public_key,
        development: false,
    };
    let contract_for =
        |max_sequence_tokens: u64, prefill_chunk_size: u64, max_active: Option<u64>| {
            runtime_wire::ServingContract {
                model_id: "unverified/model".into(),
                identity: "unverified-model".into(),
                max_sequence_tokens,
                prefill_chunk_size,
                pages: 2,
                state_rows: 2,
                page_tokens: 4,
                max_active,
                model_contract: Some(runtime_wire::ModelContract {
                    vocabulary: 16,
                    vision: Some(runtime_wire::VisionCapacity {
                        feature_slots: 2,
                        features_per_request: 1,
                        max_images: 4,
                        patches_per_feature: 3,
                        patch_capacities: vec![64, 128],
                    }),
                    frontend: Some(runtime_wire::FrontendContract {
                        tools: Some(runtime_wire::ToolParser {
                            kind: runtime_wire::ToolParserKind::QwenCoder as i32,
                            wrapping_newlines: 1,
                            tag_separator: "|".into(),
                        }),
                        reasoning: Some(runtime_wire::ReasoningParser {
                            kind: runtime_wire::ReasoningParserKind::ThinkTags as i32,
                            open: "think".into(),
                            close: "/think".into(),
                            wrapping_newlines: 1,
                            terminal_markers: vec!["done".into()],
                        }),
                        assistant_marker: "assistant".into(),
                        message_end_markers: vec!["end".into()],
                        position_dimensions: 1,
                        ..Default::default()
                    }),
                    graphs: vec![runtime_wire::GraphContract {
                        graph: "decode".into(),
                        ..Default::default()
                    }],
                    ..Default::default()
                }),
                ..Default::default()
            }
        };
    let asset_id = PayloadId::of(&tokenizer);
    fs::write(root.join("tokenizer.bin"), &tokenizer).unwrap();
    let mut deployment_requests = vec![];
    for &(
        tensor_parallel,
        max_sequence_tokens,
        prefill_chunk_size,
        max_active,
        execution_max_active,
    ) in deployments
    {
        let mut ranks = vec![];
        for rank in 0..tensor_parallel {
            let rank_weights = [
                weights.as_slice(),
                format!("tp{tensor_parallel}rank{rank}").as_bytes(),
            ]
            .concat();
            let id = PayloadId::of(&rank_weights);
            let path = root.join(format!("weights-{tensor_parallel}-{rank}.bin"));
            fs::write(&path, &rank_weights).unwrap();
            ranks.push(export::PublishRank {
                body: wire::RankBody {
                    rank,
                    deployment_id: format!("sm120-tp{tensor_parallel}"),
                    context: Some(Default::default()),
                    reservations: vec![runtime_wire::Reservation {
                        address: 0x2000,
                        nbytes: 4096,
                    }],
                    allocations: vec![wire::Allocation {
                        name: "device-0".into(),
                        nbytes: rank_weights.len() as u64,
                        alignment: 4096,
                        address: Some(0x1000),
                        ..Default::default()
                    }],
                    initializers: vec![wire::Initializer {
                        allocation: "device-0".into(),
                        destination_offset: 0,
                        source: Some(wire::PayloadUse {
                            slice: Some(wire::PayloadSlice {
                                payload_id: id.as_bytes().to_vec(),
                                offset: 0,
                                nbytes: rank_weights.len() as u64,
                            }),
                            role: wire::PayloadRole::Weights as i32,
                        }),
                    }],
                    graphs: vec![wire::Graph {
                        name: "decode".into(),
                        resource: 0,
                        node_count: 3,
                        edge_count: 2,
                        ..Default::default()
                    }],
                    execution: execution_max_active.map(|max_active| runtime_wire::Execution {
                        abi_version: 1,
                        max_active,
                        input_frame_bytes: 16,
                        buffers: vec![runtime_wire::Buffer {
                            name: "input".into(),
                            nbytes: 16,
                            dtype: "uint8".into(),
                            shape: vec![16],
                            reference: Some(runtime_wire::Reference {
                                allocation: "device-0".into(),
                                offset: 0,
                            }),
                            ..Default::default()
                        }],
                        graphs: vec![runtime_wire::ExecutionGraph {
                            name: "decode".into(),
                            kind: "decode".into(),
                            sampling_mode: "sample".into(),
                            batch_capacity: 1,
                            token_capacity: 1,
                            estimated_duration_ns: 1234,
                            ..Default::default()
                        }],
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                payloads: vec![export::PublishPayload {
                    role: wire::PayloadRole::Weights,
                    relative: format!("weights/tp{tensor_parallel}/rank{rank}.safetensors"),
                    source: container::Slice::new(
                        Arc::new(fs::File::open(&path).unwrap()),
                        0,
                        rank_weights.len() as u64,
                    )
                    .unwrap(),
                    id,
                    tensor: None,
                }],
                resources: vec![export::PublishResource {
                    kind: wire::ExecutableKind::Graph,
                    path: graph.clone(),
                    nbytes: 11,
                }],
            });
        }
        deployment_requests.push(export::PublishDeployment {
            body: wire::DeploymentBody {
                deployment_id: format!("sm120-tp{tensor_parallel}"),
                architecture: vec![12, 0],
                tensor_parallel,
                contract: Some(contract_for(
                    max_sequence_tokens,
                    prefill_chunk_size,
                    max_active,
                )),
                cluster: Some(runtime_wire::Cluster {
                    world: tensor_parallel,
                    export_id: "unverified-export".into(),
                }),
                ..Default::default()
            },
            payloads: vec![],
            ranks,
        });
    }
    let request = export::PublishRequest {
        core: wire::CoreBody {
            model_identity: "unverified/model".into(),
            hardware_identity: "sm120".into(),
            publisher: Some(wire::PublisherMetadata {
                namespace: "unverified-test".into(),
                official_source: "https://huggingface.co/unverified-test/model".into(),
                development: false,
                ..Default::default()
            }),
            frontend: Some(runtime_wire::FrontendSettings {
                tokenizer: Some(runtime_wire::TokenizerSettings {
                    bos_token: Some("bos".into()),
                    eos_token: Some("eos".into()),
                    ..Default::default()
                }),
                eos_token_ids: vec![1, 2, 3],
                ..Default::default()
            }),
            sampling_defaults: Some(runtime_wire::DefaultSamplingParameters::default()),
            assets: vec![wire::FrontendAsset {
                name: "tokenizer".into(),
                payload: Some(wire::PayloadUse {
                    slice: Some(wire::PayloadSlice {
                        payload_id: asset_id.as_bytes().to_vec(),
                        offset: 0,
                        nbytes: tokenizer.len() as u64,
                    }),
                    role: wire::PayloadRole::Frontend as i32,
                }),
                sha256: protection::sha256(&tokenizer).to_vec(),
            }],
            ..Default::default()
        },
        payloads: vec![export::PublishPayload {
            role: wire::PayloadRole::Frontend,
            relative: String::new(),
            source: container::Slice::new(
                Arc::new(fs::File::open(root.join("tokenizer.bin")).unwrap()),
                0,
                tokenizer.len() as u64,
            )
            .unwrap(),
            id: asset_id,
            tensor: Some("tokenizer".into()),
        }],
        deployments: deployment_requests,
        core_id: [41u8; 32],
        public_key,
        accepted: "unverified-test".into(),
        no_weights: false,
    };
    export::publish(&request, &key, export::Destination::Directory(root)).unwrap();
    publisher
}

fn section_identities(text: &str, title: &str) -> Vec<(String, usize)> {
    text.lines()
        .filter(|line| line.contains(title))
        .map(|line| {
            let words: Vec<&str> = line.split_whitespace().collect();
            let deployment = words
                .iter()
                .find(|word| word.starts_with("sm120-tp"))
                .unwrap_or_else(|| panic!("section lacks deployment: {line}"))
                .to_string();
            let rank = words
                .iter()
                .position(|word| *word == "rank")
                .and_then(|position| words.get(position + 1))
                .and_then(|value| value.parse().ok())
                .unwrap_or_else(|| panic!("section lacks rank: {line}"));
            (deployment, rank)
        })
        .collect()
}

fn plain_section<'a>(text: &'a str, title: &str) -> &'a str {
    text.trim()
        .split("\n\n")
        .find(|section| section.lines().next() == Some(title))
        .unwrap_or_else(|| panic!("missing section {title}:\n{text}"))
}

fn assert_plain_row(section: &str, key: &str, value: &str) {
    let expected = format!("{key} {value}");
    assert!(
        section
            .lines()
            .any(|line| { line.split_whitespace().collect::<Vec<_>>().join(" ") == expected }),
        "missing row {expected}:\n{section}"
    );
}

fn assert_plain_graph(text: &str, deployment: &str, rank: usize) {
    let graph = plain_section(
        text,
        &format!("Graphs · rank {rank} · deployment {deployment}"),
    );
    assert_plain_row(graph, "Graph", "decode");
    assert_plain_row(graph, "Nodes / edges", "3 / 2");
    assert_plain_row(graph, "Kind / sampling", "decode / sample");
    assert_plain_row(graph, "Batch / tokens", "1 / 1");
    assert_plain_row(graph, "Duration", "1 µs");
}

fn assert_plain_memory(text: &str, deployment: &str, rank: usize) {
    let memory = plain_section(
        text,
        &format!("Memory map · rank {rank} · deployment {deployment}"),
    );
    assert_plain_row(memory, "device-0", "128 B");
    assert_plain_row(memory, "Reservation", "8192 · 4 KiB");
    assert_plain_row(memory, "Buffer", "input");
    assert_plain_row(memory, "Allocation / offset", "device-0 / 0");
    assert_plain_row(memory, "Initializer", "device-0 + 0");
    assert_plain_row(memory, "Offset / size", "0 / 128 B");
    let tensor_parallel = deployment.strip_prefix("sm120-tp").unwrap();
    let weights = [
        b"payload weights".repeat(8),
        format!("tp{tensor_parallel}rank{rank}").into_bytes(),
    ]
    .concat();
    let payload = PayloadId::of(&weights);
    // Payload identifiers can wrap across aligned continuation rows.
    assert!(
        memory
            .split_whitespace()
            .collect::<String>()
            .contains(&format!("Payload{payload}")),
        "{memory}"
    );
}

fn companion_files(root: &Path) -> Vec<PathBuf> {
    let metadata = reader::Metadata::open(root).unwrap();
    metadata
        .locations
        .locations
        .iter()
        .filter(|location| !location.file.is_empty())
        .map(|location| root.join(&location.file))
        .collect()
}

#[test]
fn unverified_payload_directory_and_packed_read_metadata_without_companions() {
    let temp = tempfile::tempdir().unwrap();
    let directory = temp.path().join("core");
    fs::create_dir_all(&directory).unwrap();
    publish(&directory, &[2]);
    let packed = temp.path().join("packed.safetensors");
    conversion::pack(&directory, &packed).unwrap();
    let companions = companion_files(&directory);
    assert!(!companions.is_empty());
    for companion in &companions {
        fs::remove_file(companion).unwrap();
    }

    let value =
        unverified_metadata(&directory, options(false, false, false, false), Some(1)).unwrap();
    assert_eq!(value["core"]["signatures_verified"], false);
    assert_eq!(value["core"]["metadata"], "unverified");
    assert_eq!(value["model"]["model_id"], "unverified/model");
    assert_eq!(value["model"]["max_sequence_tokens"], 32);
    assert_eq!(value["model"]["prefill_chunk_size"], 4);
    assert_eq!(value["model"]["vocabulary"], 16);
    assert_eq!(value["model"]["max_active"], 7);
    assert_eq!(value["ranks"].as_array().unwrap().len(), 1);
    assert_eq!(value["ranks"][0]["rank"], 1);
    assert!(
        value["deployments"]
            .as_array()
            .unwrap()
            .iter()
            .any(|d| { d["id"] == "sm120-tp2" && d["tensor_parallel"] == 2 })
    );

    let memory =
        unverified_metadata(&directory, options(true, false, false, false), Some(1)).unwrap();
    assert_eq!(memory["ranks"][0]["allocations"][0]["name"], "device-0");
    assert_eq!(memory["ranks"][0]["reservations"][0]["address"], 0x2000);
    assert_eq!(memory["ranks"][0]["execution_buffers"][0]["name"], "input");
    assert_eq!(
        memory["ranks"][0]["initializers"][0]["allocation"],
        "device-0"
    );
    assert!(!memory.to_string().contains("decode"), "{memory}");

    let graphs =
        unverified_metadata(&directory, options(false, true, false, false), Some(1)).unwrap();
    assert_eq!(
        graphs["ranks"][0]["execution_graphs"][0]["batch_capacity"],
        1
    );
    assert_eq!(
        graphs["ranks"][0]["execution_graphs"][0]["estimated_duration_ns"],
        1234
    );
    assert_eq!(graphs["ranks"][0]["graphs"][0]["node_count"], 3);
    assert_eq!(graphs["ranks"][0]["graphs"][0]["edge_count"], 2);
    assert!(!graphs.to_string().contains("device-0"), "{graphs}");

    let frontend =
        unverified_metadata(&directory, options(false, false, true, false), Some(1)).unwrap();
    let detail = &frontend["deployments"][0]["frontend"];
    assert_eq!(detail["tokenizer"]["bos_token"], "bos");
    assert_eq!(detail["tokenizer"]["eos_token"], "eos");
    assert_eq!(detail["eos_token_ids"], serde_json::json!([1, 2, 3]));
    assert_eq!(detail["tools"]["kind"], "QWEN_CODER");
    assert_eq!(detail["tools"]["wrapping_newlines"], 1);
    assert_eq!(detail["tools"]["tag_separator"], "|");
    assert_eq!(detail["reasoning"]["kind"], "THINK_TAGS");
    assert_eq!(detail["reasoning"]["open"], "think");
    assert_eq!(detail["reasoning"]["close"], "/think");
    assert_eq!(
        detail["reasoning"]["terminal_markers"],
        serde_json::json!(["done"])
    );
    assert_eq!(detail["assistant_marker"], "assistant");
    assert_eq!(detail["message_end_markers"], serde_json::json!(["end"]));
    assert_eq!(detail["position_dimensions"], 1);
    let vision = &frontend["deployments"][0]["vision"];
    assert_eq!(vision["feature_slots"], 2);
    assert_eq!(vision["max_image_tokens_per_prompt"], 32);
    assert_eq!(vision["max_images"], 32);
    assert_eq!(vision["image_feature_window_tokens"], 1);
    assert_eq!(vision["patches_per_feature"], 3);
    assert_eq!(vision["patch_capacities"], serde_json::json!([64, 128]));
    assert_eq!(frontend["assets"][0]["name"], "tokenizer");
    assert!(frontend["assets"][0]["bytes"].as_u64().unwrap() > 0);
    assert!(frontend["assets"][0]["sha256"].as_str().is_some());

    let all = unverified_metadata(&directory, options(false, false, false, true), Some(1)).unwrap();
    assert!(all.to_string().contains("device-0"), "{all}");
    assert!(all.to_string().contains("decode"), "{all}");
    assert!(all.to_string().contains("tokenizer"), "{all}");

    let value = unverified_metadata(&packed, options(false, false, false, true), Some(1)).unwrap();
    assert_eq!(value["core"]["signatures_verified"], false);
    assert_eq!(value["core"]["metadata"], "unverified");
    assert_eq!(value["model"]["model_id"], "unverified/model");
    assert_eq!(value["model"]["max_sequence_tokens"], 32);
    assert_eq!(value["ranks"][0]["rank"], 1);
}

fn write_profile(path: &Path, publisher: &Publisher) {
    fs::write(
        path,
        serde_json::to_vec(std::slice::from_ref(publisher)).unwrap(),
    )
    .unwrap();
}

#[test]
fn authenticated_payload_uses_verified_counts_and_execution_fallback() {
    let temp = tempfile::tempdir().unwrap();
    let fallback = temp.path().join("fallback");
    fs::create_dir_all(&fallback).unwrap();
    let publisher = publish_configured(&fallback, &[(2, 32, 4, None, Some(4))]);
    let profile = temp.path().join("fallback-publisher.json");
    write_profile(&profile, &publisher);
    let path = fallback.to_str().unwrap();

    let summary = flare(&[
        "inspect",
        path,
        "--publisher-keys",
        profile.to_str().unwrap(),
        "--plain",
    ]);
    assert!(
        summary.status.success(),
        "{}",
        String::from_utf8_lossy(&summary.stderr)
    );
    let text = String::from_utf8(summary.stdout).unwrap();
    assert!(text.contains("signatures verified"), "{text}");
    assert!(text.contains("32 tokens"), "{text}");
    assert!(text.contains("4 requests"), "{text}");
    assert!(!text.contains("Graphs ·"), "{text}");
    assert!(!text.contains("Memory map"), "{text}");

    let summary = flare(&[
        "inspect",
        path,
        "--publisher-keys",
        profile.to_str().unwrap(),
        "--json",
    ]);
    assert!(
        summary.status.success(),
        "{}",
        String::from_utf8_lossy(&summary.stderr)
    );
    let summary: serde_json::Value = serde_json::from_slice(&summary.stdout).unwrap();
    assert_eq!(summary["core"]["signatures_verified"], true);
    assert_eq!(summary["model"]["max_sequence_tokens"], 32);
    assert_eq!(summary["model"]["max_active"], 4);
    assert_eq!(summary["ranks"].as_array().unwrap().len(), 2);
    assert_eq!(summary["ranks"][0]["rank"], 0);
    assert_eq!(summary["ranks"][1]["rank"], 1);
    assert_eq!(summary["ranks"][0]["allocations"], 1);
    assert_eq!(summary["ranks"][0]["graphs"], 1);
    assert!(summary["ranks"][0].get("execution_graphs").is_none());
    assert!(summary["ranks"][0].get("memory_map").is_none());

    let details = flare(&[
        "inspect",
        path,
        "--publisher-keys",
        profile.to_str().unwrap(),
        "--graphs",
        "--memory-map",
        "--plain",
    ]);
    assert!(
        details.status.success(),
        "{}",
        String::from_utf8_lossy(&details.stderr)
    );
    let text = String::from_utf8(details.stdout).unwrap();
    assert!(text.contains("Graphs · rank"), "{text}");
    assert!(text.contains("Memory map"), "{text}");
    let graph = text
        .lines()
        .find(|line| line.contains("decode"))
        .unwrap_or_else(|| panic!("{text}"));
    let fields: Vec<&str> = graph.split_whitespace().collect();
    assert!(fields.contains(&"3"), "node count: {graph}");
    assert!(fields.contains(&"2"), "edge count: {graph}");
    assert!(text.contains("device-0"), "{text}");
    let initializers = text
        .split("Initializers · rank 0")
        .nth(1)
        .unwrap_or_else(|| panic!("{text}"));
    let initializers = initializers
        .split('╰')
        .next()
        .unwrap_or_else(|| panic!("{text}"));
    assert!(initializers.contains("weights/tp2/rank0"), "{text}");
    assert!(initializers.contains(".safetensors"), "{text}");
    assert!(initializers.contains("flare.payload"), "{text}");
    assert!(initializers.contains("0 + 128"), "{text}");
    assert!(!initializers.contains("primary"), "{text}");

    let explicit = temp.path().join("explicit");
    fs::create_dir_all(&explicit).unwrap();
    let publisher = publish_configured(&explicit, &[(2, 32, 4, Some(7), Some(4))]);
    let profile = temp.path().join("explicit-publisher.json");
    write_profile(&profile, &publisher);
    let output = flare(&[
        "inspect",
        explicit.to_str().unwrap(),
        "--publisher-keys",
        profile.to_str().unwrap(),
        "--json",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["model"]["max_active"], 7);
}

#[test]
fn unverified_payload_plain_renders_capacities_and_real_detail_names() {
    let temp = tempfile::tempdir().unwrap();
    let directory = temp.path().join("core");
    fs::create_dir_all(&directory).unwrap();
    publish(&directory, &[2]);
    let path = directory.to_str().unwrap();

    let default = flare(&["inspect", path, "--unverified", "--plain"]);
    assert!(
        default.status.success(),
        "{}",
        String::from_utf8_lossy(&default.stderr)
    );
    let text = String::from_utf8(default.stdout).unwrap();
    let summary = plain_section(&text, "unverified/model");
    assert_plain_row(summary, "Format", "payload");
    assert_plain_row(summary, "Signatures", "unverified");
    assert_plain_row(summary, "Publisher", "unverified-test");
    let deployment = plain_section(&text, "Deployment sm120-tp2");
    assert_plain_row(deployment, "Context", "32 tokens");
    assert_plain_row(deployment, "Concurrency", "7 requests");
    assert!(!deployment.contains("tokens tokens"), "{text}");
    assert!(!text.contains("Graphs ·"), "{text}");
    assert!(!text.contains("Memory map"), "{text}");

    let graphs = flare(&["inspect", path, "--unverified", "--plain", "--graphs"]);
    assert!(
        graphs.status.success(),
        "{}",
        String::from_utf8_lossy(&graphs.stderr)
    );
    let text = String::from_utf8(graphs.stdout).unwrap();
    let graph_heads = section_identities(&text, "Graphs ·");
    assert_eq!(
        graph_heads.into_iter().collect::<BTreeSet<_>>(),
        BTreeSet::from([("sm120-tp2".to_owned(), 0), ("sm120-tp2".to_owned(), 1),]),
        "{text}"
    );
    for rank in 0..2 {
        assert_plain_graph(&text, "sm120-tp2", rank);
    }

    let memory = flare(&["inspect", path, "--unverified", "--plain", "--memory-map"]);
    assert!(
        memory.status.success(),
        "{}",
        String::from_utf8_lossy(&memory.stderr)
    );
    let text = String::from_utf8(memory.stdout).unwrap();
    let memory_heads = section_identities(&text, "Memory map ·");
    assert_eq!(
        memory_heads.into_iter().collect::<BTreeSet<_>>(),
        BTreeSet::from([("sm120-tp2".to_owned(), 0), ("sm120-tp2".to_owned(), 1),]),
        "{text}"
    );
    for rank in 0..2 {
        assert_plain_memory(&text, "sm120-tp2", rank);
    }

    let frontend = flare(&["inspect", path, "--unverified", "--plain", "--frontend"]);
    assert!(
        frontend.status.success(),
        "{}",
        String::from_utf8_lossy(&frontend.stderr)
    );
    let text = String::from_utf8(frontend.stdout).unwrap();
    assert!(text.contains("tokenizer"), "{text}");
}

#[test]
fn inspect_selects_a_tensor_parallel_deployment_for_payload_cores() {
    let temp = tempfile::tempdir().unwrap();
    let directory = temp.path().join("core");
    fs::create_dir_all(&directory).unwrap();
    let publisher = publish_configured(
        &directory,
        &[(2, 32, 4, Some(7), Some(7)), (4, 64, 8, Some(9), Some(9))],
    );
    let profile = directory.join("publisher.json");
    fs::write(&profile, serde_json::to_vec(&vec![publisher]).unwrap()).unwrap();
    let path = directory.to_str().unwrap();
    let profile = profile.to_str().unwrap();

    let selected = flare(&[
        "inspect",
        path,
        "--publisher-keys",
        profile,
        "--tensor-parallel",
        "4",
        "--json",
    ]);
    assert!(
        selected.status.success(),
        "{}",
        String::from_utf8_lossy(&selected.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&selected.stdout).unwrap();
    assert_eq!(value["model"]["max_sequence_tokens"], 64);
    assert_eq!(value["model"]["prefill_chunk_size"], 8);
    assert_eq!(value["ranks"].as_array().unwrap().len(), 4);

    let default = flare(&["inspect", path, "--publisher-keys", profile, "--json"]);
    assert!(
        !default.status.success(),
        "multi-deployment default should require a tensor-parallel selection"
    );
    let error = String::from_utf8_lossy(&default.stderr);
    assert!(error.contains("select a tensor-parallel size"), "{error}");

    let unverified = flare(&[
        "inspect",
        path,
        "--unverified",
        "--tensor-parallel",
        "4",
        "--json",
    ]);
    assert!(
        unverified.status.success(),
        "{}",
        String::from_utf8_lossy(&unverified.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&unverified.stdout).unwrap();
    assert_eq!(value["deployments"].as_array().unwrap().len(), 1);
    assert_eq!(value["deployments"][0]["id"], "sm120-tp4");
    assert_eq!(value["selected_tensor_parallel"], 4);
    assert_eq!(value["rank_count"], 4);

    let missing = flare(&[
        "inspect",
        path,
        "--unverified",
        "--tensor-parallel",
        "3",
        "--json",
    ]);
    assert!(!missing.status.success());
    let error = String::from_utf8_lossy(&missing.stderr);
    assert!(error.contains("available: [2, 4]"), "{error}");
}

#[test]
fn unverified_payload_multi_deployment_capacities_and_details_are_qualified() {
    let temp = tempfile::tempdir().unwrap();
    let directory = temp.path().join("core");
    fs::create_dir_all(&directory).unwrap();
    publish_configured(
        &directory,
        &[(2, 32, 4, Some(7), Some(7)), (4, 64, 8, Some(9), Some(9))],
    );

    let value = unverified_metadata(&directory, options(false, false, false, false), None).unwrap();
    let deployments = value["deployments"].as_array().unwrap();
    let tp2 = deployments
        .iter()
        .find(|deployment| deployment["id"] == "sm120-tp2")
        .unwrap();
    let tp4 = deployments
        .iter()
        .find(|deployment| deployment["id"] == "sm120-tp4")
        .unwrap();
    assert_eq!(tp2["max_sequence_tokens"], 32);
    assert_eq!(tp2["prefill_chunk_size"], 4);
    assert_eq!(tp2["max_active"], 7);
    assert_eq!(tp4["max_sequence_tokens"], 64);
    assert_eq!(tp4["prefill_chunk_size"], 8);
    assert_eq!(tp4["max_active"], 9);
    assert_eq!(tp2["ranks"], serde_json::json!([0, 1]));
    assert_eq!(tp4["ranks"], serde_json::json!([0, 1, 2, 3]));

    let path = directory.to_str().unwrap();
    let summary = flare(&["inspect", path, "--unverified", "--plain"]);
    assert!(
        summary.status.success(),
        "{}",
        String::from_utf8_lossy(&summary.stderr)
    );
    let text = String::from_utf8(summary.stdout).unwrap();
    let model = plain_section(&text, "unverified/model");
    assert_plain_row(model, "Format", "payload");
    assert_plain_row(model, "Signatures", "unverified");
    for (deployment, context, prefill, concurrency) in [
        ("sm120-tp2", "32 tokens", "4 tokens", "7 requests"),
        ("sm120-tp4", "64 tokens", "8 tokens", "9 requests"),
    ] {
        let section = plain_section(&text, &format!("Deployment {deployment}"));
        assert_plain_row(section, "Context", context);
        assert_plain_row(section, "Prefill", prefill);
        assert_plain_row(section, "Concurrency", concurrency);
    }

    let details = flare(&[
        "inspect",
        path,
        "--unverified",
        "--plain",
        "--graphs",
        "--memory-map",
    ]);
    assert!(
        details.status.success(),
        "{}",
        String::from_utf8_lossy(&details.stderr)
    );
    let text = String::from_utf8(details.stdout).unwrap();
    let expected: BTreeSet<(String, usize)> = [
        ("sm120-tp2", 0),
        ("sm120-tp2", 1),
        ("sm120-tp4", 0),
        ("sm120-tp4", 1),
        ("sm120-tp4", 2),
        ("sm120-tp4", 3),
    ]
    .into_iter()
    .map(|(deployment, rank)| (deployment.to_owned(), rank))
    .collect();
    for title in ["Graphs ·", "Memory map ·"] {
        let identified = section_identities(&text, title)
            .into_iter()
            .collect::<BTreeSet<_>>();
        assert_eq!(identified, expected, "{title}: {text}");
    }
    for (deployment, rank) in expected {
        assert_plain_graph(&text, &deployment, rank);
        assert_plain_memory(&text, &deployment, rank);
    }
}

#[test]
fn authenticated_payload_plain_uses_the_verified_renderer() {
    let temp = tempfile::tempdir().unwrap();
    let directory = temp.path().join("core");
    fs::create_dir_all(&directory).unwrap();
    let publisher = publish(&directory, &[2]);
    let profile = temp.path().join("publisher.json");
    fs::write(&profile, serde_json::to_vec(&[publisher]).unwrap()).unwrap();
    let output = flare(&[
        "inspect",
        directory.to_str().unwrap(),
        "--publisher-keys",
        profile.to_str().unwrap(),
        "--plain",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("signatures verified"), "{text}");
    assert!(text.contains("32 tokens"), "{text}");
    assert!(!text.contains("(unverified)"), "{text}");
}

#[test]
fn unverified_payload_rank_selection_rejects_unknown_and_ambiguous_ranks() {
    let temp = tempfile::tempdir().unwrap();
    let directory = temp.path().join("core");
    fs::create_dir_all(&directory).unwrap();
    publish(&directory, &[2]);
    assert!(
        unverified_metadata(&directory, options(false, false, false, false), Some(3)).is_err(),
        "an absent rank must not be reported"
    );
    assert!(
        unverified_metadata(
            &directory,
            options(false, false, false, false),
            Some(u32::MAX as usize + 1),
        )
        .is_err(),
        "a rank outside the u32 rank domain must not wrap to rank zero"
    );

    let ambiguous = temp.path().join("ambiguous");
    fs::create_dir_all(&ambiguous).unwrap();
    publish(&ambiguous, &[2, 4]);
    let error = unverified_metadata(&ambiguous, options(false, false, false, false), Some(1))
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("ambiguous") || error.contains("deployment"),
        "{error}"
    );
}
