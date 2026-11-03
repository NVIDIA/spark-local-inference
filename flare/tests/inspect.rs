/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::{
    path::PathBuf,
    process::{Command, Output},
};

fn flare(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_flare"))
        .args(arguments)
        .arg("--unverified")
        .output()
        .unwrap()
}

fn varint(mut n: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    while n >= 128 {
        bytes.push(n as u8 | 128);
        n >>= 7;
    }
    bytes.push(n as u8);
    bytes
}

fn integer(field: u64, n: u64) -> Vec<u8> {
    [varint(field << 3), varint(n)].concat()
}

fn message(field: u64, bytes: &[u8]) -> Vec<u8> {
    [
        varint(field << 3 | 2),
        varint(bytes.len() as u64),
        bytes.to_vec(),
    ]
    .concat()
}

fn fixture(directory: &std::path::Path, version: u64, maximum: u64) -> PathBuf {
    let allocation = [
        message(1, b"cache"),
        integer(3, 256),
        integer(7, 0x200000),
        message(
            9,
            &[integer(1, 1), integer(2, 300), integer(3, maximum)].concat(),
        ),
    ]
    .concat();
    let fixed = [
        message(1, b"weights"),
        integer(2, 4096),
        integer(3, 256),
        message(4, b"weights.bin"),
        message(5, &integer(2, 4096)),
    ]
    .concat();
    let payload = [
        message(1, b"weights.bin"),
        message(
            2,
            &[integer(2, 4096), message(3, b"absent-data.safetensors")].concat(),
        ),
    ]
    .concat();
    let graph = [message(1, b"decode-1"), integer(4, 12), integer(5, 11)].concat();
    let execution_graph = [
        message(1, b"decode-1"),
        message(2, b"decode"),
        message(3, b"sample"),
        integer(4, 1),
        integer(5, 1),
    ]
    .concat();
    let bundle = [
        message(1, &[12, 0]),
        message(2, &allocation),
        message(2, &fixed),
        message(5, &graph),
        message(7, &[integer(2, 4), message(5, &execution_graph)].concat()),
        message(9, &payload),
    ]
    .concat();
    let contract = [
        message(1, b"owner/model"),
        message(3, &integer(2, 128)),
        integer(4, 32768),
        integer(5, 4096),
        integer(16, 2),
        message(17, b"local-model"),
    ]
    .concat();
    // Optional explicit zeroes and absent fields must remain distinguishable.
    let sampling = [
        integer(1, u64::MAX),
        varint(2 << 3 | 5),
        0f32.to_le_bytes().to_vec(),
        integer(3, u64::MAX),
    ]
    .concat();
    let index = [
        integer(1, version),
        message(7, &sampling),
        message(8, &[]),
        message(9, &integer(1, 2)),
        message(10, &contract),
    ]
    .concat();
    let metadata = [
        message(1, &index),
        message(2, b"example"),
        message(3, b"https://huggingface.co/example/model"),
        integer(6, 2),
    ]
    .concat();
    let group = |kind, body: &[u8]| message(4, &[integer(1, kind), message(2, body)].concat());
    let rank = |n| {
        message(
            5,
            &[
                integer(1, 4),
                message(2, &[integer(1, n), message(2, &bundle)].concat()),
            ]
            .concat(),
        )
    };
    let envelope = [
        integer(1, 6),
        group(1, &metadata),
        group(2, &[]),
        group(3, &[]),
        rank(0),
        rank(1),
    ]
    .concat();
    let header = serde_json::json!({
        "__metadata__": {"format": "flare"},
        "flare.payload": {"dtype": "U8", "shape": [0], "data_offsets": [0, 0]},
        "flare.metadata": {"dtype": "U8", "shape": [envelope.len()], "data_offsets": [0, envelope.len()]},
    });
    let mut bytes = 4088u64.to_le_bytes().to_vec();
    bytes.extend(serde_json::to_vec(&header).unwrap());
    bytes.resize(4096, b' ');
    bytes.extend(envelope);
    let path = directory.join(format!("model-{version}.safetensors"));
    std::fs::write(&path, bytes).unwrap();
    path
}

#[test]
fn inspection_reads_primary_metadata_without_cuda_trust_or_companions() {
    let temp = tempfile::tempdir().unwrap();
    for version in [4, 5] {
        let path = fixture(temp.path(), version, 10);
        let path = path.to_str().unwrap();
        let basic = flare(&["inspect", path]);
        assert!(basic.status.success(), "{basic:?}");
        assert!(basic.stderr.is_empty(), "{basic:?}");
        let text = String::from_utf8(basic.stdout).unwrap();
        for expected in [
            "local-model",
            "32,768",
            "Sampling",
            "unverified",
            "Rank resources",
            "2 requests",
        ] {
            assert!(text.contains(expected), "{expected}: {text}");
        }
        assert!(!text.contains("owner/model"));
        assert!(!text.contains("Graphs · rank"));
        assert!(!text.contains("Memory map"));
        for (flag, expected, omitted) in [
            ("--memory-map", "Memory map", "Graphs · rank"),
            ("--graphs", "Graphs · rank", "Memory map"),
            ("--all", "FRONTEND", "never-present"),
        ] {
            let output = flare(&["inspect", path, flag]);
            assert!(output.status.success(), "{output:?}");
            let text = String::from_utf8(output.stdout).unwrap();
            assert!(text.contains(expected), "{text}");
            assert!(!text.contains(omitted), "{text}");
            if flag == "--all" {
                assert!(text.contains("owner/model"), "{text}");
            }
            if flag == "--memory-map" {
                assert!(text.contains("0x0000000000200000"), "{text}");
                assert!(text.contains("max 10 units"), "{text}");
                assert!(text.contains("absent-data.safetensors"), "{text}");
            }
        }
        let summary = flare(&["inspect", path, "--json"]);
        assert!(summary.status.success(), "{summary:?}");
        let summary: serde_json::Value = serde_json::from_slice(&summary.stdout).unwrap();
        assert!(summary.get("frontend").is_none());
        assert!(summary["ranks"][0].get("memory_map").is_none());
        assert!(summary["ranks"][0].get("execution_graphs").is_none());
        let output = flare(&["inspect", path, "--all", "--json"]);
        assert!(output.status.success(), "{output:?}");
        let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(json["package"]["signatures_verified"], false);
        assert_eq!(json["sampling_defaults"]["temperature"], 0.0);
        assert_eq!(json["sampling_defaults"]["seed"], u64::MAX);
        assert_eq!(json["sampling_defaults"]["top_k"], -1);
        assert!(json["sampling_defaults"]["top_p"].is_null());
        assert_eq!(
            json["ranks"][0]["memory_map"]["allocations"][0]["virtual_bytes"],
            3072
        );
        assert_eq!(json["ranks"][1]["execution_graphs"][0]["packed_tokens"], 1);
    }
}

#[test]
fn rank_selection_preserves_the_package_rank_number() {
    let temp = tempfile::tempdir().unwrap();
    let path = fixture(temp.path(), 5, 10);
    let output = flare(&[
        "inspect",
        path.to_str().unwrap(),
        "--rank",
        "1",
        "--graphs",
        "--json",
    ]);
    assert!(output.status.success(), "{output:?}");
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json["ranks"].as_array().unwrap().len(), 1);
    assert_eq!(json["ranks"][0]["rank"], 1);
    assert_eq!(json["package"]["cluster"]["world"], 2);
}

#[test]
fn inspection_authenticates_unless_unverified_was_explicitly_selected() {
    let temp = tempfile::tempdir().unwrap();
    let path = fixture(temp.path(), 5, 10);
    let output = Command::new(env!("CARGO_BIN_EXE_flare"))
        .args(["inspect", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let output = Command::new(env!("CARGO_BIN_EXE_flare"))
        .args([
            "inspect",
            path.to_str().unwrap(),
            "--unverified",
            "--publisher-keys",
            "/nonexistent/profile",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn invalid_packages_fail_before_loading() {
    let temp = tempfile::tempdir().unwrap();
    let path = fixture(temp.path(), 5, u64::MAX);
    let output = flare(&["inspect", path.to_str().unwrap(), "--memory-map"]);
    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("allocation extent overflow"), "{error}");
    std::fs::write(&path, 65537u64.to_le_bytes()).unwrap();
    let output = flare(&["inspect", path.to_str().unwrap()]);
    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("invalid safetensors header size"), "{error}");
    let missing = "/nonexistent/package.safetensors";
    // The operating system words the missing-path error.
    let reason = std::fs::metadata(missing).unwrap_err().to_string();
    let output = flare(&["inspect", missing]);
    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains(&reason), "{error}");
}
