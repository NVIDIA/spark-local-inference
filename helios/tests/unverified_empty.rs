/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Unverified inspection of a structurally empty payload envelope must stay bounded.

#![cfg(feature = "runtime-core")]

use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

use flare_helios::core::{container, reader, schema, schema::wire};
use flare_helios::runtime_core::{InspectionOptions, unverified_metadata};
use prost::Message;

#[test]
fn empty_payload_envelope_does_not_panic() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = std::env::temp_dir().join(format!(
        "flare-unverified-empty-{}-{unique}",
        std::process::id()
    ));
    fs::create_dir(&directory).unwrap();
    let path = directory.join("empty.safetensors");
    let core = wire::CoreBody {
        model_identity: "empty".into(),
        hardware_identity: "sm120".into(),
        ..Default::default()
    };
    let envelope = wire::ProtectedCore {
        version: schema::FORMAT_VERSION,
        core_id: vec![7; 32],
        signer: vec![8; 32],
        core_body: core.encode_to_vec(),
        ..Default::default()
    };
    let locations = wire::LocationSet {
        version: schema::FORMAT_VERSION,
        ..Default::default()
    };
    let mut writer =
        container::Writer::new(fs::File::create(&path).unwrap(), container::PAYLOAD_FORMAT)
            .unwrap();
    let manifest = envelope.encode_to_vec();
    writer
        .add(
            reader::MANIFEST,
            manifest.len() as u64,
            container::Source::Bytes(manifest),
        )
        .unwrap();
    let locations = locations.encode_to_vec();
    writer
        .add(
            reader::LOCATIONS,
            locations.len() as u64,
            container::Source::Bytes(locations),
        )
        .unwrap();
    writer.finish().unwrap();

    // An error is acceptable for a structurally empty catalog; unwinding is not.
    if let Ok(value) = unverified_metadata(&path, InspectionOptions::default(), None) {
        assert_eq!(
            value["deployments"].as_array().map(Vec::len),
            Some(0)
        );
        assert_eq!(value["ranks"].as_array().map(Vec::len), Some(0));
        assert_eq!(value["core"]["signatures_verified"], false);
        assert_eq!(value["core"]["metadata"], "unverified");
    }
    let _ = fs::remove_dir_all(directory);
}
