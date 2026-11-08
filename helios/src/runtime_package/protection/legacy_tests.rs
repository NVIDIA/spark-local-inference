/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use ring::{
    aead, hkdf,
    signature::{Ed25519KeyPair, KeyPair},
};
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

const EXECUTABLE_OFFSET: usize = 64;

struct Fixture {
    path: PathBuf,
    directory: Option<PathBuf>,
    envelope: wire::ProtectedIndex,
    payload: Vec<u8>,
    plain: Vec<u8>,
    publisher: Publisher,
    key: Ed25519KeyPair,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
        if let Some(directory) = &self.directory {
            let _ = std::fs::remove_dir_all(directory);
        }
    }
}

impl Fixture {
    fn new(version: u32, length: usize) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "flare-signed-reader-{}-{}.safetensors",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let key = Ed25519KeyPair::from_seed_unchecked(&[42; 32]).unwrap();
        let publisher = Publisher {
            name: "Flare test publisher".into(),
            namespace: "flare-tests".into(),
            public_key: key.public_key().as_ref().try_into().unwrap(),
            development: false,
        };
        let package_id = vec![7; 32];
        let plain: Vec<u8> = (0..length).map(|i| (i % 251) as u8).collect();
        let (payload, hashes) = if version == 4 {
            let cipher = fixture_key(
                &publisher.public_key,
                &package_id,
                EXECUTABLE_OFFSET as u64,
                &publisher.namespace,
            )
            .unwrap();
            let mut payload = Vec::new();
            let mut hashes = Vec::new();
            for (i, chunk) in plain.chunks(CHUNK).enumerate() {
                let mut bytes = chunk.to_vec();
                let mut nonce = [0; 12];
                nonce[4..].copy_from_slice(&(i as u64).to_be_bytes());
                cipher
                    .seal_in_place_append_tag(
                        aead::Nonce::assume_unique_for_key(nonce),
                        aead::Aad::from(fixture_aad(
                            &package_id,
                            EXECUTABLE_OFFSET as u64,
                            length as u64,
                            i as u64,
                        )),
                        &mut bytes,
                    )
                    .unwrap();
                hashes.push(sha256(&bytes));
                payload.extend(bytes);
            }
            (payload, hashes)
        } else {
            let payload = zstd::stream::encode_all(plain.as_slice(), 3).unwrap();
            let hashes = payload.chunks(CHUNK).map(sha256).collect();
            (payload, hashes)
        };
        let stored_bytes = payload.len() as u64;
        let supporting = vec![13; EXECUTABLE_OFFSET];
        let supporting_hash = sha256(&supporting);
        let payload = [supporting, payload].concat();
        let span = wire::Span {
            offset: EXECUTABLE_OFFSET as u64,
            nbytes: stored_bytes,
            filename: String::new(),
        };
        let bundle = wire::BundleIndex {
            modules: vec![wire::Module {
                name: "kernel".into(),
                file: "kernel".into(),
                globals: vec![],
            }],
            resources: [("kernel".into(), span)].into_iter().collect(),
            ..Default::default()
        };
        let data = wire::StoredData {
            sha256: sha256(&[]),
            ..Default::default()
        };
        let support_span = wire::Span {
            offset: 0,
            nbytes: EXECUTABLE_OFFSET as u64,
            filename: String::new(),
        };
        let support = wire::ResourceCatalog {
            resources: vec![wire::ProtectedResource {
                offset: 0,
                stored_bytes: EXECUTABLE_OFFSET as u64,
                plain_bytes: EXECUTABLE_OFFSET as u64,
                sha256: vec![supporting_hash.clone()],
            }],
        }
        .encode_to_vec();
        let data_body = data.encode_to_vec();
        let metadata = wire::PublisherMetadata {
            index: Some(wire::ModelIndex {
                version: 4,
                assets: vec![wire::Asset {
                    name: "fixture.text".into(),
                    payload: Some(support_span),
                    sha256: hex(&supporting_hash),
                }],
                contract: Some(wire::ServingContract {
                    model_id: "fixture".into(),
                    model_contract: Some(wire::ModelContract::default()),
                    ..Default::default()
                }),
                frontend: Some(wire::FrontendSettings::default()),
                sampling_defaults: Some(wire::DefaultSamplingParameters::default()),
                ..Default::default()
            }),
            namespace: publisher.namespace.clone(),
            official_source: "https://huggingface.co/flare-tests/model".into(),
            ranks: 1,
            group_hashes: vec![sha256(&support), sha256(&data_body)],
            ..Default::default()
        };
        let rank = wire::RankProtection {
            bundle: Some(bundle),
            data: Some(data),
            executables: Some(wire::ResourceCatalog {
                resources: vec![wire::ProtectedResource {
                    offset: EXECUTABLE_OFFSET as u64,
                    stored_bytes,
                    plain_bytes: length as u64,
                    sha256: hashes,
                }],
            }),
            ..Default::default()
        };
        let envelope = wire::ProtectedIndex {
            version,
            package_id,
            signer: sha256(&publisher.public_key),
            groups: vec![(1, metadata.encode_to_vec()), (2, support), (3, data_body)]
                .into_iter()
                .map(|(kind, body)| wire::SignedGroup {
                    kind,
                    body,
                    signature: vec![],
                })
                .collect(),
            ranks: vec![wire::SignedGroup {
                kind: 4,
                body: rank.encode_to_vec(),
                signature: vec![],
            }],
        };
        let mut fixture = Self {
            path,
            directory: None,
            envelope,
            payload,
            plain,
            publisher,
            key,
        };
        fixture.sign();
        fixture.write();
        fixture
    }
    fn sign(&mut self) {
        for group in self
            .envelope
            .groups
            .iter_mut()
            .chain(&mut self.envelope.ranks)
        {
            let domain = match self.envelope.version {
                4 => b"flare-package-signature-v4",
                6 => b"flare-package-signature-v6",
                _ => panic!("fixture version"),
            };
            let message = fields(&[
                domain,
                &self.envelope.version.to_le_bytes(),
                &self.envelope.package_id,
                &self.envelope.signer,
                &group.kind.to_le_bytes(),
                &group.body,
            ]);
            group.signature = self.key.sign(&message).as_ref().to_vec();
        }
    }
    fn write(&self) {
        let envelope = self.envelope.encode_to_vec();
        let payload = self.payload.len();
        let mut header = serde_json::to_vec(&serde_json::json!({
            "__metadata__": {"format": "flare"},
            "flare.payload": {"dtype": "U8", "shape": [payload], "data_offsets": [0, payload]},
            "flare.metadata": {"dtype": "U8", "shape": [envelope.len()], "data_offsets": [payload, payload + envelope.len()]},
        })).unwrap();
        header.resize(4088, b' ');
        let mut file = File::create(&self.path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&self.payload).unwrap();
        file.write_all(&envelope).unwrap();
    }
    fn verify(&self) -> Result<LegacyVerified> {
        LegacyVerified::open(&self.path, std::slice::from_ref(&self.publisher), None)
    }
    fn read(&self) -> Result<Vec<u8>> {
        let authorized = self.verify()?.authorize(&self.publisher.namespace, false)?;
        let archive = authorized.legacy_archive().unwrap();
        let span = &authorized.legacy_index().unwrap().ranks[0].resources["kernel"];
        let (file, start) = archive.source(span)?;
        let mut reader = archive.protection.reader(file.file.clone(), start, span)?;
        assert_eq!(reader.read(&mut []).unwrap(), 0);
        let mut result = Vec::new();
        reader
            .read_to_end(&mut result)
            .map_err(|error| error.to_string())?;
        Ok(result)
    }
}

#[test]
fn signed_formats_four_and_six_load_from_directories_with_companions() {
    for version in [4, 6] {
        let mut fixture = Fixture::new(version, 256);
        let directory = fixture.path.with_extension("");
        std::fs::create_dir(&directory).unwrap();
        let primary = directory.join("model.safetensors");
        std::fs::rename(&fixture.path, &primary).unwrap();
        fixture.path = primary;
        fixture.directory = Some(directory.clone());
        for companion in [None, Some(format!("{}.safetensors", "a".repeat(64)))] {
            if let Some(name) = &companion {
                let envelope = fixture.envelope.encode_to_vec();
                let layout = serde_json::to_vec(&serde_json::json!({
                    "version": 1, "files": {"": [[fixture.payload.len(), name]]},
                })).unwrap();
                let mut header = serde_json::to_vec(&serde_json::json!({
                    "__metadata__": {"format": "flare-snapshot"},
                    "flare.payload": {"dtype": "U8", "shape": [0], "data_offsets": [0, 0]},
                    "flare.metadata": {"dtype": "U8", "shape": [envelope.len()], "data_offsets": [0, envelope.len()]},
                    "flare.layout": {"dtype": "U8", "shape": [layout.len()], "data_offsets": [envelope.len(), envelope.len() + layout.len()]},
                })).unwrap();
                header.resize(4088, b' ');
                let mut file = File::create(&fixture.path).unwrap();
                file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
                file.write_all(&header).unwrap();
                file.write_all(&envelope).unwrap();
                file.write_all(&layout).unwrap();
                let length = fixture.payload.len();
                let mut header = serde_json::to_vec(&serde_json::json!({
                    "__metadata__": {"format": "flare-data"},
                    "flare.payload": {"dtype": "U8", "shape": [length], "data_offsets": [0, length]},
                })).unwrap();
                header.resize(4088, b' ');
                let mut file = File::create(directory.join(name)).unwrap();
                file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
                file.write_all(&header).unwrap();
                file.write_all(&fixture.payload).unwrap();
            }
            for path in [&fixture.path, &directory] {
                let claim = PackageClaim::read(path).unwrap();
                assert_eq!(claim.signer, fixture.publisher.fingerprint());
                assert_eq!(claim.namespace, fixture.publisher.namespace);
                assert_eq!(
                    super::super::package_companions(path).unwrap(),
                    companion.iter().cloned().collect::<Vec<_>>()
                );
                let verified = VerifiedPackage::open(
                    path, std::slice::from_ref(&fixture.publisher), None,
                ).unwrap();
                let authorized = verified.authorize(&fixture.publisher.namespace, false).unwrap();
                let archive = authorized.legacy_archive().unwrap();
                let span = &authorized.legacy_index().unwrap().ranks[0].resources["kernel"];
                let (file, start) = archive.source(span).unwrap();
                let mut reader = archive.protection.reader(file.file.clone(), start, span).unwrap();
                let mut bytes = Vec::new();
                reader.read_to_end(&mut bytes).unwrap();
                assert_eq!(bytes, fixture.plain);
            }
        }
    }
}

#[test]
fn signed_formats_four_and_six_require_acceptance_and_read_executables() {
    for version in [4, 6] {
        let fixture = Fixture::new(version, CHUNK + 37);
        assert!(
            super::super::package_companions(&fixture.path)
                .unwrap()
                .is_empty()
        );
        let metadata = super::super::unverified_metadata(
            &fixture.path,
            super::super::InspectionOptions::default(),
            None,
        )
        .unwrap();
        assert_eq!(metadata["package"]["signatures_verified"], false);
        assert_eq!(metadata["model"]["model_id"], "fixture");
        let claim = PackageClaim::read(&fixture.path).unwrap();
        assert_eq!(claim.signer, fixture.publisher.fingerprint());
        assert_eq!(claim.namespace, fixture.publisher.namespace);
        let verified = fixture.verify().unwrap();
        let span = &verified.index.ranks[0].resources["kernel"];
        let (file, start) = verified.archive.source(span).unwrap();
        assert!(
            verified
                .archive
                .protection
                .reader(file.file.clone(), start, span)
                .is_err()
        );
        assert!(
            fixture
                .verify()
                .unwrap()
                .authorize("another-publisher", false)
                .is_err()
        );
        assert_eq!(fixture.read().unwrap(), fixture.plain);
    }
}

#[test]
fn signed_format_four_rejects_changed_signatures_and_version() {
    for rank in [false, true] {
        let mut fixture = Fixture::new(4, 256);
        let group = if rank {
            &mut fixture.envelope.ranks[0]
        } else {
            &mut fixture.envelope.groups[0]
        };
        group.signature[0] ^= 1;
        fixture.write();
        assert!(
            fixture
                .verify()
                .err()
                .unwrap()
                .to_string()
                .contains("publisher signature failed")
        );
    }
    let mut fixture = Fixture::new(4, 256);
    fixture.envelope.version = 6;
    fixture.write();
    assert!(
        fixture
            .verify()
            .err()
            .unwrap()
            .to_string()
            .contains("publisher signature failed")
    );
}

#[test]
fn signed_format_four_checks_ciphertext_hashes_and_authentication_tags() {
    for resign in [false, true] {
        let mut fixture = Fixture::new(4, CHUNK + 37);
        fixture.payload[EXECUTABLE_OFFSET + CHUNK + 17] ^= 1;
        if resign {
            let mut rank =
                wire::RankProtection::decode(fixture.envelope.ranks[0].body.as_slice()).unwrap();
            rank.executables.as_mut().unwrap().resources[0].sha256[1] =
                sha256(&fixture.payload[EXECUTABLE_OFFSET + CHUNK + 16..]);
            fixture.envelope.ranks[0].body = rank.encode_to_vec();
            fixture.sign();
        }
        fixture.write();
        let error = fixture.read().unwrap_err().to_string();
        assert!(
            error.contains(if resign {
                "executable decryption failed"
            } else {
                "publisher resource content verification failed"
            }),
            "{error}"
        );
    }
}

#[test]
fn signed_format_four_validates_encrypted_catalog_lengths() {
    let mut fixture = Fixture::new(4, 256);
    let mut rank = wire::RankProtection::decode(fixture.envelope.ranks[0].body.as_slice()).unwrap();
    rank.executables.as_mut().unwrap().resources[0].plain_bytes += 1;
    fixture.envelope.ranks[0].body = rank.encode_to_vec();
    fixture.sign();
    fixture.write();
    assert!(
        fixture
            .verify()
            .err()
            .unwrap()
            .to_string()
            .contains("invalid protected resource catalog")
    );
}

fn fixture_key(
    public_key: &[u8],
    package: &[u8],
    offset: u64,
    namespace: &str,
) -> Result<aead::LessSafeKey> {
    let info = fields(&[
        b"flare-executable-v3",
        &offset.to_le_bytes(),
        namespace.as_bytes(),
    ]);
    let prk = hkdf::Salt::new(hkdf::HKDF_SHA256, package).extract(public_key);
    let parts = [info.as_slice()];
    let mut key = [0; 32];
    prk.expand(&parts, hkdf::HKDF_SHA256)
        .unwrap()
        .fill(&mut key)
        .unwrap();
    Ok(aead::LessSafeKey::new(
        aead::UnboundKey::new(&aead::AES_256_GCM, &key).unwrap(),
    ))
}
fn fixture_aad(package: &[u8], offset: u64, length: u64, chunk: u64) -> Vec<u8> {
    fields(&[
        b"flare-executable-chunk-v3",
        package,
        &offset.to_le_bytes(),
        &length.to_le_bytes(),
        &chunk.to_le_bytes(),
    ])
}
