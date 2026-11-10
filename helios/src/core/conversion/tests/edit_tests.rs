/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;

const TEMPLATE: &str = "{{ reasoning_effort|default('xhigh') }}{%- mark -%}";
const MEDIUM: &str = "{{ reasoning_effort|default('medium') }}{%- mark -%}";

fn editable_fixture(
    scratch: &Scratch,
    key: &signature::Ed25519KeyPair,
    derived_template: bool,
) -> Fixture {
    let mut source = fixture(scratch);
    let meta = Metadata::open(&source.directory).unwrap();
    let envelope = protect::envelope(&meta.envelope).unwrap();
    let mut core = wire::CoreBody::decode(envelope.core_body.as_slice()).unwrap();
    let id = PayloadId::of(TEMPLATE.as_bytes());
    core.payloads.push(descriptor(id, TEMPLATE.as_bytes()));
    core.assets.push(wire::FrontendAsset {
        name: "chat_template".into(),
        payload: Some(wire::PayloadUse {
            slice: Some(slice(id, TEMPLATE.len() as u64)),
            role: wire::PayloadRole::Frontend as i32,
        }),
        sha256: id.as_bytes().to_vec(),
    });
    let mut derivations = meta.derivations.clone().unwrap();
    if derived_template {
        let prefix = &TEMPLATE.as_bytes()[..7];
        let prefix_id = PayloadId::of(prefix);
        core.payloads.push(descriptor(prefix_id, prefix));
        derivations.derivations.push(wire::Derivation {
            target_payload_id: prefix_id.as_bytes().to_vec(),
            operations: vec![wire::DerivationOperation {
                operation: Some(wire::derivation_operation::Operation::Copy(
                    wire::DerivationCopy {
                        source_payload_id: id.as_bytes().to_vec(),
                        source_offset: 0,
                        destination_offset: 0,
                        contiguous_bytes: 7,
                        ..Default::default()
                    },
                )),
            }],
        });
    }
    let public_key = key.public_key().as_ref().try_into().unwrap();
    source.publishers[0].public_key = public_key;
    let mut builder = protect::EnvelopeBuilder::new(
        core.encode_to_vec(),
        public_key,
        envelope.core_id.as_slice().try_into().unwrap(),
    )
    .unwrap();
    for d in envelope.deployments {
        builder.deployment(d.index, d.body).unwrap();
    }
    for r in envelope.ranks {
        builder.rank(r.deployment, r.rank, r.body).unwrap();
    }
    let manifest = builder.sign(key).unwrap();
    let mut locations = meta.locations.clone();
    locations.locations.push(wire::PayloadLocation {
        payload_id: id.as_bytes().to_vec(),
        file: String::new(),
        tensor: "frontend.chat-template".into(),
    });
    let temp = source.directory.join("replacement");
    let mut writer = Writer::new(File::create(&temp).unwrap(), container::PAYLOAD_FORMAT).unwrap();
    for name in meta.primary().tensor_names() {
        if name.starts_with(container::ALIGNMENT_PREFIX)
            || name == reader::MANIFEST
            || name == reader::LOCATIONS
            || name == reader::DERIVATIONS
        {
            continue;
        }
        let slice = meta.primary().tensor(name).unwrap();
        writer.add(name, slice.len(), Source::Slice(slice)).unwrap();
    }
    writer
        .add(
            reader::MANIFEST,
            manifest.len() as u64,
            Source::Bytes(manifest),
        )
        .unwrap();
    let encoded = locations.encode_to_vec();
    writer
        .add(
            reader::LOCATIONS,
            encoded.len() as u64,
            Source::Bytes(encoded),
        )
        .unwrap();
    writer
        .add(
            "frontend.chat-template",
            TEMPLATE.len() as u64,
            Source::Bytes(TEMPLATE.as_bytes().to_vec()),
        )
        .unwrap();
    let encoded = derivations.encode_to_vec();
    writer
        .add(
            reader::DERIVATIONS,
            encoded.len() as u64,
            Source::Bytes(encoded),
        )
        .unwrap();
    writer.finish().unwrap();
    fs::rename(temp, source.directory.join("model.safetensors")).unwrap();
    source
}

fn key() -> signature::Ed25519KeyPair {
    let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&SystemRandom::new()).unwrap();
    signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap()
}

#[test]
fn frontend_edits_preserve_payloads_and_source_and_verify_signatures() {
    let scratch = Scratch::new("edit");
    let signing = key();
    let source = editable_fixture(&scratch, &signing, false);
    for (suffix, template) in [("alias", None), ("both", Some(MEDIUM.as_bytes()))] {
        let before = Metadata::open(&source.directory).unwrap();
        let original = fs::read(source.directory.join("model.safetensors")).unwrap();
        let output = scratch.join(suffix);
        edit(&EditRequest {
            source: &source.directory,
            output: &output,
            publishers: &source.publishers,
            key: &signing,
            served_model_name: Some("nvidia/model"),
            chat_template: template,
        })
        .unwrap();
        let after = Metadata::open(&output).unwrap();
        let publisher = Publisher {
            name: "Development".into(),
            namespace: "local-development".into(),
            public_key: signing.public_key().as_ref().try_into().unwrap(),
            development: true,
        };
        let verified = protect::verify(
            &after.envelope,
            &[publisher],
            Selection::Deployment {
                deployment: 0,
                rank: None,
            },
        )
        .unwrap();
        assert_eq!(
            verified
                .deployment()
                .unwrap()
                .contract
                .as_ref()
                .unwrap()
                .served_names,
            ["nvidia/model"]
        );
        assert_eq!(verified.core().model_identity, "fixture");
        assert_eq!(
            fs::read(source.directory.join("model.safetensors")).unwrap(),
            original
        );
        let old_envelope = protect::envelope(&before.envelope).unwrap();
        let new_envelope = protect::envelope(&after.envelope).unwrap();
        assert_ne!(old_envelope.core_id, new_envelope.core_id);
        assert_eq!(old_envelope.ranks[0].body, new_envelope.ranks[0].body);
        let checked = Verified::open(&output, None).unwrap();
        checked.payloads().unwrap();
        for relative in [
            "weights/layer-000.safetensors",
            "deployments/sm120-tp1/rank-0/graphs.safetensors",
        ] {
            assert_eq!(
                fs::read(source.directory.join(relative)).unwrap(),
                fs::read(output.join(relative)).unwrap()
            );
        }
        assert_eq!(
            tensor_bytes(after.primary(), "frontend.chat-template"),
            template.unwrap_or(TEMPLATE.as_bytes())
        );
        assert_eq!(
            tensor_bytes(before.primary(), "frontend.tokenizer"),
            tensor_bytes(after.primary(), "frontend.tokenizer")
        );
    }
}

#[test]
fn invalid_edits_and_untrusted_sources_leave_no_output() {
    let scratch = Scratch::new("edit-reject");
    let signing = key();
    let source = editable_fixture(&scratch, &signing, false);
    for (name, alias, template, trusted) in [
        ("empty", None, None, true),
        ("bad-name", Some(""), None, true),
        ("unmarked", None, Some("{{ messages }}".as_bytes()), true),
        ("syntax", None, Some("{% bad %}".as_bytes()), true),
        ("utf8", None, Some(b"\xff".as_slice()), true),
        ("untrusted", Some("name"), None, false),
    ] {
        let output = scratch.join(name);
        assert!(
            edit(&EditRequest {
                source: &source.directory,
                output: &output,
                publishers: if trusted { &source.publishers } else { &[] },
                key: &signing,
                served_model_name: alias,
                chat_template: template
            })
            .is_err()
        );
        assert!(!output.exists());
    }
}

#[test]
fn template_payload_remains_available_to_derivations() {
    let scratch = Scratch::new("edit-template-derivation");
    let signing = key();
    let source = editable_fixture(&scratch, &signing, true);
    let output = scratch.join("edited");
    edit(&EditRequest {
        source: &source.directory,
        output: &output,
        publishers: &source.publishers,
        key: &signing,
        served_model_name: None,
        chat_template: Some(MEDIUM.as_bytes()),
    })
    .unwrap();
    let verified = Verified::open(&output, None).unwrap();
    verified.payloads().unwrap();
    assert_eq!(
        tensor_bytes(verified.metadata.primary(), "frontend.chat-template"),
        TEMPLATE.as_bytes()
    );
    assert_eq!(
        tensor_bytes(verified.metadata.primary(), "frontend.chat-template-1"),
        MEDIUM.as_bytes()
    );
}

#[test]
fn packed_input_keeps_stored_executable_bytes() {
    let scratch = Scratch::new("edit-packed");
    let signing = key();
    let source = editable_fixture(&scratch, &signing, false);
    let packed = scratch.join("input.safetensors");
    pack(&source.directory, &packed).unwrap();
    let output = scratch.join("edited");
    edit(&EditRequest {
        source: &packed,
        output: &output,
        publishers: &source.publishers,
        key: &signing,
        served_model_name: Some("name"),
        chat_template: Some(MEDIUM.as_bytes()),
    })
    .unwrap();
    let verified = Verified::open(&output, None).unwrap();
    let payloads = verified.payloads().unwrap();
    for expected in [&source.payload, &source.stored] {
        let stored = payloads.get(PayloadId::of(expected).as_bytes()).unwrap();
        let mut actual = vec![0; expected.len()];
        stored.read_at(0, &mut actual).unwrap();
        assert_eq!(&actual, expected);
    }
}
