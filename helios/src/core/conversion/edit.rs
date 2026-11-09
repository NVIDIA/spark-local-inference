/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Frontend edits that preserve stored model and executable payloads.

use super::*;
use ring::{rand::SecureRandom, signature::KeyPair};

pub struct EditRequest<'a> {
    pub source: &'a Path,
    pub output: &'a Path,
    pub publishers: &'a [crate::runtime_core::Publisher],
    pub key: &'a ring::signature::Ed25519KeyPair,
    pub served_model_name: Option<&'a str>,
    pub chat_template: Option<&'a [u8]>,
}

/// Writes a development-signed directory without recompiling or recompressing payloads.
pub fn edit(request: &EditRequest<'_>) -> Result<()> {
    if request.served_model_name.is_none() && request.chat_template.is_none() {
        return Err(invalid(
            "edit requires a served model name or chat template",
        ));
    }
    if request.output.exists() {
        return Err(invalid(format!(
            "destination exists: {}",
            request.output.display()
        )));
    }
    if let Some(name) = request.served_model_name {
        if name.trim().is_empty() || name.len() > 256 || name.chars().any(char::is_control) {
            return Err(invalid(
                "served model name must contain 1..256 bytes without control characters",
            ));
        }
    }
    let verified = Verified::open(request.source, None)?;
    let metadata = &verified.metadata;
    let envelope = protection::envelope(&metadata.envelope)?;
    let mut core = wire::CoreBody::decode(envelope.core_body.as_slice())
        .map_err(|e| invalid(format!("invalid core body: {e}")))?;
    let mut deployments = Vec::new();
    for signed in &envelope.deployments {
        let authenticated = protection::verify(
            &metadata.envelope,
            request.publishers,
            protection::Selection::Deployment {
                deployment: signed.index as usize,
                rank: None,
            },
        )?;
        let mut body = authenticated
            .deployment()
            .cloned()
            .ok_or_else(|| invalid("core has no prepared deployment"))?;
        if let Some(name) = request.served_model_name {
            body.contract
                .as_mut()
                .ok_or_else(|| invalid("deployment has no serving contract"))?
                .served_names = vec![name.to_owned()];
        }
        deployments.push((signed.index, body));
    }
    let _payloads = verified.payload_files()?;
    let mut locations = metadata.locations.clone();
    let mut removed_tensor = None;
    let mut replacement = None;
    if let Some(bytes) = request.chat_template {
        if core.frontend.as_ref().map_or(0, |f| f.prompt_encoder)
            != runtime::PromptEncoder::PromptJinja as i32
        {
            return Err(invalid(
                "chat template edits require a Jinja prompt encoder",
            ));
        }
        let source =
            std::str::from_utf8(bytes).map_err(|_| invalid("chat template must be UTF-8"))?;
        crate::frontend::validate_chat_template(source)?;
        let asset = core
            .assets
            .iter_mut()
            .find(|asset| asset.name == "chat_template")
            .ok_or_else(|| invalid("core has no chat template asset"))?;
        let use_ = asset
            .payload
            .as_mut()
            .ok_or_else(|| invalid("chat template has no payload"))?;
        let slice = use_
            .slice
            .as_mut()
            .ok_or_else(|| invalid("chat template has no payload slice"))?;
        let previous = slice.payload_id.clone();
        let id = protection::sha256(bytes).to_vec();
        if id != asset.sha256 {
            slice.payload_id = id.clone();
            slice.offset = 0;
            slice.nbytes = bytes.len() as u64;
            asset.sha256 = id.clone();
            let still_used = core.assets.iter().any(|a| {
                a.payload
                    .as_ref()
                    .and_then(|p| p.slice.as_ref())
                    .is_some_and(|s| s.payload_id == previous)
            }) || deployments
                .iter()
                .any(|(_, d)| d.payloads.iter().any(|p| p.id == previous))
                || verified
                    .ranks
                    .iter()
                    .any(|r| r.payloads.iter().any(|p| p.id == previous));
            let still_used = still_used || metadata.derivations.as_ref().is_some_and(|set| {
                set.derivations.iter().any(|d| d.target_payload_id == previous || d.operations.iter().any(|op| {
                    matches!(&op.operation, Some(wire::derivation_operation::Operation::Copy(c)) if c.source_payload_id == previous)
                }))
            });
            if !still_used {
                core.payloads.retain(|p| p.id != previous);
                if let Some(old) = locations
                    .locations
                    .iter()
                    .find(|l| l.payload_id == previous)
                {
                    if old.file.is_empty()
                        && !locations.locations.iter().any(|l| {
                            l.payload_id != previous && l.file.is_empty() && l.tensor == old.tensor
                        })
                    {
                        removed_tensor = Some(old.tensor.clone());
                    }
                }
                locations.locations.retain(|l| l.payload_id != previous);
            }
            if !core.payloads.iter().any(|p| p.id == id) {
                core.payloads.push(wire::PayloadDescriptor {
                    id: id.clone(),
                    nbytes: bytes.len() as u64,
                });
            }
            if !locations.locations.iter().any(|l| l.payload_id == id) {
                let mut tensor = "frontend.chat-template".to_owned();
                let mut suffix = 0;
                while metadata.primary().tensor_names().any(|n| n == tensor)
                    && removed_tensor.as_deref() != Some(&tensor)
                {
                    suffix += 1;
                    tensor = format!("frontend.chat-template-{suffix}");
                }
                locations.locations.push(wire::PayloadLocation {
                    payload_id: id,
                    file: String::new(),
                    tensor: tensor.clone(),
                });
                replacement = Some((tensor, bytes.to_vec()));
            }
        }
    }
    let mut core_id = [0; 32];
    ring::rand::SystemRandom::new()
        .fill(&mut core_id)
        .map_err(|_| invalid("generate core identity"))?;
    let public_key: [u8; 32] = request
        .key
        .public_key()
        .as_ref()
        .try_into()
        .map_err(|_| invalid("invalid signing key"))?;
    core.core_id = core_id.to_vec();
    let publisher = core
        .publisher
        .as_mut()
        .ok_or_else(|| invalid("core has no publisher"))?;
    publisher.namespace = "local-development".into();
    publisher.official_source = "local-development".into();
    publisher.development = true;
    let bodies: Vec<_> = deployments
        .iter()
        .map(|(index, body)| (*index, body.encode_to_vec()))
        .collect();
    for (index, bytes) in &bodies {
        core.deployments[*index as usize].body_digest = protection::sha256(bytes).to_vec();
    }
    let mut builder =
        protection::EnvelopeBuilder::new(core.encode_to_vec(), public_key, core_id)?;
    for (index, body) in bodies {
        builder.deployment(index, body)?;
    }
    for rank in &envelope.ranks {
        builder.rank(rank.deployment, rank.rank, rank.body.clone())?;
    }
    let manifest = builder.sign(request.key)?;
    schema::validate_locations(&locations)?;
    let staging = unique_staging(request.output, "edit")?;
    fs::create_dir(&staging).map_err(|e| invalid(format!("create edit staging: {e}")))?;
    let _guard = TempDirectory(staging.clone());
    let root = metadata.path.parent().unwrap_or(Path::new("."));
    let files: BTreeSet<_> = locations
        .locations
        .iter()
        .filter(|l| !l.file.is_empty())
        .map(|l| &l.file)
        .collect();
    for relative in files {
        let destination = staging.join(relative);
        fs::create_dir_all(destination.parent().unwrap())
            .map_err(|e| invalid(format!("create payload directory: {e}")))?;
        if fs::hard_link(root.join(relative), &destination).is_err() {
            crate::core::cache::copy_file(&root.join(relative), &destination)
                .map_err(|e| invalid(format!("copy payload companion: {e}")))?;
        }
    }
    let primary = staging.join("model.safetensors");
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&primary)
        .map_err(|e| invalid(format!("create edited primary: {e}")))?;
    let mut writer = Writer::new(file, metadata.primary().format())?;
    for name in metadata.primary().tensor_names() {
        if name.starts_with(container::ALIGNMENT_PREFIX)
            || name == reader::MANIFEST
            || name == reader::LOCATIONS
            || removed_tensor.as_deref() == Some(name)
        {
            continue;
        }
        let slice = metadata.primary().tensor(name)?;
        writer.add(name, slice.len(), Source::Slice(slice))?;
    }
    if let Some((name, bytes)) = replacement {
        writer.add(&name, bytes.len() as u64, Source::Bytes(bytes))?;
    }
    writer.add(
        reader::MANIFEST,
        manifest.len() as u64,
        Source::Bytes(manifest),
    )?;
    let encoded_locations = locations.encode_to_vec();
    writer.add(
        reader::LOCATIONS,
        encoded_locations.len() as u64,
        Source::Bytes(encoded_locations),
    )?;
    writer.finish()?;
    // Verification pins directory handles; release them before Windows publishes the tree.
    drop(Verified::open(&staging, None)?);
    crate::core::cache::synchronize(&staging)?;
    crate::core::cache::rename_noreplace(&staging, request.output)
        .map_err(|e| invalid(format!("publish edited core: {e}")))?;
    sync_parent(request.output)
}
