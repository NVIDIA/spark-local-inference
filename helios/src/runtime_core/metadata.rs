/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Metadata-only inspection of a core's primary file.

use super::{
    InspectionOptions, archive::Archive, format::invalid, protection::hex, protobuf::wire,
};
use crate::Result;
use prost::Message;
use serde_json::{Value, json};
use std::path::Path;

/// Read bounded core metadata without authenticating signatures, opening companion files,
/// reading payloads, or initializing CUDA. The returned object describes recorded preparation
/// limits and address layouts, not the physical memory selected when serving.
fn core_primary(path: &Path) -> Result<std::path::PathBuf> {
    if !path.is_dir() {
        return Ok(path.to_owned());
    }
    let preferred = path.join("model.safetensors");
    if preferred.is_file() {
        return Ok(preferred);
    }
    Err(invalid(
        "core directory has no model.safetensors primary",
    ))
}

pub fn unverified_metadata(
    path: &Path,
    options: InspectionOptions,
    rank: Option<usize>,
) -> Result<Value> {
    unverified_metadata_selected(path, options, rank, None)
}

/// Unverified inspection with an optional payload deployment selection.
pub fn unverified_metadata_selected(
    path: &Path,
    options: InspectionOptions,
    rank: Option<usize>,
    tensor_parallel: Option<u32>,
) -> Result<Value> {
    let primary = core_primary(path)?;
    if super::NativeResources::core_format(&primary)? == super::CoreFormat::Payload {
        return unverified_payload(&primary, options, rank, tensor_parallel);
    }
    if tensor_parallel.is_some() {
        return Err(invalid(
            "tensor-parallel selection applies only to payload cores",
        ));
    }
    let path = &primary;
    let (file, _, payload_bytes, bytes) = Archive::envelope(path)?;
    let envelope = wire::ProtectedIndex::decode(bytes.as_slice())
        .map_err(|e| invalid(format!("invalid signed envelope: {e}")))?;
    super::protection::ExecutableEncoding::from_version(envelope.version)?;
    if envelope.groups.len() != 3
        || envelope
            .groups
            .iter()
            .enumerate()
            .any(|(i, g)| g.kind != i as u32 + 1)
        || envelope.ranks.is_empty()
    {
        return Err(invalid("invalid protected core envelope"));
    }
    let metadata = wire::PublisherMetadata::decode(envelope.groups[0].body.as_slice())
        .map_err(|e| invalid(format!("invalid publisher metadata: {e}")))?;
    let mut index = metadata
        .index
        .ok_or_else(|| invalid("missing model index"))?;
    if !matches!(index.version, 4 | 5) {
        return Err(invalid("unsupported model index"));
    }
    if !index.ranks.is_empty() || metadata.ranks as usize != envelope.ranks.len() {
        return Err(invalid("inconsistent core rank count"));
    }
    let mut ranks = Vec::new();
    for (position, group) in envelope.ranks.iter().enumerate() {
        if group.kind != 4 {
            return Err(invalid("invalid rank signature group"));
        }
        let rank = wire::RankProtection::decode(group.body.as_slice())
            .map_err(|e| invalid(format!("invalid rank metadata: {e}")))?;
        if rank.rank as usize != position {
            return Err(invalid("unordered core ranks"));
        }
        let bundle = rank.bundle.ok_or_else(|| invalid("missing rank bundle"))?;
        ranks.push(bundle);
    }
    if let Some(rank) = rank {
        if index.cluster.is_none() || rank >= ranks.len() {
            return Err(invalid(
                "rank is not a separately served rank of this core",
            ));
        }
        ranks = vec![ranks.remove(rank)];
    }
    index.ranks = ranks;
    let mut value = super::describe::describe_index(&index, options, rank)?;
    value["core"] = json!({
        "signatures_verified": false,
        "format_version": index.version,
        "bytes": file.metadata().map_err(|e| e.to_string())?.len(),
        "primary_payload_bytes": payload_bytes,
        "companions": null,
        "cluster": index.cluster,
        "publisher": {"name": null, "namespace": metadata.namespace,
            "development": metadata.development, "fingerprint": hex(&envelope.signer)},
        "official_source": metadata.official_source,
        "license": metadata.license,
    });
    Ok(value)
}

/// Bounded, unauthenticated metadata for a payload-format primary. Rank bodies are always decoded
/// so the default summary reports the real rank set and a requested rank is validated; detail
/// flags only gate additional arrays. No companions, payloads, signatures, or CUDA are touched.
fn unverified_payload(
    path: &std::path::Path,
    options: InspectionOptions,
    rank: Option<usize>,
    tensor_parallel: Option<u32>,
) -> Result<Value> {
    use prost::Message as _;
    let metadata = crate::core::reader::Metadata::open(path)?;
    let envelope = crate::core::protection::envelope(&metadata.envelope)?;
    let core =
        crate::core::schema::wire::CoreBody::decode(envelope.core_body.as_slice())
            .map_err(|e| invalid(format!("invalid core body: {e}")))?;
    let mut deployments = Vec::new();
    for signed in &envelope.deployments {
        deployments.push(
            crate::core::schema::wire::DeploymentBody::decode(signed.body.as_slice())
                .map_err(|e| invalid(format!("invalid deployment body: {e}")))?,
        );
    }
    let mut rank_bodies = Vec::new();
    for signed in &envelope.ranks {
        rank_bodies.push(
            crate::core::schema::wire::RankBody::decode(signed.body.as_slice())
                .map_err(|e| invalid(format!("invalid rank body: {e}")))?,
        );
    }
    let all_rank_count = rank_bodies.len();
    if let Some(tensor_parallel) = tensor_parallel {
        let matching: Vec<usize> = deployments
            .iter()
            .enumerate()
            .filter(|(_, body)| body.tensor_parallel == tensor_parallel)
            .map(|(index, _)| index)
            .collect();
        let available: Vec<u32> = deployments
            .iter()
            .map(|body| body.tensor_parallel)
            .collect();
        match matching.as_slice() {
            [index] => {
                let selected_id = deployments[*index].deployment_id.clone();
                let selected = deployments.remove(*index);
                deployments = vec![selected];
                rank_bodies.retain(|body| body.deployment_id == selected_id);
            }
            [] => {
                return Err(invalid(format!(
                    "core has no deployment with tensor-parallel size {tensor_parallel}; available: {available:?}"
                )));
            }
            _ => {
                return Err(invalid(format!(
                    "core has several deployments with tensor-parallel size {tensor_parallel}; inspect the catalog"
                )));
            }
        }
    }

    let selected_ranks: Vec<usize> = match rank {
        None => (0..rank_bodies.len()).collect(),
        Some(selected) => {
            let matches: Vec<usize> = rank_bodies
                .iter()
                .enumerate()
                .filter(|(_, body)| usize::try_from(body.rank).is_ok_and(|body| body == selected))
                .map(|(index, _)| index)
                .collect();
            match matches.len() {
                0 => return Err(invalid("requested rank is not part of this core")),
                1 => matches,
                _ => {
                    let ambiguous: Vec<&str> = matches
                        .iter()
                        .map(|&index| rank_bodies[index].deployment_id.as_str())
                        .collect();
                    return Err(invalid(format!(
                        "rank {selected} is ambiguous across deployments {ambiguous:?}; omit --rank to inspect the catalog"
                    )));
                }
            }
        }
    };

    let mut deployment_values = Vec::with_capacity(deployments.len());
    let mut vocabularies = Vec::new();
    for body in &deployments {
        let contract = body.contract.as_ref();
        let vocabulary = contract
            .and_then(|contract| contract.model_contract.as_ref())
            .map(|model| Value::from(model.vocabulary))
            .unwrap_or(Value::Null);
        vocabularies.push(vocabulary.clone());
        let max_active = contract
            .and_then(|contract| contract.max_active)
            .or_else(|| {
                rank_bodies
                    .iter()
                    .find(|rank| rank.deployment_id == body.deployment_id)
                    .and_then(|rank| rank.execution.as_ref())
                    .map(|execution| execution.max_active)
            });
        let model_contract = contract.and_then(|contract| contract.model_contract.as_ref());
        let frontend_default = crate::runtime_core::wire::FrontendSettings::default();
        let frontend = core.frontend.as_ref().unwrap_or(&frontend_default);
        let frontend_detail = if options.frontend || options.all {
            super::describe::frontend_json(
                frontend,
                model_contract.and_then(|model| model.frontend.as_ref()),
            )
        } else {
            Value::Null
        };
        let vision_detail = if options.frontend || options.all {
            super::describe::vision_json(
                model_contract.and_then(|model| model.vision.as_ref()),
                frontend,
                contract.map(|contract| contract.max_sequence_tokens),
            )
        } else {
            Value::Null
        };
        deployment_values.push(json!({
            "id": body.deployment_id,
            "frontend": frontend_detail,
            "vision": vision_detail,
            "tensor_parallel": body.tensor_parallel,
            "architecture": body.architecture,
            "pages": contract.map(|c| c.pages),
            "state_rows": contract.map(|c| c.state_rows),
            "max_sequence_tokens": contract.map(|c| c.max_sequence_tokens),
            "prefill_chunk_size": contract.map(|c| c.prefill_chunk_size),
            "max_active": max_active.map(Value::from).unwrap_or(Value::Null),
            "vocabulary": vocabulary,
            "ranks": body.ranks.iter().map(|rank| rank.rank).collect::<Vec<_>>(),
        }));
    }
    let common_vocabulary = match vocabularies.split_first() {
        None => Value::Null,
        Some((first, rest)) if !first.is_null() && rest.iter().all(|value| value == first) => {
            first.clone()
        }
        Some(_) => Value::Null,
    };

    // The selected deployment comes from the decoded rank body, never from commitment lists that
    // an unverified envelope could leave inconsistent.
    let selected_deployment = match rank {
        Some(selected) => rank_bodies
            .iter()
            .find(|body| usize::try_from(body.rank).is_ok_and(|rank| rank == selected))
            .and_then(|body| {
                deployment_values
                    .iter()
                    .find(|deployment| deployment["id"] == body.deployment_id)
            }),
        None if deployment_values.len() == 1 => deployment_values.first(),
        None => None,
    };
    let capacity = |name: &str| {
        selected_deployment
            .and_then(|deployment| deployment.get(name).cloned())
            .unwrap_or(Value::Null)
    };

    let mut ranks = Vec::with_capacity(selected_ranks.len());
    for index in selected_ranks {
        let body = &rank_bodies[index];
        let mut entry = json!({ "rank": body.rank, "deployment": body.deployment_id });
        if options.memory_map || options.all {
            entry["reservations"] = json!(
                body.reservations
                    .iter()
                    .map(|reservation| json!({
                        "address": reservation.address, "nbytes": reservation.nbytes,
                    }))
                    .collect::<Vec<_>>()
            );
            entry["initializers"] = json!(body.initializers.iter().map(|initializer| json!({
                "allocation": initializer.allocation,
                "destination_offset": initializer.destination_offset,
                "payload": initializer.source.as_ref().and_then(|use_| use_.slice.as_ref()).map(|slice| json!({
                    "id": slice.payload_id.iter().map(|byte| format!("{byte:02x}")).collect::<String>(),
                    "offset": slice.offset, "nbytes": slice.nbytes,
                })),
            })).collect::<Vec<_>>());
            entry["execution_buffers"] = json!(body.execution.as_ref().map(|execution| execution.buffers.iter().map(|buffer| json!({
                "name": buffer.name, "nbytes": buffer.nbytes, "dtype": buffer.dtype,
                "shape": buffer.shape, "strides": buffer.strides,
                "allocation": buffer.reference.as_ref().map(|reference| reference.allocation.clone()),
                "offset": buffer.reference.as_ref().map(|reference| reference.offset),
            })).collect::<Vec<_>>()).unwrap_or_default());
            entry["allocations"] = json!(
                body.allocations
                    .iter()
                    .map(|allocation| json!({
                        "name": allocation.name, "nbytes": allocation.nbytes,
                        "address": allocation.address, "alignment": allocation.alignment,
                        "memory": allocation.memory, "sizing": allocation.sizing,
                    }))
                    .collect::<Vec<_>>()
            );
        }
        if options.graphs || options.all {
            entry["execution_graphs"] = json!(
                body.execution
                    .as_ref()
                    .map(|execution| {
                        execution.graphs.iter().map(|graph| json!({
                "name": graph.name, "kind": graph.kind, "sampling_mode": graph.sampling_mode,
                "batch_capacity": graph.batch_capacity, "token_capacity": graph.token_capacity,
                "packed_token_capacity": graph.packed_token_capacity,
                "prefill_capacity": graph.prefill_capacity,
                "draft_tokens": graph.draft_tokens, "verify_tokens": graph.verify_tokens,
                "estimated_duration_ns": graph.estimated_duration_ns,
            })).collect::<Vec<_>>()
                    })
                    .unwrap_or_default()
            );
            entry["graphs"] = json!(
                body.graphs
                    .iter()
                    .map(|graph| json!({
                        "name": graph.name, "node_count": graph.node_count,
                        "edge_count": graph.edge_count, "resource": graph.resource,
                    }))
                    .collect::<Vec<_>>()
            );
        }
        ranks.push(entry);
    }
    let assets = if options.frontend || options.all {
        core.assets.iter().map(|asset| json!({
            "name": asset.name,
            "bytes": asset.payload.as_ref().and_then(|use_| use_.slice.as_ref()).map(|slice| slice.nbytes).unwrap_or(0),
            "sha256": asset.sha256.iter().map(|byte| format!("{byte:02x}")).collect::<String>(),
        })).collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    Ok(json!({
        "core": {
            "signatures_verified": false,
            "metadata": "unverified",
            "format": "payload",
            "format_version": core.format_version,
            "path": path.display().to_string(),
            "frontend_settings": if options.frontend || options.all {
                core.frontend.as_ref().and_then(|frontend| serde_json::to_value(frontend).ok()).unwrap_or(Value::Null)
            } else {
                Value::Null
            },
            "publisher": core.publisher.as_ref().map(|publisher| json!({
                "namespace": publisher.namespace,
                "official_source": publisher.official_source,
                "development": publisher.development,
            })),
        },
        "model": {
            "model_id": core.model_identity,
            "identity": core.model_identity,
            "hardware": core.hardware_identity,
            "max_sequence_tokens": capacity("max_sequence_tokens"),
            "prefill_chunk_size": capacity("prefill_chunk_size"),
            "max_active": capacity("max_active"),
            "vocabulary": common_vocabulary,
        },
        "deployments": deployment_values,
        "assets": assets,
        "payloads": core.payloads.len(),
        "ranks": ranks,
        "rank_count": if tensor_parallel.is_some() { rank_bodies.len() } else { all_rank_count },
        "selected_tensor_parallel": tensor_parallel,
        "selected_rank": rank,
    }))
}

#[cfg(test)]
mod tests {
    use std::fs;

    #[test]
    fn core_primary_requires_canonical_directory_primary() {
        let root = std::env::temp_dir().join(format!("flare-unverified-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        assert!(super::core_primary(&root).is_err(), "no primary");
        fs::write(root.join("model.safetensors"), b"x").unwrap();
        assert_eq!(
            super::core_primary(&root).unwrap(),
            root.join("model.safetensors")
        );
        let file = root.join("packed.safetensors");
        fs::write(&file, b"x").unwrap();
        assert_eq!(super::core_primary(&file).unwrap(), file);
        let _ = fs::remove_dir_all(root);
    }
}
