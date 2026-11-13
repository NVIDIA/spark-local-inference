/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Reuse signed weight payloads with compatible declared preparation layouts.

use std::collections::{BTreeMap, BTreeSet};

use helios_core::core::{cache, protection, reader, schema::wire as pwire};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::{Deserialize, Serialize};

use super::{
    CaptureRequest, RankOutput, Result, WeightPayloadGroup, WeightPayloadSourceSpan,
    WeightPayloadSources, invalid, weight_payload,
};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct Source {
    #[serde(with = "envelope_base64")]
    envelope: Vec<u8>,
    publishers: Vec<helios_core::runtime_core::Publisher>,
    #[serde(default)]
    rebuild_manifest: Option<serde_json::Value>,
}

mod envelope_base64 {
    use base64::{Engine, display::Base64Display, engine::general_purpose::STANDARD};
    use serde::{Deserializer, Serializer, de};

    pub(super) fn serialize<S>(envelope: &[u8], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.collect_str(&Base64Display::new(envelope, &STANDARD))
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct EnvelopeVisitor;

        impl de::Visitor<'_> for EnvelopeVisitor {
            type Value = Vec<u8>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a base64-encoded signed weight envelope")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                STANDARD.decode(value).map_err(E::custom)
            }
        }

        deserializer.deserialize_str(EnvelopeVisitor)
    }
}

#[pyclass(module = "flare_heliopause")]
pub(crate) struct NativeWeightSource {
    pub(crate) source: Source,
    // The entry's hardlinks retain all payloads without one open file per weight.
    _entry: cache::Entry,
}

#[pymethods]
impl NativeWeightSource {
    fn metadata(&self) -> PyResult<String> {
        serde_json::to_string(&self.source)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

#[pyfunction]
fn export_reuse_weights(
    py: Python<'_>,
    cache: String,
    model: String,
    profiles: String,
) -> PyResult<NativeWeightSource> {
    py.detach(|| open_source(std::path::Path::new(&cache), &model, &profiles))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

pub(crate) fn open_source(
    cache: &std::path::Path,
    model: &str,
    profiles: &str,
) -> Result<NativeWeightSource> {
    let root = cache::Root::open(cache)?;
    let entry = root.resolve(&cache::model_key(&model)?)?.ok_or_else(|| {
        invalid(
            "--reuse-weights requires a cached default model version; export with weights first",
        )
    })?;
    let metadata = reader::Metadata::open(&root.primary(&entry)?)?;
    let publishers = super::conversion::payload_publishers(&profiles)?;
    let catalog = protection::verify(
        &metadata.envelope,
        &publishers,
        protection::Selection::Catalog,
    )?;
    if catalog.core().model_identity != model {
        return Err(invalid("cached weight source belongs to a different model"));
    }
    let mut payloads = BTreeSet::new();
    for deployment in 0..catalog.deployment_count() {
        let verified = protection::verify(
            &metadata.envelope,
            &publishers,
            protection::Selection::Deployment {
                deployment,
                rank: None,
            },
        )?;
        for rank in verified.ranks() {
            for initializer in &rank.initializers {
                let Some(source) = initializer
                    .source
                    .as_ref()
                    .filter(|source| is_weight(source.role))
                else {
                    continue;
                };
                let slice = source
                    .slice
                    .as_ref()
                    .ok_or_else(|| invalid("weight initializer has no payload slice"))?;
                if payloads.contains(&slice.payload_id) {
                    continue;
                }
                let id = helios_core::core::PayloadId::from_bytes(
                    slice.payload_id.as_slice().try_into()?,
                );
                let object = root.payload(id)?.ok_or_else(|| {
                    invalid(format!("--reuse-weights is missing cached payload {id}"))
                })?;
                if object.length() != verified.catalog().length(&slice.payload_id)? {
                    return Err(invalid(format!(
                        "cached weight payload {id} has a different length"
                    )));
                }
                payloads.insert(slice.payload_id.clone());
            }
        }
        if let Some(engram) = verified.deployment().and_then(|body| body.engram.as_ref()) {
            for source in engram_uses(engram) {
                let slice = source
                    .slice
                    .as_ref()
                    .ok_or_else(|| invalid("cached Engram plane has no slice"))?;
                let id = helios_core::core::PayloadId::from_bytes(
                    slice.payload_id.as_slice().try_into()?,
                );
                let object = root.payload(id)?.ok_or_else(|| {
                    invalid(format!(
                        "--reuse-weights is missing cached Engram payload {id}"
                    ))
                })?;
                if object.length() != verified.catalog().length(&slice.payload_id)? {
                    return Err(invalid("cached Engram payload has a different length"));
                }
                payloads.insert(slice.payload_id.clone());
            }
        }
    }
    if payloads.is_empty() {
        return Err(invalid(
            "--reuse-weights requires a cached model version containing weights",
        ));
    }
    Ok(NativeWeightSource {
        source: Source {
            envelope: metadata.envelope,
            publishers,
            rebuild_manifest: None,
        },
        _entry: entry,
    })
}

fn is_weight(role: i32) -> bool {
    role == pwire::PayloadRole::Weights as i32 || role == pwire::PayloadRole::TunedWeights as i32
}

pub(crate) fn numeric_ranges(
    request: &CaptureRequest,
    allocations: &[super::Allocation],
) -> Result<Vec<(u64, u64)>> {
    if request.options.reuse_weights.is_none() {
        return Ok(Vec::new());
    }
    let declaration: weight_payload::WeightPayloadDeclaration = serde_json::from_value(
        request
            .weight_payload
            .clone()
            .ok_or_else(|| invalid("--reuse-weights requires semantic weight declarations"))?,
    )?;
    let owners = super::capture::weight_allocations(allocations);
    let addresses: BTreeMap<_, _> = owners
        .iter()
        .map(|owner| (owner.name.as_str(), owner.address))
        .collect();
    let catalog = weight_payload::extract(
        &declaration.into_request(owners.clone(), request.zero_ranges.clone()),
    )?;
    let mut ranges = Vec::new();
    for group in &catalog.groups {
        for span in &group.spans {
            let start = addresses[span.allocation.as_str()] + span.destination_offset;
            let end = start + span.nbytes;
            for &(numeric_start, numeric_end) in &request.numeric {
                let begin = start.max(numeric_start);
                let stop = end.min(numeric_end);
                if begin < stop {
                    ranges.push((begin, stop));
                }
            }
        }
    }
    Ok(super::reachability::merge_ranges(ranges))
}

fn same_draft_body_quantization(
    previous: &super::wire::RebuildManifest,
    expected: &super::wire::RebuildManifest,
) -> bool {
    previous
        .speculation
        .as_ref()
        .and_then(|s| s.draft_body_quantization.as_deref())
        == expected
            .speculation
            .as_ref()
            .and_then(|s| s.draft_body_quantization.as_deref())
}

impl Source {
    pub(crate) fn engram(
        &self,
        request: &CaptureRequest,
        output: &RankOutput,
        data: &serde_json::Value,
    ) -> Result<pwire::Engram> {
        let verified = self.deployment(request, output)?;
        let previous = verified
            .deployment()
            .and_then(|body| body.engram.as_ref())
            .ok_or_else(|| invalid("--reuse-weights requires cached Engram embedding tables"))?;
        let mut metadata = data.clone();
        for table in metadata["tables"]
            .as_array_mut()
            .ok_or_else(|| invalid("Engram tables must be an array"))?
        {
            plane_lengths(table)?;
            if let Some(shards) = table
                .get_mut("shards")
                .and_then(|value| value.as_array_mut())
            {
                for shard in shards {
                    plane_lengths(shard)?;
                }
            }
        }
        let current: super::wire::Engram = super::binding::message("Engram", &metadata)?;
        rebind_engram(previous, &current)
    }

    pub(crate) fn groups(
        &self,
        request: &CaptureRequest,
        output: &RankOutput,
        catalog: &weight_payload::WeightPayloadCatalog,
        views: &WeightPayloadSources,
    ) -> Result<Vec<WeightPayloadGroup>> {
        let verified = self.deployment(request, output)?;
        map_groups(
            verified
                .rank(request.rank as u32)
                .ok_or_else(|| invalid("cached deployment lacks the preparation rank"))?,
            verified.catalog(),
            output,
            catalog,
            views,
        )
    }

    fn deployment(
        &self,
        request: &CaptureRequest,
        output: &RankOutput,
    ) -> Result<protection::VerifiedCore> {
        let expected: super::wire::RebuildManifest = super::binding::message(
            "RebuildManifest",
            self.rebuild_manifest
                .as_ref()
                .ok_or_else(|| invalid("--reuse-weights requires preparation input metadata"))?,
        )?;
        let shared = protection::verify(
            &self.envelope,
            &self.publishers,
            protection::Selection::Catalog,
        )?;
        for deployment in 0..shared.deployment_count() {
            let body = shared.deployment_body(deployment)?;
            if body.tensor_parallel as usize != request.world_size
                || body.architecture != output.index.ranks[0].architecture
            {
                continue;
            }
            let previous = body
                .rebuild
                .as_ref()
                .ok_or_else(|| invalid("cached deployment has no preparation input metadata"))?;
            if previous.recipe != expected.recipe
                || previous.checkpoint_repository != expected.checkpoint_repository
                || previous.checkpoint_revision != expected.checkpoint_revision
                || previous.local_checkpoint != expected.local_checkpoint
                || previous.tensor_parallel != expected.tensor_parallel
                || previous.online_mxfp8 != expected.online_mxfp8
                || (if previous.online_mxfp8_scope.is_empty() { "all" } else { &previous.online_mxfp8_scope })
                    != (if expected.online_mxfp8_scope.is_empty() { "all" } else { &expected.online_mxfp8_scope })
                || !same_draft_body_quantization(previous, &expected)
            {
                return Err(invalid(
                    "--reuse-weights requires the same recipe, checkpoint revision, tensor-parallel size, and draft conversion settings",
                ));
            }
            return Ok(protection::verify(
                &self.envelope,
                &self.publishers,
                protection::Selection::Deployment {
                    deployment,
                    rank: Some(request.rank as u32),
                },
            )?);
        }
        Err(invalid(
            "--reuse-weights has no cached deployment with the requested architecture and tensor-parallel size",
        ))
    }
}

pub(crate) fn engram_uses(engram: &pwire::Engram) -> Vec<&pwire::PayloadUse> {
    engram
        .tables
        .iter()
        .flat_map(|table| {
            table.weights.iter().chain(table.scales.iter()).chain(
                table
                    .shards
                    .iter()
                    .flat_map(|shard| shard.weights.iter().chain(shard.scales.iter())),
            )
        })
        .collect()
}

fn plane_lengths(value: &mut serde_json::Value) -> Result<()> {
    for plane in ["weights", "scales"] {
        if let Some(parts) = value.get_mut(plane).filter(|parts| !parts.is_null()) {
            let sources: Vec<super::FileSource> = serde_json::from_value(parts.clone())?;
            let nbytes = sources.iter().try_fold(0u64, |sum, source| {
                sum.checked_add(source.nbytes)
                    .ok_or_else(|| invalid("Engram plane length overflow"))
            })?;
            *parts = serde_json::json!({"offset":0,"nbytes":nbytes});
        }
    }
    Ok(())
}

fn rebind_engram(previous: &pwire::Engram, current: &super::wire::Engram) -> Result<pwire::Engram> {
    if previous.token_map != current.token_map
        || previous.pad_token_id != current.pad_token_id
        || previous.compressed_vocabulary != current.compressed_vocabulary
        || previous.tables.len() != current.tables.len()
    {
        return Err(invalid(
            "Engram vocabulary or table catalog differs from the cached weights",
        ));
    }
    let plane = |old: &Option<pwire::PayloadUse>,
                 new: &Option<super::wire::Span>|
     -> Result<Option<pwire::PayloadUse>> {
        match (old, new) {
            (None, None) => Ok(None),
            (Some(old), Some(new))
                if old
                    .slice
                    .as_ref()
                    .is_some_and(|slice| slice.nbytes == new.nbytes) =>
            {
                Ok(Some(old.clone()))
            }
            _ => Err(invalid(
                "Engram weight or scale plane differs from the cached layout",
            )),
        }
    };
    let mut result = previous.clone();
    for (old, new) in result.tables.iter_mut().zip(&current.tables) {
        let hash_matches = match (&old.hash, &new.hash) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                a.method == b.method
                    && a.orders == b.orders
                    && a.multipliers == b.multipliers
                    && a.primes == b.primes
                    && a.offsets == b.offsets
            }
            _ => false,
        };
        if old.layer != new.layer
            || old.rows != new.rows
            || old.embedding_width != new.embedding_width
            || old.scale_block != new.scale_block
            || old.encoding != new.encoding
            || old.global_scale.to_bits() != new.global_scale.to_bits()
            || !hash_matches
            || old.shards.len() != new.shards.len()
        {
            return Err(invalid(
                "Engram embedding table geometry, encoding, or hashing differs from the cached weights",
            ));
        }
        old.weights = plane(&old.weights, &new.weights)?;
        old.scales = plane(&old.scales, &new.scales)?;
        for (old, new) in old.shards.iter_mut().zip(&new.shards) {
            if old.rank != new.rank || old.first_row != new.first_row || old.rows != new.rows {
                return Err(invalid(
                    "Engram embedding shard partition differs from the cached weights",
                ));
            }
            old.weights = plane(&old.weights, &new.weights)?;
            old.scales = plane(&old.scales, &new.scales)?;
        }
        old.weight_buffer = new.weight_buffer.clone();
        old.scale_buffer = new.scale_buffer.clone();
        old.hash_buffer = new.hash_buffer.clone();
        old.ready_event = new.ready_event.clone();
        old.request_buffer = new.request_buffer.clone();
    }
    Ok(result)
}

fn map_groups(
    previous: &pwire::RankBody,
    payloads: &helios_core::core::schema::Catalog,
    output: &RankOutput,
    catalog: &weight_payload::WeightPayloadCatalog,
    views: &WeightPayloadSources,
) -> Result<Vec<WeightPayloadGroup>> {
    let bundle = &output.index.ranks[0];
    let rebuild = bundle
        .rebuild_weights
        .as_ref()
        .ok_or_else(|| invalid("--reuse-weights requires declared weight views"))?;
    let current = super::core::convert_prepared(rebuild, bundle, Some(views))?;
    let prior = helios_core::core::prepared_weights::catalog(previous)?.ok_or_else(|| {
        invalid("cached rank has no prepared weight catalog; export with weights first")
    })?;
    let old_views: BTreeMap<_, _> = prior
        .views
        .iter()
        .map(|view| (view.name.as_str(), view))
        .collect();
    // Every view in an allocation must preserve the same relative placement.
    // Allocation names and absolute addresses may change between preparations.
    let mut placements = BTreeMap::<String, (String, u64, u64)>::new();
    for view in &current.views {
        let old = old_views.get(view.name.as_str()).ok_or_else(|| {
            invalid(format!(
                "cached weights lack view {}; export with weights",
                view.name
            ))
        })?;
        let mut normalized = (*old).clone();
        normalized.allocation = view.allocation;
        normalized.offset = view.offset;
        if normalized != *view {
            return Err(invalid(format!(
                "weight view {} has a different layout or checkpoint mapping; export without --reuse-weights",
                view.name
            )));
        }
        let old_allocation = previous
            .allocations
            .get(old.allocation as usize)
            .ok_or_else(|| invalid("cached weight allocation is absent"))?;
        let allocation = &bundle.allocations[view.allocation as usize];
        let placement = (allocation.name.clone(), old.offset, view.offset);
        if let Some((name, old_origin, origin)) = placements.get(&old_allocation.name) {
            if name != &placement.0
                || i128::from(view.offset) - i128::from(old.offset)
                    != i128::from(*origin) - i128::from(*old_origin)
            {
                return Err(invalid(format!(
                    "weight allocation {} has a different relative layout; export without --reuse-weights",
                    old_allocation.name
                )));
            }
        } else {
            placements.insert(old_allocation.name.clone(), placement);
        }
    }
    // Publication coalesces adjacent initializers across owner spans. Coverage
    // merges those spans without changing their zero-elision boundaries.
    let coverage: Vec<_> = catalog
        .groups
        .iter()
        .map(|group| {
            let mut allocations = BTreeMap::<&str, Vec<(u64, u64)>>::new();
            for span in &group.spans {
                allocations.entry(&span.allocation).or_default().push((
                    span.destination_offset,
                    span.destination_offset + span.nbytes,
                ));
            }
            for ranges in allocations.values_mut() {
                *ranges = super::reachability::merge_ranges(std::mem::take(ranges));
            }
            (group, allocations)
        })
        .collect();
    let mut result = BTreeMap::<String, WeightPayloadGroup>::new();
    for initializer in &previous.initializers {
        let Some(source) = initializer
            .source
            .as_ref()
            .filter(|source| is_weight(source.role))
        else {
            continue;
        };
        let slice = source
            .slice
            .as_ref()
            .ok_or_else(|| invalid("cached weight initializer has no slice"))?;
        let Some((allocation, old_origin, origin)) = placements.get(&initializer.allocation) else {
            return Err(invalid(
                "cached weight allocation has no matching declared views",
            ));
        };
        let offset = i128::from(initializer.destination_offset) + i128::from(*origin)
            - i128::from(*old_origin);
        let offset = u64::try_from(offset)
            .map_err(|_| invalid("reused weight offset is outside its allocation"))?;
        let end = offset
            .checked_add(slice.nbytes)
            .ok_or_else(|| invalid("reused weight extent overflow"))?;
        let storage = if source.role == pwire::PayloadRole::TunedWeights as i32 {
            weight_payload::StorageClass::Tuned
        } else {
            weight_payload::StorageClass::Stable
        };
        let group = coverage.iter().find(|(group, allocations)| group.storage == storage &&
            allocations.get(allocation.as_str()).is_some_and(|ranges|
                ranges.iter().any(|&(start, stop)| start <= offset && end <= stop)))
            .map(|(group, _)| *group)
            .ok_or_else(|| invalid(format!(
                "rank {} cached {} weight initializer {}[{offset}..{end}] is not covered by one declared weight group; export with weights",
                previous.rank, storage.as_str(), allocation,
            )))?;
        let resolved = result
            .entry(group.name.clone())
            .or_insert_with(|| WeightPayloadGroup {
                name: group.name.clone(),
                storage: group.storage.as_str().to_string(),
                nbytes: 0,
                spans: Vec::new(),
                payload_id: slice.payload_id.clone(),
            });
        if resolved.payload_id != slice.payload_id {
            return Err(invalid(
                "cached weight group uses multiple payload objects; export with weights",
            ));
        }
        resolved.nbytes = payloads.length(&slice.payload_id)?;
        resolved.spans.push(WeightPayloadSourceSpan {
            allocation: allocation.clone(),
            destination_offset: offset,
            source_offset: slice.offset,
            nbytes: slice.nbytes,
        });
    }
    Ok(result.into_values().collect())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(export_reuse_weights, module)?)?;
    module.add_class::<NativeWeightSource>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Source, same_draft_body_quantization};
    use helios_core::runtime_core::wire::{RebuildManifest, RebuildSpeculation};

    #[test]
    fn draft_body_quantization_must_match_for_weight_reuse() {
        let manifest = |value: Option<&str>| RebuildManifest {
            speculation: Some(RebuildSpeculation {
                draft_body_quantization: value.map(str::to_string),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(same_draft_body_quantization(&manifest(None), &manifest(None)));
        assert!(same_draft_body_quantization(
            &manifest(Some("mxfp8")),
            &manifest(Some("mxfp8"))
        ));
        for (previous, expected) in [(None, Some("mxfp8")), (Some("mxfp8"), None)] {
            assert!(!same_draft_body_quantization(
                &manifest(previous),
                &manifest(expected)
            ));
        }
    }

    #[test]
    fn source_metadata_preserves_envelope_bytes() {
        let bytes: Vec<u8> = (0..=255).collect();
        for length in [0, 1, 2, 3, 255, 256] {
            let source = Source {
                envelope: bytes[..length].to_vec(),
                publishers: Vec::new(),
                rebuild_manifest: Some(serde_json::json!({"recipe":"fixture"})),
            };
            let json = serde_json::to_string(&source).unwrap();
            let metadata: serde_json::Value = serde_json::from_str(&json).unwrap();
            let encoded = metadata["envelope"].as_str().unwrap();
            assert_eq!(encoded.len(), length.div_ceil(3) * 4);
            let from_json: Source = serde_json::from_str(&json).unwrap();
            let from_value: Source = serde_json::from_value(metadata).unwrap();
            for restored in [from_json, from_value] {
                assert_eq!(restored.envelope, source.envelope);
                assert_eq!(restored.rebuild_manifest, source.rebuild_manifest);
            }
        }
    }

    #[test]
    fn source_metadata_uses_standard_padded_base64() {
        let source = Source {
            envelope: vec![0xfb, 0xff],
            publishers: Vec::new(),
            rebuild_manifest: None,
        };
        let metadata = serde_json::to_value(&source).unwrap();
        assert_eq!(metadata["envelope"], "+/8=");
        let restored: Source = serde_json::from_value(metadata).unwrap();
        assert_eq!(restored.envelope, source.envelope);
    }

    #[test]
    fn source_metadata_rejects_invalid_envelope_encoding() {
        for envelope in [
            serde_json::json!("*"),
            serde_json::json!("+/8"),
            serde_json::json!("+/9="),
            serde_json::json!("+/8=\n"),
            serde_json::json!([251, 255]),
            serde_json::Value::Null,
        ] {
            let metadata = serde_json::json!({"envelope":envelope,"publishers":[]});
            assert!(
                serde_json::from_value::<Source>(metadata.clone()).is_err(),
                "{metadata}"
            );
        }
    }
}
