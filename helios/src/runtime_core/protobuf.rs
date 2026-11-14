/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Decode generated Protobuf messages into the validated runtime description.

use super::format::*;
use crate::Result;
use crate::core::schema::wire as payload;
use std::collections::HashMap;

pub mod wire {
    include!(concat!(env!("OUT_DIR"), "/flare.runtime.rs"));
}

impl wire::RebuildManifest {
    /// Resolve the recorded draft-body conversion for preparation compatibility.
    /// An omitted DFlash2 override inherits online conversion's `all` scope;
    /// an omitted scope itself means `all` in persisted rebuild metadata.
    pub fn effective_draft_body_quantization(&self) -> Option<&str> {
        let speculation = self.speculation.as_ref()?;
        speculation.draft_body_quantization.as_deref().or_else(|| {
            (speculation.method == "dflash2"
                && self.online_mxfp8
                && matches!(self.online_mxfp8_scope.as_str(), "" | "all"))
            .then_some("mxfp8")
        })
    }
}

impl wire::ExecutionGraph {
    pub(super) fn packed_tokens(&self) -> u64 {
        if self.kind == "mixed" {
            self.prefill_capacity.saturating_add(
                self.batch_capacity
                    .saturating_sub(1)
                    .saturating_mul(self.token_capacity),
            )
        } else if self.packed_token_capacity != 0 {
            self.packed_token_capacity
        } else {
            self.batch_capacity.saturating_mul(self.token_capacity)
        }
    }
}

pub(super) fn size(value: u64) -> Result<usize> {
    usize::try_from(value).map_err(|_| invalid("resource size exceeds host address space"))
}

pub(super) fn serving_contract(
    value: wire::ServingContract,
    _prepared_concurrency: u64,
) -> Result<serde_json::Value> {
    if value.model_contract.as_ref().is_none_or(|contract| !(1..=3).contains(&contract.version)) {
        return Err(invalid("unsupported prepared model contract version; re-export the model core"));
    }
    let mut value = serde_json::to_value(value).map_err(|e| invalid(e.to_string()))?;
    value.as_object_mut().unwrap().retain(|_, v| !v.is_null());
    Ok(value)
}

pub(super) fn sampling_defaults(
    value: wire::DefaultSamplingParameters,
) -> Result<serde_json::Value> {
    let sampling = crate::model::SamplingParameters {
        seed: required(value.seed, "sampling seed")?,
        temperature: required(value.temperature, "sampling temperature")?,
        top_k: required(value.top_k, "sampling top_k")?,
        top_p: required(value.top_p, "sampling top_p")?,
        min_p: required(value.min_p, "sampling min_p")?,
        repetition_penalty: required(value.repetition_penalty, "sampling repetition_penalty")?,
        presence_penalty: required(value.presence_penalty, "sampling presence_penalty")?,
        frequency_penalty: required(value.frequency_penalty, "sampling frequency_penalty")?,
    };
    sampling
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    let mut result = serde_json::json!({
        "seed": sampling.seed, "temperature": sampling.temperature, "top_k": sampling.top_k,
        "top_p": sampling.top_p, "min_p": sampling.min_p,
        "repetition_penalty": sampling.repetition_penalty, "presence_penalty": sampling.presence_penalty,
        "frequency_penalty": sampling.frequency_penalty,
    });
    if let Some(ranges) = value.ranges {
        let ranges = sampling_ranges(ranges)?;
        ranges
            .validate(&sampling)
            .map_err(|error| invalid(error.to_string()))?;
        result["ranges"] =
            serde_json::to_value(ranges).map_err(|error| invalid(error.to_string()))?;
    }
    Ok(result)
}

fn sampling_ranges(value: wire::SamplingRanges) -> Result<crate::model::SamplingRanges> {
    use crate::model::{SamplingRange, SamplingRanges};
    let float = |range: wire::SamplingFloatRange| -> Result<SamplingRange<f32>> {
        Ok(SamplingRange {
            min: required(range.min, "sampling range min")?,
            max: required(range.max, "sampling range max")?,
        })
    };
    Ok(SamplingRanges {
        temperature: value.temperature.map(float).transpose()?,
        top_p: value.top_p.map(float).transpose()?,
        min_p: value.min_p.map(float).transpose()?,
        repetition_penalty: value.repetition_penalty.map(float).transpose()?,
        presence_penalty: value.presence_penalty.map(float).transpose()?,
        frequency_penalty: value.frequency_penalty.map(float).transpose()?,
        top_k: value
            .top_k
            .map(|range| -> Result<SamplingRange<i32>> {
                Ok(SamplingRange {
                    min: required(range.min, "sampling range min")?,
                    max: required(range.max, "sampling range max")?,
                })
            })
            .transpose()?,
    })
}

pub(super) fn frontend_settings(value: wire::FrontendSettings) -> Result<serde_json::Value> {
    wire::PromptEncoder::try_from(value.prompt_encoder)
        .map_err(|_| invalid("unsupported prompt encoder"))?;
    let tokens = required(value.tokenizer, "tokenizer settings")?;
    if value.eos_token_ids.is_empty()
        || value.eos_token_ids.len() > 8
        || value
            .eos_token_ids
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            != value.eos_token_ids.len()
    {
        return Err(invalid(
            "EOS token IDs must be one through eight distinct values",
        ));
    }
    let tokenizer = serde_json::json!({
        "bos_token": tokens.bos_token, "eos_token": tokens.eos_token, "unk_token": tokens.unk_token,
        "sep_token": tokens.sep_token, "pad_token": tokens.pad_token, "cls_token": tokens.cls_token,
        "mask_token": tokens.mask_token,
        "additional_special_tokens": tokens.additional_special_tokens.map(|tokens| serde_json::json!({"values": tokens.values})),
    });
    let image = value
        .image
        .map(|image| -> Result<serde_json::Value> {
            use wire::image_settings::Profile;
            Ok(match required(image.profile, "image profile")? {
                Profile::Qwen(config) => serde_json::json!({"qwen": {
                    "min_pixels": config.min_pixels, "max_pixels": config.max_pixels,
                    "position_side": config.position_side, "image_token_id": config.image_token_id,
                    "video_token_id": config.video_token_id,
                }}),
                Profile::Mimo(config) => serde_json::json!({"mimo": {
                    "min_pixels": config.min_pixels, "max_pixels": config.max_pixels,
                    "image_mean": config.image_mean, "image_std": config.image_std,
                }}),
                Profile::PatchMerge(config) => serde_json::json!({"patch_merge": {
                    "patch_size": config.patch_size, "min_pixels": config.min_pixels, "max_pixels": config.max_pixels,
                    "image_mean": config.image_mean, "image_std": config.image_std,
                    "image_token": config.image_token, "start_token": config.start_token, "end_token": config.end_token,
                }}),
                Profile::DeepseekRowMajor(config) => serde_json::json!({"deepseek_row_major": {
                    "min_pixels": config.min_pixels, "max_tokens": config.max_tokens,
                    "max_width_height_ratio": config.max_width_height_ratio,
                }}),
                Profile::Deepseek4(_) => serde_json::json!({"deepseek4": {}}),
                Profile::Moonvit(_) => serde_json::json!({"moonvit": {}}),
                Profile::Step(_) => serde_json::json!({"step": {}}),
                Profile::Radio(_) => serde_json::json!({"radio": {}}),
                Profile::Glm(_) => serde_json::json!({"glm": {}}),
            })
        })
        .transpose()?;
    Ok(
        serde_json::json!({"tokenizer": tokenizer, "image": image, "eos_token_ids": value.eos_token_ids,
                          "template_encoding": value.template_encoding,
                          "prompt_encoder": value.prompt_encoder,
                          "image_placeholder_token_id": value.image_placeholder_token_id}),
    )
}
fn required<T>(value: Option<T>, name: &str) -> Result<T> {
    value.ok_or_else(|| invalid(format!("missing protobuf {name}")))
}
fn convert<T, U>(values: Vec<T>, f: impl Fn(T) -> Result<U>) -> Result<Vec<U>> {
    values.into_iter().map(f).collect()
}
fn array<T, const N: usize>(values: Vec<T>, name: &str) -> Result<[T; N]> {
    values
        .try_into()
        .map_err(|_| invalid(format!("invalid protobuf {name} width")))
}
fn reference(value: wire::Reference) -> Result<Reference> {
    Ok(Reference {
        allocation: value.allocation,
        offset: size(value.offset)?,
    })
}
fn relocation(value: wire::Relocation) -> Result<Relocation> {
    Ok(Relocation {
        offset: size(value.offset)?,
        target: reference(required(value.target, "relocation target")?)?,
        kind: match wire::RelocationKind::try_from(value.kind) {
            Ok(wire::RelocationKind::Pointer) => RelocationKind::Pointer,
            Ok(wire::RelocationKind::TensorMap) => RelocationKind::TensorMap,
            Err(_) => return Err(invalid("unknown protobuf relocation kind")),
        },
    })
}
fn blob(value: wire::Blob) -> Result<Blob> {
    Ok(Blob {
        data: value.data,
        relocations: convert(value.relocations, relocation)?,
    })
}
pub(super) fn node(value: wire::Node) -> Result<Node> {
    use wire::node::Operation;
    Ok(match required(value.operation, "node operation")? {
        Operation::Kernel(value) => Node::Kernel {
            function: size(value.function)?,
            grid: array(value.grid, "kernel grid")?,
            block: array(value.block, "kernel block")?,
            shared: value.shared,
            arguments: convert(value.arguments, blob)?,
            attributes: convert(value.attributes, |value| {
                Ok(Attribute {
                    id: value.id,
                    value: blob(required(value.value, "node attribute")?)?,
                })
            })?,
        },
        Operation::Memset(value) => Node::Memset {
            destination: reference(required(value.destination, "memset destination")?)?,
            pitch: size(value.pitch)?,
            value: value.value,
            element_size: value.element_size,
            width: size(value.width)?,
            height: size(value.height)?,
        },
        Operation::Copy(value) => Node::Copy {
            source: reference(required(value.source, "copy source")?)?,
            destination: reference(required(value.destination, "copy destination")?)?,
            nbytes: size(value.nbytes)?,
        },
        Operation::Empty(_) => Node::Empty,
        Operation::EventWait(value) => Node::EventWait { event: value.event },
    })
}

pub(super) fn decode(mut value: wire::BundleIndex) -> Result<Core> {
    if let Some(execution) = &mut value.execution {
        for graph in &mut execution.graphs {
            if graph.kind == "speculative" && graph.packed_token_capacity == 0 {
                graph.packed_token_capacity = graph.batch_capacity.checked_mul(graph.token_capacity)
                    .ok_or_else(|| invalid("rectangular speculative capacity overflow"))?;
            }
        }
    }
    let context = required(value.context, "CUDA context")?;
    let core = Core {
        resources: value.resources,
        events: value.events,
        architecture: array(value.architecture, "architecture")?,
        context: Context {
            flags: context.flags,
            cache: context.cache,
            limits: convert(context.limits, |v| Ok([size(v.id)?, size(v.value)?]))?,
        },
        reservations: convert(value.reservations, |v| {
            Ok(Reservation {
                address: v.address,
                nbytes: size(v.nbytes)?,
            })
        })?,
        allocations: convert(value.allocations, |v| {
            Ok(Allocation {
                name: v.name,
                sizing: v.sizing,
                nbytes: size(v.nbytes)?,
                alignment: size(v.alignment)?,
                file: v.file,
                data_spans: convert(v.data_spans, |v| Ok([size(v.offset)?, size(v.nbytes)?]))?,
                sources: vec![],
                relocations: convert(v.relocations, relocation)?,
                address: v.address,
                memory: match wire::Memory::try_from(v.memory) {
                    Ok(wire::Memory::Device) => Memory::Device,
                    Ok(wire::Memory::MappedHost) => Memory::MappedHost,
                    Ok(wire::Memory::Peer) => Memory::Peer,
                    Err(_) => return Err(invalid("unknown protobuf memory kind")),
                },
            })
        })?,
        modules: convert(value.modules, |v| {
            Ok(Module {
                name: v.name,
                file: v.file,
                globals: convert(v.globals, |v| {
                    Ok(Global {
                        symbol: v.symbol,
                        file: v.file,
                        nbytes: size(v.nbytes)?,
                        relocations: convert(v.relocations, relocation)?,
                    })
                })?,
            })
        })?,
        functions: convert(
            value.functions.into_iter().enumerate().collect(),
            |(id, v)| {
                Ok(Function {
                    id,
                    module: v.module,
                    symbol: v.symbol,
                    parameter_sizes: convert(v.parameter_sizes, size)?,
                    attributes: v.attributes.into_iter().map(|v| [v.id, v.value]).collect(),
                })
            },
        )?,
        graphs: convert(value.graphs, |v| {
            Ok(Graph {
                name: v.name,
                flags: v.flags,
                resource: v.resource,
                node_count: size(v.node_count)?,
                edge_count: size(v.edge_count)?,
                functions: convert(v.functions, size)?,
            })
        })?,
        execution: value.execution,
    };
    Ok(core)
}

pub(super) fn edge(value: wire::Edge) -> Result<Edge> {
    Ok(Edge {
        source: size(value.source)?,
        destination: size(value.destination)?,
        data: array(value.data, "edge data")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    #[test]
    fn draft_body_precision_resolves_only_documented_inheritance() {
        for (method, online, scope, explicit, expected) in [
            ("dflash2", true, "all", None, Some("mxfp8")),
            ("dflash2", true, "", None, Some("mxfp8")),
            ("dflash2", false, "all", None, None),
            ("dflash2", false, "", None, None),
            ("dflash2", true, "kda-input", None, None),
            ("dflash2", true, "unknown", None, None),
            ("mtp", true, "all", None, None),
            ("", true, "all", None, None),
            ("dflash2", true, "all", Some("mxfp8"), Some("mxfp8")),
            ("dflash2", true, "kda-input", Some("mxfp8"), Some("mxfp8")),
            ("dflash2", false, "all", Some("mxfp8"), Some("mxfp8")),
            ("dflash2", true, "all", Some("unknown"), Some("unknown")),
            ("dflash2", true, "all", Some(""), Some("")),
        ] {
            let manifest = wire::RebuildManifest {
                online_mxfp8: online,
                online_mxfp8_scope: scope.into(),
                speculation: Some(wire::RebuildSpeculation {
                    method: method.into(),
                    draft_body_quantization: explicit.map(str::to_owned),
                    ..Default::default()
                }),
                ..Default::default()
            };
            let encoded = manifest.encode_to_vec();
            let decoded = wire::RebuildManifest::decode(encoded.as_slice()).unwrap();
            assert_eq!(decoded.effective_draft_body_quantization(), expected);
            assert_eq!(decoded.encode_to_vec(), encoded);
        }
        let no_draft = wire::RebuildManifest {
            online_mxfp8: true,
            online_mxfp8_scope: "all".into(),
            ..Default::default()
        };
        assert_eq!(no_draft.effective_draft_body_quantization(), None);
    }

    #[test]
    fn supported_contract_versions_preserve_their_declared_version() {
        for version in [1, 2, 3] {
            let contract = wire::ServingContract {
                model_contract: Some(wire::ModelContract { version, ..Default::default() }),
                ..Default::default()
            };
            let config = serving_contract(contract, 8).unwrap();
            assert_eq!(config["model_contract"]["version"], version);
        }
    }

    #[test]
    fn omitted_speculative_target_capacity_preserves_rectangular_geometry() {
        let bundle = |batch, width, packed| wire::BundleIndex {
            architecture: vec![12, 1],
            context: Some(wire::Context::default()),
            execution: Some(wire::Execution {
                abi_version: 3,
                graphs: vec![wire::ExecutionGraph {
                    kind: "speculative".into(), batch_capacity: batch,
                    token_capacity: width, packed_token_capacity: packed,
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        for (batch, width, packed, expected) in [(2, 4, 0, 8), (8, 1, 0, 8), (2, 4, 6, 6)] {
            let core = decode(bundle(batch, width, packed)).unwrap();
            assert_eq!(core.execution.unwrap().graphs[0].packed_token_capacity, expected);
        }
        assert!(decode(bundle(u64::MAX, 2, 0)).is_err());
    }

    #[test]
    fn rebuild_catalogs_do_not_change_native_bundle_loading() {
        // SM120 architecture and an empty CUDA context, serialized without a catalog.
        let bytes = [10, 2, 12, 0, 66, 0];
        let legacy = wire::BundleIndex::decode(bytes.as_slice()).unwrap();
        assert!(legacy.rebuild_weights.is_none());
        let expected = decode(legacy.clone()).unwrap();
        for version in [0, 1, 99] {
            let mut described = legacy.clone();
            described.rebuild_weights = Some(wire::RebuildWeights {
                version,
                unavailable: vec!["weights".into()],
                ..Default::default()
            });
            let actual = decode(described).unwrap();
            assert_eq!(actual.architecture, expected.architecture);
            assert_eq!(actual.context.flags, expected.context.flags);
            assert_eq!(actual.allocations.len(), expected.allocations.len());
            assert_eq!(actual.graphs.len(), expected.graphs.len());
        }
    }

    fn defaults() -> wire::DefaultSamplingParameters {
        wire::DefaultSamplingParameters {
            seed: Some(u64::MAX),
            temperature: Some(1.),
            top_k: Some(-1),
            top_p: Some(0.95),
            min_p: Some(0.),
            repetition_penalty: Some(1.),
            presence_penalty: Some(0.),
            frequency_penalty: Some(0.),
            ranges: None,
        }
    }

    #[test]
    fn sampling_round_trip_preserves_explicit_zeroes_and_uint64_seed() {
        let value = defaults();
        let decoded =
            wire::DefaultSamplingParameters::decode(value.encode_to_vec().as_slice()).unwrap();
        let json = sampling_defaults(decoded).unwrap();
        assert_eq!(json["seed"].as_u64(), Some(u64::MAX));
        assert_eq!(json["top_k"], -1);
        assert_eq!(json["frequency_penalty"], 0.);
        #[cfg(feature = "server")]
        {
            let sampling: crate::model::SamplingParameters = serde_json::from_value(json).unwrap();
            assert_eq!(sampling.seed, u64::MAX);
            assert_eq!(sampling.top_k, -1);
        }
    }

    #[test]
    fn serialized_sampling_without_ranges_preserves_defaults() {
        // DefaultSamplingParameters fields 1 through 8, without field 9.
        let bytes = [
            0x08, 0x00, 0x15, 0x00, 0x00, 0x80, 0x3f, 0x18, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0x01, 0x25, 0x33, 0x33, 0x73, 0x3f, 0x2d, 0x00, 0x00, 0x00, 0x00,
            0x35, 0x00, 0x00, 0x80, 0x3f, 0x3d, 0x00, 0x00, 0x00, 0x00, 0x45, 0x00, 0x00, 0x00,
            0x00,
        ];
        let decoded = wire::DefaultSamplingParameters::decode(bytes.as_slice()).unwrap();
        assert!(decoded.ranges.is_none());
        assert_eq!(
            decoded,
            wire::DefaultSamplingParameters {
                seed: Some(0),
                ..defaults()
            }
        );
        let json = sampling_defaults(decoded).unwrap();
        assert!(json.get("ranges").is_none());
        assert_eq!(json["seed"], 0);
        assert_eq!(json["top_k"], -1);
        assert_eq!(json["temperature"], 1.);
    }

    #[test]
    fn sampling_ranges_preserve_explicit_zero_bounds_and_reject_invalid_metadata() {
        let mut value = defaults();
        value.ranges = Some(wire::SamplingRanges {
            temperature: Some(wire::SamplingFloatRange {
                min: Some(0.5),
                max: Some(1.5),
            }),
            top_k: Some(wire::SamplingIntRange {
                min: Some(-1),
                max: Some(100),
            }),
            presence_penalty: Some(wire::SamplingFloatRange {
                min: Some(0.),
                max: Some(0.),
            }),
            ..Default::default()
        });
        let decoded =
            wire::DefaultSamplingParameters::decode(value.encode_to_vec().as_slice()).unwrap();
        let json = sampling_defaults(decoded).unwrap();
        assert_eq!(
            json["ranges"]["presence_penalty"],
            serde_json::json!({"min":0.,"max":0.})
        );
        assert_eq!(
            json["ranges"]["top_k"],
            serde_json::json!({"min":-1,"max":100})
        );
        for (min, max) in [
            (None, Some(1.5)),
            (Some(0.5), None),
            (Some(0.), Some(1.5)),
            (Some(1.5), Some(2.)),
            (Some(2.), Some(0.5)),
            (Some(f32::NAN), Some(1.5)),
            (Some(0.5), Some(f32::INFINITY)),
        ] {
            value.ranges.as_mut().unwrap().temperature =
                Some(wire::SamplingFloatRange { min, max });
            assert!(
                sampling_defaults(value.clone()).is_err(),
                "{min:?}..={max:?}"
            );
        }
        value.ranges = Some(wire::SamplingRanges {
            top_k: Some(wire::SamplingIntRange {
                min: None,
                max: Some(100),
            }),
            ..Default::default()
        });
        assert!(sampling_defaults(value).is_err());
    }

    #[test]
    fn incomplete_or_invalid_defaults_never_inherit_model_policy() {
        let mut value = defaults();
        value.top_k = None;
        assert!(
            sampling_defaults(value)
                .unwrap_err()
                .to_string()
                .contains("sampling top_k")
        );
        let mut value = defaults();
        value.top_p = Some(0.);
        assert!(sampling_defaults(value).is_err());
        let mut value = defaults();
        value.temperature = Some(f32::NAN);
        assert!(sampling_defaults(value).is_err());
    }

    #[test]
    fn frontend_round_trip_keeps_eos_and_empty_special_token_list_distinct_from_absence() {
        let value = wire::FrontendSettings {
            template_encoding: 0,
            prompt_encoder: 0,
            image_placeholder_token_id: None,
            tokenizer: Some(wire::TokenizerSettings {
                bos_token: Some("<bos>".into()),
                eos_token: Some("<eos>".into()),
                additional_special_tokens: Some(wire::TokenList { values: Vec::new() }),
                ..Default::default()
            }),
            eos_token_ids: vec![17, 23],
            image: None,
        };
        let decoded = wire::FrontendSettings::decode(value.encode_to_vec().as_slice()).unwrap();
        let json = frontend_settings(decoded).unwrap();
        assert_eq!(json["eos_token_ids"], serde_json::json!([17, 23]));
        assert_eq!(json["tokenizer"]["bos_token"], "<bos>");
        assert_eq!(
            json["tokenizer"]["additional_special_tokens"]["values"],
            serde_json::json!([])
        );
        assert!(json["tokenizer"]["pad_token"].is_null());
        #[cfg(feature = "server")]
        {
            let frontend: crate::frontend::FrontendSettings = serde_json::from_value(json).unwrap();
            assert!(
                frontend
                    .tokenizer
                    .additional_special_tokens
                    .unwrap()
                    .values
                    .is_empty()
            );
        }
    }

    #[test]
    fn serialized_text_and_deepseek_image_profiles_keep_their_meaning() {
        // Tokenizer field 1, optional image field 2, and packed EOS field 3
        // use the wire representation exported before row-major image profiles.
        for (bytes, image) in [
            (&[0x0a, 0x00, 0x1a, 0x01, 0x01][..], false),
            (
                &[0x0a, 0x00, 0x12, 0x02, 0x1a, 0x00, 0x1a, 0x01, 0x01][..],
                true,
            ),
        ] {
            let decoded = wire::FrontendSettings::decode(bytes).unwrap();
            let json = frontend_settings(decoded).unwrap();
            assert_eq!(json["eos_token_ids"], serde_json::json!([1]));
            assert_eq!(json["prompt_encoder"], 0);
            assert!(json["image_placeholder_token_id"].is_null());
            if image {
                assert_eq!(json["image"], serde_json::json!({"deepseek4": {}}));
            } else {
                assert!(json["image"].is_null());
            }
            #[cfg(feature = "server")]
            {
                let frontend: crate::frontend::FrontendSettings =
                    serde_json::from_value(json).unwrap();
                assert_eq!(
                    frontend.prompt_encoder,
                    crate::frontend::PromptEncoder::Jinja
                );
                assert_eq!(frontend.image_placeholder_token_id, None);
                if let Some(settings) = frontend.image {
                    let processor =
                        crate::frontend::images::ImageProcessor::load(settings, Default::default())
                            .unwrap();
                    assert!(processor.geometry(701, 503).unwrap().tokens <= 384);
                }
            }
        }
    }

    #[test]
    fn serialized_glm_profile_selects_only_image_preparation() {
        // Field 8 adds the GLM profile without changing any preceding profile tag.
        let bytes = &[0x0a, 0x00, 0x12, 0x02, 0x42, 0x00, 0x1a, 0x01, 0x01][..];
        let decoded = wire::FrontendSettings::decode(bytes).unwrap();
        let json = frontend_settings(decoded).unwrap();
        assert_eq!(json["image"], serde_json::json!({"glm": {}}));
        assert_eq!(json["prompt_encoder"], 0);
        #[cfg(feature = "server")]
        {
            let frontend: crate::frontend::FrontendSettings = serde_json::from_value(json).unwrap();
            let settings = frontend.image.unwrap();
            assert_eq!(
                settings.preparation_kind(),
                crate::model::contract::ImagePreparationKind::Glm
            );
            let processor =
                crate::frontend::images::ImageProcessor::load(settings, Default::default())
                    .unwrap();
            assert_eq!(processor.geometry(112, 112).unwrap().tokens, 16);
        }
    }

    #[test]
    fn prompt_encoder_round_trip_preserves_explicit_image_token_and_rejects_unknown_encoder() {
        for encoder in [1, 2] {
            let value = wire::FrontendSettings {
                tokenizer: Some(wire::TokenizerSettings::default()),
                eos_token_ids: vec![1],
                prompt_encoder: encoder,
                image_placeholder_token_id: Some(129264),
                ..Default::default()
            };
            let decoded = wire::FrontendSettings::decode(value.encode_to_vec().as_slice()).unwrap();
            let json = frontend_settings(decoded).unwrap();
            assert_eq!(json["prompt_encoder"], encoder);
            assert_eq!(json["image_placeholder_token_id"], 129264);
            #[cfg(feature = "server")]
            {
                let frontend: crate::frontend::FrontendSettings =
                    serde_json::from_value(json).unwrap();
                assert_eq!(frontend.prompt_encoder as i32, encoder);
                assert_eq!(frontend.image_placeholder_token_id, Some(129264));
            }
        }
        let error = frontend_settings(wire::FrontendSettings {
            prompt_encoder: 37,
            ..Default::default()
        })
        .unwrap_err();
        assert!(error.to_string().contains("unsupported prompt encoder"));
    }

    #[test]
    fn image_contract_contains_only_the_selected_profile() {
        let value = wire::FrontendSettings {
            template_encoding: 0,
            prompt_encoder: 0,
            image_placeholder_token_id: None,
            tokenizer: Some(wire::TokenizerSettings::default()),
            eos_token_ids: vec![1],
            image: Some(wire::ImageSettings {
                profile: Some(wire::image_settings::Profile::Deepseek4(wire::Empty {})),
            }),
        };
        let json = frontend_settings(value).unwrap();
        assert_eq!(json["image"], serde_json::json!({"deepseek4": {}}));
        #[cfg(feature = "server")]
        {
            let frontend: crate::frontend::FrontendSettings = serde_json::from_value(json).unwrap();
            let _ = crate::frontend::images::ImageProcessor::load(
                frontend.image.unwrap(),
                Default::default(),
            )
            .unwrap();
        }
    }
    #[test]
    fn serialized_scored_prefill_preserves_operation_and_prompt_target_values() {
        // GraphContract(kind=12) and ValueBinding(source=39) predate packed rounds.
        let graph = wire::GraphContract::decode(
            b"\x0a\x05score\x12\x0d\x0a\x07targets\x10\x27\x18\x04\x40\x0c".as_slice(),
        )
        .unwrap();
        let contract: crate::model::contract::GraphContract =
            serde_json::from_value(serde_json::to_value(graph).unwrap()).unwrap();
        assert_eq!(
            contract.kind,
            crate::model::contract::GraphOperation::ScoredPrefill
        );
        assert_eq!(
            contract.inputs[0].source,
            crate::model::contract::ValueSource::PromptTargetToken
        );
        assert_eq!(
            contract.inputs[0].indexing,
            crate::model::contract::InputIndexing::PackedToken
        );
        let execution =
            wire::ExecutionGraph::decode(b"\x12\x0escored_prefill\x20\x01\x28\x40".as_slice())
                .unwrap();
        assert_eq!(
            crate::execution::GraphKind::try_from(execution.kind.as_str()).unwrap(),
            crate::execution::GraphKind::ScoredPrefill
        );
        assert_eq!(execution.draft_tokens, None);
        assert_eq!(execution.packed_tokens(), 64);
    }

    #[test]
    fn unified_generation_rejects_removed_operations_and_preserves_geometry() {
        for operation in [0, 10, 11, 13, 99] {
            assert!(crate::model::contract::GraphOperation::try_from(operation).is_err());
        }
        for kind in ["proposal", "verification", "packed_speculative"] {
            assert!(crate::execution::GraphKind::try_from(kind).is_err());
        }
        let graph = wire::ExecutionGraph {
            kind: "speculative".into(), batch_capacity: 32, token_capacity: 6,
            draft_tokens: Some(7), verify_tokens: Some(6), packed_token_capacity: 128,
            draft_source: 1, ..Default::default()
        };
        let restored = wire::ExecutionGraph::decode(graph.encode_to_vec().as_slice()).unwrap();
        assert_eq!((restored.draft_tokens, restored.verify_tokens, restored.packed_token_capacity),
            (Some(7), Some(6), 128));
    }

    #[test]
    fn completion_capacity_wire_default_and_positive_values_roundtrip() {
        for capacity in [0u32, 1, 8, 16, 33, 65] {
            let execution = wire::Execution { abi_version: 6, completion_token_capacity: capacity,
                ..Default::default() };
            let restored = wire::Execution::decode(execution.encode_to_vec().as_slice()).unwrap();
            assert_eq!(restored.completion_token_capacity, capacity);
            assert_eq!(crate::execution::resolve_completion_token_capacity(restored.completion_token_capacity),
                if capacity == 0 { 8 } else { capacity as usize });
        }
    }

    #[test]
    fn startup_tuning_metadata_roundtrips_physical_and_semantic_bindings() {
        let binding = wire::TunedBinding { buffer: "winner".into(), tuning_graph: "tune".into() };
        let execution = wire::Execution {
            tuning_graphs: vec![wire::TuningGraph {
                name: "tune".into(), outputs: vec!["winner".into()], status_buffer: "status".into(),
            }],
            graphs: vec![wire::ExecutionGraph { tuned_bindings: vec![binding.clone()], ..Default::default() }],
            ..Default::default()
        };
        let restored = wire::Execution::decode(execution.encode_to_vec().as_slice()).unwrap();
        assert_eq!(restored, execution);
        let semantic = wire::GraphContract { kind: 5, tuned_bindings: vec![binding], ..Default::default() };
        let restored = wire::GraphContract::decode(semantic.encode_to_vec().as_slice()).unwrap();
        let contract: crate::model::contract::GraphContract =
            serde_json::from_value(serde_json::to_value(restored).unwrap()).unwrap();
        assert_eq!(contract.tuned_bindings[0].buffer, "winner");
        assert_eq!(contract.tuned_bindings[0].tuning_graph, "tune");
    }

    #[test]
    fn unknown_prepared_contract_versions_are_rejected() {
        for version in [0, 4, u32::MAX] {
            let value = wire::ServingContract { model_contract: Some(wire::ModelContract {
                version, ..Default::default()
            }), ..Default::default() };
            assert!(serving_contract(value, 4).unwrap_err().to_string().contains("re-export"));
        }
    }

    #[cfg(feature = "server")]
    #[test]
    fn serialized_tag_parsers_keep_existing_ids_defaults_and_serving_behavior() {
        use crate::model::contract::{FrontendContract, ReasoningParserKind, ToolParserKind};
        use crate::output::tagged::OutputConfig as TaggedConfig;
        use crate::output::{Input, OutputConfig, OutputPipeline, Termination};
        use crate::server::output::Accumulator;
        for (kind, expected, payload) in [
            (
                1,
                ToolParserKind::QwenCoder,
                "<tool_call><function=lookup></function></tool_call>",
            ),
            (
                2,
                ToolParserKind::GlmArgumentTags,
                "<tool_call>lookup</tool_call>",
            ),
            (
                3,
                ToolParserKind::Dsml,
                "<｜DSML｜tool_calls><｜DSML｜invoke name=\"lookup\"></｜DSML｜invoke></｜DSML｜tool_calls>",
            ),
        ] {
            let tool = wire::ToolParser::decode(&[0x08, kind][..]).unwrap();
            let reason =
                wire::ReasoningParser::decode(&b"\x08\x01\x12\x07<think>\x1a\x08</think>"[..])
                    .unwrap();
            let value = wire::FrontendContract {
                tools: Some(tool),
                reasoning: Some(reason),
                assistant_marker: "assistant".into(),
                ..Default::default()
            };
            let syntax: FrontendContract =
                serde_json::from_value(serde_json::to_value(value).unwrap()).unwrap();
            assert_eq!(syntax.tools.as_ref().unwrap().kind, expected);
            assert_eq!(syntax.tools.as_ref().unwrap().wrapping_newlines, 0);
            assert_eq!(
                syntax.reasoning.as_ref().unwrap().kind,
                ReasoningParserKind::ThinkTags
            );
            let mut parser = OutputPipeline::new(OutputConfig {
                syntax,
                control_tokens: vec![],
                content: TaggedConfig {
                    structured: false,
                    start: crate::output::tagged::OutputStart::Reasoning,
                    tools: vec![serde_json::json!({"function":{"name":"lookup"}})],
                },
            });
            let mut output = Accumulator::new("core".into());
            let text = format!("plan</think>{payload}");
            for character in text.chars() {
                output
                    .apply(parser.push(Input::Text(&character.to_string())).unwrap())
                    .unwrap();
            }
            output
                .apply(parser.finish(Termination::Eos).unwrap())
                .unwrap();
            output.finish().unwrap();
            assert_eq!(output.message()["reasoning_content"], "plan");
            assert_eq!(
                output.message()["tool_calls"][0]["function"]["name"],
                "lookup"
            );
        }
        assert_eq!(
            ToolParserKind::try_from(wire::ToolParser::decode(&[0x08, 4][..]).unwrap().kind)
                .unwrap(),
            ToolParserKind::Xtml
        );
        assert_eq!(
            ReasoningParserKind::try_from(
                wire::ReasoningParser::decode(&[0x08, 2][..]).unwrap().kind
            )
            .unwrap(),
            ReasoningParserKind::XtmlThink
        );
        assert_eq!(
            ToolParserKind::try_from(wire::ToolParser::decode(&[0x08, 6][..]).unwrap().kind)
                .unwrap(),
            ToolParserKind::GemmaCall
        );
        assert_eq!(
            ReasoningParserKind::try_from(
                wire::ReasoningParser::decode(&[0x08, 4][..]).unwrap().kind
            )
            .unwrap(),
            ReasoningParserKind::ChannelThought
        );
        assert!(ToolParserKind::try_from(wire::ToolParser::decode(&[][..]).unwrap().kind).is_err());
        let syntax: FrontendContract = serde_json::from_value(
            serde_json::to_value(wire::FrontendContract {
                assistant_marker: "assistant".into(),
                ..Default::default()
            })
            .unwrap(),
        )
        .unwrap();
        assert!(syntax.tools.is_none() && syntax.reasoning.is_none());
    }
}

/// Adapt one authenticated payload-core rank into the loader's runtime description.
///
/// Payload identity and binding are already authenticated by `core::protection`; this function
/// only translates validated payload-relative records into the resource keys and data spans the
/// shared loader consumes. Every resource key it creates names an open container and a byte range
/// the caller resolved, so reads never depend on a persisted logical-file layout.
pub(super) fn payload_rank(
    rank: &payload::RankBody,
    architecture: [i32; 2],
    placements: &Placements,
) -> Result<Core> {
    let mut resources: HashMap<String, wire::Span> = HashMap::new();
    let mut span_for = |key: String, use_: &payload::PayloadUse| -> Result<String> {
        let slice = use_
            .slice
            .as_ref()
            .ok_or_else(|| invalid("payload use has no slice"))?;
        let placement = placements
            .get(slice.payload_id.as_slice())
            .ok_or_else(|| invalid("bound payload has no physical placement"))?;
        let offset = placement
            .offset
            .checked_add(slice.offset)
            .ok_or_else(|| invalid("payload placement offset overflow"))?;
        if slice
            .offset
            .checked_add(slice.nbytes)
            .is_none_or(|end| end > placement.nbytes)
        {
            return Err(invalid("payload slice exceeds its placed payload"));
        }
        resources.insert(
            key.clone(),
            wire::Span {
                filename: placement.token.clone(),
                offset,
                nbytes: slice.nbytes,
            },
        );
        Ok(key)
    };

    let mut allocations = Vec::with_capacity(rank.allocations.len());
    let mut by_allocation: HashMap<&str, Vec<(u64, u64, Source)>> = HashMap::new();
    for (index, initializer) in rank.initializers.iter().enumerate() {
        let use_ = initializer
            .source
            .as_ref()
            .ok_or_else(|| invalid("initializer has no source"))?;
        let slice = use_
            .slice
            .as_ref()
            .ok_or_else(|| invalid("initializer source has no slice"))?;
        let key = span_for(format!("initializer/{index}"), use_)?;
        by_allocation
            .entry(initializer.allocation.as_str())
            .or_default()
            .push((
                initializer.destination_offset,
                slice.nbytes,
                Source {
                    resource: key,
                    offset: 0,
                    nbytes: slice.nbytes,
                },
            ));
    }
    for allocation in &rank.allocations {
        let mut spans = by_allocation
            .remove(allocation.name.as_str())
            .unwrap_or_default();
        spans.sort_by_key(|(offset, _, _)| *offset);
        allocations.push(Allocation {
            name: allocation.name.clone(),
            nbytes: size(allocation.nbytes)?,
            alignment: size(allocation.alignment)?,
            file: None,
            data_spans: spans
                .iter()
                .map(|(offset, nbytes, _)| Ok([size(*offset)?, size(*nbytes)?]))
                .collect::<Result<Vec<_>>>()?,
            sources: spans.into_iter().map(|(_, _, source)| source).collect(),
            relocations: convert(allocation.relocations.clone(), relocation)?,
            address: allocation.address,
            memory: match wire::Memory::try_from(allocation.memory) {
                Ok(wire::Memory::Device) => Memory::Device,
                Ok(wire::Memory::MappedHost) => Memory::MappedHost,
                Ok(wire::Memory::Peer) => Memory::Peer,
                Err(_) => return Err(invalid("unknown protobuf memory kind")),
            },
            sizing: allocation.sizing.clone(),
        });
    }

    let mut executable_keys = Vec::with_capacity(rank.resources.len());
    for resource in &rank.resources {
        let use_ = payload::PayloadUse {
            slice: resource.stored.clone(),
            role: payload::PayloadRole::Kernels as i32,
        };
        executable_keys.push(span_for(
            format!("executable/{}", resource.resource_id),
            &use_,
        )?);
    }

    let mut modules = Vec::with_capacity(rank.modules.len());
    for (module_index, module) in rank.modules.iter().enumerate() {
        let cubin = executable_keys
            .get(module.cubin as usize)
            .ok_or_else(|| invalid("module cubin reference is outside the rank"))?
            .clone();
        let mut globals = Vec::with_capacity(module.globals.len());
        for (global_index, global) in module.globals.iter().enumerate() {
            let contents = global
                .contents
                .as_ref()
                .ok_or_else(|| invalid("module global has no contents"))?;
            // Structural indices, not names: bounded names may contain any separator character,
            // so string concatenation could collide two distinct globals.
            let key = span_for(format!("global/{module_index}/{global_index}"), contents)?;
            globals.push(Global {
                symbol: global.symbol.clone(),
                file: key,
                nbytes: size(global.nbytes)?,
                relocations: convert(global.relocations.clone(), relocation)?,
            });
        }
        modules.push(Module {
            name: module.name.clone(),
            file: cubin,
            globals,
        });
    }

    let functions = rank
        .functions
        .iter()
        .enumerate()
        .map(|(id, function)| {
            Ok(Function {
                id,
                module: function.module.clone(),
                symbol: function.symbol.clone(),
                parameter_sizes: convert(function.parameter_sizes.clone(), size)?,
                attributes: function
                    .attributes
                    .iter()
                    .map(|value| [value.id, value.value])
                    .collect(),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let graphs = rank
        .graphs
        .iter()
        .map(|graph| {
            Ok(Graph {
                name: graph.name.clone(),
                flags: graph.flags,
                resource: executable_keys
                    .get(graph.resource as usize)
                    .ok_or_else(|| invalid("graph resource reference is outside the rank"))?
                    .clone(),
                node_count: size(graph.node_count)?,
                edge_count: size(graph.edge_count)?,
                functions: convert(graph.functions.clone(), size)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let context = rank
        .context
        .clone()
        .ok_or_else(|| invalid("rank has no CUDA context description"))?;
    let core = Core {
        resources,
        architecture,
        allocations,
        modules,
        functions,
        graphs,
        reservations: convert(rank.reservations.clone(), |value| {
            Ok(Reservation {
                address: value.address,
                nbytes: size(value.nbytes)?,
            })
        })?,
        execution: rank.execution.clone(),
        context: Context {
            flags: context.flags,
            cache: context.cache,
            limits: convert(context.limits, |value| {
                Ok([size(value.id)?, size(value.value)?])
            })?,
        },
        events: rank.events.clone(),
    };
    Ok(core)
}

#[cfg(test)]
mod payload_tests {
    use super::*;

    fn placement(token: &str, offset: u64, nbytes: u64) -> Placement {
        Placement {
            token: token.into(),
            offset,
            nbytes,
        }
    }
    fn use_of(id: u8, offset: u64, nbytes: u64) -> payload::PayloadUse {
        payload::PayloadUse {
            slice: Some(payload::PayloadSlice {
                payload_id: vec![id; 32],
                offset,
                nbytes,
            }),
            role: payload::PayloadRole::Weights as i32,
        }
    }

    fn rank() -> payload::RankBody {
        payload::RankBody {
            rank: 0,
            deployment_id: "sm120-tp1".into(),
            context: Some(wire::Context::default()),
            reservations: vec![wire::Reservation {
                address: 0x1000_0000,
                nbytes: 8192,
            }],
            allocations: vec![payload::Allocation {
                name: "weights".into(),
                nbytes: 8192,
                alignment: 4096,
                address: Some(0x1000_0000),
                memory: wire::Memory::Device as i32,
                sizing: None,
                relocations: vec![],
            }],
            // Deliberately out of destination order and drawn from two different payloads: the
            // adapter must order destinations and keep each source exact.
            initializers: vec![
                payload::Initializer {
                    allocation: "weights".into(),
                    destination_offset: 4096,
                    source: Some(use_of(2, 16, 4096)),
                },
                payload::Initializer {
                    allocation: "weights".into(),
                    destination_offset: 0,
                    source: Some(use_of(1, 0, 4096)),
                },
            ],
            payloads: vec![],
            resources: vec![payload::ExecutableResource {
                resource_id: 9,
                kind: payload::ExecutableKind::Graph as i32,
                stored: Some(payload::PayloadSlice {
                    payload_id: vec![3; 32],
                    offset: 0,
                    nbytes: 32,
                }),
                decoded_nbytes: 16,
                chunk_nbytes: 4 << 20,
                chunk_sha256: vec![vec![0; 32]],
            }],
            modules: vec![payload::Module {
                name: "model".into(),
                cubin: 0,
                globals: vec![payload::ModuleGlobal {
                    symbol: "constant".into(),
                    nbytes: 8,
                    contents: Some(payload::PayloadUse {
                        slice: Some(payload::PayloadSlice {
                            payload_id: vec![4; 32],
                            offset: 8,
                            nbytes: 8,
                        }),
                        role: payload::PayloadRole::InitialState as i32,
                    }),
                    relocations: vec![],
                }],
            }],
            functions: vec![payload::Function {
                module: "model".into(),
                symbol: "kernel".into(),
                parameter_sizes: vec![8, 8],
                attributes: vec![],
            }],
            graphs: vec![payload::Graph {
                name: "decode".into(),
                flags: 0,
                resource: 0,
                node_count: 1,
                edge_count: 0,
                functions: vec![0],
            }],
            events: vec!["ready".into()],
            peers: vec![],
            execution: None,
            roce: None,
            prepared_weights: None,
            compressed_prepared_weights: None,
        }
    }

    #[test]
    fn payload_rank_keeps_every_initializer_source_exact() {
        let placements = Placements::from([
            (vec![1u8; 32], placement("weights-a", 100, 4096)),
            (vec![2u8; 32], placement("weights-b", 200, 8192)),
            (vec![3u8; 32], placement("graphs", 4096, 64)),
            (vec![4u8; 32], placement("globals", 0, 32)),
        ]);
        let core = payload_rank(&rank(), [12, 0], &placements).unwrap();
        core.validate().unwrap();
        let allocation = &core.allocations[0];
        assert!(allocation.is_payload_bound());
        assert!(allocation.file.is_none());
        assert_eq!(allocation.data_spans, vec![[0, 4096], [4096, 4096]]);
        assert_eq!(
            allocation.sources,
            vec![
                Source {
                    resource: "initializer/1".into(),
                    offset: 0,
                    nbytes: 4096,
                },
                Source {
                    resource: "initializer/0".into(),
                    offset: 0,
                    nbytes: 4096,
                },
            ]
        );
        let first = &core.resources["initializer/1"];
        assert_eq!(
            (first.filename.as_str(), first.offset, first.nbytes),
            ("weights-a", 100, 4096)
        );
        let second = &core.resources["initializer/0"];
        // Payload-relative slice offsets add to the container-relative placement.
        assert_eq!(
            (second.filename.as_str(), second.offset, second.nbytes),
            ("weights-b", 216, 4096)
        );
        let executable = &core.resources["executable/9"];
        assert_eq!(
            (executable.filename.as_str(), executable.offset),
            ("graphs", 4096)
        );
        assert_eq!(core.modules[0].file, "executable/9");
        assert_eq!(core.modules[0].globals[0].file, "global/0/0");
        assert_eq!(core.resources["global/0/0"].offset, 8);
        assert_eq!(core.graphs[0].resource, "executable/9");
        assert_eq!(core.graphs[0].functions, vec![0]);
        assert_eq!(core.functions[0].parameter_sizes, vec![8, 8]);
        assert_eq!(core.context.limits.len(), 0);
        assert_eq!(core.events, vec!["ready".to_string()]);
    }

    #[test]
    fn module_global_keys_do_not_depend_on_name_concatenation() {
        let mut body = rank();
        body.modules = vec![
            payload::Module {
                name: "a".into(),
                cubin: 0,
                globals: vec![payload::ModuleGlobal {
                    symbol: "b:c".into(),
                    nbytes: 8,
                    contents: Some(payload::PayloadUse {
                        slice: Some(payload::PayloadSlice {
                            payload_id: vec![4; 32],
                            offset: 0,
                            nbytes: 8,
                        }),
                        role: payload::PayloadRole::InitialState as i32,
                    }),
                    relocations: vec![],
                }],
            },
            payload::Module {
                name: "a:b".into(),
                cubin: 0,
                globals: vec![payload::ModuleGlobal {
                    symbol: "c".into(),
                    nbytes: 8,
                    contents: Some(payload::PayloadUse {
                        slice: Some(payload::PayloadSlice {
                            payload_id: vec![4; 32],
                            offset: 8,
                            nbytes: 8,
                        }),
                        role: payload::PayloadRole::InitialState as i32,
                    }),
                    relocations: vec![],
                }],
            },
        ];
        let placements = Placements::from([
            (vec![1u8; 32], placement("weights-a", 100, 4096)),
            (vec![2u8; 32], placement("weights-b", 200, 8192)),
            (vec![3u8; 32], placement("graphs", 0, 64)),
            (vec![4u8; 32], placement("globals", 0, 32)),
        ]);
        let core = payload_rank(&body, [12, 0], &placements).unwrap();
        assert_eq!(core.modules[0].globals[0].file, "global/0/0");
        assert_eq!(core.modules[1].globals[0].file, "global/1/0");
        assert_eq!(core.resources["global/0/0"].offset, 0);
        assert_eq!(core.resources["global/1/0"].offset, 8);
    }

    #[test]
    fn payload_rank_rejects_slices_without_a_placement() {
        let placements = Placements::from([(vec![1u8; 32], placement("weights-a", 0, 4096))]);
        assert!(payload_rank(&rank(), [12, 0], &placements).is_err());
    }

    #[test]
    fn payload_rank_rejects_a_slice_past_its_placed_payload() {
        let placements = Placements::from([
            (vec![1u8; 32], placement("weights-a", 0, 4096)),
            (vec![2u8; 32], placement("weights-b", 0, 4096)),
            (vec![3u8; 32], placement("graphs", 0, 64)),
            (vec![4u8; 32], placement("globals", 0, 32)),
        ]);
        assert!(payload_rank(&rank(), [12, 0], &placements).is_err());
        let placements = Placements::from([
            (vec![1u8; 32], placement("weights-a", 0, 4096)),
            (vec![2u8; 32], placement("weights-b", 0, 8192)),
            (vec![3u8; 32], placement("graphs", 0, 64)),
            (vec![4u8; 32], placement("globals", 0, 32)),
        ]);
        payload_rank(&rank(), [12, 0], &placements).unwrap();
    }
}
