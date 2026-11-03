/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::{NativeResources, format::invalid, protobuf::size};
use crate::{
    Result,
    execution::{BufferDescriptor, ExecutionManifest, GraphDescriptor, InputBinding, TunedBinding, TuningGraphDescriptor},
};

impl NativeResources {
    pub fn execution_manifest(&self) -> Result<ExecutionManifest> {
        let execution = self
            .execution
            .as_ref()
            .ok_or_else(|| invalid("package has no serving contract"))?;
        let buffers = execution
            .buffers
            .iter()
            .map(|b| {
                let reference = b
                    .reference
                    .as_ref()
                    .ok_or_else(|| invalid("missing execution buffer reference"))?;
                let allocation = self.allocation(&reference.allocation)?;
                let offset = size(reference.offset)?;
                let nbytes = size(b.nbytes)?;
                if offset
                    .checked_add(nbytes)
                    .is_none_or(|end| end > allocation.nbytes)
                {
                    return Err(invalid("execution buffer exceeds its allocation"));
                }
                Ok(BufferDescriptor {
                    name: b.name.clone(),
                    address: allocation.address + offset,
                    nbytes,
                    dtype: b.dtype.as_str().try_into()?,
                    shape: b.shape.iter().copied().map(size).collect::<Result<_>>()?,
                    strides: b.strides.iter().copied().map(size).collect::<Result<_>>()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let graphs = execution
            .graphs
            .iter()
            .map(|g| {
                let index = self
                    .names
                    .iter()
                    .position(|name| *name == g.name)
                    .ok_or_else(|| invalid("execution references an unknown graph"))?;
                Ok(GraphDescriptor {
                    tuned_bindings: g.tuned_bindings.iter().map(|binding| TunedBinding {
                        buffer: binding.buffer.clone(), tuning_graph: binding.tuning_graph.clone(),
                    }).collect(),
                    confidence_width: g.confidence_width as usize,
                    draft_tokens: if g.kind == "speculative" {
                        g.draft_tokens.ok_or_else(|| {
                            invalid("speculative graph lacks proposal depth")
                        })? as usize
                    } else {
                        0
                    },
                    packed_token_capacity: size(g.packed_token_capacity)?,
                    name: g.name.clone(),
                    kind: g.kind.as_str().try_into()?,
                    sampling_mode: g.sampling_mode.as_str().try_into()?,
                    handle: self.executables[index] as usize,
                    batch_capacity: size(g.batch_capacity)?,
                    token_capacity: size(g.token_capacity)?,
                    prefill_capacity: size(g.prefill_capacity)?,
                    estimated_duration_ns: g.estimated_duration_ns,
                    input_bindings: g
                        .input_bindings
                        .iter()
                        .map(|b| {
                            Ok(InputBinding {
                                buffer_name: b.buffer_name.clone(),
                                host_offset: size(b.host_offset)?,
                                nbytes: size(b.nbytes)?,
                            })
                        })
                        .collect::<Result<_>>()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let tuning_graphs = execution.tuning_graphs.iter().map(|graph| {
            let index = self.names.iter().position(|name| *name == graph.name)
                .ok_or_else(|| invalid("tuning metadata references an unknown graph"))?;
            Ok(TuningGraphDescriptor {
                name: graph.name.clone(), handle: self.executables[index] as usize,
                outputs: graph.outputs.clone(), status_buffer: graph.status_buffer.clone(),
            })
        }).collect::<Result<Vec<_>>>()?;
        let manifest = ExecutionManifest {
            tuning_graphs,
            collectives: None,
            engram: None,
            cache: self.cache.clone(),
            offload: None,
            abi_version: execution.abi_version,
            completion_token_capacity: crate::execution::resolve_completion_token_capacity(execution.completion_token_capacity),
            context: self.context as usize,
            device: self.device,
            max_active: size(execution.max_active)?,
            input_frame_bytes: size(execution.input_frame_bytes)?,
            buffers,
            graphs,
        };
        manifest.validate()?;
        Ok(manifest)
    }
}
