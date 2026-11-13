/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Select serving graph variants without changing captured storage layouts.

use std::collections::{BTreeSet, HashSet};

use serde_json::Value;

use super::format::{Core, invalid};
use crate::{Result, execution::BATCH_CAPACITIES};

#[derive(Clone, Debug, Default)]
pub struct LoadOptions {
    pub kv_cache_limit: super::KvCacheLimit,
    pub memory_limit: Option<super::MemoryLimit>,
    /// Auxiliary embedding-table storage; defaults to disk when absent.
    pub embedding_table_storage: Option<crate::engram::Storage>,
    /// Per-host RAM bytes for disk-backed embedding rows; zero disables, absent selects by memory topology.
    pub embedding_table_cache_bytes: Option<u64>,
    /// Keep original scale-plane bytes in host RAM while embedding weights remain on disk.
    /// When absent, enabled for disk tables on discrete GPUs.
    pub embedding_table_resident_scales: Option<bool>,
    /// Disk workers per table: four on Windows, one elsewhere. Share 128 slots and 8 MiB staging.
    pub embedding_table_workers: Option<usize>,
    pub max_concurrency: Option<usize>,
    /// Retain eligible mixed graphs by default; explicit enable requires an eligible graph.
    pub mixed_prefill: Option<bool>,
    pub mtp_draft_tokens: Option<usize>,
    pub history_drafts: Option<bool>,
    /// Select one exact packed verification capacity per batch before proposal sampling.
    pub mtp_verification_token_budget: Option<usize>,
    /// Retain every captured packed verification capacity at or below this total-row ceiling.
    pub mtp_max_verification_token_budget: Option<usize>,
    pub adaptive_verification: Option<bool>,
    /// User-visible ordinals when the launcher renumbers CUDA devices.
    pub device_labels: Option<Vec<i32>>,
    pub prefix_cache: crate::offload::PrefixCacheOffload,
    /// For a core whose ranks run on separate hosts: the one rank this host loads.
    pub cluster_rank: Option<usize>,
}

impl LoadOptions {
    /// Resolve the disk-reader default without treating it as an explicit storage override.
    pub fn embedding_workers(&self) -> usize {
        self.embedding_table_workers
            .unwrap_or(if cfg!(windows) { 4 } else { 1 })
    }

    pub(super) fn embedding_resident_scales(&self, has_tables: bool, integrated: bool) -> bool {
        has_tables
            && self.embedding_table_storage.unwrap_or_default() == crate::engram::Storage::Disk
            && self.embedding_table_resident_scales.unwrap_or(!integrated)
    }

    pub(super) fn embedding_cache_bytes(&self, has_tables: bool, integrated: bool) -> u64 {
        self.embedding_table_cache_bytes.unwrap_or_else(|| {
            if has_tables
                && self.embedding_table_storage.unwrap_or_default() == crate::engram::Storage::Disk
            {
                if integrated { 1 << 30 } else { 4 << 30 }
            } else {
                0
            }
        })
    }

    pub(super) fn select(&self, cores: &mut [Core], config: &mut Value) -> Result<usize> {
        if self.mtp_verification_token_budget.is_some()
            && self.mtp_max_verification_token_budget.is_some()
        {
            return Err(invalid(
                "exact and maximum verification token budgets are mutually exclusive",
            ));
        }
        if let Some(limit) = self.memory_limit {
            limit.validate()?;
        }
        if let Some(workers) = self.embedding_table_workers {
            if !matches!(workers, 1 | 2 | 4 | 8) {
                return Err(invalid("embedding table workers must be 1, 2, 4, or 8"));
            }
            if self.embedding_table_storage == Some(crate::engram::Storage::Ram) {
                return Err(invalid("embedding table workers require disk storage"));
            }
            #[cfg(unix)]
            if workers != 1 {
                return Err(invalid("parallel embedding disk workers require Windows"));
            }
        }
        if self
            .embedding_table_cache_bytes
            .is_some_and(|bytes| bytes != 0)
            && self.embedding_table_storage == Some(crate::engram::Storage::Ram)
        {
            return Err(invalid("embedding row cache requires disk table storage"));
        }
        if self.embedding_table_resident_scales == Some(true)
            && self.embedding_table_storage == Some(crate::engram::Storage::Ram)
        {
            return Err(invalid(
                "resident embedding scales require disk table storage; RAM storage already retains all planes",
            ));
        }
        let execution = cores
            .first()
            .and_then(|p| p.execution.as_ref())
            .ok_or_else(|| invalid("model core has no serving contract"))?;
        let completion_token_capacity = crate::execution::resolve_completion_token_capacity(execution.completion_token_capacity);
        let prepared = usize::try_from(execution.max_active)
            .map_err(|_| invalid("prepared concurrency exceeds usize"))?;
        let active = self.max_concurrency.unwrap_or(prepared);
        if !(1..=64).contains(&prepared) || active == 0 || active > prepared {
            return Err(invalid(format!(
                "runtime concurrency must be 1 through the captured maximum {prepared}"
            )));
        }
        let widths = config
            .get("speculative_widths")
            .map(|value| {
                value
                    .as_array()
                    .ok_or_else(|| invalid("speculative_widths must be an array"))?
                    .iter()
                    .map(|value| {
                        value
                            .as_u64()
                            .and_then(|n| usize::try_from(n).ok())
                            .ok_or_else(|| invalid("invalid speculative width"))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();
        crate::scheduler::speculation::validate_widths(&widths)?;
        let token_capacity = config
            .get("speculative_token_capacity")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if !widths.is_empty() && token_capacity == 0 {
            return Err(invalid("speculative token capacity must be positive"));
        }
        let maximum = execution.graphs.iter().filter(|g| g.kind == "speculative")
            .filter_map(|g| g.draft_tokens).max().unwrap_or(0) as usize;
        let drafts = self.mtp_draft_tokens.unwrap_or(maximum);
        if drafts > maximum {
            return Err(invalid(format!("runtime MTP draft tokens must be 0 through the captured maximum {maximum}")));
        }
        let captured_fixed = config.get("speculative_fixed_width")
            .filter(|value| !value.is_null())
            .map(|value| value.as_u64().and_then(|width| usize::try_from(width).ok())
                .ok_or_else(|| invalid("speculative_fixed_width must be an integer or null")))
            .transpose()?;
        if captured_fixed.is_some_and(|width| !widths.contains(&width)) {
            return Err(invalid("fixed verification width is outside the captured widths"));
        }
        if widths.is_empty() && self.adaptive_verification.is_some() {
            return Err(invalid("verification policy requires a core captured with speculation"));
        }
        let source = match config.get("speculative_proposal_source").and_then(Value::as_str).unwrap_or("mtp") {
            "mtp" => 1,
            "copy" => 2,
            _ => return Err(invalid("unknown speculative proposal source")),
        };
        let adaptive = self.adaptive_verification.unwrap_or(captured_fixed.is_none());
        let eligible_widths: BTreeSet<_> = execution.graphs.iter().filter(|graph|
            graph.kind == "speculative" && graph.batch_capacity <= active.next_power_of_two() as u64
                && graph.draft_source == source && graph.draft_tokens.is_some_and(|depth| depth <= drafts as u32)
                && widths.contains(&(graph.token_capacity as usize))
                && (graph.token_capacity == 1 || graph.packed_token_capacity <= token_capacity))
            .map(|graph| graph.token_capacity as usize).collect();
        let fixed = if adaptive || drafts == 0 || widths.is_empty() { None }
            else if let Some(width) = captured_fixed { Some(width) }
            else { Some(eligible_widths.iter().copied().filter(|width| *width > 1).max()
                .ok_or_else(|| invalid("no captured speculative width fits the requested draft depth"))?) };
        let enabled = drafts > 0 && fixed != Some(1) && !widths.is_empty();
        let fixed = enabled.then_some(fixed).flatten();
        if self.history_drafts == Some(true) && (!enabled || source != 1) {
            return Err(invalid("--history-drafts requires enabled learned speculation"));
        }
        if enabled && (eligible_widths.iter().all(|width| *width == 1)
            || fixed.is_some_and(|width| !eligible_widths.contains(&width))) {
            return Err(invalid("no captured speculative width fits the requested draft depth and fixed width"));
        }
        let verification_token_limit = self.mtp_verification_token_budget.or(self.mtp_max_verification_token_budget);
        if verification_token_limit.is_some_and(|budget| budget == 0 || !enabled || !adaptive) {
            return Err(invalid("verification token budget requires enabled adaptive speculation and a positive capacity"));
        }
        let selected_token_capacity = verification_token_limit.map_or(token_capacity, |budget| token_capacity.min(budget as u64));
        let widths: Vec<_> = widths.into_iter().filter(|_| enabled).collect();
        let mut rank_catalog = None;
        let mut rank_tuning_catalog = None;
        for (rank, core) in cores.iter_mut().enumerate() {
            let execution = core.execution.as_mut().ok_or_else(|| invalid("rank lacks serving contract"))?;
            if crate::execution::resolve_completion_token_capacity(execution.completion_token_capacity) != completion_token_capacity {
                return Err(invalid("rank completion token capacities differ"));
            }
            if execution.max_active != prepared as u64 {
                return Err(invalid("rank concurrency capacities differ"));
            }
            let all_serving_names: HashSet<_> = execution.graphs.iter().map(|graph| graph.name.clone()).collect();
            let mut names = HashSet::new();
            let mut variants = HashSet::new();
            for graph in &execution.graphs {
                use crate::execution::GraphKind;
                let kind = GraphKind::try_from(graph.kind.as_str())?;
                crate::execution::SamplingMode::try_from(graph.sampling_mode.as_str())?;
                let speculative = kind == GraphKind::Speculative || (kind == GraphKind::Mixed && graph.speculation_enabled == Some(true));
                let valid = match kind {
                    GraphKind::Mixed => graph.batch_capacity >= 2
                        && BATCH_CAPACITIES.contains(&(graph.batch_capacity as usize))
                        && graph.prefill_capacity >= 64 && graph.prefill_capacity.is_multiple_of(64)
                        && graph.prefill_capacity <= config["prefill_chunk_size"].as_u64().unwrap_or(0)
                        && graph.speculation_enabled.is_some()
                        && if speculative { (1..=crate::execution::resolve_completion_token_capacity(execution.completion_token_capacity) as u64).contains(&graph.token_capacity) } else { graph.token_capacity == 1 },
                    GraphKind::Prefill | GraphKind::ScoredPrefill | GraphKind::Vision => graph.batch_capacity == 1 && graph.token_capacity > 0,
                    GraphKind::Speculative => BATCH_CAPACITIES.contains(&(graph.batch_capacity as usize)) && (1..=crate::execution::resolve_completion_token_capacity(execution.completion_token_capacity) as u64).contains(&graph.token_capacity),
                    _ => BATCH_CAPACITIES.contains(&(graph.batch_capacity as usize)) && graph.token_capacity == 1,
                };
                if !valid || graph.batch_capacity > prepared.next_power_of_two() as u64
                    || (kind != GraphKind::Mixed && graph.prefill_capacity != 0)
                    || (kind == GraphKind::Speculative && (graph.packed_token_capacity < graph.batch_capacity
                        || graph.packed_token_capacity > graph.batch_capacity.saturating_mul(graph.token_capacity)))
                    || (kind != GraphKind::Speculative && graph.packed_token_capacity != 0)
                    || (graph.confidence_width > 0 && (kind != GraphKind::Speculative
                        || graph.draft_tokens != Some(graph.confidence_width)))
                    || (kind == GraphKind::Speculative && graph.speculation_enabled == Some(false))
                    || (kind == GraphKind::Decode && graph.speculation_enabled == Some(true))
                    || if speculative {
                        graph.verify_tokens != Some(graph.token_capacity as u32)
                            || graph.draft_tokens.is_none_or(|n| n as usize >= crate::execution::resolve_completion_token_capacity(execution.completion_token_capacity))
                            || (kind == GraphKind::Speculative && graph.token_capacity == 1 && graph.draft_tokens != Some(0))
                            || ![1, 2].contains(&graph.draft_source)
                    } else {
                        graph.draft_tokens.is_some() || graph.verify_tokens.is_some() || graph.draft_source != 0
                    }
                    || (kind == GraphKind::Speculative && graph.token_capacity > 1 && graph.packed_token_capacity > token_capacity)
                    || !names.insert(graph.name.clone())
                    || !variants.insert((graph.kind.clone(), graph.sampling_mode.clone(), graph.batch_capacity,
                        graph.token_capacity, graph.prefill_capacity, graph.packed_token_capacity, graph.draft_tokens, graph.draft_source, graph.speculation_enabled)) {
                    return Err(invalid("invalid or duplicate serving graph variant"));
                }
            }
            for graph in &execution.tuning_graphs {
                if graph.name.is_empty() || !names.insert(graph.name.clone()) {
                    return Err(invalid("duplicate or unnamed startup tuning graph"));
                }
            }
            let resources: HashSet<_> = core.graphs.iter().map(|g| g.name.clone()).collect();
            if names != resources || resources.len() != core.graphs.len() {
                return Err(invalid("execution catalog must describe every captured graph exactly once"));
            }
            execution.graphs.retain(|graph| {
                graph.batch_capacity <= active.next_power_of_two() as u64
                    && (graph.kind != "mixed" || self.mixed_prefill != Some(false))
                    && graph.speculation_enabled.is_none_or(|value| value == enabled)
                    && (graph.kind != "decode" || !enabled)
                    && (!(graph.kind == "speculative" || graph.kind == "mixed" && enabled)
                        || enabled && graph.draft_source == source
                            && graph.draft_tokens.is_some_and(|depth| depth <= drafts as u32)
                            && widths.contains(&(graph.token_capacity as usize))
                            && (graph.token_capacity == 1 || fixed.is_none_or(|width| graph.token_capacity == width as u64)))
                    && (graph.kind != "speculative" || graph.token_capacity == 1
                        || graph.packed_token_capacity <= selected_token_capacity
                            && self.mtp_verification_token_budget.is_none_or(|budget|
                                graph.packed_token_capacity == (budget as u64).min(graph.batch_capacity * graph.token_capacity)))
            });
            let mut depths = std::collections::HashMap::new();
            for graph in &execution.graphs {
                if let Some(depth) = graph.draft_tokens {
                    let key = (graph.kind.clone(), graph.batch_capacity, graph.token_capacity,
                        graph.packed_token_capacity, graph.prefill_capacity, graph.sampling_mode.clone());
                    let selected = depths.entry(key).or_insert(depth);
                    *selected = (*selected).max(depth);
                }
            }
            execution.graphs.retain(|graph| graph.draft_tokens.is_none_or(|depth|
                depths[&(graph.kind.clone(), graph.batch_capacity, graph.token_capacity,
                    graph.packed_token_capacity, graph.prefill_capacity, graph.sampling_mode.clone())] == depth));
            if self.mixed_prefill == Some(true)
                && !execution.graphs.iter().any(|graph| graph.kind == "mixed")
            {
                return Err(invalid(format!(
                    "rank {rank} has no mixed prefill graph within selected generation geometry"
                )));
            }
            for batch in BATCH_CAPACITIES.into_iter().filter(|batch| *batch <= active.next_power_of_two()) {
                for kind in ["init", "restore", "snapshot", "resume", if enabled { "speculative" } else { "decode" }] {
                    if !execution.graphs.iter().any(|g| g.kind == kind && g.batch_capacity == batch as u64
                        && (kind != "speculative" || g.token_capacity == 1)) {
                        return Err(invalid(format!("rank {rank} lacks {kind} graph for batch {batch} within selected generation geometry")));
                    }
                }
                if enabled && batch as u64 * 2 <= selected_token_capacity
                    && !execution.graphs.iter().any(|g| g.kind == "speculative" && g.batch_capacity == batch as u64 && g.token_capacity > 1) {
                    return Err(invalid(format!("rank {rank} lacks speculative graph for batch {batch} within selected generation geometry")));
                }
            }
            let mut tuning_names = HashSet::new();
            let mut outputs = HashSet::new();
            for tuner in &execution.tuning_graphs {
                if tuner.name.is_empty() || all_serving_names.contains(&tuner.name)
                    || !tuning_names.insert(tuner.name.as_str()) || tuner.outputs.is_empty()
                    || tuner.status_buffer.is_empty()
                    || !core.graphs.iter().any(|graph| graph.name == tuner.name) {
                    return Err(invalid("invalid startup tuning graph catalog"));
                }
                for output in tuner.outputs.iter().chain(std::iter::once(&tuner.status_buffer)) {
                    if !outputs.insert(output.as_str())
                        || !execution.buffers.iter().any(|buffer| buffer.name == *output) {
                        return Err(invalid("startup tuning storage requires one registered producer"));
                    }
                }
            }
            let mut referenced = HashSet::new();
            for graph in &execution.graphs {
                for binding in &graph.tuned_bindings {
                    if !execution.tuning_graphs.iter().any(|tuner|
                        tuner.name == binding.tuning_graph && tuner.outputs.contains(&binding.buffer)) {
                        return Err(invalid("tuned binding has no declared startup producer"));
                    }
                    referenced.insert(binding.tuning_graph.clone());
                }
            }
            execution.tuning_graphs.retain(|graph| referenced.contains(&graph.name));
            let tuning_catalog: Vec<_> = execution.tuning_graphs.iter().map(|graph| {
                let storage: Vec<_> = graph.outputs.iter().chain(std::iter::once(&graph.status_buffer))
                    .map(|name| {
                        let buffer = execution.buffers.iter().find(|buffer| buffer.name == *name).unwrap();
                        (name.clone(), buffer.dtype.clone(), buffer.nbytes, buffer.shape.clone(), buffer.strides.clone())
                    }).collect();
                (graph.name.clone(), graph.outputs.clone(), graph.status_buffer.clone(), storage)
            }).collect();
            if rank_tuning_catalog.as_ref().is_some_and(|expected| *expected != tuning_catalog) {
                return Err(invalid("selected startup tuning catalogs differ between ranks"));
            }
            rank_tuning_catalog = Some(tuning_catalog);
            let catalog: BTreeSet<_> = execution.graphs.iter().map(|g|
                (g.name.clone(), g.kind.clone(), g.sampling_mode.clone(), g.batch_capacity,
                    g.token_capacity, g.prefill_capacity, g.packed_token_capacity, g.draft_tokens,
                    g.verify_tokens, g.draft_source, g.confidence_width,
                    g.tuned_bindings.iter().map(|binding| (binding.buffer.clone(), binding.tuning_graph.clone())).collect::<Vec<_>>())).collect();
            if rank_catalog.as_ref().is_some_and(|expected| *expected != catalog) {
                return Err(invalid("selected graph catalogs differ between ranks"));
            }
            rank_catalog = Some(catalog);
            let selected: HashSet<_> = execution.graphs.iter().map(|g| g.name.as_str())
                .chain(execution.tuning_graphs.iter().map(|g| g.name.as_str())).collect();
            core.graphs.retain(|g| selected.contains(g.name.as_str()));
            prune_kernels(core)?;
        }
        let object = config
            .as_object_mut()
            .ok_or_else(|| invalid("server configuration must be an object"))?;
        object.insert("max_active".into(), active.into());
        object.insert("speculative_widths".into(), serde_json::json!(widths));
        object.insert("speculative_fixed_width".into(), serde_json::json!(fixed));
        if verification_token_limit.is_some() {
            object.insert(
                "speculative_token_capacity".into(),
                selected_token_capacity.into(),
            );
        }
        if let Some(history) = self.history_drafts {
            object.insert("history_drafts".into(), history.into());
        } else if !enabled {
            object.insert("history_drafts".into(), false.into());
        }
        Ok(active)
    }
}

fn prune_kernels(core: &mut Core) -> Result<()> {
    let used: HashSet<_> = core
        .graphs
        .iter()
        .flat_map(|g| g.functions.iter().copied())
        .collect();
    core.functions.retain(|f| used.contains(&f.id));
    let modules: HashSet<_> = core
        .functions
        .iter()
        .map(|f| f.module.as_str())
        .collect();
    // Module globals can own persistent data referenced through opaque kernel arguments.
    core
        .modules
        .retain(|module| modules.contains(module.name.as_str()) || !module.globals.is_empty());
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use super::*;
    use crate::runtime_core::{format::*, protobuf::wire};
    use prost::Message;

    fn core() -> Core {
        let mut execution = wire::Execution {
            abi_version: crate::execution::ABI_VERSION,
            max_active: 4,
            input_frame_bytes: 8192,
            ..Default::default()
        };
        for batch in [1, 2, 4] {
            for enabled in [false, true] {
                for kind in ["init", "restore", "snapshot", "resume", "decode"] {
                    if enabled && kind == "decode" {
                        continue;
                    }
                    execution.graphs.push(wire::ExecutionGraph {
                        name: format!("{kind}{batch}-{}", if enabled { "mtp" } else { "no-mtp" }),
                        kind: kind.try_into().unwrap(),
                        sampling_mode: "general".into(),
                        estimated_duration_ns: 1000,
                        batch_capacity: batch,
                        token_capacity: 1,
                        speculation_enabled: Some(enabled),
                        ..Default::default()
                    });
                }
            }
            for width in 1..=6 {
                execution.graphs.push(wire::ExecutionGraph {
                    name: format!("speculative{batch}x{width}"),
                    kind: "speculative".into(),
                    sampling_mode: "general".into(),
                    estimated_duration_ns: 1000,
                    batch_capacity: batch,
                    token_capacity: width,
                    packed_token_capacity: batch * width,
                    draft_tokens: Some(width as u32 - 1),
                    verify_tokens: Some(width as u32),
                    draft_source: wire::DraftSource::Mtp as i32,
                    ..Default::default()
                });
            }
        }
        let graphs = execution
            .graphs
            .iter()
            .enumerate()
            .map(|(function, g)| Graph {
                name: g.name.clone(),
                flags: 0,
                resource: g.name.clone(),
                node_count: 1,
                edge_count: 0,
                functions: vec![function],
            })
            .collect();
        let functions = execution
            .graphs
            .iter()
            .enumerate()
            .map(|(id, g)| Function {
                id,
                module: g.name.clone(),
                symbol: "kernel".into(),
                parameter_sizes: vec![],
                attributes: vec![],
            })
            .collect();
        let modules = execution
            .graphs
            .iter()
            .map(|g| Module {
                name: g.name.clone(),
                file: format!("{}.cubin", g.name),
                globals: vec![],
            })
            .collect();
        Core {
            resources: std::collections::HashMap::new(),
            events: vec![],
            architecture: [12, 0],
            graphs,
            functions,
            modules,
            execution: Some(execution),
            context: Context::default(),
            reservations: vec![],
            allocations: ["weights", "mtp-weights", "scratch"]
                .map(|name| Allocation {
                    name: name.into(),
                    nbytes: 8192,
                    alignment: 256,
                    file: None,
                    data_spans: vec![],
                    sources: vec![],
                    relocations: vec![],
                    address: None,
                    memory: Memory::Device,
                    sizing: None,
                })
                .into(),
        }
    }

    fn tuning_core() -> Core {
        let mut core = core();
        let function = core.graphs.last().unwrap().functions[0];
        for name in ["tune", "unused_tune"] {
            core.graphs.push(Graph {
                name: name.into(), flags: 0, resource: name.into(), node_count: 1, edge_count: 0,
                functions: vec![function],
            });
        }
        let execution = core.execution.as_mut().unwrap();
        for name in ["winner", "status", "unused_winner", "unused_status"] {
            execution.buffers.push(wire::Buffer {
                name: name.into(), nbytes: 4, dtype: "int32".into(), shape: vec![1], strides: vec![1],
                ..Default::default()
            });
        }
        execution.tuning_graphs = vec![
            wire::TuningGraph { name: "tune".into(), outputs: vec!["winner".into()], status_buffer: "status".into() },
            wire::TuningGraph { name: "unused_tune".into(), outputs: vec!["unused_winner".into()], status_buffer: "unused_status".into() },
        ];
        for graph in &mut execution.graphs {
            if graph.name == "decode1-no-mtp" {
                graph.tuned_bindings.push(wire::TunedBinding { buffer: "winner".into(), tuning_graph: "tune".into() });
            }
        }
        core
    }

    #[test]
    fn selected_graphs_retain_startup_producers_and_their_kernels() {
        let core = tuning_core();
        let function = core.graphs.iter().find(|graph| graph.name == "tune").unwrap().functions[0];
        let mut cores = [core];
        LoadOptions { max_concurrency: Some(1), mtp_draft_tokens: Some(0), ..Default::default() }
            .select(&mut cores, &mut config()).unwrap();
        let core = &cores[0];
        assert!(core.graphs.iter().any(|graph| graph.name == "tune"));
        assert!(!core.graphs.iter().any(|graph| graph.name == "unused_tune"));
        assert!(core.functions.iter().any(|item| item.id == function));
        assert_eq!(core.execution.as_ref().unwrap().tuning_graphs.len(), 1);
    }

    #[test]
    fn startup_producer_selection_rejects_dangling_and_rank_mismatched_schemas() {
        for invalid in 0..4 {
            let mut cores = [tuning_core(), tuning_core()];
            let execution = cores[1].execution.as_mut().unwrap();
            match invalid {
                0 => execution.tuning_graphs[0].outputs[0] = "missing".into(),
                1 => execution.buffers.iter_mut().find(|buffer| buffer.name == "winner").unwrap().dtype = "float32".into(),
                2 => execution.tuning_graphs[0].name = "decode1-no-mtp".into(),
                _ => execution.tuning_graphs[1].outputs[0] = "winner".into(),
            }
            assert!(LoadOptions { max_concurrency: Some(1), mtp_draft_tokens: Some(0), ..Default::default() }
                .select(&mut cores, &mut config()).is_err());
        }
    }

    fn config() -> Value {
        serde_json::json!({"speculative_token_capacity": 24, "speculative_widths": [1,2,3,4,5,6], "speculative_fixed_width": null})
    }

    #[test]
    fn mtp_only_catalog_loads_with_drafts_and_rejects_ordinary_execution() {
        for ranks in [1, 2, 16] {
            let catalog = || (0..ranks).map(|_| {
                let mut core = core();
                let execution = core.execution.as_mut().unwrap();
                execution.graphs.retain(|graph| graph.speculation_enabled != Some(false));
                for graph in &mut execution.graphs {
                    graph.speculation_enabled = Some(true);
                }
                let names: HashSet<_> = execution.graphs.iter().map(|graph| graph.name.clone()).collect();
                core.graphs.retain(|graph| names.contains(&graph.name));
                core
            }).collect::<Vec<_>>();
            let mut cores = catalog();
            LoadOptions::default().select(&mut cores, &mut config()).unwrap();
            for core in &cores {
                let execution = core.execution.as_ref().unwrap();
                assert!(execution.graphs.iter().all(|graph| graph.speculation_enabled == Some(true)));
                for batch in [1, 2, 4] {
                    assert!(execution.graphs.iter().any(|graph|
                        graph.kind == "speculative" && graph.batch_capacity == batch && graph.token_capacity == 1));
                }
            }
            let error = LoadOptions { mtp_draft_tokens: Some(0), ..Default::default() }
                .select(&mut catalog(), &mut config()).unwrap_err();
            assert!(error.to_string().contains("rank 0 lacks init graph"), "{error}");
        }
    }

    #[test]
    fn rebuild_speculation_preserves_mtp_only_and_defaults_its_absence_to_false() {
        for mtp_only in [false, true] {
            let metadata = wire::RebuildSpeculation { mtp_only, ..Default::default() };
            let decoded = wire::RebuildSpeculation::decode(metadata.encode_to_vec().as_slice()).unwrap();
            assert_eq!(decoded.mtp_only, mtp_only);
        }
        let mut metadata = serde_json::to_value(wire::RebuildSpeculation::default()).unwrap();
        metadata.as_object_mut().unwrap().remove("mtp_only");
        assert!(!serde_json::from_value::<wire::RebuildSpeculation>(metadata).unwrap().mtp_only);
    }

    #[test]
    fn history_override_preserves_the_selected_graphs() {
        let mut baseline = [core()];
        LoadOptions::default()
            .select(&mut baseline, &mut config())
            .unwrap();
        for stored in [None, Some(false), Some(true)] {
            for selected in [None, Some(false), Some(true)] {
                let mut configuration = config();
                if let Some(stored) = stored {
                    let bytes = [0x68, u8::from(stored)];
                    let mut contract = wire::ServingContract::decode(bytes.as_slice()).unwrap();
                    contract.model_contract = Some(wire::ModelContract { version: 3, ..Default::default() });
                    let decoded = super::super::protobuf::serving_contract(contract, 4).unwrap();
                    configuration["history_drafts"] = decoded["history_drafts"].clone();
                }
                let mut cores = [core()];
                LoadOptions {
                    history_drafts: selected,
                    ..Default::default()
                }
                .select(&mut cores, &mut configuration)
                .unwrap();
                assert_eq!(
                    configuration.get("history_drafts").and_then(Value::as_bool),
                    selected.or(stored)
                );
                assert_eq!(
                    cores[0].execution.as_ref().unwrap().encode_to_vec(),
                    baseline[0].execution.as_ref().unwrap().encode_to_vec()
                );
            }
        }
    }

    #[test]
    fn history_requires_enabled_learned_proposals() {
        let mut configuration = config();
        let error = LoadOptions {
            history_drafts: Some(true),
            mtp_draft_tokens: Some(0),
            ..Default::default()
        }
        .select(&mut [core()], &mut configuration)
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("requires enabled learned speculation")
        );
        configuration["speculative_proposal_source"] = "copy".into();
        let error = LoadOptions {
            history_drafts: Some(true),
            ..Default::default()
        }
        .select(&mut [core()], &mut configuration)
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("requires enabled learned speculation")
        );
    }

    #[test]
    fn embedding_worker_options_reject_invalid_counts_and_resident_storage() {
        assert_eq!(
            LoadOptions::default().embedding_workers(),
            if cfg!(windows) { 4 } else { 1 }
        );
        let resident = LoadOptions {
            embedding_table_storage: Some(crate::engram::Storage::Ram),
            ..Default::default()
        };
        resident.select(&mut [core()], &mut config()).unwrap();
        let single = LoadOptions {
            embedding_table_workers: Some(1),
            ..Default::default()
        };
        assert_eq!(single.embedding_workers(), 1);
        single.select(&mut [core()], &mut config()).unwrap();
        for workers in [0, 3, 5] {
            let options = LoadOptions {
                embedding_table_workers: Some(workers),
                ..Default::default()
            };
            assert!(
                options
                    .select(&mut [core()], &mut config())
                    .unwrap_err()
                    .to_string()
                    .contains("workers must be")
            );
        }
        let options = LoadOptions {
            embedding_table_workers: Some(2),
            embedding_table_storage: Some(crate::engram::Storage::Ram),
            ..Default::default()
        };
        assert!(
            options
                .select(&mut [core()], &mut config())
                .unwrap_err()
                .to_string()
                .contains("require disk storage")
        );
    }

    #[test]
    fn indeterminate_graph_durations_are_loadable() {
        let mut core = core();
        core.execution.as_mut().unwrap().graphs[0].estimated_duration_ns = 0;
        LoadOptions::default()
            .select(&mut [core], &mut config())
            .unwrap();
    }

    #[test]
    fn embedding_cache_defaults_follow_storage_and_memory_topology() {
        use crate::engram::Storage;
        for storage in [None, Some(Storage::Disk)] {
            let mut options = LoadOptions {
                embedding_table_storage: storage,
                ..Default::default()
            };
            options.select(&mut [core()], &mut config()).unwrap();
            assert_eq!(options.embedding_cache_bytes(true, false), 4 << 30);
            assert_eq!(options.embedding_cache_bytes(true, true), 1 << 30);
            assert_eq!(options.embedding_cache_bytes(false, false), 0);
            assert_eq!(options.embedding_cache_bytes(false, true), 0);
            for bytes in [0, 2 << 30, 8 << 30] {
                options.embedding_table_cache_bytes = Some(bytes);
                options.select(&mut [core()], &mut config()).unwrap();
                assert_eq!(options.embedding_cache_bytes(true, false), bytes);
                assert_eq!(options.embedding_cache_bytes(true, true), bytes);
            }
        }
        for bytes in [None, Some(0)] {
            let options = LoadOptions {
                embedding_table_storage: Some(Storage::Ram),
                embedding_table_cache_bytes: bytes,
                ..Default::default()
            };
            options.select(&mut [core()], &mut config()).unwrap();
            assert_eq!(options.embedding_cache_bytes(true, false), 0);
            assert_eq!(options.embedding_cache_bytes(true, true), 0);
        }
    }

    #[test]
    fn resident_scale_defaults_follow_storage_and_memory_topology() {
        use crate::engram::Storage;
        for storage in [None, Some(Storage::Disk), Some(Storage::Ram)] {
            for override_value in [None, Some(false), Some(true)] {
                let options = LoadOptions {
                    embedding_table_storage: storage,
                    embedding_table_resident_scales: override_value,
                    ..Default::default()
                };
                for integrated in [false, true] {
                    assert!(!options.embedding_resident_scales(false, integrated));
                    assert_eq!(
                        options.embedding_resident_scales(true, integrated),
                        storage != Some(Storage::Ram) && override_value.unwrap_or(!integrated)
                    );
                }
            }
        }
    }

    #[test]
    fn cached_disk_rows_reject_resident_table_storage_before_cuda() {
        let options = LoadOptions {
            embedding_table_storage: Some(crate::engram::Storage::Ram),
            embedding_table_cache_bytes: Some(1 << 30),
            ..Default::default()
        };
        assert!(
            options
                .select(&mut [core()], &mut config())
                .unwrap_err()
                .to_string()
                .contains("embedding row cache requires disk table storage")
        );
    }

    #[test]
    fn resident_scales_allow_cached_disk_rows_and_reject_whole_table_ram() {
        let mut options = LoadOptions {
            embedding_table_resident_scales: Some(true),
            embedding_table_cache_bytes: Some(1 << 30),
            ..Default::default()
        };
        options.select(&mut [core()], &mut config()).unwrap();
        options.embedding_table_cache_bytes = Some(0);
        options.embedding_table_storage = Some(crate::engram::Storage::Ram);
        assert!(
            options
                .select(&mut [core()], &mut config())
                .unwrap_err()
                .to_string()
                .contains("resident embedding scales require disk table storage")
        );
    }

    #[test]
    fn offload_identity_separates_selected_state_without_fragmenting_compatible_limits() {
        let mut identities = HashMap::new();
        for source in ["mtp", "copy"] {
            for concurrency in 1..=4 {
                for drafts in 0..=5 {
                    for adaptive in [false, true] {
                        let mut core = core();
                        for graph in &mut core.execution.as_mut().unwrap().graphs {
                            if graph.kind == "speculative" {
                                graph.draft_source = if source == "mtp" { 1 } else { 2 };
                            }
                        }
                        let mut config = config();
                        config["identity"] = "01".repeat(32).into();
                        config["speculative_proposal_source"] = source.into();
                        LoadOptions {
                            max_concurrency: Some(concurrency),
                            mtp_draft_tokens: Some(drafts),
                            adaptive_verification: Some(adaptive),
                            ..Default::default()
                        }
                        .select(std::slice::from_mut(&mut core), &mut config)
                        .unwrap();
                        let identity = super::super::model::offload_identity(&config).unwrap();
                        assert_ne!(identity, [1; 32]);
                        let state = if drafts == 0 { "ordinary" } else { source };
                        assert_eq!(*identities.entry(state).or_insert(identity), identity);
                    }
                }
            }
        }
        assert_eq!(identities.values().collect::<HashSet<_>>().len(), 3);

        let mut config = config();
        config["identity"] = "01".repeat(32).into();
        config["speculative_fixed_width"] = 1.into();
        LoadOptions::default()
            .select(&mut [core()], &mut config)
            .unwrap();
        assert_eq!(
            super::super::model::offload_identity(&config).unwrap(),
            identities["ordinary"]
        );
        config["identity"] = "02".repeat(32).into();
        assert_ne!(
            super::super::model::offload_identity(&config).unwrap(),
            identities["ordinary"]
        );
    }

    #[test]
    fn offload_identity_respects_default_proposals_and_overridden_fixed_policy() {
        let mut adaptive_config = config();
        adaptive_config["identity"] = "ab".repeat(32).into();
        LoadOptions::default()
            .select(&mut [core()], &mut adaptive_config)
            .unwrap();
        let speculative = super::super::model::offload_identity(&adaptive_config).unwrap();
        adaptive_config["speculative_proposal_source"] = "mtp".into();
        adaptive_config["identity"] = "AB".repeat(32).into();
        assert_eq!(
            super::super::model::offload_identity(&adaptive_config).unwrap(),
            speculative
        );
        for adaptive in [None, Some(false), Some(true)] {
            let mut config = config();
            config["identity"] = "ab".repeat(32).into();
            config["speculative_fixed_width"] = 1.into();
            LoadOptions {
                adaptive_verification: adaptive,
                ..Default::default()
            }
            .select(&mut [core()], &mut config)
            .unwrap();
            assert_eq!(
                super::super::model::offload_identity(&config).unwrap() == speculative,
                adaptive == Some(true)
            );
        }
    }

    #[test]
    fn draft_count_and_verification_width_are_independent_selection_axes() {
        for limit in [3, 5] {
            let mut p = core();
            for drafts in [3, 5] {
                let name = format!("speculative2-mtp{drafts}-verify2");
                p.execution
                    .as_mut()
                    .unwrap()
                    .graphs
                    .push(wire::ExecutionGraph {
                        name: name.clone(),
                        kind: "speculative".into(),
                        sampling_mode: "general".into(),
                        estimated_duration_ns: 1000,
                        batch_capacity: 2,
                        token_capacity: 2,
                        packed_token_capacity: 4,
                        draft_tokens: Some(drafts),
                        verify_tokens: Some(2),
                        draft_source: 1,
                        ..Default::default()
                    });
                p.graphs.push(Graph {
                    name,
                    resource: "graph".into(),
                    node_count: 1,
                    edge_count: 0,
                    flags: 0,
                    functions: vec![0],
                });
            }
            LoadOptions {
                max_concurrency: Some(2),
                mtp_draft_tokens: Some(limit),
                adaptive_verification: Some(true),
                ..Default::default()
            }
            .select(std::slice::from_mut(&mut p), &mut config())
            .unwrap();
            let variants: Vec<_> = p
                .execution
                .unwrap()
                .graphs
                .into_iter()
                .filter(|g| {
                    g.kind == "speculative" && g.batch_capacity == 2 && g.verify_tokens == Some(2)
                })
                .collect();
            assert_eq!(variants.len(), 1);
            assert_eq!(variants[0].draft_tokens, Some(limit as u32));
        }
    }

    #[test]
    fn draft_limit_comes_from_draft_declarations_not_verification_capacity() {
        let mut p = core();
        p.execution
            .as_mut()
            .unwrap()
            .graphs
            .retain(|g| g.token_capacity <= 2);
        for graph in &mut p.execution.as_mut().unwrap().graphs {
            if graph.kind == "speculative" && graph.token_capacity == 2 {
                graph.draft_tokens = Some(5);
            }
        }
        let names: HashSet<_> = p
            .execution
            .as_ref()
            .unwrap()
            .graphs
            .iter()
            .map(|g| g.name.clone())
            .collect();
        p.graphs.retain(|g| names.contains(&g.name));
        let mut config =
            serde_json::json!({"speculative_token_capacity": 24, "speculative_widths": [1,2]});
        LoadOptions {
            mtp_draft_tokens: Some(5),
            adaptive_verification: Some(false),
            ..Default::default()
        }
        .select(std::slice::from_mut(&mut p), &mut config)
        .unwrap();
        assert_eq!(config["speculative_fixed_width"], 2);
        assert!(
            p.execution
                .unwrap()
                .graphs
                .iter()
                .filter(|g| g.kind == "speculative")
                .all(
                    |g| (g.draft_tokens == Some(0) && g.verify_tokens == Some(1))
                        || (g.draft_tokens == Some(5) && g.verify_tokens == Some(2))
                )
        );
    }

    #[test]
    fn every_concurrency_and_width_reduction_selects_only_required_graphs() {
        for concurrency in 1usize..=4 {
            for drafts in 0..=5 {
                for adaptive in [false, true] {
                    let mut cores = vec![core(), core()];
                    let mut config = config();
                    let options = LoadOptions {
                        max_concurrency: Some(concurrency),
                        mtp_draft_tokens: Some(drafts),
                        adaptive_verification: Some(adaptive),
                        ..Default::default()
                    };
                    options.select(&mut cores, &mut config).unwrap();
                    let batches = concurrency.next_power_of_two().ilog2() as usize + 1;
                    let expected = batches
                        * (4 + if adaptive {
                            drafts + 1
                        } else if drafts == 0 {
                            1
                        } else {
                            2
                        });
                    for core in &cores {
                        let execution = core.execution.as_ref().unwrap();
                        assert_eq!(execution.max_active, 4);
                        assert_eq!(execution.input_frame_bytes, 8192);
                        assert_eq!(core.graphs.len(), expected);
                        assert_eq!(core.functions.len(), expected);
                        assert_eq!(core.modules.len(), expected);
                        assert_eq!(core.allocations.len(), 3);
                        assert_eq!(core.allocations[1].name, "mtp-weights");
                        for graph in &core.graphs {
                            assert_eq!(
                                core
                                    .functions
                                    .iter()
                                    .find(|f| f.id == graph.functions[0])
                                    .unwrap()
                                    .module,
                                graph.name
                            );
                        }
                    }
                    assert_eq!(config["max_active"], concurrency);
                    assert_eq!(
                        config["speculative_widths"].as_array().unwrap().len(),
                        if drafts == 0 { 0 } else { 6 }
                    );
                }
            }
        }
    }

    #[test]
    fn invalid_limits_and_missing_graphs_fail_before_resource_loading() {
        for options in [
            LoadOptions {
                max_concurrency: Some(0),
                ..Default::default()
            },
            LoadOptions {
                max_concurrency: Some(5),
                ..Default::default()
            },
            LoadOptions {
                mtp_draft_tokens: Some(6),
                ..Default::default()
            },
        ] {
            assert!(options.select(&mut [core()], &mut config()).is_err());
        }
        let mut p = core();
        p.graphs.retain(|g| g.name != "speculative2x3");
        p.execution
            .as_mut()
            .unwrap()
            .graphs
            .retain(|g| g.name != "speculative2x3");
        let options = LoadOptions {
            mtp_draft_tokens: Some(2),
            adaptive_verification: Some(false),
            ..Default::default()
        };
        assert!(
            options
                .select(&mut [p], &mut config())
                .unwrap_err()
                .to_string()
                .contains("batch 2")
        );
    }

    #[test]
    fn tensor_parallel_catalogs_must_match() {
        let mut cores = vec![core(), core()];
        cores[1].graphs.retain(|g| g.name != "speculative2x3");
        cores[1]
            .execution
            .as_mut()
            .unwrap()
            .graphs
            .retain(|g| g.name != "speculative2x3");
        assert!(
            LoadOptions::default()
                .select(&mut cores, &mut config())
                .unwrap_err()
                .to_string()
                .contains("catalogs differ")
        );
    }

    #[test]
    fn policy_defaults_and_overrides_select_the_recorded_catalog() {
        let mut fixed_config = config();
        fixed_config["speculative_fixed_width"] = 6.into();
        let mut cores = [core()];
        LoadOptions::default()
            .select(&mut cores, &mut fixed_config)
            .unwrap();
        assert_eq!(cores[0].graphs.len(), 18);
        let mut cores = [core()];
        LoadOptions {
            mtp_draft_tokens: Some(2),
            adaptive_verification: Some(true),
            ..Default::default()
        }
        .select(&mut cores, &mut fixed_config)
        .unwrap();
        assert!(fixed_config["speculative_fixed_width"].is_null());
        assert_eq!(cores[0].graphs.len(), 21);
    }

    #[test]
    fn globals_remain_even_when_their_module_has_no_selected_kernel() {
        let mut core = core();
        core.modules.push(Module {
            name: "data".into(),
            file: "data.cubin".into(),
            globals: vec![Global {
                symbol: "weights".into(),
                file: "weights.bin".into(),
                nbytes: 8,
                relocations: vec![],
            }],
        });
        let mut cores = [core];
        LoadOptions {
            max_concurrency: Some(1),
            mtp_draft_tokens: Some(0),
            ..Default::default()
        }
        .select(&mut cores, &mut config())
        .unwrap();
        assert_eq!(cores[0].graphs.len(), 5);
        assert_eq!(cores[0].modules.len(), 6);
        assert_eq!(cores[0].modules.last().unwrap().name, "data");
    }

    #[test]
    fn pinned_zero_selects_ordinary_operations_and_disables_draft_scheduling() {
        for adaptive in [None, Some(false), Some(true)] {
            let mut p = core();
            let mut config = config();
            config["history_drafts"] = true.into();
            LoadOptions {
                mtp_draft_tokens: Some(0),
                adaptive_verification: adaptive,
                ..Default::default()
            }
            .select(std::slice::from_mut(&mut p), &mut config)
            .unwrap();
            assert_eq!(config["speculative_widths"], serde_json::json!([]));
            assert!(config["speculative_fixed_width"].is_null());
            assert_eq!(config["history_drafts"], false);
            assert!(
                p.execution
                    .as_ref()
                    .unwrap()
                    .graphs
                    .iter()
                    .all(|g| g.speculation_enabled == Some(false))
            );
            assert_eq!(
                p.execution
                    .as_ref()
                    .unwrap()
                    .graphs
                    .iter()
                    .filter(|g| g.kind == "decode")
                    .count(),
                3
            );
        }
        let mut config = config();
        config["speculative_fixed_width"] = 1.into();
        LoadOptions::default()
            .select(&mut [core()], &mut config)
            .unwrap();
        assert_eq!(config["speculative_widths"], serde_json::json!([]));
    }

    #[test]
    fn pinned_zero_requires_ordinary_decode_instead_of_using_speculative_width_one() {
        let mut p = core();
        let names: HashSet<_> = p
            .execution
            .as_ref()
            .unwrap()
            .graphs
            .iter()
            .filter(|g| g.kind == "decode")
            .map(|g| g.name.clone())
            .collect();
        p.execution
            .as_mut()
            .unwrap()
            .graphs
            .retain(|g| !names.contains(&g.name));
        p.graphs.retain(|g| !names.contains(&g.name));
        let error = LoadOptions {
            mtp_draft_tokens: Some(0),
            ..Default::default()
        }
        .select(&mut [p], &mut config())
        .unwrap_err();
        assert!(error.to_string().contains("lacks decode graph"));
    }

    #[test]
    fn prefill_selection_matches_the_decode_mode() {
        for drafts in [0, 5] {
            let mut p = core();
            for enabled in [false, true] {
                let name = format!("prefill64-{enabled}");
                p.execution
                    .as_mut()
                    .unwrap()
                    .graphs
                    .push(wire::ExecutionGraph {
                        name: name.clone(),
                        kind: "prefill".into(),
                        sampling_mode: "general".into(),
                        estimated_duration_ns: 1000,
                        batch_capacity: 1,
                        token_capacity: 64,
                        speculation_enabled: Some(enabled),
                        ..Default::default()
                    });
                p.graphs.push(Graph {
                    name,
                    resource: "prefill".into(),
                    node_count: 1,
                    edge_count: 0,
                    flags: 0,
                    functions: vec![0],
                });
            }
            LoadOptions {
                mtp_draft_tokens: Some(drafts),
                ..Default::default()
            }
            .select(std::slice::from_mut(&mut p), &mut config())
            .unwrap();
            let selected: Vec<_> = p
                .execution
                .as_ref()
                .unwrap()
                .graphs
                .iter()
                .filter(|g| g.kind == "prefill")
                .collect();
            assert_eq!(selected.len(), 1);
            assert_eq!(selected[0].speculation_enabled, Some(drafts > 0));
        }
    }
    fn mixed_core() -> Core {
        let mut p = core();
        for enabled in [false, true] {
            let name = format!("prefill512-{enabled}");
            p.execution.as_mut().unwrap().graphs.push(wire::ExecutionGraph {
                name: name.clone(),
                kind: "prefill".into(),
                sampling_mode: "general".into(),
                batch_capacity: 1,
                token_capacity: 512,
                speculation_enabled: Some(enabled),
                ..Default::default()
            });
            let function = p.functions.len();
            p.functions.push(Function {
                id: function,
                module: name.clone(),
                symbol: "kernel".into(),
                parameter_sizes: vec![],
                attributes: vec![],
            });
            p.modules.push(Module {
                name: name.clone(),
                file: format!("{name}.cubin"),
                globals: vec![],
            });
            p.graphs.push(Graph {
                resource: name.clone(),
                name,
                node_count: 1,
                edge_count: 0,
                flags: 0,
                functions: vec![function],
            });
            for prefill in [64, 512] {
                for width in if enabled { vec![1, 3, 6] } else { vec![1] } {
                    let name = format!("mixed-4-{prefill}-{width}-{enabled}");
                    p.execution
                        .as_mut()
                        .unwrap()
                        .graphs
                        .push(wire::ExecutionGraph {
                            name: name.clone(),
                            kind: "mixed".into(),
                            sampling_mode: "general".into(),
                            batch_capacity: 4,
                            token_capacity: width,
                            prefill_capacity: prefill,
                            speculation_enabled: Some(enabled),
                            draft_tokens: enabled.then_some(width as u32 - 1),
                            verify_tokens: enabled.then_some(width as u32),
                            draft_source: if enabled { 1 } else { 0 },
                            ..Default::default()
                        });
                    let function = p.functions.len();
                    p.functions.push(Function {
                        id: function,
                        module: name.clone(),
                        symbol: "kernel".into(),
                        parameter_sizes: vec![],
                        attributes: vec![],
                    });
                    p.modules.push(Module {
                        name: name.clone(),
                        file: format!("{name}.cubin"),
                        globals: vec![],
                    });
                    p.graphs.push(Graph {
                        resource: name.clone(),
                        name,
                        node_count: 1,
                        edge_count: 0,
                        flags: 0,
                        functions: vec![function],
                    });
                }
            }
        }
        p
    }

    #[test]
    fn mixed_selection_preserves_prompt_variants_and_applies_decode_limits() {
        for (drafts, adaptive, widths) in [
            (0, true, vec![1]),
            (2, true, vec![1, 3]),
            (2, false, vec![1, 3]),
        ] {
            let mut p = mixed_core();
            let mut c = config();
            c["prefill_chunk_size"] = 512.into();
            LoadOptions {
                mtp_draft_tokens: Some(drafts),
                adaptive_verification: Some(adaptive),
                ..Default::default()
            }
            .select(std::slice::from_mut(&mut p), &mut c)
            .unwrap();
            let mixed: Vec<_> = p
                .execution
                .as_ref()
                .unwrap()
                .graphs
                .iter()
                .filter(|g| g.kind == "mixed")
                .collect();
            assert_eq!(mixed.len(), 2 * widths.len());
            for prefill in [64, 512] {
                for &width in &widths {
                    let graph = mixed
                        .iter()
                        .find(|g| g.prefill_capacity == prefill && g.token_capacity == width)
                        .unwrap();
                    assert_eq!(graph.speculation_enabled, Some(drafts > 0));
                    assert_eq!(graph.packed_tokens(), prefill + 3 * width);
                }
            }
        }
    }

    #[test]
    fn mixed_prefill_override_retains_separate_graphs_and_prunes_resources() {
        for (drafts, adaptive) in [(0, true), (2, true), (2, false)] {
            let mut baseline = [mixed_core(), mixed_core()];
            let mut c = config();
            c["prefill_chunk_size"] = 512.into();
            let options = LoadOptions {
                mtp_draft_tokens: Some(drafts),
                adaptive_verification: Some(adaptive),
                ..Default::default()
            };
            let mut enabled = [mixed_core(), mixed_core()];
            let mut disabled = [mixed_core(), mixed_core()];
            let mut enabled_config = c.clone();
            let mut disabled_config = c.clone();
            assert_eq!(options.select(&mut baseline, &mut c).unwrap(), 4);
            assert_eq!(
                LoadOptions { mixed_prefill: Some(true), ..options.clone() }
                    .select(&mut enabled, &mut enabled_config).unwrap(),
                4
            );
            assert_eq!(
                LoadOptions { mixed_prefill: Some(false), ..options }
                    .select(&mut disabled, &mut disabled_config).unwrap(),
                4
            );
            assert_eq!(enabled_config, c);
            assert_eq!(disabled_config, c);
            for ((baseline, enabled), disabled) in
                baseline.iter().zip(&enabled).zip(&disabled)
            {
                let graphs = &baseline.execution.as_ref().unwrap().graphs;
                assert!(graphs.iter().any(|g| g.kind == "mixed"));
                assert!(graphs.iter().any(|g| g.kind == "prefill"));
                assert_eq!(&enabled.execution.as_ref().unwrap().graphs, graphs);
                let separate: Vec<_> = graphs.iter()
                    .filter(|g| g.kind != "mixed").cloned().collect();
                assert_eq!(&disabled.execution.as_ref().unwrap().graphs, &separate);
                let names: HashSet<_> = separate.iter().map(|g| g.name.as_str()).collect();
                assert_eq!(
                    disabled.graphs.iter().map(|g| g.name.as_str()).collect::<HashSet<_>>(),
                    names
                );
                assert_eq!(
                    disabled.modules.iter().map(|m| m.name.as_str()).collect::<HashSet<_>>(),
                    names
                );
                assert!(disabled.functions.iter().all(|f| names.contains(f.module.as_str())));
                assert_eq!(disabled.functions.len(), separate.len());
                assert_eq!(disabled.allocations.len(), baseline.allocations.len());
            }
        }
    }

    #[test]
    fn mixed_prefill_explicit_enable_requires_eligible_capture() {
        for (make_core, concurrency) in [
            (core as fn() -> Core, 4), (mixed_core, 1), (mixed_core, 2),
        ] {
            let mut c = config();
            c["prefill_chunk_size"] = 512.into();
            let options = LoadOptions {
                max_concurrency: Some(concurrency),
                ..Default::default()
            };
            for mixed_prefill in [None, Some(false)] {
                LoadOptions { mixed_prefill, ..options.clone() }
                    .select(&mut [make_core()], &mut c.clone()).unwrap();
            }
            let error = LoadOptions { mixed_prefill: Some(true), ..options }
                .select(&mut [make_core()], &mut c).unwrap_err();
            assert!(error.to_string().contains("no mixed prefill graph"));
        }
    }

    #[test]
    fn mixed_prefill_disable_preserves_capture_validation() {
        let mut p = mixed_core();
        p.execution.as_mut().unwrap().graphs.iter_mut()
            .find(|g| g.kind == "mixed").unwrap().prefill_capacity = 65;
        let mut c = config();
        c["prefill_chunk_size"] = 512.into();
        let error = LoadOptions { mixed_prefill: Some(false), ..Default::default() }
            .select(&mut [p], &mut c).unwrap_err();
        assert!(error.to_string().contains("invalid or duplicate serving graph variant"));
    }

    #[test]
    fn mixed_prefill_disable_prunes_only_unreferenced_startup_producers() {
        for mixed_prefill in [Some(true), Some(false)] {
            let mut p = mixed_core();
            let tuning = tuning_core();
            let execution = p.execution.as_mut().unwrap();
            let tuning_execution = tuning.execution.unwrap();
            execution.buffers = tuning_execution.buffers;
            execution.tuning_graphs = tuning_execution.tuning_graphs;
            p.graphs.extend(tuning.graphs.into_iter().filter(|g| {
                g.name == "tune" || g.name == "unused_tune"
            }));
            execution.graphs.iter_mut().find(|g| g.name == "decode1-no-mtp").unwrap()
                .tuned_bindings.push(wire::TunedBinding {
                    buffer: "winner".into(), tuning_graph: "tune".into(),
                });
            execution.graphs.iter_mut().find(|g| {
                g.kind == "mixed" && g.speculation_enabled == Some(false)
            }).unwrap().tuned_bindings.push(wire::TunedBinding {
                buffer: "unused_winner".into(), tuning_graph: "unused_tune".into(),
            });
            let mut c = config();
            c["prefill_chunk_size"] = 512.into();
            LoadOptions {
                mtp_draft_tokens: Some(0),
                mixed_prefill,
                ..Default::default()
            }.select(std::slice::from_mut(&mut p), &mut c).unwrap();
            let keep_mixed = mixed_prefill == Some(true);
            assert_eq!(p.execution.as_ref().unwrap().tuning_graphs.len(),
                if keep_mixed { 2 } else { 1 });
            assert!(p.graphs.iter().any(|g| g.name == "tune"));
            assert_eq!(p.graphs.iter().any(|g| g.name == "unused_tune"), keep_mixed);
        }
    }

    #[test]
    fn mixed_selection_rejects_rank_shape_mismatch_and_invalid_mode() {
        let mut ranks = [mixed_core(), mixed_core()];
        let mut c = config();
        c["prefill_chunk_size"] = 512.into();
        ranks[1]
            .execution
            .as_mut()
            .unwrap()
            .graphs
            .iter_mut()
            .find(|g| g.kind == "mixed" && g.speculation_enabled == Some(true))
            .unwrap()
            .prefill_capacity = 128;
        assert!(
            LoadOptions::default()
                .select(&mut ranks, &mut c)
                .unwrap_err()
                .to_string()
                .contains("catalogs differ")
        );
        for mutation in 0..4 {
            let mut p = mixed_core();
            let g = p
                .execution
                .as_mut()
                .unwrap()
                .graphs
                .iter_mut()
                .find(|g| g.kind == "mixed")
                .unwrap();
            match mutation {
                0 => g.speculation_enabled = None,
                1 => g.token_capacity = 3,
                2 => g.prefill_capacity = 65,
                _ => g.prefill_capacity = 1024,
            }
            let mut c = config();
            c["prefill_chunk_size"] = 512.into();
            assert!(LoadOptions::default().select(&mut [p], &mut c).is_err());
        }
    }

    fn refresh_graph_resources(core: &mut Core) {
        core.graphs = core.execution.as_ref().unwrap().graphs.iter().map(|graph| Graph {
            name: graph.name.clone(), flags: 0, resource: graph.name.clone(),
            node_count: 1, edge_count: 0, functions: vec![0],
        }).collect();
    }

    fn round_core(active: usize, budget: usize, confidence: bool) -> Core {
        let mut core = core();
        let execution = core.execution.as_mut().unwrap();
        execution.max_active = active as u64;
        execution.graphs.retain(|graph| graph.kind != "speculative" && graph.batch_capacity <= active.next_power_of_two() as u64);
        let state_templates: Vec<_> = execution.graphs.iter().filter(|graph| graph.batch_capacity == 4).cloned().collect();
        for batch in [8u64, 16, 32, 64].into_iter().filter(|batch| *batch <= active.next_power_of_two() as u64) {
            for template in &state_templates {
                let mut graph = template.clone();
                graph.name = format!("{}-b{batch}-{:?}", graph.kind, graph.speculation_enabled);
                graph.batch_capacity = batch;
                execution.graphs.push(graph);
            }
        }
        for batch in BATCH_CAPACITIES.into_iter().filter(|batch| *batch <= active.next_power_of_two()) {
            execution.graphs.push(wire::ExecutionGraph {
                name: format!("round-b{batch}-d0-w1-p{batch}"), kind: "speculative".into(),
                sampling_mode: "general".into(), batch_capacity: batch as u64,
                token_capacity: 1, packed_token_capacity: batch as u64,
                draft_tokens: Some(0), verify_tokens: Some(1), draft_source: 1,
                ..Default::default()
            });
            for depth in [1u32, 3, 5] {
                for rows in batch..=(batch * 6).min(budget) {
                    execution.graphs.push(wire::ExecutionGraph {
                        name: format!("round-b{batch}-d{depth}-w6-p{rows}"), kind: "speculative".into(),
                        sampling_mode: "general".into(), batch_capacity: batch as u64,
                        token_capacity: 6, packed_token_capacity: rows as u64,
                        draft_tokens: Some(depth), verify_tokens: Some(6), draft_source: 1,
                        confidence_width: if confidence { depth } else { 0 },
                        ..Default::default()
                    });
                }
            }
        }
        refresh_graph_resources(&mut core);
        core
    }

    fn round_config(budget: usize) -> Value {
        serde_json::json!({"speculative_token_capacity": budget, "speculative_widths": [1,6]})
    }

    #[test]
    fn complete_round_selection_uses_true_captured_depth_and_retains_wide_targets() {
        for requested in [1usize, 2, 3, 4, 5] {
            let mut ranks = [round_core(8, 48, true), round_core(8, 48, true)];
            LoadOptions { mtp_draft_tokens: Some(requested), ..Default::default() }
                .select(&mut ranks, &mut round_config(48)).unwrap();
            let expected = [1, 3, 5].into_iter().filter(|depth| *depth <= requested).max().unwrap() as u32;
            for core in ranks {
                let graphs = &core.execution.unwrap().graphs;
                for batch in [1u64, 2, 4, 8] {
                    let rounds: Vec<_> = graphs.iter().filter(|graph| graph.kind == "speculative" && graph.batch_capacity == batch).collect();
                    assert!(rounds.iter().any(|graph| graph.token_capacity == 1 && graph.draft_tokens == Some(0) && graph.packed_token_capacity == batch));
                    assert!(rounds.iter().filter(|graph| graph.token_capacity > 1).all(|graph| graph.token_capacity == 6 && graph.draft_tokens == Some(expected) && graph.confidence_width == expected));
                    assert!(rounds.iter().any(|graph| graph.token_capacity == 6));
                }
            }
        }
    }

    #[test]
    fn exact_and_maximum_round_budgets_preserve_partial_and_pending_only_geometry() {
        for active in [1usize, 3, 8, 32] {
            for budget in [1usize, 3, 7, 8, 9, 24, 32, 128] {
                for exact in [false, true] {
                    let mut ranks = [round_core(active, 128, false), round_core(active, 128, false)];
                    let options = LoadOptions {
                        mtp_verification_token_budget: exact.then_some(budget),
                        mtp_max_verification_token_budget: (!exact).then_some(budget),
                        ..Default::default()
                    };
                    let mut config = round_config(128);
                    options.select(&mut ranks, &mut config).unwrap();
                    assert_eq!(config["speculative_token_capacity"], budget);
                    for core in ranks {
                        let graphs = &core.execution.unwrap().graphs;
                        for batch in BATCH_CAPACITIES.into_iter().filter(|batch| *batch <= active.next_power_of_two()) {
                            assert!(graphs.iter().any(|graph| graph.kind == "speculative" && graph.batch_capacity == batch as u64 && graph.token_capacity == 1 && graph.draft_tokens == Some(0)));
                            let rows: BTreeSet<_> = graphs.iter().filter(|graph| graph.kind == "speculative" && graph.batch_capacity == batch as u64 && graph.token_capacity == 6).map(|graph| graph.packed_token_capacity as usize).collect();
                            let upper = budget.min(batch * 6);
                            let expected: BTreeSet<_> = if upper < batch { BTreeSet::new() }
                                else if exact { [upper].into_iter().collect() }
                                else { (batch..=upper).collect() };
                            assert_eq!(rows, expected, "active={active} budget={budget} batch={batch} exact={exact}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn budget_selection_preserves_captured_limits_and_concurrency_reductions() {
        for concurrency in 1usize..=8 {
            for exact in [false, true] {
                let mut core = round_core(8, 9, true);
                LoadOptions {
                    max_concurrency: Some(concurrency),
                    mtp_verification_token_budget: exact.then_some(9),
                    mtp_max_verification_token_budget: (!exact).then_some(24),
                    ..Default::default()
                }.select(std::slice::from_mut(&mut core), &mut round_config(9)).unwrap();
                assert!(core.execution.unwrap().graphs.iter().all(|graph|
                    graph.batch_capacity <= concurrency.next_power_of_two() as u64
                    && (graph.kind != "speculative" || graph.token_capacity == 1 || graph.packed_token_capacity <= 9)));
            }
        }
    }

    #[test]
    fn round_budgets_reject_missing_required_shapes_on_either_rank() {
        for exact in [false, true] {
            for rank in [0usize, 1] {
                for width in [1, 6] {
                    let mut ranks = [round_core(8, 48, false), round_core(8, 48, false)];
                    ranks[rank].execution.as_mut().unwrap().graphs.retain(|graph|
                        !(graph.kind == "speculative" && graph.batch_capacity == 4 && graph.token_capacity == width));
                    refresh_graph_resources(&mut ranks[rank]);
                    let error = LoadOptions {
                        mtp_verification_token_budget: exact.then_some(24),
                        mtp_max_verification_token_budget: (!exact).then_some(24),
                        ..Default::default()
                    }.select(&mut ranks, &mut round_config(48)).unwrap_err().to_string();
                    assert!(error.contains(&format!("rank {rank}")) && error.contains("batch 4"), "{error}");
                }
            }
        }
        let mut ranks = [round_core(8, 48, false), round_core(8, 48, false)];
        ranks[1].execution.as_mut().unwrap().graphs.retain(|graph|
            !(graph.kind == "speculative" && graph.batch_capacity == 4 && graph.packed_token_capacity == 7));
        refresh_graph_resources(&mut ranks[1]);
        assert!(LoadOptions { mtp_max_verification_token_budget: Some(24), ..Default::default() }
            .select(&mut ranks, &mut round_config(48)).unwrap_err().to_string().contains("catalogs differ"));
    }

    #[test]
    fn exact_budget_requires_its_capacity_while_maximum_budget_keeps_other_captures() {
        for rank in [0usize, 1] {
            let mut ranks = [round_core(8, 48, false), round_core(8, 48, false)];
            ranks[rank].execution.as_mut().unwrap().graphs.retain(|graph|
                !(graph.kind == "speculative" && graph.batch_capacity == 4 && graph.packed_token_capacity == 24));
            refresh_graph_resources(&mut ranks[rank]);
            let error = LoadOptions { mtp_verification_token_budget: Some(24), ..Default::default() }
                .select(&mut ranks, &mut round_config(48)).unwrap_err().to_string();
            assert!(error.contains(&format!("rank {rank}")) && error.contains("batch 4"), "{error}");
        }
        let mut core = round_core(8, 48, false);
        core.execution.as_mut().unwrap().graphs.retain(|graph|
            !(graph.kind == "speculative" && graph.batch_capacity == 4 && graph.packed_token_capacity == 24));
        refresh_graph_resources(&mut core);
        LoadOptions { mtp_max_verification_token_budget: Some(24), ..Default::default() }
            .select(std::slice::from_mut(&mut core), &mut round_config(48)).unwrap();
        let rows: BTreeSet<_> = core.execution.unwrap().graphs.into_iter().filter(|graph|
            graph.kind == "speculative" && graph.batch_capacity == 4 && graph.token_capacity == 6)
            .map(|graph| graph.packed_token_capacity).collect();
        assert_eq!(rows, (4..24).collect());
    }

    #[test]
    fn no_head_and_confidence_rounds_preserve_their_readback_contracts() {
        for confidence in [false, true] {
            let mut core = round_core(8, 48, confidence);
            LoadOptions::default().select(std::slice::from_mut(&mut core), &mut round_config(48)).unwrap();
            assert!(core.execution.unwrap().graphs.iter().filter(|graph| graph.kind == "speculative")
                .all(|graph| graph.confidence_width == if confidence { graph.draft_tokens.unwrap() } else { 0 }));
        }
        let mut ranks = [round_core(8, 48, false), round_core(8, 48, true)];
        assert!(LoadOptions::default().select(&mut ranks, &mut round_config(48)).unwrap_err().to_string().contains("catalogs differ"));
    }

    #[test]
    fn execution_abi_labels_do_not_reject_core_selection() {
        for rank in [0usize, 1] {
            for abi in [0, 1, 2, 3, 4, 5, 6, 7, u32::MAX] {
                let mut ranks = [round_core(8, 48, false), round_core(8, 48, false)];
                ranks[rank].execution.as_mut().unwrap().abi_version = abi;
                LoadOptions::default().select(&mut ranks, &mut round_config(48)).unwrap();
            }
        }
    }

    #[test]
    fn malformed_rounds_reject_before_loading() {
        assert!(LoadOptions { mtp_verification_token_budget: Some(24), ..Default::default() }
            .select(&mut [round_core(8, 9, false)], &mut round_config(9)).is_err());
        for mutation in 0..10 {
            let mut core = round_core(8, 48, true);
            let graph = core.execution.as_mut().unwrap().graphs.iter_mut()
                .find(|graph| graph.kind == "speculative" && graph.batch_capacity == 4 && graph.token_capacity == 6).unwrap();
            match mutation {
                0 => graph.packed_token_capacity = 0,
                1 => graph.packed_token_capacity = graph.batch_capacity - 1,
                2 => graph.packed_token_capacity = graph.batch_capacity * graph.token_capacity + 1,
                3 => graph.draft_tokens = None,
                4 => graph.draft_tokens = Some(16),
                5 => graph.verify_tokens = Some(5),
                6 => graph.confidence_width = 7,
                7 => graph.draft_source = 0,
                8 => graph.kind = "proposal".into(),
                _ => graph.kind = "verification".into(),
            }
            assert!(LoadOptions::default().select(&mut [core], &mut round_config(48)).is_err());
        }
    }

    #[test]
    fn round_budget_options_require_positive_adaptive_speculation() {
        for options in [
            LoadOptions { mtp_verification_token_budget: Some(0), ..Default::default() },
            LoadOptions { mtp_max_verification_token_budget: Some(0), ..Default::default() },
            LoadOptions { mtp_verification_token_budget: Some(8), mtp_draft_tokens: Some(0), ..Default::default() },
            LoadOptions { mtp_max_verification_token_budget: Some(8), adaptive_verification: Some(false), ..Default::default() },
            LoadOptions { mtp_verification_token_budget: Some(8), mtp_max_verification_token_budget: Some(8), ..Default::default() },
        ] {
            assert!(options.select(&mut [round_core(8, 48, false)], &mut round_config(48)).is_err());
        }
    }
    #[test]
    fn captured_row_budget_below_verification_width_selects_without_relabeling() {
        for budget in [1usize, 3] {
            let mut core = round_core(8, budget, false);
            let mut config = round_config(budget);
            LoadOptions::default().select(std::slice::from_mut(&mut core), &mut config).unwrap();
            let rounds: Vec<_> = core.execution.unwrap().graphs.into_iter()
                .filter(|graph| graph.kind == "speculative" && graph.token_capacity > 1).collect();
            assert!(!rounds.is_empty());
            assert!(rounds.iter().all(|graph| graph.draft_tokens == Some(5)
                && graph.token_capacity == 6 && graph.packed_token_capacity <= budget as u64));
            assert_eq!(config["speculative_token_capacity"], budget);
            assert_eq!(config["speculative_widths"], serde_json::json!([1, 6]));
        }
    }

    #[test]
    fn width_one_rounds_require_zero_actual_proposal_work() {
        for depth in [1, 7] {
            let mut core = core();
            core.execution.as_mut().unwrap().graphs.iter_mut()
                .find(|graph| graph.kind == "speculative" && graph.token_capacity == 1).unwrap().draft_tokens = Some(depth);
            assert!(LoadOptions::default().select(&mut [core], &mut config()).is_err());
        }
    }

    #[test]
    fn malformed_explicit_fixed_width_rejects_even_with_an_adaptive_override() {
        for value in [serde_json::json!("6"), serde_json::json!(-1), serde_json::json!(2.5),
            serde_json::json!(true), serde_json::json!([]), serde_json::json!({})] {
            for adaptive in [None, Some(false), Some(true)] {
                let mut configuration = config();
                configuration["speculative_fixed_width"] = value.clone();
                let error = LoadOptions { adaptive_verification: adaptive, ..Default::default() }
                    .select(&mut [core()], &mut configuration).unwrap_err().to_string();
                assert!(error.contains("speculative_fixed_width must be an integer or null"), "{error}");
            }
        }
    }

    fn host_wide_round_core() -> Core {
        let mut core = round_core(4, 24, false);
        core.execution.as_mut().unwrap().graphs.retain(|graph|
            graph.kind != "speculative" || graph.token_capacity == 1 || graph.draft_tokens == Some(5));
        for graph in &mut core.execution.as_mut().unwrap().graphs {
            if graph.kind == "speculative" && graph.token_capacity > 1 {
                graph.token_capacity = 8;
                graph.verify_tokens = Some(8);
                graph.name = graph.name.replace("-w6-", "-w8-");
            }
        }
        refresh_graph_resources(&mut core);
        core
    }

    #[test]
    fn fixed_host_rounds_keep_actual_width_independently_of_draft_depth() {
        for fixed in [None, Some(8)] {
            let mut configuration = serde_json::json!({"speculative_widths": [1,8], "speculative_token_capacity": 24,
                "speculative_fixed_width": fixed});
            let mut ranks = [host_wide_round_core(), host_wide_round_core()];
            LoadOptions { mtp_draft_tokens: Some(5), adaptive_verification: Some(false), ..Default::default() }
                .select(&mut ranks, &mut configuration).unwrap();
            assert_eq!(configuration["speculative_fixed_width"], 8);
            for core in ranks {
                let rounds: Vec<_> = core.execution.unwrap().graphs.into_iter()
                    .filter(|graph| graph.kind == "speculative" && graph.token_capacity > 1).collect();
                assert!(!rounds.is_empty());
                assert!(rounds.iter().all(|graph| graph.token_capacity == 8 && graph.verify_tokens == Some(8)
                    && graph.draft_tokens == Some(5) && graph.confidence_width == 0));
            }
        }
    }

    #[test]
    fn explicit_fixed_width_is_not_replaced_by_a_smaller_captured_width() {
        let mut configuration = config();
        configuration["speculative_fixed_width"] = 6.into();
        let error = LoadOptions { mtp_draft_tokens: Some(1), adaptive_verification: Some(false), ..Default::default() }
            .select(&mut [core()], &mut configuration).unwrap_err().to_string();
        assert!(error.contains("no captured speculative width fits"), "{error}");
        assert_eq!(configuration["speculative_fixed_width"], 6);
    }

    #[test]
    fn unsupported_positive_draft_limit_rejects_without_selecting_ordinary_decode() {
        for adaptive in [false, true] {
            for budget in [1, 24] {
                let make_core = || {
                    let mut core = host_wide_round_core();
                    core.execution.as_mut().unwrap().graphs.retain(|graph|
                        graph.kind != "speculative" || graph.token_capacity == 1 || graph.packed_token_capacity <= budget);
                    refresh_graph_resources(&mut core);
                    core
                };
                let mut configuration = serde_json::json!({"speculative_widths": [1,8], "speculative_token_capacity": budget});
                let error = LoadOptions { mtp_draft_tokens: Some(1), adaptive_verification: Some(adaptive), ..Default::default() }
                    .select(&mut [make_core()], &mut configuration).unwrap_err().to_string();
                assert!(error.contains("no captured speculative width fits"), "{error}");
                LoadOptions { mtp_draft_tokens: Some(0), adaptive_verification: Some(adaptive), ..Default::default() }
                    .select(&mut [make_core()], &mut configuration).unwrap();
                assert_eq!(configuration["speculative_widths"], serde_json::json!([]));
            }
        }
    }

}
