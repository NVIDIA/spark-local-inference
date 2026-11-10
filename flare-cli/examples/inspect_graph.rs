/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Inspect recorded graph launches without CUDA or weight reads.
//!
//! Prints JSON Lines; without --graph, lists the selected rank's graph catalog.
//! Argument relocations describe pointers and tensor maps patched during loading.
//! Cooperative grids are recorded dimensions: the loader can reduce them when
//! the target GPU rejects the cooperative launch's residency requirement.

use std::{
    collections::BTreeSet,
    fs::OpenOptions,
    io::{self, BufWriter, Read, Write},
    path::PathBuf,
};

use clap::Parser;
use flare_helios::{
    core::{
        container::Slice,
        protection,
        reader::Metadata,
        schema::{self, wire},
    },
    runtime_core::wire as runtime,
};
use prost::Message;
use serde_json::json;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Parser)]
#[command(
    about = "Inspect payload-format graph records on CPU without reading weights",
    after_help = "Output is JSON Lines. Metadata signatures are not verified by this diagnostic.\nCompressed resource chunk checks remain enabled. Recorded cooperative grids may\nbe reduced by the native loader to fit the GPU; pointer/TMA relocations are applied\nat load time. --cubin writes one selected module only, and refuses to overwrite."
)]
struct Args {
    core: PathBuf,
    #[arg(long, default_value_t = 0)]
    rank: u32,
    /// Deployment ID; required when the core contains several deployments.
    #[arg(long)]
    deployment: Option<String>,
    /// Exact graph name; omit to list graph metadata only.
    #[arg(long)]
    graph: Option<String>,
    /// Exact module name to extract as a cubin.
    #[arg(long, requires = "output")]
    cubin: Option<String>,
    #[arg(long, requires = "cubin")]
    output: Option<PathBuf>,
}

struct SliceReader {
    slice: Slice,
    offset: u64,
}

impl Read for SliceReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let count = buffer.len().min((self.slice.len() - self.offset) as usize);
        self.slice
            .read_at(self.offset, &mut buffer[..count])
            .map_err(io::Error::other)?;
        self.offset += count as u64;
        Ok(count)
    }
}

fn record<T: Message + Default>(reader: &mut impl Read) -> Result<T> {
    let mut prefix = [0; 4];
    reader.read_exact(&mut prefix)?;
    let size = u32::from_le_bytes(prefix) as usize;
    if size > 1 << 20 {
        return Err("graph record exceeds 1 MiB".into());
    }
    let mut bytes = vec![0; size];
    reader.read_exact(&mut bytes)?;
    Ok(T::decode(bytes.as_slice())?)
}

fn finish(reader: &mut impl Read) -> Result<()> {
    if reader.read(&mut [0])? != 0 {
        return Err("trailing graph record bytes".into());
    }
    Ok(())
}

fn emit(output: &mut impl Write, value: serde_json::Value) -> Result<()> {
    serde_json::to_writer(&mut *output, &value)?;
    writeln!(output)?;
    Ok(())
}

fn resource_reader(
    metadata: &Metadata,
    catalog: &schema::Catalog,
    resource: &wire::ExecutableResource,
) -> Result<impl Read> {
    let stored = resource.stored.as_ref().ok_or("missing executable slice")?;
    let files = metadata.payloads(catalog, [stored.payload_id.as_slice()])?;
    let slice = files.slice(stored)?;
    Ok(protection::open_compressed_reader(
        SliceReader { slice, offset: 0 },
        stored.nbytes,
        resource.decoded_nbytes,
        &resource.chunk_sha256,
    )?)
}

fn run(args: Args) -> Result<()> {
    let metadata = Metadata::open(&args.core)?;
    let envelope = protection::envelope(&metadata.envelope)?;
    let core = wire::CoreBody::decode(envelope.core_body.as_slice())?;
    let mut catalog = schema::validate_core(&core)?;
    let deployments = envelope
        .deployments
        .iter()
        .map(|signed| {
            Ok((
                signed.index,
                wire::DeploymentBody::decode(signed.body.as_slice())?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let selected = deployments
        .iter()
        .filter(|(_, body)| {
            args.deployment
                .as_ref()
                .is_none_or(|name| name == &body.deployment_id)
        })
        .collect::<Vec<_>>();
    if selected.len() != 1 {
        return Err("select exactly one deployment with --deployment".into());
    }
    let (deployment_index, deployment) = selected[0];
    let ranks = schema::validate_deployment(deployment, &mut catalog)?;
    let selected = envelope
        .ranks
        .iter()
        .filter(|signed| signed.deployment == *deployment_index && signed.rank == args.rank)
        .collect::<Vec<_>>();
    if selected.len() != 1 {
        return Err("selected rank is missing or duplicated".into());
    }
    let rank = wire::RankBody::decode(selected[0].body.as_slice())?;
    if rank.rank != args.rank || rank.deployment_id != deployment.deployment_id {
        return Err("rank metadata disagrees with its deployment".into());
    }
    schema::validate_rank(&rank, ranks, &mut catalog, &mut BTreeSet::new())?;
    let mut output = BufWriter::new(io::stdout().lock());
    emit(
        &mut output,
        json!({"type": "rank", "deployment": deployment.deployment_id,
        "rank": rank.rank, "architecture": deployment.architecture,
        "signatures_verified": false, "graphs": rank.graphs,
        "execution": rank.execution}),
    )?;

    if let Some(name) = &args.graph {
        let graph = rank
            .graphs
            .iter()
            .find(|graph| &graph.name == name)
            .ok_or("graph name is absent from selected rank")?;
        let resource = &rank.resources[graph.resource as usize];
        let mut reader = resource_reader(&metadata, &catalog, resource)?;
        emit(
            &mut output,
            json!({"type": "graph", "graph": graph,
            "allocations": rank.allocations, "events": rank.events}),
        )?;
        for index in 0..graph.node_count {
            let node: runtime::Node = record(&mut reader)?;
            let operation = node
                .operation
                .as_ref()
                .ok_or("graph node has no supported operation")?;
            let function = if let runtime::node::Operation::Kernel(kernel) = operation {
                let function = rank
                    .functions
                    .get(usize::try_from(kernel.function)?)
                    .ok_or("kernel function index is outside rank")?;
                if !graph.functions.contains(&kernel.function)
                    || kernel.grid.len() != 3
                    || kernel.grid.contains(&0)
                    || kernel.block.len() != 3
                    || kernel.block.contains(&0)
                    || kernel.arguments.len() != function.parameter_sizes.len()
                    || kernel
                        .arguments
                        .iter()
                        .zip(&function.parameter_sizes)
                        .any(|(argument, size)| argument.data.len() as u64 != *size)
                {
                    return Err(
                        "invalid kernel geometry, function dependency or argument sizes".into(),
                    );
                }
                Some(function)
            } else {
                None
            };
            emit(
                &mut output,
                json!({"type": "node", "index": index,
                "node": node, "function": function}),
            )?;
        }
        for index in 0..graph.edge_count {
            let edge: runtime::Edge = record(&mut reader)?;
            if edge.source >= graph.node_count || edge.destination >= graph.node_count {
                return Err("edge endpoint is outside graph".into());
            }
            emit(
                &mut output,
                json!({"type": "edge", "index": index, "edge": edge}),
            )?;
        }
        finish(&mut reader)?;
        let mut modules = BTreeSet::new();
        for function in &graph.functions {
            let function = &rank.functions[*function as usize];
            if !modules.insert(function.module.as_str()) {
                continue;
            }
            let module = rank
                .modules
                .iter()
                .find(|module| module.name == function.module)
                .ok_or("function module is absent")?;
            emit(&mut output, json!({"type": "module", "module": module}))?;
        }
    }
    if let Some(name) = &args.cubin {
        let module = rank
            .modules
            .iter()
            .find(|module| &module.name == name)
            .ok_or("module name is absent from selected rank")?;
        let resource = &rank.resources[module.cubin as usize];
        let mut reader = resource_reader(&metadata, &catalog, resource)?;
        let path = args
            .output
            .as_ref()
            .ok_or("--output is required with --cubin")?;
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        let nbytes = io::copy(&mut reader, &mut file)?;
        file.flush()?;
        emit(
            &mut output,
            json!({"type": "cubin", "module": module.name,
            "output": path, "nbytes": nbytes}),
        )?;
    }
    output.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    run(Args::parse())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_and_edge_records_preserve_arguments_and_dependencies() {
        let node = runtime::Node {
            operation: Some(runtime::node::Operation::Kernel(runtime::Kernel {
                function: 3,
                grid: vec![32, 1, 1],
                block: vec![96, 1, 1],
                shared: 48128,
                arguments: vec![runtime::Blob {
                    data: vec![0; 8],
                    relocations: vec![runtime::Relocation {
                        offset: 0,
                        target: Some(runtime::Reference {
                            allocation: "scratch".into(),
                            offset: 128,
                        }),
                        kind: 0,
                    }],
                }],
                attributes: vec![],
            })),
        };
        let edge = runtime::Edge {
            source: 0,
            destination: 1,
            data: vec![1, 0, 0],
        };
        let mut bytes = Vec::new();
        for body in [node.encode_to_vec(), edge.encode_to_vec()] {
            bytes.extend((body.len() as u32).to_le_bytes());
            bytes.extend(body);
        }
        let mut reader = bytes.as_slice();
        assert_eq!(record::<runtime::Node>(&mut reader).unwrap(), node);
        assert_eq!(record::<runtime::Edge>(&mut reader).unwrap(), edge);
        finish(&mut reader).unwrap();
    }

    #[test]
    fn rejects_truncated_oversized_and_trailing_records() {
        for bytes in [
            vec![1, 0],
            vec![8, 0, 0, 0, 1],
            ((1u32 << 20) + 1).to_le_bytes().to_vec(),
        ] {
            assert!(record::<runtime::Node>(&mut bytes.as_slice()).is_err());
        }
        assert!(finish(&mut [1u8].as_slice()).is_err());
    }
}
