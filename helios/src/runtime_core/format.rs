/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::{HashMap, HashSet};

use crate::{Error, ExecutionError, Result};

pub(super) fn invalid(message: impl Into<String>) -> Error {
    Error::execution(ExecutionError::Manifest, message)
}

#[derive(Debug)]
pub(super) struct Core {
    pub resources: HashMap<String, super::protobuf::wire::Span>,
    pub architecture: [i32; 2],
    pub allocations: Vec<Allocation>,
    pub modules: Vec<Module>,
    pub functions: Vec<Function>,
    pub graphs: Vec<Graph>,
    pub reservations: Vec<Reservation>,
    pub execution: Option<super::protobuf::wire::Execution>,
    pub context: Context,
    pub events: Vec<String>,
}

#[derive(Debug, Default)]
pub(super) struct Context {
    pub flags: u32,
    pub cache: i32,
    pub limits: Vec<[usize; 2]>,
}

#[derive(Debug)]
pub(super) struct Reservation {
    pub address: u64,
    pub nbytes: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) enum Memory {
    #[default]
    Device,
    MappedHost,
    /// Another rank's device allocation, referenced at its recorded address.
    Peer,
}

/// One contiguous stored run of a payload-bound allocation: which payload resource supplies the
/// bytes and where they start inside it. The destination offset comes from the parallel entry in
/// `Allocation::data_spans`, so a single allocation may be assembled from several payloads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Source {
    pub resource: String,
    pub offset: u64,
    pub nbytes: u64,
}

/// Where one authenticated payload lives in an opened container, as resolved by the core
/// reader. `token` identifies an open file handle the archive holds; `offset` is the byte offset of
/// the payload inside that container.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Placement {
    pub token: String,
    pub offset: u64,
    pub nbytes: u64,
}

pub(super) type Placements = std::collections::BTreeMap<Vec<u8>, Placement>;

#[derive(Debug)]
pub(super) struct Allocation {
    pub name: String,
    pub nbytes: usize,
    pub alignment: usize,
    pub file: Option<String>,
    pub data_spans: Vec<[usize; 2]>,
    /// Parallel to `data_spans` when nonempty. Empty keeps the legacy rule: the allocation's whole
    /// payload is its resource, consumed sequentially in span order.
    pub sources: Vec<Source>,
    pub relocations: Vec<Relocation>,
    pub address: Option<u64>,
    pub memory: Memory,
    pub sizing: Option<super::protobuf::wire::Sizing>,
}

impl Allocation {
    pub fn extent(&self) -> usize {
        self.sizing.as_ref().map_or(self.nbytes, |s| {
            (s.unit_bytes as usize * s.maximum as usize).div_ceil(self.alignment) * self.alignment
        })
    }

    pub fn spans(&self) -> impl Iterator<Item = [usize; 2]> + '_ {
        self.data_spans.iter().copied()
    }

    pub fn stored_bytes(&self) -> usize {
        self.spans().map(|span| span[1]).sum()
    }
    /// True when every span names its own payload resource.
    pub fn is_payload_bound(&self) -> bool {
        !self.sources.is_empty()
    }
}

#[derive(Debug)]
pub(super) struct Module {
    pub name: String,
    pub file: String,
    pub globals: Vec<Global>,
}

#[derive(Debug)]
pub(super) struct Global {
    pub symbol: String,
    pub file: String,
    pub nbytes: usize,
    pub relocations: Vec<Relocation>,
}

#[derive(Debug)]
pub(super) struct Function {
    pub id: usize,
    pub module: String,
    pub symbol: String,
    pub parameter_sizes: Vec<usize>,
    pub attributes: Vec<[i32; 2]>,
}

#[derive(Debug)]
pub(super) struct Reference {
    pub allocation: String,
    pub offset: usize,
}

#[derive(Debug)]
pub(super) enum RelocationKind {
    Pointer,
    TensorMap,
}

#[derive(Debug)]
pub(super) struct Relocation {
    pub offset: usize,
    pub target: Reference,
    pub kind: RelocationKind,
}

impl Relocation {
    pub fn width(&self) -> usize {
        match self.kind {
            RelocationKind::Pointer => 8,
            RelocationKind::TensorMap => 128,
        }
    }
}

#[derive(Debug)]
pub(super) struct Blob {
    pub data: Vec<u8>,
    pub relocations: Vec<Relocation>,
}

#[derive(Debug)]
pub(super) struct Attribute {
    pub id: i32,
    pub value: Blob,
}

#[derive(Debug)]
pub(super) enum Node {
    Kernel {
        function: usize,
        grid: [u32; 3],
        block: [u32; 3],
        shared: u32,
        arguments: Vec<Blob>,
        attributes: Vec<Attribute>,
    },
    Memset {
        destination: Reference,
        pitch: usize,
        value: u32,
        element_size: u32,
        width: usize,
        height: usize,
    },
    Copy {
        source: Reference,
        destination: Reference,
        nbytes: usize,
    },
    Empty,
    EventWait {
        event: String,
    },
}

#[derive(Debug)]
pub(super) struct Edge {
    pub source: usize,
    pub destination: usize,
    pub data: [u8; 8],
}

#[derive(Debug)]
pub(super) struct Graph {
    pub name: String,
    pub flags: u64,
    pub resource: String,
    pub node_count: usize,
    pub edge_count: usize,
    pub functions: Vec<usize>,
}

pub(super) struct Dependencies {
    seen: HashSet<(usize, usize)>,
    indegree: Vec<usize>,
    successors: Vec<Vec<usize>>,
}
impl Dependencies {
    pub fn new(nodes: usize) -> Self {
        Self {
            seen: HashSet::new(),
            indegree: vec![0; nodes],
            successors: vec![vec![]; nodes],
        }
    }
    pub fn add(&mut self, edge: &Edge) -> Result<()> {
        if edge.source >= self.indegree.len()
            || edge.destination >= self.indegree.len()
            || edge.data[0] > 2
            || edge.data[1] != 0
            || edge.data[2] > 1
            || edge.data[3..].iter().any(|v| *v != 0)
            || !self.seen.insert((edge.source, edge.destination))
        {
            return Err(invalid("invalid graph dependency"));
        }
        self.indegree[edge.destination] += 1;
        self.successors[edge.source].push(edge.destination);
        Ok(())
    }
    pub fn finish(mut self) -> Result<()> {
        let mut ready: Vec<_> = self
            .indegree
            .iter()
            .enumerate()
            .filter_map(|(i, n)| (*n == 0).then_some(i))
            .collect();
        let mut visited = 0;
        while let Some(node) = ready.pop() {
            visited += 1;
            for &next in &self.successors[node] {
                self.indegree[next] -= 1;
                if self.indegree[next] == 0 {
                    ready.push(next);
                }
            }
        }
        if visited != self.indegree.len() {
            return Err(invalid("graph dependency cycle"));
        }
        Ok(())
    }
}

impl Core {
    pub fn validate(&self) -> Result<()> {
        if self.events.iter().any(String::is_empty)
            || self.events.iter().collect::<HashSet<_>>().len() != self.events.len()
        {
            return Err(invalid("external event names must be nonempty and unique"));
        }
        if self.allocations.is_empty() || self.graphs.is_empty() {
            return Err(invalid("unsupported or empty runtime core"));
        }
        if self.context.limits.iter().any(|limit| limit[0] > 6)
            || !(0..=3).contains(&self.context.cache)
        {
            return Err(invalid("unsupported CUDA context configuration"));
        }
        let mut allocations = HashMap::new();
        let mut addresses = Vec::new();
        let mut reservations = Vec::new();
        for r in &self.reservations {
            if r.address == 0 || r.nbytes == 0 || r.address.checked_add(r.nbytes as u64).is_none() {
                return Err(invalid("invalid device address reservation"));
            }
            reservations.push((r.address, r.address + r.nbytes as u64));
        }
        reservations.sort_unstable();
        if reservations.windows(2).any(|r| r[0].1 > r[1].0) {
            return Err(invalid("device address reservations overlap"));
        }
        for a in &self.allocations {
            if let Some(s) = &a.sizing {
                if a.nbytes != 0
                    || a.address.is_none()
                    || !matches!(a.memory, Memory::Device)
                    || a.file.is_some()
                    || !a.data_spans.is_empty()
                    || !a.relocations.is_empty()
                    || ![0, 1].contains(&s.capacity)
                    || s.unit_bytes == 0
                    || s.maximum == 0
                    || s.maximum > i32::MAX as u64
                    || !a.alignment.is_power_of_two()
                    || s.unit_bytes
                        .checked_mul(s.maximum)
                        .and_then(|n| n.checked_add(a.alignment as u64 - 1))
                        .is_none_or(|n| n > usize::MAX as u64)
                {
                    return Err(invalid("invalid runtime-sized allocation"));
                }
            }
            let extent = a.extent();
            let mut end = 0;
            for [offset, size] in a.spans() {
                if size == 0
                    || offset < end
                    || offset.checked_add(size).is_none_or(|n| n > a.nbytes)
                {
                    return Err(invalid(
                        "allocation data spans must be ordered, disjoint, and in bounds",
                    ));
                }
                end = offset + size;
            }
            if a.is_payload_bound() {
                if a.file.is_some() || a.sources.len() != a.data_spans.len() {
                    return Err(invalid(
                        "payload-bound allocation must name one source per data span",
                    ));
                }
                for (source, span) in a.sources.iter().zip(&a.data_spans) {
                    if source.nbytes != span[1] as u64
                        || source.resource.is_empty()
                        || source
                            .offset
                            .checked_add(source.nbytes)
                            .is_none_or(|end| end > usize::MAX as u64)
                    {
                        return Err(invalid("allocation source does not match its data span"));
                    }
                }
                if a.stored_bytes() == 0 {
                    return Err(invalid("payload-bound allocation carries no stored bytes"));
                }
            } else if a.file.is_some() != (a.stored_bytes() > 0) {
                return Err(invalid("allocation payload file must match its data spans"));
            }
            if matches!(a.memory, Memory::Peer)
                && (a.address.is_none()
                    || a.file.is_some()
                    || a.is_payload_bound()
                    || !a.relocations.is_empty())
            {
                return Err(invalid(
                    "peer allocations require a recorded address and carry no payload",
                ));
            }
            if matches!(a.memory, Memory::MappedHost)
                && (a.nbytes % 4096 != 0 || a.address.is_none_or(|v| v % 4096 != 0))
            {
                return Err(invalid(
                    "mapped host allocations must cover complete host pages",
                ));
            }
            if a.name.is_empty()
                || extent == 0
                || !a.alignment.is_power_of_two()
                || a.nbytes.checked_add(a.alignment - 1).is_none()
                || allocations.insert(a.name.as_str(), extent).is_some()
            {
                return Err(invalid("invalid or duplicate runtime allocation"));
            }
            if let Some(address) = a.address {
                let end = address
                    .checked_add(extent as u64)
                    .ok_or_else(|| invalid("allocation address overflow"))?;
                if address == 0 || address % a.alignment as u64 != 0 {
                    return Err(invalid("invalid recorded allocation address"));
                }
                if matches!(a.memory, Memory::Device)
                    && !self
                        .reservations
                        .iter()
                        .any(|r| r.address <= address && end <= r.address + r.nbytes as u64)
                {
                    return Err(invalid(
                        "recorded device allocation lies outside its reservation",
                    ));
                }
                addresses.push((address, end));
            } else if matches!(a.memory, Memory::MappedHost) {
                return Err(invalid(
                    "mapped host allocation requires a recorded address",
                ));
            }
            // A tensor-map descriptor embeds a global address, so its 128 bytes must be stored
            // bytes rather than zero fill. The carrier is irrelevant: a legacy allocation file and
            // a payload-bound allocation both initialize through their data spans.
            let has_descriptor = a
                .relocations
                .iter()
                .any(|relocation| matches!(relocation.kind, RelocationKind::TensorMap));
            if has_descriptor {
                if a.file.is_none() && !a.is_payload_bound() {
                    return Err(invalid(
                        "tensor-map storage requires an initialized descriptor",
                    ));
                }
                if a.is_payload_bound() {
                    let mut spans: Vec<(usize, usize)> = a
                        .spans()
                        .map(|[offset, nbytes]| (offset, offset + nbytes))
                        .collect();
                    spans.sort_unstable();
                    let mut merged: Vec<(usize, usize)> = Vec::new();
                    for (start, end) in spans {
                        match merged.last_mut() {
                            Some(previous) if start <= previous.1 => {
                                previous.1 = previous.1.max(end)
                            }
                            _ => merged.push((start, end)),
                        }
                    }
                    for relocation in a
                        .relocations
                        .iter()
                        .filter(|relocation| matches!(relocation.kind, RelocationKind::TensorMap))
                    {
                        let end = relocation
                            .offset
                            .checked_add(128)
                            .ok_or_else(|| invalid("tensor-map relocation offset overflow"))?;
                        if !merged
                            .iter()
                            .any(|(start, stop)| *start <= relocation.offset && end <= *stop)
                        {
                            return Err(invalid(
                                "tensor-map descriptor bytes are not initialized by any source span",
                            ));
                        }
                    }
                }
            }
        }
        addresses.sort_unstable();
        if addresses.windows(2).any(|a| a[0].1 > a[1].0) {
            return Err(invalid("recorded allocation addresses overlap"));
        }
        for a in &self.allocations {
            self.validate_relocations(a.nbytes, &a.relocations)?;
        }
        let mut modules = HashSet::new();
        for module in &self.modules {
            if module.name.is_empty() || !modules.insert(&module.name) {
                return Err(invalid("invalid or duplicate module name"));
            }
            let mut globals = HashSet::new();
            for global in &module.globals {
                if global.symbol.is_empty()
                    || global.symbol.contains('\0')
                    || global.nbytes == 0
                    || !globals.insert(&global.symbol)
                {
                    return Err(invalid("invalid or duplicate module global"));
                }
                self.validate_relocations(global.nbytes, &global.relocations)?;
            }
        }
        for function in &self.functions {
            if !modules.contains(&function.module)
                || function.symbol.is_empty()
                || function.symbol.contains('\0')
                || function.parameter_sizes.contains(&0)
                || function
                    .attributes
                    .iter()
                    .any(|a| ![8, 9, 14, 15].contains(&a[0]))
            {
                return Err(invalid("invalid runtime kernel definition"));
            }
        }
        let mut names = HashSet::new();
        for graph in &self.graphs {
            if graph.name.is_empty()
                || !names.insert(&graph.name)
                || graph.node_count == 0
                || graph.flags & !9 != 0
            {
                return Err(invalid("invalid or duplicate graph name"));
            }
            if graph.resource.is_empty()
                || graph
                    .functions
                    .iter()
                    .any(|id| !self.functions.iter().any(|f| f.id == *id))
            {
                return Err(invalid("invalid graph resource or function dependency"));
            }
        }
        Ok(())
    }
    pub fn validate_reference(&self, r: &Reference, size: usize) -> Result<()> {
        if !self
            .allocations
            .iter()
            .find(|a| a.name == r.allocation)
            .is_some_and(|a| {
                r.offset
                    .checked_add(size)
                    .is_some_and(|end| end <= a.extent())
            })
        {
            return Err(invalid(format!(
                "reference exceeds allocation {:?}",
                r.allocation
            )));
        }
        Ok(())
    }
    pub fn validate_relocations(&self, size: usize, values: &[Relocation]) -> Result<()> {
        let mut fields: Vec<_> = values.iter().collect();
        fields.sort_by_key(|r| r.offset);
        let mut end = 0;
        for r in fields {
            if r.offset < end || r.offset.checked_add(r.width()).is_none_or(|v| v > size) {
                return Err(invalid("invalid or overlapping relocation fields"));
            }
            self.validate_reference(&r.target, 1)?;
            end = r.offset + r.width();
        }
        Ok(())
    }
    pub fn validate_node(&self, node: &Node) -> Result<()> {
        match node {
            Node::Kernel {
                function,
                grid,
                block,
                arguments,
                attributes,
                ..
            } => {
                let f = self
                    .functions
                    .iter()
                    .find(|f| f.id == *function)
                    .ok_or_else(|| invalid("unknown kernel"))?;
                if grid.contains(&0)
                    || block.contains(&0)
                    || arguments
                        .iter()
                        .map(|a| a.data.len())
                        .ne(f.parameter_sizes.iter().copied())
                {
                    return Err(invalid(
                        "kernel argument layout or launch dimensions are invalid",
                    ));
                }
                for a in arguments {
                    self.validate_relocations(a.data.len(), &a.relocations)?;
                }
                let mut ids = HashSet::new();
                for a in attributes {
                    if ![1, 2, 4, 5, 8, 9, 10, 11, 14].contains(&a.id)
                        || a.value.data.len() != 64
                        || !ids.insert(a.id)
                    {
                        return Err(invalid("unsupported or duplicate kernel-node attribute"));
                    }
                    self.validate_relocations(64, &a.value.relocations)?;
                }
            }
            Node::Memset {
                destination,
                pitch,
                element_size,
                width,
                height,
                ..
            } => {
                let row = width
                    .checked_mul(*element_size as usize)
                    .ok_or_else(|| invalid("memset extent overflow"))?;
                if ![1, 2, 4].contains(element_size)
                    || row == 0
                    || *height == 0
                    || (*height > 1 && *pitch < row)
                {
                    return Err(invalid("invalid memset dimensions"));
                }
                let extent = (height - 1)
                    .checked_mul(*pitch)
                    .and_then(|v| v.checked_add(row))
                    .ok_or_else(|| invalid("memset extent overflow"))?;
                self.validate_reference(destination, extent)?;
            }
            Node::Copy {
                source,
                destination,
                nbytes,
            } => {
                if *nbytes == 0 {
                    return Err(invalid("empty graph copy"));
                }
                self.validate_reference(source, *nbytes)?;
                self.validate_reference(destination, *nbytes)?;
            }
            Node::Empty => {}
            Node::EventWait { event } => {
                if !self.events.contains(event) {
                    return Err(invalid("graph waits on an undeclared external event"));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn core() -> Core {
        Core {
            resources: HashMap::new(),
            events: vec![],
            architecture: [12, 0],
            allocations: vec![Allocation {
                name: "state".into(),
                nbytes: 4096,
                alignment: 256,
                file: Some("state.bin".into()),
                data_spans: vec![[0, 4096]],
                sources: vec![],
                relocations: vec![],
                address: None,
                memory: Memory::Device,
                sizing: None,
            }],
            modules: vec![],
            functions: vec![],
            reservations: vec![],
            execution: None,
            context: Context::default(),
            graphs: vec![Graph {
                name: "copy".into(),
                flags: 0,
                resource: "copy".into(),
                node_count: 2,
                edge_count: 0,
                functions: vec![],
            }],
        }
    }

    #[test]
    fn execution_abi_labels_do_not_reject_core_resources() {
        for abi in [0, 1, 5, 6, 7, u32::MAX] {
            let mut value = core();
            value.execution = Some(super::super::protobuf::wire::Execution {
                abi_version: abi, ..Default::default()
            });
            value.validate().unwrap();
            value.allocations.clear();
            assert!(value.validate().is_err());
        }
    }

    #[test]
    fn streamed_dependencies_reject_cycles_duplicates_and_invalid_nodes() {
        let edge = |source, destination| Edge {
            source,
            destination,
            data: [0; 8],
        };
        let mut dependencies = Dependencies::new(4);
        for (source, destination) in [(0, 1), (0, 2), (1, 3), (2, 3)] {
            dependencies.add(&edge(source, destination)).unwrap();
        }
        dependencies.finish().unwrap();

        let mut dependencies = Dependencies::new(4);
        for (source, destination) in [(0, 1), (1, 2), (2, 1)] {
            dependencies.add(&edge(source, destination)).unwrap();
        }
        assert!(
            dependencies
                .finish()
                .unwrap_err()
                .to_string()
                .contains("cycle")
        );

        let mut dependencies = Dependencies::new(2);
        dependencies.add(&edge(0, 1)).unwrap();
        assert!(dependencies.add(&edge(0, 1)).is_err());
        assert!(dependencies.add(&edge(0, 2)).is_err());
    }

    #[test]
    fn recorded_device_spans_require_a_containing_reservation() {
        let mut value = core();
        value.allocations[0].address = Some(0x200000000000);
        assert!(value.validate().is_err());
        value.reservations.push(Reservation {
            address: 0x200000000000,
            nbytes: 8192,
        });
        assert!(value.validate().is_ok());
        value.allocations[0].nbytes = 8193;
        assert!(value.validate().is_err());
    }

    #[test]
    fn runtime_sizing_reserves_its_maximum_without_a_payload() {
        use super::super::protobuf::wire::Sizing;

        let mut value = core();
        let a = &mut value.allocations[0];
        a.address = Some(0x200000000000);
        a.nbytes = 0;
        a.file = None;
        a.data_spans.clear();
        a.sizing = Some(Sizing {
            capacity: 0,
            unit_bytes: 300,
            maximum: 100,
        });
        assert_eq!(a.extent(), 30208);
        value.reservations.push(Reservation {
            address: 0x200000000000,
            nbytes: 30208,
        });
        value.validate().unwrap();
        value.reservations[0].nbytes -= 1;
        assert!(value.validate().is_err());
        value.reservations[0].nbytes += 1;
        value.allocations[0].nbytes = 256;
        assert!(value.validate().is_err());
        value.allocations[0].nbytes = 0;
        value.allocations[0].sizing.as_mut().unwrap().unit_bytes = u64::MAX;
        assert!(value.validate().is_err());
    }

    #[test]
    fn packed_payload_spans_are_bounded_and_match_the_file() {
        let mut value = core();
        for spans in [
            vec![[0, 0]],
            vec![[4095, 2]],
            vec![[16, 32], [32, 16]],
            vec![],
        ] {
            value.allocations[0].data_spans = spans;
            assert!(value.validate().is_err());
        }
        value.allocations[0].data_spans = vec![[16, 32], [128, 64]];
        assert!(value.validate().is_ok());
        assert_eq!(value.allocations[0].stored_bytes(), 96);
        value.allocations[0].file = None;
        assert!(value.validate().is_err());
        value.allocations[0].data_spans.clear();
        assert!(value.validate().is_ok());
    }

    #[test]
    fn tensor_map_descriptors_must_be_initialized_by_source_spans() {
        let mut value = core();
        let relocation = |offset: usize| Relocation {
            offset,
            kind: RelocationKind::TensorMap,
            target: Reference {
                allocation: "state".into(),
                offset: 0,
            },
        };
        // Payload-bound storage: the descriptor must be covered by stored spans.
        value.allocations[0].file = None;
        value.allocations[0].data_spans = vec![[0, 128], [128, 128]];
        value.allocations[0].sources = vec![
            Source {
                resource: "a".into(),
                offset: 0,
                nbytes: 128,
            },
            Source {
                resource: "b".into(),
                offset: 0,
                nbytes: 128,
            },
        ];
        value.allocations[0].relocations = vec![relocation(0)];
        assert!(value.validate().is_ok());
        value.allocations[0].relocations = vec![relocation(64)];
        // Contiguous spans still cover the descriptor as one initialized range.
        assert!(value.validate().is_ok());
        value.allocations[0].data_spans = vec![[0, 128], [256, 128]];
        value.allocations[0].sources = vec![
            Source {
                resource: "a".into(),
                offset: 0,
                nbytes: 128,
            },
            Source {
                resource: "b".into(),
                offset: 0,
                nbytes: 128,
            },
        ];
        value.allocations[0].relocations = vec![relocation(128)];
        assert!(
            value.validate().is_err(),
            "a gap must not initialize a descriptor"
        );
        // Legacy carrier keeps its original rule.
        let mut legacy = core();
        legacy.allocations[0].relocations = vec![relocation(0)];
        assert!(legacy.validate().is_ok());
        legacy.allocations[0].file = None;
        legacy.allocations[0].data_spans.clear();
        assert!(
            legacy.validate().is_err(),
            "an uninitialized descriptor is rejected"
        );
    }

    #[test]
    fn pointer_fields_cannot_overlap_or_reference_an_allocation_end() {
        let mut value = core();
        let relocation = || Relocation {
            offset: 0,
            kind: RelocationKind::Pointer,
            target: Reference {
                allocation: "state".into(),
                offset: 4095,
            },
        };
        value.allocations[0].relocations.push(relocation());
        assert!(value.validate().is_ok());
        value.allocations[0].relocations[0].target.offset = 4096;
        assert!(value.validate().is_err());
        value.allocations[0].relocations = vec![relocation(), relocation()];
        assert!(value.validate().is_err());
    }

    #[test]
    fn mapped_host_extents_cover_complete_pages() {
        let mut value = core();
        value.allocations[0].memory = Memory::MappedHost;
        value.allocations[0].address = Some(0x100000000000);
        assert!(value.validate().is_ok());
        value.allocations[0].nbytes = 4000;
        value.allocations[0].data_spans = vec![[0, 4000]];
        assert!(value.validate().is_err());
    }
}
