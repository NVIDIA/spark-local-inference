/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed payload-package schema and bounded semantic validation.
//!
//! The generated Protobuf types describe signed payload identity, rank placement, initialization,
//! and execution geometry. Physical container placement is the separate, unsigned
//! [`wire::LocationSet`]. Every validator here is bounded: it checks sizes, identifier shapes,
//! arithmetic, and cross-references before any consumer allocates from the data.

use super::invalid;
use crate::Result;
use crate::runtime_package::wire as runtime;
use std::collections::{BTreeMap, BTreeSet};

pub mod wire {
    include!(concat!(env!("OUT_DIR"), "/flare.package.rs"));
}

/// Envelope and body format version this build understands.
pub const FORMAT_VERSION: u32 = 2;
/// Bound on one serialized signed body, matching the metadata tensor limit.
pub const MAX_BODY: usize = 128 << 20;
/// Bound on the complete serialized envelope.
pub const MAX_ENVELOPE: usize = 128 << 20;
/// Bound on reconstructed rank metadata selected by one verification.
pub const MAX_SELECTED_RANK_METADATA: u64 = 512 << 20;
/// Executable compressed chunk size. Authentication covers chunks of exactly this size.
pub const MAX_CHUNK: usize = 4 << 20;
/// Bound on compressed chunks for one executable resource (4 TiB at the fixed chunk size).
/// Bulk executable storage is not subject to the metadata limit; only this structural bound and
/// checked arithmetic apply.
pub const MAX_EXECUTABLE_CHUNKS: u64 = 1 << 20;
/// Bound on payload descriptors per catalog.
pub const MAX_DESCRIPTORS: usize = 1 << 20;
/// Bound on physical locations.
pub const MAX_LOCATIONS: usize = 1 << 20;
/// Bound on outer axes in one piece; the innermost run is `contiguous_bytes`.
pub const MAX_PIECE_AXES: usize = 6;
/// Bound on declared logical source axes. Source tensors and flattened prepared
/// owners have independent coordinate domains, so this is not the owner rank.
pub const MAX_SOURCE_AXES: usize = 8;


fn bounded_name(value: &str, limit: usize, what: &str) -> Result<()> {
    if value.is_empty() || value.len() > limit || value.chars().any(char::is_control) {
        return Err(invalid(format!("invalid {what}")));
    }
    Ok(())
}

// Mangled CUDA symbols use the enclosing signed metadata body's size bound.
fn symbol_name(value: &str, what: &str) -> Result<()> {
    if value.is_empty() || value.chars().any(char::is_control) {
        return Err(invalid(format!("invalid {what}")));
    }
    Ok(())
}

fn digest(id: &[u8], what: &str) -> Result<()> {
    if id.len() != 32 {
        return Err(invalid(format!("{what} must be a 32-byte SHA-256 digest")));
    }
    Ok(())
}

/// Payload lengths authenticated by the package, deployment, and rank catalogs.
///
/// A digest identifies bytes; the catalog records how many. Two descriptors for one digest must
/// agree on length, and every slice is checked against the recorded length before use.
#[derive(Clone, Debug, Default)]
pub struct Catalog {
    lengths: BTreeMap<Vec<u8>, u64>,
}

impl Catalog {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn len(&self) -> usize {
        self.lengths.len()
    }
    pub fn is_empty(&self) -> bool {
        self.lengths.is_empty()
    }
    pub fn insert(&mut self, descriptor: &wire::PayloadDescriptor) -> Result<()> {
        digest(&descriptor.id, "payload id")?;
        if descriptor.nbytes == 0 {
            return Err(invalid("payload descriptor has zero length"));
        }
        if self.lengths.len() >= MAX_DESCRIPTORS {
            return Err(invalid("too many payload descriptors"));
        }
        match self.lengths.get(&descriptor.id) {
            Some(&previous) if previous != descriptor.nbytes => {
                Err(invalid("conflicting payload lengths for one digest"))
            }
            _ => {
                self.lengths
                    .insert(descriptor.id.clone(), descriptor.nbytes);
                Ok(())
            }
        }
    }
    pub fn insert_all<'a>(
        &mut self,
        descriptors: impl IntoIterator<Item = &'a wire::PayloadDescriptor>,
    ) -> Result<()> {
        for descriptor in descriptors {
            self.insert(descriptor)?;
        }
        Ok(())
    }
    /// Every authenticated payload id in catalog order.
    pub fn ids(&self) -> Vec<&[u8]> {
        self.lengths.keys().map(|id| id.as_slice()).collect()
    }
    pub fn length(&self, id: &[u8]) -> Result<u64> {
        self.lengths
            .get(id)
            .copied()
            .ok_or_else(|| invalid("slice references an absent payload descriptor"))
    }
    pub fn check_slice(&self, slice: &wire::PayloadSlice) -> Result<u64> {
        let length = self.length(slice.payload_id.as_slice())?;
        let end = slice
            .offset
            .checked_add(slice.nbytes)
            .ok_or_else(|| invalid("payload slice overflow"))?;
        if slice.nbytes == 0 || end > length {
            return Err(invalid("payload slice exceeds its payload"));
        }
        Ok(length)
    }
    /// Validate one authenticated use and the role permitted for it.
    pub fn check_use(
        &self,
        value: &wire::PayloadUse,
        permitted: &[wire::PayloadRole],
    ) -> Result<u64> {
        let slice = value
            .slice
            .as_ref()
            .ok_or_else(|| invalid("payload use has no slice"))?;
        let role = role_of(value)?;
        if !permitted.is_empty() && !permitted.contains(&role) {
            return Err(invalid("payload use role is not permitted here"));
        }
        self.check_slice(slice)
    }
}

/// The role of one authenticated use, rejecting the unspecified value.
pub fn size(value: u64) -> Result<usize> {
    usize::try_from(value).map_err(|_| invalid("size exceeds the host address space"))
}

/// The role of one authenticated use, rejecting the unspecified value.
pub fn role_of(value: &wire::PayloadUse) -> Result<wire::PayloadRole> {
    let role =
        wire::PayloadRole::try_from(value.role).map_err(|_| invalid("unknown payload role"))?;
    if role == wire::PayloadRole::Unspecified {
        return Err(invalid("payload use has no role"));
    }
    Ok(role)
}

/// Validate publisher identity fields in a signed package body or an unverified claim.
pub fn validate_publisher_metadata(publisher: &wire::PublisherMetadata) -> Result<()> {
    let namespace = publisher.namespace.as_str();
    if namespace.is_empty()
        || namespace.len() > 96
        || !namespace.as_bytes()[0].is_ascii_alphanumeric()
        || !namespace
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err(invalid("invalid publisher namespace"));
    }
    let source = publisher.official_source.as_str();
    if publisher.development {
        if namespace != "local-development" || source != "local-development" {
            return Err(invalid("invalid development publisher identity"));
        }
    } else {
        let prefix = format!("https://huggingface.co/{namespace}/");
        let Some(repository) = source.strip_prefix(&prefix) else {
            return Err(invalid(
                "official source is not under the publisher namespace",
            ));
        };
        if repository.is_empty()
            || repository.len() > 96
            || !repository.as_bytes()[0].is_ascii_alphanumeric()
            || !repository
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
        {
            return Err(invalid("invalid official source repository"));
        }
    }
    let name = publisher.license_name.as_str();
    let link = publisher.license_link.as_str();
    if !name.is_empty() && (name.len() > 128 || name.chars().any(char::is_control)) {
        return Err(invalid("invalid license name"));
    }
    if name.is_empty() && !link.is_empty() {
        return Err(invalid("license link requires a license name"));
    }
    if !link.is_empty()
        && (link.len() > 512
            || !link.starts_with("https://")
            || !link.bytes().all(|b| b.is_ascii_graphic()))
    {
        return Err(invalid("invalid license link"));
    }
    Ok(())
}

/// Validate the shared body and build the shared payload catalog.
pub fn validate_package(body: &wire::PackageBody) -> Result<Catalog> {
    if body.format_version != FORMAT_VERSION {
        return Err(invalid("unsupported package format version"));
    }
    digest(&body.package_id, "package id")?;
    bounded_name(&body.model_identity, 256, "model identity")?;
    bounded_name(&body.hardware_identity, 256, "hardware identity")?;
    validate_publisher_metadata(
        body.publisher
            .as_ref()
            .ok_or_else(|| invalid("package has no publisher metadata"))?,
    )?;
    if body.frontend.is_none() {
        return Err(invalid("package has no frontend settings"));
    }
    if body.sampling_defaults.is_none() {
        return Err(invalid("package has no sampling defaults"));
    }
    let mut catalog = Catalog::new();
    catalog.insert_all(&body.payloads)?;
    let mut names = BTreeSet::new();
    for asset in &body.assets {
        bounded_name(&asset.name, 128, "asset name")?;
        if !names.insert(asset.name.as_str()) {
            return Err(invalid("duplicate frontend asset name"));
        }
        let payload = asset
            .payload
            .as_ref()
            .ok_or_else(|| invalid("frontend asset has no payload"))?;
        let length = catalog.check_use(payload, &[wire::PayloadRole::Frontend])?;
        digest(&asset.sha256, "asset digest")?;
        if payload
            .slice
            .as_ref()
            .is_some_and(|slice| slice.nbytes != length)
        {
            return Err(invalid("frontend asset must use its whole payload"));
        }
    }
    if body.deployments.is_empty() {
        return Err(invalid("package has no deployments"));
    }
    let mut ids = BTreeSet::new();
    for commitment in &body.deployments {
        bounded_name(&commitment.deployment_id, 128, "deployment id")?;
        if !ids.insert(commitment.deployment_id.as_str()) {
            return Err(invalid("duplicate deployment id"));
        }
        digest(&commitment.body_digest, "deployment commitment")?;
    }
    Ok(catalog)
}

/// Validate one deployment body and extend the catalog with its payloads.
pub fn validate_deployment(body: &wire::DeploymentBody, catalog: &mut Catalog) -> Result<u32> {
    bounded_name(&body.deployment_id, 128, "deployment id")?;
    // Compute capability is (major, minor); a minor of zero is the normal packaged form.
    if body.architecture.len() != 2
        || body.architecture[0] <= 0
        || body.architecture[1] < 0
        || body.architecture.iter().any(|&value| value > 1024)
    {
        return Err(invalid(
            "deployment architecture must name a compute capability",
        ));
    }
    if body.contract.is_none() {
        return Err(invalid("deployment has no serving contract"));
    }
    let ranks = u32::try_from(body.ranks.len()).map_err(|_| invalid("too many ranks"))?;
    if ranks == 0 {
        return Err(invalid("deployment has no ranks"));
    }
    if body.tensor_parallel != 0 && body.tensor_parallel != ranks {
        return Err(invalid("tensor-parallel size differs from the rank count"));
    }
    for (position, commitment) in body.ranks.iter().enumerate() {
        if commitment.rank as usize != position {
            return Err(invalid("rank commitments must be ordered from zero"));
        }
        digest(&commitment.body_digest, "rank commitment")?;
    }
    catalog.insert_all(&body.payloads)?;
    if let Some(engram) = &body.engram {
        validate_engram(engram, ranks, catalog)?;
    }
    Ok(ranks)
}

fn validate_engram(engram: &wire::Engram, ranks: u32, catalog: &Catalog) -> Result<()> {
    if engram.tables.is_empty() {
        return Err(invalid("engram catalog has no tables"));
    }
    if engram.compressed_vocabulary == 0 {
        return Err(invalid("engram catalog has no compressed vocabulary"));
    }
    let token_map_empty = engram.token_map.is_empty();
    if token_map_empty {
        if engram.pad_token_id >= engram.compressed_vocabulary {
            return Err(invalid("engram pad token is outside the compressed vocabulary"));
        }
    } else {
        if engram.pad_token_id >= engram.token_map.len() as u32
            || engram
                .token_map
                .iter()
                .any(|token| *token >= engram.compressed_vocabulary)
        {
            return Err(invalid("engram token map is outside the compressed vocabulary"));
        }
    }
    let roles = [wire::PayloadRole::Weights, wire::PayloadRole::TunedWeights];
    for table in &engram.tables {
        if wire::engram_table::Encoding::try_from(table.encoding).map_or(true, |value| {
            value == wire::engram_table::Encoding::Unspecified
        }) {
            return Err(invalid("engram table has no encoding"));
        }
        if table.rows == 0 || table.embedding_width == 0 {
            return Err(invalid("engram table has no rows or width"));
        }
        bounded_name(&table.weight_buffer, 64, "engram weight buffer")?;
        bounded_name(&table.scale_buffer, 64, "engram scale buffer")?;
        let hash = table
            .hash
            .as_ref()
            .ok_or_else(|| invalid("engram table has no hash description"))?;
        let method = wire::engram_hash::Method::try_from(hash.method)
            .map_err(|_| invalid("engram hash has no method"))?;
        if method == wire::engram_hash::Method::Unspecified {
            return Err(invalid("engram hash has no method"));
        }
        if token_map_empty {
            if method != wire::engram_hash::Method::EosXorModulo
                || table.request_buffer.is_empty()
            {
                return Err(invalid(
                    "engram without a token map requires EOS XOR hash and request delivery",
                ));
            }
        } else if method != wire::engram_hash::Method::XorModulo {
            return Err(invalid(
                "engram with a token map requires XOR-modulo hashing",
            ));
        }
        if !hash.orders.is_empty()
            && (hash.orders.len() != hash.primes.len() || hash.orders.len() != hash.offsets.len())
        {
            return Err(invalid("engram hash head arrays disagree"));
        }
        if table.shards.is_empty() {
            for (plane, label) in [(&table.weights, "weights"), (&table.scales, "scales")] {
                let plane = plane
                    .as_ref()
                    .ok_or_else(|| invalid(format!("engram table has no {label} plane")))?;
                catalog.check_use(plane, &roles)?;
            }
        } else {
            if table.weights.is_some() || table.scales.is_some() {
                return Err(invalid("partitioned table carries shared planes too"));
            }
            let mut end = 0u64;
            for shard in &table.shards {
                if shard.rank >= ranks {
                    return Err(invalid("engram partition names an absent rank"));
                }
                if shard.rows == 0 || shard.first_row < end {
                    return Err(invalid("engram partitions overlap or are empty"));
                }
                end = shard
                    .first_row
                    .checked_add(shard.rows)
                    .ok_or_else(|| invalid("engram partition overflow"))?;
                if end > table.rows {
                    return Err(invalid("engram partition exceeds table rows"));
                }
                for (plane, label) in [(&shard.weights, "weights"), (&shard.scales, "scales")] {
                    let plane = plane
                        .as_ref()
                        .ok_or_else(|| invalid(format!("engram partition has no {label} plane")))?;
                    catalog.check_use(plane, &roles)?;
                }
            }
        }
    }
    Ok(())
}

fn allocation_extent(allocation: &wire::Allocation) -> Result<u64> {
    match &allocation.sizing {
        Some(sizing) => {
            if allocation.nbytes != 0
                || allocation.address.is_none()
                || !allocation.relocations.is_empty()
                || allocation.memory != runtime::Memory::Device as i32
                || !matches!(sizing.capacity, 0 | 1)
                || sizing.unit_bytes == 0
                || sizing.maximum == 0
            {
                return Err(invalid("invalid runtime-sized allocation"));
            }
            let bytes = sizing
                .unit_bytes
                .checked_mul(sizing.maximum)
                .ok_or_else(|| invalid("runtime-sized allocation overflow"))?;
            bytes
                .div_ceil(allocation.alignment)
                .checked_mul(allocation.alignment)
                .ok_or_else(|| invalid("runtime-sized allocation extent overflow"))
        }
        None => Ok(allocation.nbytes),
    }
}

/// Validate one rank body against the authenticated catalog and package-wide resource ids.
pub fn validate_rank(
    body: &wire::RankBody,
    ranks: u32,
    catalog: &mut Catalog,
    resource_ids: &mut BTreeSet<u64>,
) -> Result<()> {
    bounded_name(&body.deployment_id, 128, "deployment id")?;
    catalog.insert_all(&body.payloads)?;
    if body.rank >= ranks {
        return Err(invalid("rank id is outside its deployment"));
    }
    if body.context.is_none() {
        return Err(invalid("rank has no CUDA context description"));
    }
    let mut extents = BTreeMap::new();
    for allocation in &body.allocations {
        bounded_name(&allocation.name, 128, "allocation name")?;
        if !allocation.alignment.is_power_of_two() {
            return Err(invalid("allocation alignment must be a power of two"));
        }
        let extent = allocation_extent(allocation)?;
        if extent == 0 {
            return Err(invalid("allocation has no extent"));
        }
        if let Some(address) = allocation.address {
            if address == 0 || address % allocation.alignment != 0 {
                return Err(invalid("allocation address is unaligned"));
            }
        } else if allocation.memory == runtime::Memory::MappedHost as i32 {
            return Err(invalid("mapped host allocation has no address"));
        }
        if allocation.memory == runtime::Memory::MappedHost as i32
            && (extent % 4096 != 0 || allocation.address.is_none_or(|v| v % 4096 != 0))
        {
            return Err(invalid("mapped host allocation must cover complete pages"));
        }
        if allocation.memory == runtime::Memory::Peer as i32 && allocation.address.is_none() {
            return Err(invalid("peer allocation has no address"));
        }
        if extents.insert(allocation.name.as_str(), extent).is_some() {
            return Err(invalid("duplicate allocation name"));
        }
    }
    for (name, &extent) in &extents {
        if let Some(allocation) = body.allocations.iter().find(|a| a.name == *name) {
            for relocation in &allocation.relocations {
                let target = relocation
                    .target
                    .as_ref()
                    .ok_or_else(|| invalid("relocation has no target"))?;
                let target_extent = extents
                    .get(target.allocation.as_str())
                    .ok_or_else(|| invalid("relocation names an absent allocation"))?;
                if target.offset >= *target_extent {
                    return Err(invalid("relocation target is outside its allocation"));
                }
                let width = match runtime::RelocationKind::try_from(relocation.kind) {
                    Ok(runtime::RelocationKind::Pointer) => 8,
                    Ok(runtime::RelocationKind::TensorMap) => 128,
                    Err(_) => return Err(invalid("unknown relocation kind")),
                };
                if relocation
                    .offset
                    .checked_add(width)
                    .is_none_or(|end| end > extent)
                {
                    return Err(invalid("relocation is outside its allocation"));
                }
            }
        }
    }
    validate_initializers(body, &extents, catalog)?;

    for (index, resource) in body.resources.iter().enumerate() {
        validate_resource(resource, catalog)?;
        if !resource_ids.insert(resource.resource_id) {
            return Err(invalid("duplicate executable resource id"));
        }
        let _ = index;
    }

    let mut module_names = BTreeSet::new();
    for module in &body.modules {
        bounded_name(&module.name, 128, "module name")?;
        if !module_names.insert(module.name.as_str()) {
            return Err(invalid("duplicate module name"));
        }
        if body
            .resources
            .get(module.cubin as usize)
            .is_none_or(|resource| resource.kind != wire::ExecutableKind::Cubin as i32)
        {
            return Err(invalid("module cubin reference is not a cubin resource"));
        }
        let mut symbols = BTreeSet::new();
        for global in &module.globals {
            symbol_name(&global.symbol, "module symbol")?;
            if !symbols.insert(global.symbol.as_str()) {
                return Err(invalid("duplicate module global symbol"));
            }
            let contents = global
                .contents
                .as_ref()
                .ok_or_else(|| invalid("module global has no contents"))?;
            catalog.check_use(contents, &[wire::PayloadRole::InitialState])?;
            // Globals may be slices of a combined initial-state payload, so the length that must
            // agree is this use's slice, not the whole payload.
            let length = contents.slice.as_ref().map_or(0, |slice| slice.nbytes);
            if global.nbytes == 0 || length != global.nbytes {
                return Err(invalid(
                    "module global length differs from its payload slice",
                ));
            }
        }
    }
    for function in &body.functions {
        if !module_names.contains(function.module.as_str()) {
            return Err(invalid("function names an absent module"));
        }
        symbol_name(&function.symbol, "function symbol")?;
        if function.parameter_sizes.len() > 128 {
            return Err(invalid("function declares too many parameters"));
        }
    }
    let mut graph_names = BTreeSet::new();
    for graph in &body.graphs {
        bounded_name(&graph.name, 128, "graph name")?;
        if !graph_names.insert(graph.name.as_str()) {
            return Err(invalid("duplicate graph name"));
        }
        if body
            .resources
            .get(graph.resource as usize)
            .is_none_or(|resource| resource.kind != wire::ExecutableKind::Graph as i32)
        {
            return Err(invalid("graph resource reference is not a graph resource"));
        }
        if graph.node_count == 0 {
            return Err(invalid("graph has no nodes"));
        }
        for function in &graph.functions {
            if (*function as usize) >= body.functions.len() {
                return Err(invalid("graph names an absent function"));
            }
        }
    }
    if body.graphs.is_empty() || body.allocations.is_empty() {
        return Err(invalid("rank has no graphs or allocations"));
    }
    let mut events = BTreeSet::new();
    for event in &body.events {
        bounded_name(event, 128, "event name")?;
        if !events.insert(event.as_str()) {
            return Err(invalid("duplicate event name"));
        }
    }
    let mut peer_addresses = BTreeSet::new();
    for peer in &body.peers {
        if peer.owner >= ranks
            || peer.nbytes == 0
            || peer.address == 0
            || !peer_addresses.insert(peer.address)
        {
            return Err(invalid("invalid peer memory record"));
        }
        let mut readers = BTreeSet::new();
        if peer.readers.is_empty()
            || !peer.readers.contains(&peer.owner)
            || peer
                .readers
                .iter()
                .any(|&reader| reader >= ranks || !readers.insert(reader))
        {
            return Err(invalid("invalid peer memory reader"));
        }
    }
    super::prepared_weights::validate_wrapper(body)?;
    if let Some(weights) = &body.prepared_weights {
        validate_rank_prepared_weights(body, weights)?;
    }
    Ok(())
}

/// Validate an inline catalog or a catalog decoded from authenticated rank metadata.
pub fn validate_rank_prepared_weights(
    body: &wire::RankBody,
    weights: &wire::PreparedWeightCatalog,
) -> Result<()> {
    let ordered: Vec<(&str, u64)> = body.allocations.iter()
        .map(|allocation| Ok((allocation.name.as_str(), allocation_extent(allocation)?)))
        .collect::<Result<_>>()?;
    validate_prepared_weights(weights, &ordered)
}

fn validate_initializers(
    body: &wire::RankBody,
    extents: &BTreeMap<&str, u64>,
    catalog: &Catalog,
) -> Result<()> {
    let roles = [
        wire::PayloadRole::Weights,
        wire::PayloadRole::TunedWeights,
        wire::PayloadRole::InitialState,
    ];
    let mut ranges: BTreeMap<&str, Vec<(u64, u64)>> = BTreeMap::new();
    for initializer in &body.initializers {
        let extent = extents
            .get(initializer.allocation.as_str())
            .ok_or_else(|| invalid("initializer names an absent allocation"))?;
        let end = initializer
            .destination_offset
            .checked_add(
                initializer
                    .source
                    .as_ref()
                    .and_then(|source| source.slice.as_ref())
                    .map_or(0, |slice| slice.nbytes),
            )
            .ok_or_else(|| invalid("initializer destination overflow"))?;
        if end > *extent || end == initializer.destination_offset {
            return Err(invalid("initializer destination is outside its allocation"));
        }
        let source = initializer
            .source
            .as_ref()
            .ok_or_else(|| invalid("initializer has no source"))?;
        catalog.check_use(source, &roles)?;
        ranges
            .entry(initializer.allocation.as_str())
            .or_default()
            .push((initializer.destination_offset, end));
    }
    for (name, mut spans) in ranges {
        spans.sort_unstable();
        if spans.windows(2).any(|pair| pair[1].0 < pair[0].1) {
            return Err(invalid(format!("initializers for {name} overlap")));
        }
    }
    Ok(())
}

fn validate_resource(resource: &wire::ExecutableResource, catalog: &Catalog) -> Result<()> {
    if resource.resource_id == 0 {
        return Err(invalid("executable resource id must be assigned"));
    }
    let kind = wire::ExecutableKind::try_from(resource.kind)
        .map_err(|_| invalid("unknown executable resource kind"))?;
    let role = match kind {
        wire::ExecutableKind::Cubin => wire::PayloadRole::Kernels,
        wire::ExecutableKind::Graph => wire::PayloadRole::Graphs,
        wire::ExecutableKind::Unspecified => {
            return Err(invalid("executable resource has no kind"));
        }
    };
    let stored = resource
        .stored
        .as_ref()
        .ok_or_else(|| invalid("executable resource has no stored bytes"))?;
    catalog.check_slice(stored)?;
    let maximum = MAX_EXECUTABLE_CHUNKS * MAX_CHUNK as u64;
    if resource.decoded_nbytes == 0 || resource.decoded_nbytes > maximum
        || stored.nbytes == 0 || stored.nbytes > maximum
    {
        return Err(invalid("invalid executable resource length"));
    }
    if resource.chunk_nbytes as usize != MAX_CHUNK {
        return Err(invalid("executable chunks must use the fixed chunk size"));
    }
    let chunks = stored.nbytes.div_ceil(MAX_CHUNK as u64);
    if resource.chunk_sha256.len() as u64 != chunks
        || resource
            .chunk_sha256
            .iter()
            .any(|digest| digest.len() != 32)
    {
        return Err(invalid("executable chunk commitments are inconsistent"));
    }
    let use_ = wire::PayloadUse {
        slice: Some(stored.clone()),
        role: role as i32,
    };
    catalog.check_use(&use_, &[role])?;
    Ok(())
}

fn validate_prepared_weights(
    catalog: &wire::PreparedWeightCatalog,
    extents: &[(&str, u64)],
) -> Result<()> {
    if catalog.version == 0 {
        return Err(invalid("prepared weight catalog has no version"));
    }
    let mut names = BTreeSet::new();
    for view in &catalog.views {
        bounded_name(&view.name, 256, "weight view name")?;
        if !names.insert(view.name.as_str()) {
            return Err(invalid("duplicate prepared weight view"));
        }
        let extent = extents
            .get(view.allocation as usize)
            .map(|(_, extent)| *extent)
            .ok_or_else(|| invalid("weight view names an absent allocation"))?;
        if view.nbytes == 0 {
            return Err(invalid("weight view has no bytes"));
        }
        let end = view
            .offset
            .checked_add(view.nbytes)
            .ok_or_else(|| invalid("weight view overflow"))?;
        if end > extent {
            return Err(invalid("weight view exceeds its allocation"));
        }
        // `layout_type` names a packing owner when one exists. A plain strided tensor has none,
        // because the container owner is optional and packing is described separately.
        if view.element_bytes == 0 || view.dtype.is_empty() {
            return Err(invalid("weight view has no dtype"));
        }
        // A scalar owner has empty shape and stride; every other view must carry
        // exactly one stride per dimension. A missing stride must not be
        // interpreted as an implicit contiguous layout.
        if view.stride.len() != view.shape.len() {
            return Err(invalid("weight view strides disagree with its shape"));
        }
        let elements = if view.shape.is_empty() {
            1u64
        } else {
            let mut highest = 0u64;
            for (dimension, stride) in view.shape.iter().zip(&view.stride) {
                if *dimension == 0 {
                    return Err(invalid("weight view has an empty axis"));
                }
                highest = highest
                    .checked_add(
                        (*dimension - 1)
                            .checked_mul(*stride)
                            .ok_or_else(|| invalid("weight view stride extent overflow"))?,
                    )
                    .ok_or_else(|| invalid("weight view stride extent overflow"))?;
            }
            highest
                .checked_add(1)
                .ok_or_else(|| invalid("weight view extent overflow"))?
        };
        let covered = elements
            .checked_mul(view.element_bytes as u64)
            .ok_or_else(|| invalid("weight view byte extent overflow"))?;
        if covered > view.nbytes {
            return Err(invalid(
                "weight view bytes do not cover its shape and strides",
            ));
        }
        if let Some(packing) = &view.packing
            && packing.format.is_empty()
        {
            return Err(invalid("weight view packing has no format"));
        }
        for source in &view.sources {
            bounded_name(&source.tensor, 512, "weight source tensor")?;
            if source.ranges.len() > MAX_SOURCE_AXES {
                return Err(invalid("weight source has too many axes"));
            }
            for range in &source.ranges {
                if range.begin >= range.end {
                    return Err(invalid("weight source range is empty"));
                }
            }
            if !source.piece_encoding.is_empty()
                && (source.piece_encoding.len() > 128
                    || !source
                        .piece_encoding
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric()
                            || matches!(byte, b'_' | b'.' | b'-')))
            {
                return Err(invalid("weight source has an invalid piece encoding"));
            }
            validate_weight_pieces(source, view)?;
        }
        for padding in &view.padding {
            if padding.nbytes == 0 || padding.fill.is_empty() {
                return Err(invalid("weight padding has no fill"));
            }
            if padding
                .offset
                .checked_add(padding.nbytes)
                .is_none_or(|end| end > view.nbytes)
            {
                return Err(invalid("weight padding is outside its view"));
            }
        }
        for source in &view.sources {
            bounded_name(&source.tensor, 512, "weight source tensor")?;
            for range in &source.ranges {
                if range.begin >= range.end {
                    return Err(invalid("weight source range is empty"));
                }
            }
        }
    }
    Ok(())
}

/// Validate affine byte pieces. Empty pieces retain the legacy semantics and
/// are not interpreted here; present pieces are bounded raw-byte maps whose
/// destination must stay inside the owning view. Source bounds remain the
/// producer's checkpoint declaration because the catalog carries no source
/// tensor length.
fn validate_weight_pieces(source: &wire::WeightSource, view: &wire::PreparedWeightView) -> Result<()> {
    if source.pieces.is_empty() {
        return Ok(());
    }
    for piece in &source.pieces {
        let axes = piece.extents.len();
        if axes > MAX_PIECE_AXES
            || piece.source_strides.len() != axes
            || piece.destination_strides.len() != axes
        {
            return Err(invalid("weight piece axes disagree with its strides"));
        }
        if piece.contiguous_bytes == 0 || piece.contiguous_bytes > view.nbytes {
            return Err(invalid("weight piece has an invalid contiguous run"));
        }
        let mut source_span = piece.contiguous_bytes;
        let mut destination_span = piece.contiguous_bytes;
        for ((extent, source_stride), destination_stride) in piece
            .extents
            .iter()
            .zip(&piece.source_strides)
            .zip(&piece.destination_strides)
        {
            if *extent == 0 || *source_stride == 0 || *destination_stride == 0 {
                return Err(invalid("weight piece has an empty extent or stride"));
            }
            source_span = source_span
                .checked_add(
                    (*extent - 1)
                        .checked_mul(*source_stride)
                        .ok_or_else(|| invalid("weight piece source extent overflow"))?,
                )
                .ok_or_else(|| invalid("weight piece source extent overflow"))?;
            destination_span = destination_span
                .checked_add(
                    (*extent - 1)
                        .checked_mul(*destination_stride)
                        .ok_or_else(|| invalid("weight piece destination extent overflow"))?,
                )
                .ok_or_else(|| invalid("weight piece destination extent overflow"))?;
        }
        piece
            .source_offset
            .checked_add(source_span)
            .ok_or_else(|| invalid("weight piece source bounds overflow"))?;
        let destination_end = piece
            .destination_offset
            .checked_add(destination_span)
            .ok_or_else(|| invalid("weight piece destination bounds overflow"))?;
        if destination_end > view.nbytes {
            return Err(invalid("weight piece is outside its owner view"));
        }
    }
    Ok(())
}

/// Validate physical locations. These are unsigned hints; a package that omits or misplaces one
/// fails its payload digest check when the payload is read, never with wrong bytes.
pub fn validate_locations(set: &wire::LocationSet) -> Result<()> {
    if set.version != FORMAT_VERSION {
        return Err(invalid("unsupported location set version"));
    }
    if set.locations.len() > MAX_LOCATIONS {
        return Err(invalid("too many payload locations"));
    }
    let mut seen = BTreeSet::new();
    for location in &set.locations {
        digest(&location.payload_id, "payload id")?;
        if !seen.insert(location.payload_id.as_slice()) {
            return Err(invalid("payload resolves to more than one location"));
        }
        location_path(&location.file)?;
        bounded_name(&location.tensor, 512, "payload tensor name")?;
    }
    for deployment in &set.available_deployments {
        bounded_name(deployment, 128, "deployment id")?;
    }
    Ok(())
}

/// Validate a location path: relative, beneath the package root, no traversal or separators
/// outside the allowed set. The empty path names the primary.
pub fn location_path(file: &str) -> Result<()> {
    if file.is_empty() {
        return Ok(());
    }
    if file.len() > 512 || file.bytes().any(|b| b == b'\\' || b == 0) {
        return Err(invalid("invalid location path"));
    }
    if file.starts_with('/') || file.ends_with('/') {
        return Err(invalid("location path must be relative"));
    }
    let mut components = 0usize;
    for component in file.split('/') {
        components += 1;
        if components > 16
            || component.is_empty()
            || component == "."
            || component == ".."
            || component.len() > 128
            || !component
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
        {
            return Err(invalid("invalid location path component"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    fn descriptor(id: u8, nbytes: u64) -> wire::PayloadDescriptor {
        wire::PayloadDescriptor {
            id: vec![id; 32],
            nbytes,
        }
    }
    fn slice(id: u8, offset: u64, nbytes: u64) -> wire::PayloadSlice {
        wire::PayloadSlice {
            payload_id: vec![id; 32],
            offset,
            nbytes,
        }
    }
    fn use_of(id: u8, offset: u64, nbytes: u64, role: wire::PayloadRole) -> wire::PayloadUse {
        wire::PayloadUse {
            slice: Some(slice(id, offset, nbytes)),
            role: role as i32,
        }
    }
    fn publisher() -> wire::PublisherMetadata {
        wire::PublisherMetadata {
            namespace: "example".into(),
            official_source: "https://huggingface.co/example/model".into(),
            development: false,
            license_name: "Example".into(),
            license_link: "https://example.com/license".into(),
            ranks: 0,
        }
    }
    fn initializer(offset: u64, nbytes: u64) -> wire::Initializer {
        wire::Initializer {
            allocation: "state".into(),
            destination_offset: offset,
            source: Some(use_of(1, 0, nbytes, wire::PayloadRole::Weights)),
        }
    }

    #[test]
    fn catalog_rejects_conflicting_lengths_and_bad_ids() {
        let mut catalog = Catalog::new();
        catalog.insert(&descriptor(1, 4096)).unwrap();
        catalog.insert(&descriptor(1, 4096)).unwrap();
        assert!(catalog.insert(&descriptor(1, 8192)).is_err());
        assert!(catalog.insert(&descriptor(1, 0)).is_err());
        assert!(
            catalog
                .insert(&wire::PayloadDescriptor {
                    id: vec![1; 31],
                    nbytes: 1
                })
                .is_err()
        );
        assert!(catalog.check_slice(&slice(2, 0, 1)).is_err());
        assert!(catalog.check_slice(&slice(1, 1, 4096)).is_err());
        assert!(catalog.check_slice(&slice(1, 0, 0)).is_err());
        assert_eq!(catalog.check_slice(&slice(1, 4096 - 16, 16)).unwrap(), 4096);
    }

    #[test]
    fn use_roles_are_checked_per_consuming_binding() {
        let mut catalog = Catalog::new();
        catalog.insert(&descriptor(1, 64)).unwrap();
        let weights = use_of(1, 0, 64, wire::PayloadRole::Weights);
        assert!(
            catalog
                .check_use(&weights, &[wire::PayloadRole::Weights])
                .is_ok()
        );
        assert!(
            catalog
                .check_use(&weights, &[wire::PayloadRole::Frontend])
                .is_err()
        );
        let unknown = wire::PayloadUse {
            slice: Some(slice(1, 0, 64)),
            role: 0,
        };
        assert!(
            catalog
                .check_use(&unknown, &[wire::PayloadRole::Weights])
                .is_err()
        );
        let absent = use_of(9, 0, 64, wire::PayloadRole::Weights);
        assert!(
            catalog
                .check_use(&absent, &[wire::PayloadRole::Weights])
                .is_err()
        );
    }

    #[test]
    fn package_body_requires_identity_contract_and_unique_assets() {
        let body = wire::PackageBody {
            format_version: FORMAT_VERSION,
            package_id: vec![7; 32],
            model_identity: "model".into(),
            hardware_identity: "sm120".into(),
            publisher: Some(publisher()),
            frontend: Some(runtime::FrontendSettings::default()),
            sampling_defaults: Some(runtime::DefaultSamplingParameters::default()),
            payloads: vec![descriptor(1, 64)],
            assets: vec![wire::FrontendAsset {
                name: "tokenizer".into(),
                payload: Some(use_of(1, 0, 64, wire::PayloadRole::Frontend)),
                sha256: vec![0; 32],
            }],
            deployments: vec![wire::DeploymentCommitment {
                deployment_id: "sm120-tp1".into(),
                body_digest: vec![3; 32],
            }],
        };
        assert!(validate_package(&body).is_ok());
        let mut wrong_version = body.clone();
        wrong_version.format_version = FORMAT_VERSION + 1;
        assert!(validate_package(&wrong_version).is_err());
        let mut partial_asset = body.clone();
        partial_asset.assets[0]
            .payload
            .as_mut()
            .unwrap()
            .slice
            .as_mut()
            .unwrap()
            .nbytes = 32;
        assert!(validate_package(&partial_asset).is_err());
        let mut duplicate = body.clone();
        duplicate.assets.push(body.assets[0].clone());
        assert!(validate_package(&duplicate).is_err());
        let mut no_link = body.clone();
        no_link.publisher.as_mut().unwrap().license_link.clear();
        assert!(validate_package(&no_link).is_ok());
        let mut link_only = body.clone();
        link_only.publisher.as_mut().unwrap().license_name.clear();
        assert!(validate_package(&link_only).is_err());
        let mut no_license = body;
        no_license.publisher.as_mut().unwrap().license_name.clear();
        no_license.publisher.as_mut().unwrap().license_link.clear();
        assert!(validate_package(&no_license).is_ok());
    }

    fn module_rank() -> (wire::RankBody, Catalog) {
        let mut catalog = Catalog::new();
        catalog.insert(&descriptor(1, 64)).unwrap();
        catalog.insert(&descriptor(2, 32)).unwrap();
        catalog.insert(&descriptor(3, 32)).unwrap();
        let resource = |id, kind| wire::ExecutableResource {
            resource_id: id,
            kind,
            stored: Some(slice(2 + (id as u8 - 4).min(1), 0, 32)),
            decoded_nbytes: 16,
            chunk_nbytes: MAX_CHUNK as u32,
            chunk_sha256: vec![vec![0; 32]],
        };
        let rank = wire::RankBody {
            rank: 0,
            deployment_id: "sm120-tp1".into(),
            context: Some(runtime::Context::default()),
            allocations: vec![wire::Allocation {
                name: "state".into(),
                nbytes: 4096,
                alignment: 4096,
                address: Some(0x1000_0000),
                memory: runtime::Memory::Device as i32,
                sizing: None,
                relocations: vec![],
            }],
            initializers: vec![wire::Initializer {
                allocation: "state".into(),
                destination_offset: 0,
                source: Some(use_of(1, 0, 64, wire::PayloadRole::InitialState)),
            }],
            resources: vec![
                resource(4, wire::ExecutableKind::Cubin as i32),
                resource(5, wire::ExecutableKind::Graph as i32),
            ],
            modules: vec![wire::Module {
                name: "model".into(),
                cubin: 0,
                globals: vec![
                    wire::ModuleGlobal {
                        symbol: "first".into(),
                        nbytes: 8,
                        contents: Some(use_of(1, 0, 8, wire::PayloadRole::InitialState)),
                        relocations: vec![],
                    },
                    wire::ModuleGlobal {
                        symbol: "second".into(),
                        nbytes: 8,
                        contents: Some(use_of(1, 8, 8, wire::PayloadRole::InitialState)),
                        relocations: vec![],
                    },
                ],
            }],
            graphs: vec![wire::Graph {
                name: "decode".into(),
                flags: 0,
                resource: 1,
                node_count: 1,
                edge_count: 0,
                functions: vec![],
            }],
            ..Default::default()
        };
        (rank, catalog)
    }

    #[test]
    fn cuda_symbols_preserve_long_names_and_reject_invalid_strings() {
        for symbol in ["_f0".to_owned(), format!("_Z{}", "template_parameter_".repeat(4096))] {
            let (mut rank, mut catalog) = module_rank();
            rank.functions.push(wire::Function {
                module: "model".into(),
                symbol: symbol.clone(),
                ..Default::default()
            });
            rank.modules[0].globals[0].symbol = symbol;
            validate_rank(&rank, 1, &mut catalog, &mut BTreeSet::new()).unwrap();
        }
        for invalid_symbol in ["", "kernel\0suffix", "kernel\nsuffix"] {
            for global in [false, true] {
                let (mut rank, mut catalog) = module_rank();
                if global {
                    rank.modules[0].globals[0].symbol = invalid_symbol.into();
                } else {
                    rank.functions.push(wire::Function {
                        module: "model".into(),
                        symbol: invalid_symbol.into(),
                        ..Default::default()
                    });
                }
                let error = validate_rank(&rank, 1, &mut catalog, &mut BTreeSet::new()).unwrap_err();
                assert!(error.to_string().contains(if global { "module symbol" } else { "function symbol" }));
            }
        }
    }

    #[test]
    fn module_globals_may_share_slices_of_one_payload() {
        let (rank, mut catalog) = module_rank();
        let mut ids = BTreeSet::new();
        assert!(validate_rank(&rank, 1, &mut catalog, &mut ids).is_ok());
        let mut wrong = rank.clone();
        wrong.modules[0].globals[0].nbytes = 16;
        let mut catalog = Catalog::new();
        catalog.insert(&descriptor(1, 64)).unwrap();
        catalog.insert(&descriptor(2, 32)).unwrap();
        catalog.insert(&descriptor(3, 32)).unwrap();
        let mut ids = BTreeSet::new();
        assert!(validate_rank(&wrong, 1, &mut catalog, &mut ids).is_err());
    }

    #[test]
    fn runtime_sized_extent_rounding_is_checked() {
        let allocation = wire::Allocation {
            name: "pages".into(),
            nbytes: 0,
            alignment: 4096,
            address: Some(0x1000_0000),
            memory: runtime::Memory::Device as i32,
            sizing: Some(runtime::Sizing {
                capacity: 1,
                unit_bytes: u64::MAX / 2,
                maximum: 2,
            }),
            relocations: vec![],
        };
        assert!(allocation_extent(&allocation).is_err());
        let mut ok = allocation;
        ok.sizing = Some(runtime::Sizing {
            capacity: 1,
            unit_bytes: 300,
            maximum: 100,
        });
        assert_eq!(allocation_extent(&ok).unwrap(), 32768);
    }

    #[test]
    fn weight_views_must_fit_their_allocation_and_declare_geometry() {
        let extents = [("weights", 4096u64)];
        let view = |offset: u64, nbytes: u64| wire::PreparedWeightView {
            name: "layer.0".into(),
            group: "layer".into(),
            dtype: "f16".into(),
            element_bytes: 2,
            shape: vec![64, 32],
            stride: vec![32, 1],
            layout_type: "row".into(),
            allocation: 0,
            offset,
            nbytes,
            packing: None,
            sources: vec![],
            padding: vec![],
            statistics_domain: String::new(),
        };
        let catalog = |view| wire::PreparedWeightCatalog {
            version: 1,
            views: vec![view],
            unavailable: vec![],
        };
        assert!(validate_prepared_weights(&catalog(view(0, 4096)), &extents).is_ok());
        assert!(validate_prepared_weights(&catalog(view(1, 4096)), &extents).is_err());
        assert!(validate_prepared_weights(&catalog(view(4096, 1)), &extents).is_err());
        // A direct tensor view: no packing owner, empty layout type, torch strides.
        let mut direct = view(0, 4096);
        direct.layout_type.clear();
        direct.packing = None;
        assert!(validate_prepared_weights(&catalog(direct), &extents).is_ok());
        // A scalar owner: empty shape and stride, one element of bytes.
        let scalar = |nbytes: u64| wire::PreparedWeightView {
            name: "scale".into(),
            group: "layer".into(),
            dtype: "f32".into(),
            element_bytes: 4,
            shape: vec![],
            stride: vec![],
            layout_type: String::new(),
            allocation: 0,
            offset: 0,
            nbytes,
            packing: None,
            sources: vec![],
            padding: vec![],
            statistics_domain: String::new(),
        };
        assert!(validate_prepared_weights(&catalog(scalar(4)), &extents).is_ok());
        assert!(validate_prepared_weights(&catalog(scalar(2)), &extents).is_err());
        let mut wrong_stride = view(0, 64);
        wrong_stride.stride = vec![1];
        assert!(validate_prepared_weights(&catalog(wrong_stride), &extents).is_err());
        let mut missing_stride = view(0, 2);
        missing_stride.shape = vec![u64::MAX];
        missing_stride.stride.clear();
        assert!(validate_prepared_weights(&catalog(missing_stride), &extents).is_err());
        let mut empty_axis = view(0, 2);
        empty_axis.shape = vec![0, 2];
        empty_axis.stride = vec![2, 1];
        assert!(validate_prepared_weights(&catalog(empty_axis), &extents).is_err());
        let mut no_dtype = view(0, 64);
        no_dtype.dtype.clear();
        assert!(validate_prepared_weights(&catalog(no_dtype), &extents).is_err());
    }

    #[test]
    fn weight_pieces_are_bounded_and_legacy_sources_remain_empty() {
        let extents = [("weights", 16 << 30)];
        let view = |nbytes: u64| wire::PreparedWeightView {
            name: "experts.w1_fp4".into(),
            group: "experts".into(),
            dtype: "u8".into(),
            element_bytes: 1,
            shape: vec![nbytes],
            stride: vec![1],
            layout_type: String::new(),
            allocation: 0,
            offset: 0,
            nbytes,
            packing: None,
            sources: vec![
                wire::WeightSource {
                    tensor: "model.experts.0.up_proj.weight".into(),
                    ranges: vec![wire::TensorRange {
                        begin: 0,
                        end: 516096,
                    }],
                    concatenation_axis: 0,
                    piece_encoding: String::new(),
                    pieces: vec![wire::WeightPiece {
                        source_offset: 0,
                        destination_offset: 0,
                        contiguous_bytes: 4,
                        extents: vec![9, 4, 32, 112],
                        source_strides: vec![57344, 14336, 448, 4],
                        destination_strides: vec![57344, 4, 16, 512],
                    }],
                },
            ],
            padding: vec![],
            statistics_domain: String::new(),
        };
        let catalog = |view| wire::PreparedWeightCatalog {
            version: 1,
            views: vec![view],
            unavailable: vec![],
        };
        assert!(validate_prepared_weights(&catalog(view(516096)), &extents).is_ok());

        // A prior source without pieces serializes to the same legacy wire form.
        let legacy = wire::WeightSource {
            tensor: "model.embed.weight".into(),
            ranges: vec![wire::TensorRange { begin: 0, end: 8 }],
            concatenation_axis: 0,
            piece_encoding: String::new(),
            pieces: vec![],
        };
        let decoded = wire::WeightSource::decode(legacy.encode_to_vec().as_slice()).unwrap();
        assert!(decoded.pieces.is_empty());
        assert_eq!(decoded.ranges, legacy.ranges);

        // Prior wire data may carry field 4 pieces without field 5 encoding.
        let prior_pieces = [
            0x0a, 0x06, b'w', b'e', b'i', b'g', b'h', b't',
            0x12, 0x04, 0x08, 0x00, 0x10, 0x08,
            0x22, 0x06, 0x18, 0x04, 0x20, 0x01, 0x28, 0x01, 0x30, 0x01,
        ];
        let prior = wire::WeightSource::decode(&prior_pieces[..]).unwrap();
        assert_eq!(prior.pieces.len(), 1);
        assert_eq!(prior.pieces[0].contiguous_bytes, 4);
        assert!(prior.piece_encoding.is_empty());

        let mapped = view(516096).sources[0].clone();
        let decoded = wire::WeightSource::decode(mapped.encode_to_vec().as_slice()).unwrap();
        assert_eq!(decoded.pieces.len(), 1);
        assert_eq!(decoded.pieces[0].destination_strides, vec![57344, 4, 16, 512]);
        let roundtrip = wire::PreparedWeightView {
            sources: vec![decoded],
            ..view(516096)
        };
        assert!(validate_prepared_weights(&catalog(roundtrip), &extents).is_ok());

        // A serialized prior WeightSource has no field 4; its fixed bytes must
        // continue to decode and validate without reinterpretation.
        let prior_bytes = [
            0x0a, 0x06, b'w', b'e', b'i', b'g', b'h', b't',
            0x12, 0x04, 0x08, 0x00, 0x10, 0x08,
            0x18, 0x01,
        ];
        let prior_source = wire::WeightSource::decode(&prior_bytes[..]).unwrap();
        assert_eq!(prior_source.tensor, "weight");
        assert_eq!(prior_source.concatenation_axis, 1);
        assert!(prior_source.pieces.is_empty());
        let prior_view = wire::PreparedWeightView {
            sources: vec![prior_source],
            ..view(4096)
        };
        assert!(validate_prepared_weights(&catalog(prior_view), &extents).is_ok());

        // The w13 owner maps one local packed row per outer index; the
        // contiguous run is the packed row width, not local * row width.
        let w13_nbytes = 1152 * 3584;
        let w13_view = wire::PreparedWeightView {
            name: "experts.w1_fp4.actual".into(),
            shape: vec![w13_nbytes],
            stride: vec![1],
            nbytes: w13_nbytes,
            sources: vec![wire::WeightSource {
                tensor: "model.experts.0.up_proj.weight".into(),
                ranges: vec![wire::TensorRange { begin: 0, end: 1152 }],
                concatenation_axis: 0,
                piece_encoding: String::new(),
                pieces: vec![wire::WeightPiece {
                    source_offset: 0,
                    destination_offset: 0,
                    contiguous_bytes: 3584,
                    extents: vec![1152],
                    source_strides: vec![3584],
                    destination_strides: vec![3584],
                }],
            }],
            ..view(w13_nbytes)
        };
        assert!(validate_prepared_weights(&catalog(w13_view), &extents).is_ok());
        let mut overlapping = wire::PreparedWeightView {
            name: "experts.w1_fp4.overlap".into(),
            nbytes: w13_nbytes,
            ..view(w13_nbytes)
        };
        overlapping.sources[0].ranges[0].end = 1152;
        overlapping.sources[0].pieces[0].contiguous_bytes = 1152 * 3584;
        assert!(validate_prepared_weights(&catalog(overlapping), &extents).is_err());

        // A valid contiguous run may exceed 4 GiB; the owner bound is the limit.
        let large = wire::PreparedWeightView {
            name: "experts.large".into(),
            shape: vec![6 << 30],
            nbytes: 6 << 30,
            sources: vec![wire::WeightSource {
                tensor: "model.large.weight".into(),
                ranges: vec![wire::TensorRange { begin: 0, end: 6 << 30 }],
                concatenation_axis: 0,
                piece_encoding: String::new(),
                pieces: vec![wire::WeightPiece {
                    source_offset: 0,
                    destination_offset: 0,
                    contiguous_bytes: 5 << 30,
                    extents: vec![],
                    source_strides: vec![],
                    destination_strides: vec![],
                }],
            }],
            ..view(6 << 30)
        };
        assert!(validate_prepared_weights(&catalog(large), &extents).is_ok());

        // Source tensor rank is independent of a flattened prepared owner rank.
        let mut rank3 = view(4096);
        rank3.sources[0].pieces.clear();
        rank3.sources[0].ranges = vec![
            wire::TensorRange { begin: 0, end: 8 },
            wire::TensorRange { begin: 0, end: 4 },
            wire::TensorRange { begin: 0, end: 2 },
        ];
        assert!(validate_prepared_weights(&catalog(rank3), &extents).is_ok());
        let mut encoded_rank3 = view(4096);
        encoded_rank3.sources[0].ranges = vec![
            wire::TensorRange { begin: 0, end: 8 },
            wire::TensorRange { begin: 0, end: 4 },
            wire::TensorRange { begin: 0, end: 2 },
        ];
        encoded_rank3.sources[0].piece_encoding = "iq2_xs_desc_n64_k128_v1".into();
        encoded_rank3.sources[0].pieces = vec![wire::WeightPiece {
            source_offset: 0,
            destination_offset: 0,
            contiguous_bytes: 512,
            extents: vec![],
            source_strides: vec![],
            destination_strides: vec![],
        }];
        assert!(validate_prepared_weights(&catalog(encoded_rank3), &extents).is_ok());
        let mut too_many = view(4096);
        too_many.sources[0].pieces.clear();
        too_many.sources[0].ranges = (0..9).map(|_| wire::TensorRange { begin: 0, end: 1 }).collect();
        assert!(validate_prepared_weights(&catalog(too_many), &extents).is_err());

        let mut encoded = view(516096);
        encoded.sources[0].piece_encoding = "w4a16_mma_k16_n64_v1".into();
        assert!(validate_prepared_weights(&catalog(encoded), &extents).is_ok());
        let mut bad_encoding = view(516096);
        bad_encoding.sources[0].piece_encoding = "bad encoding".into();
        assert!(validate_prepared_weights(&catalog(bad_encoding), &extents).is_err());

        let mut invalid = view(516096);
        let piece = &mut invalid.sources[0].pieces[0];
        piece.extents.push(1);
        assert!(validate_prepared_weights(&catalog(invalid), &extents).is_err());

        let mut overflow = view(516096);
        overflow.sources[0].pieces[0].destination_offset = u64::MAX;
        assert!(validate_prepared_weights(&catalog(overflow), &extents).is_err());

        let mut outside = view(100);
        outside.sources[0].pieces[0].destination_strides = vec![57344, 4, 16, 512];
        outside.sources[0].pieces[0].destination_offset = 1;
        assert!(validate_prepared_weights(&catalog(outside), &extents).is_err());

        let mut no_run = view(516096);
        no_run.sources[0].pieces[0].contiguous_bytes = 0;
        assert!(validate_prepared_weights(&catalog(no_run), &extents).is_err());
    }

    #[test]
    fn locations_reject_traversal_and_duplicates() {
        let set = wire::LocationSet {
            version: FORMAT_VERSION,
            locations: vec![wire::PayloadLocation {
                payload_id: vec![1; 32],
                file: "weights/layer-000.safetensors".into(),
                tensor: "flare.payload".into(),
            }],
            available_deployments: vec!["sm120-tp1".into()],
        };
        assert!(validate_locations(&set).is_ok());
        for file in [
            "../escape.safetensors",
            "/abs.safetensors",
            "weights/",
            "weights//x",
            "weights\\x",
            "a/../b",
        ] {
            let mut bad = set.clone();
            bad.locations[0].file = file.into();
            assert!(validate_locations(&bad).is_err(), "{file}");
        }
        let mut duplicate = set;
        duplicate.locations.push(duplicate.locations[0].clone());
        assert!(validate_locations(&duplicate).is_err());
        assert!(location_path("").is_ok());
    }

    #[test]
    fn executable_resources_are_not_limited_by_the_metadata_bound() {
        let decoded = MAX_BODY as u64 + 1;
        let stored = decoded / 2;
        let chunks = stored.div_ceil(MAX_CHUNK as u64);
        let mut catalog = Catalog::new();
        catalog.insert(&descriptor(1, stored)).unwrap();
        let resource = wire::ExecutableResource {
            resource_id: 3,
            kind: wire::ExecutableKind::Cubin as i32,
            stored: Some(slice(1, 0, stored)),
            decoded_nbytes: decoded,
            chunk_nbytes: MAX_CHUNK as u32,
            chunk_sha256: vec![vec![0; 32]; chunks as usize],
        };
        assert!(validate_resource(&resource, &catalog).is_ok());
        let mut too_many = resource.clone();
        too_many.decoded_nbytes = (MAX_EXECUTABLE_CHUNKS + 1) * MAX_CHUNK as u64;
        assert!(validate_resource(&too_many, &catalog).is_err());
        let mut inconsistent = resource.clone();
        inconsistent.chunk_sha256.pop();
        assert!(validate_resource(&inconsistent, &catalog).is_err());
    }

    #[test]
    fn initializers_must_be_disjoint_and_in_bounds() {
        let mut catalog = Catalog::new();
        catalog.insert(&descriptor(1, 4096)).unwrap();
        let body = |initializers| wire::RankBody {
            initializers,
            ..Default::default()
        };
        let extents = BTreeMap::from([("state", 8192u64)]);
        assert!(
            validate_initializers(
                &body(vec![initializer(0, 4096), initializer(4096, 4096)]),
                &extents,
                &catalog
            )
            .is_ok()
        );
        assert!(
            validate_initializers(
                &body(vec![initializer(0, 4096), initializer(2048, 4096)]),
                &extents,
                &catalog
            )
            .is_err()
        );
        // Leading holes stay zero: allocations start zeroed and need no initializer there.
        assert!(
            validate_initializers(&body(vec![initializer(4096, 4096)]), &extents, &catalog).is_ok()
        );
        assert!(
            validate_initializers(&body(vec![initializer(4096, 8192)]), &extents, &catalog)
                .is_err()
        );
        assert!(
            validate_initializers(&body(vec![initializer(0, 8192)]), &extents, &catalog).is_err()
        );
        let mut absent = initializer(0, 16);
        absent.allocation = "missing".into();
        assert!(validate_initializers(&body(vec![absent]), &extents, &catalog).is_err());
        let mut wrong_role = initializer(0, 16);
        wrong_role.source.as_mut().unwrap().role = wire::PayloadRole::Kernels as i32;
        assert!(validate_initializers(&body(vec![wrong_role]), &extents, &catalog).is_err());
    }
    #[test]
    fn qwen_eos_engram_without_token_map_is_valid() {
        let mut catalog = Catalog::new();
        catalog.insert(&descriptor(1, 80)).unwrap();
        catalog.insert(&descriptor(2, 20)).unwrap();
        let use_ = |id: u8, nbytes: u64| wire::PayloadUse {
            slice: Some(wire::PayloadSlice {
                payload_id: vec![id; 32],
                offset: 0,
                nbytes,
            }),
            role: wire::PayloadRole::Weights as i32,
        };
        let table = wire::EngramTable {
            layer: 1,
            rows: 320001536,
            embedding_width: 160,
            scale_block: 16,
            weights: Some(use_(1, 80)),
            scales: Some(use_(2, 20)),
            hash: Some(wire::EngramHash {
                method: wire::engram_hash::Method::EosXorModulo as i32,
                orders: (2..=3).flat_map(|order| [order; 8]).collect(),
                primes: vec![2; 16],
                offsets: vec![0; 16],
                ..Default::default()
            }),
            weight_buffer: "ple_weights".into(),
            scale_buffer: "ple_scales".into(),
            hash_buffer: "ple_hashes".into(),
            request_buffer: "ple_requests".into(),
            ready_event: "ple_ready".into(),
            encoding: wire::engram_table::Encoding::Nvfp4E4m3Global as i32,
            global_scale: 1.0,
            ..Default::default()
        };
        let engram = wire::Engram {
            token_map: vec![],
            pad_token_id: 1,
            compressed_vocabulary: 248320,
            tables: vec![table.clone()],
        };
        assert!(validate_engram(&engram, 1, &catalog).is_ok());

        let mut wrong_hash = engram.clone();
        wrong_hash.tables[0].hash.as_mut().unwrap().method =
            wire::engram_hash::Method::XorModulo as i32;
        assert!(validate_engram(&wrong_hash, 1, &catalog).is_err());

        let mut no_request = engram.clone();
        no_request.tables[0].request_buffer.clear();
        assert!(validate_engram(&no_request, 1, &catalog).is_err());

        let mut mapped = engram.clone();
        mapped.token_map = vec![0, 1];
        let error = validate_engram(&mapped, 1, &catalog).unwrap_err().to_string();
        assert!(error.contains("requires XOR-modulo"), "{error}");

        let mut mapped_xor = mapped;
        mapped_xor.tables[0].hash.as_mut().unwrap().method =
            wire::engram_hash::Method::XorModulo as i32;
        assert!(validate_engram(&mapped_xor, 1, &catalog).is_ok());
    }

    #[test]
    fn peer_memory_requires_owner_inclusive_unique_readers_and_unique_addresses() {
        let peer = |owner, address, readers| runtime::PeerMemory {
            owner,
            address,
            nbytes: 4096,
            readers,
        };
        let validate = |peers| {
            let (mut rank, mut catalog) = module_rank();
            rank.peers = peers;
            validate_rank(&rank, 2, &mut catalog, &mut BTreeSet::new())
        };
        assert!(
            validate(vec![
                peer(0, 0x1000, vec![0, 1]),
                peer(1, 0x2000, vec![1, 0]),
            ])
            .is_ok()
        );
        assert!(validate(vec![peer(0, 0x1000, vec![1])]).is_err());
        assert!(validate(vec![peer(0, 0x1000, vec![0, 0])]).is_err());
        assert!(validate(vec![peer(0, 0x1000, vec![0, 2])]).is_err());
        assert!(validate(vec![peer(0, 0x1000, vec![])]).is_err());
        assert!(
            validate(vec![
                peer(0, 0x1000, vec![0, 1]),
                peer(1, 0x1000, vec![0, 1]),
            ])
            .is_err()
        );
    }
}
