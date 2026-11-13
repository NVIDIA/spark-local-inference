/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Stable semantic weight payload extraction from preparation declarations.
//!
//! Python supplies a thin declaration of semantic groups, storage classes, and
//! explicit aliases; this module resolves those views against retained
//! allocations. It never infers immutability, grouping, or alias identity from
//! pointer values, object paths, or tensor names. Bytes that no declaration
//! claims are returned as the initial-state payload, never discarded.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;

use super::{invalid, Result};

/// Fixed zero-block granularity for stored weight payloads.
pub(crate) const ELIDE_BLOCK: u64 = 64 << 10;

const MAX_RUNS: usize = 1 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StorageClass {
    /// Stable prepared weight bytes: checkpoint bytes, declared packing, TP part.
    Stable,
    /// Layout selected by tuning; grouped separately from stable weights.
    Tuned,
    /// Residual retained bytes with no owner: pointers, tensor maps, counters,
    /// warm-up residue. Never stable weight storage, never tuning output.
    State,
}

impl StorageClass {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            StorageClass::Stable => "stable",
            StorageClass::Tuned => "tuned",
            StorageClass::State => "state",
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct WeightAllocationInput {
    pub name: String,
    pub address: u64,
    pub nbytes: u64,
}

/// Declared packing geometry of one prepared view.
///
/// Only fields the producer actually records are present; absent numbers stay
/// zero rather than being inferred from a format name.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub(crate) struct WeightPackingParameters {
    pub format: String,
    #[serde(default)]
    pub block_size: u64,
    #[serde(default)]
    pub group_size: u64,
    #[serde(default)]
    pub scale_atom_rows: u64,
    #[serde(default)]
    pub tile_order: u64,
    #[serde(default)]
    pub row_rotation: u64,
    #[serde(default)]
    pub locality_tile: u64,
}

/// One declared tensor view. `alias_of` names an earlier declared view that
/// owns the same bytes; an alias contributes no stored span.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct WeightViewInput {
    pub name: String,
    pub group: String,
    pub storage: StorageClass,
    #[serde(default)]
    pub alias_of: Option<String>,
    #[serde(default)]
    pub sources: Vec<super::WeightPayloadSource>,
    #[serde(default)]
    pub packing: String,
    #[serde(default)]
    pub packing_parameters: Option<WeightPackingParameters>,
    #[serde(default)]
    pub statistics_domain: String,
    pub element_bytes: u64,
    pub shape: Vec<u64>,
    pub stride: Vec<u64>,
    pub address: u64,
}

/// The thin declaration Python emits: semantic views only. Allocations and
/// zero ranges come from the exporter's resolved capture state.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct WeightPayloadDeclaration {
    pub version: u32,
    pub views: Vec<WeightViewInput>,
}

impl WeightPayloadDeclaration {
    pub(crate) fn into_request(
        self,
        allocations: Vec<WeightAllocationInput>,
        zero_ranges: Vec<(u64, u64)>,
    ) -> WeightPayloadRequest {
        WeightPayloadRequest {
            version: self.version,
            allocations,
            zero_ranges,
            views: self.views,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct WeightPayloadRequest {
    pub version: u32,
    pub allocations: Vec<WeightAllocationInput>,
    /// Ranges that preparation declares zero or otherwise not stored.
    #[serde(default)]
    pub zero_ranges: Vec<(u64, u64)>,
    pub views: Vec<WeightViewInput>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PayloadSpan {
    pub payload_offset: u64,
    pub allocation: String,
    pub destination_offset: u64,
    pub nbytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WeightGroupPayload {
    pub name: String,
    pub storage: StorageClass,
    pub nbytes: u64,
    pub spans: Vec<PayloadSpan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WeightAlias {
    pub name: String,
    pub of: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WeightPayloadCatalog {
    pub groups: Vec<WeightGroupPayload>,
    /// Every retained byte no stable or tuned declaration claims.
    pub state: WeightGroupPayload,
    pub aliases: Vec<WeightAlias>,
}

/// Name of the payload that holds unclassified retained bytes.
pub(crate) const INITIAL_STATE_PAYLOAD: &str = "initial-state";

pub(crate) fn extract(request: &WeightPayloadRequest) -> Result<WeightPayloadCatalog> {
    let retained = request.allocations.iter().map(|a| a.name.as_str()).collect();
    extract_retained(request, &retained)
}

/// Validate every declared view before excluding allocations pruned by graph reachability.
pub(crate) fn extract_retained(
    request: &WeightPayloadRequest,
    retained: &BTreeSet<&str>,
) -> Result<WeightPayloadCatalog> {
    if request.version != 1 {
        return Err(invalid(format!(
            "unsupported weight payload declaration version {}",
            request.version
        )));
    }
    let allocations = AllocationIndex::new(&request.allocations)?;
    if retained.iter().any(|name| !request.allocations.iter().any(|a| a.name == *name)) {
        return Err(invalid("retained weight allocation is absent from capture storage"));
    }
    let mut names = BTreeSet::new();
    for view in &request.views {
        if view.name.is_empty() || !names.insert(view.name.as_str()) {
            return Err(invalid("duplicate or empty weight view name"));
        }
        if view.group.is_empty() {
            return Err(invalid(format!("weight view {:?} has no group", view.name)));
        }
        if view.storage == StorageClass::State {
            return Err(invalid("state storage cannot be declared for a weight view"));
        }
    }
    for view in &request.views {
        if let Some(target) = &view.alias_of {
            if target == &view.name {
                return Err(invalid(format!("weight view {:?} aliases itself", view.name)));
            }
            if !names.contains(target.as_str()) {
                return Err(invalid(format!(
                    "weight view {:?} aliases undeclared view {target:?}",
                    view.name
                )));
            }
        }
    }
    let alias_targets: BTreeSet<&str> = request
        .views
        .iter()
        .filter_map(|view| view.alias_of.as_deref())
        .collect();
    for view in &request.views {
        if alias_targets.contains(view.name.as_str()) && view.alias_of.is_some() {
            return Err(invalid(format!(
                "weight view {:?} aliases another alias",
                view.name
            )));
        }
    }

    // Packing is declared per view, not per semantic layer: one layer legitimately
    // mixes nvfp4 experts, bfloat16 attention, and float32 norms/scales, so no
    // uniformity is required here. The stable/tuned split is the declaration's
    // authority; tuning cannot be inferred from a packing string. Only an alias
    // that disagrees with the view it aliases is invalid (checked below).

    for view in &request.views {
        if let Some(parameters) = &view.packing_parameters {
            if parameters.format.is_empty() {
                return Err(invalid(format!(
                    "weight view {:?} declares packing parameters without a format",
                    view.name
                )));
            }
            if !view.packing.is_empty() && view.packing != parameters.format {
                return Err(invalid(format!(
                    "weight view {:?} packing {:?} differs from its declared parameters {:?}",
                    view.name, view.packing, parameters.format
                )));
            }
        }
    }

    let by_name: BTreeMap<&str, usize> = request
        .views
        .iter()
        .enumerate()
        .map(|(index, view)| (view.name.as_str(), index))
        .collect();
    let mut placement: Vec<(String, Vec<(u64, u64)>)> = Vec::with_capacity(request.views.len());
    for view in &request.views {
        let allocation = allocations.find(view)?;
        let relative = view.address - allocation.address;
        let placed = occupied_runs(view)?
            .into_iter()
            .map(|(start, end)| {
                let start = start
                    .checked_add(relative)
                    .ok_or_else(|| invalid("weight view offset overflow"))?;
                let end = end
                    .checked_add(relative)
                    .ok_or_else(|| invalid("weight view offset overflow"))?;
                Ok((start, end))
            })
            .collect::<Result<Vec<_>>>()?;
        placement.push((allocation.name.clone(), placed));
    }

    let mut aliases = Vec::new();
    for (index, view) in request.views.iter().enumerate() {
        let Some(target) = &view.alias_of else {
            continue;
        };
        let target_index = by_name[target.as_str()];
        let target_view = &request.views[target_index];
        // Aliases must share the same allocation and the same absolute occupied
        // storage. Allocation-relative intervals alone would conflate distinct
        // tensors that happen to sit at equal offsets in different allocations.
        if view.packing != target_view.packing
            || view.packing_parameters != target_view.packing_parameters
            || view.storage != target_view.storage
        {
            return Err(invalid(format!(
                "weight view {:?} aliases {target:?} with a different packing or storage class",
                view.name
            )));
        }
        if view.element_bytes != target_view.element_bytes
            || placement[index] != placement[target_index]
            || view.address != target_view.address
        {
            return Err(invalid(format!(
                "weight view {:?} aliases {target:?} with different covered storage",
                view.name
            )));
        }
        if retained.contains(placement[index].0.as_str()) {
            aliases.push(WeightAlias {
                name: view.name.clone(),
                of: target.clone(),
            });
        }
    }

    let mut order: Vec<usize> = (0..request.views.len())
        .filter(|&index| request.views[index].alias_of.is_none()
            && retained.contains(placement[index].0.as_str()))
        .collect();
    order.sort_by(|&left, &right| request.views[left].name.cmp(&request.views[right].name));

    let mut claimed = Claimed::new(&request.allocations);
    let mut groups: BTreeMap<(String, StorageClass), Vec<PayloadSpan>> = BTreeMap::new();
    for index in order {
        let view = &request.views[index];
        let allocation = allocations.find(view)?;
        let spans = groups
            .entry((view.group.clone(), view.storage))
            .or_default();
        let (placed_allocation, placed) = &placement[index];
        debug_assert_eq!(placed_allocation, &allocation.name);
        for (relative_start, relative_end) in placed {
            let start = allocation
                .address
                .checked_add(*relative_start)
                .ok_or_else(|| invalid("weight view address overflow"))?;
            let end = allocation
                .address
                .checked_add(*relative_end)
                .ok_or_else(|| invalid("weight view address overflow"))?;
            for (piece_start, piece_end) in claimed.claim(&allocation.name, start, end) {
                spans.push(PayloadSpan {
                    payload_offset: 0,
                    allocation: allocation.name.clone(),
                    destination_offset: piece_start - allocation.address,
                    nbytes: piece_end - piece_start,
                });
            }
        }
    }

    let mut groups: Vec<WeightGroupPayload> = groups
        .into_iter()
        .map(|((group, storage), spans)| -> Result<WeightGroupPayload> {
            let name = match storage {
                StorageClass::Stable => group,
                StorageClass::Tuned => format!("{group}:tuned"),
                // State is emitted separately below, never as a declared group.
                StorageClass::State => {
                    return Err(invalid("state storage cannot be declared for a weight view"));
                }
            };
            let spans = number_payload(spans);
            let nbytes = spans.iter().map(|span| span.nbytes).sum();
            Ok(WeightGroupPayload {
                name,
                storage,
                nbytes,
                spans,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    groups.sort_by(|left, right| left.name.cmp(&right.name));

    let mut state = Vec::new();
    for allocation in &request.allocations {
        if !retained.contains(allocation.name.as_str()) {
            continue;
        }
        let base = allocation.address;
        let end = base
            .checked_add(allocation.nbytes)
            .ok_or_else(|| invalid("weight allocation extent overflow"))?;
        let mut excluded: Vec<(u64, u64)> = claimed.ranges(&allocation.name).to_vec();
        for (start, length) in &request.zero_ranges {
            let zero_end = start
                .checked_add(*length)
                .ok_or_else(|| invalid("zero range extent overflow"))?;
            let (start, zero_end) = (*start, zero_end);
            if let Some((lo, hi)) = intersect(start, zero_end, base, end) {
                excluded.push((lo, hi));
            }
        }
        excluded.sort_unstable();
        for (start, stop) in complement(&excluded, base, end) {
            state.push(PayloadSpan {
                payload_offset: 0,
                allocation: allocation.name.clone(),
                destination_offset: start - base,
                nbytes: stop - start,
            });
        }
    }
    let state = number_payload(state);
    let state = WeightGroupPayload {
        name: INITIAL_STATE_PAYLOAD.to_string(),
        storage: StorageClass::State,
        nbytes: state.iter().map(|span| span.nbytes).sum(),
        spans: state,
    };

    Ok(WeightPayloadCatalog {
        groups,
        state,
        aliases,
    })
}

/// Stream the nonzero 64 KiB blocks of every span, anchored at the span start
/// rather than at the containing allocation. Blocks that are entirely zero are
/// omitted, so a group's stored bytes do not depend on allocation placement.
pub(crate) fn for_each_owner_block<F, G>(
    spans: &[PayloadSpan],
    read: &mut F,
    emit: &mut G,
) -> Result<()>
where
    F: FnMut(&str, u64, &mut [u8]) -> Result<()>,
    G: FnMut(&PayloadSpan, u64, &[u8]) -> Result<()>,
{
    let mut buffer = vec![0u8; ELIDE_BLOCK as usize];
    for span in spans {
        let mut offset = 0;
        while offset < span.nbytes {
            let nbytes = ELIDE_BLOCK.min(span.nbytes - offset);
            let block = &mut buffer[..nbytes as usize];
            block.fill(0);
            let destination = span
                .destination_offset
                .checked_add(offset)
                .ok_or_else(|| invalid("weight span offset overflow"))?;
            read(&span.allocation, destination, block)?;
            if block.iter().any(|byte| *byte != 0) {
                emit(span, offset, block)?;
            }
            offset += nbytes;
        }
    }
    Ok(())
}

fn number_payload(spans: Vec<PayloadSpan>) -> Vec<PayloadSpan> {
    let mut offset = 0;
    spans
        .into_iter()
        .map(|span| {
            let numbered = PayloadSpan {
                payload_offset: offset,
                ..span
            };
            offset += numbered.nbytes;
            numbered
        })
        .collect()
}

/// Element extent of a view, validating shape, stride, and element width.
fn element_extent(view: &WeightViewInput) -> Result<u64> {
    if view.element_bytes == 0 {
        return Err(invalid(format!("weight view {:?} has no element width", view.name)));
    }
    if view.shape.len() != view.stride.len() {
        return Err(invalid(format!(
            "weight view {:?} shape and stride ranks differ",
            view.name
        )));
    }
    if view.shape.iter().any(|size| *size == 0) {
        return Err(invalid(format!("weight view {:?} has an empty dimension", view.name)));
    }
    let extent = view
        .shape
        .iter()
        .zip(&view.stride)
        .try_fold(1u64, |extent, (&size, &stride)| {
            (size - 1)
                .checked_mul(stride)
                .and_then(|span| extent.checked_add(span))
        })
        .and_then(|extent| extent.checked_mul(view.element_bytes))
        .ok_or_else(|| invalid(format!("weight view {:?} extent overflow", view.name)))?;
    Ok(extent)
}

/// Exact occupied byte intervals of a view, relative to its own address.
/// A strided dimension contributes separate runs; a view never owns its holes.
fn occupied_runs(view: &WeightViewInput) -> Result<Vec<(u64, u64)>> {
    let element = view.element_bytes;
    let mut dimensions: Vec<(u64, u64)> = view
        .shape
        .iter()
        .copied()
        .zip(view.stride.iter().copied())
        .filter(|(size, _)| *size > 1)
        .collect();
    for (size, stride) in &dimensions {
        if *stride == 0 {
            return Err(invalid(format!(
                "weight view {:?} broadcasts dimension {size} over one address",
                view.name
            )));
        }
    }
    if dimensions.is_empty() {
        return Ok(vec![(0, element)]);
    }
    // Fold dimensions by byte stride so a transposed or permuted contiguous
    // view becomes one run instead of one run per element.
    dimensions.sort_unstable_by_key(|(_, stride)| *stride);
    let mut run_elements = 1u64;
    let mut folded = 0;
    while folded < dimensions.len() {
        let (size, stride) = dimensions[folded];
        if stride != run_elements {
            break;
        }
        run_elements = run_elements
            .checked_mul(size)
            .ok_or_else(|| invalid("weight view run overflow"))?;
        folded += 1;
    }
    let run = run_elements
        .checked_mul(element)
        .ok_or_else(|| invalid("weight view run overflow"))?;
    let outer = &dimensions[folded..];
    if outer.is_empty() {
        return Ok(vec![(0, run)]);
    }
    let count = outer.iter().try_fold(1u64, |count, (size, _)| {
        count
            .checked_mul(*size)
            .ok_or_else(|| invalid("weight view run count overflow"))
    })?;
    if count > MAX_RUNS as u64 {
        return Err(invalid(format!(
            "weight view {:?} needs {count} runs, above the {MAX_RUNS} run limit",
            view.name
        )));
    }
    let mut runs = Vec::with_capacity(count as usize);
    let mut indices = vec![0u64; outer.len()];
    loop {
        let mut start = 0u64;
        for (index, (_, stride)) in indices.iter().zip(outer) {
            start = start
                .checked_add(
                    index
                        .checked_mul(*stride)
                        .ok_or_else(|| invalid("weight view offset overflow"))?,
                )
                .ok_or_else(|| invalid("weight view offset overflow"))?;
        }
        let begin = start
            .checked_mul(element)
            .ok_or_else(|| invalid("weight view offset overflow"))?;
        runs.push((begin, begin + run));
        let mut dimension = 0;
        loop {
            if dimension == outer.len() {
                runs.sort_unstable();
                return Ok(merge_runs(runs));
            }
            indices[dimension] += 1;
            if indices[dimension] < outer[dimension].0 {
                break;
            }
            indices[dimension] = 0;
            dimension += 1;
        }
    }
}

fn merge_runs(runs: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(runs.len());
    for (start, end) in runs {
        match merged.last_mut() {
            Some(last) if last.1 == start => last.1 = end,
            Some(last) if last.1 >= end => {}
            _ => merged.push((start, end)),
        }
    }
    merged
}

struct AllocationIndex<'a> {
    allocations: &'a [WeightAllocationInput],
}

impl<'a> AllocationIndex<'a> {
    fn new(allocations: &'a [WeightAllocationInput]) -> Result<Self> {
        let mut ordered: Vec<(u64, u64)> = Vec::with_capacity(allocations.len());
        let mut names = BTreeSet::new();
        for allocation in allocations {
            if allocation.name.is_empty() || !names.insert(allocation.name.as_str()) {
                return Err(invalid("duplicate or empty weight allocation name"));
            }
            let end = allocation
                .address
                .checked_add(allocation.nbytes)
                .ok_or_else(|| invalid("weight allocation extent overflow"))?;
            ordered.push((allocation.address, end));
        }
        ordered.sort_unstable();
        for pair in ordered.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(invalid("weight allocations overlap"));
            }
        }
        Ok(Self { allocations })
    }

    fn find(&self, view: &WeightViewInput) -> Result<&'a WeightAllocationInput> {
        let address = view.address;
        let extent = element_extent(view)?;
        let end = address
            .checked_add(extent)
            .ok_or_else(|| invalid("weight view address overflow"))?;
        self.allocations
            .iter()
            .find(|allocation| {
                allocation.address <= address
                    && end <= allocation.address.saturating_add(allocation.nbytes)
            })
            .ok_or_else(|| invalid(format!(
                "weight view {:?} at {address:#x} with extent {extent} lies outside every declared allocation",
                view.name
            )))
    }
}

/// Deterministic first-declaration ownership of shared bytes.
struct Claimed {
    ranges: BTreeMap<String, Vec<(u64, u64)>>,
}

impl Claimed {
    fn new(allocations: &[WeightAllocationInput]) -> Self {
        let ranges = allocations
            .iter()
            .map(|allocation| (allocation.name.clone(), Vec::new()))
            .collect();
        Self { ranges }
    }

    fn claim(&mut self, allocation: &str, start: u64, end: u64) -> Vec<(u64, u64)> {
        let claimed = self.ranges.entry(allocation.to_string()).or_default();
        let mut pieces = Vec::new();
        let mut cursor = start;
        for (taken_start, taken_end) in claimed.iter().copied() {
            if taken_end <= cursor || taken_start >= end {
                continue;
            }
            if taken_start > cursor {
                pieces.push((cursor, taken_start.min(end)));
            }
            cursor = cursor.max(taken_end);
            if cursor >= end {
                break;
            }
        }
        if cursor < end {
            pieces.push((cursor, end));
        }
        for piece in &pieces {
            claimed.push(*piece);
        }
        claimed.sort_unstable();
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(claimed.len());
        for (start, end) in claimed.drain(..) {
            match merged.last_mut() {
                Some(last) if last.1 == start => last.1 = end,
                Some(last) if last.1 >= end => {}
                _ => merged.push((start, end)),
            }
        }
        *claimed = merged;
        pieces
    }

    fn ranges(&self, allocation: &str) -> &[(u64, u64)] {
        self.ranges.get(allocation).map(Vec::as_slice).unwrap_or(&[])
    }
}

fn intersect(start: u64, end: u64, low: u64, high: u64) -> Option<(u64, u64)> {
    let start = start.max(low);
    let end = end.min(high);
    (start < end).then_some((start, end))
}

fn complement(ranges: &[(u64, u64)], low: u64, high: u64) -> Vec<(u64, u64)> {
    let mut pieces = Vec::new();
    let mut cursor = low;
    for (start, end) in ranges.iter().copied() {
        if end <= cursor {
            continue;
        }
        if start >= high {
            break;
        }
        if start > cursor {
            pieces.push((cursor, start.min(high)));
        }
        cursor = cursor.max(end);
        if cursor >= high {
            return pieces;
        }
    }
    if cursor < high {
        pieces.push((cursor, high));
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allocation(name: &str, address: u64, nbytes: u64) -> WeightAllocationInput {
        WeightAllocationInput {
            name: name.into(),
            address,
            nbytes,
        }
    }

    fn view(name: &str, address: u64, shape: Vec<u64>, stride: Vec<u64>) -> WeightViewInput {
        WeightViewInput {
            name: name.into(),
            group: "layer-000".into(),
            storage: StorageClass::Stable,
            alias_of: None,
            sources: Vec::new(),
            packing: "bfloat16".into(),
            packing_parameters: None,
            statistics_domain: String::new(),
            element_bytes: 2,
            shape,
            stride,
            address,
        }
    }

    fn request(allocations: Vec<WeightAllocationInput>, views: Vec<WeightViewInput>) -> WeightPayloadRequest {
        WeightPayloadRequest {
            version: 1,
            allocations,
            zero_ranges: Vec::new(),
            views,
        }
    }

    #[test]
    fn declared_packing_parameters_are_carried_and_validated() {
        let mut declared = view("weights.layers[0].experts", 0, vec![4], vec![1]);
        declared.packing = "nvfp4".into();
        declared.packing_parameters = Some(WeightPackingParameters {
            format: "nvfp4".into(),
            group_size: 16,
            ..Default::default()
        });
        let catalog = extract(&request(
            vec![allocation("device-0", 0, 0x1000)],
            vec![declared.clone()],
        ))
        .unwrap();
        assert_eq!(catalog.groups.len(), 1);

        // Alias must agree structurally, including the declared geometry.
        let mut alias = declared.clone();
        alias.name = "weights.layers[0].experts_alias".into();
        alias.alias_of = Some(declared.name.clone());
        let mismatched = {
            let mut view = alias.clone();
            view.packing_parameters = Some(WeightPackingParameters {
                format: "nvfp4".into(),
                group_size: 8,
                ..Default::default()
            });
            view
        };
        assert!(extract(&request(
            vec![allocation("device-0", 0, 0x1000)],
            vec![declared.clone(), mismatched],
        ))
        .is_err());
        assert!(extract(&request(
            vec![allocation("device-0", 0, 0x1000)],
            vec![declared.clone(), alias],
        ))
        .is_ok());

        // A format disagreement between the string and the parameters is rejected.
        let mut disagreeing = declared.clone();
        disagreeing.packing = "tensor_fp8".into();
        assert!(extract(&request(
            vec![allocation("device-0", 0, 0x1000)],
            vec![disagreeing],
        ))
        .is_err());

        // Negative geometry cannot deserialize into the unsigned fields.
        let negative: std::result::Result<WeightPayloadDeclaration, serde_json::Error> =
            serde_json::from_str(
            r#"{"version":1,"views":[{"name":"v","group":"g","storage":"stable",
                "packing":"nvfp4","packing_parameters":{"format":"nvfp4","group_size":-1},
                "element_bytes":1,"shape":[1],"stride":[1],"address":0}]}"#,
        );
        assert!(negative.is_err(), "negative group_size was accepted");
    }

    #[test]
    fn mixed_packings_in_one_semantic_layer_are_preserved() {
        let nvfp4 = {
            let mut view = view("weights.layers[0].experts", 0, vec![8], vec![1]);
            view.packing = "nvfp4".into();
            view
        };
        let bfloat16 = {
            let mut view = view("weights.layers[0].attention", 16, vec![8], vec![1]);
            view.packing = "bfloat16".into();
            view
        };
        let float32 = {
            let mut view = view("weights.layers[0].norm", 32, vec![8], vec![1]);
            view.packing = "float32".into();
            view.element_bytes = 4;
            view
        };
        let catalog = extract(&request(
            vec![allocation("device-0", 0, 0x1000)],
            vec![nvfp4, bfloat16, float32],
        ))
        .unwrap();
        // One declared layer remains one payload; every view's bytes are kept.
        assert_eq!(catalog.groups.len(), 1);
        assert_eq!(catalog.groups[0].name, "layer-000");
        assert_eq!(
            catalog.groups[0].nbytes,
            8 * 2 + 8 * 2 + 8 * 4,
            "mixed packings lost bytes"
        );
        assert_eq!(catalog.groups[0].spans.len(), 3);

        // Distinct layers may of course declare different packings too.
        let other = {
            let mut view = view("weights.layers[1].experts", 64, vec![8], vec![1]);
            view.group = "layer-001".into();
            view.packing = "nvfp4".into();
            view
        };
        let mut all = request(
            vec![allocation("device-0", 0, 0x1000)],
            vec![
                {
                    let mut view = view("weights.layers[0].norm", 0, vec![8], vec![1]);
                    view.packing = "float32".into();
                    view.element_bytes = 4;
                    view
                },
                other,
            ],
        );
        all.views[0].packing = "float32".into();
        assert_eq!(extract(&all).unwrap().groups.len(), 2);
    }

    #[test]
    fn aliases_must_match_their_target_packing() {
        let mut alias = view("weights.head", 0x1000, vec![4], vec![1]);
        alias.alias_of = Some("weights.embedding".into());
        alias.packing = "nvfp4".into();
        let catalog = extract(&request(
            vec![allocation("device-0", 0, 0x4000)],
            vec![view("weights.embedding", 0x1000, vec![4], vec![1]), alias],
        ));
        assert!(catalog.is_err(), "alias with a different packing was accepted");
    }

    #[test]
    fn residual_bytes_are_state_storage_not_stable_weights() {
        let catalog = extract(&request(
            vec![allocation("device-0", 0, 0x1000)],
            vec![view("weights.layers[0].q", 0, vec![4], vec![1])],
        ))
        .unwrap();
        assert_eq!(catalog.state.name, INITIAL_STATE_PAYLOAD);
        assert_eq!(catalog.state.storage, StorageClass::State);
        assert_eq!(catalog.state.storage.as_str(), "state");
        assert!(catalog.state.nbytes > 0);
        assert!(catalog.groups.iter().all(|group| group.storage != StorageClass::State));

        let declared = extract(&request(
            vec![allocation("device-0", 0, 0x1000)],
            vec![{
                let mut view = view("weights.layers[0].q", 0, vec![4], vec![1]);
                view.storage = StorageClass::State;
                view
            }],
        ));
        assert!(declared.is_err(), "a view declared state storage was accepted");
    }

    #[test]
    fn unreachable_weight_allocations_contribute_no_payload_or_aliases() {
        let mut alias = view("weights.source_alias", 0x1000, vec![16], vec![1]);
        alias.alias_of = Some("weights.source".into());
        let request = request(
            vec![allocation("source", 0x1000, 128), allocation("packed", 0x2000, 64)],
            vec![view("weights.source", 0x1000, vec![16], vec![1]), alias,
                 view("weights.packed", 0x2000, vec![8], vec![1])],
        );
        let catalog = extract_retained(&request, &BTreeSet::from(["packed"])).unwrap();
        assert!(catalog.aliases.is_empty());
        assert_eq!(catalog.groups.len(), 1);
        assert_eq!(catalog.groups[0].nbytes, 16);
        assert_eq!(catalog.groups[0].spans[0].allocation, "packed");
        assert_eq!(catalog.groups[0].spans[0].payload_offset, 0);
        assert_eq!(catalog.state.nbytes, 48);
        assert_eq!(catalog.state.spans[0].allocation, "packed");
        assert_eq!(catalog.state.spans[0].destination_offset, 16);
    }

    #[test]
    fn unreachable_weight_views_still_require_valid_declarations() {
        let request = request(
            vec![allocation("source", 0x1000, 128), allocation("packed", 0x2000, 64)],
            vec![view("weights.source", 0x1000, vec![16], vec![1]),
                 view("weights.packed", 0x2000, vec![8], vec![1])],
        );
        let retained = BTreeSet::from(["packed"]);
        for case in 0..5 {
            let mut invalid = request.clone();
            match case {
                0 => invalid.views[0].address = 0x3000,
                1 => invalid.views[0].shape = vec![128],
                2 => invalid.views[0].stride = vec![0],
                3 => invalid.views[0].alias_of = Some("weights.packed".into()),
                4 => invalid.views[0].storage = StorageClass::State,
                _ => unreachable!(),
            }
            assert!(extract_retained(&invalid, &retained).is_err(), "case {case}");
        }
        assert!(extract_retained(&request, &BTreeSet::from(["absent"])).is_err());
    }

    #[test]
    fn contiguous_views_become_one_span() {
        let catalog = extract(&request(
            vec![allocation("device-0", 0x1000, 0x1000)],
            vec![view("weights.layers[0].q", 0x1000, vec![4, 8], vec![8, 1])],
        ))
        .unwrap();
        assert_eq!(catalog.groups.len(), 1);
        let group = &catalog.groups[0];
        assert_eq!(group.nbytes, 4 * 8 * 2);
        assert_eq!(group.spans.len(), 1);
        assert_eq!(group.spans[0].destination_offset, 0);
        assert_eq!(catalog.state.nbytes, 0x1000 - 64);
    }

    #[test]
    fn strided_columns_do_not_own_their_holes() {
        let catalog = extract(&request(
            vec![allocation("device-0", 0, 0x1000)],
            vec![view("weights.layers[0].rows", 0, vec![2, 2], vec![16, 4])],
        ))
        .unwrap();
        let spans = &catalog.groups[0].spans;
        assert_eq!(
            spans
                .iter()
                .map(|span| (span.destination_offset, span.nbytes))
                .collect::<Vec<_>>(),
            vec![(0, 2), (8, 2), (32, 2), (40, 2)]
        );
        assert_eq!(catalog.groups[0].nbytes, 8);
    }

    #[test]
    fn declared_alias_contributes_no_stored_bytes() {
        let mut alias = view("weights.head", 0x1000, vec![4], vec![1]);
        alias.alias_of = Some("weights.embedding".into());
        let catalog = extract(&request(
            vec![allocation("device-0", 0, 0x4000)],
            vec![view("weights.embedding", 0x1000, vec![4], vec![1]), alias],
        ))
        .unwrap();
        assert_eq!(catalog.aliases.len(), 1);
        assert_eq!(catalog.aliases[0].name, "weights.head");
        assert_eq!(catalog.aliases[0].of, "weights.embedding");
        assert_eq!(catalog.groups[0].nbytes, 8);
    }

    #[test]
    fn overlapping_views_are_claimed_once_in_declaration_order() {
        let catalog = extract(&request(
            vec![allocation("device-0", 0, 0x1000)],
            vec![
                view("weights.first", 0, vec![8], vec![1]),
                view("weights.second", 8, vec![8], vec![1]),
            ],
        ))
        .unwrap();
        assert_eq!(catalog.groups[0].nbytes, 24);
        assert_eq!(
            catalog.groups[0]
                .spans
                .iter()
                .map(|span| (span.destination_offset, span.nbytes))
                .collect::<Vec<_>>(),
            vec![(0, 16), (16, 8)]
        );
    }

    #[test]
    fn unclaimed_live_bytes_become_initial_state() {
        let catalog = extract(&request(
            vec![allocation("device-0", 0x4000, 0x10000)],
            vec![view("weights.embedding", 0x4000, vec![16], vec![1])],
        ))
        .unwrap();
        assert_eq!(catalog.state.nbytes, 0x10000 - 32);
        assert_eq!(catalog.state.spans.len(), 1);
        assert_eq!(catalog.state.spans[0].destination_offset, 32);
        assert_eq!(catalog.state.name, INITIAL_STATE_PAYLOAD);
    }

    #[test]
    fn declared_zero_ranges_are_not_state() {
        let mut request = request(
            vec![allocation("device-0", 0x4000, 0x1000)],
            vec![view("weights.embedding", 0x4000, vec![16], vec![1])],
        );
        request.zero_ranges = vec![(0x4100, 0x100)];
        let catalog = extract(&request).unwrap();
        assert_eq!(catalog.state.nbytes, 0x1000 - 32 - 0x100);
    }

    fn owner_blocks(span: PayloadSpan, bytes: &[u8]) -> Vec<(u64, u64, Vec<u8>)> {
        let mut blocks = Vec::new();
        let mut blocks_read = 0;
        for_each_owner_block(
            std::slice::from_ref(&span),
            &mut |_allocation, _destination, buffer| {
                let start = blocks_read;
                blocks_read += buffer.len();
                buffer.copy_from_slice(&bytes[start..start + buffer.len()]);
                Ok(())
            },
            &mut |_span, offset, block| {
                blocks.push((offset, block.len() as u64, block.to_vec()));
                Ok(())
            },
        )
        .unwrap();
        blocks
    }

    fn span(destination_offset: u64, nbytes: u64) -> PayloadSpan {
        PayloadSpan {
            payload_offset: 0,
            allocation: "device-0".into(),
            destination_offset,
            nbytes,
        }
    }

    #[test]
    fn owner_blocks_are_anchored_at_the_span_start() {
        let mut bytes = vec![0u8; 3 * ELIDE_BLOCK as usize];
        bytes[ELIDE_BLOCK as usize] = 7;
        let blocks = owner_blocks(span(0x1000, bytes.len() as u64), &bytes);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].0, ELIDE_BLOCK);
        assert_eq!(blocks[0].2[0], 7);
    }

    #[test]
    fn owner_blocks_do_not_depend_on_allocation_offset() {
        let mut bytes = vec![0u8; 2 * ELIDE_BLOCK as usize];
        bytes[3] = 0xAB;
        bytes[ELIDE_BLOCK as usize + 9] = 0xCD;
        let low = owner_blocks(span(0x1000, bytes.len() as u64), &bytes);
        let high = owner_blocks(span(0x11000, bytes.len() as u64), &bytes);
        assert_eq!(low, high);
        assert_eq!(low.len(), 2);
    }

    #[test]
    fn owner_blocks_omit_fully_zero_blocks() {
        let mut bytes = vec![0u8; 3 * ELIDE_BLOCK as usize];
        bytes[0] = 1;
        bytes[2 * ELIDE_BLOCK as usize + 5] = 2;
        let blocks = owner_blocks(span(0, bytes.len() as u64), &bytes);
        assert_eq!(
            blocks.iter().map(|(offset, nbytes, _)| (*offset, *nbytes)).collect::<Vec<_>>(),
            vec![(0, ELIDE_BLOCK), (2 * ELIDE_BLOCK, ELIDE_BLOCK)]
        );
    }

    #[test]
    fn alias_must_cover_exactly_the_declared_target_storage() {
        let mut alias = view("weights.head", 0x1008, vec![4], vec![1]);
        alias.alias_of = Some("weights.embedding".into());
        assert!(
            extract(&request(
                vec![allocation("device-0", 0, 0x4000)],
                vec![view("weights.embedding", 0x1000, vec![4], vec![1]), alias]
            ))
            .is_err()
        );
    }

    #[test]
    fn thin_python_declaration_deserializes_without_allocation_fields() {
        let declaration: WeightPayloadDeclaration = serde_json::from_str(
            r#"{"version":1,"views":[{"name":"weights.embedding","group":"embedding",
                "storage":"stable","element_bytes":2,"shape":[4],"stride":[1],
                "address":4096}]}"#,
        )
        .unwrap();
        assert_eq!(declaration.version, 1);
        assert_eq!(declaration.views.len(), 1);
        assert_eq!(declaration.views[0].group, "embedding");
        assert!(declaration.views[0].alias_of.is_none());
        let request = declaration.into_request(
            vec![allocation("device-0", 0x1000, 0x1000)],
            Vec::new(),
        );
        assert!(extract(&request).is_ok());
    }

    #[test]
    fn alias_across_distinct_allocations_with_equal_relative_offsets_is_rejected() {
        let mut alias = view("weights.head", 0x2000, vec![4], vec![1]);
        alias.alias_of = Some("weights.embedding".into());
        assert!(
            extract(&request(
                vec![
                    allocation("device-0", 0x1000, 0x1000),
                    allocation("device-1", 0x2000, 0x1000),
                ],
                vec![view("weights.embedding", 0x1000, vec![4], vec![1]), alias]
            ))
            .is_err()
        );
    }

    #[test]
    fn transposed_contiguous_view_coalesces_to_one_run() {
        let catalog = extract(&request(
            vec![allocation("device-0", 0, 0x1000)],
            vec![{
                let mut column_major = view("weights.column_major", 0, vec![3, 4], vec![1, 3]);
                column_major.element_bytes = 4;
                column_major
            }],
        ))
        .unwrap();
        assert_eq!(catalog.groups[0].spans.len(), 1);
        assert_eq!(catalog.groups[0].nbytes, 12 * 4);
    }

    #[test]
    fn tuned_and_stable_groups_of_one_layer_stay_separate() {
        let mut tuned = view("weights.layers[0].tuned", 0x100, vec![4], vec![1]);
        tuned.storage = StorageClass::Tuned;
        let catalog = extract(&request(
            vec![allocation("device-0", 0, 0x1000)],
            vec![view("weights.layers[0].q", 0, vec![4], vec![1]), tuned],
        ))
        .unwrap();
        assert_eq!(
            catalog.groups.iter().map(|group| (group.name.as_str(), group.storage))
                .collect::<Vec<_>>(),
            vec![("layer-000", StorageClass::Stable), ("layer-000:tuned", StorageClass::Tuned)]
        );
    }

    #[test]
    fn duplicate_allocation_names_are_rejected() {
        assert!(
            extract(&request(
                vec![allocation("device-0", 0, 0x1000), allocation("device-0", 0x2000, 0x1000)],
                vec![]
            ))
            .is_err()
        );
    }

    #[test]
    fn undetermined_declarations_are_rejected() {
        let allocations = vec![allocation("device-0", 0, 0x1000)];
        let mut bad_version = request(allocations.clone(), vec![view("a", 0, vec![4], vec![1])]);
        bad_version.version = 2;
        assert!(extract(&bad_version).is_err());
        assert!(extract(&request(allocations.clone(), vec![view("a", 0x2000, vec![4], vec![1])])).is_err());
        assert!(extract(&request(allocations.clone(), vec![view("a", 0, vec![1 << 40], vec![1])])).is_err());
        let mut alias = view("b", 0, vec![4], vec![1]);
        alias.alias_of = Some("missing".into());
        assert!(extract(&request(allocations.clone(), vec![alias])).is_err());
        let mut broadcast = view("c", 8, vec![2, 2], vec![1, 0]);
        broadcast.element_bytes = 2;
        assert!(
            extract(&request(
                allocations.clone(),
                vec![broadcast]
            ))
            .is_err()
        );
        assert!(extract(&request(
            vec![allocation("device-0", 0, 0x1000), allocation("device-1", 0x800, 0x1000)],
            vec![]
        ))
        .is_err());
    }
}
