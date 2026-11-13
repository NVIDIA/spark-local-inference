/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bounded byte-copy and fill derivations for missing signed payloads.
//!
//! A derivation reconstructs one payload whose SHA-256 the signed metadata
//! already fixes. Operations are pure byte copies and pattern fills: no dtype
//! interpretation, no quantization, no arithmetic. All bounds are checked before
//! any source or output byte is touched.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use super::{
    PayloadId,
    container::Slice,
    invalid,
    schema::{Catalog, wire},
};
use crate::Result;
use ring::digest;

/// Wire version of the derivation set this build understands.
pub const DERIVATION_VERSION: u32 = 1;
/// Maximum operations in one derivation.
pub const MAX_OPERATIONS: usize = 1 << 20;
/// Maximum expanded destination runs in one derivation.
pub const MAX_RUNS: usize = 1 << 22;
/// Maximum derivations in one dependency set.
pub const MAX_DERIVATIONS: usize = 1 << 16;
/// Maximum derivation dependency depth.
pub const MAX_DEPTH: usize = 8;
/// Maximum fill pattern length.
pub const MAX_PATTERN: usize = 4096;

/// Explicit limits applied before any derivation work begins.
#[derive(Clone, Debug)]
pub struct Limits {
    pub max_operations: usize,
    pub max_runs: usize,
    pub max_derivations: usize,
    pub max_depth: usize,
    pub max_pattern: usize,
    pub chunk: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_operations: MAX_OPERATIONS,
            max_runs: MAX_RUNS,
            max_derivations: MAX_DERIVATIONS,
            max_depth: MAX_DEPTH,
            max_pattern: MAX_PATTERN,
            chunk: 1 << 20,
        }
    }
}

/// One strided outer axis of a copy. `count` runs are spaced by the strides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Axis {
    pub count: u64,
    pub source_stride: u64,
    pub destination_stride: u64,
}

/// Copies a contiguous run, repeated over up to two outer axes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Copy {
    pub source: PayloadId,
    pub source_offset: u64,
    pub destination_offset: u64,
    pub contiguous: u64,
    pub outer: Vec<Axis>,
}

/// Repeats `pattern` over a destination range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fill {
    pub destination_offset: u64,
    pub length: u64,
    pub pattern: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Operation {
    Copy(Copy),
    Fill(Fill),
}

/// Reconstructs the payload identified by `target` with exact `length` bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Derivation {
    pub target: PayloadId,
    pub length: u64,
    pub operations: Vec<Operation>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunSource {
    Copy { payload: PayloadId, offset: u64 },
    Fill { pattern: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Run {
    destination: u64,
    length: u64,
    source: RunSource,
}

fn checked_add(value: u64, increment: u64, what: &str) -> Result<u64> {
    value
        .checked_add(increment)
        .ok_or_else(|| invalid(format!("derivation {what} overflow")))
}

fn checked_mul(value: u64, factor: u64, what: &str) -> Result<u64> {
    value
        .checked_mul(factor)
        .ok_or_else(|| invalid(format!("derivation {what} overflow")))
}

fn expand(derivation: &Derivation, limits: &Limits) -> Result<(Vec<Run>, Vec<Vec<u8>>)> {
    if derivation.operations.is_empty() {
        return Err(invalid("derivation has no operations"));
    }
    if derivation.operations.len() > limits.max_operations {
        return Err(invalid("derivation exceeds its operation limit"));
    }
    if derivation.length == 0 {
        return Err(invalid("derivation target must not be empty"));
    }
    let mut runs: Vec<Run> = Vec::new();
    let mut patterns: Vec<Vec<u8>> = Vec::new();
    for operation in &derivation.operations {
        match operation {
            Operation::Fill(fill) => {
                if fill.length == 0
                    || fill.pattern.is_empty()
                    || fill.pattern.len() > limits.max_pattern
                {
                    return Err(invalid("invalid derivation fill"));
                }
                checked_add(fill.destination_offset, fill.length, "fill destination")?;
                let pattern = patterns.len();
                patterns.push(fill.pattern.clone());
                push_run(
                    &mut runs,
                    Run {
                        destination: fill.destination_offset,
                        length: fill.length,
                        source: RunSource::Fill { pattern },
                    },
                    limits,
                )?;
            }
            Operation::Copy(copy) => {
                if copy.contiguous == 0 || copy.outer.len() > 2 {
                    return Err(invalid("invalid derivation copy"));
                }
                let mut count = 1u64;
                for axis in &copy.outer {
                    if axis.count == 0 {
                        return Err(invalid("derivation copy axis must not be empty"));
                    }
                    count = checked_mul(count, axis.count, "copy run count")?;
                }
                if runs.len() as u64 + count > limits.max_runs as u64 {
                    return Err(invalid("derivation exceeds its run limit"));
                }
                let mut indexes = vec![0u64; copy.outer.len()];
                loop {
                    let mut source = copy.source_offset;
                    let mut destination = copy.destination_offset;
                    for (axis, index) in copy.outer.iter().zip(&indexes) {
                        source = checked_add(
                            source,
                            checked_mul(*index, axis.source_stride, "copy source stride")?,
                            "copy source",
                        )?;
                        destination = checked_add(
                            destination,
                            checked_mul(
                                *index,
                                axis.destination_stride,
                                "copy destination stride",
                            )?,
                            "copy destination",
                        )?;
                    }
                    checked_add(source, copy.contiguous, "copy source extent")?;
                    checked_add(destination, copy.contiguous, "copy destination extent")?;
                    push_run(
                        &mut runs,
                        Run {
                            destination,
                            length: copy.contiguous,
                            source: RunSource::Copy {
                                payload: copy.source,
                                offset: source,
                            },
                        },
                        limits,
                    )?;
                    let mut axis_index = 0;
                    loop {
                        if axis_index == copy.outer.len() {
                            break;
                        }
                        indexes[axis_index] += 1;
                        if indexes[axis_index] < copy.outer[axis_index].count {
                            break;
                        }
                        indexes[axis_index] = 0;
                        axis_index += 1;
                    }
                    if axis_index == copy.outer.len() {
                        break;
                    }
                }
            }
        }
    }
    Ok((runs, patterns))
}

fn push_run(runs: &mut Vec<Run>, run: Run, limits: &Limits) -> Result<()> {
    if runs.len() >= limits.max_runs {
        return Err(invalid("derivation exceeds its run limit"));
    }
    runs.push(run);
    Ok(())
}

/// Requires the expanded runs to tile the target exactly: no gaps or overlaps.
fn tile(runs: &mut [Run], length: u64) -> Result<()> {
    runs.sort_by_key(|run| run.destination);
    let mut cursor = 0u64;
    for run in runs.iter() {
        if run.destination != cursor {
            return Err(invalid(
                "derivation destination ranges overlap or leave a gap",
            ));
        }
        cursor = checked_add(cursor, run.length, "destination")?;
    }
    if cursor != length {
        return Err(invalid("derivation does not cover its target exactly"));
    }
    Ok(())
}

/// Expands and tiles a derivation, then coalesces only exact adjacent copy runs.
fn prepare_runs(derivation: &Derivation, limits: &Limits) -> Result<(Vec<Run>, Vec<Vec<u8>>)> {
    let (mut runs, patterns) = expand(derivation, limits)?;
    tile(&mut runs, derivation.length)?;
    coalesce_runs(&mut runs)?;
    Ok((runs, patterns))
}

fn coalesce_runs(runs: &mut Vec<Run>) -> Result<()> {
    let mut merged: Vec<Run> = Vec::with_capacity(runs.len());
    for run in runs.drain(..) {
        let mergeable = match merged.last() {
            Some(previous) => match (previous.source, run.source) {
                (
                    RunSource::Copy {
                        payload: previous_payload,
                        offset: previous_offset,
                    },
                    RunSource::Copy {
                        payload,
                        offset,
                    },
                ) => {
                    previous_payload == payload
                        && previous_offset
                            .checked_add(previous.length)
                            .is_some_and(|end| end == offset)
                        && previous
                            .destination
                            .checked_add(previous.length)
                            .is_some_and(|end| end == run.destination)
                }
                _ => false,
            },
            None => false,
        };
        if mergeable {
            let previous = merged.last_mut().expect("coalescing requires a previous run");
            previous.length = previous
                .length
                .checked_add(run.length)
                .ok_or_else(|| invalid("coalesced derivation run overflow"))?;
        } else {
            merged.push(run);
        }
    }
    *runs = merged;
    Ok(())
}

/// Validates structure and exact destination coverage without reading payloads.
pub fn validate(derivation: &Derivation, limits: &Limits) -> Result<()> {
    let (mut runs, _) = expand(derivation, limits)?;
    tile(&mut runs, derivation.length)
}

/// Converts authenticated wire derivations into executable byte operations.
///
/// Target lengths come from the signed payload catalog; a source must be a
/// signed payload or another target in the same set. This is the only wire
/// conversion for derivations: consumers must not reimplement the mapping.
pub fn from_wire(
    set: &wire::DerivationSet,
    catalog: &Catalog,
    limits: &Limits,
) -> Result<Vec<Derivation>> {
    if set.version != DERIVATION_VERSION {
        return Err(invalid("unsupported derivation set version"));
    }
    if set.derivations.len() > limits.max_derivations {
        return Err(invalid("derivation set exceeds its entry limit"));
    }
    let mut derivations = Vec::with_capacity(set.derivations.len());
    for entry in &set.derivations {
        let target = wire_payload_id(&entry.target_payload_id)?;
        let length = catalog.length(entry.target_payload_id.as_slice())?;
        let mut operations = Vec::with_capacity(entry.operations.len());
        for operation in &entry.operations {
            let operation = operation
                .operation
                .as_ref()
                .ok_or_else(|| invalid("derivation operation has no body"))?;
            operations.push(match operation {
                wire::derivation_operation::Operation::Copy(copy) => Operation::Copy(Copy {
                    source: wire_payload_id(&copy.source_payload_id)?,
                    source_offset: copy.source_offset,
                    destination_offset: copy.destination_offset,
                    contiguous: copy.contiguous_bytes,
                    outer: wire_axes(copy)?,
                }),
                wire::derivation_operation::Operation::Fill(fill) => Operation::Fill(Fill {
                    destination_offset: fill.destination_offset,
                    length: fill.byte_length,
                    pattern: fill.pattern.clone(),
                }),
            });
        }
        derivations.push(Derivation {
            target,
            length,
            operations,
        });
    }
    validate_sources(&derivations, catalog, limits)?;
    let targets: BTreeSet<PayloadId> = derivations.iter().map(|entry| entry.target).collect();
    for derivation in &derivations {
        for operation in &derivation.operations {
            if let Operation::Copy(copy) = operation
                && !targets.contains(&copy.source)
                && catalog.length(copy.source.as_bytes()).is_err()
            {
                return Err(invalid(format!(
                    "derivation source {} is not an authenticated payload",
                    copy.source
                )));
            }
        }
    }
    Ok(derivations)
}

fn wire_payload_id(bytes: &[u8]) -> Result<PayloadId> {
    let bytes: [u8; 32] = bytes
        .try_into()
        .map_err(|_| invalid("derivation payload identity must be 32 bytes"))?;
    Ok(PayloadId::from_bytes(bytes))
}

fn wire_axes(copy: &wire::DerivationCopy) -> Result<Vec<Axis>> {
    let count = copy.extents.len();
    if count > 2 || copy.source_strides.len() != count || copy.destination_strides.len() != count {
        return Err(invalid(
            "derivation copy requires matching extents and strides for up to two axes",
        ));
    }
    Ok((0..count)
        .map(|index| Axis {
            count: copy.extents[index],
            source_stride: copy.source_strides[index],
            destination_stride: copy.destination_strides[index],
        })
        .collect())
}

/// Checks candidate copy/fill extents against signed payload lengths before any
/// acquisition work begins.
pub fn validate_sources(
    derivations: &[Derivation],
    catalog: &Catalog,
    limits: &Limits,
) -> Result<()> {
    for derivation in derivations {
        if derivation.operations.len() > limits.max_operations {
            return Err(invalid("derivation exceeds its operation limit"));
        }
        for operation in &derivation.operations {
            match operation {
                Operation::Copy(copy) => {
                    let length = catalog.length(copy.source.as_bytes())?;
                    let mut span = copy.contiguous;
                    let mut destination_span = copy.contiguous;
                    for axis in &copy.outer {
                        if axis.count == 0 {
                            return Err(invalid("derivation copy axis must not be empty"));
                        }
                        let advance = (axis.count - 1)
                            .checked_mul(axis.source_stride)
                            .ok_or_else(|| invalid("derivation source stride overflow"))?;
                        span = span
                            .checked_add(advance)
                            .ok_or_else(|| invalid("derivation source extent overflow"))?;
                        let advance = (axis.count - 1)
                            .checked_mul(axis.destination_stride)
                            .ok_or_else(|| invalid("derivation destination stride overflow"))?;
                        destination_span = destination_span
                            .checked_add(advance)
                            .ok_or_else(|| invalid("derivation destination extent overflow"))?;
                    }
                    if copy
                        .source_offset
                        .checked_add(span)
                        .is_none_or(|end| end > length)
                    {
                        return Err(invalid("derivation source exceeds its signed payload"));
                    }
                    if copy
                        .destination_offset
                        .checked_add(destination_span)
                        .is_none_or(|end| end > derivation.length)
                    {
                        return Err(invalid("derivation destination exceeds the target payload"));
                    }
                }
                Operation::Fill(fill) => {
                    if fill
                        .destination_offset
                        .checked_add(fill.length)
                        .is_none_or(|end| end > derivation.length)
                    {
                        return Err(invalid("derivation fill exceeds the target payload"));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Validates every derivation and returns a safe execution order.
///
/// This is the single derivation validator: schema and cache consumers convert
/// their wire records into [`Derivation`] and call this function rather than
/// re-deriving the coverage and cycle rules.
pub fn validate_set(derivations: &[Derivation], limits: &Limits) -> Result<Vec<usize>> {
    for derivation in derivations {
        validate(derivation, limits)?;
    }
    order(derivations, limits)
}

/// Full pre-acquisition validation: coverage, dependency order, and source bounds
/// against the signed payload catalog.
pub fn validate_against_catalog(
    derivations: &[Derivation],
    catalog: &Catalog,
    limits: &Limits,
) -> Result<Vec<usize>> {
    validate_sources(derivations, catalog, limits)?;
    validate_set(derivations, limits)
}

/// Orders derivations so every source produced by another derivation precedes
/// its consumer. Cycles, duplicate targets, and excessive depth are rejected.
pub fn order(derivations: &[Derivation], limits: &Limits) -> Result<Vec<usize>> {
    if derivations.len() > limits.max_derivations {
        return Err(invalid("derivation set exceeds its entry limit"));
    }
    let mut by_target = BTreeMap::new();
    for (index, derivation) in derivations.iter().enumerate() {
        if by_target.insert(derivation.target, index).is_some() {
            return Err(invalid("duplicate derivation target"));
        }
    }
    let dependencies: Vec<Vec<usize>> = derivations
        .iter()
        .map(|derivation| {
            derivation
                .operations
                .iter()
                .filter_map(|operation| match operation {
                    Operation::Copy(copy) => by_target.get(&copy.source).copied(),
                    Operation::Fill(_) => None,
                })
                .collect()
        })
        .collect();

    // Iterative post-order DFS: recursion depth would otherwise grow with the
    // chain length before the depth limit could reject it.
    let mut state = vec![0u8; derivations.len()];
    let mut depths = vec![0usize; derivations.len()];
    let mut ordered = Vec::with_capacity(derivations.len());
    for root in 0..derivations.len() {
        if state[root] == 2 {
            continue;
        }
        let mut stack: Vec<(usize, usize)> = vec![(root, 0)];
        while let Some((node, next)) = stack.pop() {
            if next == 0 {
                match state[node] {
                    2 => continue,
                    1 => return Err(invalid("derivation dependency cycle")),
                    _ => state[node] = 1,
                }
            }
            if next < dependencies[node].len() {
                let child = dependencies[node][next];
                stack.push((node, next + 1));
                match state[child] {
                    1 => return Err(invalid("derivation dependency cycle")),
                    2 => {}
                    _ => stack.push((child, 0)),
                }
            } else {
                state[node] = 2;
                let mut depth = 0usize;
                for dependency in &dependencies[node] {
                    depth = depth.max(depths[*dependency] + 1);
                }
                if depth > limits.max_depth {
                    return Err(invalid("derivation dependency depth exceeded"));
                }
                depths[node] = depth;
                ordered.push(node);
            }
        }
    }
    Ok(ordered)
}

struct Staging {
    path: PathBuf,
    file: Option<BufWriter<File>>,
    keep: bool,
}

impl Staging {
    fn create(path: &Path, capacity: usize) -> Result<Self> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| invalid(format!("create derivation output: {e}")))?;
        Ok(Self {
            path: path.to_path_buf(),
            file: Some(BufWriter::with_capacity(capacity, file)),
            keep: false,
        })
    }
    fn write(&mut self, bytes: &[u8]) -> Result<()> {
        self.file
            .as_mut()
            .ok_or_else(|| invalid("derivation output is closed"))?
            .write_all(bytes)
            .map_err(|e| invalid(format!("write derivation output: {e}")))
    }
    fn finish(mut self) -> Result<()> {
        let mut file = self
            .file
            .take()
            .ok_or_else(|| invalid("derivation output is closed"))?;
        file.flush()
            .map_err(|e| invalid(format!("flush derivation output: {e}")))?;
        file.get_ref()
            .sync_all()
            .map_err(|e| invalid(format!("sync derivation output: {e}")))?;
        self.keep = true;
        Ok(())
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        if !self.keep {
            drop(self.file.take());
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// One expanded destination run of a candidate derivation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DestinationRun {
    pub destination: u64,
    pub length: u64,
    /// Copied bytes name their payload and offset; a fill has no source.
    pub source: Option<(PayloadId, u64)>,
}

/// Expands a candidate into destination runs so a caller can map subranges,
/// such as one initializer at a time, back to canonical payload bytes.
pub fn destination_runs(derivation: &Derivation, limits: &Limits) -> Result<Vec<DestinationRun>> {
    let (mut runs, _) = expand(derivation, limits)?;
    tile(&mut runs, derivation.length)?;
    Ok(runs
        .into_iter()
        .map(|run| DestinationRun {
            destination: run.destination,
            length: run.length,
            source: match run.source {
                RunSource::Copy { payload, offset } => Some((payload, offset)),
                RunSource::Fill { .. } => None,
            },
        })
        .collect())
}

struct SourceWindow {
    payload: Option<PayloadId>,
    start: u64,
    length: usize,
    buffer: Vec<u8>,
}

impl SourceWindow {
    fn new(capacity: usize) -> Self {
        Self {
            payload: None,
            start: 0,
            length: 0,
            buffer: vec![0u8; capacity],
        }
    }

    fn bytes<'a>(
        &'a mut self,
        sources: &BTreeMap<PayloadId, Slice>,
        payload: PayloadId,
        offset: u64,
        length: u64,
        progress: Option<&crate::core::progress::AcquisitionProgress>,
    ) -> Result<&'a [u8]> {
        let requested = usize::try_from(length).map_err(|_| invalid("derivation run exceeds host range"))?;
        if requested > self.buffer.len() {
            return Err(invalid("derivation run exceeds source window"));
        }
        let slice = sources
            .get(&payload)
            .ok_or_else(|| invalid(format!("derivation lacks source payload {payload}")))?;
        let end = checked_add(offset, length, "copy source extent")?;
        if end > slice.len() {
            return Err(invalid(format!(
                "derivation source range {offset}+{length} exceeds payload {payload}"
            )));
        }
        let cached_end = self
            .start
            .checked_add(self.length as u64)
            .ok_or_else(|| invalid("derivation source window overflow"))?;
        if self.payload != Some(payload) || offset < self.start || end > cached_end {
            let count = (slice.len() - offset)
                .min(self.buffer.len() as u64) as usize;
            if count == 0 {
                return Err(invalid("derivation source window is empty"));
            }
            slice.read_at(offset, &mut self.buffer[..count])?;
            if let Some(progress) = progress {
                progress.record_local_source_bytes(count as u64);
            }
            self.payload = Some(payload);
            self.start = offset;
            self.length = count;
        }
        let start = usize::try_from(offset - self.start)
            .map_err(|_| invalid("derivation source window offset exceeds host range"))?;
        Ok(&self.buffer[start..start + requested])
    }
}

struct TargetWindow {
    start: u64,
    length: usize,
    buffer: Vec<u8>,
}

impl TargetWindow {
    fn new(capacity: usize) -> Self {
        Self {
            start: 0,
            length: 0,
            buffer: vec![0u8; capacity],
        }
    }

    fn bytes<'a>(
        &'a mut self,
        target: &Slice,
        offset: u64,
        length: u64,
        progress: Option<&crate::core::progress::AcquisitionProgress>,
    ) -> Result<&'a [u8]> {
        let requested = usize::try_from(length).map_err(|_| invalid("derivation run exceeds host range"))?;
        if requested > self.buffer.len() {
            return Err(invalid("derivation run exceeds target window"));
        }
        let end = checked_add(offset, length, "target read extent")?;
        if end > target.len() {
            return Err(invalid("derivation target read exceeds its payload"));
        }
        let cached_end = self
            .start
            .checked_add(self.length as u64)
            .ok_or_else(|| invalid("derivation target window overflow"))?;
        if offset < self.start || end > cached_end {
            let count = (target.len() - offset)
                .min(self.buffer.len() as u64) as usize;
            if count == 0 {
                return Err(invalid("derivation target window is empty"));
            }
            target.read_at(offset, &mut self.buffer[..count])?;
            if let Some(progress) = progress {
                progress.record_verification_bytes(count as u64);
            }
            self.start = offset;
            self.length = count;
        }
        let start = usize::try_from(offset - self.start)
            .map_err(|_| invalid("derivation target window offset exceeds host range"))?;
        Ok(&self.buffer[start..start + requested])
    }
}

/// Compares a declared payload byte-for-byte against a candidate derivation.
///
/// This is the combiner's acceptance test: only exact equality permits sharing,
/// and no output file or digest list is trusted in place of the bytes.
pub fn verify_output(
    target: &Slice,
    derivation: &Derivation,
    sources: &BTreeMap<PayloadId, Slice>,
    limits: &Limits,
) -> Result<()> {
    verify_output_with(target, derivation, sources, limits, None)
}

pub fn verify_output_with(
    target: &Slice,
    derivation: &Derivation,
    sources: &BTreeMap<PayloadId, Slice>,
    limits: &Limits,
    progress: Option<&crate::core::progress::AcquisitionProgress>,
) -> Result<()> {
    let (runs, patterns) = prepare_runs(derivation, limits)?;
    if target.len() != derivation.length {
        return Err(invalid(
            "candidate target length differs from its derivation",
        ));
    }
    if limits.chunk == 0 {
        return Err(invalid("derivation copy buffer must not be empty"));
    }
    let capacity = limits.chunk;
    let mut source_window = SourceWindow::new(capacity);
    let mut target_window = TargetWindow::new(capacity);
    let mut fill_buffer = vec![0u8; capacity];
    for run in &runs {
        match run.source {
            RunSource::Copy { payload, offset } => {
                let mut read = 0u64;
                while read < run.length {
                    let count = (run.length - read).min(capacity as u64) as usize;
                    let source = source_window.bytes(
                        sources,
                        payload,
                        offset + read,
                        count as u64,
                        progress,
                    )?;
                    let expected = target_window.bytes(
                        target,
                        run.destination + read,
                        count as u64,
                        progress,
                    )?;
                    if source != expected {
                        return Err(invalid(
                            "candidate derivation does not reproduce target bytes",
                        ));
                    }
                    read += count as u64;
                }
            }
            RunSource::Fill { pattern } => {
                let pattern = &patterns[pattern];
                let mut checked = 0u64;
                let mut phase = 0usize;
                while checked < run.length {
                    let count = (run.length - checked).min(capacity as u64) as usize;
                    for (index, slot) in fill_buffer[..count].iter_mut().enumerate() {
                        *slot = pattern[(phase + index) % pattern.len()];
                    }
                    let expected = target_window.bytes(
                        target,
                        run.destination + checked,
                        count as u64,
                        progress,
                    )?;
                    if fill_buffer[..count] != expected[..] {
                        return Err(invalid("candidate fill does not reproduce target bytes"));
                    }
                    phase = (phase + count) % pattern.len();
                    checked += count as u64;
                }
            }
        }
    }
    Ok(())
}

/// Materializes one derivation into a new file and verifies the target digest.
///
/// The output is exactly the target payload bytes: raw stored bytes with no
/// safetensors header, framing, or alignment padding. Acquisition installs that
/// file as a payload object (or wraps it in a companion) keyed by the target
/// identity; it must not open the result as a container.
///
/// The destination must not exist. On any failure, including a digest mismatch,
/// the partially written file is removed, so a caller never publishes unverified
/// bytes.
pub fn materialize(
    output: &Path,
    derivation: &Derivation,
    sources: &BTreeMap<PayloadId, Slice>,
    limits: &Limits,
) -> Result<()> {
    materialize_with(output, derivation, sources, limits, None)
}

pub fn materialize_with(
    output: &Path,
    derivation: &Derivation,
    sources: &BTreeMap<PayloadId, Slice>,
    limits: &Limits,
    progress: Option<&crate::core::progress::AcquisitionProgress>,
) -> Result<()> {
    let (runs, patterns) = prepare_runs(derivation, limits)?;
    if limits.chunk == 0 {
        return Err(invalid("derivation copy buffer must not be empty"));
    }
    let capacity = limits.chunk;
    for run in &runs {
        if let RunSource::Copy { payload, offset } = run.source {
            let slice = sources
                .get(&payload)
                .ok_or_else(|| invalid(format!("derivation lacks source payload {payload}")))?;
            let end = checked_add(offset, run.length, "copy source extent")?;
            if end > slice.len() {
                return Err(invalid(format!(
                    "derivation source range {offset}+{} exceeds payload {payload}",
                    run.length
                )));
            }
        }
    }
    let mut staging = Staging::create(output, capacity)?;
    let mut source_window = SourceWindow::new(capacity);
    let mut fill_buffer = vec![0u8; capacity];
    let mut hasher = digest::Context::new(&digest::SHA256);
    for run in &runs {
        match run.source {
            RunSource::Copy { payload, offset } => {
                let mut read = 0u64;
                while read < run.length {
                    let count = (run.length - read).min(capacity as u64) as usize;
                    let source = source_window.bytes(
                        sources,
                        payload,
                        offset + read,
                        count as u64,
                        progress,
                    )?;
                    staging.write(source)?;
                    if let Some(progress) = progress {
                        progress.record_materialized_copy_bytes(count as u64);
                    }
                    hasher.update(source);
                    if let Some(progress) = progress {
                        progress.record_verification_bytes(count as u64);
                    }
                    read += count as u64;
                }
            }
            RunSource::Fill { pattern } => {
                let pattern = &patterns[pattern];
                let mut written = 0u64;
                let mut phase = 0usize;
                while written < run.length {
                    let count = (run.length - written).min(capacity as u64) as usize;
                    for (index, slot) in fill_buffer[..count].iter_mut().enumerate() {
                        *slot = pattern[(phase + index) % pattern.len()];
                    }
                    staging.write(&fill_buffer[..count])?;
                    if let Some(progress) = progress {
                        progress.record_materialized_fill_bytes(count as u64);
                    }
                    hasher.update(&fill_buffer[..count]);
                    if let Some(progress) = progress {
                        progress.record_verification_bytes(count as u64);
                    }
                    phase = (phase + count) % pattern.len();
                    written += count as u64;
                }
            }
        }
    }

    let actual = hasher.finish();
    if actual.as_ref() != derivation.target.as_bytes() {
        let actual = PayloadId::from_digest(actual);
        return Err(invalid(format!(
            "derivation output digest {actual} does not match target {}",
            derivation.target
        )));
    }
    staging.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::container::{COMPANION_TENSOR, Container, Slice, write_companion};
    use crate::core::schema::{Catalog, wire};
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn payload_id(byte: u8) -> PayloadId {
        PayloadId::from_bytes([byte; 32])
    }

    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    struct Scratch(PathBuf);
    impl Scratch {
        fn new(tag: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "flare-derivation-{}-{}-{tag}",
                std::process::id(),
                SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
        fn join(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn payload(scratch: &Scratch, name: &str, bytes: &[u8]) -> (PayloadId, Slice) {
        let raw = scratch.join(&format!("{name}.raw"));
        let mut file = File::create(&raw).unwrap();
        file.write_all(bytes).unwrap();
        file.sync_all().unwrap();
        let raw = std::sync::Arc::new(File::open(&raw).unwrap());
        let payload = Slice::new(raw, 0, bytes.len() as u64).unwrap();
        let path = scratch.join(name);
        let mut output = File::create(&path).unwrap();
        write_companion(&mut output, &payload).unwrap();
        output.sync_all().unwrap();
        let container = Container::open(&path).unwrap();
        let slice = container.tensor(COMPANION_TENSOR).unwrap();
        (slice.digest().unwrap(), slice)
    }

    fn fill(offset: u64, length: u64, pattern: &[u8]) -> Operation {
        Operation::Fill(Fill {
            destination_offset: offset,
            length,
            pattern: pattern.to_vec(),
        })
    }

    fn copy(
        source: PayloadId,
        source_offset: u64,
        destination_offset: u64,
        contiguous: u64,
        outer: Vec<Axis>,
    ) -> Operation {
        Operation::Copy(Copy {
            source,
            source_offset,
            destination_offset,
            contiguous,
            outer,
        })
    }

    #[test]
    fn contiguous_copy_and_pattern_fill_materialize_exactly() {
        let scratch = Scratch::new("copy-fill");
        let (id, slice) = payload(&scratch, "source.safetensors", b"abcdefgh");
        let expected = b"abcdefgh\xab\xab\xab".to_vec();
        let derivation = Derivation {
            target: PayloadId::of(&expected),
            length: expected.len() as u64,
            operations: vec![copy(id, 0, 0, 8, vec![]), fill(8, 3, &[0xab])],
        };
        let sources = BTreeMap::from([(id, slice)]);
        let output = scratch.join("target.bin");
        materialize(&output, &derivation, &sources, &Limits::default()).unwrap();
        assert_eq!(fs::read(&output).unwrap(), expected);
    }

    #[test]
    fn progress_counts_copy_fill_and_verification_bytes_exactly() {
        let scratch = Scratch::new("progress-copy-fill");
        let (id, slice) = payload(&scratch, "source.safetensors", b"abcdefgh");
        let expected = b"abcdefgh\xab\xab\xab".to_vec();
        let derivation = Derivation {
            target: PayloadId::of(&expected),
            length: expected.len() as u64,
            operations: vec![copy(id, 0, 0, 8, vec![]), fill(8, 3, &[0xab])],
        };
        let progress = crate::core::progress::AcquisitionProgress::new();
        let output = scratch.join("progress-target.bin");
        materialize_with(
            &output,
            &derivation,
            &BTreeMap::from([(id, slice)]),
            &Limits::default(),
            Some(&progress),
        )
        .unwrap();
        let snapshot = progress.snapshot();
        assert_eq!(snapshot.local_source_bytes, 8);
        assert_eq!(snapshot.materialized_copy_bytes, 8);
        assert_eq!(snapshot.materialized_fill_bytes, 3);
        assert_eq!(snapshot.materialized_bytes(), 11);
        assert_eq!(snapshot.verification_bytes, 11);
        assert_eq!(snapshot.network_bytes, 0);
        assert_eq!(fs::read(&output).unwrap(), expected);
    }

    #[test]
    fn failed_digest_counts_work_but_removes_output() {
        let scratch = Scratch::new("progress-failed");
        let (id, slice) = payload(&scratch, "source.safetensors", b"12345678");
        let derivation = Derivation {
            target: PayloadId::of(b"not the output"),
            length: 8,
            operations: vec![copy(id, 0, 0, 8, vec![])],
        };
        let progress = crate::core::progress::AcquisitionProgress::new();
        let output = scratch.join("failed-target.bin");
        assert!(
            materialize_with(
                &output,
                &derivation,
                &BTreeMap::from([(id, slice)]),
                &Limits::default(),
                Some(&progress),
            )
            .is_err()
        );
        let snapshot = progress.snapshot();
        assert_eq!(snapshot.local_source_bytes, 8);
        assert_eq!(snapshot.materialized_copy_bytes, 8);
        assert_eq!(snapshot.verification_bytes, 8);
        assert!(!output.exists());
    }

    #[test]
    fn coalescing_and_windows_preserve_different_payloads_and_actual_reads() {
        let scratch = Scratch::new("window-payloads");
        let (first_id, first) = payload(&scratch, "first.safetensors", b"abcdefgh");
        let (second_id, second) = payload(&scratch, "second.safetensors", b"ijklmnop");
        let expected = b"abcdefghijklmnop".to_vec();
        let derivation = Derivation {
            target: PayloadId::of(&expected),
            length: expected.len() as u64,
            operations: vec![
                copy(first_id, 0, 0, 4, vec![]),
                copy(first_id, 4, 4, 4, vec![]),
                copy(second_id, 0, 8, 4, vec![]),
                copy(second_id, 4, 12, 4, vec![]),
            ],
        };
        let sources = BTreeMap::from([(first_id, first), (second_id, second)]);
        let limits = Limits {
            chunk: 4,
            ..Limits::default()
        };
        assert_eq!(prepare_runs(&derivation, &limits).unwrap().0.len(), 2);
        let progress = crate::core::progress::AcquisitionProgress::new();
        let output = scratch.join("window-target.bin");
        materialize_with(&output, &derivation, &sources, &limits, Some(&progress)).unwrap();
        assert_eq!(fs::read(&output).unwrap(), expected);
        let snapshot = progress.snapshot();
        assert_eq!(snapshot.local_source_bytes, 16);
        assert_eq!(snapshot.materialized_copy_bytes, 16);
        assert_eq!(snapshot.verification_bytes, 16);

        let target = payload(&scratch, "window-expected.safetensors", &expected).1;
        let progress = crate::core::progress::AcquisitionProgress::new();
        verify_output_with(
            &target,
            &derivation,
            &sources,
            &limits,
            Some(&progress),
        )
        .unwrap();
        let snapshot = progress.snapshot();
        assert_eq!(snapshot.local_source_bytes, 16);
        assert_eq!(snapshot.verification_bytes, 16);
        assert_eq!(snapshot.materialized_bytes(), 0);
    }

    #[test]
    fn strided_copy_and_pattern_fill_preserve_window_and_phase_edges() {
        let scratch = Scratch::new("window-strided-fill");
        let (id, slice) = payload(&scratch, "strided.safetensors", b"AAAABBBBCCCC");
        let expected = b"AAAACCCC12312".to_vec();
        let derivation = Derivation {
            target: PayloadId::of(&expected),
            length: expected.len() as u64,
            operations: vec![
                copy(
                    id,
                    0,
                    0,
                    4,
                    vec![Axis {
                        count: 2,
                        source_stride: 8,
                        destination_stride: 4,
                    }],
                ),
                fill(8, 5, b"123"),
            ],
        };
        let limits = Limits {
            chunk: 3,
            ..Limits::default()
        };
        let sources = BTreeMap::from([(id, slice)]);
        let output = scratch.join("strided-fill.bin");
        materialize(&output, &derivation, &sources, &limits).unwrap();
        assert_eq!(fs::read(&output).unwrap(), expected);

        let target = payload(&scratch, "strided-target.safetensors", &expected).1;
        verify_output(&target, &derivation, &sources, &limits).unwrap();
    }

    #[test]
    fn repeated_source_reads_are_counted_per_read() {
        let scratch = Scratch::new("progress-repeat");
        let (id, slice) = payload(&scratch, "source.safetensors", b"abcdefgh");
        let derivation = Derivation {
            target: PayloadId::of(b"abcdefgh"),
            length: 8,
            operations: vec![copy(id, 0, 0, 4, vec![]), copy(id, 4, 4, 4, vec![])],
        };
        let progress = crate::core::progress::AcquisitionProgress::new();
        let output = scratch.join("repeat-target.bin");
        materialize_with(
            &output,
            &derivation,
            &BTreeMap::from([(id, slice)]),
            &Limits::default(),
            Some(&progress),
        )
        .unwrap();
        let snapshot = progress.snapshot();
        assert_eq!(snapshot.local_source_bytes, 8);
        assert_eq!(snapshot.materialized_copy_bytes, 8);
        assert_eq!(snapshot.verification_bytes, 8);
        assert_eq!(snapshot.materialized_fill_bytes, 0);
        assert_eq!(fs::read(&output).unwrap(), b"abcdefgh");
    }

    #[test]
    fn progress_counts_verification_target_reads() {
        let scratch = Scratch::new("progress-verify");
        let (id, slice) = payload(&scratch, "source.safetensors", b"abcdefgh");
        let target = scratch.join("target.raw");
        let mut target_file = File::create(&target).unwrap();
        target_file.write_all(b"abcdefgh").unwrap();
        target_file.sync_all().unwrap();
        let target = Slice::new(
            std::sync::Arc::new(File::open(&target).unwrap()),
            0,
            8,
        )
        .unwrap();
        let derivation = Derivation {
            target: PayloadId::of(b"abcdefgh"),
            length: 8,
            operations: vec![copy(id, 0, 0, 8, vec![])],
        };
        let progress = crate::core::progress::AcquisitionProgress::new();
        verify_output_with(
            &target,
            &derivation,
            &BTreeMap::from([(id, slice)]),
            &Limits::default(),
            Some(&progress),
        )
        .unwrap();
        let snapshot = progress.snapshot();
        assert_eq!(snapshot.local_source_bytes, 8);
        assert_eq!(snapshot.verification_bytes, 8);
        assert_eq!(snapshot.materialized_bytes(), 0);
    }

    #[test]
    fn multi_axis_copy_extracts_rows() {
        let scratch = Scratch::new("rows");
        let (id, slice) = payload(&scratch, "source.safetensors", b"AAAABBBBCCCC");
        let expected = b"AAAACCCC".to_vec();
        let derivation = Derivation {
            target: PayloadId::of(&expected),
            length: 8,
            operations: vec![copy(
                id,
                0,
                0,
                4,
                vec![Axis {
                    count: 2,
                    source_stride: 8,
                    destination_stride: 4,
                }],
            )],
        };
        let output = scratch.join("rows.bin");
        materialize(
            &output,
            &derivation,
            &BTreeMap::from([(id, slice)]),
            &Limits::default(),
        )
        .unwrap();
        assert_eq!(fs::read(&output).unwrap(), expected);
    }

    #[test]
    fn overlapping_or_gapped_destinations_are_rejected() {
        let source = payload_id(7);
        for operations in [
            vec![copy(source, 0, 0, 8, vec![]), copy(source, 0, 4, 8, vec![])],
            vec![copy(source, 0, 0, 4, vec![]), copy(source, 0, 8, 4, vec![])],
            vec![copy(source, 0, 0, 7, vec![])],
        ] {
            let derivation = Derivation {
                target: payload_id(0),
                length: 16,
                operations,
            };
            assert!(validate(&derivation, &Limits::default()).is_err());
        }
    }

    #[test]
    fn source_bounds_are_checked_before_writing() {
        let scratch = Scratch::new("bounds");
        let (id, slice) = payload(&scratch, "source.safetensors", b"12345678");
        let derivation = Derivation {
            target: payload_id(0),
            length: 8,
            operations: vec![copy(id, 4, 0, 8, vec![])],
        };
        let output = scratch.join("target.bin");
        assert!(
            materialize(
                &output,
                &derivation,
                &BTreeMap::from([(id, slice)]),
                &Limits::default()
            )
            .is_err()
        );
        assert!(!output.exists());
    }

    #[test]
    fn digest_mismatch_removes_the_unverified_output() {
        let scratch = Scratch::new("digest");
        let (id, slice) = payload(&scratch, "source.safetensors", b"12345678");
        let derivation = Derivation {
            target: payload_id(0x11),
            length: 8,
            operations: vec![copy(id, 0, 0, 8, vec![])],
        };
        let output = scratch.join("target.bin");
        let error = materialize(
            &output,
            &derivation,
            &BTreeMap::from([(id, slice)]),
            &Limits::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("does not match target"), "{error}");
        assert!(!output.exists());
    }

    #[test]
    fn malformed_fills_are_rejected() {
        for (length, pattern) in [(0, vec![1u8]), (4, Vec::new())] {
            let derivation = Derivation {
                target: payload_id(0),
                length: 4,
                operations: vec![Operation::Fill(Fill {
                    destination_offset: 0,
                    length,
                    pattern,
                })],
            };
            assert!(validate(&derivation, &Limits::default()).is_err());
        }
        let oversized = Derivation {
            target: payload_id(0),
            length: 8,
            operations: vec![fill(0, 8, &[0; MAX_PATTERN + 1])],
        };
        assert!(validate(&oversized, &Limits::default()).is_err());
    }

    #[test]
    fn dependency_order_and_cycles() {
        let first = Derivation {
            target: payload_id(1),
            length: 4,
            operations: vec![Operation::Fill(Fill {
                destination_offset: 0,
                length: 4,
                pattern: vec![1],
            })],
        };
        let second = Derivation {
            target: payload_id(2),
            length: 4,
            operations: vec![copy(payload_id(1), 0, 0, 4, vec![])],
        };
        let ordered = order(&[second.clone(), first.clone()], &Limits::default()).unwrap();
        assert_eq!(ordered, vec![1, 0]);

        let cycle = Derivation {
            target: payload_id(1),
            operations: vec![copy(payload_id(1), 0, 0, 4, vec![])],
            ..second.clone()
        };
        assert!(order(&[cycle], &Limits::default()).is_err());
        assert!(order(&[first.clone(), first], &Limits::default()).is_err());
    }

    #[test]
    fn depth_and_run_limits_are_enforced() {
        let mut chain = Vec::new();
        for index in 0..6u8 {
            chain.push(Derivation {
                target: payload_id(index.wrapping_add(1)),
                length: 4,
                operations: vec![copy(payload_id(index), 0, 0, 4, vec![])],
            });
        }
        let limits = Limits {
            max_depth: 2,
            ..Limits::default()
        };
        assert!(order(&chain, &limits).is_err());

        let wide = Derivation {
            target: payload_id(0),
            length: 4,
            operations: vec![copy(
                payload_id(1),
                0,
                0,
                4,
                vec![Axis {
                    count: 8,
                    source_stride: 4,
                    destination_stride: 4,
                }],
            )],
        };
        let limits = Limits {
            max_runs: 4,
            ..Limits::default()
        };
        assert!(validate(&wide, &limits).is_err());
    }

    #[test]
    fn wire_derivations_convert_validate_and_materialize() {
        let scratch = Scratch::new("wire");
        let source = b"abcdefgh";
        let (source_id, slice) = payload(&scratch, "source.safetensors", source);
        let expected = b"abcdefgh\xab".to_vec();
        let target_id = PayloadId::of(&expected);

        let mut catalog = Catalog::new();
        catalog
            .insert(&wire::PayloadDescriptor {
                id: source_id.as_bytes().to_vec(),
                nbytes: source.len() as u64,
            })
            .unwrap();
        catalog
            .insert(&wire::PayloadDescriptor {
                id: target_id.as_bytes().to_vec(),
                nbytes: expected.len() as u64,
            })
            .unwrap();

        let set = wire::DerivationSet {
            version: 1,
            derivations: vec![wire::Derivation {
                target_payload_id: target_id.as_bytes().to_vec(),
                operations: vec![
                    wire::DerivationOperation {
                        operation: Some(wire::derivation_operation::Operation::Copy(
                            wire::DerivationCopy {
                                source_payload_id: source_id.as_bytes().to_vec(),
                                source_offset: 0,
                                destination_offset: 0,
                                contiguous_bytes: source.len() as u64,
                                extents: vec![],
                                source_strides: vec![],
                                destination_strides: vec![],
                            },
                        )),
                    },
                    wire::DerivationOperation {
                        operation: Some(wire::derivation_operation::Operation::Fill(
                            wire::DerivationFill {
                                destination_offset: source.len() as u64,
                                byte_length: 1,
                                pattern: vec![0xab],
                            },
                        )),
                    },
                ],
            }],
        };
        let derivations = from_wire(&set, &catalog, &Limits::default()).unwrap();
        assert_eq!(
            validate_set(&derivations, &Limits::default()).unwrap(),
            vec![0]
        );
        let output = scratch.join("wire.bin");
        materialize(
            &output,
            &derivations[0],
            &BTreeMap::from([(source_id, slice)]),
            &Limits::default(),
        )
        .unwrap();
        assert_eq!(fs::read(&output).unwrap(), expected);

        let mut bad = set.clone();
        if let wire::derivation_operation::Operation::Copy(copy) =
            bad.derivations[0].operations[0].operation.as_mut().unwrap()
        {
            copy.extents = vec![1, 2, 3];
        }
        assert!(from_wire(&bad, &catalog, &Limits::default()).is_err());

        let mut missing = set.clone();
        if let wire::derivation_operation::Operation::Copy(copy) = missing.derivations[0].operations
            [0]
        .operation
        .as_mut()
        .unwrap()
        {
            copy.source_payload_id = vec![9u8; 32];
        }
        assert!(from_wire(&missing, &catalog, &Limits::default()).is_err());

        let mut unsigned = set;
        unsigned.derivations[0].target_payload_id = vec![3u8; 32];
        assert!(from_wire(&unsigned, &catalog, &Limits::default()).is_err());
    }

    #[test]
    fn candidate_derivations_are_byte_verified_against_the_target() {
        let scratch = Scratch::new("verify");
        let (id, slice) = payload(&scratch, "source.safetensors", b"abcdefgh");
        let expected = b"abcdefgh\x01\x01".to_vec();
        let target = payload(&scratch, "target.safetensors", &expected).1;
        let derivation = Derivation {
            target: PayloadId::of(&expected),
            length: expected.len() as u64,
            operations: vec![copy(id, 0, 0, 8, vec![]), fill(8, 2, &[1])],
        };
        let sources = BTreeMap::from([(id, slice)]);
        verify_output(&target, &derivation, &sources, &Limits::default()).unwrap();

        let wrong = Derivation {
            target: PayloadId::of(&expected),
            length: expected.len() as u64,
            operations: vec![copy(id, 0, 0, 8, vec![]), fill(8, 2, &[2])],
        };
        assert!(verify_output(&target, &wrong, &sources, &Limits::default()).is_err());
        let short = Derivation {
            target: PayloadId::of(&expected),
            length: 9,
            operations: vec![copy(id, 0, 0, 9, vec![])],
        };
        assert!(verify_output(&target, &short, &sources, &Limits::default()).is_err());
    }

    #[test]
    fn long_forward_chains_are_rejected_without_recursion() {
        fn numbered(index: u32) -> PayloadId {
            let mut bytes = [0u8; 32];
            bytes[..4].copy_from_slice(&index.to_le_bytes());
            PayloadId::from_bytes(bytes)
        }
        let chain: Vec<Derivation> = (0..20_000u32)
            .map(|index| Derivation {
                target: numbered(index + 1),
                length: 4,
                operations: vec![copy(numbered(index), 0, 0, 4, vec![])],
            })
            .collect();
        let error = order(&chain, &Limits::default()).unwrap_err().to_string();
        assert!(error.contains("depth"), "{error}");
    }

    #[test]
    fn checked_arithmetic_rejects_overflowing_axes() {
        let derivation = Derivation {
            target: payload_id(0),
            length: 8,
            operations: vec![copy(
                payload_id(1),
                0,
                0,
                4,
                vec![Axis {
                    count: 2,
                    source_stride: u64::MAX,
                    destination_stride: 4,
                }],
            )],
        };
        assert!(validate(&derivation, &Limits::default()).is_err());
    }
}
