/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Inverse byte-copy planning for missing canonical payloads.
//!
//! A validated forward derivation maps canonical source bytes into a derived
//! target. When the canonical payload is absent but a derived target is present,
//! the same affine byte copies can be inverted. This module plans only those
//! inverse copies: it emits the existing [`Copy`] operations, never fill
//! operations, new schema fields, or source bytes.
//!
//! Fills are not invertible. A missing payload is reported as resolved only when
//! every one of its bytes is covered by an injective inverse mapping from bytes
//! that are available, directly or through another planned missing payload.
//! Overlapping candidates are clipped to their uncovered regions so the selected
//! output remains injective. The caller must run each returned derivation through
//! the existing materializer, which verifies the exact target digest.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::Result;

use super::derivation::{self, Axis, Copy, Derivation, Limits, Operation};
use super::schema::Catalog;
use super::{PayloadId, invalid};

/// Result of inverse planning.
///
/// `derivations` is ordered by dependency and contains the requested payloads
/// that could be recovered plus any missing intermediate payloads they need.
/// `unresolved` names requested payloads with no complete inverse coverage; the
/// acquisition path must fall back to its normal forward or network source for
/// those.
#[derive(Clone, Debug)]
pub struct InversePlan {
    pub derivations: Vec<Derivation>,
    pub unresolved: BTreeSet<PayloadId>,
}

/// Plans inverse derivations for `missing` payloads from `available` bytes.
///
/// `forward` must contain byte-copy mappings that pass the existing derivation
/// validators. `catalog` supplies authenticated lengths. `available` names
/// payloads whose exact bytes are already present. Pass every missing canonical
/// dependency, not only the caller's required set: an alternative target held
/// locally may be needed to recover a dependency of a required payload. Every
/// returned operation is a bounded [`Copy`]; fills are ignored because they
/// cannot recover a source.
///
/// The result is an acquisition hint, not authorization. Callers must still
/// materialize each derivation through [`derivation::materialize`] to verify the
/// exact target digest before publishing it.
pub fn plan(
    catalog: &Catalog,
    forward: &[Derivation],
    missing: &BTreeSet<PayloadId>,
    available: &BTreeSet<PayloadId>,
    limits: &Limits,
) -> Result<InversePlan> {
    if forward.len() > limits.max_derivations {
        return Err(invalid("forward derivation set exceeds its entry limit"));
    }

    // Build the raw source -> target graph first. It contains only identities;
    // hints are validated later and only when their source lies in the reverse
    // dependency cone of the requested missing payload.
    let mut raw: BTreeMap<PayloadId, BTreeSet<PayloadId>> = BTreeMap::new();
    for derivation in forward {
        for operation in &derivation.operations {
            if let Operation::Copy(copy) = operation {
                raw.entry(copy.source)
                    .or_default()
                    .insert(derivation.target);
            }
        }
    }
    let mut needed: BTreeSet<PayloadId> = BTreeSet::new();
    let mut queue: VecDeque<PayloadId> = VecDeque::new();
    for id in missing {
        if !available.contains(id) && needed.insert(*id) {
            queue.push_back(*id);
        }
    }
    while let Some(source) = queue.pop_front() {
        let Some(targets) = raw.get(&source) else {
            continue;
        };
        for target in targets {
            if !available.contains(target) && needed.insert(*target) {
                queue.push_back(*target);
            }
        }
    }

    let mut by_target: BTreeMap<PayloadId, &Derivation> = BTreeMap::new();
    let mut consumers: BTreeMap<PayloadId, BTreeSet<PayloadId>> = BTreeMap::new();
    for derivation in forward {
        let relevant = derivation.operations.iter().any(
            |operation| matches!(operation, Operation::Copy(copy) if needed.contains(&copy.source)),
        );
        if !relevant {
            continue;
        }
        // Optional acquisition hints are untrusted. A malformed or otherwise
        // unusable unrelated hint must not poison a valid path, so reject only
        // this hint and let the requested payload remain unresolved if no other
        // candidate covers it.
        let target_length_ok = matches!(
            catalog.length(derivation.target.as_bytes()),
            Ok(length) if length == derivation.length
        );
        if !target_length_ok
            || derivation::validate(derivation, limits).is_err()
            || derivation::validate_sources(std::slice::from_ref(derivation), catalog, limits)
                .is_err()
        {
            continue;
        }
        if by_target.insert(derivation.target, derivation).is_some() {
            return Err(invalid("duplicate forward derivation target"));
        }
        for operation in &derivation.operations {
            if let Operation::Copy(copy) = operation {
                if needed.contains(&copy.source) {
                    consumers
                        .entry(copy.source)
                        .or_default()
                        .insert(derivation.target);
                }
            }
        }
    }

    let mut unresolved: BTreeSet<PayloadId> = needed;
    let mut budget = Budget {
        operations: limits.max_operations,
        runs: limits.max_runs,
    };
    let mut plans: BTreeMap<PayloadId, Plan> = BTreeMap::new();
    loop {
        if plans.len() >= limits.max_derivations {
            break;
        }
        let pending: Vec<PayloadId> = unresolved.iter().copied().collect();
        let mut progress = false;
        for id in pending {
            if plans.contains_key(&id) {
                unresolved.remove(&id);
                continue;
            }
            if budget.operations == 0 || budget.runs == 0 {
                break;
            }
            if let Some((plan, usage)) = build_plan(
                id, catalog, &by_target, &consumers, &plans, available, limits, &budget,
            )? {
                budget.operations -= usage.operations;
                budget.runs -= usage.runs;
                plans.insert(id, plan);
                unresolved.remove(&id);
                progress = true;
            }
        }
        if !progress {
            break;
        }
    }

    let mut selected: BTreeMap<PayloadId, Derivation> = BTreeMap::new();
    let mut visiting: BTreeSet<PayloadId> = BTreeSet::new();
    let mut unresolved_requested: BTreeSet<PayloadId> = BTreeSet::new();
    for id in missing {
        if available.contains(id) {
            continue;
        }
        if !collect(*id, &plans, &mut selected, &mut visiting) {
            unresolved_requested.insert(*id);
        }
    }
    let derivations: Vec<Derivation> = selected.into_values().collect();
    if derivations.len() > limits.max_derivations {
        return Err(invalid("inverse derivation set exceeds its entry limit"));
    }
    let order = derivation::validate_set(&derivations, limits)?;
    Ok(InversePlan {
        derivations: order
            .into_iter()
            .map(|index| derivations[index].clone())
            .collect(),
        unresolved: unresolved_requested,
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct Budget {
    operations: usize,
    runs: usize,
}

#[derive(Clone, Copy, Debug)]
struct Usage {
    operations: usize,
    runs: usize,
}

#[derive(Clone, Debug)]
struct Plan {
    derivation: Derivation,
    depth: usize,
    dependencies: BTreeSet<PayloadId>,
}

struct Candidate {
    copy: Copy,
    runs: Vec<Run>,
    target: PayloadId,
    depth: usize,
    dependency: Option<PayloadId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Run {
    destination: u64,
    length: u64,
    source: PayloadId,
    source_offset: u64,
}

struct Accepted {
    runs: Vec<Run>,
    clipped: bool,
}

/// Builds one missing payload from all currently resolved inverse candidates.
///
/// A `None` result means the payload is not yet fully covered. It may become
/// covered on a later round after another missing intermediate payload resolves.
#[allow(clippy::too_many_arguments)]
fn build_plan(
    id: PayloadId,
    catalog: &Catalog,
    by_target: &BTreeMap<PayloadId, &Derivation>,
    consumers: &BTreeMap<PayloadId, BTreeSet<PayloadId>>,
    plans: &BTreeMap<PayloadId, Plan>,
    available: &BTreeSet<PayloadId>,
    limits: &Limits,
    budget: &Budget,
) -> Result<Option<(Plan, Usage)>> {
    let length = catalog.length(id.as_bytes())?;
    let Some(entries) = consumers.get(&id) else {
        return Ok(None);
    };
    let mut candidates: Vec<Candidate> = Vec::new();
    let mut candidate_operations = 0usize;
    let mut candidate_runs = 0usize;
    for target in entries {
        let Some(derivation) = by_target.get(target) else {
            continue;
        };
        if *target == id {
            continue;
        }
        let (depth, dependency) = if available.contains(target) {
            (0usize, None)
        } else if let Some(plan) = plans.get(target) {
            if plan.depth + 1 > limits.max_depth {
                continue;
            }
            (plan.depth + 1, Some(*target))
        } else {
            continue;
        };
        let source_length = catalog.length(target.as_bytes())?;
        for operation in &derivation.operations {
            let Operation::Copy(copy) = operation else {
                continue;
            };
            if copy.source != id {
                continue;
            }
            // Bound candidate storage and expansion before cloning or
            // materializing runs from an untrusted forward map.
            if candidate_operations + 1 > limits.max_operations
                || candidate_operations + 1 > budget.operations
            {
                continue;
            }
            let Ok(count) = copy_run_count(copy) else {
                continue;
            };
            let Ok(count) = usize::try_from(count) else {
                continue;
            };
            if candidate_runs + count > limits.max_runs || candidate_runs + count > budget.runs {
                continue;
            }
            let inverse = invert(copy, *target);
            let Ok(runs) = expand_copy(&inverse, source_length, length, limits) else {
                continue;
            };
            if candidate_operations + 1 > limits.max_operations
                || candidate_operations + 1 > budget.operations
                || candidate_runs + runs.len() > limits.max_runs
                || candidate_runs + runs.len() > budget.runs
            {
                continue;
            }
            candidate_operations += 1;
            candidate_runs += runs.len();
            candidates.push(Candidate {
                copy: inverse,
                runs,
                target: *target,
                depth,
                dependency,
            });
        }
    }
    if candidates.is_empty() {
        return Ok(None);
    }
    candidates.sort_by(|left, right| {
        let left_first = left.runs.first().expect("inverse copy produces a run");
        let right_first = right.runs.first().expect("inverse copy produces a run");
        left_first
            .destination
            .cmp(&right_first.destination)
            .then_with(|| right_first.length.cmp(&left_first.length))
            .then_with(|| left.target.cmp(&right.target))
            .then_with(|| {
                left.copy
                    .destination_offset
                    .cmp(&right.copy.destination_offset)
            })
            .then_with(|| left.copy.source_offset.cmp(&right.copy.source_offset))
            .then_with(|| left.copy.contiguous.cmp(&right.copy.contiguous))
    });

    let mut coverage = Coverage::new(length);
    let mut operations = Vec::new();
    let mut selected_runs = 0usize;
    let mut depth = 0usize;
    let mut dependencies = BTreeSet::new();
    for candidate in candidates {
        let Some(accepted) = coverage.accept(&candidate.runs) else {
            continue;
        };
        if accepted.runs.is_empty() {
            continue;
        }
        let operation_count = if accepted.clipped {
            accepted.runs.len()
        } else {
            1
        };
        if operations.len() + operation_count > limits.max_operations
            || operations.len() + operation_count > budget.operations
        {
            continue;
        }
        if selected_runs + accepted.runs.len() > limits.max_runs
            || selected_runs + accepted.runs.len() > budget.runs
        {
            continue;
        }
        if accepted.clipped {
            for run in &accepted.runs {
                operations.push(Operation::Copy(Copy {
                    source: run.source,
                    source_offset: run.source_offset,
                    destination_offset: run.destination,
                    contiguous: run.length,
                    outer: Vec::new(),
                }));
            }
        } else {
            operations.push(Operation::Copy(candidate.copy));
        }
        selected_runs += accepted.runs.len();
        coverage.insert(&accepted.runs);
        depth = depth.max(candidate.depth);
        if let Some(dependency) = candidate.dependency {
            dependencies.insert(dependency);
        }
        if coverage.complete() {
            break;
        }
    }
    if !coverage.complete() {
        return Ok(None);
    }
    let operation_count = operations.len();
    Ok(Some((
        Plan {
            derivation: Derivation {
                target: id,
                length,
                operations,
            },
            depth,
            dependencies,
        },
        Usage {
            operations: operation_count,
            runs: selected_runs,
        },
    )))
}

fn collect(
    id: PayloadId,
    plans: &BTreeMap<PayloadId, Plan>,
    selected: &mut BTreeMap<PayloadId, Derivation>,
    visiting: &mut BTreeSet<PayloadId>,
) -> bool {
    if selected.contains_key(&id) {
        return true;
    }
    if !visiting.insert(id) {
        return false;
    }
    let Some(plan) = plans.get(&id) else {
        visiting.remove(&id);
        return false;
    };
    for dependency in &plan.dependencies {
        if !collect(*dependency, plans, selected, visiting) {
            visiting.remove(&id);
            return false;
        }
    }
    visiting.remove(&id);
    selected.insert(id, plan.derivation.clone());
    true
}

fn invert(copy: &Copy, target: PayloadId) -> Copy {
    Copy {
        source: target,
        source_offset: copy.destination_offset,
        destination_offset: copy.source_offset,
        contiguous: copy.contiguous,
        outer: copy
            .outer
            .iter()
            .map(|axis| Axis {
                count: axis.count,
                source_stride: axis.destination_stride,
                destination_stride: axis.source_stride,
            })
            .collect(),
    }
}

/// Number of expanded runs without allocating or reading payloads.
fn copy_run_count(copy: &Copy) -> Result<u64> {
    if copy.contiguous == 0 || copy.outer.len() > 2 {
        return Err(invalid("invalid inverse copy"));
    }
    let mut count = 1u64;
    for axis in &copy.outer {
        if axis.count == 0 {
            return Err(invalid("inverse copy axis must not be empty"));
        }
        count = count
            .checked_mul(axis.count)
            .ok_or_else(|| invalid("inverse copy run count overflow"))?;
    }
    Ok(count)
}

/// Expands one inverse copy into destination runs for coverage checking.
///
/// The returned copy keeps its rectangular structure; this expansion only
/// bounds and checks the runs. No per-byte work is performed.
fn expand_copy(
    copy: &Copy,
    source_length: u64,
    target_length: u64,
    limits: &Limits,
) -> Result<Vec<Run>> {
    let count = copy_run_count(copy)?;
    if count > limits.max_runs as u64 {
        return Err(invalid("inverse copy exceeds its run limit"));
    }

    let mut source_span = copy.contiguous;
    let mut target_span = copy.contiguous;
    for axis in &copy.outer {
        let advance = (axis.count - 1)
            .checked_mul(axis.source_stride)
            .ok_or_else(|| invalid("inverse copy source stride overflow"))?;
        source_span = source_span
            .checked_add(advance)
            .ok_or_else(|| invalid("inverse copy source extent overflow"))?;
        let advance = (axis.count - 1)
            .checked_mul(axis.destination_stride)
            .ok_or_else(|| invalid("inverse copy destination stride overflow"))?;
        target_span = target_span
            .checked_add(advance)
            .ok_or_else(|| invalid("inverse copy destination extent overflow"))?;
    }
    if copy
        .source_offset
        .checked_add(source_span)
        .is_none_or(|end| end > source_length)
    {
        return Err(invalid(
            "inverse copy source exceeds its authenticated payload",
        ));
    }
    if copy
        .destination_offset
        .checked_add(target_span)
        .is_none_or(|end| end > target_length)
    {
        return Err(invalid("inverse copy exceeds the missing payload"));
    }

    let mut runs = Vec::new();
    runs.try_reserve(count.min(1024) as usize)
        .map_err(|_| invalid("inverse copy run storage is unavailable"))?;
    let mut indexes = vec![0u64; copy.outer.len()];
    loop {
        let mut source = copy.source_offset;
        let mut destination = copy.destination_offset;
        for (axis, index) in copy.outer.iter().zip(&indexes) {
            source = source
                .checked_add(
                    index
                        .checked_mul(axis.source_stride)
                        .ok_or_else(|| invalid("inverse copy source stride overflow"))?,
                )
                .ok_or_else(|| invalid("inverse copy source offset overflow"))?;
            destination = destination
                .checked_add(
                    index
                        .checked_mul(axis.destination_stride)
                        .ok_or_else(|| invalid("inverse copy destination stride overflow"))?,
                )
                .ok_or_else(|| invalid("inverse copy destination offset overflow"))?;
        }
        runs.push(Run {
            destination,
            length: copy.contiguous,
            source: copy.source,
            source_offset: source,
        });
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
    runs.sort_by_key(|run| (run.destination, run.source_offset, run.source));
    Ok(runs)
}

#[derive(Clone, Copy)]
struct Interval {
    length: u64,
}

struct Coverage {
    length: u64,
    intervals: BTreeMap<u64, Interval>,
}

impl Coverage {
    fn new(length: u64) -> Self {
        Self {
            length,
            intervals: BTreeMap::new(),
        }
    }

    /// Returns the uncovered parts of one candidate.
    ///
    /// `None` means the candidate's own expanded runs overlap, so it is not an
    /// injective copy. Existing coverage is always retained; overlapping
    /// candidate regions are clipped away and reported through `clipped`.
    fn accept(&self, runs: &[Run]) -> Option<Accepted> {
        let mut previous_end = 0u64;
        for run in runs {
            let end = run.destination.checked_add(run.length)?;
            if run.destination < previous_end {
                return None;
            }
            previous_end = end;
        }

        let mut novel = Vec::new();
        let mut clipped = false;
        for run in runs {
            let end = run.destination.checked_add(run.length)?;
            let mut cursor = run.destination;
            for (&start, interval) in self.intervals.range(..end) {
                let existing_end = start.checked_add(interval.length)?;
                if existing_end <= cursor {
                    continue;
                }
                if start >= end {
                    break;
                }
                clipped = true;
                if start > cursor {
                    let length = start.min(end) - cursor;
                    if length > 0 {
                        novel.push(Run {
                            destination: cursor,
                            length,
                            source: run.source,
                            source_offset: run
                                .source_offset
                                .checked_add(cursor - run.destination)?,
                        });
                    }
                }
                cursor = cursor.max(existing_end);
                if cursor >= end {
                    break;
                }
            }
            if cursor < end {
                novel.push(Run {
                    destination: cursor,
                    length: end - cursor,
                    source: run.source,
                    source_offset: run.source_offset.checked_add(cursor - run.destination)?,
                });
            }
        }
        Some(Accepted {
            runs: novel,
            clipped,
        })
    }

    fn insert(&mut self, runs: &[Run]) {
        for run in runs {
            self.intervals
                .entry(run.destination)
                .or_insert(Interval { length: run.length });
        }
    }

    fn complete(&self) -> bool {
        let mut cursor = 0u64;
        for (&start, interval) in &self.intervals {
            if start != cursor {
                return false;
            }
            let Some(end) = start.checked_add(interval.length) else {
                return false;
            };
            cursor = end;
        }
        cursor == self.length
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;
    use crate::core::container::Slice;
    use crate::core::schema::wire;

    fn payload(byte: u8) -> PayloadId {
        PayloadId::from_bytes([byte; 32])
    }

    fn catalog(entries: &[(PayloadId, u64)]) -> Catalog {
        let mut catalog = Catalog::new();
        for (id, nbytes) in entries {
            catalog
                .insert(&wire::PayloadDescriptor {
                    id: id.as_bytes().to_vec(),
                    nbytes: *nbytes,
                })
                .unwrap();
        }
        catalog
    }

    fn copy(
        source: PayloadId,
        source_offset: u64,
        destination_offset: u64,
        contiguous: u64,
        outer: Vec<Axis>,
    ) -> Copy {
        Copy {
            source,
            source_offset,
            destination_offset,
            contiguous,
            outer,
        }
    }

    fn derivation(target: PayloadId, length: u64, operations: Vec<Operation>) -> Derivation {
        Derivation {
            target,
            length,
            operations,
        }
    }

    fn limits(depth: usize) -> Limits {
        Limits {
            max_depth: depth,
            ..Limits::default()
        }
    }

    #[test]
    fn contiguous_inverse_recovers_canonical_payload() {
        let canonical = payload(1);
        let derived = payload(2);
        let catalog = catalog(&[(canonical, 8), (derived, 8)]);
        let forward = vec![derivation(
            derived,
            8,
            vec![Operation::Copy(copy(canonical, 0, 0, 8, vec![]))],
        )];

        let result = plan(
            &catalog,
            &forward,
            &BTreeSet::from([canonical]),
            &BTreeSet::from([derived]),
            &Limits::default(),
        )
        .unwrap();

        assert!(result.unresolved.is_empty());
        assert_eq!(result.derivations.len(), 1);
        assert_eq!(result.derivations[0].target, canonical);
        assert_eq!(
            result.derivations[0].operations,
            vec![Operation::Copy(copy(derived, 0, 0, 8, vec![]))]
        );
        derivation::validate(&result.derivations[0], &Limits::default()).unwrap();
    }

    #[test]
    fn rectangular_inverse_preserves_the_forward_shape() {
        let canonical = payload(1);
        let derived = payload(2);
        let catalog = catalog(&[(canonical, 16), (derived, 16)]);
        let axes = vec![
            Axis {
                count: 2,
                source_stride: 8,
                destination_stride: 8,
            },
            Axis {
                count: 2,
                source_stride: 4,
                destination_stride: 4,
            },
        ];
        let forward = vec![derivation(
            derived,
            16,
            vec![Operation::Copy(copy(canonical, 0, 0, 4, axes.clone()))],
        )];

        let result = plan(
            &catalog,
            &forward,
            &BTreeSet::from([canonical]),
            &BTreeSet::from([derived]),
            &Limits::default(),
        )
        .unwrap();

        assert!(result.unresolved.is_empty());
        assert_eq!(result.derivations.len(), 1);
        assert_eq!(result.derivations[0].operations.len(), 1);
        assert_eq!(
            result.derivations[0].operations,
            vec![Operation::Copy(copy(derived, 0, 0, 4, axes))]
        );
        let runs =
            derivation::destination_runs(&result.derivations[0], &Limits::default()).unwrap();
        assert_eq!(runs.len(), 4);
        derivation::validate(&result.derivations[0], &Limits::default()).unwrap();
    }

    #[test]
    fn multiple_targets_jointly_cover_one_canonical_payload() {
        let canonical = payload(1);
        let first = payload(2);
        let second = payload(3);
        let catalog = catalog(&[(canonical, 8), (first, 4), (second, 4)]);
        let forward = vec![
            derivation(
                first,
                4,
                vec![Operation::Copy(copy(canonical, 0, 0, 4, vec![]))],
            ),
            derivation(
                second,
                4,
                vec![Operation::Copy(copy(canonical, 4, 0, 4, vec![]))],
            ),
        ];

        let result = plan(
            &catalog,
            &forward,
            &BTreeSet::from([canonical]),
            &BTreeSet::from([first, second]),
            &Limits::default(),
        )
        .unwrap();

        assert!(result.unresolved.is_empty());
        assert_eq!(result.derivations.len(), 1);
        assert_eq!(result.derivations[0].operations.len(), 2);
        derivation::validate(&result.derivations[0], &Limits::default()).unwrap();
    }

    #[test]
    fn incomplete_source_coverage_is_unresolved() {
        let canonical = payload(1);
        let derived = payload(2);
        let catalog = catalog(&[(canonical, 8), (derived, 4)]);
        let forward = vec![derivation(
            derived,
            4,
            vec![Operation::Copy(copy(canonical, 0, 0, 4, vec![]))],
        )];

        let result = plan(
            &catalog,
            &forward,
            &BTreeSet::from([canonical]),
            &BTreeSet::from([derived]),
            &Limits::default(),
        )
        .unwrap();
        assert!(result.derivations.is_empty());
        assert_eq!(result.unresolved, BTreeSet::from([canonical]));
    }

    #[test]
    fn non_injective_forward_mapping_is_unresolved() {
        let canonical = payload(1);
        let derived = payload(2);
        let catalog = catalog(&[(canonical, 8), (derived, 8)]);
        let forward = vec![derivation(
            derived,
            8,
            vec![Operation::Copy(copy(
                canonical,
                0,
                0,
                4,
                vec![Axis {
                    count: 2,
                    source_stride: 0,
                    destination_stride: 4,
                }],
            ))],
        )];

        let result = plan(
            &catalog,
            &forward,
            &BTreeSet::from([canonical]),
            &BTreeSet::from([derived]),
            &Limits::default(),
        )
        .unwrap();
        assert!(result.derivations.is_empty());
        assert_eq!(result.unresolved, BTreeSet::from([canonical]));
    }

    #[test]
    fn partially_overlapping_targets_are_clipped_to_a_complete_cover() {
        let canonical_bytes: Vec<u8> = (0..8).collect();
        let first_bytes = canonical_bytes[0..6].to_vec();
        let second_bytes = canonical_bytes[4..8].to_vec();
        let canonical = PayloadId::of(&canonical_bytes);
        let first = PayloadId::of(&first_bytes);
        let second = PayloadId::of(&second_bytes);
        let catalog = catalog(&[(canonical, 8), (first, 6), (second, 4)]);
        let forward = vec![
            derivation(
                first,
                6,
                vec![Operation::Copy(copy(canonical, 0, 0, 6, vec![]))],
            ),
            derivation(
                second,
                4,
                vec![Operation::Copy(copy(canonical, 4, 0, 4, vec![]))],
            ),
        ];

        let result = plan(
            &catalog,
            &forward,
            &BTreeSet::from([canonical]),
            &BTreeSet::from([first, second]),
            &Limits::default(),
        )
        .unwrap();

        assert!(result.unresolved.is_empty());
        assert_eq!(result.derivations.len(), 1);
        assert_eq!(result.derivations[0].operations.len(), 2);
        match &result.derivations[0].operations[1] {
            Operation::Copy(copy) => {
                assert_eq!(copy.source, second);
                assert_eq!(copy.source_offset, 2);
                assert_eq!(copy.destination_offset, 6);
                assert_eq!(copy.contiguous, 2);
                assert!(copy.outer.is_empty());
            }
            Operation::Fill(_) => panic!("inverse planning must not emit fills"),
        }
        derivation::validate(&result.derivations[0], &Limits::default()).unwrap();

        let scratch = Scratch::new("overlap-digest");
        let (first_id, first_slice) = slice(&scratch, "first", &first_bytes);
        let (second_id, second_slice) = slice(&scratch, "second", &second_bytes);
        assert_eq!(first_id, first);
        assert_eq!(second_id, second);
        let output = scratch.join("canonical.bin");
        derivation::materialize(
            &output,
            &result.derivations[0],
            &BTreeMap::from([(first, first_slice), (second, second_slice)]),
            &Limits::default(),
        )
        .unwrap();
        assert_eq!(fs::read(output).unwrap(), canonical_bytes);
    }

    #[test]
    fn repeated_region_is_deduplicated() {
        let canonical = payload(1);
        let first = payload(2);
        let second = payload(3);
        let catalog = catalog(&[(canonical, 4), (first, 4), (second, 4)]);
        let forward = vec![
            derivation(
                first,
                4,
                vec![Operation::Copy(copy(canonical, 0, 0, 4, vec![]))],
            ),
            derivation(
                second,
                4,
                vec![Operation::Copy(copy(canonical, 0, 0, 4, vec![]))],
            ),
        ];

        let result = plan(
            &catalog,
            &forward,
            &BTreeSet::from([canonical]),
            &BTreeSet::from([first, second]),
            &Limits::default(),
        )
        .unwrap();

        assert!(result.unresolved.is_empty());
        assert_eq!(result.derivations.len(), 1);
        assert_eq!(result.derivations[0].operations.len(), 1);
        derivation::validate(&result.derivations[0], &Limits::default()).unwrap();
    }

    #[test]
    fn mixed_duplicate_and_novel_rectangular_runs_are_clipped() {
        let canonical_bytes: Vec<u8> = (0..16).collect();
        let first_bytes = canonical_bytes[0..8].to_vec();
        let second_bytes: Vec<u8> = [0usize, 8, 4, 12]
            .into_iter()
            .flat_map(|start| canonical_bytes[start..start + 4].to_vec())
            .collect();
        let canonical = PayloadId::of(&canonical_bytes);
        let first = PayloadId::of(&first_bytes);
        let second = PayloadId::of(&second_bytes);
        let catalog = catalog(&[(canonical, 16), (first, 8), (second, 16)]);
        let forward = vec![
            derivation(
                first,
                8,
                vec![Operation::Copy(copy(canonical, 0, 0, 8, vec![]))],
            ),
            derivation(
                second,
                16,
                vec![Operation::Copy(copy(
                    canonical,
                    0,
                    0,
                    4,
                    vec![
                        Axis {
                            count: 2,
                            source_stride: 8,
                            destination_stride: 4,
                        },
                        Axis {
                            count: 2,
                            source_stride: 4,
                            destination_stride: 8,
                        },
                    ],
                ))],
            ),
        ];

        let result = plan(
            &catalog,
            &forward,
            &BTreeSet::from([canonical]),
            &BTreeSet::from([first, second]),
            &Limits::default(),
        )
        .unwrap();

        assert!(result.unresolved.is_empty());
        assert_eq!(result.derivations.len(), 1);
        assert_eq!(result.derivations[0].operations.len(), 3);
        for (operation, source_offset, destination_offset) in [
            (&result.derivations[0].operations[1], 4, 8),
            (&result.derivations[0].operations[2], 12, 12),
        ] {
            match operation {
                Operation::Copy(copy) => {
                    assert_eq!(copy.source, second);
                    assert_eq!(copy.source_offset, source_offset);
                    assert_eq!(copy.destination_offset, destination_offset);
                    assert_eq!(copy.contiguous, 4);
                    assert!(copy.outer.is_empty());
                }
                Operation::Fill(_) => panic!("inverse planning must not emit fills"),
            }
        }
        derivation::validate(&result.derivations[0], &Limits::default()).unwrap();

        let scratch = Scratch::new("mixed-digest");
        let (first_id, first_slice) = slice(&scratch, "first", &first_bytes);
        let (second_id, second_slice) = slice(&scratch, "second", &second_bytes);
        assert_eq!(first_id, first);
        assert_eq!(second_id, second);
        let output = scratch.join("canonical.bin");
        derivation::materialize(
            &output,
            &result.derivations[0],
            &BTreeMap::from([(first, first_slice), (second, second_slice)]),
            &Limits::default(),
        )
        .unwrap();
        assert_eq!(fs::read(output).unwrap(), canonical_bytes);
    }

    #[test]
    fn unrelated_inverse_path_does_not_consume_budget() {
        let requested = payload(2);
        let unrelated = payload(1);
        let requested_target = payload(3);
        let unrelated_target = payload(4);
        let catalog = catalog(&[
            (requested, 4),
            (unrelated, 4),
            (requested_target, 4),
            (unrelated_target, 4),
        ]);
        let forward = vec![
            derivation(
                requested_target,
                4,
                vec![Operation::Copy(copy(requested, 0, 0, 4, vec![]))],
            ),
            derivation(
                unrelated_target,
                4,
                vec![Operation::Copy(copy(unrelated, 0, 0, 4, vec![]))],
            ),
        ];
        let limits = Limits {
            max_operations: 1,
            ..Limits::default()
        };

        let result = plan(
            &catalog,
            &forward,
            &BTreeSet::from([requested]),
            &BTreeSet::from([requested_target, unrelated_target]),
            &limits,
        )
        .unwrap();
        assert!(result.unresolved.is_empty());
        assert_eq!(result.derivations.len(), 1);
        assert_eq!(result.derivations[0].target, requested);
    }

    #[test]
    fn unrelated_malformed_hint_does_not_poison_a_valid_path() {
        let requested = payload(1);
        let available_target = payload(2);
        let unknown_source = payload(3);
        let unknown_target = payload(4);
        let catalog = catalog(&[(requested, 4), (available_target, 4)]);
        let forward = vec![
            derivation(
                available_target,
                4,
                vec![Operation::Copy(copy(requested, 0, 0, 4, vec![]))],
            ),
            derivation(
                unknown_target,
                4,
                vec![Operation::Copy(copy(unknown_source, 0, 0, 4, vec![]))],
            ),
        ];

        let result = plan(
            &catalog,
            &forward,
            &BTreeSet::from([requested]),
            &BTreeSet::from([available_target]),
            &Limits::default(),
        )
        .unwrap();
        assert!(result.unresolved.is_empty());
        assert_eq!(result.derivations.len(), 1);
        assert_eq!(result.derivations[0].target, requested);
    }

    #[test]
    fn chained_inverse_respects_the_depth_limit() {
        let canonical = payload(1);
        let intermediate = payload(2);
        let available = payload(3);
        let catalog = catalog(&[(canonical, 4), (intermediate, 4), (available, 4)]);
        let forward = vec![
            derivation(
                intermediate,
                4,
                vec![Operation::Copy(copy(canonical, 0, 0, 4, vec![]))],
            ),
            derivation(
                available,
                4,
                vec![Operation::Copy(copy(intermediate, 0, 0, 4, vec![]))],
            ),
        ];

        let result = plan(
            &catalog,
            &forward,
            &BTreeSet::from([canonical]),
            &BTreeSet::from([available]),
            &limits(0),
        )
        .unwrap();
        assert!(result.derivations.is_empty());
        assert_eq!(result.unresolved, BTreeSet::from([canonical]));

        let result = plan(
            &catalog,
            &forward,
            &BTreeSet::from([canonical]),
            &BTreeSet::from([available]),
            &limits(1),
        )
        .unwrap();
        assert!(result.unresolved.is_empty());
        assert_eq!(result.derivations.len(), 2);
        assert_eq!(result.derivations[0].target, intermediate);
        assert_eq!(result.derivations[1].target, canonical);
    }

    #[test]
    fn dependency_cycle_is_unresolved() {
        let canonical = payload(1);
        let derived = payload(2);
        let catalog = catalog(&[(canonical, 4), (derived, 4)]);
        let forward = vec![
            derivation(
                derived,
                4,
                vec![Operation::Copy(copy(canonical, 0, 0, 4, vec![]))],
            ),
            derivation(
                canonical,
                4,
                vec![Operation::Copy(copy(derived, 0, 0, 4, vec![]))],
            ),
        ];

        let result = plan(
            &catalog,
            &forward,
            &BTreeSet::from([canonical]),
            &BTreeSet::from([]),
            &Limits::default(),
        )
        .unwrap();
        assert!(result.derivations.is_empty());
        assert_eq!(result.unresolved, BTreeSet::from([canonical]));
    }

    #[test]
    fn operation_and_run_limits_return_unresolved() {
        let canonical = payload(1);
        let first = payload(2);
        let second = payload(3);
        let catalog = catalog(&[(canonical, 8), (first, 4), (second, 4)]);
        let forward = vec![
            derivation(
                first,
                4,
                vec![Operation::Copy(copy(canonical, 0, 0, 4, vec![]))],
            ),
            derivation(
                second,
                4,
                vec![Operation::Copy(copy(canonical, 4, 0, 4, vec![]))],
            ),
        ];

        let one_operation = Limits {
            max_operations: 1,
            ..Limits::default()
        };
        let result = plan(
            &catalog,
            &forward,
            &BTreeSet::from([canonical]),
            &BTreeSet::from([first, second]),
            &one_operation,
        )
        .unwrap();
        assert!(result.derivations.is_empty());
        assert_eq!(result.unresolved, BTreeSet::from([canonical]));

        let one_run = Limits {
            max_runs: 1,
            ..Limits::default()
        };
        let result = plan(
            &catalog,
            &forward,
            &BTreeSet::from([canonical]),
            &BTreeSet::from([first, second]),
            &one_run,
        )
        .unwrap();
        assert!(result.derivations.is_empty());
        assert_eq!(result.unresolved, BTreeSet::from([canonical]));
    }

    #[test]
    fn overbudget_candidate_is_skipped_not_fatal() {
        let canonical = payload(1);
        let oversized = payload(2);
        let valid = payload(3);
        let catalog = catalog(&[(canonical, 8), (oversized, 8), (valid, 8)]);
        let oversized_hint = derivation(
            oversized,
            8,
            vec![Operation::Copy(copy(
                canonical,
                0,
                0,
                4,
                vec![Axis {
                    count: 2,
                    source_stride: 4,
                    destination_stride: 4,
                }],
            ))],
        );
        let valid_hint = derivation(
            valid,
            8,
            vec![Operation::Copy(copy(canonical, 0, 0, 8, vec![]))],
        );
        let by_target = BTreeMap::from([(oversized, &oversized_hint), (valid, &valid_hint)]);
        let consumers = BTreeMap::from([(canonical, BTreeSet::from([oversized, valid]))]);
        let available = BTreeSet::from([oversized, valid]);
        let budget = Budget {
            operations: 4,
            runs: 1,
        };

        let (plan, usage) = build_plan(
            canonical,
            &catalog,
            &by_target,
            &consumers,
            &BTreeMap::new(),
            &available,
            &Limits::default(),
            &budget,
        )
        .unwrap()
        .unwrap();

        assert_eq!(plan.derivation.operations.len(), 1);
        assert_eq!(usage.runs, 1);
        match &plan.derivation.operations[0] {
            Operation::Copy(copy) => assert_eq!(copy.source, valid),
            Operation::Fill(_) => panic!("inverse planning must not emit fills"),
        }
    }

    #[test]
    fn cumulative_budget_returns_a_supported_subset() {
        let first_canonical = payload(1);
        let second_canonical = payload(2);
        let first_derived = payload(3);
        let second_derived = payload(4);
        let catalog = catalog(&[
            (first_canonical, 4),
            (second_canonical, 4),
            (first_derived, 4),
            (second_derived, 4),
        ]);
        let forward = vec![
            derivation(
                first_derived,
                4,
                vec![Operation::Copy(copy(first_canonical, 0, 0, 4, vec![]))],
            ),
            derivation(
                second_derived,
                4,
                vec![Operation::Copy(copy(second_canonical, 0, 0, 4, vec![]))],
            ),
        ];
        let limits = Limits {
            max_operations: 1,
            ..Limits::default()
        };

        let result = plan(
            &catalog,
            &forward,
            &BTreeSet::from([first_canonical, second_canonical]),
            &BTreeSet::from([first_derived, second_derived]),
            &limits,
        )
        .unwrap();
        assert_eq!(result.derivations.len(), 1);
        assert_eq!(result.unresolved, BTreeSet::from([second_canonical]));
    }

    #[test]
    fn oversized_forward_set_is_rejected_before_planning() {
        let canonical = payload(1);
        let derived = payload(2);
        let catalog = catalog(&[(canonical, 4), (derived, 4)]);
        let forward = vec![derivation(
            derived,
            4,
            vec![Operation::Copy(copy(canonical, 0, 0, 4, vec![]))],
        )];
        let limits = Limits {
            max_derivations: 0,
            ..Limits::default()
        };
        assert!(
            plan(
                &catalog,
                &forward,
                &BTreeSet::from([canonical]),
                &BTreeSet::from([derived]),
                &limits,
            )
            .is_err()
        );
    }

    #[test]
    fn oversized_inverse_copy_is_rejected_without_panic() {
        let inverse = copy(
            payload(2),
            0,
            0,
            2,
            vec![Axis {
                count: u64::MAX,
                source_stride: u64::MAX,
                destination_stride: 0,
            }],
        );
        let limits = Limits {
            max_runs: usize::MAX,
            ..Limits::default()
        };
        assert!(expand_copy(&inverse, u64::MAX, u64::MAX, &limits).is_err());
    }

    struct Scratch(std::path::PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            static SEQUENCE: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "flare-inverse-{}-{}-{tag}",
                std::process::id(),
                SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn join(&self, name: &str) -> std::path::PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn slice(scratch: &Scratch, name: &str, bytes: &[u8]) -> (PayloadId, Slice) {
        let path = scratch.join(&format!("{name}.raw"));
        fs::write(&path, bytes).unwrap();
        let file = Arc::new(fs::File::open(&path).unwrap());
        let slice = Slice::new(file, 0, bytes.len() as u64).unwrap();
        (PayloadId::of(bytes), slice)
    }

    #[test]
    fn inverse_materializes_bytes_through_the_existing_materializer() {
        let canonical: Vec<u8> = (0..16).collect();
        let derived: Vec<u8> = [0u8, 8, 4, 12]
            .into_iter()
            .flat_map(|start| canonical[start as usize..start as usize + 4].to_vec())
            .collect();
        let canonical_id = PayloadId::of(&canonical);
        let scratch = Scratch::new("materialize");
        let (derived_id, derived_slice) = slice(&scratch, "derived", &derived);
        let catalog = catalog(&[(canonical_id, 16), (derived_id, 16)]);
        let axes = vec![
            Axis {
                count: 2,
                source_stride: 8,
                destination_stride: 4,
            },
            Axis {
                count: 2,
                source_stride: 4,
                destination_stride: 8,
            },
        ];
        let forward = vec![derivation(
            derived_id,
            16,
            vec![Operation::Copy(copy(canonical_id, 0, 0, 4, axes))],
        )];

        let result = plan(
            &catalog,
            &forward,
            &BTreeSet::from([canonical_id]),
            &BTreeSet::from([derived_id]),
            &Limits::default(),
        )
        .unwrap();
        assert!(result.unresolved.is_empty());
        let output = scratch.join("canonical.bin");
        derivation::materialize(
            &output,
            &result.derivations[0],
            &BTreeMap::from([(derived_id, derived_slice)]),
            &Limits::default(),
        )
        .unwrap();
        assert_eq!(fs::read(output).unwrap(), canonical);
    }

    #[test]
    fn corrupt_source_fails_materialization_without_publication() {
        let canonical: Vec<u8> = (0..16).collect();
        let mut derived: Vec<u8> = [0u8, 8, 4, 12]
            .into_iter()
            .flat_map(|start| canonical[start as usize..start as usize + 4].to_vec())
            .collect();
        derived[5] ^= 0xff;
        let canonical_id = PayloadId::of(&canonical);
        let scratch = Scratch::new("corrupt");
        let (derived_id, derived_slice) = slice(&scratch, "derived", &derived);
        let catalog = catalog(&[(canonical_id, 16), (derived_id, 16)]);
        let axes = vec![
            Axis {
                count: 2,
                source_stride: 8,
                destination_stride: 4,
            },
            Axis {
                count: 2,
                source_stride: 4,
                destination_stride: 8,
            },
        ];
        let forward = vec![derivation(
            derived_id,
            16,
            vec![Operation::Copy(copy(canonical_id, 0, 0, 4, axes))],
        )];

        let result = plan(
            &catalog,
            &forward,
            &BTreeSet::from([canonical_id]),
            &BTreeSet::from([derived_id]),
            &Limits::default(),
        )
        .unwrap();
        assert!(result.unresolved.is_empty());
        let output = scratch.join("canonical.bin");
        assert!(
            derivation::materialize(
                &output,
                &result.derivations[0],
                &BTreeMap::from([(derived_id, derived_slice)]),
                &Limits::default(),
            )
            .is_err()
        );
        assert!(!output.exists());
    }
}
