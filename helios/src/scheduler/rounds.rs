/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Completed speculative forecasts and unified captured generation rounds.

use super::{policy::{Candidate, bits}, utility::{LaneValue, ValueRate, best_cohort}, *};

const POPULATION_WEIGHT: f32 = 1.0 / 64.0;

#[derive(Clone, Copy)]
struct Forecast {
    request: RequestHandle,
    prefixes: [speculation::PrefixEstimate; 7],
}

impl Forecast {
    fn new(request: RequestHandle, prior: [f32; 7], generated: u32) -> Self {
        Self { request, prefixes: prior.map(|p| speculation::PrefixEstimate::new(f64::from(p), generated)) }
    }
}

pub(super) struct Forecasts {
    requests: [Option<Forecast>; REQUEST_LANES],
    pub(super) population: [f32; 7],
}

impl Default for Forecasts {
    fn default() -> Self {
        Self { requests: [None; REQUEST_LANES], population: std::array::from_fn(|i| 0.5f32.powi(i as i32 + 1)) }
    }
}

pub(super) struct HostPrefixRanking {
    probabilities: [[f64; 7]; REQUEST_LANES],
    ranks_by_slot: [Vec<usize>; REQUEST_LANES],
    slots_by_rank: Vec<usize>,
}

impl HostPrefixRanking {
    pub(super) fn extend_cohort(&self, slot: usize, lengths: &mut [usize; REQUEST_LANES],
        frontier: &mut std::collections::BinaryHeap<usize>, draft_budget: usize) {
        lengths[slot] = 1;
        for &rank in &self.ranks_by_slot[slot] {
            frontier.push(rank);
            lengths[slot] += 1;
        }
        while frontier.len() > draft_budget {
            lengths[self.slots_by_rank[frontier.pop().unwrap()]] -= 1;
        }
    }
}

fn rank_prefixes(positions: &mut [(f64, usize, usize)]) {
    // Survival is nonincreasing with depth; ties keep predecessors before successors.
    positions.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
}

impl RequestScheduler {
    pub(super) fn pending_ceiling(&self, request: &LiveRequest) -> usize {
        // Completion reconciles the scheduled cursor with the remaining submissions.
        let pending = (request.scheduled_consumed - request.consumed_tokens) as usize;
        debug_assert_eq!(pending, self.pending_token_ceiling(request.handle));
        pending
    }

    pub(super) fn pending_token_ceiling(&self, handle: RequestHandle) -> usize {
        self.submissions.values().map(|submission| submission.lanes.iter()
            .filter(|lane| lane.request == handle)
            .map(|lane| submission.lane_advance_ceiling(lane)).sum::<usize>()).sum()
    }

    pub(super) fn speculative_curve(&self, request: &LiveRequest, optimistic: bool) -> [f64; 7] {
        if self.proposal_source == speculation::ProposalSource::Mtp
            && self.capabilities.speculative_allocations.iter().any(|shape| shape.4 == crate::execution::PrefixAllocation::Gpu) {
            let forecast = self.forecasts.requests[request.handle.slot as usize]
                .filter(|forecast| forecast.request == request.handle)
                .unwrap_or_else(|| Forecast::new(request.handle, self.forecasts.population, request.generated_tokens));
            let mut bound = 1.0f64;
            forecast.prefixes.map(|estimate| {
                bound = bound.min(if optimistic { estimate.upper(request.generated_tokens) } else { estimate.mean() });
                bound
            })
        } else {
            request.acceptance.curve(self.proposal_source, request.generated_tokens, optimistic)
        }
    }

    pub(super) fn gpu_prefix_allocation(&self, batch: usize, drafts: usize, width: usize, packed: usize) -> bool {
        self.capabilities.speculative_allocations.iter().any(|&(b, d, w, p, allocation)|
            (b, d, w, p) == (batch, drafts, width, packed) && allocation == crate::execution::PrefixAllocation::Gpu)
    }

    fn round_ceilings(&self, mask: u64, drafts: usize, width: usize) -> [usize; REQUEST_LANES] {
        std::array::from_fn(|slot| {
            if mask & (1 << slot) == 0 { 0 } else {
                self.speculative_tokens(self.slots[slot].request.as_ref().unwrap(), width.min(drafts + 1))
            }
        })
    }

    pub(super) fn round_lengths(&self, mask: u64, batch: usize, drafts: usize, width: usize, packed: usize) -> [usize; REQUEST_LANES] {
        self.round_lengths_optimistic(mask, batch, drafts, width, packed, false)
    }

    pub(super) fn round_lengths_optimistic(&self, mask: u64, batch: usize, drafts: usize, width: usize, packed: usize, optimistic: bool) -> [usize; REQUEST_LANES] {
        let ceilings = self.round_ceilings(mask, drafts, width);
        if self.gpu_prefix_allocation(batch, drafts, width, packed)
            || self.speculative_fixed_width.is_some() || ceilings.iter().sum::<usize>() <= packed {
            return ceilings;
        }
        let (values, _, _) = self.lane_values(self.window_remaining().unwrap_or(0));
        self.prefix_lengths_with_values(mask, batch, drafts, width, packed, true, &values, optimistic)
    }

    pub(super) fn estimated_round_lengths(&self, mask: u64, batch: usize, drafts: usize, width: usize, packed: usize, optimistic: bool) -> [usize; REQUEST_LANES] {
        let (values, _, _) = self.lane_values(self.window_remaining().unwrap_or(0));
        self.prefix_lengths_with_values(mask, batch, drafts, width, packed, false, &values, optimistic)
    }

    pub(super) fn prefix_lengths_with_values(&self, mask: u64, batch: usize, drafts: usize, width: usize, packed: usize,
        preserve_gpu_ceilings: bool, values: &[LaneValue; REQUEST_LANES], optimistic: bool) -> [usize; REQUEST_LANES] {
        let mut lengths = [0; REQUEST_LANES];
        let ceiling = self.round_ceilings(mask, drafts, width);
        let gpu = self.gpu_prefix_allocation(batch, drafts, width, packed);
        if (gpu && preserve_gpu_ceilings) || (!gpu && self.speculative_fixed_width.is_some())
            || ceiling.iter().sum::<usize>() <= packed { return ceiling; }
        for slot in bits(mask) { lengths[slot] = 1; }
        let remaining = packed.saturating_sub(mask.count_ones() as usize);
        if remaining == 0 { return lengths; }
        let probabilities: [[f64; 7]; REQUEST_LANES] = std::array::from_fn(|slot| {
            if mask & (1 << slot) == 0 { return [0.0; 7]; }
            let request = self.slots[slot].request.as_ref().unwrap();
            if gpu { self.speculative_curve(request, optimistic) }
            else { request.acceptance.curve(self.proposal_source, request.generated_tokens, optimistic)
                .map(|probability| probability.powf(self.policy.adaptive_verification_alpha) * values[slot].output_token) }
        });
        let mut positions = Vec::new();
        for slot in bits(mask) {
            for depth in 0..ceiling[slot].saturating_sub(1) {
                positions.push((probabilities[slot][depth], depth, slot));
            }
        }
        rank_prefixes(&mut positions);
        for (_, _, slot) in positions.into_iter().take(remaining) {
            lengths[slot] += 1;
        }
        lengths
    }

    pub(super) fn host_prefix_ranking(&self, available: &[usize], drafts: usize, width: usize,
        values: &[LaneValue; REQUEST_LANES], optimistic: bool) -> HostPrefixRanking {
        let mut probabilities = [[0.0; 7]; REQUEST_LANES];
        let mut scores = Vec::new();
        for &slot in available {
            let request = self.slots[slot].request.as_ref().unwrap();
            let limit = self.speculative_tokens(request, width.min(drafts + 1));
            probabilities[slot] = request.acceptance.curve(self.proposal_source, request.generated_tokens, optimistic)
                .map(|probability| probability.powf(self.policy.adaptive_verification_alpha));
            for position in 0..limit.saturating_sub(1) {
                scores.push((probabilities[slot][position] * values[slot].output_token, position, slot));
            }
        }
        rank_prefixes(&mut scores);
        let mut ranks_by_slot = std::array::from_fn(|_| Vec::new());
        let mut slots_by_rank = Vec::with_capacity(scores.len());
        for (rank, &(_, _, slot)) in scores.iter().enumerate() {
            ranks_by_slot[slot].push(rank);
            slots_by_rank.push(slot);
        }
        HostPrefixRanking { probabilities, ranks_by_slot, slots_by_rank }
    }

    pub(super) fn speculative_information(&self, request: &LiveRequest, depth: usize) -> f64 {
        self.forecasts.requests[request.handle.slot as usize].filter(|forecast| forecast.request == request.handle)
            .map_or_else(|| request.acceptance.information(self.proposal_source, depth, request.generated_tokens),
                |forecast| forecast.prefixes[depth].weight_at(request.generated_tokens))
    }

    pub(super) fn expected_speculative_tokens(&self, request: &LiveRequest, width: usize) -> f64 {
        self.proposal_forecast_tokens(request, width, false)
    }

    fn proposal_forecast_tokens(&self, request: &LiveRequest, width: usize, optimistic: bool) -> f64 {
        forecast_tokens(&self.powered_curve(request, optimistic), width)
    }

    /// Prefix survival raised to the adaptive verification exponent.
    pub(super) fn powered_curve(&self, request: &LiveRequest, optimistic: bool) -> [f64; 7] {
        self.speculative_curve(request, optimistic).map(|p| p.powf(self.policy.adaptive_verification_alpha))
    }

    pub(super) fn speculative_duration(&self, request: &LiveRequest, batch: usize, width: usize) -> Option<u64> {
        if self.speculative_fixed_width.is_none() {
            // The full B x W graph is among the rounds this table minimizes over.
            return self.costs.packed_round_duration(batch, width);
        }
        let length = self.speculative_tokens(request, width);
        self.costs.packed_round_shapes().filter(|shape| shape.batch == batch && shape.width == width
            && shape.drafts + 1 >= length && shape.tokens >= length)
            .map(|shape| shape.nanos).min()
    }

    pub(super) fn observe_packed_forecasts(&mut self, step: &ScheduledStep,
        records: &[CompletionRecord], confidence: &[[f32; 7]]) {
        for (row, lane) in step.lanes.iter().enumerate() {
            let record = &records[lane.request.slot as usize];
            let Some(vector) = record.vector else { continue; };
            if !matches!(record.status, 1 | 2) || vector.emitted_count == 0
                || self.request(lane.request).is_err() { continue; }
            let generated = (record.generated_tokens - vector.emitted_count) as u32;
            let mut forecast = self.forecasts.requests[lane.request.slot as usize]
                .filter(|forecast| forecast.request == lane.request)
                .unwrap_or_else(|| Forecast::new(lane.request, self.forecasts.population, generated));
            let mut prefix = 1.0f32;
            for depth in 0..lane.tokens.saturating_sub(1).min(step.draft_tokens) {
                prefix *= confidence[row][depth];
                forecast.prefixes[depth].observe(f64::from(prefix), generated);
                self.forecasts.population[depth] += POPULATION_WEIGHT * (prefix - self.forecasts.population[depth]);
            }
            self.forecasts.requests[lane.request.slot as usize] = Some(forecast);
        }
    }
}

impl RequestScheduler {
    fn fixed_round_candidate(&self, shape: &cost::PackedRoundShape, prefixes: &[(usize, usize, f64)],
        values: &[LaneValue; REQUEST_LANES], age: f64) -> Option<Candidate> {
        let offers: Vec<_> = prefixes.iter().filter_map(|&(slot, length, expected)| {
            (length != 0 && length <= shape.tokens).then(||
                (slot, length, values[slot].decode(expected, shape.nanos as f64 / 1e6)))
        }).collect();
        let &(_, first_length, _) = offers.first()?;
        let scores: Vec<_> = offers.iter().map(|&(slot, _, value)| (slot, value)).collect();
        let unconstrained = best_cohort::<REQUEST_LANES>(&scores, shape.batch);
        let full_rows: usize = offers.iter().filter(|&&(slot, _, _)| unconstrained & (1 << slot) != 0)
            .map(|offer| offer.1).sum();
        let mask = if full_rows <= shape.tokens {
            unconstrained
        } else if offers.iter().all(|&(_, length, _)| length == first_length) {
            best_cohort::<REQUEST_LANES>(&scores, shape.batch.min(shape.tokens / first_length))
        } else {
            let budget = shape.tokens.min(offers.iter().map(|offer| offer.1).sum());
            let mut lengths: Vec<_> = offers.iter().map(|offer| offer.1).collect();
            lengths.sort_unstable();
            let mut minimum_rows = vec![0];
            for length in lengths {
                minimum_rows.push(minimum_rows.last().unwrap() + length);
            }
            let maximum_lanes = minimum_rows.iter().take_while(|&&rows| rows <= budget).count() - 1;
            let count = shape.batch.min(maximum_lanes);
            // A row budget that already bounds cohort size needs no cardinality dimension.
            let cardinality = maximum_lanes > shape.batch;
            let stride = budget + 1;
            let layers = if cardinality { count + 1 } else { 1 };
            let mut states = vec![None; layers * stride];
            states[0] = Some((0.0, 0u64));
            let maximum_length = offers.iter().map(|offer| offer.1).max().unwrap();
            for (index, &(slot, length, value)) in offers.iter().enumerate() {
                let first = usize::from(cardinality);
                let last = if cardinality { count.min(index + 1) } else { 0 };
                for lanes in (first..=last).rev() {
                    let previous = if cardinality { lanes - 1 } else { 0 };
                    let lower = length + if cardinality { minimum_rows[previous] } else { 0 };
                    let upper = budget.min(length + if cardinality { previous * maximum_length } else { budget });
                    for rows in (lower..=upper).rev() {
                        let Some((before, mask)) = states[previous * stride + rows - length] else { continue; };
                        let candidate = (before + value, mask | (1 << slot));
                        let destination = &mut states[lanes * stride + rows];
                        if destination.is_none_or(|old: (f64, u64)| candidate.0 > old.0
                            || candidate.0 == old.0 && (candidate.1.count_ones() > old.1.count_ones()
                                || candidate.1.count_ones() == old.1.count_ones() && candidate.1 < old.1)) {
                            *destination = Some(candidate);
                        }
                    }
                }
            }
            states.into_iter().flatten().max_by(|a, b| a.0.total_cmp(&b.0)
                .then(a.1.count_ones().cmp(&b.1.count_ones())).then(b.1.cmp(&a.1)))?.1
        };
        if mask == 0 { return None; }
        let value = offers.iter().filter(|&&(slot, _, _)| mask & (1 << slot) != 0)
            .map(|offer| offer.2).sum();
        let mut host_lengths = [0; REQUEST_LANES];
        for &(slot, length, _) in &offers {
            if mask & (1 << slot) != 0 { host_lengths[slot] = length as u8; }
        }
        Some(Candidate { host_lengths: Some(host_lengths), draft_tokens: shape.drafts, kind: GraphKind::Speculative,
            mask, batch: shape.batch, capacity: shape.width, packed_token_capacity: shape.tokens,
            prefill_capacity: 0, prefill_slot: None, work: 0, duration_ns: shape.nanos,
            rate: ValueRate::new(shape.nanos, value, age), feedback: false })
    }

    pub(super) fn packed_round_candidates(&self, output: &mut Vec<Candidate>,
        values: &[LaneValue; REQUEST_LANES], age: f64, optimistic: bool) {
        let available: Vec<_> = self.slots.iter().enumerate().filter_map(|(slot, entry)| {
            let request = entry.request.as_ref()?;
            (request.phase == RequestPhase::Decode && self.eligible(request)).then_some(slot)
        }).collect();
        let mut forecasts = HashMap::new();
        let mut fixed_prefixes = HashMap::new();
        let mut host_rankings = HashMap::new();
        // Lane forecasts stay fixed while shapes compete; evaluate each at most once.
        let mut curves: [Option<[f64; 7]>; REQUEST_LANES] = [None; REQUEST_LANES];
        let mut order = Vec::with_capacity(available.len());
        let mut frontier = std::collections::BinaryHeap::new();
        for shape in self.costs.packed_round_shapes() {
            if shape.batch > self.limits.max_active.next_power_of_two()
                || !self.speculative_catalog[shape.batch.ilog2() as usize].contains(&shape.width) { continue; }
            if self.speculative_fixed_width.is_some()
                && !self.gpu_prefix_allocation(shape.batch, shape.drafts, shape.width, shape.tokens) {
                let prefixes = fixed_prefixes.entry((shape.drafts, shape.width)).or_insert_with(|| {
                    available.iter().map(|&slot| {
                        let request = self.slots[slot].request.as_ref().unwrap();
                        let length = self.speculative_tokens(request, shape.width.min(shape.drafts + 1));
                        let expected = f64::from(length != 0) + request.acceptance.curve(
                            self.proposal_source, request.generated_tokens, false)[..length.saturating_sub(1)]
                            .iter().map(|&probability| {
                                if self.policy.adaptive_verification_alpha == 1.0 { probability }
                                else { probability.powf(self.policy.adaptive_verification_alpha) }
                            }).sum::<f64>();
                        (slot, length, expected)
                    }).collect::<Vec<_>>()
                });
                if let Some(candidate) = self.fixed_round_candidate(&shape, prefixes, values, age) {
                    output.push(candidate);
                }
                continue;
            }
            // Speculative positions ranked by marginal forecast tokens; shapes with equal
            // draft and width bounds share the ranking.
            let forecast = forecasts.entry((shape.drafts, shape.width)).or_insert_with(|| {
                let mut scores = Vec::new();
                for &slot in &available {
                    let request = self.slots[slot].request.as_ref().unwrap();
                    let limit = self.speculative_tokens(request, (shape.drafts + 1).min(shape.width));
                    let curve = curves[slot].get_or_insert_with(|| self.powered_curve(request, optimistic));
                    let mut before = 1.0;
                    for position in 0..limit.saturating_sub(1) {
                        let total = forecast_tokens(curve, position + 2);
                        scores.push((total - before, position, slot));
                        before = total;
                    }
                }
                scores.sort_unstable_by(|a, b|
                    b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
                let gains: Vec<_> = scores.iter().map(|&(score, _, slot)|
                    score * values[slot].output_token).collect();
                let mut ranks: [Vec<usize>; REQUEST_LANES] = std::array::from_fn(|_| Vec::new());
                for (rank, &(_, _, slot)) in scores.iter().enumerate() { ranks[slot].push(rank); }
                (gains, ranks)
            });
            let (gains, ranks) = (&forecast.0, &forecast.1);
            let host_ranking = if self.gpu_prefix_allocation(shape.batch, shape.drafts, shape.width, shape.tokens) {
                None
            } else {
                Some(host_rankings.entry((shape.drafts, shape.width)).or_insert_with(||
                    self.host_prefix_ranking(&available, shape.drafts, shape.width, values, optimistic)))
            };
            let nanos = shape.nanos;
            let mut base = [0.0; REQUEST_LANES];
            let mut full = [0.0; REQUEST_LANES];
            let mut limits = [0; REQUEST_LANES];
            for &slot in &available {
                let request = self.slots[slot].request.as_ref().unwrap();
                base[slot] = values[slot].decode(1.0, nanos as f64 / 1e6);
                full[slot] = base[slot] + ranks[slot].iter().take(shape.tokens - 1)
                    .map(|&rank| gains[rank]).sum::<f64>();
                limits[slot] = self.speculative_tokens(request, (shape.drafts + 1).min(shape.width));
            }
            let mut best: Option<(u64, f64, Option<[u8; REQUEST_LANES]>)> = None;
            // Alternate bounded orderings expose long useful prefixes, delivery
            // fairness, and short tails. Every cohort is priced by its causal
            // prefix allocation and the full measured physical B/D/W/P round.
            for ordering in 0..3 {
                order.clear();
                order.extend(available.iter().map(|&slot| (slot, match ordering {
                    0 => full[slot], 1 => base[slot], _ => full[slot] / limits[slot] as f64 })));
                // Only the leading cohort-size entries of the total order are used.
                let rank = |a: &(usize, f64), b: &(usize, f64)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
                let cohort = shape.batch.min(shape.tokens);
                if (1..order.len()).contains(&cohort) {
                    order.select_nth_unstable_by(cohort - 1, rank);
                    order.truncate(cohort);
                }
                order.sort_unstable_by(rank);
                frontier.clear();
                let mut host_lengths = [0; REQUEST_LANES];
                // A lane's value changes only when its host prefix length does.
                let mut lane_terms = [(0, 0.0); REQUEST_LANES];
                let (mut mask, mut pending_value, mut draft_value) = (0u64, 0., 0.);
                for (count, &(slot, _)) in order.iter().take(shape.batch.min(shape.tokens)).enumerate() {
                    mask |= 1 << slot;
                    pending_value += base[slot];
                    let value = if let Some(ranking) = &host_ranking {
                        ranking.extend_cohort(slot, &mut host_lengths, &mut frontier, shape.tokens - count - 1);
                        bits(mask).map(|slot| {
                            if lane_terms[slot].0 != host_lengths[slot] {
                                let expected = 1.0 + ranking.probabilities[slot][..host_lengths[slot] - 1]
                                    .iter().sum::<f64>();
                                lane_terms[slot] = (host_lengths[slot], values[slot].decode(expected, nanos as f64 / 1e6));
                            }
                            lane_terms[slot].1
                        }).sum()
                    } else {
                        // The heap retains the best P-count speculative positions.
                        // Its capacity decreases as pending rows enter; discarded
                        // positions cannot re-enter at a subsequent cohort size.
                        for &rank in &ranks[slot] {
                            frontier.push(rank);
                            draft_value += gains[rank];
                        }
                        while frontier.len() > shape.tokens - count - 1 {
                            draft_value -= gains[frontier.pop().unwrap()];
                        }
                        pending_value + draft_value
                    };
                    if best.is_none_or(|(old_mask, old_value, _)|
                        value > old_value || (value == old_value &&
                            (mask.count_ones(), std::cmp::Reverse(mask)) >
                            (old_mask.count_ones(), std::cmp::Reverse(old_mask)))) {
                        best = Some((mask, value, host_ranking.as_ref().map(|_| host_lengths.map(|length| length as u8))));
                    }
                }
            }
            if let Some((mask, value, host_lengths)) = best {
                output.push(Candidate { host_lengths, draft_tokens: shape.drafts, kind: GraphKind::Speculative,
                    mask, batch: shape.batch, capacity: shape.width, packed_token_capacity: shape.tokens,
                    prefill_capacity: 0, prefill_slot: None, work: 0, duration_ns: nanos,
                    rate: ValueRate::new(nanos, value, age), feedback: false,
                    });
            }
        }
    }
}

/// Expected tokens from a round of `width` positions with powered prefix survival.
pub(super) fn forecast_tokens(powered: &[f64; 7], width: usize) -> f64 {
    f64::from(width != 0) + powered[..width.saturating_sub(1)].iter().sum::<f64>()
}
