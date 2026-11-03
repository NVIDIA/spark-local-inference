/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Completion-timed service accounting for prepared graph submissions.

use super::{utility::ServicePrice, *};
use crate::execution::ExecutionTiming;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ServiceAccounting {
    pub tokens: Option<ServicePrice>,
    pub deliveries: Option<ServicePrice>,
    pending_delivery: Option<DeliveryObservation>,
}

#[derive(Clone, Copy, Debug)]
struct DeliveryObservation {
    fence: Fence,
    before: ServicePrice,
    duration_ms: f64,
    maximum_rate: f64,
}

#[derive(Clone, Copy)]
pub(super) struct ServiceParticipant {
    handle: RequestHandle,
    token_rate: Option<f64>,
    delivery_rate: Option<f64>,
}

impl ServiceAccounting {
    pub fn publish(&mut self, fence: Fence, adaptation_ms: f64) {
        if let Some(observation) = self.pending_delivery.take() {
            debug_assert_eq!(observation.fence, fence);
            let mut price = observation.before;
            price.observe(
                observation.duration_ms,
                1.0,
                observation.maximum_rate,
                adaptation_ms,
            );
            self.deliveries = Some(price);
        }
    }
}

impl RequestScheduler {
    pub(super) fn delivery_reference_intervals(
        &self,
        rates: &[Option<f64>; REQUEST_LANES],
    ) -> [Option<f64>; REQUEST_LANES] {
        let count = rates.iter().flatten().count() as f64;
        let mut references = rates.map(|r| r.map(|r| count / r));
        let family = |r: &LiveRequest| match r.phase {
            RequestPhase::Decode if self.speculative_widths.is_empty() => Some(GraphKind::Decode),
            RequestPhase::Decode => Some(GraphKind::Speculative),
            RequestPhase::Resume => Some(GraphKind::Resume),
            RequestPhase::Snapshot if policy::terminal_cleanup(r) => Some(GraphKind::Snapshot),
            _ => None,
        };
        for kind in [
            GraphKind::Decode,
            GraphKind::Speculative,
            GraphKind::Resume,
            GraphKind::Snapshot,
        ] {
            let mask = self
                .slots
                .iter()
                .enumerate()
                .filter_map(|(slot, entry)| {
                    entry
                        .request
                        .as_ref()
                        .filter(|r| rates[slot].is_some() && family(r) == Some(kind))
                        .map(|_| 1u64 << slot)
                })
                .fold(0, |a, b| a | b);
            let streams = mask.count_ones() as usize;
            if streams == 0 {
                continue;
            }
            let mut per_delivery_ms = [f64::INFINITY; REQUEST_LANES];
            let mut consider = |nanos: u64, batch: usize, eligible: u64| {
                if nanos == 0 || eligible == 0 { return; }
                let cost = nanos as f64 / 1e6 / batch.min(eligible.count_ones() as usize) as f64;
                for slot in policy::bits(eligible) {
                    per_delivery_ms[slot] = per_delivery_ms[slot].min(cost);
                }
            };
            for batch in crate::execution::BATCH_CAPACITIES
                .into_iter()
                .take_while(|b| *b <= self.limits.max_active.next_power_of_two())
            {
                let widths: &[usize] = if matches!(kind, GraphKind::Speculative) {
                    &self.speculative_catalog[batch.ilog2() as usize]
                } else {
                    &[1]
                };
                for &width in widths {
                    if matches!(kind, GraphKind::Speculative) {
                        let mut eligible = 0;
                        let mut nanos = None;
                        for slot in policy::bits(mask) {
                            if let Some(ns) = self.speculative_duration(
                                self.slots[slot].request.as_ref().unwrap(), batch, width) {
                                eligible |= 1 << slot;
                                nanos = Some(ns);
                            }
                        }
                        if let Some(ns) = nanos { consider(ns, batch, eligible); }
                    } else if let Some(ns) = self.costs.duration(kind, batch, width) {
                        consider(ns, batch, mask);
                    }
                }
            }
            if matches!(kind, GraphKind::Decode | GraphKind::Speculative) {
                for &(batch, width, prefill) in &self.mixed_catalog {
                    if let Some(ns) = self
                        .costs
                        .shape_duration(GraphKind::Mixed, batch, width, prefill)
                        .filter(|ns| *ns > 0)
                    {
                        consider(ns, batch, mask);
                    }
                }
            }
            for (slot, reference) in references.iter_mut().enumerate() {
                if mask & (1 << slot) != 0 && per_delivery_ms[slot].is_finite() {
                    *reference = Some(count * per_delivery_ms[slot]);
                }
            }
        }
        references
    }

    /// Individually feasible service rates, in tokens and deliveries per millisecond.
    /// Unknown or expired graph times contribute no normalization evidence.
    pub(super) fn service_rates(&self, request: &LiveRequest) -> (Option<f64>, Option<f64>) {
        let mut token_rate = 0.0f64;
        let mut delivery_rate = 0.0f64;
        let prefill = request.consumed_tokens < request.prompt_tokens as u64;
        let mut consider = |kind, batch, capacity, tokens: f64, delivery: bool| {
            let ns = if kind == GraphKind::Speculative {
                self.speculative_duration(request, batch, capacity)
            } else { self.costs.duration(kind, batch, capacity) };
            if let Some(ns) = ns.filter(|n| *n > 0)
            {
                let per_ms = 1e6 / ns as f64;
                token_rate = token_rate.max(tokens * per_ms);
                if delivery {
                    delivery_rate = delivery_rate.max(per_ms);
                }
            }
        };
        if policy::terminal_cleanup(request) {
            for batch in crate::execution::BATCH_CAPACITIES
                .into_iter()
                .take_while(|b| *b <= self.limits.max_active.next_power_of_two())
            {
                consider(GraphKind::Snapshot, batch, 1, 0.0, true);
            }
        } else if prefill {
            for &capacity in &self.prefill_catalog {
                let work = self.prefill_work(request, capacity);
                if work == 0 {
                    continue;
                }
                let first_output =
                    request.consumed_tokens + work as u64 == request.prompt_tokens as u64;
                consider(
                    GraphKind::Prefill,
                    1,
                    capacity,
                    work as f64 + f64::from(first_output),
                    false,
                );
            }
        } else {
            let mut curve = None;
            for batch in crate::execution::BATCH_CAPACITIES {
                if batch > self.limits.max_active.next_power_of_two() {
                    break;
                }
                if request.generated_tokens == 0 {
                    consider(GraphKind::Resume, batch, 1, 1.0, true);
                } else if request.generated_tokens < request.max_new_tokens {
                    if self.speculative_widths.is_empty() {
                        consider(GraphKind::Decode, batch, 1, 1.0, true);
                    } else {
                        for &width in &self.speculative_catalog[batch.ilog2() as usize] {
                            let tokens = self.speculative_tokens(request, width);
                            if tokens > 0 {
                                let curve = curve.get_or_insert_with(|| self.powered_curve(request, false));
                                consider(
                                    GraphKind::Speculative,
                                    batch,
                                    width,
                                    super::rounds::forecast_tokens(curve, tokens),
                                    true,
                                );
                            }
                        }
                    }
                }
            }
        }
        if !policy::terminal_cleanup(request) {
            for &(batch, width, capacity) in &self.mixed_catalog {
                let Some(ns) = self
                    .costs
                    .shape_duration(GraphKind::Mixed, batch, width, capacity)
                    .filter(|ns| *ns > 0)
                else {
                    continue;
                };
                let per_ms = 1e6 / ns as f64;
                if prefill {
                    let work = self.prefill_work(request, capacity);
                    if work > 0 {
                        let first =
                            request.consumed_tokens + work as u64 == request.prompt_tokens as u64;
                        token_rate = token_rate.max((work as f64 + f64::from(first)) * per_ms);
                    }
                } else if request.generated_tokens > 0
                    && request.generated_tokens < request.max_new_tokens
                {
                    let tokens = self.speculative_tokens(request, width);
                    if tokens > 0 {
                        token_rate = token_rate.max(
                            request
                                .acceptance
                                .expected_tokens(self.proposal_source, tokens)
                                * per_ms,
                        );
                        delivery_rate = delivery_rate.max(per_ms);
                    }
                }
            }
        }
        (
            (token_rate > 0.0).then_some(token_rate),
            (delivery_rate > 0.0).then_some(delivery_rate),
        )
    }

    pub(super) fn projected_service(&self, request: &LiveRequest) -> ServiceAccounting {
        let mut service = request.accounting.service;
        for submission in self.submissions.values() {
            let Some(participant) = submission.participants[request.handle.slot as usize]
                .filter(|p| p.handle == request.handle)
            else {
                continue;
            };
            let dt = submission.estimated_duration_ns as f64 / 1e6;
            if dt <= 0.0 {
                continue;
            }
            let lane = submission
                .lanes
                .iter()
                .find(|l| l.request == request.handle);
            let (tokens, delivery) = match (submission.kind, lane) {
                (GraphKind::Prefill | GraphKind::ScoredPrefill, Some(l)) => (l.tokens as f64, 0.0),
                (GraphKind::Decode | GraphKind::Speculative, Some(_)) =>
                    (submission.expected_tokens[request.handle.slot as usize], 1.0),
                _ => (0.0, 0.0),
            };
            if let (Some(price), Some(rate)) = (&mut service.tokens, participant.token_rate) {
                price.observe(dt, tokens, rate, self.policy.adaptation_ms);
            }
            if let (Some(price), Some(rate)) = (&mut service.deliveries, participant.delivery_rate)
            {
                price.observe(dt, delivery, rate, self.policy.adaptation_ms);
            }
        }
        service
    }

    pub(super) fn service_participants(&mut self) -> [Option<ServiceParticipant>; REQUEST_LANES] {
        let mut participants = [None; REQUEST_LANES];
        let mut count = 0;
        for (slot, entry) in self.slots.iter().enumerate() {
            let Some(request) = entry
                .request
                .as_ref()
                .filter(|r| !matches!(r.phase, RequestPhase::Inbound | RequestPhase::Finishing))
            else {
                continue;
            };
            count += 1;
            let (token_rate, delivery_rate) = self.service_rates(request);
            participants[slot] = Some(ServiceParticipant {
                handle: request.handle,
                token_rate,
                delivery_rate,
            });
        }
        let references = self
            .delivery_reference_intervals(&participants.map(|p| p.and_then(|p| p.delivery_rate)));
        for (slot, participant) in participants
            .iter()
            .enumerate()
            .filter_map(|(s, p)| p.map(|p| (s, p)))
        {
            let service = &mut self.slots[participant.handle.slot as usize]
                .request
                .as_mut()
                .unwrap()
                .accounting
                .service;
            if let Some(rate) = participant.token_rate {
                service
                    .tokens
                    .get_or_insert_with(|| ServicePrice::initial(rate / count as f64));
            }
            if let Some(reference) = references[slot] {
                service
                    .deliveries
                    .get_or_insert_with(|| ServicePrice::initial(1.0 / reference));
            }
        }
        participants
    }

    pub(super) fn observe_service(
        &mut self,
        fence: Fence,
        participants: &[Option<ServiceParticipant>; REQUEST_LANES],
        useful_tokens: &[u64; REQUEST_LANES],
        timing: ExecutionTiming,
    ) {
        for request in self.slots.iter_mut().filter_map(|s| s.request.as_mut()) {
            request.accounting.service.pending_delivery = None;
        }
        let Some(ns) = cost::observed_duration_ns(timing) else {
            return;
        };
        let duration_ms = ns as f64 / 1e6;
        let adaptation_ms = self.policy.adaptation_ms;
        for participant in participants.iter().flatten() {
            let Some(request) = self.slots[participant.handle.slot as usize]
                .request
                .as_mut()
                .filter(|r| r.handle == participant.handle)
            else {
                continue;
            };
            let output = request.accounting.pending_publication_fence == Some(fence)
                && useful_tokens[participant.handle.slot as usize] > 0;
            let service = &mut request.accounting.service;
            if let (Some(price), Some(rate)) = (&mut service.tokens, participant.token_rate) {
                price.observe(
                    duration_ms,
                    useful_tokens[participant.handle.slot as usize] as f64,
                    rate,
                    adaptation_ms,
                );
            }
            if let (Some(price), Some(rate)) = (&mut service.deliveries, participant.delivery_rate)
            {
                if output {
                    service.pending_delivery = Some(DeliveryObservation {
                        fence,
                        before: *price,
                        duration_ms,
                        maximum_rate: rate,
                    });
                }
                price.observe(duration_ms, 0.0, rate, adaptation_ms);
            } else if output && service.deliveries.is_none() {
                let rate = 1.0 / duration_ms;
                service.pending_delivery = Some(DeliveryObservation {
                    fence,
                    before: ServicePrice::initial(rate),
                    duration_ms,
                    maximum_rate: rate,
                });
            }
        }
    }
}
