/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Exact-prefix capture selection and request-local predicted work accounting.

use super::*;

pub(super) type SelectionCache = HashMap<RequestHandle, (CheckpointAccounting, bool, bool)>;

pub(super) struct PlanningScope<'a>(&'a std::cell::RefCell<Option<SelectionCache>>);

impl Drop for PlanningScope<'_> {
    fn drop(&mut self) {
        *self.0.borrow_mut() = None;
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CheckpointAccounting {
    pub demand_epoch: u64,
    pub initial_prefill_ns: Option<u64>,
    pub productive_decode_ns: u64,
    pub extra_spent_ns: u64,
    baseline_interval: Option<(usize, usize)>,
    window_ends: Vec<usize>,
    reserved_ns: u64,
    cuts: Vec<Cut>,
}

impl CheckpointAccounting {
    pub(super) fn for_epoch(demand_epoch: u64, consumed: usize, prompt: usize) -> Self {
        Self {
            demand_epoch,
            baseline_interval: Some((consumed, prompt)),
            ..Self::default()
        }
    }
    pub(super) fn retain_window_ends(&mut self, ends: impl Iterator<Item = usize>) {
        self.window_ends.extend(ends);
        self.window_ends.sort_unstable();
        self.window_ends.dedup();
    }

    pub(super) fn clear_reservations(&mut self) {
        self.cuts.clear();
        self.reserved_ns = 0;
    }

    pub(super) fn refund_copy(&mut self, cost: u64) {
        self.reserved_ns = self.reserved_ns.saturating_sub(cost);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Cut {
    endpoint: usize,
    role: RetentionRole,
    split_ns: u64,
    copy_ns: u64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ArmedCapture {
    pub snapshot: SnapshotId,
    pub endpoint: u64,
    pub role: RetentionRole,
    pub copy_ns: u64,
}

#[derive(Clone, Debug)]
pub(super) struct CapturePlan {
    handle: RequestHandle,
    accounting: CheckpointAccounting,
    anchor: usize,
    cut: Option<Cut>,
    quote: Option<cache_memory::SnapshotQuote>,
    reserve_guard: bool,
    unknown: bool,
    budget: bool,
}

fn allowance(work: u64, fraction: f64) -> u64 {
    ((work as f64) * fraction).floor().min(u64::MAX as f64) as u64
}

impl LiveRequest {
    pub(super) fn checkpoint_anchor(&self) -> usize {
        self.message_boundaries
            .iter()
            .rev()
            .copied()
            .find(|&end| self.checkpoint_endpoint_eligible(end))
            .unwrap_or(self.prompt_tokens)
    }

    fn checkpoint_endpoint_eligible(&self, end: usize) -> bool {
        !self
            .prefill_spans
            .iter()
            .any(|span| span.start < end && end < span.end)
            && (end == self.prompt_tokens
                || self.checkpoints.window_ends.binary_search(&end).is_err())
    }
}

impl RequestScheduler {
    pub(super) fn checkpoint_planning_scope(&self) -> PlanningScope<'_> {
        *self.checkpoint_selections.borrow_mut() = Some(HashMap::new());
        PlanningScope(&self.checkpoint_selections)
    }
    pub(super) fn revoke_unused_anchor_guards(&mut self) -> bool {
        let handle = self
            .slots
            .iter()
            .filter_map(|slot| slot.request.as_ref())
            .filter(|request| request.anchor_guard.is_some())
            .min_by_key(|request| {
                (
                    request.checkpoints.demand_epoch,
                    request.handle.slot,
                    request.handle.generation,
                )
            })
            .map(|request| request.handle);
        let Some(handle) = handle else {
            return false;
        };
        let guard = self
            .request_mut(handle)
            .unwrap()
            .anchor_guard
            .take()
            .unwrap();
        self.release_snapshot(guard);
        true
    }

    fn checkpoint_headroom(&self) -> usize {
        self.slots
            .iter()
            .filter_map(|slot| slot.request.as_ref())
            .filter(|request| request.growable_output && request.phase == RequestPhase::Decode)
            .map(|request| {
                self.pages_through(
                    request,
                    request
                        .scheduled_consumed
                        .saturating_add(self.decode_horizon()),
                )
                .saturating_sub(request.reserved_pages.len())
            })
            .fold(0usize, usize::saturating_add)
    }

    fn interval_prefill_ns(
        &self,
        request: &LiveRequest,
        mut start: usize,
        end: usize,
    ) -> Option<u64> {
        let catalog = if request.prompt_logprobs {
            &self.scored_prefill_catalog
        } else {
            &self.prefill_catalog
        };
        let mut total = 0u64;
        while start < end {
            let limit = request
                .checkpoints
                .window_ends
                .iter()
                .copied()
                .filter(|&limit| start < limit)
                .min()
                .unwrap_or(end)
                .min(end);
            let mut best: Option<(usize, u64)> = None;
            for &capacity in catalog {
                let mut through = start.saturating_add(capacity).min(limit);
                if let Some(span) = request
                    .prefill_spans
                    .iter()
                    .find(|span| span.start < through && through < span.end)
                {
                    through = span.start;
                }
                let work = through.saturating_sub(start);
                if work == 0 {
                    continue;
                }
                let nanos = self
                    .costs
                    .fresh_duration(request.prefill_kind(), capacity)?;
                if best.is_none_or(|(old_work, old_ns)| {
                    u128::from(nanos) * (old_work as u128) < u128::from(old_ns) * work as u128
                        || (u128::from(nanos) * old_work as u128
                            == u128::from(old_ns) * work as u128
                            && work > old_work)
                }) {
                    best = Some((work, nanos));
                }
            }
            let (work, nanos) = best?;
            start += work;
            total = total.checked_add(nanos)?;
        }
        Some(total)
    }

    fn checkpoint_selection(&self, request: &LiveRequest) -> (CheckpointAccounting, bool, bool) {
        if let Some(value) = self
            .checkpoint_selections
            .borrow()
            .as_ref()
            .and_then(|cache| cache.get(&request.handle))
        {
            return value.clone();
        }
        let selection = self.checkpoint_selection_uncached(request);
        if let Some(cache) = self.checkpoint_selections.borrow_mut().as_mut() {
            cache.insert(request.handle, selection.clone());
        }
        selection
    }

    fn checkpoint_selection_uncached(
        &self,
        request: &LiveRequest,
    ) -> (CheckpointAccounting, bool, bool) {
        let mut accounting = request.checkpoints.clone();
        if accounting
            .cuts
            .iter()
            .any(|cut| cut.role != RetentionRole::Anchor)
        {
            return (accounting, false, false);
        }
        accounting.cuts.clear();
        accounting.reserved_ns = 0;
        let anchor = request.checkpoint_anchor();
        let start = request.consumed_tokens as usize;
        let (baseline_start, baseline_end) = *accounting.baseline_interval.get_or_insert((
            request.admission_reused_tokens as usize,
            request.prompt_tokens,
        ));
        if accounting.initial_prefill_ns.is_none() {
            accounting.initial_prefill_ns =
                self.interval_prefill_ns(request, baseline_start, baseline_end);
        }
        if anchor > start {
            accounting.cuts.push(Cut {
                endpoint: anchor,
                role: RetentionRole::Anchor,
                split_ns: 0,
                copy_ns: 0,
            });
        }
        let Some(prefill) = accounting.initial_prefill_ns else {
            return (accounting, true, false);
        };
        let Some(copy_ns) = self.costs.fresh_duration(GraphKind::Snapshot, 1) else {
            return (accounting, true, false);
        };
        let mut unknown = false;
        let mut budget = false;
        let available = allowance(prefill, self.policy.checkpoint_extra_work_fraction);
        let mut candidates: Vec<_> = request
            .message_boundaries
            .iter()
            .rev()
            .copied()
            .filter(|&end| start < end && end < anchor && request.checkpoint_endpoint_eligible(end))
            .map(|end| (end, RetentionRole::Message))
            .collect();
        if start < request.prompt_tokens && request.prompt_tokens != anchor {
            candidates.push((request.prompt_tokens, RetentionRole::Endpoint));
        }
        for (endpoint, role) in candidates {
            let left = accounting
                .cuts
                .iter()
                .map(|cut| cut.endpoint)
                .filter(|&cut| cut < endpoint)
                .max()
                .unwrap_or(start);
            let right = accounting
                .cuts
                .iter()
                .map(|cut| cut.endpoint)
                .filter(|&cut| endpoint < cut)
                .min()
                .unwrap_or(request.prompt_tokens);
            let split = if endpoint == right {
                Some(0)
            } else {
                self.interval_prefill_ns(request, left, right)
                    .zip(
                        self.interval_prefill_ns(request, left, endpoint)
                            .zip(self.interval_prefill_ns(request, endpoint, right)),
                    )
                    .map(|(whole, (before, after))| {
                        before.saturating_add(after).saturating_sub(whole)
                    })
            };
            let Some(split_ns) = split else {
                unknown = true;
                continue;
            };
            let charge = split_ns.saturating_add(copy_ns);
            if accounting
                .extra_spent_ns
                .saturating_add(accounting.reserved_ns)
                .saturating_add(charge)
                > available
            {
                budget = true;
                continue;
            }
            let rank = RetentionRank {
                use_epoch: accounting.demand_epoch,
                role,
                consumed_tokens: endpoint as u64,
            };
            let rows = 1 + accounting
                .cuts
                .iter()
                .filter(|cut| cut.endpoint > start)
                .count()
                - usize::from(request.anchor_guard.is_some() && start < anchor);
            if self
                .quote_snapshots_preserving(1, rows, self.checkpoint_headroom(), rank)
                .is_none()
            {
                continue;
            }
            accounting.reserved_ns = accounting.reserved_ns.saturating_add(charge);
            accounting.cuts.push(Cut {
                endpoint,
                role,
                split_ns,
                copy_ns,
            });
        }
        accounting.cuts.sort_unstable_by_key(|cut| cut.endpoint);
        (accounting, unknown, budget)
    }

    pub(super) fn checkpoint_cut(&self, request: &LiveRequest) -> Option<usize> {
        let (accounting, _, _) = self.checkpoint_selection(request);
        accounting
            .cuts
            .iter()
            .find(|cut| cut.endpoint > request.consumed_tokens as usize)
            .map(|cut| cut.endpoint)
    }

    pub(super) fn plan_checkpoints(&self, step: &ScheduledStep) -> Vec<CapturePlan> {
        let mut plans = Vec::new();
        for (row, lane) in step.lanes.iter().enumerate() {
            let request = self.request(lane.request).unwrap();
            let kind = step.lane_kind(row);
            if !kind.is_prefill() {
                continue;
            }
            let anchor = request.checkpoint_anchor();
            let (accounting, unknown, budget) = self.checkpoint_selection(request);
            let end = lane.start_position as usize + lane.tokens;
            let cut = accounting
                .cuts
                .iter()
                .find(|cut| cut.endpoint == end)
                .copied();
            let reserve_guard = cut.is_some_and(|cut| cut.endpoint < anchor)
                && request.anchor_guard.is_none()
                && anchor > request.consumed_tokens as usize;
            let rank = RetentionRank {
                use_epoch: accounting.demand_epoch,
                role: cut.map_or(RetentionRole::Endpoint, |cut| cut.role),
                consumed_tokens: end as u64,
            };
            let rows = usize::from(cut.is_some()) + usize::from(reserve_guard);
            let planned_rows = accounting
                .cuts
                .len()
                .saturating_sub(usize::from(request.anchor_guard.is_some()))
                .max(rows);
            let quote = if cut.is_some_and(|cut| cut.role == RetentionRole::Anchor)
                && request.anchor_guard.is_some()
            {
                None
            } else if rows > 0 {
                self.quote_snapshots_preserving(
                    rows,
                    planned_rows,
                    self.checkpoint_headroom(),
                    rank,
                )
            } else {
                None
            };
            plans.push(CapturePlan {
                handle: lane.request,
                accounting,
                anchor,
                cut,
                quote,
                reserve_guard,
                unknown,
                budget,
            });
        }
        plans
    }

    pub(super) fn checkpoint_anchor_waiting(&self, plans: &[CapturePlan]) -> bool {
        !self.submissions.is_empty()
            && plans.iter().any(|plan| {
                plan.cut
                    .is_some_and(|cut| cut.role == RetentionRole::Anchor)
                    && self
                        .request(plan.handle)
                        .is_ok_and(|request| request.anchor_guard.is_none())
                    && plan.quote.is_none()
                    && self.pending_resources(self.checkpoint_headroom(), 1)
            })
    }

    pub(super) fn promote_request_anchor(&mut self, handle: RequestHandle) -> Result<()> {
        let request = self.request(handle)?;
        let key = request.history.key_at(request.checkpoint_anchor() as u64);
        let epoch = request.checkpoints.demand_epoch;
        if let Some(id) = key.and_then(|key| self.cache.ready_by_prefix(&key)) {
            self.cache.promote(id, RetentionRole::Anchor, epoch)?;
        }
        Ok(())
    }

    pub(super) fn commit_checkpoint_plans(&mut self, plans: &[CapturePlan]) -> Result<()> {
        for plan in plans {
            let handle = plan.handle;
            let mut accounting = plan.accounting.clone();
            self.stats.checkpoint_skip_unknown_cost += u64::from(plan.unknown);
            self.stats.checkpoint_skip_work_budget += u64::from(plan.budget);
            let Some(cut) = plan.cut else {
                self.request_mut(handle)?.checkpoints = accounting;
                continue;
            };
            {
                accounting
                    .cuts
                    .retain(|candidate| candidate.endpoint != cut.endpoint);
                accounting.reserved_ns = accounting.reserved_ns.saturating_sub(cut.split_ns);
                accounting.extra_spent_ns = accounting.extra_spent_ns.saturating_add(cut.split_ns);
                self.stats.checkpoint_extra_work_ns = self
                    .stats
                    .checkpoint_extra_work_ns
                    .saturating_add(cut.split_ns);
            }
            let mut destination = if cut.role == RetentionRole::Anchor {
                self.request_mut(handle)?.anchor_guard.take()
            } else {
                None
            };
            if destination.is_none() {
                let mut quote = plan.quote.clone();
                if cut.role == RetentionRole::Anchor {
                    let rank = RetentionRank {
                        use_epoch: accounting.demand_epoch,
                        role: cut.role,
                        consumed_tokens: cut.endpoint as u64,
                    };
                    quote = self.quote_snapshots(1, self.checkpoint_headroom(), rank);
                }
                if let Some(quote) = quote {
                    if let Some(mut rows) = self.apply_snapshot_quote(&quote)? {
                        if plan.reserve_guard {
                            let guard = rows.remove(0);
                            self.request_mut(handle)?.anchor_guard = Some(guard);
                        }
                        destination = rows.pop();
                    }
                }
            }
            if destination.is_none()
                && plan.reserve_guard
                && self.request(handle)?.anchor_guard.is_none()
            {
                let rank = RetentionRank {
                    use_epoch: accounting.demand_epoch,
                    role: RetentionRole::Anchor,
                    consumed_tokens: plan.anchor as u64,
                };
                if let Some(quote) = self.quote_snapshots(1, self.checkpoint_headroom(), rank) {
                    if let Some(rows) = self.apply_snapshot_quote(&quote)? {
                        self.request_mut(handle)?.anchor_guard = rows.first().copied();
                    }
                }
            }
            if let Some(snapshot) = destination {
                let capture = ArmedCapture {
                    snapshot,
                    endpoint: cut.endpoint as u64,
                    role: cut.role,
                    copy_ns: cut.copy_ns,
                };
                self.request_mut(handle)?.checkpoint_capture = Some(capture);
            } else {
                accounting.reserved_ns = accounting.reserved_ns.saturating_sub(cut.copy_ns);
                self.stats.checkpoint_skip_storage += 1;
            }
            self.request_mut(handle)?.checkpoints = accounting;
        }
        Ok(())
    }

    pub(super) fn checkpoint_copy_allowed(
        &self,
        handle: RequestHandle,
        capture: ArmedCapture,
    ) -> Result<bool> {
        if capture.role == RetentionRole::Anchor {
            return Ok(true);
        }
        if self.costs.fresh_duration(GraphKind::Snapshot, 1).is_none() {
            return Ok(false);
        }
        let accounting = &self.request(handle)?.checkpoints;
        let Some(prefill) = accounting.initial_prefill_ns else {
            return Ok(false);
        };
        Ok(accounting
            .extra_spent_ns
            .saturating_add(accounting.reserved_ns)
            <= allowance(prefill, self.policy.checkpoint_extra_work_fraction))
    }

    pub(super) fn credit_checkpoint_decode(
        &mut self,
        step: &ScheduledStep,
        useful: &[u64; REQUEST_LANES],
        generated: &[GeneratedToken],
        timing: crate::execution::ExecutionTiming,
    ) {
        let total: u128 = useful.iter().map(|&tokens| u128::from(tokens)).sum();
        if total == 0 {
            return;
        }
        let Some(gpu_ns) = timing
            .gpu_ns
            .into_iter()
            .flatten()
            .max()
            .filter(|&nanos| nanos > 0)
        else {
            return;
        };
        let work =
            gpu_ns.saturating_sub(timing.readiness.map_or(0, |readiness| readiness.wait_ns()));
        for (row, lane) in step.lanes.iter().enumerate() {
            if step.lane_kind(row).is_prefill() || useful[lane.request.slot as usize] == 0 {
                continue;
            }
            let output = generated
                .iter()
                .filter(|event| event.request == lane.request)
                .count();
            let credit = ((u128::from(work) * output as u128) / total) as u64;
            if let Ok(request) = self.request_mut(lane.request) {
                request.checkpoints.productive_decode_ns = request
                    .checkpoints
                    .productive_decode_ns
                    .saturating_add(credit);
            }
        }
    }

    pub(super) fn allocate_terminal_capture(
        &mut self,
        handle: RequestHandle,
    ) -> Result<Option<SnapshotId>> {
        let request = self.request(handle)?;
        let (mut accounting, _, _) = self.checkpoint_selection(request);
        accounting.cuts.clear();
        accounting.reserved_ns = 0;
        let Some(cost) = self.costs.fresh_duration(GraphKind::Snapshot, 1) else {
            self.stats.checkpoint_skip_unknown_cost += 1;
            return Ok(None);
        };
        let Some(prefill) = accounting.initial_prefill_ns else {
            self.stats.checkpoint_skip_unknown_cost += 1;
            return Ok(None);
        };
        if accounting.extra_spent_ns.saturating_add(cost)
            > allowance(
                prefill.saturating_add(accounting.productive_decode_ns),
                self.policy.checkpoint_extra_work_fraction,
            )
        {
            self.stats.checkpoint_skip_work_budget += 1;
            return Ok(None);
        }
        let rank = RetentionRank {
            use_epoch: accounting.demand_epoch,
            role: RetentionRole::Endpoint,
            consumed_tokens: request.consumed_tokens,
        };
        let quote = self.quote_snapshots(1, self.checkpoint_headroom(), rank);
        let rows = match quote {
            Some(quote) => self.apply_snapshot_quote(&quote)?,
            None => None,
        };
        let snapshot = rows.and_then(|rows| rows.first().copied());
        if snapshot.is_some() {
            accounting.reserved_ns = cost;
            let request = self.request_mut(handle)?;
            request.checkpoint_copy_ns = cost;
            request.checkpoints = accounting;
        } else {
            self.stats.checkpoint_skip_storage += 1;
        }
        Ok(snapshot)
    }

    pub(super) fn commit_checkpoint_copies(&mut self, step: &ScheduledStep) -> Result<()> {
        for (row, lane) in step.lanes.iter().enumerate() {
            if step.lane_kind(row) != GraphKind::Snapshot {
                continue;
            }
            let request = self.request_mut(lane.request)?;
            let cost = std::mem::take(&mut request.checkpoint_copy_ns);
            request.checkpoints.reserved_ns = request.checkpoints.reserved_ns.saturating_sub(cost);
            request.checkpoints.extra_spent_ns =
                request.checkpoints.extra_spent_ns.saturating_add(cost);
            self.stats.checkpoint_extra_work_ns =
                self.stats.checkpoint_extra_work_ns.saturating_add(cost);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measure(scheduler: &mut RequestScheduler, kind: GraphKind, tokens: usize, nanos: u64) {
        scheduler.costs.set_measured(
            &ScheduledStep {
                draft_tokens: 0,
                kind,
                batch_capacity: 1,
                token_capacity: tokens,
                packed_token_capacity: 0,
                mixed: None,
                lanes: Vec::new(),
                requires_feedback: true,
            },
            nanos,
        );
    }

    fn linear_costs(scheduler: &mut RequestScheduler) {
        for capacity in scheduler.prefill_catalog.clone() {
            measure(
                scheduler,
                GraphKind::Prefill,
                capacity,
                capacity as u64 * 1000,
            );
        }
        measure(scheduler, GraphKind::Snapshot, 1, 1);
    }

    #[test]
    fn missing_cost_retries_original_uncached_interval_after_progress() {
        let mut scheduler = super::super::tests::scheduler(1, 128, 16);
        scheduler
            .graph_durations(&super::super::tests::duration_graphs(1))
            .unwrap();
        let mut spec = super::super::tests::spec(1024, 3);
        spec.message_boundaries = vec![128, 256, 512, 896];
        let handle = scheduler.admit(spec).unwrap();
        let (cold, unknown, _) = scheduler.checkpoint_selection(scheduler.request(handle).unwrap());
        assert!(unknown);
        assert_eq!(cold.initial_prefill_ns, None);
        assert_eq!(
            cold.cuts.iter().map(|cut| cut.endpoint).collect::<Vec<_>>(),
            [896]
        );
        scheduler.request_mut(handle).unwrap().checkpoints = cold;
        scheduler.request_mut(handle).unwrap().consumed_tokens = 128;
        linear_costs(&mut scheduler);
        let (warm, unknown, _) = scheduler.checkpoint_selection(scheduler.request(handle).unwrap());
        assert!(!unknown);
        assert_eq!(warm.initial_prefill_ns, Some(1_024_000));
        assert!(warm.cuts.len() > 1);
        assert!(warm.cuts.iter().all(|cut| cut.endpoint > 128));
    }

    #[test]
    fn unknown_or_stale_competing_shape_disables_extras() {
        let mut scheduler = super::super::tests::scheduler(1, 128, 16);
        let mut spec = super::super::tests::spec(1024, 3);
        spec.message_boundaries = vec![128, 256, 512, 896];
        let handle = scheduler.admit(spec).unwrap();
        linear_costs(&mut scheduler);
        measure(&mut scheduler, GraphKind::Prefill, 512, 0);
        let (unknown, _, _) = scheduler.checkpoint_selection(scheduler.request(handle).unwrap());
        assert_eq!(unknown.initial_prefill_ns, None);
        assert_eq!(unknown.cuts.len(), 1);
        linear_costs(&mut scheduler);
        scheduler
            .advance_to(scheduler.policy.estimate_ttl_ns)
            .unwrap();
        let (stale, _, _) = scheduler.checkpoint_selection(scheduler.request(handle).unwrap());
        assert_eq!(stale.initial_prefill_ns, None);
        assert_eq!(stale.cuts.len(), 1);
    }

    #[test]
    fn joint_storage_selection_keeps_supplementaries_nearest_anchor() {
        for snapshots in [2, 3] {
            let mut scheduler = super::super::tests::scheduler(1, 128, snapshots);
            linear_costs(&mut scheduler);
            let mut spec = super::super::tests::spec(1024, 3);
            spec.message_boundaries = (128..1024).step_by(128).collect();
            let handle = scheduler.admit(spec).unwrap();
            let (selected, _, _) =
                scheduler.checkpoint_selection(scheduler.request(handle).unwrap());
            let endpoints: Vec<_> = selected.cuts.iter().map(|cut| cut.endpoint).collect();
            assert_eq!(
                endpoints,
                if snapshots == 2 {
                    vec![768, 896]
                } else {
                    vec![640, 768, 896]
                }
            );
        }
    }

    #[test]
    fn decode_attribution_excludes_prefill_and_uses_mixed_useful_denominator() {
        let mut scheduler = super::super::tests::scheduler(2, 128, 16);
        let prefill = scheduler.admit(super::super::tests::spec(128, 3)).unwrap();
        let decoder = scheduler.admit(super::super::tests::spec(128, 3)).unwrap();
        let step = ScheduledStep {
            draft_tokens: 0,
            kind: GraphKind::Mixed,
            batch_capacity: 2,
            token_capacity: 1,
            packed_token_capacity: 0,
            mixed: Some(MixedStep {
                prefill_capacity: 128,
                prefill_row: Some(0),
                speculative: false,
            }),
            lanes: vec![
                ScheduledLane {
                    request: prefill,
                    start_position: 0,
                    tokens: 128,
                },
                ScheduledLane {
                    request: decoder,
                    start_position: 128,
                    tokens: 1,
                },
            ],
            requires_feedback: true,
        };
        let mut useful = [0; REQUEST_LANES];
        useful[prefill.slot as usize] = 128;
        useful[decoder.slot as usize] = 1;
        let generated = [
            GeneratedToken {
                request: prefill,
                token: 1,
                finished: false,
            },
            GeneratedToken {
                request: decoder,
                token: 1,
                finished: false,
            },
        ];
        scheduler.credit_checkpoint_decode(&step, &useful, &generated, Default::default());
        assert_eq!(
            scheduler
                .request(decoder)
                .unwrap()
                .checkpoints
                .productive_decode_ns,
            0
        );
        let mut timing = crate::execution::ExecutionTiming::default();
        timing.gpu_ns[0] = Some(1290);
        timing.gpu_ns[1] = Some(2580);
        scheduler.credit_checkpoint_decode(&step, &useful, &generated, timing);
        assert_eq!(
            scheduler
                .request(prefill)
                .unwrap()
                .checkpoints
                .productive_decode_ns,
            0
        );
        assert_eq!(
            scheduler
                .request(decoder)
                .unwrap()
                .checkpoints
                .productive_decode_ns,
            20
        );
        let mut readiness = crate::execution::ReadinessIntervals::default();
        readiness.len = 1;
        readiness.intervals[0] = [1500, 210];
        timing.readiness = Some(readiness);
        scheduler.credit_checkpoint_decode(&step, &useful, &generated, timing);
        assert_eq!(
            scheduler
                .request(decoder)
                .unwrap()
                .checkpoints
                .productive_decode_ns,
            30
        );
    }

    #[test]
    fn joint_split_ledger_prices_interacting_cuts_and_keeps_spent_work_on_stale_quote() {
        let mut scheduler = super::super::tests::scheduler(1, 128, 16);
        scheduler
            .configure_policy(PolicyConfig {
                checkpoint_extra_work_fraction: 0.5,
                ..Default::default()
            })
            .unwrap();
        for (tokens, nanos) in [(128, 100), (256, 110), (512, 200)] {
            measure(&mut scheduler, GraphKind::Prefill, tokens, nanos);
        }
        measure(&mut scheduler, GraphKind::Snapshot, 1, 1);
        let mut spec = super::super::tests::spec(1024, 3);
        spec.message_boundaries = (128..1024).step_by(128).collect();
        let handle = scheduler.admit(spec).unwrap();
        let (selected, _, _) = scheduler.checkpoint_selection(scheduler.request(handle).unwrap());
        assert_eq!(selected.initial_prefill_ns, Some(400));
        assert_eq!(
            selected
                .cuts
                .iter()
                .map(|cut| cut.endpoint)
                .collect::<Vec<_>>(),
            [256, 512, 640, 768, 896, 1024]
        );
        assert_eq!(selected.reserved_ns, 115);
        assert_eq!(
            selected
                .cuts
                .iter()
                .find(|cut| cut.endpoint == 256)
                .unwrap()
                .split_ns,
            20
        );
        assert_eq!(
            selected
                .cuts
                .iter()
                .find(|cut| cut.endpoint == 640)
                .unwrap()
                .split_ns,
            90
        );
        scheduler.request_mut(handle).unwrap().phase = RequestPhase::Prefill;
        let plan = scheduler.plan().unwrap();
        assert_eq!(plan.lanes[0].tokens, 256);
        let _held = scheduler.reserve_snapshot(0).unwrap();
        scheduler.commit(&plan, Fence(1)).unwrap();
        let accounting = scheduler.request(handle).unwrap().checkpoint_accounting();
        assert_eq!(accounting.extra_spent_ns, 20);
        assert_eq!(accounting.reserved_ns, 94);
        assert!(
            scheduler
                .request(handle)
                .unwrap()
                .checkpoint_capture
                .is_none()
        );
        assert!(scheduler.request(handle).unwrap().anchor_guard.is_some());
    }

    #[test]
    fn restored_prefix_promotes_exact_anchor_outside_its_parent_chain_without_rewind() {
        let mut scheduler = super::super::tests::scheduler(1, 128, 16);
        let mut history = PrefixHistory::new([9; 32]);
        history.append(&(0..1024).collect::<Vec<_>>(), &[]).unwrap();
        let mut ids = Vec::new();
        for consumed in [1024u64, 512] {
            let snapshot = scheduler.reserve_snapshot(0).unwrap();
            let pages = (0..consumed / 64)
                .map(|_| scheduler.free_pages.pop_first().unwrap())
                .collect();
            ids.push(
                scheduler
                    .cache
                    .publish(
                        &history,
                        &Publication {
                            parent: None,
                            consumed_tokens: consumed,
                            tail_pages: pages,
                            snapshot: StateSnapshot {
                                id: snapshot,
                                consumed_tokens: consumed,
                            },
                            ready_after: Fence(0),
                        },
                    )
                    .unwrap()
                    .id(),
            );
        }
        let mut spec = super::super::tests::spec(1024, 3);
        spec.message_boundaries = vec![512];
        let handle = scheduler.admit(spec).unwrap();
        assert_eq!(scheduler.request(handle).unwrap().prefix, Some(ids[0]));
        let plan = scheduler.plan().unwrap();
        assert_eq!(plan.kind, GraphKind::Restore);
        scheduler.commit(&plan, Fence(1)).unwrap();
        assert_eq!(scheduler.request(handle).unwrap().consumed_tokens, 1024);
        assert_eq!(
            scheduler
                .cache
                .checkpoint(ids[1])
                .unwrap()
                .retention_rank()
                .role,
            RetentionRole::Anchor
        );
    }

    #[test]
    fn completed_conditioned_windows_preserve_anchor_eligibility_and_baseline_cuts() {
        let mut scheduler = super::super::tests::scheduler(1, 128, 16);
        let mut spec = super::super::tests::spec(768, 3);
        spec.message_boundaries = vec![256, 400, 512];
        let handle = scheduler.admit(spec).unwrap();
        scheduler
            .request_mut(handle)
            .unwrap()
            .checkpoints
            .retain_window_ends([256, 512, 768].into_iter());
        assert!(
            scheduler
                .request(handle)
                .unwrap()
                .conditioned_windows
                .is_empty()
        );
        assert_eq!(scheduler.request(handle).unwrap().checkpoint_anchor(), 400);
        let (accounting, _, _) = scheduler.checkpoint_selection(scheduler.request(handle).unwrap());
        assert_eq!(accounting.initial_prefill_ns, Some(3000));
    }

    #[test]
    fn zero_fraction_never_reserves_or_charges_extra_copy_work() {
        let mut scheduler = super::super::tests::scheduler(1, 128, 16);
        scheduler
            .configure_policy(PolicyConfig {
                checkpoint_extra_work_fraction: 0.0,
                ..Default::default()
            })
            .unwrap();
        linear_costs(&mut scheduler);
        let mut spec = super::super::tests::spec(1024, 3);
        spec.message_boundaries = vec![128, 512, 896];
        let handle = scheduler.admit(spec).unwrap();
        let (selected, _, _) = scheduler.checkpoint_selection(scheduler.request(handle).unwrap());
        assert_eq!(selected.cuts.len(), 1);
        assert_eq!(selected.cuts[0].role, RetentionRole::Anchor);
        assert!(
            scheduler
                .allocate_terminal_capture(handle)
                .unwrap()
                .is_none()
        );
        assert_eq!(scheduler.free_snapshot_count(), 16);
        assert_eq!(scheduler.stats().checkpoint_extra_work_ns, 0);
    }
}
