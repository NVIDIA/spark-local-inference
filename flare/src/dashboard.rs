// SPDX-License-Identifier: Apache-2.0
//! Native dashboard snapshots derived from the terminal dashboard's telemetry.
use flare_helios::server::{
    ServerSnapshot,
    view::{History, Rates, Snapshot},
};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Default)]
pub struct DashboardPoint {
    pub elapsed_seconds: f64,
    pub prefill_tokens_per_second: f64,
    pub generated_tokens_per_second: f64,
    pub verify_steps_per_second: f64,
    /// Accepted drafts per draft-offering request round, not a fraction.
    pub acceptance: Option<f64>,
    /// Allocated drafts per progressing verification request round.
    pub depth: Option<f64>,
}
#[derive(Clone, Debug, Default)]
pub struct DashboardView {
    pub available: bool,
    pub session_id: u64,
    pub uptime_seconds: f64,
    pub prefill_tokens_per_second: f64,
    pub generated_tokens_per_second: f64,
    pub verify_steps_per_second: f64,
    pub acceptance: Option<f64>,
    pub depth: Option<f64>,
    pub weight_bytes_per_second: Option<f64>,
    pub prefill_tokens: u64,
    pub generated_tokens: u64,
    pub reused_tokens: u64,
    pub active: u64,
    pub pending: u64,
    pub completed: u64,
    pub failed: u64,
    pub ingress: u64,
    pub cached_pages: u64,
    pub free_pages: u64,
    pub total_pages: u64,
    pub page_tokens: u64,
    pub host_hits: u64,
    pub host_reused_tokens: u64,
    pub host_memory_records: u64,
    pub host_disk_records: u64,
    pub host_bytes_out: u64,
    pub host_bytes_in: u64,
    pub points: Vec<DashboardPoint>,
}

#[derive(Default)]
pub(crate) struct DashboardHistory {
    timeline: Option<Timeline>,
}
struct Timeline {
    session: u64,
    origin: Instant,
    origin_ns: u64,
    origin_uptime: f64,
    last_seen_ns: u64,
    last_point_ns: u64,
    uptime: f64,
    ingress: u64,
    latest: Snapshot,
    history: History,
}
const SAMPLE_NS: u64 = 500_000_000;
impl DashboardHistory {
    pub(crate) fn clear(&mut self) {
        self.timeline = None;
    }
    pub(crate) fn sample(&mut self, sample: &ServerSnapshot) -> DashboardView {
        self.observe(
            sample.session_id,
            sample.sampled_at_ns,
            sample.uptime,
            sample.ingress_active as u64,
            sample.model,
        )
    }
    fn observe(
        &mut self,
        session: u64,
        at: u64,
        uptime: Duration,
        ingress: u64,
        model: Snapshot,
    ) -> DashboardView {
        if let Some(previous) = &self.timeline {
            if previous.session == session && at <= previous.last_seen_ns {
                return previous.view();
            }
        }
        let reset = self.timeline.as_ref().is_none_or(|previous| {
            previous.session != session
                || counters(model)
                    .into_iter()
                    .zip(counters(previous.latest))
                    .any(|(after, before)| after < before)
        });
        if reset {
            let origin = Instant::now();
            self.timeline = Some(Timeline {
                session,
                origin,
                origin_ns: at,
                origin_uptime: uptime.as_secs_f64(),
                last_seen_ns: at,
                last_point_ns: at,
                uptime: uptime.as_secs_f64(),
                ingress,
                latest: model,
                history: History::new(origin, model),
            });
        } else if let Some(timeline) = &mut self.timeline {
            timeline.last_seen_ns = at;
            timeline.uptime = uptime.as_secs_f64();
            timeline.ingress = ingress;
            timeline.latest = model;
            if at - timeline.last_point_ns >= SAMPLE_NS {
                let instant = timeline.origin + Duration::from_nanos(at - timeline.origin_ns);
                timeline.history.sample(instant, model);
                timeline.last_point_ns = at;
            }
        }
        self.timeline.as_ref().unwrap().view()
    }
}
fn counters(s: Snapshot) -> [u64; 15] {
    [
        s.prefill_tokens,
        s.generated_tokens,
        s.verify_steps,
        s.draft_steps,
        s.verification_drafts,
        s.verification_rounds,
        s.accepted_drafts,
        s.draft_rounds,
        s.reused_tokens,
        s.completed,
        s.failed,
        s.host_hits,
        s.host_reused_tokens,
        s.host_bytes_out,
        s.host_bytes_in,
    ]
}
fn point(elapsed_seconds: f64, rates: Rates) -> DashboardPoint {
    DashboardPoint {
        elapsed_seconds,
        prefill_tokens_per_second: rates.prefill,
        generated_tokens_per_second: rates.generated,
        verify_steps_per_second: rates.verify,
        acceptance: rates.acceptance,
        depth: rates.depth,
    }
}
impl Timeline {
    fn view(&self) -> DashboardView {
        let s = self.latest;
        let r = self.history.rates();
        DashboardView {
            available: true,
            session_id: self.session,
            uptime_seconds: self.uptime,
            prefill_tokens_per_second: r.prefill,
            generated_tokens_per_second: r.generated,
            verify_steps_per_second: r.verify,
            acceptance: r.acceptance,
            depth: r.depth,
            weight_bytes_per_second: (s.decode_weight_bytes > 0).then_some(r.weights),
            prefill_tokens: s.prefill_tokens,
            generated_tokens: s.generated_tokens,
            reused_tokens: s.reused_tokens,
            active: s.active,
            pending: s.pending,
            completed: s.completed,
            failed: s.failed,
            ingress: self.ingress,
            cached_pages: s.cached_pages,
            free_pages: s.free_pages,
            total_pages: s.total_pages,
            page_tokens: s.page_tokens,
            host_hits: s.host_hits,
            host_reused_tokens: s.host_reused_tokens,
            host_memory_records: s.host_memory_records,
            host_disk_records: s.host_disk_records,
            host_bytes_out: s.host_bytes_out,
            host_bytes_in: s.host_bytes_in,
            points: self
                .history
                .points()
                .iter()
                .map(|(elapsed, rates)| point(self.origin_uptime + elapsed, *rates))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample(history: &mut DashboardHistory, at_ms: u64, model: Snapshot) -> DashboardView {
        history.observe(
            1,
            at_ms * 1_000_000,
            Duration::from_millis(10_000 + at_ms),
            7,
            model,
        )
    }
    #[test]
    fn rates_preserve_cli_units_and_speculation_ratios_are_not_percentages() {
        let mut history = DashboardHistory::default();
        sample(&mut history, 0, Snapshot::default());
        let view = sample(
            &mut history,
            2000,
            Snapshot {
                prefill_tokens: 100,
                generated_tokens: 40,
                verify_steps: 8,
                accepted_drafts: 12,
                draft_rounds: 4,
                verification_drafts: 20,
                verification_rounds: 8,
                decode_weight_bytes: 1000,
                ..Default::default()
            },
        );
        assert_eq!(view.prefill_tokens_per_second, 50.0);
        assert_eq!(view.generated_tokens_per_second, 20.0);
        assert_eq!(view.verify_steps_per_second, 4.0);
        assert_eq!(view.acceptance, Some(3.0));
        assert_eq!(view.depth, Some(2.5));
        assert_eq!(view.weight_bytes_per_second, Some(4000.0));
        assert_eq!(view.uptime_seconds, 12.0);
        assert_eq!(view.points[0].elapsed_seconds, 12.0);
    }
    #[test]
    fn generation_and_prefill_follow_the_cli_two_second_window() {
        let mut history = DashboardHistory::default();
        sample(&mut history, 0, Snapshot::default());
        sample(
            &mut history,
            1000,
            Snapshot {
                generated_tokens: 100,
                prefill_tokens: 200,
                ..Default::default()
            },
        );
        sample(
            &mut history,
            2000,
            Snapshot {
                generated_tokens: 200,
                prefill_tokens: 400,
                ..Default::default()
            },
        );
        let view = sample(
            &mut history,
            3000,
            Snapshot {
                generated_tokens: 200,
                prefill_tokens: 400,
                ..Default::default()
            },
        );
        assert_eq!(view.generated_tokens_per_second, 50.0);
        assert_eq!(view.prefill_tokens_per_second, 100.0);
    }
    #[test]
    fn unknown_ratios_and_undeclared_weights_remain_unknown() {
        let mut history = DashboardHistory::default();
        sample(&mut history, 0, Snapshot::default());
        let view = sample(
            &mut history,
            500,
            Snapshot {
                verify_steps: 5,
                ..Default::default()
            },
        );
        assert!(view.available);
        assert_eq!(view.acceptance, None);
        assert_eq!(view.depth, None);
        assert_eq!(view.weight_bytes_per_second, None);
        assert_eq!(view.points[0].acceptance, None);
        let declared = sample(
            &mut history,
            1000,
            Snapshot {
                verify_steps: 5,
                decode_weight_bytes: 123,
                ..Default::default()
            },
        );
        assert!(declared.weight_bytes_per_second.is_some());
    }
    #[test]
    fn sampling_is_throttled_and_nonadvancing_samples_do_not_replace_totals() {
        let mut history = DashboardHistory::default();
        sample(&mut history, 0, Snapshot::default());
        for tick in 1..5 {
            assert!(
                sample(
                    &mut history,
                    tick * 100,
                    Snapshot {
                        generated_tokens: tick,
                        ..Default::default()
                    }
                )
                .points
                .is_empty()
            );
        }
        let sampled = sample(
            &mut history,
            500,
            Snapshot {
                generated_tokens: 5,
                ..Default::default()
            },
        );
        assert_eq!(sampled.points.len(), 1);
        let stale = sample(
            &mut history,
            500,
            Snapshot {
                generated_tokens: 1000,
                ..Default::default()
            },
        );
        assert_eq!(stale.generated_tokens, 5);
        assert_eq!(stale.points.len(), 1);
        let older = sample(&mut history, 400, Snapshot::default());
        assert_eq!(older.generated_tokens, 5);
        assert_eq!(older.generated_tokens_per_second, 10.0);
    }
    #[test]
    fn session_changes_and_counter_resets_clear_prior_rates_and_history() {
        let mut history = DashboardHistory::default();
        sample(&mut history, 0, Snapshot::default());
        sample(
            &mut history,
            1000,
            Snapshot {
                generated_tokens: 50,
                ..Default::default()
            },
        );
        let reset = sample(
            &mut history,
            2000,
            Snapshot {
                generated_tokens: 2,
                ..Default::default()
            },
        );
        assert!(reset.points.is_empty());
        assert_eq!(reset.generated_tokens_per_second, 0.0);
        let replaced = history.observe(
            2,
            3_000_000_000,
            Duration::from_millis(50),
            0,
            Snapshot::default(),
        );
        assert_eq!(replaced.session_id, 2);
        assert_eq!(replaced.generated_tokens, 0);
        assert!(replaced.points.is_empty());
        assert_eq!(replaced.uptime_seconds, 0.05);
        history.clear();
        assert!(history.timeline.is_none());
        assert!(!DashboardView::default().available);
    }
    #[test]
    fn history_is_bounded_and_cache_gauges_are_copied_without_inventing_used_pages() {
        let mut history = DashboardHistory::default();
        sample(&mut history, 0, Snapshot::default());
        let mut view = DashboardView::default();
        for tick in 1..1000 {
            view = sample(
                &mut history,
                tick * 500,
                Snapshot {
                    generated_tokens: tick,
                    cached_pages: 3,
                    free_pages: 5,
                    total_pages: 16,
                    page_tokens: 32,
                    host_hits: 8,
                    host_reused_tokens: 256,
                    host_memory_records: 2,
                    host_disk_records: 4,
                    host_bytes_out: 1024,
                    host_bytes_in: 512,
                    reused_tokens: 128,
                    active: 2,
                    pending: 3,
                    completed: 4,
                    failed: 1,
                    ..Default::default()
                },
            );
        }
        assert!(view.points.len() <= 241);
        assert!(
            view.points.last().unwrap().elapsed_seconds
                - view.points.first().unwrap().elapsed_seconds
                <= 120.0
        );
        assert_eq!(
            (
                view.cached_pages,
                view.free_pages,
                view.total_pages,
                view.page_tokens
            ),
            (3, 5, 16, 32)
        );
        assert_eq!(
            (
                view.host_hits,
                view.host_reused_tokens,
                view.host_memory_records,
                view.host_disk_records
            ),
            (8, 256, 2, 4)
        );
        assert_eq!(
            (view.host_bytes_out, view.host_bytes_in, view.reused_tokens),
            (1024, 512, 128)
        );
        assert_eq!(
            (
                view.active,
                view.pending,
                view.completed,
                view.failed,
                view.ingress
            ),
            (2, 3, 4, 1, 7)
        );
    }
}
