/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Per-acquisition byte and elapsed-time accounting and its presentation.

use std::{
    sync::{Arc, Mutex, mpsc, atomic::{AtomicBool, Ordering}},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crate::startup::{Phase, StartupProgress};

/// Redraw interval for one acquisition row on a terminal.
const TTY_INTERVAL: Duration = Duration::from_millis(125);
/// Counter resampling interval for redirected output. `StartupProgress` prints a
/// redirected row every ten seconds, so the detail is rewritten more often than
/// it is printed but far less often than a redraw.
const LOG_INTERVAL: Duration = Duration::from_secs(1);
/// Display cap for an estimated remaining time. Longer estimates are noise from
/// an early sample and would be truncated to a meaningless number anyway.
const ETA_CAP: Duration = Duration::from_secs(359_999);

/// One shared handle for an acquisition. Clones observe the same counters.
#[derive(Clone, Debug)]
pub struct AcquisitionProgress {
    start: Instant,
    counters: Arc<Mutex<Counters>>,
    cancelled: Arc<AtomicBool>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Counters {
    phase: AcquisitionPhase,
    planning_completed: u64,
    planning_total: Option<u64>,
    network_bytes: u64,
    network_total: Option<u64>,
    local_source_bytes: u64,
    materialized_copy_bytes: u64,
    materialized_fill_bytes: u64,
    verification_bytes: u64,
}

/// Immutable acquisition counters for phase and final reporting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AcquisitionSnapshot {
    pub phase: AcquisitionPhase,
    /// Companion file identities resolved or attempted while preparing a download.
    pub planning_completed: u64,
    pub planning_total: Option<u64>,
    pub network_bytes: u64,
    /// Declared cumulative value `network_bytes` should reach, when the acquired
    /// byte count is known. `None` means unknown; it never describes verification
    /// or materialization work.
    pub network_total: Option<u64>,
    /// Bytes successfully read from local source payloads.
    pub local_source_bytes: u64,
    /// Bytes successfully written by derivation copy operations.
    pub materialized_copy_bytes: u64,
    /// Bytes successfully written by derivation fill operations.
    pub materialized_fill_bytes: u64,
    /// Bytes successfully read for byte comparison or fed to the final target hash.
    pub verification_bytes: u64,
    pub elapsed: Duration,
}

/// Actual acquisition work, independent of how a client presents progress.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AcquisitionPhase {
    #[default]
    Resolving,
    Downloading,
    PreparingDownload,
    Checking,
    Installing,
    Complete,
}

impl AcquisitionPhase {
    pub fn label(self) -> &'static str {
        match self {
            Self::Resolving => "Resolving source",
            Self::Downloading => "Downloading",
            Self::PreparingDownload => "Preparing download",
            Self::Checking => "Checking core",
            Self::Installing => "Installing core",
            Self::Complete => "Installed",
        }
    }
}

impl AcquisitionSnapshot {
    pub fn materialized_bytes(&self) -> u64 {
        self.materialized_copy_bytes
            .saturating_add(self.materialized_fill_bytes)
    }

    /// Read one counter selected by name.
    pub fn counter(&self, counter: Counter) -> u64 {
        match counter {
            Counter::Network => self.network_bytes,
            Counter::LocalSource => self.local_source_bytes,
            Counter::Materialized => self.materialized_bytes(),
            Counter::Verification => self.verification_bytes,
        }
    }
}

/// An acquisition counter that a progress row can meter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Counter {
    Network,
    LocalSource,
    Materialized,
    Verification,
}

impl Default for AcquisitionProgress {
    fn default() -> Self {
        Self::new()
    }
}

impl AcquisitionProgress {
    pub fn begin_download_planning(&self, files: u64) {
        self.update(|counters| {
            counters.phase = AcquisitionPhase::PreparingDownload;
            counters.planning_completed = 0;
            counters.planning_total = Some(files);
            counters.network_total = None;
        });
    }

    pub fn record_planned_file(&self) {
        self.update(|counters| {
            counters.planning_completed = counters.planning_completed.saturating_add(1);
        });
    }

    pub fn set_phase(&self, phase: AcquisitionPhase) {
        self.update(|counters| counters.phase = phase);
    }
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            counters: Arc::new(Mutex::new(Counters::default())),
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Requests cooperative cancellation at the acquisition's next safe boundary.
    pub fn request_cancel(&self) { self.cancelled.store(true, Ordering::Release); }

    pub fn is_cancelled(&self) -> bool { self.cancelled.load(Ordering::Acquire) }

    pub fn record_network_bytes(&self, bytes: u64) {
        self.update(|counters| {
            counters.network_bytes = counters.network_bytes.saturating_add(bytes)
        });
    }

    /// Declare the cumulative network byte count this acquisition expects to
    /// reach, including bytes already counted, or `None` when it is unknown.
    pub fn record_network_total(&self, total: Option<u64>) {
        self.update(|counters| counters.network_total = total);
    }

    pub fn record_local_source_bytes(&self, bytes: u64) {
        self.update(|counters| {
            counters.local_source_bytes = counters.local_source_bytes.saturating_add(bytes)
        });
    }

    pub fn record_materialized_copy_bytes(&self, bytes: u64) {
        self.update(|counters| {
            counters.materialized_copy_bytes =
                counters.materialized_copy_bytes.saturating_add(bytes)
        });
    }

    pub fn record_materialized_fill_bytes(&self, bytes: u64) {
        self.update(|counters| {
            counters.materialized_fill_bytes =
                counters.materialized_fill_bytes.saturating_add(bytes)
        });
    }

    pub fn record_verification_bytes(&self, bytes: u64) {
        self.update(|counters| {
            counters.verification_bytes = counters.verification_bytes.saturating_add(bytes)
        });
    }

    /// Copy every counter and the elapsed time under one lock. Safe to call from
    /// any thread at any rate while workers record.
    pub fn snapshot(&self) -> AcquisitionSnapshot {
        let counters = *self
            .counters
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        AcquisitionSnapshot {
            phase: counters.phase,
            planning_completed: counters.planning_completed,
            planning_total: counters.planning_total,
            network_bytes: counters.network_bytes,
            network_total: counters.network_total,
            local_source_bytes: counters.local_source_bytes,
            materialized_copy_bytes: counters.materialized_copy_bytes,
            materialized_fill_bytes: counters.materialized_fill_bytes,
            verification_bytes: counters.verification_bytes,
            elapsed: self.start.elapsed(),
        }
    }

    fn update(&self, update: impl FnOnce(&mut Counters)) {
        let mut counters = self
            .counters
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        update(&mut counters);
    }
}

/// One counter's transfer rate over a baseline interval.
#[derive(Clone, Debug)]
pub struct Rate {
    started: Instant,
    baseline: u64,
}

impl Rate {
    /// Begin sampling a counter whose current value is `done`.
    pub fn start(done: u64) -> Self {
        Self {
            started: Instant::now(),
            baseline: done,
        }
    }

    /// Restart sampling at the counter's current value.
    pub fn reset(&mut self, done: u64) {
        *self = Self::start(done);
    }

    /// Bytes per second since the baseline. Zero until time passes.
    pub fn bytes_per_second(&self, done: u64) -> f64 {
        let elapsed = self.started.elapsed().as_secs_f64();
        if elapsed <= 0.0 {
            return 0.0;
        }
        done.saturating_sub(self.baseline) as f64 / elapsed
    }

    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }
}

/// Formats a byte count in binary units with one decimal for scaled values.
pub fn format_bytes(bytes: u64) -> String {
    let (value, unit, divisor) = scale(bytes);
    if divisor == 1 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {unit}")
    }
}

fn scale(bytes: u64) -> (f64, &'static str, u64) {
    const UNITS: [(u64, &str); 4] = [
        (1 << 40, "TiB"),
        (1 << 30, "GiB"),
        (1 << 20, "MiB"),
        (1 << 10, "KiB"),
    ];
    for (divisor, unit) in UNITS {
        if bytes >= divisor {
            return (bytes as f64 / divisor as f64, unit, divisor);
        }
    }
    (bytes as f64, "B", 1)
}

/// Formats a transfer rate. A non-finite or non-positive rate renders as idle.
pub fn format_rate(bytes_per_second: f64) -> String {
    if !bytes_per_second.is_finite() || bytes_per_second < 1.0 {
        return "0 B/s".to_owned();
    }
    format!("{}/s", format_bytes(bytes_per_second as u64))
}

/// Formats a remaining duration at the precision its magnitude needs.
pub fn format_eta(remaining: Duration) -> String {
    let seconds = remaining.as_secs().min(ETA_CAP.as_secs());
    if seconds >= 3600 {
        format!("{}h{:02}m", seconds / 3600, seconds % 3600 / 60)
    } else if seconds >= 60 {
        format!("{}m{:02}s", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s")
    }
}

/// Formats one transfer row body: transferred bytes, optional total, rate, and
/// the ETA that the rate implies for the remaining bytes. `done` and `total`
/// must share a base: both cumulative (`AcquisitionSnapshot::network_total`
/// with `network_bytes`) or both measured from the same stage start. A count
/// above the total is reported as-is; the total and ETA are dropped rather than
/// clamping the reading.
///
/// The result carries no label, no newline, and no terminal escape sequence, so
/// a caller that owns its own layout (Orbit frames, tests) can place it anywhere.
pub fn format_progress(done: u64, total: Option<u64>, rate: Option<f64>) -> String {
    let total = total.filter(|total| *total > 0 && *total >= done);
    let mut text = match total {
        Some(total) => {
            let (_, unit, divisor) = scale(total);
            if divisor == 1 {
                format!("{done} / {total} B")
            } else {
                format!(
                    "{:.1} / {:.1} {unit}",
                    done as f64 / divisor as f64,
                    total as f64 / divisor as f64
                )
            }
        }
        None => format_bytes(done),
    };
    let rate = rate.filter(|rate| rate.is_finite() && *rate >= 1.0);
    if let Some(rate) = rate {
        text.push_str("  ");
        text.push_str(&format_rate(rate));
        if let Some(remaining) = total
            .filter(|total| *total > done)
            .map(|total| (total - done) as f64 / rate)
            .filter(|remaining| remaining.is_finite() && *remaining <= ETA_CAP.as_secs_f64())
            .map(Duration::from_secs_f64)
        {
            text.push_str("  ETA ");
            text.push_str(&format_eta(remaining));
        }
    }
    text
}

/// Presents one acquisition on `StartupProgress`, one stage row at a time.
///
/// A worker resamples the counters and rewrites the active row's detail, so the
/// row shows transferred bytes, rate, and ETA while the work is running rather
/// than only when it finishes. Callers announce stages as the acquisition moves
/// between network transfer, materialization, verification, and installation.
pub struct AcquisitionDisplay<'a> {
    progress: &'a StartupProgress,
    counters: AcquisitionProgress,
    stage: Arc<Mutex<Option<Stage>>>,
    worker: Option<Worker>,
}

struct Stage {
    phase: Phase,
    counter: Counter,
    total: Option<u64>,
    baseline: u64,
    rate: Rate,
    advanced: u64,
    detail: String,
}

struct Worker {
    stop: mpsc::Sender<()>,
    handle: JoinHandle<()>,
}

impl<'a> AcquisitionDisplay<'a> {
    /// Begin presenting `counters`. The display holds no stage until `stage`.
    pub fn new(progress: &'a StartupProgress, counters: &AcquisitionProgress) -> Self {
        let stage: Arc<Mutex<Option<Stage>>> = Arc::new(Mutex::new(None));
        let worker = progress.is_enabled().then(|| {
            let interval = if progress.is_terminal() {
                TTY_INTERVAL
            } else {
                LOG_INTERVAL
            };
            let counters = counters.clone();
            let shared = stage.clone();
            let (stop, receiver) = mpsc::channel();
            let handle = thread::spawn(move || {
                while matches!(
                    receiver.recv_timeout(interval),
                    Err(mpsc::RecvTimeoutError::Timeout)
                ) {
                    if let Some(stage) = shared
                        .lock()
                        .unwrap_or_else(|error| error.into_inner())
                        .as_mut()
                    {
                        stage.refresh(&counters);
                    }
                }
            });
            Worker { stop, handle }
        });
        Self {
            progress,
            counters: counters.clone(),
            stage,
            worker,
        }
    }

    /// Finish the active row and open one for `label`, metered by `counter`.
    ///
    /// `total` is how many units this stage expects to add to the counter, not
    /// the counter's absolute value: a stage that begins with the counter at
    /// 3 GiB and expects to transfer 4 GiB passes 4 GiB, not 7 GiB. A total
    /// enables a filled bar and an ETA; without one the row shows an
    /// indeterminate bar, the transferred bytes, and the rate.
    pub fn stage(&self, label: &str, counter: Counter, total: Option<u64>) {
        let baseline = self.counters.snapshot().counter(counter);
        let mut slot = self.stage.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(previous) = slot.take() {
            previous.complete(&self.counters);
        }
        let mut stage = Stage {
            phase: self.progress.phase(label, total),
            counter,
            total,
            baseline,
            rate: Rate::start(baseline),
            advanced: 0,
            detail: String::new(),
        };
        stage.refresh(&self.counters);
        *slot = Some(stage);
    }

    /// Settle the active row as completed and stop the worker.
    pub fn finish(mut self) {
        self.settle();
    }

    fn settle(&mut self) {
        let stage = self
            .stage
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        if let Some(stage) = stage {
            stage.complete(&self.counters);
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.stop.send(());
            let _ = worker.handle.join();
        }
    }
}

impl Drop for AcquisitionDisplay<'_> {
    fn drop(&mut self) {
        // An unfinished stage reports itself as failed when the caller drops the
        // display before settling it.
        if let Some(worker) = self.worker.take() {
            let _ = worker.stop.send(());
            let _ = worker.handle.join();
        }
        let stage = self
            .stage
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        if let Some(mut stage) = stage {
            // A stage that failed before its next tick still reports the bytes it
            // reached.
            stage.refresh(&self.counters);
        }
    }
}

impl Stage {
    fn refresh(&mut self, counters: &AcquisitionProgress) {
        let done = counters.snapshot().counter(self.counter);
        let advanced = done
            .saturating_sub(self.baseline)
            .saturating_sub(self.advanced);
        if advanced > 0 {
            self.phase.advance_bytes(advanced);
            self.advanced = self.advanced.saturating_add(advanced);
        }
        self.detail = format_progress(
            done.saturating_sub(self.baseline),
            self.total,
            Some(self.rate.bytes_per_second(done)),
        );
        self.phase.detail(self.detail.clone());
    }

    fn complete(mut self, counters: &AcquisitionProgress) {
        // A stage can finish between two ticks, so settle from the counters
        // rather than from the last sampled detail. The phase row renders its
        // own elapsed time; the detail carries only the counters.
        self.refresh(counters);
        self.phase.finish(self.detail);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn download_planning_reports_shared_counts_without_a_transfer_total() {
        let progress = AcquisitionProgress::new();
        progress.record_network_bytes(18 << 20);
        progress.record_network_total(Some(18 << 20));
        progress.begin_download_planning(8);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let observer = progress.clone();
                scope.spawn(move || observer.record_planned_file());
            }
        });
        let snapshot = progress.snapshot();
        assert_eq!(snapshot.phase, AcquisitionPhase::PreparingDownload);
        assert_eq!(snapshot.phase.label(), "Preparing download");
        assert_eq!(snapshot.planning_completed, 8);
        assert_eq!(snapshot.planning_total, Some(8));
        assert_eq!(snapshot.network_total, None);
        assert_eq!(snapshot.network_bytes, 18 << 20);
        progress.begin_download_planning(0);
        assert_eq!(progress.snapshot().planning_completed, 0);
        assert_eq!(progress.snapshot().planning_total, Some(0));
    }

    #[test]
    fn cancellation_is_shared_across_clones_without_discarding_progress() {
        let progress = AcquisitionProgress::new();
        let worker = progress.clone();
        worker.record_network_bytes(1024);
        assert!(!worker.is_cancelled());
        progress.request_cancel();
        std::thread::spawn(move || assert!(worker.is_cancelled())).join().unwrap();
        assert!(progress.is_cancelled());
        assert_eq!(progress.snapshot().network_bytes, 1024);
        assert!(!AcquisitionProgress::new().is_cancelled());
    }

    #[test]
    fn byte_rate_and_eta_text_is_unit_scaled_and_label_free() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1023), "1023 B");
        assert_eq!(format_bytes(1536), "1.5 KiB");
        assert_eq!(format_bytes(3 << 20), "3.0 MiB");
        assert_eq!(format_bytes(3 << 30), "3.0 GiB");
        assert_eq!(format_rate(0.0), "0 B/s");
        assert_eq!(format_rate(f64::NAN), "0 B/s");
        assert_eq!(format_rate(1.5 * (1u64 << 20) as f64), "1.5 MiB/s");
        assert_eq!(format_eta(Duration::from_secs(9)), "9s");
        assert_eq!(format_eta(Duration::from_secs(615)), "10m15s");
        assert_eq!(format_eta(Duration::from_secs(3725)), "1h02m");
        assert!(!format_progress(1, None, None).contains('\x1b'));
        assert!(!format_progress(1, None, None).contains('\n'));
    }

    #[test]
    fn progress_body_adds_eta_only_when_a_total_and_rate_remain() {
        let rate = 2.0 * (1u64 << 20) as f64;
        assert_eq!(format_progress(0, None, None), "0 B");
        assert_eq!(
            format_progress(4 << 20, Some(8 << 20), Some(rate)),
            "4.0 / 8.0 MiB  2.0 MiB/s  ETA 2s"
        );
        assert_eq!(
            format_progress(4 << 20, None, Some(rate)),
            "4.0 MiB  2.0 MiB/s"
        );
        assert_eq!(
            format_progress(8 << 20, Some(8 << 20), Some(rate)),
            "8.0 / 8.0 MiB  2.0 MiB/s"
        );
        assert_eq!(
            format_progress(9 << 20, Some(8 << 20), Some(rate)),
            "9.0 MiB  2.0 MiB/s"
        );
    }

    #[test]
    fn counters_select_the_metered_bytes() {
        let progress = AcquisitionProgress::new();
        progress.record_network_bytes(3);
        progress.record_local_source_bytes(5);
        progress.record_materialized_copy_bytes(7);
        progress.record_materialized_fill_bytes(11);
        progress.record_verification_bytes(13);
        progress.record_network_total(Some(21));
        let snapshot = progress.snapshot();
        assert_eq!(snapshot.network_total, Some(21));
        assert_eq!(snapshot.counter(Counter::Network), 3);
        assert_eq!(snapshot.counter(Counter::LocalSource), 5);
        assert_eq!(snapshot.counter(Counter::Materialized), 18);
        assert_eq!(snapshot.counter(Counter::Verification), 13);
    }

    #[test]
    fn declared_network_total_is_replaced_or_cleared() {
        let progress = AcquisitionProgress::new();
        assert_eq!(progress.snapshot().network_total, None);
        progress.record_network_bytes(4);
        progress.record_network_total(Some(9));
        assert_eq!(progress.snapshot().network_total, Some(9));
        progress.record_network_total(Some(12));
        assert_eq!(progress.snapshot().network_total, Some(12));
        progress.record_network_total(None);
        assert_eq!(progress.snapshot().network_total, None);
        assert_eq!(progress.snapshot().network_bytes, 4);
    }

    #[test]
    fn rate_reports_bytes_since_its_baseline() {
        let mut rate = Rate::start(1000);
        thread::sleep(Duration::from_millis(20));
        assert!(rate.bytes_per_second(500) == 0.0);
        rate.reset(2000);
        thread::sleep(Duration::from_millis(20));
        assert!(rate.bytes_per_second(2000) == 0.0);
    }

    #[test]
    fn a_stage_that_ends_between_ticks_settles_from_the_counters() {
        let startup = StartupProgress::silent();
        let counters = AcquisitionProgress::new();
        let display = AcquisitionDisplay::new(&startup, &counters);
        display.stage("Download core", Counter::Network, Some(1 << 20));
        counters.record_network_bytes(3 << 20);
        let detail = display.settle_detail().expect("a settled row");
        assert!(detail.starts_with("3.0 MiB  "), "{detail}");
        assert!(!detail.contains("ETA"), "{detail}");
    }

    #[test]
    fn concurrent_records_are_visible_and_monotonic() {
        let progress = AcquisitionProgress::new();
        let stop = Arc::new(AtomicBool::new(false));
        let workers: Vec<_> = (0..4)
            .map(|_| {
                let progress = progress.clone();
                let stop = stop.clone();
                thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        progress.record_verification_bytes(1);
                    }
                })
            })
            .collect();
        let mut previous = 0;
        for _ in 0..100 {
            let observed = progress.snapshot().verification_bytes;
            assert!(observed >= previous);
            previous = observed;
        }
        stop.store(true, Ordering::Relaxed);
        for worker in workers {
            worker.join().unwrap();
        }
        assert!(progress.snapshot().verification_bytes >= previous);
    }

    #[test]
    fn stages_follow_counters_and_settle_on_finish() {
        let startup = StartupProgress::silent();
        let counters = AcquisitionProgress::new();
        let display = AcquisitionDisplay::new(&startup, &counters);
        assert!(display.worker.is_none());
        counters.record_network_bytes(1 << 20);
        display.stage("Download core", Counter::Network, Some(4 << 20));
        {
            let stage = display.selected();
            assert_eq!(stage.0, "0.0 / 4.0 MiB");
            assert_eq!(stage.1, 0);
        }
        counters.record_network_bytes(3 << 20);
        {
            let mut slot = display.stage.lock().unwrap();
            slot.as_mut().unwrap().refresh(&counters);
            let stage = slot.as_ref().unwrap();
            assert_eq!(stage.advanced, 3 << 20);
            assert!(
                stage.detail.starts_with("3.0 / 4.0 MiB  "),
                "{}",
                stage.detail
            );
        }
        display.stage("Prepare core", Counter::Materialized, None);
        counters.record_materialized_fill_bytes(1 << 19);
        {
            let mut slot = display.stage.lock().unwrap();
            slot.as_mut().unwrap().refresh(&counters);
            let stage = slot.as_ref().unwrap();
            assert_eq!(stage.advanced, 1 << 19);
            assert!(stage.detail.starts_with("512.0 KiB  "), "{}", stage.detail);
        }
        display.finish();
    }

    impl<'a> AcquisitionDisplay<'a> {
        fn settle_detail(&self) -> Option<String> {
            let mut stage = self.stage.lock().unwrap().take()?;
            stage.refresh(&self.counters);
            let detail = stage.detail.clone();
            stage.phase.finish(stage.detail);
            Some(detail)
        }

        fn selected(&self) -> (String, u64) {
            let slot = self.stage.lock().unwrap();
            let stage = slot.as_ref().expect("a selected stage");
            (stage.detail.clone(), stage.advanced)
        }
    }
}
