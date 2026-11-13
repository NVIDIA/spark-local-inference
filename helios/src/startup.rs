/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Animated startup banner and loading rows with measured progress and elapsed time.

mod banner;

use std::{
    io::{IsTerminal, Write},
    sync::{Arc, Mutex, mpsc},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

/// Displays one loading stage at a time on stderr; redirected output is plain text.
pub struct StartupProgress {
    enabled: bool,
    terminal: bool,
    output: Arc<Mutex<Output>>,
    animation: Option<(mpsc::Sender<AnimationStop>, JoinHandle<()>)>,
    observation: Option<StartupObservation>,
}

/// Stable identifiers for core loading stages; labels are presentation text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StartupStage {
    Other, VerifyStoredData, PrepareModel, InitializeGpu, PlanCache, AllocateCache,
    AllocateModelMemory, LoadWeights, PrepareKernels, PrepareExecution,
    PrepareEmbeddingTables, PrepareHostPrefixCache, LoadTokenizer, StartServer, ShutdownServer,
}

impl StartupStage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Other => "other",
            Self::VerifyStoredData => "verify_stored_data",
            Self::PrepareModel => "prepare_model",
            Self::InitializeGpu => "initialize_gpu",
            Self::PlanCache => "plan_cache",
            Self::AllocateCache => "allocate_cache",
            Self::AllocateModelMemory => "allocate_model_memory",
            Self::LoadWeights => "load_weights",
            Self::PrepareKernels => "prepare_kernels",
            Self::PrepareExecution => "prepare_execution",
            Self::PrepareEmbeddingTables => "prepare_embedding_tables",
            Self::PrepareHostPrefixCache => "prepare_host_prefix_cache",
            Self::LoadTokenizer => "load_tokenizer",
            Self::StartServer => "start_server",
            Self::ShutdownServer => "shutdown_server",
        }
    }

    fn from_label(label: &str) -> Self {
        match label {
            "Verify stored data" => Self::VerifyStoredData,
            "Prepare model" => Self::PrepareModel,
            "Initialize GPU" | "Initialize GPUs" => Self::InitializeGpu,
            "Plan cache" => Self::PlanCache,
            "Allocate cache" => Self::AllocateCache,
            "Allocate model memory" => Self::AllocateModelMemory,
            "Load weights" => Self::LoadWeights,
            "Prepare kernels" => Self::PrepareKernels,
            "Prepare execution" => Self::PrepareExecution,
            "Prepare embedding tables" => Self::PrepareEmbeddingTables,
            "Prepare host prefix cache" => Self::PrepareHostPrefixCache,
            "Load tokenizer" => Self::LoadTokenizer,
            "Start server" => Self::StartServer,
            "Shut down server" => Self::ShutdownServer,
            _ => Self::Other,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StartupStatus { Running, Complete, Failed }

/// Latest startup update, retaining at most 512 characters in each text field.
#[derive(Clone, Debug)]
pub struct StartupSnapshot {
    pub sequence: u64,
    pub stage: StartupStage,
    pub status: StartupStatus,
    pub label: String,
    pub detail: String,
    pub completed: u64,
    pub total: Option<u64>,
    pub elapsed: Duration,
}

/// A bounded observation independent of terminal rendering or event consumers.
#[derive(Clone, Default)]
pub struct StartupObservation(Arc<Mutex<Option<StartupSnapshot>>>);

impl StartupObservation {
    pub fn snapshot(&self) -> Option<StartupSnapshot> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn publish(&self, state: &State, status: StartupStatus) {
        let mut latest = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let sequence = latest.as_ref().map_or(1, |last| last.sequence.saturating_add(1));
        *latest = Some(StartupSnapshot {
            sequence, stage: state.stage, status,
            label: state.label.chars().take(512).collect(),
            detail: state.detail.chars().take(512).collect(),
            completed: state.completed, total: state.total, elapsed: state.started.elapsed(),
        });
    }
}

enum AnimationStop {
    Finish,
    Pause,
}

impl StartupProgress {
    pub fn stderr() -> Self {
        let mut progress = Self::stderr_rows();
        progress.animate_banner();
        progress
    }

    /// Displays progress rows without the Flare startup banner.
    pub fn stderr_rows() -> Self {
        Self {
            enabled: true,
            terminal: std::io::stderr().is_terminal(),
            output: Arc::new(Mutex::new(Output::default())),
            animation: None,
            observation: None,
        }
    }

    pub fn silent() -> Self {
        Self {
            enabled: false,
            terminal: false,
            output: Arc::new(Mutex::new(Output::default())),
            animation: None,
            observation: None,
        }
    }

    /// Observes loading without writing stderr or starting terminal workers.
    pub fn observed() -> (Self, StartupObservation) {
        let observation = StartupObservation::default();
        let mut progress = Self::silent();
        progress.observation = Some(observation.clone());
        (progress, observation)
    }

    /// Whether phases write anything at all.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Whether stderr is a terminal that can redraw in place.
    pub fn is_terminal(&self) -> bool {
        self.terminal
    }

    fn animate_banner(&mut self) {
        if !self.terminal {
            return;
        }
        let (width, height) = terminal_size();
        if cfg!(windows) || banner::frame(0.0, width).lines().count() + 2 >= height {
            let mut out = std::io::stderr().lock();
            let _ = out.write_all(banner::frame(1.0, width).as_bytes());
            let _ = out.flush();
            return;
        }
        let started = Instant::now();
        let mut output = self.output.lock().unwrap();
        output.started = Some(started);
        output.redraw(0.0);
        drop(output);
        self.spawn_animation(started);
    }

    fn spawn_animation(&mut self, started: Instant) {
        let output = self.output.clone();
        let (stop, receiver) = mpsc::channel();
        let worker = thread::spawn(move || {
            loop {
                let stopped = match receiver.recv_timeout(
                    banner::DURATION
                        .saturating_sub(started.elapsed())
                        .min(Duration::from_micros(62_500)),
                ) {
                    Ok(AnimationStop::Pause) => return,
                    Ok(AnimationStop::Finish) | Err(mpsc::RecvTimeoutError::Disconnected) => true,
                    Err(mpsc::RecvTimeoutError::Timeout) => false,
                };
                let mut output = output.lock().unwrap();
                let progress = output.progress();
                if stopped || progress >= 1.0 {
                    output.redraw(1.0);
                    output.started = None;
                    output.completed.clear();
                    output.active = None;
                    output.drawn_lines = 0;
                    break;
                }
                output.redraw(progress);
            }
        });
        self.animation = Some((stop, worker));
    }

    /// Joins the banner worker for an operation, then resumes its original timeline.
    /// Callers must finish any loading phases first to stop their progress workers.
    pub fn with_animation_paused<T>(&mut self, operation: impl FnOnce() -> T) -> T {
        if let Some((stop, worker)) = self.animation.take() {
            let _ = stop.send(AnimationStop::Pause);
            let _ = worker.join();
        }
        let result = operation();
        let started = self.output.lock().unwrap().started;
        if let Some(started) = started {
            self.spawn_animation(started);
        }
        result
    }

    /// Settles the header before another terminal UI takes ownership of stderr.
    pub fn finish_animation(&mut self) {
        if let Some((stop, worker)) = self.animation.take() {
            let _ = stop.send(AnimationStop::Finish);
            let _ = worker.join();
        }
    }

    pub fn phase(&self, label: &str, total: Option<u64>) -> Phase {
        self.phase_named(StartupStage::from_label(label), label, total)
    }

    pub fn phase_named(&self, stage: StartupStage, label: &str, total: Option<u64>) -> Phase {
        let state = Arc::new(Mutex::new(State {
            stage,
            label: label.to_owned(),
            detail: String::new(),
            started: Instant::now(),
            elapsed: None,
            completed: 0,
            total,
        }));
        let mut phase = Phase {
            state: state.clone(),
            enabled: self.enabled,
            terminal: self.terminal,
            output: self.output.clone(),
            finished: false,
            animation: None,
            observation: self.observation.clone(),
        };
        phase.publish(StartupStatus::Running);
        if self.enabled {
            self.output.lock().unwrap().write(
                &state.lock().unwrap(),
                self.terminal,
                Status::Running,
            );
        }
        if self.enabled {
            let (stop, receiver) = mpsc::channel();
            let output = self.output.clone();
            let terminal = self.terminal;
            // A terminal redraws smoothly. Redirected output (an agent's retained log, or
            // `flare orbit attach`) cannot redraw, so it reports a progress line periodically
            // instead of staying on a bare PENDING row for minutes.
            let interval = if terminal {
                Duration::from_millis(125)
            } else {
                Duration::from_secs(10)
            };
            let worker = thread::spawn(move || {
                while matches!(
                    receiver.recv_timeout(interval),
                    Err(mpsc::RecvTimeoutError::Timeout)
                ) {
                    output
                        .lock()
                        .unwrap()
                        .write(&state.lock().unwrap(), terminal, Status::Running);
                }
            });
            phase.animation = Some((stop, worker));
        }
        phase
    }
}

impl Drop for StartupProgress {
    fn drop(&mut self) {
        self.finish_animation();
    }
}

/// A stage prints FAILED if dropped without an explicit successful finish.
pub struct Phase {
    state: Arc<Mutex<State>>,
    enabled: bool,
    terminal: bool,
    output: Arc<Mutex<Output>>,
    finished: bool,
    animation: Option<(mpsc::Sender<()>, JoinHandle<()>)>,
    observation: Option<StartupObservation>,
}

impl Phase {
    fn publish(&self, status: StartupStatus) {
        if let Some(observation) = &self.observation {
            observation.publish(&self.state.lock().unwrap(), status);
        }
    }

    pub fn advance(&self, count: u64) {
        let mut state = self.state.lock().unwrap();
        state.completed = state.completed.saturating_add(count);
        drop(state);
        self.publish(StartupStatus::Running);
    }

    pub fn advance_bytes(&self, count: u64) {
        let mut state = self.state.lock().unwrap();
        state.completed = state.completed.saturating_add(count);
        let completed = state.completed as f64 / (1u64 << 30) as f64;
        state.detail = match state.total {
            Some(total) => format!("{completed:.1} / {:.1} GiB", total as f64 / (1u64 << 30) as f64),
            None => format!("{completed:.1} GiB"),
        };
        drop(state);
        self.publish(StartupStatus::Running);
    }

    pub fn detail(&self, detail: impl Into<String>) {
        self.state.lock().unwrap().detail = detail.into();
        self.publish(StartupStatus::Running);
    }

    pub fn finish(mut self, detail: impl Into<String>) {
        self.detail(detail);
        self.finished = true;
    }
}

impl Drop for Phase {
    fn drop(&mut self) {
        self.publish(if self.finished { StartupStatus::Complete } else { StartupStatus::Failed });
        if let Some((stop, worker)) = self.animation.take() {
            let _ = stop.send(());
            let _ = worker.join();
        }
        if self.enabled {
            self.output.lock().unwrap().write(
                &self.state.lock().unwrap(),
                self.terminal,
                if self.finished {
                    Status::Ok
                } else {
                    Status::Failed
                },
            );
        }
    }
}

#[derive(Clone)]
struct State {
    stage: StartupStage,
    label: String,
    detail: String,
    started: Instant,
    elapsed: Option<Duration>,
    completed: u64,
    total: Option<u64>,
}

#[derive(Clone, Copy)]
enum Status {
    Running,
    Ok,
    Failed,
}

impl State {
    fn bar(&self, elapsed: f64) -> String {
        use std::fmt::Write;

        let mut bar = String::new();
        if let Some(total) = self.total {
            let halves = if total == 0 {
                20
            } else {
                (20.0 * self.completed.min(total) as f64 / total as f64) as usize
            };
            let full = halves / 2;
            let half = halves % 2;
            let color = if self.completed >= total { 32 } else { 36 };
            let _ = write!(bar, "\x1b[{color}m{}", "━".repeat(full));
            if half != 0 {
                bar.push('╸');
            }
            let mut remaining = 10 - full - half;
            if remaining != 0 {
                bar.push_str("\x1b[38;2;58;58;58m");
                if half == 0 && full != 0 {
                    bar.push('╺');
                    remaining -= 1;
                }
                bar.push_str(&"━".repeat(remaining));
            }
        } else {
            let offset = (20 - (elapsed * 15.0) as usize % 20) % 20;
            for index in offset..offset + 10 {
                let fade = 0.5 + (index as f64 / 20.0 * std::f64::consts::TAU).cos() / 2.0;
                let red = (58.0 * fade) as u8;
                let cyan = (128.0 * (1.0 - fade) + 58.0 * fade) as u8;
                let _ = write!(bar, "\x1b[38;2;{red};{cyan};{cyan}m━");
            }
        }
        bar
    }

    fn render(&self, terminal: bool, status: Status) -> String {
        use std::fmt::Write;
        let elapsed = self
            .elapsed
            .unwrap_or_else(|| self.started.elapsed())
            .as_secs_f64();
        let status_text = match status {
            Status::Ok => "    OK    ".to_owned(),
            Status::Failed => "  FAILED  ".to_owned(),
            Status::Running if terminal => self.bar(elapsed),
            Status::Running => " PENDING  ".to_owned(),
        };
        let label = format!("] {}.", self.label);
        let detail = if self.detail.is_empty() {
            String::new()
        } else {
            format!("  {}", self.detail)
        };
        let timing = format!("  {elapsed:.1}s");
        let width = if terminal {
            terminal_width()
        } else {
            usize::MAX
        };
        let available = width.saturating_sub(11 + timing.chars().count());
        let body = truncate(&(label.clone() + &detail), available);
        let label_end = body
            .char_indices()
            .nth(label.chars().count())
            .map_or(body.len(), |(i, _)| i);
        let (label, detail) = body.split_at(label_end);
        let padding = if terminal {
            available.saturating_sub(body.chars().count())
        } else {
            0
        };
        let mut out = String::new();
        if terminal {
            let color = match status {
                Status::Running => "36",
                Status::Ok => "1;32",
                Status::Failed => "1;31",
            };
            let _ = write!(
                out,
                "\r\x1b[2K[\x1b[{color}m{status_text}\x1b[0m{label}\x1b[2m{detail}{:padding$}{timing}\x1b[0m",
                ""
            );
        } else {
            let _ = write!(out, "[{status_text}{label}{detail}{timing}");
        }
        out
    }
}

#[derive(Default)]
struct Output {
    started: Option<Instant>,
    completed: Vec<(State, Status)>,
    active: Option<State>,
    drawn_lines: usize,
}

impl Output {
    fn progress(&self) -> f64 {
        self.started.map_or(1.0, |started| {
            (started.elapsed().as_secs_f64() / banner::DURATION.as_secs_f64()).min(1.0)
        })
    }

    fn write(&mut self, state: &State, terminal: bool, status: Status) {
        if self.started.is_some() {
            if matches!(status, Status::Running) {
                self.active = Some(state.clone());
            } else {
                let mut state = state.clone();
                state.elapsed = Some(state.started.elapsed());
                self.completed.push((state, status));
                self.active = None;
            }
            self.redraw(self.progress());
        } else {
            let mut out = std::io::stderr().lock();
            let _ = out.write_all(state.render(terminal, status).as_bytes());
            if !terminal || !matches!(status, Status::Running) {
                let _ = writeln!(out);
            }
            let _ = out.flush();
        }
    }

    fn redraw(&mut self, progress: f64) {
        if self.started.is_none() {
            return;
        }
        let (width, height) = terminal_size();
        let mut content = banner::frame(progress, width);
        for (state, status) in &self.completed {
            content.push_str(&state.render(true, *status));
            content.push('\n');
        }
        if let Some(state) = &self.active {
            content.push_str(&state.render(true, Status::Running));
        }
        let lines: Vec<_> = content.split('\n').collect();
        let visible = &lines[lines.len().saturating_sub(height.saturating_sub(1).max(1))..];
        let mut out = std::io::stderr().lock();
        if self.drawn_lines > 1 {
            let _ = write!(
                out,
                "\x1b[{}A",
                (self.drawn_lines - 1).min(height.saturating_sub(1))
            );
        }
        let _ = write!(out, "\r\x1b[J{}", visible.join("\n"));
        let _ = out.flush();
        self.drawn_lines = visible.len();
    }
}

fn truncate(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        return text.to_owned();
    }
    if width == 0 {
        return String::new();
    }
    text.chars().take(width - 1).collect::<String>() + "…"
}

fn terminal_width() -> usize {
    terminal_size().0
}

fn terminal_size() -> (usize, usize) {
    crate::os::terminal_size()
}

#[cfg(test)]
mod observation_tests {
    use super::*;

    #[test]
    fn observed_stages_publish_counts_and_terminal_status_without_terminal_workers() {
        let (progress, observation) = StartupProgress::observed();
        assert!(!progress.is_enabled());
        assert!(!progress.is_terminal());
        let phase = progress.phase("Load weights", Some(10));
        assert!(phase.animation.is_none());
        let first = observation.snapshot().unwrap();
        assert_eq!(first.stage, StartupStage::LoadWeights);
        assert_eq!(first.status, StartupStatus::Running);
        phase.advance_bytes(4);
        assert_eq!(observation.snapshot().unwrap().completed, 4);
        phase.finish("loaded");
        let last = observation.snapshot().unwrap();
        assert_eq!(last.total, Some(10));
        assert_eq!(last.status, StartupStatus::Complete);
        assert!(last.sequence > first.sequence);
        assert_eq!(last.detail, "loaded");
        drop(progress.phase_named(StartupStage::StartServer, "Listen", None));
        assert_eq!(observation.snapshot().unwrap().status, StartupStatus::Failed);
    }

    #[test]
    fn observer_retains_one_bounded_update_and_survives_producer_drop() {
        let (progress, observation) = StartupProgress::observed();
        let phase = progress.phase_named(StartupStage::Other, &"λ".repeat(2000), None);
        for _ in 0..100 { phase.advance(1); }
        phase.finish("δ".repeat(2000));
        drop(progress);
        let snapshot = observation.snapshot().unwrap();
        assert_eq!(snapshot.label.chars().count(), 512);
        assert_eq!(snapshot.detail.chars().count(), 512);
        assert_eq!(snapshot.completed, 100);
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn pausing_banner_preserves_playback_and_loading_rows() {
        let mut progress = StartupProgress::silent();
        progress.enabled = true;
        progress.terminal = true;
        progress.animate_banner();
        let started = progress.output.lock().unwrap().started.unwrap();
        progress.phase("Verify core", None).finish("signatures");
        let output = progress.output.clone();
        let result = progress.with_animation_paused(|| {
            let output = output.lock().unwrap();
            assert_eq!(output.started, Some(started));
            assert!(output.progress() < 1.0);
            assert_eq!(output.completed.len(), 1);
            Err::<(), _>("configuration failed")
        });
        assert_eq!(result, Err("configuration failed"));
        assert_eq!(progress.output.lock().unwrap().started, Some(started));

        let (stop, worker) = progress.animation.take().unwrap();
        worker.join().unwrap();
        drop(stop);
        assert!(started.elapsed() >= banner::DURATION);
        assert!(progress.output.lock().unwrap().started.is_none());
        progress.with_animation_paused(|| ());
        assert!(progress.animation.is_none());
    }

    #[test]
    fn pausing_without_a_banner_does_not_start_animation() {
        let mut progress = StartupProgress::silent();
        assert_eq!(progress.with_animation_paused(|| 42), 42);
        assert!(progress.animation.is_none());
        assert!(progress.output.lock().unwrap().started.is_none());
    }
}
