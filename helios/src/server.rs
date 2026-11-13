/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Native HTTP admission, bounded generation streams and direct graph scheduling.

mod config;
mod dashboard;
mod http;
mod inference;
mod listener;
mod latency;
pub use latency::FirstResponseSnapshot;
mod metrics;
mod models;
mod observation;
pub(crate) mod output;
mod logprobs;
mod protocol;
mod request;
pub use request::{PreparedRequest, prepare as prepare_request};
mod runtime;
mod stops;
#[cfg(feature = "runtime-core")]
mod session;
#[cfg(feature = "runtime-core")]
pub use session::{LoadedSession, ServingSession};
pub(crate) mod token_spans;
pub use config::Bootstrap;
pub use dashboard::view;
pub use metrics::Snapshot;
pub use runtime::{Generation, GenerationEvent, ModelRuntime, Prompt};

pub fn restore_terminal_for_exit() {
    dashboard::restore_for_exit();
}

use crate::{
    AdmissionError, Error, ExecutionError, Failure, Result, Shutdown,
    execution::ExecutionManifest,
    frontend::{Frontend, images::ImageProcessor},
    model::ModelConfig,
};
use axum::{
    Extension, Json, Router,
    extract::{DefaultBodyLimit, Request, State},
    http::{HeaderMap, StatusCode, Uri, header},
    middleware::{self, Next},
    response::{IntoResponse, Response, Sse, sse::KeepAlive},
    routing::{get, post},
};
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    convert::Infallible,
    net::SocketAddr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

pub struct ServerConfig {
    /// Write operational diagnostics to stderr for command-line callers.
    pub log_to_stderr: bool,
    pub bind: SocketAddr,
    pub queue_capacity: usize,
    /// Minimum event capacity; each response also reserves its complete output budget.
    pub event_capacity: usize,
    pub frontend_workers: usize,
    pub max_body_bytes: usize,
    /// Serving key; Messages also accepts `x-api-key`. `/health` and `/metrics`
    /// are unauthenticated.
    pub api_key: Option<String>,
    /// Names requests may use for the model in addition to its identifier.
    pub served_names: Vec<String>,
    /// Default reasoning-token cap for conversational requests without a budget.
    pub thinking_token_budget: Option<u32>,
    /// Setting that raises the per-request context, appended to context rejections.
    pub context_hint: String,
    /// Setting that raises request or page capacity, appended to capacity rejections.
    pub capacity_hint: String,
    pub scheduling_policy: crate::scheduler::PolicyConfig,
}
impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            log_to_stderr: true,
            bind: "127.0.0.1:8000".parse().unwrap(),
            queue_capacity: 16,
            event_capacity: 128,
            frontend_workers: 2,
            max_body_bytes: 32 * 1024 * 1024,
            api_key: None,
            served_names: Vec::new(),
            thinking_token_budget: None,
            context_hint: String::new(),
            capacity_hint: String::new(),
            scheduling_policy: crate::scheduler::PolicyConfig::default(),
        }
    }
}
const GENERATED_WINDOW: Duration = Duration::from_secs(60);
const GENERATED_SAMPLE_PERIOD: Duration = Duration::from_secs(5);
struct HttpState {
    first_response: Arc<latency::FirstResponseSamples>,
    log_to_stderr: bool,
    model: Arc<ModelRuntime>,
    frontend: Arc<Frontend>,
    images: Option<Arc<ImageProcessor>>,
    grammar: Arc<crate::grammar::GrammarCompiler>,
    frontend_slots: Arc<Semaphore>,
    ingress_slots: Arc<Semaphore>,
    max_body_bytes: usize,
    frontend_capacity: usize,
    ingress_capacity: usize,
    /// Keeps request and tool-call IDs unique across restarts and hosts.
    run: u64,
    ids: AtomicU64,
    api_key: Option<String>,
    thinking_token_budget: Option<u32>,
    context_hint: String,
    capacity_hint: String,
    started: Instant,
    generated: Mutex<VecDeque<(Instant, u64)>>,
    accepting: Mutex<bool>,
}
/// Distinguishes one server start from another; the standard hasher's keys are random.
fn run_id() -> u64 {
    use std::hash::BuildHasher;
    std::collections::hash_map::RandomState::new().hash_one(0u8)
}

impl HttpState {
    /// Records the generated-token counter, retaining one sample at or before the window start.
    fn sample_generated(&self) {
        let now = Instant::now();
        let tokens = self.model.metrics().snapshot(false).generated_tokens;
        let mut samples = self.generated.lock().unwrap_or_else(|e| e.into_inner());
        samples.push_back((now, tokens));
        while samples.len() > 2 && now.duration_since(samples[1].0) >= GENERATED_WINDOW {
            samples.pop_front();
        }
    }

    fn generated_last_minute(&self) -> u64 {
        let samples = self.generated.lock().unwrap_or_else(|e| e.into_inner());
        match (samples.front(), samples.back()) {
            (Some(first), Some(last)) => last.1.saturating_sub(first.1),
            _ => 0,
        }
    }
}

pub struct Server {
    address: SocketAddr,
    model: Arc<ModelRuntime>,
    control: Mutex<HttpControl>,
    shutdown: Mutex<Option<Shutdown>>,
    dashboard: Mutex<Option<dashboard::Dashboard>>,
    state: Arc<HttpState>,
    http_alive: Arc<AtomicBool>,
}

/// Serving state and bounded HTTP metrics sampled without querying the GPU.
#[derive(Clone, Debug)]
pub struct ServerSnapshot {
    pub first_response: FirstResponseSnapshot,
    pub session_id: u64,
    pub sampled_at_ns: u64,
    pub uptime: Duration,
    pub healthy: bool,
    pub ready: bool,
    pub accepting: bool,
    pub ingress_active: usize,
    pub ingress_capacity: usize,
    pub frontend_active: usize,
    pub frontend_capacity: usize,
    pub model: Snapshot,
}
struct HttpControl {
    stop: Option<oneshot::Sender<()>>,
    thread: Option<JoinHandle<Result<()>>>,
}

/// Readiness must disappear before HTTP runtime cleanup can wait on other work.
struct HttpLifetime(Arc<AtomicBool>);
impl Drop for HttpLifetime {
    fn drop(&mut self) { self.0.store(false, Ordering::Release); }
}

#[derive(Debug)]
pub struct StartFailure {
    pub message: Error,
    pub drained: bool,
}
impl From<Error> for StartFailure {
    fn from(message: Error) -> Self {
        Self {
            message,
            drained: true,
        }
    }
}
impl From<&str> for StartFailure {
    fn from(message: &str) -> Self {
        Error::from(message).into()
    }
}
impl Server {
    /// # Safety
    /// The caller owns stable, exclusive prepared resources until close confirms
    /// both CUDA streams drained. No Python code runs on either native worker.
    pub unsafe fn start(
        manifest: ExecutionManifest,
        model: ModelConfig,
        frontend: Frontend,
        images: Option<ImageProcessor>,
        config: ServerConfig,
        peers: Option<crate::execution::RankPeers>,
    ) -> std::result::Result<Self, StartFailure> {
        if model.contract.vision.enabled() != images.is_some() {
            return Err("image processor must match prepared vision capabilities".into());
        }
        if !(1..=64).contains(&config.frontend_workers)
            || config.max_body_bytes == 0
            || config.max_body_bytes > 256 * 1024 * 1024
        {
            return Err("invalid HTTP admission limits".into());
        }
        let delimiters =
            crate::model::reasoning_delimiter_tokens(&frontend, &model.contract.frontend)?;
        config.validate_thinking_budget(&frontend, &model)?;
        let grammar = Arc::new(crate::grammar::GrammarCompiler::new(
            frontend.tokenizer(),
            model.contract.vocabulary,
            &model.eos_token_ids,
            &delimiters,
        )?);
        let ingress_capacity = config
            .queue_capacity
            .checked_add(model.limits.max_active)
            .ok_or("HTTP request capacity overflows")?;
        let frontend = Arc::new(frontend);
        // SAFETY: The caller grants exclusive graph/storage access and retains
        // their owners until drain. This server propagates that obligation to the
        // model worker and reports uncertain drain through StartFailure or close.
        let model = Arc::new(unsafe {
            ModelRuntime::start_retaining_drain(
                manifest,
                model,
                config.queue_capacity,
                config.event_capacity,
                frontend.clone(),
                peers,
                config.scheduling_policy,
                config.log_to_stderr,
            )?
        });
        let state = Arc::new(HttpState {
            first_response: Arc::new(latency::FirstResponseSamples::default()),
            log_to_stderr: config.log_to_stderr,
            model: model.clone(),
            frontend,
            images: images.map(Arc::new),
            grammar,
            frontend_slots: Arc::new(Semaphore::new(config.frontend_workers)),
            ingress_slots: Arc::new(Semaphore::new(ingress_capacity)),
            max_body_bytes: config.max_body_bytes,
            frontend_capacity: config.frontend_workers,
            ingress_capacity,
            run: run_id(),
            ids: AtomicU64::new(1),
            api_key: config.api_key.clone(),
            thinking_token_budget: config.thinking_token_budget,
            context_hint: config.context_hint.clone(),
            capacity_hint: config.capacity_hint.clone(),
            started: Instant::now(),
            generated: Mutex::new(VecDeque::new()),
            accepting: Mutex::new(true),
        });
        let server_state = state.clone();
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let runtime_model = model.clone();
        let http_alive = Arc::new(AtomicBool::new(false));
        let worker_alive = http_alive.clone();
        let thread = match thread::Builder::new()
            .name("flare-http".into())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                    .map_err(|error| Error::Execution(ExecutionError::Worker(Failure::caused_by(format!("cannot create HTTP runtime: {error}"), error))))?;
                runtime.block_on(async move {
                    let _lifetime = HttpLifetime(worker_alive.clone());
                    let shutdown_state = state.clone();
                    let sampler = state.clone();
                    tokio::spawn(async move {
                        let mut ticks = tokio::time::interval(GENERATED_SAMPLE_PERIOD);
                        loop {
                            ticks.tick().await;
                            sampler.sample_generated();
                        }
                    });
                    let router = router(state, &config.served_names, config.max_body_bytes);
                    let listener = match tokio::net::TcpListener::bind(config.bind).await {
                        Ok(listener) => listener,
                        Err(error) => {
                            let error = Error::Execution(ExecutionError::Worker(Failure::caused_by(format!("cannot bind HTTP listener: {error}"), error)));
                            let _ = ready_tx.send(Err(error.clone()));
                            return Err(error);
                        }
                    };
                    let address = listener.local_addr().map_err(|error| Error::Execution(ExecutionError::Worker(Failure::caused_by(error.to_string(), error))))?;
                    let listener = listener::HttpListener::with_diagnostics(listener, config.log_to_stderr);
                    worker_alive.store(true, Ordering::Release);
                    let _ = ready_tx.send(Ok(address));
                    let (drain_tx, drain_rx) = oneshot::channel();
                    let mut serving = tokio::spawn(async move {
                        axum::serve(listener, router)
                            .with_graceful_shutdown(async {
                                let _ = drain_rx.await;
                            })
                            .await
                    });
                    tokio::select! {
                        result=&mut serving=>{runtime_model.request_stop();return result.map_err(|error| Error::Execution(ExecutionError::Worker(Failure::caused_by(error.to_string(), error))))?.map_err(|error| Error::Execution(ExecutionError::Worker(Failure::caused_by(error.to_string(), error))));},
                        _=shutdown_rx=>{}
                    }
                    shutdown_state.frontend_slots.close();
                    shutdown_state.ingress_slots.close();
                    let _ = drain_tx.send(());
                    match tokio::time::timeout(Duration::from_secs(5), &mut serving).await {
                        Ok(result) => result
                            .map_err(|error| Error::Execution(ExecutionError::Worker(Failure::caused_by(error.to_string(), error))))?
                            .map_err(|error| Error::Execution(ExecutionError::Worker(Failure::caused_by(error.to_string(), error)))),
                        Err(_) => {
                            serving.abort();
                            let _ = serving.await;
                            Ok(())
                        }
                    }
                })
            }) {
            Ok(thread) => thread,
            Err(error) => {
                let shutdown = model.close();
                return Err(StartFailure {
                    message: Error::Execution(ExecutionError::Worker(Failure::caused_by(format!("cannot start HTTP thread: {error}"), error))),
                    drained: shutdown.drained,
                });
            }
        };
        match ready_rx.recv() {
            Ok(Ok(address)) => Ok(Self {
                address,
                model,
                control: Mutex::new(HttpControl {
                    stop: Some(shutdown_tx),
                    thread: Some(thread),
                }),
                shutdown: Mutex::new(None),
                dashboard: Mutex::new(None),
                state: server_state,
                http_alive,
            }),
            result => {
                let returned = thread.join().ok().and_then(Result::err);
                let shutdown = model.close();
                Err(StartFailure {
                    message: match result {
                        Ok(Err(error)) => error,
                        Err(disconnected) => readiness_failure(
                            "HTTP thread exited before readiness",
                            disconnected,
                            returned,
                        ),
                        Ok(Ok(_)) => unreachable!("ready success handled above"),
                    },
                    drained: shutdown.drained,
                })
            }
        }
    }
    pub fn address(&self) -> SocketAddr {
        self.address
    }
    /// Both the HTTP serving loop and the native model worker remain live.
    pub fn healthy(&self) -> bool {
        self.http_alive.load(Ordering::Acquire) && self.model.healthy()
    }
    pub fn snapshot(&self) -> ServerSnapshot {
        let accepting = *self.state.accepting.lock().unwrap_or_else(|e| e.into_inner());
        let healthy = self.healthy();
        ServerSnapshot {
            first_response: self.state.first_response.snapshot(),
            session_id: self.state.run,
            sampled_at_ns: observation::monotonic_ns(),
            uptime: self.state.started.elapsed(),
            healthy,
            ready: healthy && accepting && !self.state.ingress_slots.is_closed(),
            accepting,
            ingress_active: self.state.ingress_capacity.saturating_sub(self.state.ingress_slots.available_permits()),
            ingress_capacity: self.state.ingress_capacity,
            frontend_active: self.state.frontend_capacity.saturating_sub(self.state.frontend_slots.available_permits()),
            frontend_capacity: self.state.frontend_capacity,
            model: self.model.metrics().snapshot(self.model.config().has_learned_drafts()),
        }
    }
    /// Rejects subsequent HTTP generation requests while admitted responses finish.
    pub fn pause_admission(&self) {
        *self.state.accepting.lock().unwrap_or_else(|e| e.into_inner()) = false;
    }
    pub fn resume_admission(&self) {
        let mut accepting = self.state.accepting.lock().unwrap_or_else(|e| e.into_inner());
        if self.healthy() && !self.state.ingress_slots.is_closed() {
            *accepting = true;
        }
    }
    /// Waits for HTTP preparation and response owners after admission is paused.
    /// A timed-out wait leaves the server paused and permits resuming admission.
    pub fn wait_for_requests(&self, timeout: Duration) -> bool {
        let started = Instant::now();
        loop {
            {
                let accepting = self.state.accepting.lock().unwrap_or_else(|e| e.into_inner());
                if *accepting { return false; }
                if self.state.ingress_slots.available_permits() == self.state.ingress_capacity {
                    return true;
                }
            }
            if !self.healthy() || started.elapsed() >= timeout { return false; }
            thread::sleep(Duration::from_millis(10).min(timeout.saturating_sub(started.elapsed())));
        }
    }
    pub fn wait(&self, timeout: Duration) -> bool {
        self.model.wait(timeout) && self.healthy()
    }
    pub fn start_dashboard(&self) -> std::io::Result<bool> {
        let _control = self.control.lock().unwrap_or_else(|e| e.into_inner());
        let mut dashboard = self.dashboard.lock().unwrap_or_else(|e| e.into_inner());
        if dashboard.is_none() && self.healthy() {
            *dashboard = dashboard::Dashboard::start(
                self.model.metrics_handle(),
                format!("{} · {}", self.model.config().model_id, self.address),
                self.model.config().has_learned_drafts(),
            )?;
        }
        Ok(dashboard.is_some())
    }
    pub fn stop_dashboard(&self) {
        self.dashboard
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
    }
    pub fn close(&self) -> Shutdown {
        let mut control = self.control.lock().unwrap_or_else(|e| e.into_inner());
        self.stop_dashboard();
        if let Some(shutdown) = self
            .shutdown
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
        {
            return shutdown;
        }
        {
            let mut accepting = self.state.accepting.lock().unwrap_or_else(|e| e.into_inner());
            *accepting = false;
            // Closing retains admitted permits but permanently prevents resume,
            // before the HTTP worker can observe its shutdown signal.
            self.state.ingress_slots.close();
        }
        if let Some(stop) = control.stop.take() {
            let _ = stop.send(());
        }
        let mut http_error = None;
        if let Some(thread) = control.thread.take() {
            match thread.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    http_error = Some(error);
                }
                Err(_) => {
                    http_error = Some(Error::execution(
                        ExecutionError::Worker,
                        "HTTP thread panicked",
                    ));
                }
            }
        }
        let mut shutdown = self.model.close();
        if shutdown.error.is_none() { shutdown.error = http_error; }
        *self.shutdown.lock().unwrap_or_else(|e| e.into_inner()) = Some(shutdown.clone());
        shutdown
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.close();
    }
}

/// Routes generation and model discovery behind API-key authentication; health
/// and metrics expose no request content and stay unauthenticated.
fn router(state: Arc<HttpState>, served_names: &[String], max_body_bytes: usize) -> Router {
    Router::new()
        .merge(models::router(
            state.model.config(),
            served_names,
            state.thinking_token_budget,
        ))
        .route("/v1/chat/completions", post(http::generate).layer(middleware::from_fn_with_state(state.clone(), admit_http)))
        .route("/v1/completions", post(http::generate).layer(middleware::from_fn_with_state(state.clone(), admit_http)))
        .route("/v1/messages", post(http::generate).layer(middleware::from_fn_with_state(state.clone(), admit_http)))
        .route("/v1/messages/count_tokens", post(http::generate).layer(middleware::from_fn_with_state(state.clone(), admit_http)))
        .route("/v1/responses", post(http::generate).get(http::no_websocket).layer(middleware::from_fn_with_state(state.clone(), admit_http)))
        .route("/v1/systemone", post(protocol::decisions::generate).layer(middleware::from_fn_with_state(state.clone(), admit_http)))
        .route("/v1/decisions", post(protocol::decisions::generate).layer(middleware::from_fn_with_state(state.clone(), admit_http)))
        .route_layer(middleware::from_fn_with_state(state.clone(), authorize))
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/loads", get(loads))
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .with_state(state)
}

async fn health(State(state): State<Arc<HttpState>>) -> Response {
    if state.model.healthy() {
        Json(json!({"status":"ok"})).into_response()
    } else {
        error(
            StatusCode::SERVICE_UNAVAILABLE,
            "prepared model runtime is unavailable",
        )
    }
}

async fn loads(State(state): State<Arc<HttpState>>) -> Response {
    let metrics = state.model.metrics();
    let snapshot = metrics.snapshot(false);
    Json(load_report(
        &snapshot,
        metrics.cache_hit_rate(),
        state.generated_last_minute() as f64 / 60.0,
        state.model.config().limits.max_active,
    ))
    .into_response()
}

fn load_report(
    snapshot: &Snapshot,
    cache_hit_rate: f64,
    gen_throughput: f64,
    max_running: usize,
) -> Value {
    let used_pages = snapshot.total_pages.saturating_sub(snapshot.free_pages);
    let token_usage = if snapshot.total_pages == 0 {
        0.0
    } else {
        used_pages as f64 / snapshot.total_pages as f64
    };
    let utilization = if max_running == 0 {
        0.0
    } else {
        snapshot.active as f64 / max_running as f64
    };
    json!({
        "timestamp": "",
        "dp_rank_count": 1,
        "loads": [{
            "dp_rank": 0,
            "num_running_reqs": snapshot.active,
            "num_waiting_reqs": snapshot.pending,
            "num_waiting_uncached_tokens": 0,
            "num_total_reqs": snapshot.active + snapshot.pending,
            "num_used_tokens": used_pages.saturating_mul(snapshot.page_tokens),
            "max_total_num_tokens": snapshot.total_pages.saturating_mul(snapshot.page_tokens),
            "token_usage": token_usage,
            "gen_throughput": gen_throughput,
            "cache_hit_rate": cache_hit_rate,
            "utilization": utilization,
            "max_running_requests": max_running,
        }],
    })
}
async fn metrics(State(state): State<Arc<HttpState>>, uri: Uri, headers: HeaderMap) -> Response {
    let first_response = state.first_response.snapshot();
    let accept = headers.get(header::ACCEPT).and_then(|value| value.to_str().ok());
    let format = match metrics::Format::negotiate(uri.query(), accept) {
        Ok(format) => format,
        Err(message) => return error(StatusCode::BAD_REQUEST, message),
    };
    let input_metrics = metrics::input_metrics(
        state.model.input_store().map(|store| store.stats()).unwrap_or_default(),
        state.images.as_ref().map(|processor| processor.preparation_counts()).unwrap_or_default(),
    );
    if format == metrics::Format::Json {
        let learned_drafter = state.model.config().has_learned_drafts();
        let snapshot = state.model.metrics().snapshot(learned_drafter);
        let mut value = snapshot.json(learned_drafter);
        if let Some(object) = value.as_object_mut() {
            object.insert("first_response".into(), serde_json::to_value(first_response).unwrap());
            for (name, _, _, count) in input_metrics {
                object.insert(name.into(), count.into());
            }
        }
        return ([(header::VARY, "accept")], Json(value)).into_response();
    }
    let mut text = state.model.metrics().render();
    for (name, help, value) in [
        ("first_response_samples", "Retained eligible server-side first-response samples in the rolling window.", first_response.samples),
        ("first_response_window_seconds", "Retention window for first-response observations.", first_response.window_seconds),
        ("first_response_capacity", "Maximum retained first-response observations.", first_response.capacity as u64),
        ("first_response_truncated", "Eligible observations were evicted before their window expired.", u64::from(first_response.truncated)),
    ] { metrics::metric(&mut text, name, "gauge", help, value); }
    if let Some(median) = first_response.median_seconds {
        use std::fmt::Write;
        let _ = writeln!(text, "# HELP flare_first_response_median_seconds Server-side HTTP admission to first offered answer or reasoning text.\n# TYPE flare_first_response_median_seconds gauge\nflare_first_response_median_seconds {median}");
    }
    for (name, kind, help, value) in input_metrics {
        metrics::metric(&mut text, name, kind, help, value);
    }
    for (name, help, value) in [
        (
            "http_inflight_requests",
            "Reserved HTTP admission slots; a batch holds one slot per generation.",
            state.ingress_capacity - state.ingress_slots.available_permits(),
        ),
        (
            "http_request_capacity",
            "Maximum simultaneous reserved HTTP generation slots.",
            state.ingress_capacity,
        ),
        (
            "frontend_active_workers",
            "CPU workers preparing tokenizer, image and grammar inputs.",
            state.frontend_capacity - state.frontend_slots.available_permits(),
        ),
        (
            "frontend_worker_capacity",
            "Maximum simultaneous CPU preparation workers.",
            state.frontend_capacity,
        ),
        (
            "uptime_seconds",
            "Seconds since the HTTP listener started.",
            state.started.elapsed().as_secs() as usize,
        ),
        (
            "generated_tokens_last_minute",
            "Committed generated tokens during the most recent minute, sampled every five seconds.",
            state.generated_last_minute() as usize,
        ),
    ] {
        metrics::metric(&mut text, name, "gauge", help, value as u64);
    }
    (
        [
            (header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8"),
            (header::VARY, "accept"),
        ],
        text,
    )
        .into_response()
}
fn readiness_failure(
    message: &'static str,
    disconnected: std::sync::mpsc::RecvError,
    returned: Option<Error>,
) -> Error {
    let cause = match returned {
        Some(error) => Failure::caused_by(message, error),
        None => Failure::caused_by(message, disconnected),
    };
    ExecutionError::Worker(cause).into()
}

fn status(failure: &Error) -> StatusCode {
    match failure {
        Error::Admission(AdmissionError::Capacity(_) | AdmissionError::Unavailable(_)) => {
            StatusCode::SERVICE_UNAVAILABLE
        }
        Error::Admission(_) => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

fn error_code(failure: &Error) -> &'static str {
    match failure {
        Error::Admission(AdmissionError::Capacity(_)) => "capacity_exhausted",
        Error::Admission(AdmissionError::Unavailable(_)) => "server_unavailable",
        Error::Admission(AdmissionError::Context(_)) => "context_length_exceeded",
        Error::Admission(AdmissionError::Feature(_)) => "unsupported_feature",
        Error::Admission(AdmissionError::Grammar(_)) => "invalid_grammar",
        Error::Admission(AdmissionError::Image(_)) => "invalid_image",
        Error::Admission(AdmissionError::Invalid(_)) => "invalid_request",
        Error::Execution(ExecutionError::Capacity(_)) => "execution_capacity_exceeded",
        _ => "server_error",
    }
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .fold(0u8, |difference, (a, b)| difference | (a ^ b))
            == 0
}

async fn authorize(State(state): State<Arc<HttpState>>, request: Request, next: Next) -> Response {
    let protocol = protocol::Protocol::from_path(request.uri().path());
    if let Some(key) = &state.api_key {
        let authorization = request.headers().get(axum::http::header::AUTHORIZATION);
        let bearer = authorization.and_then(|value| value.as_bytes().strip_prefix(b"Bearer "));
        let anthropic = (protocol == protocol::Protocol::Messages)
            .then(|| request.headers().get("x-api-key"))
            .flatten()
            .map(|v| v.as_bytes());
        let valid = (authorization.is_none() || bearer.is_some())
            && bearer
                .or(anthropic)
                .is_some_and(|v| constant_time_eq(v, key.as_bytes()))
            && bearer
                .into_iter()
                .chain(anthropic)
                .all(|v| constant_time_eq(v, key.as_bytes()));
        if !valid {
            return http::error(
                protocol,
                StatusCode::UNAUTHORIZED,
                "missing or invalid API key",
            );
        }
    }
    next.run(request).await
}

fn error(status: StatusCode, message: impl std::fmt::Display) -> Response {
    (
        status,
        Json(json!({"error":{"message":message.to_string(),"type":"invalid_request_error"}})),
    )
        .into_response()
}

async fn admit_http(
    State(state): State<Arc<HttpState>>,
    mut request: Request,
    next: Next,
) -> Response {
    let protocol = protocol::Protocol::from_path(request.uri().path());
    if !state.model.healthy() {
        return http::error(
            protocol,
            StatusCode::SERVICE_UNAVAILABLE,
            "prepared model runtime is unavailable",
        );
    }
    let ingress = match acquire_ingress(&state.accepting, &state.ingress_slots) {
        Err(IngressRejection::Paused) => return http::error(protocol, StatusCode::SERVICE_UNAVAILABLE, "HTTP admission is paused"),
        Ok(permit) => Arc::new(permit),
        Err(IngressRejection::Full) => {
            let mut message = format!(
                "native HTTP admission is full: {} requests are active or queued",
                state.ingress_capacity
            );
            if !state.capacity_hint.is_empty() {
                message = format!("{message}; {}", state.capacity_hint);
            }
            return http::error(protocol, StatusCode::TOO_MANY_REQUESTS, message);
        }
    };
    request.extensions_mut().insert(ingress);
    request.extensions_mut().insert(AdmittedAt(Instant::now()));
    next.run(request).await
}

enum IngressRejection { Paused, Full }
#[derive(Clone, Copy)]
struct AdmittedAt(Instant);

fn acquire_ingress(accepting: &Mutex<bool>, slots: &Arc<Semaphore>) -> std::result::Result<OwnedSemaphorePermit, IngressRejection> {
    let accepting = accepting.lock().unwrap_or_else(|e| e.into_inner());
    if !*accepting { return Err(IngressRejection::Paused); }
    slots.clone().try_acquire_owned().map_err(|_| IngressRejection::Full)
}

#[cfg(test)]
mod admission_tests {
    use super::*;

    #[test]
    fn pausing_preserves_admitted_permits_and_resuming_reopens_capacity() {
        let accepting = Mutex::new(true);
        let slots = Arc::new(Semaphore::new(2));
        let admitted = acquire_ingress(&accepting, &slots).unwrap_or_else(|_| panic!("initial admission failed"));
        *accepting.lock().unwrap() = false;
        assert!(matches!(acquire_ingress(&accepting, &slots), Err(IngressRejection::Paused)));
        assert_eq!(slots.available_permits(), 1);
        drop(admitted);
        assert_eq!(slots.available_permits(), 2);
        *accepting.lock().unwrap() = true;
        assert!(acquire_ingress(&accepting, &slots).is_ok());
    }

    #[test]
    fn pause_barrier_prevents_concurrent_admissions_until_resume() {
        let accepting = Arc::new(Mutex::new(true));
        let slots = Arc::new(Semaphore::new(128));
        let begin = Arc::new(std::sync::Barrier::new(2));
        let worker_accepting = accepting.clone();
        let worker_slots = slots.clone();
        let worker_begin = begin.clone();
        let worker = thread::spawn(move || {
            worker_begin.wait();
            for _ in 0..10_000 {
                assert!(matches!(acquire_ingress(&worker_accepting, &worker_slots), Err(IngressRejection::Paused)));
            }
        });
        *accepting.lock().unwrap() = false;
        begin.wait();
        worker.join().unwrap();
        assert_eq!(slots.available_permits(), 128);
        *accepting.lock().unwrap() = true;
        assert!(acquire_ingress(&accepting, &slots).is_ok());
    }
}

fn chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: Value,
    finish: Value,
    include_usage: bool,
) -> Value {
    let mut value = json!({"id":id,"object":"chat.completion.chunk","created":created,"model":model,"choices":[{"index":0,"delta":delta,"finish_reason":finish}]});
    if include_usage {
        value["usage"] = Value::Null;
    }
    value
}

fn observed_chunk(
    id: &str,
    created: u64,
    model: &str,
    token: u32,
    timestamp_ns: Option<u64>,
    include_usage: bool,
) -> Value {
    let mut value = chunk(id, created, model, json!({}), Value::Null, include_usage);
    value["choices"][0]["token_ids"] = json!([token]);
    if let Some(timestamp) = timestamp_ns {
        value["choices"][0]["token_timestamps_ns"] = json!([timestamp]);
    }
    value
}

#[cfg(test)]
mod observation_tests {
    use super::*;

    #[test]
    fn readiness_failure_keeps_worker_cause_with_unchanged_fallback_message() {
        use std::error::Error as _;
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let joined = std::thread::spawn(move || -> Result<()> {
            drop(tx);
            Err(ExecutionError::Worker(Failure::caused_by(
                "cannot create HTTP runtime: denied",
                std::io::Error::from_raw_os_error(13),
            ))
            .into())
        });
        let disconnected = rx.recv().unwrap_err();
        let returned = joined.join().unwrap().err();
        let error = readiness_failure(
            "HTTP thread exited before readiness",
            disconnected,
            returned,
        );
        assert!(matches!(error, Error::Execution(ExecutionError::Worker(_))));
        assert_eq!(error.to_string(), "HTTP thread exited before readiness");
        let worker = error.source().unwrap().downcast_ref::<Error>().unwrap();
        assert_eq!(
            worker
                .source()
                .unwrap()
                .downcast_ref::<std::io::Error>()
                .unwrap()
                .raw_os_error(),
            Some(13)
        );
    }

    #[test]
    fn load_report_exposes_one_rank_of_scheduler_gauges() {
        let snapshot = Snapshot {
            active: 3,
            pending: 2,
            free_pages: 90,
            total_pages: 100,
            page_tokens: 128,
            ..Snapshot::default()
        };
        let report = load_report(&snapshot, 0.75, 42.0, 32);
        assert_eq!(report["dp_rank_count"], 1);
        assert_eq!(report["loads"].as_array().unwrap().len(), 1);
        let rank = &report["loads"][0];
        assert_eq!(rank["dp_rank"], 0);
        assert_eq!(rank["num_running_reqs"], 3);
        assert_eq!(rank["num_waiting_reqs"], 2);
        assert_eq!(rank["num_total_reqs"], 5);
        assert_eq!(rank["num_used_tokens"], 10 * 128);
        assert_eq!(rank["max_total_num_tokens"], 100 * 128);
        assert_eq!(rank["token_usage"].as_f64().unwrap(), 0.1);
        assert_eq!(rank["gen_throughput"].as_f64().unwrap(), 42.0);
        assert_eq!(rank["cache_hit_rate"].as_f64().unwrap(), 0.75);
        assert_eq!(rank["utilization"].as_f64().unwrap(), 3.0 / 32.0);
        assert_eq!(rank["max_running_requests"], 32);
        let empty = load_report(&Snapshot::default(), 0.0, 0.0, 0);
        assert_eq!(empty["loads"][0]["token_usage"].as_f64().unwrap(), 0.0);
        assert_eq!(empty["loads"][0]["utilization"].as_f64().unwrap(), 0.0);
        assert_eq!(empty["loads"][0]["max_total_num_tokens"], 0);
        let text = empty.to_string();
        assert!(!text.contains("NaN"), "{text}");
    }

    #[tokio::test]
    async fn typed_status_and_wire_messages_preserve_admission_and_execution_boundaries() {
        let cases = [
            (
                Error::admission(AdmissionError::Invalid, "invalid sampling parameters"),
                StatusCode::BAD_REQUEST,
            ),
            (
                Error::admission(
                    AdmissionError::Feature,
                    "parallel_tool_calls=false is not implemented",
                ),
                StatusCode::BAD_REQUEST,
            ),
            (
                Error::admission(AdmissionError::Grammar, "invalid grammar"),
                StatusCode::BAD_REQUEST,
            ),
            (
                Error::admission(AdmissionError::Image, "invalid image base64"),
                StatusCode::BAD_REQUEST,
            ),
            (
                Error::admission(AdmissionError::Capacity, "native request queue is full"),
                StatusCode::SERVICE_UNAVAILABLE,
            ),
            (
                Error::admission(AdmissionError::Context, "prompt exceeds the context"),
                StatusCode::BAD_REQUEST,
            ),
            (
                Error::admission(
                    AdmissionError::Unavailable,
                    "prepared model runtime is unavailable",
                ),
                StatusCode::SERVICE_UNAVAILABLE,
            ),
            (
                Error::execution(ExecutionError::Poisoned, "executor failed"),
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
        ];
        for (failure, status) in cases {
            let message = failure.to_string();
            let response = error(super::status(&failure), failure);
            assert_eq!(response.status(), status);
            let bytes = axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap();
            let body: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(body["error"]["message"], message);
        }
    }

    #[test]
    fn raw_metadata_chunks_never_duplicate_text_or_hide_eos() {
        let ordinary = chunk(
            "x",
            1,
            "model",
            json!({"content":"answer"}),
            Value::Null,
            false,
        );
        assert!(ordinary.get("usage").is_none());
        assert!(ordinary["choices"][0].get("token_ids").is_none());
        let observed = observed_chunk("x", 1, "model", 248046, Some(42), true);
        assert_eq!(observed["choices"][0]["delta"], json!({}));
        assert_eq!(observed["choices"][0]["token_ids"], json!([248046]));
        assert_eq!(observed["choices"][0]["token_timestamps_ns"], json!([42]));
        assert!(observed["choices"][0]["finish_reason"].is_null());
        assert!(observed["usage"].is_null());
        let ids_only = observed_chunk("x", 1, "model", 7, None, false);
        assert!(ids_only["choices"][0].get("token_timestamps_ns").is_none());
    }
}
