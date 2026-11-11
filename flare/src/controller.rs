// SPDX-License-Identifier: Apache-2.0
//! Asynchronous model-library and owned serving sessions for application clients.
pub use crate::details::{Compatibility, ModelDetailView};
#[path = "dashboard.rs"]
mod dashboard;
pub use dashboard::{DashboardPoint, DashboardView};
#[path = "recommendations.rs"]
mod recommendations;
use crate::{
    application::{
        AuthorizedModel, ModelStore, Source, TrustAcceptance, TrustOptions, TrustReview,
    },
    settings::{Settings, RuntimeIdentity},
};
pub use crate::settings::RuntimeSettings;
use flare_helios::{
    core::{cache::Entry, progress::{AcquisitionPhase, AcquisitionProgress}},
    runtime_core::NativeModel,
    server::{LoadedSession, ServingSession},
    startup::{StartupObservation, StartupProgress},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SessionState {
    #[default]
    Idle,
    Loading,
    Ready,
    Draining,
    Stopping,
    Failed,
    RestartRequired,
    ReviewRequired,
}

#[derive(Clone, Debug)]
pub enum Command {
    InspectModel(String),
    RemoveReference(String),
    LocateReference { id: String, path: PathBuf },
    Refresh,
    Discover(String),
    OpenModel(PathBuf),
    Download(String),
    ClearFailedDownloads,
    Activate(String),
    Stop,
    Resume(u64),
    AcceptTrust { decision: u64, publisher: String },
    RejectTrust(u64),
    SetModelDirectory(PathBuf),
    SetAppearance { theme: u32, compact: bool },
    SetRuntimeSettings(RuntimeSettings),
}
#[derive(Clone, Debug, Default)]
pub struct ModelMetadata {
    pub author: String,
    pub task: String,
    pub license: String,
    pub quantization: String,
    pub hardware: String,
    pub revision: String,
    pub revision_commit: String,
    pub base_model: String,
    pub context_tokens: Option<u64>,
    pub parameter_count: Option<u64>,
    pub revision_bytes: Option<u64>,
    pub rank_count: Option<u32>,
}
#[derive(Clone, Debug, Default)]
pub struct ModelView {
    pub id: String,
    pub name: String,
    pub version: String,
    pub source: String,
    pub path: String,
    pub description: String,
    pub error: String,
    pub logical_bytes: Option<u64>,
    pub flags: u32,
    pub metadata: ModelMetadata,
}
#[derive(Clone, Debug, Default)]
pub struct DownloadView {
    pub installed_model: String,
    pub operation_id: u64,
    pub state: u32,
    pub name: String,
    pub source: String,
    pub phase: String,
    pub error: String,
    pub network_bytes: u64,
    pub network_total: Option<u64>,
    pub verified_bytes: u64,
    pub materialized_bytes: u64,
    pub bytes_per_second: f64,
    pub transfer_active: bool,
    pub planning_completed: u64,
    pub planning_total: Option<u64>,
    pub can_cancel: bool,
    pub can_resume: bool,
}
#[derive(Clone, Debug, Default)]
pub struct TrustView {
    pub decision_id: u64,
    pub publisher: String,
    pub fingerprint: String,
    pub official_source: String,
    pub license: String,
    pub identity_status: String,
}
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub runtime_settings: RuntimeSettings,
    pub runtime_settings_pending_reload: bool,
    pub has_api_key: bool,
    pub last_settings_operation: u64,
    pub settings_error: String,
    pub bind_address: String,
    pub dashboard: DashboardView,
    pub appearance: Appearance,
    pub model_detail: Option<ModelDetailView>,
    pub detail_loading: bool,
    pub gpus: Vec<crate::hardware::GpuInfo>,
    pub gpu_error: String,
    pub hardware_sample_age_ms: Option<u64>,
    pub preview: crate::preview::PreviewSnapshot,
    pub current_operation: u64,
    pub state: SessionState,
    pub has_metrics: bool,
    pub session_id: u64,
    pub active_model: String,
    pub detail: String,
    pub error: String,
    pub endpoint: String,
    pub model_directory: String,
    pub stage: String,
    pub stage_completed: u64,
    pub stage_total: Option<u64>,
    pub accepting: bool,
    pub active_requests: u64,
    pub waiting_requests: u64,
    pub ingress_requests: u64,
    pub generated_tokens_per_second: f64,
    pub prefill_tokens_per_second: f64,
    pub first_response_seconds: Option<f64>,
    pub first_response_samples: u64,
    pub first_response_window_seconds: u64,
    pub first_response_truncated: bool,
    pub generated_tokens: u64,
    pub cached_pages: u64,
    pub total_pages: u64,
    pub busy: bool,
    pub catalog_loading: bool,
    pub models: Vec<ModelView>,
    pub downloads: Vec<DownloadView>,
    pub trust: Option<TrustView>,
}
const QUEUED: u32 = 0;
const RUNNING: u32 = 1;
const COMPLETE: u32 = 2;
const FAILED: u32 = 3;
const CANCELLED: u32 = 4;
const PAUSED: u32 = 5;

/// Native presentation preferences: 0 follows Windows, 1 is light, 2 is dark.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Appearance {
    pub theme: u32,
    pub compact: bool,
}
impl Default for Appearance {
    fn default() -> Self {
        Self {
            theme: 2,
            compact: false,
        }
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default)]
struct Persistent {
    runtime: Option<RuntimeSettings>,
    #[serde(flatten)]
    extra: serde_json::Map<String, serde_json::Value>,
    appearance: Appearance,
    model_directory: Option<PathBuf>,
    external: Vec<PathBuf>,
    downloads: Vec<SavedDownload>,
}
#[derive(Clone, Serialize, Deserialize)]
struct SavedDownload {
    #[serde(default)]
    installed_model: String,
    id: u64,
    source: String,
    directory: PathBuf,
    state: u32,
    error: String,
}
impl Persistent {
    fn load(path: &Path) -> Result<Self, String> {
        let file = match std::fs::File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self::default());
            }
            Err(error) => return Err(format!("Cannot read application state: {error}")),
        };
        let mut bytes = Vec::new();
        file.take(2 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        if bytes.len() > 2 * 1024 * 1024 {
            return Err("Application state exceeds 2 MiB".into());
        }
        let mut value: Self = serde_json::from_slice(&bytes)
            .map_err(|e| format!("Invalid application state: {e}"))?;
        if value.external.len() > 256 || value.downloads.len() > 128 {
            return Err("Application state exceeds its entry limit".into());
        }
        if value.appearance.theme > 2 {
            return Err("Application appearance is invalid".into());
        }
        if let Some(runtime) = &value.runtime { runtime.validate()?; }
        for download in &mut value.downloads {
            if matches!(download.state, QUEUED | RUNNING) {
                download.state = PAUSED;
            }
        }
        Ok(value)
    }
    fn save(&self, path: &Path) -> Result<(), String> {
        let bytes = serde_json::to_vec(self).map_err(|e| e.to_string())?;
        if bytes.len() > 2 * 1024 * 1024 {
            return Err("Application state exceeds 2 MiB".into());
        }
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        let mut temporary = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
        temporary.write_all(&bytes).map_err(|e| e.to_string())?;
        temporary.flush().map_err(|e| e.to_string())?;
        temporary.as_file().sync_all().map_err(|e| e.to_string())?;
        temporary.persist(path).map_err(|e| e.to_string())?;
        Ok(())
    }
}
#[derive(Clone)]
struct Operation {
    paused: Arc<AtomicBool>,
    cancel: Arc<AtomicBool>,
    acquisition: Option<AcquisitionProgress>,
}
impl Operation {
    fn new() -> Self {
        Self {
            cancel: Arc::new(AtomicBool::new(false)),
            paused: Arc::new(AtomicBool::new(false)),
            acquisition: None,
        }
    }
    fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
        if let Some(p) = &self.acquisition {
            p.request_cancel();
        }
    }
}
struct DownloadSlot {
    view: DownloadView,
    progress: AcquisitionProgress,
    rate: DownloadRate,
}

impl DownloadSlot {
    fn snapshot(&mut self) -> DownloadView {
        let mut view = self.view.clone();
        let p = self.progress.snapshot();
        if view.state == RUNNING && view.can_cancel {
            view.phase = p.phase.label().into();
        }
        view.network_bytes = p.network_bytes;
        view.network_total = p.network_total;
        view.verified_bytes = p.verification_bytes;
        view.materialized_bytes = p.materialized_bytes();
        let planning_active = view.state == RUNNING
            && view.can_cancel
            && p.phase == AcquisitionPhase::PreparingDownload;
        view.planning_completed = if planning_active { p.planning_completed } else { 0 };
        view.planning_total = if planning_active { p.planning_total } else { None };
        view.transfer_active = view.state == RUNNING
            && view.can_cancel
            && p.phase == AcquisitionPhase::Downloading;
        view.bytes_per_second = self.rate.sample(view.transfer_active, p.elapsed, p.network_bytes);
        view
    }
}

#[derive(Default)]
struct DownloadRate {
    samples: VecDeque<(Duration, u64)>,
    value: f64,
}

impl DownloadRate {
    fn sample(&mut self, active: bool, elapsed: Duration, bytes: u64) -> f64 {
        const WINDOW: Duration = Duration::from_secs(2);
        const INTERVAL: Duration = Duration::from_millis(125);
        if !active {
            self.samples.clear();
            self.value = 0.0;
            return 0.0;
        }
        if let Some(&(time, count)) = self.samples.back() {
            if elapsed < time || bytes < count || elapsed.saturating_sub(time) > WINDOW {
                self.samples.clear();
                self.value = 0.0;
            } else if elapsed.saturating_sub(time) < INTERVAL {
                return self.value;
            }
        }
        self.samples.push_back((elapsed, bytes));
        while self
            .samples
            .front()
            .is_some_and(|(time, _)| elapsed.saturating_sub(*time) > WINDOW)
        {
            self.samples.pop_front();
        }
        self.value = self.samples.front().map_or(0.0, |&(time, count)| {
            let seconds = elapsed.saturating_sub(time).as_secs_f64();
            if seconds > 0.0 {
                bytes.saturating_sub(count) as f64 / seconds
            } else {
                0.0
            }
        });
        self.value
    }
}

#[cfg(test)]
mod download_rate_tests {
    use super::*;

    #[test]
    fn snapshots_show_header_planning_and_transfer_as_distinct_work() {
        let progress = AcquisitionProgress::new();
        progress.begin_download_planning(9);
        progress.record_planned_file();
        let mut slot = DownloadSlot {
            view: DownloadView { state: RUNNING, can_cancel: true, ..Default::default() },
            progress: progress.clone(),
            rate: DownloadRate::default(),
        };
        let planning = slot.snapshot();
        assert!(!planning.transfer_active);
        assert_eq!(planning.planning_completed, 1);
        assert_eq!(planning.planning_total, Some(9));
        assert_eq!(planning.network_total, None);
        assert_eq!(planning.bytes_per_second, 0.0);
        progress.set_phase(AcquisitionPhase::Downloading);
        let downloading = slot.snapshot();
        assert!(downloading.transfer_active);
        assert_eq!(downloading.planning_total, None);
        progress.set_phase(AcquisitionPhase::Checking);
        assert!(!slot.snapshot().transfer_active);
        assert!(slot.rate.samples.is_empty());
        progress.set_phase(AcquisitionPhase::Downloading);
        slot.view.can_cancel = false;
        assert!(!slot.snapshot().transfer_active);
        slot.view.can_cancel = true;
        slot.view.state = COMPLETE;
        assert!(!slot.snapshot().transfer_active);
    }

    #[test]
    fn transfer_rate_uses_recent_bytes_instead_of_acquisition_lifetime() {
        let mut rate = DownloadRate::default();
        assert_eq!(rate.sample(true, Duration::from_secs(300), 10_000), 0.0);
        assert_eq!(rate.sample(true, Duration::from_secs(301), 12_000), 2_000.0);
        assert_eq!(rate.sample(false, Duration::from_secs(302), 12_000), 0.0);
        assert!(rate.samples.is_empty());
        assert_eq!(rate.sample(true, Duration::from_secs(600), 12_000), 0.0);
        assert_eq!(rate.sample(true, Duration::from_secs(601), 15_000), 3_000.0);
    }

    #[test]
    fn stalled_transfer_discards_old_throughput_and_bounds_history() {
        let mut rate = DownloadRate::default();
        for tick in 0..=400 {
            rate.sample(true, Duration::from_millis(tick * 125), tick * 125);
            assert!(rate.samples.len() <= 17);
        }
        assert_eq!(rate.value, 1_000.0);
        for tick in 401..=417 {
            rate.sample(true, Duration::from_millis(tick * 125), 50_000);
        }
        assert_eq!(rate.value, 0.0);
    }

    #[test]
    fn counter_resets_and_snapshot_gaps_start_a_fresh_window() {
        let mut rate = DownloadRate::default();
        rate.sample(true, Duration::from_secs(1), 100);
        assert_eq!(rate.sample(true, Duration::from_secs(2), 200), 100.0);
        assert_eq!(rate.sample(true, Duration::from_secs(3), 10), 0.0);
        assert_eq!(rate.sample(true, Duration::from_secs(4), 110), 100.0);
        assert_eq!(rate.sample(true, Duration::ZERO, 110), 0.0);
        assert_eq!(rate.sample(true, Duration::from_secs(10), 1_000), 0.0);
    }
}

struct Inner {
    view: Snapshot,
    active: Option<Arc<ActiveSession>>,
    startup: Option<StartupObservation>,
    operations: HashMap<u64, Operation>,
    downloads: Vec<DownloadSlot>,
    runtime_operation: Option<u64>,
}
struct Shared {
    hardware: crate::hardware::HardwareMonitor,
    preview: crate::preview::Preview,
    inner: Mutex<Inner>,
    closing: AtomicBool,
    closed: AtomicBool,
    next: AtomicU64,
    dashboard: Mutex<dashboard::DashboardHistory>,
    restart_required: AtomicBool,
}
impl Shared {
    fn update(&self, f: impl FnOnce(&mut Inner)) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut inner);
        if self.restart_required.load(Ordering::Acquire) {
            inner.view.state = SessionState::RestartRequired;
        }
    }
    fn require_restart(&self, error: String) {
        self.restart_required.store(true, Ordering::Release);
        self.update(|s| {
            s.view.error = error;
        });
    }
    fn cancelled(&self, operation: u64) -> bool {
        self.closing.load(Ordering::Acquire)
            || self
                .inner
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .operations
                .get(&operation)
                .is_some_and(|v| v.cancel.load(Ordering::Acquire))
    }
    fn id(&self) -> u64 {
        self.next.fetch_add(1, Ordering::Relaxed)
    }
}
struct Envelope {
    id: u64,
    command: Command,
}
pub struct Controller {
    shared: Arc<Shared>,
    sender: mpsc::SyncSender<Envelope>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl Controller {
    pub fn new(config: Option<PathBuf>) -> Result<Self, String> {
        let mut settings = Settings::load(config.as_deref()).map_err(|e| e.to_string())?;
        let persistent = Persistent::load(&settings.state_file)?;
        if let Some(path) = &persistent.model_directory {
            settings.model_directory = path.clone();
        }
        if let Some(runtime) = &persistent.runtime { settings.runtime = runtime.clone(); }
        settings.runtime.validate()?;
        let next = persistent
            .downloads
            .iter()
            .map(|d| d.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        let shared = Arc::new(Shared {
            hardware: crate::hardware::HardwareMonitor::start().map_err(|e| e.to_string())?,
            preview: crate::preview::Preview::new(settings.api_key.clone())?,
            inner: Mutex::new(Inner {
                view: Snapshot {
                    models: recommendations::catalog(&[]),
                    runtime_settings: settings.runtime.clone(),
                    has_api_key: settings.api_key.is_some(),
                    appearance: persistent.appearance,
                    model_directory: settings.model_directory.display().to_string(),
                    ..Snapshot::default()
                },
                active: None,
                startup: None,
                operations: HashMap::new(),
                runtime_operation: None,
                downloads: persistent
                    .downloads
                    .iter()
                    .map(|d| DownloadSlot {
                        view: DownloadView {
                            installed_model: String::new(),
                            operation_id: d.id,
                            state: d.state,
                            name: recommendations::acquisition_name(&d.source),
                            source: d.source.clone(),
                            error: d.error.clone(),
                            can_resume: matches!(d.state, FAILED | CANCELLED | PAUSED),
                            ..DownloadView::default()
                        },
                        progress: AcquisitionProgress::new(),
                        rate: DownloadRate::default(),
                    })
                    .collect(),
            }),
            closing: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            next: AtomicU64::new(next),
            dashboard: Mutex::new(dashboard::DashboardHistory::default()),
            restart_required: AtomicBool::new(false),
        });
        let (sender, receiver) = mpsc::sync_channel(32);
        let owned = shared.clone();
        let worker = thread::Builder::new()
            .name("flare-application".into())
            .spawn(move || {
                let mut worker = Worker::new(owned.clone(), settings, persistent);
                let result =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| worker.run(receiver)));
                if result.is_err() {
                    owned.require_restart("The application worker failed; restart Flare".into());
                }
                worker.shutdown();
                owned.closed.store(true, Ordering::Release);
            })
            .map_err(|e| e.to_string())?;
        let controller = Self {
            shared,
            sender,
            worker: Mutex::new(Some(worker)),
        };
        controller.submit(Command::Refresh)?;
        Ok(controller)
    }
    pub fn submit(&self, command: Command) -> Result<u64, String> {
        if self.shared.closing.load(Ordering::Acquire) || self.is_closed() {
            return Err("The application is closing".into());
        }
        let id = self.shared.id();
        let mut interrupted = None;
        self.shared.update(|s| {
            if matches!(command, Command::Stop) {
                interrupted = s
                    .runtime_operation
                    .and_then(|id| s.operations.get(&id))
                    .cloned();
            }
            s.operations.insert(id, Operation::new());
        });
        if let Err(error) = self.sender.try_send(Envelope { id, command }) {
            self.shared.update(|s| {
                s.operations.remove(&id);
            });
            return Err(match error {
                mpsc::TrySendError::Full(_) => "The application command queue is full",
                _ => "The application worker is unavailable",
            }
            .into());
        }
        if let Some(operation) = interrupted {
            operation.cancel();
        }
        Ok(id)
    }
    pub fn cancel(&self, operation: u64) -> Result<u64, String> {
        let mut result = Err("The operation has already finished or does not exist".into());
        self.shared.update(|s| {
            if let Some(control) = s.operations.get(&operation) {
                control.cancel();
                result = Ok(());
            }
        });
        result.map(|()| self.shared.id())
    }

    /// Continue the in-memory conversation with the exact owned serving session.
    pub fn try_model(&self, prompt: String) -> Result<u64, String> {
        if self.shared.closing.load(Ordering::Acquire) {
            return Err("The application is closing".into());
        }
        let inner = self.shared.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.view.state != SessionState::Ready {
            return Err("Load a model and wait until it is ready before sending a message".into());
        }
        let active = inner.active.as_ref().ok_or("No model is loaded")?;
        let sample = active.session.server().snapshot();
        if !sample.ready {
            return Err("The model is not accepting requests".into());
        }
        let mut address = active.session.server().address();
        if address.ip().is_unspecified() {
            address.set_ip(if address.is_ipv4() {
                std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST)
            } else {
                std::net::IpAddr::V6(std::net::Ipv6Addr::LOCALHOST)
            });
        }
        self.shared
            .preview
            .start_owned(address, sample.session_id, prompt)
    }

    pub fn cancel_preview(&self) -> Result<u64, String> {
        let _inner = self.shared.inner.lock().unwrap_or_else(|e| e.into_inner());
        let preview = self.shared.preview.snapshot();
        if !preview.busy {
            return Err("No chat response is running".into());
        }
        self.shared.preview.cancel();
        Ok(preview.operation_id)
    }

    /// Clear transient conversation context without changing the serving session.
    pub fn clear_chat(&self) -> Result<(), String> {
        let _inner = self.shared.inner.lock().unwrap_or_else(|e| e.into_inner());
        if self.shared.closing.load(Ordering::Acquire) {
            return Err("The application is closing".into());
        }
        self.shared.preview.clear()
    }

    /// Retain partial transport data and mark a transfer ready for explicit resume.
    pub fn pause_download(&self, operation: u64) -> Result<u64, String> {
        let mut result = Err("The download is not running or queued".into());
        self.shared.update(|s| {
            let Some(slot) = s.downloads.iter_mut().find(|d| {
                d.view.operation_id == operation && matches!(d.view.state, QUEUED | RUNNING)
            }) else {
                return;
            };
            let Some(control) = s.operations.get(&operation) else {
                return;
            };
            control.paused.store(true, Ordering::Release);
            control.cancel();
            slot.view.phase = "Pausing; keeping partial data".into();
            slot.view.can_cancel = false;
            result = Ok(operation);
        });
        result
    }

    pub fn snapshot(&self) -> Snapshot {
        let (mut view, live, startup) = {
            let mut inner = self.shared.inner.lock().unwrap_or_else(|e| e.into_inner());
            let mut view = inner.view.clone();
            view.downloads = inner
                .downloads
                .iter_mut()
                .map(DownloadSlot::snapshot)
                .collect();
            // Borrow the session while sampling so its CUDA owner can only be
            // destroyed by the application worker after this lock is released.
            let live = inner.active.as_ref().map(|active| {
                let sample = active.session.server().snapshot();
                // Sampling under the session lock prevents a delayed reader
                // from replacing a newer session's history with older counters.
                let dashboard = self.shared.dashboard.lock().unwrap_or_else(|e| e.into_inner()).sample(&sample);
                let bind = active.session.server().address();
                view.bind_address = bind.to_string();
                (
                    sample,
                    active.name.clone(),
                    active.id.clone(),
                    active.path.clone(),
                    format!("http://{}", client_address(bind)),
                    dashboard,
                )
            });
            if live.is_none() {
                self.shared.dashboard.lock().unwrap_or_else(|e| e.into_inner()).clear();
            }
            (view, live, inner.startup.clone())
        };
        if let Some(update) = startup.and_then(|s| s.snapshot()) {
            view.stage = update.stage.as_str().into();
            view.stage_completed = update.completed;
            view.stage_total = update.total;
            if !update.detail.is_empty() {
                view.detail = update.detail;
            }
        }
        if let Some((sample, name, id, path, endpoint, dashboard)) = live {
            view.generated_tokens_per_second = dashboard.generated_tokens_per_second;
            view.prefill_tokens_per_second = dashboard.prefill_tokens_per_second;
            view.dashboard = dashboard;
            view.has_metrics = true;
            view.session_id = sample.session_id;
            view.active_model = name;
            view.endpoint = endpoint;
            view.accepting = sample.ready;
            if !sample.healthy
                && !matches!(
                    view.state,
                    SessionState::RestartRequired | SessionState::Stopping
                )
            {
                view.state = SessionState::Failed;
                if view.error.is_empty() {
                    view.error = "The model runtime is unavailable".into();
                }
            }
            view.active_requests = sample.model.active;
            view.waiting_requests = sample.model.pending;
            view.ingress_requests = sample.ingress_active as u64;
            view.generated_tokens = sample.model.generated_tokens;
            view.first_response_seconds = sample.first_response.median_seconds;
            view.first_response_samples = sample.first_response.samples;
            view.first_response_window_seconds = sample.first_response.window_seconds;
            view.first_response_truncated = sample.first_response.truncated;
            view.cached_pages = sample.model.cached_pages;
            view.total_pages = sample.model.total_pages;
            for model in &mut view.models {
                if same_core(model, &id, &path) {
                    model.flags |= 4 | 8;
                }
            }
        }
        let hardware = self.shared.hardware.snapshot();
        view.gpus = hardware.gpus;
        view.gpu_error = hardware.error;
        view.hardware_sample_age_ms = hardware.sample_age_ms;
        view.preview = self.shared.preview.snapshot();
        if view.preview.session_id != view.session_id {
            view.preview = crate::preview::PreviewSnapshot::default();
        }
        view
    }
    pub fn close(&self) {
        if !self.shared.closing.swap(true, Ordering::AcqRel) {
            self.shared.preview.close();
            self.shared.update(|s| {
                for op in s.operations.values() {
                    if op.acquisition.is_some() {
                        op.paused.store(true, Ordering::Release);
                    }
                    op.cancel();
                }
                s.view.busy = true;
                if s.view.state != SessionState::RestartRequired {
                    s.view.state = SessionState::Stopping;
                }
                s.view.detail = "Closing Flare".into();
            });
        }
    }
    pub fn is_closed(&self) -> bool {
        self.shared.closed.load(Ordering::Acquire)
    }
}
impl Drop for Controller {
    fn drop(&mut self) {
        self.close();
        if let Some(worker) = self
            .worker
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            let _ = worker.join();
        }
    }
}

struct LeaseGuard {
    lease: Option<Entry>,
    drained: bool,
}
impl Drop for LeaseGuard {
    fn drop(&mut self) {
        if !self.drained {
            if let Some(lease) = self.lease.take() {
                std::mem::forget(lease);
            }
        }
    }
}
struct ActiveSession {
    runtime: RuntimeIdentity,
    model_names: Vec<String>,
    session: ServingSession,
    lease: Mutex<LeaseGuard>,
    id: String,
    name: String,
    path: String,
}
impl ActiveSession {
    fn close(&self) -> flare_helios::Shutdown {
        let shutdown = self.session.close();
        self.lease.lock().unwrap_or_else(|e| e.into_inner()).drained = shutdown.drained;
        shutdown
    }
}
impl Drop for ActiveSession {
    fn drop(&mut self) {
        self.close();
    }
}
struct PendingReview {
    decision: u64,
    operation: u64,
    id: String,
    name: String,
    review: TrustReview,
}
struct DownloadJob {
    saved: SavedDownload,
    settings: Settings,
    progress: AcquisitionProgress,
}
struct DownloadDone {
    id: u64,
    result: Result<String, String>,
    cancelled: bool,
}
struct Worker {
    shared: Arc<Shared>,
    settings: Settings,
    persistent: Persistent,
    sources: HashMap<String, Source>,
    catalog: Vec<ModelView>,
    pending: Option<PendingReview>,
    active: Option<Arc<ActiveSession>>,
    download_sender: Option<mpsc::SyncSender<DownloadJob>>,
    download_done: mpsc::Receiver<DownloadDone>,
    download_thread: Option<JoinHandle<()>>,
}

impl Worker {
    fn new(shared: Arc<Shared>, settings: Settings, persistent: Persistent) -> Self {
        let (sender, receiver) = mpsc::sync_channel::<DownloadJob>(32);
        let (done, download_done) = mpsc::channel();
        let observed = shared.clone();
        let download_thread = thread::Builder::new()
            .name("flare-acquisition".into())
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    observed.update(|s| {
                        if let Some(slot) = s
                            .downloads
                            .iter_mut()
                            .find(|d| d.view.operation_id == job.saved.id)
                        {
                            slot.view.state = RUNNING;
                            slot.view.phase = "Acquire core".into();
                        }
                    });
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                        || -> Result<String, String> {
                            if job.progress.is_cancelled() {
                                return Err("Acquisition cancelled".into());
                            }
                            let mut options = job.settings.hub.clone();
                            let (_, revision) = crate::hub::repository(&job.saved.source)
                                .map_err(|e| e.to_string())?;
                            if options.revision.is_none() && revision.is_none() {
                                options.revision = Some(
                                    crate::hardware::detect(1)
                                        .map_err(|e| e.to_string())?
                                        .into(),
                                );
                            }
                            let store = ModelStore::open(job.saved.directory.clone())
                                .map_err(|e| e.to_string())?;
                            let profiles =
                                crate::trust::trusted_profiles(job.settings.trust_store.as_deref())
                                    .map_err(|e| e.to_string())?;
                            let entry = store
                                .acquire(
                                    &job.saved.source,
                                    &options,
                                    1,
                                    &job.settings.publisher_keys,
                                    job.settings.trust_store.as_deref(),
                                    &profiles,
                                    job.progress.clone(),
                                )
                                .map_err(|e| e.to_string())?;
                            Ok(format!("local:{}@{}", entry.model(), entry.id()))
                        },
                    ))
                    .unwrap_or_else(|_| Err("The acquisition worker failed".into()));
                    let _ = done.send(DownloadDone {
                        id: job.saved.id,
                        result,
                        cancelled: job.progress.is_cancelled(),
                    });
                }
            })
            .ok();
        let download_sender = download_thread.as_ref().map(|_| sender);
        Self {
            shared,
            settings,
            persistent,
            sources: HashMap::new(),
            catalog: Vec::new(),
            pending: None,
            active: None,
            download_sender,
            download_done,
            download_thread,
        }
    }
    fn state(&self, state: SessionState, detail: impl Into<String>) {
        let detail = detail.into();
        if state == SessionState::RestartRequired {
            self.shared.restart_required.store(true, Ordering::Release);
        }
        self.shared.update(|s| {
            s.view.state = state;
            s.view.detail = detail;
        });
    }
    fn idle_state(&self) -> SessionState {
        if self.active.is_some() {
            SessionState::Ready
        } else {
            SessionState::Idle
        }
    }
    fn run(&mut self, receiver: mpsc::Receiver<Envelope>) {
        while !self.shared.closing.load(Ordering::Acquire) {
            self.collect_downloads();
            if self
                .pending
                .as_ref()
                .is_some_and(|p| self.shared.cancelled(p.operation))
            {
                self.clear_review();
            }
            if self
                .active
                .as_ref()
                .is_some_and(|a| !a.session.server().healthy())
            {
                let drained = self.stop_active();
                if drained {
                    self.state(SessionState::Failed, "The model runtime stopped");
                }
            }
            let envelope = match receiver.recv_timeout(Duration::from_millis(100)) {
                Ok(value) => value,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(_) => break,
            };
            let id = envelope.id;
            let settings_command = matches!(&envelope.command, Command::SetRuntimeSettings(_));
            self.shared.update(|s| {
                s.view.busy = true;
                if !settings_command && s.view.state != SessionState::RestartRequired {
                    s.view.error.clear();
                }
                s.runtime_operation = Some(id);
                s.view.current_operation = id;
            });
            let result = if self.shared.cancelled(id) {
                if settings_command { Err("Settings update cancelled".into()) } else { Ok(()) }
            } else {
                self.execute(id, envelope.command)
            };
            if settings_command {
                self.shared.update(|s| {
                    s.view.last_settings_operation = id;
                    s.view.settings_error = result.as_ref().err().cloned().unwrap_or_default();
                });
            }
            // Settings errors belong to the form; they do not change the
            // loaded model's serving state or the idle model workflow.
            let result = if settings_command { Ok(()) } else { result };
            if let Err(error) = result {
                let state = if self.pending.is_some() {
                    SessionState::ReviewRequired
                } else {
                    self.idle_state()
                };
                self.shared.update(|s| {
                    if s.view.state != SessionState::RestartRequired || s.view.error.is_empty() {
                        s.view.error = error;
                    }
                    if s.view.state != SessionState::RestartRequired {
                        s.view.state = if state == SessionState::Idle {
                            SessionState::Failed
                        } else {
                            state
                        };
                    }
                });
            }
            let pending_id = self.pending.as_ref().map(|p| p.operation);
            self.shared.update(|s| {
                s.view.busy = false;
                s.runtime_operation = None;
                s.startup = None;
                s.view.current_operation = pending_id.unwrap_or(0);
                let download_active = s
                    .downloads
                    .iter()
                    .any(|d| d.view.operation_id == id && d.view.can_cancel);
                if pending_id != Some(id) && !download_active {
                    s.operations.remove(&id);
                }
            });
        }
    }
    fn execute(&mut self, id: u64, command: Command) -> Result<(), String> {
        match command {
            Command::SetRuntimeSettings(runtime) => {
                runtime.validate()?;
                let mut desired = self.settings.clone();
                desired.runtime = runtime.clone();
                let pending = if let Some(active) = &self.active {
                    let names: Vec<_> = active.model_names.iter().map(String::as_str).collect();
                    desired.runtime_identity(&names).map_err(|e| e.to_string())? != active.runtime
                } else { false };
                let mut saved = self.persistent.clone();
                saved.runtime = Some(runtime.clone());
                saved.save(&self.settings.state_file)?;
                self.persistent = saved;
                self.settings = desired;
                self.shared.update(|s| {
                    s.view.runtime_settings = runtime;
                    s.view.runtime_settings_pending_reload = pending;
                });
                Ok(())
            }
            Command::Refresh => self.refresh(),
            Command::InspectModel(target) => self.inspect_model(id, &target),
            Command::RemoveReference(target) => self.remove_reference(&target),
            Command::LocateReference { id, path } => self.locate_reference(&id, path),
            Command::Discover(query) => {
                self.shared.update(|s| s.view.catalog_loading = true);
                let hardware = crate::hardware::detect(1).ok();
                let result = crate::catalog::search_for_hardware(&query, hardware)
                    .map_err(|e| e.to_string());
                self.shared.update(|s| s.view.catalog_loading = false);
                let entries = result?;
                self.catalog = entries
                    .into_iter()
                    .map(|entry| ModelView {
                        id: format!("hub:{}", entry.id),
                        name: entry.id.split_once('/').map_or(entry.id.as_str(), |(_, name)| name).into(),
                        source: entry.id,
                        metadata: ModelMetadata {
                            author: entry.author,
                            task: entry.task,
                            license: entry.license,
                            quantization: entry.quantization,
                            hardware: entry.hardware,
                            revision: entry.revision,
                            revision_commit: entry.revision_commit,
                            base_model: entry.base_model,
                            context_tokens: entry.context_tokens,
                            parameter_count: entry.parameter_count,
                            revision_bytes: entry.revision_bytes,
                            ..Default::default()
                        },
                        flags: 16,
                        ..ModelView::default()
                    })
                    .collect();
                self.refresh()
            }
            Command::OpenModel(path) => {
                let path = path
                    .canonicalize()
                    .map_err(|e| format!("Cannot open local core: {e}"))?;
                if !self.persistent.external.contains(&path) {
                    if self.persistent.external.len() >= 256 {
                        return Err("The external model list is full".into());
                    }
                    let mut saved = self.persistent.clone();
                    saved.external.push(path.clone());
                    saved.save(&self.settings.state_file)?;
                    self.persistent = saved;
                }
                self.refresh()?;
                self.begin_review(
                    id,
                    external_id(&path),
                    path.file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into(),
                    Source::LocalPath(path),
                )
            }
            Command::Activate(model) => {
                let source = self
                    .sources
                    .get(&model)
                    .cloned()
                    .ok_or("The model is no longer in the local library")?;
                let name = self
                    .shared
                    .inner
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .view
                    .models
                    .iter()
                    .find(|m| m.id == model)
                    .map(|m| m.name.clone())
                    .unwrap_or_else(|| model.clone());
                self.begin_review(id, model, name, source)
            }
            Command::Download(source) => {
                self.queue_download(id, source, self.settings.model_directory.clone())
            }
            Command::ClearFailedDownloads => self.clear_failed_downloads(),
            Command::Resume(previous) => {
                let saved = self
                    .persistent
                    .downloads
                    .iter()
                    .find(|d| d.id == previous)
                    .cloned()
                    .ok_or("The acquisition no longer exists")?;
                if !matches!(saved.state, FAILED | CANCELLED | PAUSED) {
                    return Err("Only interrupted acquisitions can resume".into());
                }
                self.queue_download(previous, saved.source, saved.directory)
            }
            Command::Stop => {
                self.clear_review();
                if self.stop_active() {
                    self.state(SessionState::Idle, "Model stopped");
                }
                Ok(())
            }
            Command::AcceptTrust {
                decision,
                publisher,
            } => {
                let pending = self
                    .pending
                    .as_ref()
                    .ok_or("There is no publisher review awaiting a decision")?;
                if pending.decision != decision || pending.review.disclosure.namespace != publisher
                {
                    return Err(
                        "Acceptance must match the displayed review and exact publisher namespace"
                            .into(),
                    );
                }
                let pending = self.pending.take().unwrap();
                self.shared.update(|s| {
                    s.view.trust = None;
                    s.operations.remove(&pending.operation);
                });
                self.authorize_and_load(id, pending.id, pending.name, pending.review)
            }
            Command::RejectTrust(decision) => {
                if self.pending.as_ref().is_none_or(|p| p.decision != decision) {
                    return Err("The publisher review has expired".into());
                }
                self.clear_review();
                Ok(())
            }
            Command::SetModelDirectory(path) => {
                let store = ModelStore::open(path).map_err(|e| e.to_string())?;
                let path = store.path().canonicalize().map_err(|e| e.to_string())?;
                let mut saved = self.persistent.clone();
                saved.model_directory = Some(path.clone());
                saved.save(&self.settings.state_file)?;
                self.persistent = saved;
                self.settings.model_directory = path.clone();
                self.shared
                    .update(|s| {
                        s.view.model_directory = path.display().to_string();
                        for download in &mut s.downloads { download.view.installed_model.clear(); }
                    });
                self.refresh()
            }
            Command::SetAppearance { theme, compact } => {
                if theme > 2 {
                    return Err("Choose System, Light, or Dark appearance".into());
                }
                let mut saved = self.persistent.clone();
                saved.appearance = Appearance { theme, compact };
                saved.save(&self.settings.state_file)?;
                self.shared.update(|s| s.view.appearance = saved.appearance);
                self.persistent = saved;
                Ok(())
            }
        }
    }
}

fn same_core(model: &ModelView, id: &str, path: &str) -> bool {
    model.id == id && !path.is_empty() && model.path == path
}

fn active_id_in_models<'a>(id: &'a str, path: &str, models: &[ModelView]) -> Option<&'a str> {
    models
        .iter()
        .any(|model| same_core(model, id, path))
        .then_some(id)
}

fn visible_download_model(
    saved: &SavedDownload,
    directory: &Path,
    sources: &HashMap<String, Source>,
) -> String {
    let same_root = saved.directory == directory
        || match (saved.directory.canonicalize(), directory.canonicalize()) {
            (Ok(acquired), Ok(selected)) => acquired == selected,
            _ => false,
        };
    if saved.state != COMPLETE || !same_root {
        return String::new();
    }
    match sources.get(&saved.installed_model) {
        Some(Source::LocalModel {
            model,
            version: Some(version),
        }) if saved.installed_model == format!("local:{model}@{version}") => {
            saved.installed_model.clone()
        }
        _ => String::new(),
    }
}

fn client_address(mut address: std::net::SocketAddr) -> std::net::SocketAddr {
    if address.ip().is_unspecified() {
        address.set_ip(if address.is_ipv4() {
            std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST)
        } else { std::net::IpAddr::V6(std::net::Ipv6Addr::LOCALHOST) });
    }
    address
}

fn external_id(path: &Path) -> String {
    use std::hash::{Hash, Hasher};
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut hash);
    format!("external:{:016x}", hash.finish())
}

fn local_model_metadata(entry: &crate::application::LibraryEntry) -> ModelMetadata {
    let mut result = ModelMetadata {
        author: entry.hub_repositories.first()
            .and_then(|repository| repository.split_once('/'))
            .map_or_else(String::new, |(author, _)| author.to_owned()),
        ..Default::default()
    };
    if let Some(metadata) = &entry.metadata {
        result.hardware = metadata["hardware_identity"].as_str().unwrap_or_default().to_ascii_uppercase();
        if let Some(deployments) = metadata["deployments"].as_array() {
            let ready: Vec<_> = deployments.iter()
                .filter(|deployment| deployment["ready"].as_bool() == Some(true)).collect();
            let single_gpu: Vec<_> = ready.iter().copied()
                .filter(|deployment| deployment["tensor_parallel"].as_u64() == Some(1))
                .collect();
            let selected = match single_gpu.as_slice() {
                [deployment] => Some(*deployment),
                [] if ready.len() == 1 => ready.first().copied(),
                _ => None,
            };
            if let Some(deployment) = selected {
                result.context_tokens = deployment["max_sequence_tokens"].as_u64().filter(|value| *value > 0);
                result.rank_count = deployment["tensor_parallel"].as_u64()
                    .and_then(|value| u32::try_from(value).ok()).filter(|value| *value > 0);
            }
        }
    }
    result
}

fn catalog_with_library(
    catalog: &[ModelView],
    entries: &[crate::application::LibraryEntry],
    models: &[ModelView],
    active: Option<&str>,
) -> Vec<ModelView> {
    catalog
        .iter()
        .map(|candidate| {
            let installed = entries
                .iter()
                .filter(|entry| entry.hub_repositories.contains(&candidate.source))
                .filter_map(|entry| {
                    let id = format!("local:{}@{}", entry.model_key, entry.version);
                    models
                        .iter()
                        .find(|model| {
                            model.id == id && model.error.is_empty() && !model.path.is_empty()
                        })
                        .map(|model| (entry, model))
                })
                .max_by_key(|(entry, model)| {
                    (
                        active == Some(model.id.as_str()),
                        entry.selected,
                        entry.version.as_str(),
                        entry.model_key.as_str(),
                    )
                });
            let mut result = candidate.clone();
            if let Some((_, local)) = installed {
                // Acquisition provenance identifies a local choice, not the
                // repository's latest revision or an authenticated publisher.
                result.id = local.id.clone();
                result.version = local.version.clone();
                result.path = local.path.clone();
                result.logical_bytes = local.logical_bytes;
                result.metadata.context_tokens = local.metadata.context_tokens;
                result.metadata.hardware = local.metadata.hardware.clone();
                result.metadata.rank_count = local.metadata.rank_count;
                result.flags |= 1;
            }
            result
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn local_card_context_uses_an_unambiguous_deployment() {
        let mut entry = crate::application::LibraryEntry {
            model: "Model".into(), model_key: "model".into(), version: "version".into(),
            selected: true, leased: false,
            verification: crate::application::VerificationState::Unverified,
            hub_repositories: vec!["author/Model-FLARE".into()], error: None,
            metadata: Some(serde_json::json!({"hardware_identity":"gb10", "deployments":[
                {"tensor_parallel":1,"max_sequence_tokens":262144,"ready":true},
                {"tensor_parallel":2,"max_sequence_tokens":1048576,"ready":true}
            ]})),
        };
        let found = local_model_metadata(&entry);
        assert_eq!(found.author, "author");
        assert_eq!(found.hardware, "GB10");
        assert_eq!(found.context_tokens, Some(262144));
        assert_eq!(found.rank_count, Some(1));
        entry.metadata.as_mut().unwrap()["deployments"][1]["tensor_parallel"] = 1.into();
        let ambiguous = local_model_metadata(&entry);
        assert_eq!(ambiguous.context_tokens, None);
        assert_eq!(ambiguous.rank_count, None);
        entry.metadata.as_mut().unwrap()["deployments"][1]["ready"] = false.into();
        assert_eq!(local_model_metadata(&entry).context_tokens, Some(262144));
        entry.metadata.as_mut().unwrap()["deployments"][0]["ready"] = false.into();
        entry.metadata.as_mut().unwrap()["deployments"][1]["ready"] = true.into();
        entry.metadata.as_mut().unwrap()["deployments"][1]["tensor_parallel"] = 8.into();
        let multi_gpu = local_model_metadata(&entry);
        assert_eq!(multi_gpu.context_tokens, Some(1048576));
        assert_eq!(multi_gpu.rank_count, Some(8));
    }

    #[test]
    fn discovery_uses_acquisition_provenance_and_preserves_the_local_choice() {
        let repository = "publisher/Model-FLARE";
        let entries: Vec<_> = ["older", "selected", "unreadable"]
            .into_iter()
            .map(|version| crate::application::LibraryEntry {
                model: "Different local display name".into(),
                model_key: "local-model".into(),
                version: version.into(),
                selected: version == "selected",
                leased: false,
                verification: crate::application::VerificationState::Unverified,
                hub_repositories: vec![repository.into()],
                metadata: None,
                error: None,
            })
            .collect();
        let mut models: Vec<_> = entries
            .iter()
            .map(|entry| ModelView {
                id: format!("local:{}@{}", entry.model_key, entry.version),
                name: entry.model.clone(),
                version: entry.version.clone(),
                path: format!("C:/models/{}/model.safetensors", entry.version),
                logical_bytes: Some(42),
                metadata: ModelMetadata { context_tokens: Some(262144), hardware: "GB10".into(), rank_count: Some(1), ..Default::default() },
                error: if entry.version == "unreadable" {
                    "Missing core data".into()
                } else {
                    String::new()
                },
                flags: 1,
                ..Default::default()
            })
            .collect();
        let catalog: Vec<_> = [repository, "another-publisher/Model-FLARE"]
            .into_iter()
            .map(|source| ModelView {
                id: format!("hub:{source}"),
                name: source.into(),
                source: source.into(),
                metadata: ModelMetadata { context_tokens: Some(1048576), hardware: "GB202".into(), ..Default::default() },
                flags: 16,
                ..Default::default()
            })
            .collect();

        let matched = catalog_with_library(&catalog, &entries, &models, None);
        assert_eq!(matched[0].id, "local:local-model@selected");
        assert_eq!(matched[0].metadata.context_tokens, Some(262144));
        assert_eq!(matched[0].metadata.hardware, "GB10");
        assert_eq!(matched[0].metadata.rank_count, Some(1));
        assert_eq!(matched[0].name, repository);
        assert_eq!(matched[0].logical_bytes, Some(42));
        assert_eq!(matched[0].flags, 1 | 16);
        assert_eq!(matched[1].flags, 16);
        assert!(matched[1].path.is_empty());

        let matched =
            catalog_with_library(&catalog, &entries, &models, Some("local:local-model@older"));
        assert_eq!(matched[0].id, "local:local-model@older");
        let active_id = models[0].id.clone();
        let active_path = models[0].path.clone();
        let mut other_root = models.clone();
        other_root[0].path = "C:/another-library/older/model.safetensors".into();
        assert!(!same_core(&other_root[0], &active_id, &active_path));
        let matched = catalog_with_library(
            &catalog, &entries, &other_root,
            active_id_in_models(&active_id, &active_path, &other_root),
        );
        assert_eq!(matched[0].id, "local:local-model@selected");
        let matched = catalog_with_library(
            &catalog, &entries, &models,
            active_id_in_models(&active_id, &active_path, &models),
        );
        assert_eq!(matched[0].id, active_id);
        let matched = catalog_with_library(
            &catalog,
            &entries,
            &models,
            Some("local:local-model@unreadable"),
        );
        assert_eq!(matched[0].id, "local:local-model@selected");

        models[0].path.clear();
        models[1].path.clear();
        let matched = catalog_with_library(&catalog, &entries, &models, None);
        assert_eq!(matched[0].id, catalog[0].id);
        assert_eq!(matched[0].flags, 16);
    }

    fn configuration(directory: &Path) -> PathBuf {
        let path = directory.join("config.toml");
        std::fs::write(
            &path,
            format!(
                "[paths]\ncore_cache='{}'\n",
                directory.join("models").display()
            ),
        )
        .unwrap();
        path
    }
    fn wait_for(controller: &Controller, predicate: impl Fn(&Snapshot) -> bool) -> Snapshot {
        let started = Instant::now();
        loop {
            let view = controller.snapshot();
            if predicate(&view) {
                return view;
            }
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "Timed out: {view:?}"
            );
            thread::sleep(Duration::from_millis(10));
        }
    }
    fn close(controller: &Controller) {
        controller.close();
        let started = Instant::now();
        while !controller.is_closed() {
            assert!(started.elapsed() < Duration::from_secs(10));
            thread::sleep(Duration::from_millis(10));
        }
    }
    #[test]
    fn persistence_keeps_paths_and_marks_interrupted_acquisitions_resumable() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("state.json");
        let external = directory.path().join("模型.safetensors");
        let state = Persistent {
            appearance: Appearance::default(),
            model_directory: Some(directory.path().join("library")),
            external: vec![external.clone()],
            downloads: vec![SavedDownload {
                installed_model: String::new(),
                id: 17,
                source: "publisher/model@gb10".into(),
                directory: directory.path().join("models"),
                state: RUNNING,
                error: String::new(),
            }],
            ..Persistent::default()
        };
        state.save(&path).unwrap();
        let restored = Persistent::load(&path).unwrap();
        assert_eq!(restored.external, vec![external]);
        assert_eq!(restored.model_directory, state.model_directory);
        assert_eq!(restored.downloads[0].state, PAUSED);
        assert_eq!(restored.downloads[0].id, 17);
        state.save(&path).unwrap();
        assert!(Persistent::load(&path).is_ok());
    }
    #[test]
    fn clear_failed_downloads_preserves_other_entries_files_and_preferences() {
        let directory = tempfile::tempdir().unwrap();
        let config = configuration(directory.path());
        let state_file = config.with_file_name("desktop-state.json");
        let data = directory.path().join("models");
        std::fs::create_dir_all(&data).unwrap();
        let partial = data.join("core.incomplete");
        let installed = data.join("model.safetensors");
        std::fs::write(&partial, b"partial core data").unwrap();
        std::fs::write(&installed, b"installed core data").unwrap();
        let states = [FAILED, COMPLETE, FAILED, PAUSED, CANCELLED, QUEUED, RUNNING];
        let saved = Persistent {
            appearance: Appearance { compact: true, ..Default::default() },
            extra: serde_json::from_value(serde_json::json!({"retained_preference": 7})).unwrap(),
            downloads: states.iter().enumerate().map(|(index, state)| SavedDownload {
                installed_model: String::new(),
                id: index as u64 + 1,
                source: format!("publisher/model-{index}@gb10"),
                directory: data.clone(),
                state: *state,
                error: String::new(),
            }).collect(),
            ..Default::default()
        };
        saved.save(&state_file).unwrap();
        let app = Controller::new(Some(config.clone())).unwrap();
        wait_for(&app, |view| !view.busy && view.downloads.len() == states.len());
        // Restored transfers are paused; live transfer state must survive cleanup too.
        app.shared.update(|inner| {
            for (slot, state) in inner.downloads.iter_mut().zip(states) {
                slot.view.state = state;
                if matches!(state, QUEUED | RUNNING) {
                    inner.operations.insert(slot.view.operation_id, Operation::new());
                }
            }
        });
        app.submit(Command::ClearFailedDownloads).unwrap();
        let cleared = wait_for(&app, |view| {
            !view.busy && view.downloads.len() == states.len() - 2
        });
        assert!(cleared.error.is_empty());
        assert_eq!(
            cleared.downloads.iter().map(|download| download.state).collect::<Vec<_>>(),
            [COMPLETE, PAUSED, CANCELLED, QUEUED, RUNNING]
        );
        assert_eq!(
            cleared.downloads.iter().map(|download| download.operation_id).collect::<Vec<_>>(),
            [2, 4, 5, 6, 7]
        );
        {
            let inner = app.shared.inner.lock().unwrap();
            for id in [6, 7] {
                assert!(!inner.operations[&id].cancel.load(Ordering::Acquire));
            }
        }
        let persisted = Persistent::load(&state_file).unwrap();
        assert_eq!(
            persisted.downloads.iter().map(|download| download.id).collect::<Vec<_>>(),
            [2, 4, 5, 6, 7]
        );
        assert!(persisted.appearance.compact);
        assert_eq!(persisted.extra, saved.extra);
        assert_eq!(std::fs::read(&partial).unwrap(), b"partial core data");
        assert_eq!(std::fs::read(&installed).unwrap(), b"installed core data");
        close(&app);
        drop(app);
        let restored = Controller::new(Some(config)).unwrap();
        assert_eq!(restored.snapshot().downloads.len(), states.len() - 2);
        assert!(restored.snapshot().downloads.iter().all(|download| download.state != FAILED));
        close(&restored);
    }
    #[test]
    fn clear_failed_downloads_keeps_history_when_persistence_fails() {
        let directory = tempfile::tempdir().unwrap();
        let config = configuration(directory.path());
        let state_file = config.with_file_name("desktop-state.json");
        let saved = Persistent {
            downloads: vec![SavedDownload {
                installed_model: String::new(),
                id: 1,
                source: "publisher/model@gb10".into(),
                directory: directory.path().join("models"),
                state: FAILED,
                error: "Transfer failed".into(),
            }],
            ..Default::default()
        };
        saved.save(&state_file).unwrap();
        let app = Controller::new(Some(config)).unwrap();
        wait_for(&app, |view| !view.busy && view.downloads.len() == 1);
        std::fs::remove_file(&state_file).unwrap();
        std::fs::create_dir(&state_file).unwrap();
        app.submit(Command::ClearFailedDownloads).unwrap();
        let failed = wait_for(&app, |view| !view.busy && !view.error.is_empty());
        assert_eq!(failed.downloads.len(), 1);
        assert_eq!(failed.downloads[0].state, FAILED);
        assert_eq!(failed.downloads[0].error, "Transfer failed");
        std::fs::remove_dir(&state_file).unwrap();
        saved.save(&state_file).unwrap();
        app.submit(Command::ClearFailedDownloads).unwrap();
        wait_for(&app, |view| !view.busy && view.downloads.is_empty());
        assert!(Persistent::load(&state_file).unwrap().downloads.is_empty());
        close(&app);
    }
    #[test]
    fn loading_observation_exposes_stable_stage_and_measured_progress() {
        let directory = tempfile::tempdir().unwrap();
        let controller = Controller::new(Some(configuration(directory.path()))).unwrap();
        wait_for(&controller, |s| {
            !s.busy && directory.path().join("models").exists()
        });
        let (progress, observation) = StartupProgress::observed();
        controller.shared.update(|s| s.startup = Some(observation));
        let phase = progress.phase_named(
            flare_helios::startup::StartupStage::LoadWeights,
            "Loading model data",
            Some(100),
        );
        phase.advance(24);
        let sample = controller.snapshot();
        assert_eq!(sample.stage, "load_weights");
        assert_eq!(sample.stage_completed, 24);
        assert_eq!(sample.stage_total, Some(100));
        phase.finish("Model data loaded");
        close(&controller);
    }

    #[test]
    fn preview_requires_a_ready_owned_model_and_rejects_after_close() {
        let directory = tempfile::tempdir().unwrap();
        let controller = Controller::new(Some(configuration(directory.path()))).unwrap();
        assert!(controller.try_model("Hello".into()).is_err());
        assert!(!controller.snapshot().preview.busy);
        controller.clear_chat().unwrap();
        close(&controller);
        assert!(
            controller
                .try_model("Hello".into())
                .unwrap_err()
                .contains("closing")
        );
        assert!(controller.clear_chat().unwrap_err().contains("closing"));
    }

    #[test]
    fn completed_download_keeps_the_exact_core_choice_across_restart() {
        let directory = tempfile::tempdir().unwrap();
        let config = configuration(directory.path());
        let selected = "local:example-model@gb10-tp1-fixture";
        Persistent {
            downloads: vec![SavedDownload {
                id: 19,
                source: "example/model@revision".into(),
                directory: directory.path().join("models"),
                state: COMPLETE,
                error: String::new(),
                installed_model: selected.into(),
            }],
            ..Persistent::default()
        }
        .save(&config.with_file_name("desktop-state.json"))
        .unwrap();
        let controller = Controller::new(Some(config.clone())).unwrap();
        let view = controller.snapshot();
        assert_eq!(view.downloads[0].state, COMPLETE);
        assert!(view.downloads[0].installed_model.is_empty());
        assert!(!view.downloads[0].can_resume);
        close(&controller);
        assert_eq!(Persistent::load(&config.with_file_name("desktop-state.json")).unwrap().downloads[0].installed_model, selected);
    }
    #[test]
    fn model_directory_changes_survive_controller_restart_and_close_is_cooperative() {
        let directory = tempfile::tempdir().unwrap();
        let config = configuration(directory.path());
        let controller = Controller::new(Some(config.clone())).unwrap();
        wait_for(&controller, |s| {
            !s.busy && s.error.is_empty() && directory.path().join("models").exists()
        });
        let selected = directory.path().join("selected-library");
        std::fs::create_dir_all(&selected).unwrap();
        controller
            .submit(Command::SetModelDirectory(selected.clone()))
            .unwrap();
        let selected = selected.canonicalize().unwrap_or(selected);
        wait_for(&controller, |s| {
            s.model_directory == selected.display().to_string() && !s.busy
        });
        controller
            .submit(Command::SetAppearance {
                theme: 1,
                compact: true,
            })
            .unwrap();
        wait_for(&controller, |s| {
            s.appearance.theme == 1 && s.appearance.compact && !s.busy
        });
        let started = Instant::now();
        controller.close();
        assert!(started.elapsed() < Duration::from_millis(100));
        close(&controller);
        assert!(controller.submit(Command::Refresh).is_err());
        drop(controller);
        let restored = Controller::new(Some(config)).unwrap();
        assert_eq!(
            restored.snapshot().model_directory,
            selected.display().to_string()
        );
        assert_eq!(restored.snapshot().appearance.theme, 1);
        assert!(restored.snapshot().appearance.compact);
        close(&restored);
    }
    #[test]
    fn malformed_external_cores_remain_visible_after_refresh_and_restart() {
        let directory = tempfile::tempdir().unwrap();
        let config = configuration(directory.path());
        let core = directory.path().join("invalid.safetensors");
        std::fs::write(&core, b"invalid core").unwrap();
        let controller = Controller::new(Some(config.clone())).unwrap();
        controller.submit(Command::OpenModel(core.clone())).unwrap();
        let view = wait_for(&controller, |s| {
            !s.busy
                && s.models
                    .iter()
                    .any(|m| m.flags & 2 != 0 && !m.error.is_empty())
        });
        assert_eq!(view.state, SessionState::Failed);
        assert!(view.trust.is_none());
        assert!(!view.has_metrics);
        close(&controller);
        drop(controller);
        let restored = Controller::new(Some(config)).unwrap();
        let view = wait_for(&restored, |s| s.models.iter().any(|model| model.flags & 2 != 0) && !s.busy);
        let external: Vec<_> = view.models.iter().filter(|model| model.flags & 2 != 0).collect();
        assert_eq!(external.len(), 1);
        assert!(!external[0].error.is_empty());
        close(&restored);
    }
    #[test]
    fn consent_for_an_expired_review_cannot_start_a_session() {
        let directory = tempfile::tempdir().unwrap();
        let controller = Controller::new(Some(configuration(directory.path()))).unwrap();
        controller
            .submit(Command::AcceptTrust {
                decision: 91,
                publisher: "publisher".into(),
            })
            .unwrap();
        let view = wait_for(&controller, |s| !s.busy && !s.error.is_empty());
        assert!(!view.has_metrics);
        assert_eq!(view.current_operation, 0);
        assert!(view.trust.is_none());
        close(&controller);
    }

    #[test]
    fn restart_requirement_survives_commands_and_cooperative_cleanup() {
        let directory = tempfile::tempdir().unwrap();
        let controller = Controller::new(Some(configuration(directory.path()))).unwrap();
        wait_for(&controller, |s| {
            !s.busy && directory.path().join("models").exists()
        });
        controller
            .shared
            .require_restart("Drain was not confirmed".into());
        controller.submit(Command::Refresh).unwrap();
        let view = wait_for(&controller, |s| !s.busy && s.current_operation == 0);
        assert_eq!(view.state, SessionState::RestartRequired);
        assert_eq!(view.error, "Drain was not confirmed");
        close(&controller);
        assert_eq!(controller.snapshot().state, SessionState::RestartRequired);
    }

    #[test]
    fn rejected_stop_does_not_cancel_a_running_operation() {
        let operation = Operation::new();
        let shared = Arc::new(Shared {
            hardware: crate::hardware::HardwareMonitor::start().unwrap(),
            preview: crate::preview::Preview::new(None).unwrap(),
            inner: Mutex::new(Inner {
                view: Snapshot::default(),
                active: None,
                startup: None,
                operations: HashMap::from([(7, operation.clone())]),
                downloads: Vec::new(),
                runtime_operation: Some(7),
            }),
            closing: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            next: AtomicU64::new(8),
            dashboard: Mutex::new(dashboard::DashboardHistory::default()),
            restart_required: AtomicBool::new(false),
        });
        let (sender, _receiver) = mpsc::sync_channel(0);
        let controller = Controller {
            shared,
            sender,
            worker: Mutex::new(None),
        };
        assert!(controller.submit(Command::Stop).is_err());
        assert!(!operation.cancel.load(Ordering::Acquire));
    }
}

impl Worker {
    fn refresh(&mut self) -> Result<(), String> {
        let store =
            ModelStore::open(self.settings.model_directory.clone()).map_err(|e| e.to_string())?;
        let entries = store.inventory().map_err(|e| e.to_string())?;
        let mut models = Vec::new();
        let mut sources = HashMap::new();
        for entry in &entries {
            let id = format!("local:{}@{}", entry.model_key, entry.version);
            let source = Source::LocalModel {
                model: entry.model_key.clone(),
                version: Some(entry.version.clone()),
            };
            let path = store
                .resolve(&source)
                .ok()
                .and_then(|r| r.path.canonicalize().ok())
                .map(|path| path.display().to_string())
                .unwrap_or_default();
            models.push(ModelView {
                id: id.clone(),
                name: recommendations::local_name(&entry.model, &entry.hub_repositories),
                version: entry.version.clone(),
                source: "Local library".into(),
                path,
                description: if entry.selected {
                    "Selected core version".into()
                } else {
                    "Prepared core version".into()
                },
                error: entry.error.clone().unwrap_or_default(),
                logical_bytes: entry
                    .metadata
                    .as_ref()
                    .and_then(|m| m["logical_bytes"].as_u64()),
                flags: 1 | if recommendations::is_recommended(&entry.hub_repositories) { recommendations::RECOMMENDED } else { 0 },
                metadata: local_model_metadata(entry),
            });
            sources.insert(id, source);
        }
        for path in &self.persistent.external {
            let id = external_id(path);
            let source = Source::LocalPath(path.clone());
            let inspection = store.inspect(
                &source,
                &crate::application::InspectOptions {
                    unverified: true,
                    ..Default::default()
                },
            );
            let error = inspection
                .as_ref()
                .err()
                .map(ToString::to_string)
                .unwrap_or_default();
            models.push(ModelView {
                id: id.clone(),
                name: path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into(),
                source: "External core".into(),
                path: path.canonicalize().unwrap_or_else(|_| path.clone()).display().to_string(),
                description: "Local core reference".into(),
                error,
                flags: 2,
                ..ModelView::default()
            });
            sources.insert(id, source);
        }
        let catalog = catalog_with_library(
            &recommendations::catalog(&self.catalog),
            &entries,
            &models,
            self.active.as_ref().and_then(|active| active_id_in_models(&active.id, &active.path, &models)),
        );
        models.extend(catalog);
        self.sources = sources;
        let visible_downloads: HashMap<_, _> = self.persistent.downloads.iter().map(|saved| {
            (saved.id, visible_download_model(saved, &self.settings.model_directory, &self.sources))
        }).collect();
        self.shared.update(|s| {
            s.view.models = models;
            for download in &mut s.downloads {
                download.view.installed_model = visible_downloads.get(&download.view.operation_id)
                    .cloned()
                    .unwrap_or_default();
            }
        });
        Ok(())
    }
    fn clear_review(&mut self) {
        if let Some(pending) = self.pending.take() {
            self.shared.update(|s| {
                s.operations.remove(&pending.operation);
                if s.view.current_operation == pending.operation {
                    s.view.current_operation = 0;
                }
            });
        }
        let state = self.idle_state();
        self.shared.update(|s| {
            s.view.trust = None;
            if s.view.state == SessionState::ReviewRequired {
                s.view.state = state;
                s.view.detail = "Publisher review dismissed".into();
            }
        });
    }
    fn begin_review(
        &mut self,
        operation: u64,
        id: String,
        name: String,
        source: Source,
    ) -> Result<(), String> {
        if self
            .shared
            .inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .view
            .state
            == SessionState::RestartRequired
        {
            return Err(
                "Restart Flare before loading another model because GPU drain was not confirmed"
                    .into(),
            );
        }
        self.clear_review();
        self.state(SessionState::Loading, "Authenticate model metadata");
        let store =
            ModelStore::open(self.settings.model_directory.clone()).map_err(|e| e.to_string())?;
        let review = store
            .review_trust(
                &source,
                &TrustOptions {
                    publisher_keys: self.settings.publisher_keys.clone(),
                    hub: self.settings.hub.clone(),
                    trust_store: self.settings.trust_store.clone(),
                    tensor_parallel: Some(1),
                    rank: None,
                },
            )
            .map_err(|e| e.to_string())?;
        if self.shared.cancelled(operation) {
            self.state(self.idle_state(), "Model activation cancelled");
            return Ok(());
        }
        if review.disclosure.remembered {
            return self.authorize_and_load(operation, id, name, review);
        }
        let disclosure = &review.disclosure;
        let trust = TrustView {
            decision_id: operation,
            publisher: disclosure.namespace.clone(),
            fingerprint: disclosure.fingerprint.clone(),
            official_source: disclosure.official_source.clone(),
            license: match (&disclosure.license_name, &disclosure.license_url) {
                (Some(name), Some(url)) => format!("{name} ({url})"),
                (Some(name), None) => name.clone(),
                (None, Some(url)) => url.clone(),
                _ => "No license declared in the core".into(),
            },
            identity_status: match &disclosure.identity {
                crate::trust::Identity::Verified => {
                    "Publisher repository confirmed this signing key".into()
                }
                crate::trust::Identity::Unverified(reason) => {
                    format!("Local signing profile; repository identity unconfirmed: {reason}")
                }
                crate::trust::Identity::Development => "Local development publisher".into(),
            },
        };
        self.pending = Some(PendingReview {
            decision: operation,
            operation,
            id,
            name,
            review,
        });
        self.shared.update(|s| {
            s.view.state = SessionState::ReviewRequired;
            s.view.detail = "Review the publisher and license before loading".into();
            s.view.trust = Some(trust);
        });
        Ok(())
    }
    fn authorize_and_load(
        &mut self,
        operation: u64,
        id: String,
        name: String,
        review: TrustReview,
    ) -> Result<(), String> {
        if self.shared.restart_required.load(Ordering::Acquire) {
            return Err(
                "Restart Flare before loading another model because GPU drain was not confirmed"
                    .into(),
            );
        }
        let (progress, observation) = StartupProgress::observed();
        self.shared.update(|s| {
            s.startup = Some(observation.clone());
            s.view.state = SessionState::Loading;
            s.view.detail = "Verify core contents".into();
        });
        let consent = TrustAcceptance {
            fingerprint: review.disclosure.fingerprint.clone(),
            namespace: review.disclosure.namespace.clone(),
        };
        let authorized = review
            .accept(&consent, self.settings.runtime.verify_weights, true, Some(&progress))
            .map_err(|e| e.to_string())?;
        if self.shared.cancelled(operation) {
            self.state(self.idle_state(), "Model activation cancelled");
            return Ok(());
        }
        if flare_helios::runtime_core::core_cluster(&authorized.core).is_some()
            || flare_helios::runtime_core::rank_needs(&authorized.core).len() != 1
        {
            return Err(
                "The desktop application requires a core prepared for one local GPU".into(),
            );
        }
        let names = authorized.core.model_names().map_err(|e| e.to_string())?;
        let options = self
            .settings
            .load_options(&names)
            .map_err(|e| e.to_string())?;
        let runtime = RuntimeIdentity::new(&self.settings.runtime, &options)?;
        let model_names: Vec<String> = names.iter().map(|name| (*name).to_owned()).collect();
        let path = authorized.path.canonicalize().map_err(|e| e.to_string())?.display().to_string();
        if let Some(active) = &self.active {
            self.state(
                SessionState::Draining,
                "Finish admitted requests before switching models",
            );
            active.session.server().pause_admission();
            while !active
                .session
                .server()
                .wait_for_requests(Duration::from_millis(100))
            {
                if self.shared.cancelled(operation) {
                    active.session.server().resume_admission();
                    self.state(
                        if active.session.server().healthy() {
                            SessionState::Ready
                        } else {
                            SessionState::Failed
                        },
                        "Model switch cancelled",
                    );
                    return Ok(());
                }
                if !active.session.server().healthy() {
                    break;
                }
            }
            if self.shared.cancelled(operation) {
                active.session.server().resume_admission();
                self.state(
                    if active.session.server().healthy() {
                        SessionState::Ready
                    } else {
                        SessionState::Failed
                    },
                    "Model switch cancelled",
                );
                return Ok(());
            }
        }
        if !self.stop_active() {
            return Err(
                "GPU drain was not confirmed; restart Flare before loading another model".into(),
            );
        }
        if self.shared.cancelled(operation) {
            self.state(SessionState::Idle, "Model activation cancelled");
            return Ok(());
        }
        self.state(SessionState::Loading, "Load model resources");
        self.shared.update(|s| s.startup = Some(observation));
        let AuthorizedModel { core, lease, .. } = authorized;
        let mut lease = LeaseGuard {
            lease,
            drained: false,
        };
        // SAFETY: Publisher consent authorizes running a trusted prepared core.
        // Its programs must obey the loader's CUDA contract; signatures do not
        // establish memory safety. The owned session retains resources through drain.
        let owner = match unsafe { NativeModel::load(core, &[0], &options, &progress) } {
            Ok(owner) => owner,
            Err(error) => {
                // The loader does not return drain evidence. Keep the core
                // lease on this error path as well as during unwinding.
                self.state(
                    SessionState::RestartRequired,
                    "Model resource loading failed without drain confirmation",
                );
                return Err(error.to_string());
            }
        };
        let phase = progress.phase("Load tokenizer", None);
        let mut config = owner.server_config().clone();
        config["thinking_token_budget"] = serde_json::json!(self.settings.thinking_token_budget);
        let mut loaded = match LoadedSession::new(owner, config) {
            Ok(loaded) => loaded,
            Err(error) => {
                lease.drained = true;
                return Err(error.to_string());
            }
        };
        loaded.server_config_mut().bind = self.settings.runtime.bind()?;
        loaded.server_config_mut().api_key = self.settings.api_key.clone();
        phase.finish("Tokenizer loaded");
        loaded
            .server_config_mut()
            .served_names
            .push("flare-local".into());
        let phase = progress.phase("Start server", None);
        let session = match loaded.start() {
            Ok(session) => session,
            Err(failure) => {
                lease.drained = failure.drained;
                if !failure.drained {
                    self.state(SessionState::RestartRequired, "GPU drain was not confirmed");
                }
                return Err(failure.message.to_string());
            }
        };
        phase.finish(format!("http://{}", session.server().address()));
        let active = Arc::new(ActiveSession {
            runtime,
            model_names,
            session,
            lease: Mutex::new(lease),
            id,
            name,
            path,
        });
        self.active = Some(active.clone());
        self.shared.update(|s| {
            s.active = Some(active);
            s.view.runtime_settings_pending_reload = false;
            s.startup = None;
        });
        if self.shared.cancelled(operation) {
            self.stop_active();
            if self.active.is_none() {
                let restart = self
                    .shared
                    .inner
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .view
                    .state
                    == SessionState::RestartRequired;
                if !restart {
                    self.state(SessionState::Idle, "Model activation cancelled");
                }
            }
        } else {
            self.state(SessionState::Ready, "Serving local requests");
        }
        Ok(())
    }
    fn stop_active(&mut self) -> bool {
        self.shared.update(|s| s.startup = None);
        let Some(active) = self.active.take() else {
            return self
                .shared
                .inner
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .view
                .state
                != SessionState::RestartRequired;
        };
        self.state(SessionState::Stopping, "Stop HTTP and drain GPU work");
        self.shared.update(|s| {
            s.active = None;
            s.view.runtime_settings_pending_reload = false;
        });
        self.shared.preview.cancel();
        while self.shared.preview.snapshot().busy {
            thread::sleep(Duration::from_millis(5));
        }
        let _ = self.shared.preview.clear();
        let shutdown = active.close();
        drop(active);
        if !shutdown.drained {
            self.shared.require_restart(
                shutdown
                    .error
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "GPU drain was not confirmed; restart Flare".into()),
            );
            self.clear_review();
            false
        } else {
            if let Some(error) = shutdown.error {
                self.shared.update(|s| s.view.error = error.to_string());
            }
            true
        }
    }
}

impl Worker {
    fn queue_download(
        &mut self,
        id: u64,
        source: String,
        directory: PathBuf,
    ) -> Result<(), String> {
        crate::hub::repository(&source).map_err(|e| e.to_string())?;
        let sender = self
            .download_sender
            .as_ref()
            .ok_or("The acquisition worker is unavailable")?;
        if self
            .persistent
            .downloads
            .iter()
            .filter(|d| matches!(d.state, QUEUED | RUNNING))
            .count()
            >= 32
        {
            return Err("The acquisition queue is full".into());
        }
        let saved = SavedDownload {
            installed_model: String::new(),
            id,
            source: source.clone(),
            directory,
            state: QUEUED,
            error: String::new(),
        };
        let mut persistent = self.persistent.clone();
        persistent.downloads.retain(|d| d.id != id);
        while persistent.downloads.len() >= 128 {
            let Some(index) = persistent
                .downloads
                .iter()
                .position(|d| !matches!(d.state, QUEUED | RUNNING))
            else {
                return Err("The acquisition history is full".into());
            };
            persistent.downloads.remove(index);
        }
        persistent.downloads.push(saved.clone());
        persistent.save(&self.settings.state_file)?;
        self.persistent = persistent;
        let progress = AcquisitionProgress::new();
        self.shared.update(|s| {
            s.downloads.retain(|d| {
                self.persistent
                    .downloads
                    .iter()
                    .any(|saved| saved.id == d.view.operation_id)
                    && d.view.operation_id != id
            });
            let mut control = s
                .operations
                .get(&id)
                .cloned()
                .unwrap_or_else(Operation::new);
            control.acquisition = Some(progress.clone());
            if control.cancel.load(Ordering::Acquire) {
                progress.request_cancel();
            }
            s.operations.insert(id, control);
            s.downloads.push(DownloadSlot {
                view: DownloadView {
                    operation_id: id,
                    state: QUEUED,
                    name: recommendations::acquisition_name(&source),
                    source,
                    phase: "Queued".into(),
                    can_cancel: true,
                    ..DownloadView::default()
                },
                progress: progress.clone(),
                rate: DownloadRate::default(),
            });
        });
        if let Err(error) = sender.try_send(DownloadJob {
            saved,
            settings: self.settings.clone(),
            progress,
        }) {
            self.finish_download(DownloadDone {
                id,
                result: Err(error.to_string()),
                cancelled: false,
            });
        }
        Ok(())
    }
    fn clear_failed_downloads(&mut self) -> Result<(), String> {
        let mut saved = self.persistent.clone();
        saved.downloads.retain(|download| download.state != FAILED);
        if saved.downloads.len() == self.persistent.downloads.len() {
            return Ok(());
        }
        saved.save(&self.settings.state_file)?;
        self.persistent = saved;
        self.shared
            .update(|state| state.downloads.retain(|slot| slot.view.state != FAILED));
        Ok(())
    }
    fn finish_download(&mut self, done: DownloadDone) {
        let paused = self
            .shared
            .inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .operations
            .get(&done.id)
            .is_some_and(|op| op.paused.load(Ordering::Acquire));
        let installed_model = done.result.as_ref().ok().cloned().unwrap_or_default();
        let state = if done.result.is_ok() {
            COMPLETE
        } else if done.cancelled && paused {
            PAUSED
        } else if done.cancelled {
            CANCELLED
        } else {
            FAILED
        };
        let error = if state == FAILED {
            done.result.err().unwrap_or_default()
        } else {
            String::new()
        };
        if let Some(saved) = self
            .persistent
            .downloads
            .iter_mut()
            .find(|d| d.id == done.id)
        {
            saved.state = state;
            saved.installed_model = installed_model.clone();
            saved.error = error.clone();
        }
        self.shared.update(|s| {
            s.operations.remove(&done.id);
            if let Some(slot) = s
                .downloads
                .iter_mut()
                .find(|d| d.view.operation_id == done.id)
            {
                slot.view.state = state;
                // Refresh exposes the completed choice only after matching its
                // acquisition root and managed entry to the selected library.
                slot.view.installed_model.clear();
                slot.view.error = error;
                slot.view.can_cancel = false;
                slot.view.can_resume = matches!(state, CANCELLED | FAILED | PAUSED);
                slot.view.phase = match state {
                    COMPLETE => "Installed",
                    PAUSED => "Paused; partial data saved",
                    CANCELLED => "Cancelled; partial data retained",
                    _ => "Failed",
                }
                .into();
            }
        });
        if let Err(error) = self.persistent.save(&self.settings.state_file) {
            self.shared
                .update(|s| s.view.error = format!("Cannot save acquisition state: {error}"));
        }
        if state == COMPLETE {
            if let Err(error) = self.refresh() {
                self.shared.update(|s| s.view.error = error);
            }
        }
    }
    fn collect_downloads(&mut self) {
        while let Ok(done) = self.download_done.try_recv() {
            self.finish_download(done);
        }
    }
    fn shutdown(&mut self) {
        self.shared.preview.close();
        self.shared.hardware.close();
        self.clear_review();
        self.shared.update(|s| {
            for control in s.operations.values() {
                control.cancel();
            }
        });
        self.stop_active();
        self.download_sender.take();
        if let Some(worker) = self.download_thread.take() {
            let _ = worker.join();
        }
        self.collect_downloads();
        while !self.shared.preview.is_closed() {
            thread::sleep(Duration::from_millis(5));
        }
        self.shared.update(|s| {
            s.startup = None;
            s.operations.clear();
            s.view.busy = false;
            s.view.catalog_loading = false;
            s.view.current_operation = 0;
            if s.view.state != SessionState::RestartRequired {
                s.view.state = SessionState::Idle;
            }
            s.view.detail = "Flare is closed".into();
        });
    }
}

impl Worker {
    fn inspect_model(&mut self, operation: u64, id: &str) -> Result<(), String> {
        let model = {
            let inner = self
                .shared
                .inner
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            inner
                .view
                .models
                .iter()
                .find(|model| model.id == id)
                .cloned()
        };
        let gpus = self.shared.hardware.snapshot().gpus;
        let model = model.ok_or("The model is no longer listed")?;
        let active = self.active.as_ref().is_some_and(|active| same_core(&model, &active.id, &active.path));
        self.shared.update(|state| state.view.detail_loading = true);
        let source = self.sources.get(id).cloned();
        let result = (|| -> Result<ModelDetailView, String> {
            let Some(source) = source.as_ref() else {
                return Ok(ModelDetailView {
                    id: id.into(),
                    name: model.name.clone(),
                    official_source: format!("https://huggingface.co/{}", model.source),
                    metadata_status: "Hub discovery entry; core metadata has not been inspected"
                        .into(),
                    compatibility_reason: "Download the core to inspect its recorded capabilities."
                        .into(),
                    ..Default::default()
                });
            };
            let store = ModelStore::open(self.settings.model_directory.clone())
                .map_err(|error| error.to_string())?;
            crate::details::inspect(
                &store,
                source,
                &TrustOptions {
                    publisher_keys: self.settings.publisher_keys.clone(),
                    hub: self.settings.hub.clone(),
                    trust_store: self.settings.trust_store.clone(),
                    ..Default::default()
                },
                id.into(),
                model.name.clone(),
                &gpus,
            )
            .map_err(|error| error.to_string())
        })();
        let mut detail = result.unwrap_or_else(|error| ModelDetailView {
            id: id.into(),
            name: model.name,
            path: model.path,
            metadata_status: "Core metadata is unavailable".into(),
            inspection_error: error,
            compatibility_reason: "The core could not be inspected.".into(),
            ..Default::default()
        });
        detail.can_remove_reference = matches!(source, Some(Source::LocalPath(_))) && !active;
        if active && matches!(source, Some(Source::LocalModel { .. })) && detail.signatures_verified
        {
            detail.compatibility = Compatibility::Compatible;
            detail.compatibility_reason = "This core is serving in this process.".into();
        }
        let cancelled = self.shared.cancelled(operation);
        self.shared.update(|state| {
            state.view.detail_loading = false;
            if !cancelled {
                state.view.model_detail = Some(detail);
            }
        });
        Ok(())
    }
    fn external_reference(&self, id: &str) -> Result<PathBuf, String> {
        if self.active.as_ref().is_some_and(|active| active.id == id) {
            return Err("Stop the model before changing its library reference".into());
        }
        let Some(Source::LocalPath(path)) = self.sources.get(id) else {
            return Err(
                "Only external core references can be removed or located with this action".into(),
            );
        };
        Ok(path.clone())
    }
    fn remove_reference(&mut self, id: &str) -> Result<(), String> {
        let path = self.external_reference(id)?;
        let mut saved = self.persistent.clone();
        saved.external.retain(|reference| reference != &path);
        saved.save(&self.settings.state_file)?;
        self.persistent = saved;
        if self.pending.as_ref().is_some_and(|review| review.id == id) {
            self.clear_review();
        }
        self.shared.update(|state| state.view.model_detail = None);
        self.refresh()
    }
    fn locate_reference(&mut self, id: &str, path: PathBuf) -> Result<(), String> {
        let previous = self.external_reference(id)?;
        let path = path
            .canonicalize()
            .map_err(|error| format!("Cannot locate core: {error}"))?;
        if !path.is_file() && !path.is_dir() {
            return Err("Select a core file or directory".into());
        }
        let mut saved = self.persistent.clone();
        saved
            .external
            .retain(|reference| reference != &previous && reference != &path);
        saved.external.push(path);
        saved.save(&self.settings.state_file)?;
        self.persistent = saved;
        if self.pending.as_ref().is_some_and(|review| review.id == id) {
            self.clear_review();
        }
        self.shared.update(|state| state.view.model_detail = None);
        self.refresh()
    }
}

#[cfg(test)]
mod runtime_settings_tests {
    use super::*;
    fn wait_settings(app:&Controller,operation:u64)->Snapshot {
        let start=std::time::Instant::now();
        loop {let view=app.snapshot();if view.last_settings_operation==operation&&!view.busy{return view;}assert!(start.elapsed()<Duration::from_secs(10));thread::sleep(Duration::from_millis(5));}
    }
    fn close(app:&Controller){app.close();let start=std::time::Instant::now();while !app.is_closed(){assert!(start.elapsed()<Duration::from_secs(10));thread::sleep(Duration::from_millis(5));}}
    #[test]
    fn runtime_save_is_atomic_persistent_and_never_rewrites_config_or_exposes_key() {
        let temp=tempfile::tempdir().unwrap();let config=temp.path().join("config.toml");
        let source=format!("[server]\napi_key='retain-this-key'\n[paths]\ncore_cache='{}'\n[unrelated]\nvalue='preserved'\n",temp.path().join("cores").display().to_string().replace('\\',"/"));
        std::fs::write(&config,&source).unwrap();
        std::fs::write(config.with_file_name("desktop-state.json"),r#"{"unrelated_state":{"keep":7}}"#).unwrap();
        let app=Controller::new(Some(config.clone())).unwrap();
        let mut desired=app.snapshot().runtime_settings;
        assert!(!desired.verify_weights);assert!(!desired.prefix_cache_disk_enabled);
        desired.host="0.0.0.0".into();desired.port=9001;desired.memory_limit="0.8".into();
        let operation=app.submit(Command::SetRuntimeSettings(desired.clone())).unwrap();
        let saved=wait_settings(&app,operation);
        assert!(saved.settings_error.is_empty());assert_eq!(saved.runtime_settings,desired);
        assert!(!saved.runtime_settings_pending_reload);assert!(saved.has_api_key);
        assert_eq!(std::fs::read_to_string(&config).unwrap(),source);
        assert!(!std::fs::read_to_string(config.with_file_name("desktop-state.json")).unwrap().contains("retain-this-key"));
        let persisted: serde_json::Value=serde_json::from_str(&std::fs::read_to_string(config.with_file_name("desktop-state.json")).unwrap()).unwrap();
        assert_eq!(persisted["unrelated_state"]["keep"],7);
        app.shared.update(|s| s.view.error="Keep runtime error".into());
        let mut invalid=desired.clone();invalid.memory_limit="NaN".into();
        let operation=app.submit(Command::SetRuntimeSettings(invalid)).unwrap();
        let rejected=wait_settings(&app,operation);
        assert!(!rejected.settings_error.is_empty());assert_eq!(rejected.runtime_settings,desired);
        assert_eq!(rejected.state,saved.state);
        assert_eq!(rejected.error,"Keep runtime error");
        app.shared.update(|s| s.view.state=SessionState::Ready);
        let mut invalid=desired.clone();invalid.port=0;
        let operation=app.submit(Command::SetRuntimeSettings(invalid)).unwrap();
        let ready=wait_settings(&app,operation);
        assert_eq!(ready.state,SessionState::Ready);
        assert_eq!(ready.error,"Keep runtime error");
        assert_eq!(ready.runtime_settings,desired);
        close(&app);drop(app);
        let restored=Controller::new(Some(config)).unwrap();
        assert_eq!(restored.snapshot().runtime_settings,desired);assert!(restored.snapshot().has_api_key);
        close(&restored);
    }
    #[test]
    fn failed_state_publication_leaves_desired_settings_unchanged() {
        let temp=tempfile::tempdir().unwrap();let config=temp.path().join("config.toml");
        std::fs::write(&config,format!("[paths]\ncore_cache='{}'\n",temp.path().join("cores").display().to_string().replace('\\',"/"))).unwrap();
        let app=Controller::new(Some(config.clone())).unwrap();let before=app.snapshot().runtime_settings;
        std::fs::create_dir(config.with_file_name("desktop-state.json")).unwrap();
        let mut desired=before.clone();desired.port=9002;
        let operation=app.submit(Command::SetRuntimeSettings(desired)).unwrap();
        let failed=wait_settings(&app,operation);assert!(!failed.settings_error.is_empty());assert_eq!(failed.runtime_settings,before);
        close(&app);
    }
    #[test]
    fn client_endpoints_normalize_only_wildcards() {
        assert_eq!(client_address("0.0.0.0:9000".parse().unwrap()).to_string(),"127.0.0.1:9000");
        assert_eq!(client_address("[::]:9000".parse().unwrap()).to_string(),"[::1]:9000");
        assert_eq!(client_address("192.168.1.5:9000".parse().unwrap()).to_string(),"192.168.1.5:9000");
    }
}

#[cfg(test)]
mod download_root_tests {
    use super::*;
    #[test]
    fn download_choice_is_hidden_for_another_root_even_with_the_same_local_id() {
        let temp = tempfile::tempdir().unwrap();
        let root_a = temp.path().join("a");
        let root_b = temp.path().join("b");
        std::fs::create_dir(&root_a).unwrap();
        std::fs::create_dir(&root_b).unwrap();
        let id = "local:model-key@same-version";
        let saved = SavedDownload {
            installed_model: id.into(),
            id: 7,
            source: "publisher/model".into(),
            directory: root_a.clone(),
            state: COMPLETE,
            error: String::new(),
        };
        let mut sources = HashMap::from([(
            id.into(),
            Source::LocalModel {
                model: "model-key".into(),
                version: Some("same-version".into()),
            },
        )]);
        assert_eq!(visible_download_model(&saved, &root_a, &sources), id);
        assert!(visible_download_model(&saved, &root_b, &sources).is_empty());
        assert_eq!(visible_download_model(&saved, &root_a, &sources), id);
        sources.clear();
        assert!(visible_download_model(&saved, &root_a, &sources).is_empty());
        assert_eq!(saved.installed_model, id);
        assert_eq!(saved.directory, root_a);
    }
}

#[cfg(test)]
mod download_root_workflow_tests {
    use super::*;
    use flare_helios::core::{
        cache::{CoreSource, Root},
        reader,
    };
    fn wait(app: &Controller, ready: impl Fn(&Snapshot) -> bool) -> Snapshot {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            let view = app.snapshot();
            if !view.busy && ready(&view) {
                return view;
            }
            assert!(std::time::Instant::now() < deadline, "{view:?}");
            thread::sleep(Duration::from_millis(5));
        }
    }
    #[test]
    fn library_switch_hides_a_completed_choice_on_id_collision_and_restores_it() {
        let temp = tempfile::tempdir().unwrap();
        let (core, _, publisher, _) = crate::hub::tests::signed_payload_fixture(temp.path());
        let root_a = temp.path().join("a");
        let root_b = temp.path().join("b");
        let mut ids = Vec::new();
        for path in [&root_a, &root_b] {
            let store = Root::open(path).unwrap();
            let metadata = reader::Metadata::open(&core).unwrap();
            let selections = crate::cache_support::select_deployments(
                &metadata,
                std::slice::from_ref(&publisher),
                &[1],
            )
            .unwrap();
            let source = CoreSource::open(&core, std::slice::from_ref(&publisher), selections[0])
                .unwrap()
                .union(std::slice::from_ref(&publisher), &selections)
                .unwrap();
            let entry = store
                .install_core_staged(source, "model-key", "same-version")
                .unwrap();
            ids.push(format!("local:{}@{}", entry.model(), entry.id()));
        }
        assert_eq!(ids[0], ids[1]);
        let selected = ids[0].clone();
        let config = temp.path().join("config.toml");
        std::fs::write(
            &config,
            format!(
                "[paths]\ncore_cache='{}'\n",
                root_a.display().to_string().replace('\\', "/")
            ),
        )
        .unwrap();
        Persistent {
            downloads: vec![SavedDownload {
                installed_model: selected.clone(),
                id: 17,
                source: "publisher/model".into(),
                directory: root_a.clone(),
                state: COMPLETE,
                error: String::new(),
            }],
            ..Default::default()
        }
        .save(&config.with_file_name("desktop-state.json"))
        .unwrap();
        let app = Controller::new(Some(config.clone())).unwrap();
        let a = wait(&app, |s| s.downloads[0].installed_model == selected);
        let active_path = a
            .models
            .iter()
            .find(|model| model.id == selected)
            .unwrap()
            .path
            .clone();
        assert_eq!(
            active_id_in_models(&selected, &active_path, &a.models),
            Some(selected.as_str())
        );
        app.submit(Command::SetModelDirectory(root_b.clone()))
            .unwrap();
        let b = wait(&app, |s| {
            s.model_directory == root_b.canonicalize().unwrap().display().to_string()
        });
        assert!(b.models.iter().any(|model| model.id == selected));
        assert!(b.downloads[0].installed_model.is_empty());
        assert_eq!(
            active_id_in_models(&selected, &active_path, &b.models),
            None
        );
        assert!(
            b.models
                .iter()
                .filter(|model| model.id == selected)
                .all(|model| !same_core(model, &selected, &active_path))
        );
        app.submit(Command::SetModelDirectory(root_a.clone()))
            .unwrap();
        let restored = wait(&app, |s| s.downloads[0].installed_model == selected);
        assert_eq!(
            active_id_in_models(&selected, &active_path, &restored.models),
            Some(selected.as_str())
        );
        assert_eq!(
            Persistent::load(&config.with_file_name("desktop-state.json"))
                .unwrap()
                .downloads[0]
                .installed_model,
            selected
        );
        app.close();
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while !app.is_closed() {
            assert!(std::time::Instant::now() < deadline);
            thread::sleep(Duration::from_millis(5));
        }
    }
}
