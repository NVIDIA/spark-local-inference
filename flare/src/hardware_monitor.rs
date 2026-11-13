//! A bounded NVML sampler whose owner wakes and joins its worker.
use super::{GpuInfo, inventory};
use std::{
    sync::{Arc, Mutex, mpsc},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
#[derive(Clone, Debug, Default)]
pub struct HardwareSnapshot {
    pub gpus: Vec<GpuInfo>,
    pub error: String,
    pub sample_age_ms: Option<u64>,
}
struct StoredSample {
    completed: Instant,
    result: Result<Vec<GpuInfo>, String>,
}
/// Queries NVML off application and UI threads, with at most one query in flight.
pub struct HardwareMonitor {
    latest: Arc<Mutex<Option<StoredSample>>>,
    stop: mpsc::Sender<()>,
    worker: Mutex<Option<JoinHandle<()>>>,
}
impl HardwareMonitor {
    /// Start an immediate sample followed by one sample every two seconds.
    pub fn start() -> std::io::Result<Self> {
        Self::start_with(Duration::from_secs(2), || {
            inventory().map_err(|error| error.to_string())
        })
    }
    fn start_with(
        interval: Duration,
        mut sample: impl FnMut() -> Result<Vec<GpuInfo>, String> + Send + 'static,
    ) -> std::io::Result<Self> {
        let latest = Arc::new(Mutex::new(None));
        let output = latest.clone();
        let (stop, receiver) = mpsc::channel();
        let interval = interval.max(Duration::from_millis(1));
        let worker = thread::Builder::new()
            .name("flare-hardware".into())
            .spawn(move || {
                loop {
                    if !matches!(receiver.try_recv(), Err(mpsc::TryRecvError::Empty)) {
                        break;
                    }
                    let started = Instant::now();
                    let sampled =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(&mut sample));
                    let failed = sampled.is_err();
                    let result = sampled
                        .unwrap_or_else(|_| Err("The NVIDIA hardware sampler failed".into()));
                    *output.lock().unwrap_or_else(|error| error.into_inner()) =
                        Some(StoredSample {
                            completed: Instant::now(),
                            result,
                        });
                    if failed {
                        break;
                    }
                    match receiver.recv_timeout(interval.saturating_sub(started.elapsed())) {
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                        Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                }
            })?;
        Ok(Self {
            latest,
            stop,
            worker: Mutex::new(Some(worker)),
        })
    }
    /// Clone the latest completed sample without querying the driver.
    pub fn snapshot(&self) -> HardwareSnapshot {
        let latest = self
            .latest
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let Some(sample) = latest.as_ref() else {
            return HardwareSnapshot::default();
        };
        let sample_age_ms =
            Some(u64::try_from(sample.completed.elapsed().as_millis()).unwrap_or(u64::MAX));
        match &sample.result {
            Ok(gpus) => HardwareSnapshot {
                gpus: gpus.clone(),
                error: String::new(),
                sample_age_ms,
            },
            Err(error) => HardwareSnapshot {
                gpus: Vec::new(),
                error: error.clone(),
                sample_age_ms,
            },
        }
    }
    /// Wake an idle worker and join it. An in-flight synchronous NVML query finishes first.
    pub fn close(&self) {
        let mut worker = self
            .worker
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(worker) = worker.take() {
            let _ = self.stop.send(());
            let _ = worker.join();
        }
    }
}
impl Drop for HardwareMonitor {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    fn wait(
        monitor: &HardwareMonitor,
        ready: impl Fn(&HardwareSnapshot) -> bool,
    ) -> HardwareSnapshot {
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            let snapshot = monitor.snapshot();
            if ready(&snapshot) {
                return snapshot;
            }
            assert!(Instant::now() < deadline, "Timed out: {snapshot:?}");
            thread::sleep(Duration::from_millis(2));
        }
    }
    #[test]
    fn failed_samples_replace_and_clear_successful_gpu_data() {
        let fail = Arc::new(AtomicBool::new(false));
        let source = fail.clone();
        let monitor = HardwareMonitor::start_with(Duration::from_millis(10), move || {
            if source.load(Ordering::Acquire) {
                Err("driver unavailable".into())
            } else {
                Ok(vec![GpuInfo {
                    name: "detected GPU".into(),
                    ..Default::default()
                }])
            }
        })
        .unwrap();
        wait(&monitor, |snapshot| snapshot.gpus.len() == 1);
        fail.store(true, Ordering::Release);
        let snapshot = wait(&monitor, |snapshot| !snapshot.error.is_empty());
        assert!(snapshot.gpus.is_empty());
        assert_eq!(snapshot.error, "driver unavailable");
        assert!(snapshot.sample_age_ms.is_some());
        monitor.close();
    }
    #[test]
    fn stop_wakes_a_long_interval_and_joins_before_returning() {
        let calls = Arc::new(AtomicUsize::new(0));
        let source = calls.clone();
        let monitor = HardwareMonitor::start_with(Duration::from_secs(60), move || {
            source.fetch_add(1, Ordering::SeqCst);
            Ok(Vec::new())
        })
        .unwrap();
        let first = wait(&monitor, |snapshot| snapshot.sample_age_ms.is_some());
        thread::sleep(Duration::from_millis(15));
        assert!(monitor.snapshot().sample_age_ms.unwrap() >= first.sample_age_ms.unwrap() + 10);
        let started = Instant::now();
        monitor.close();
        assert!(started.elapsed() < Duration::from_secs(1));
        monitor.close();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn dropping_the_owner_joins_its_worker() {
        let alive = Arc::new(AtomicUsize::new(1));
        struct Lifetime(Arc<AtomicUsize>);
        impl Drop for Lifetime {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::SeqCst);
            }
        }
        let lifetime = Lifetime(alive.clone());
        let monitor = HardwareMonitor::start_with(Duration::from_secs(60), move || {
            let _ = &lifetime;
            Ok(Vec::new())
        })
        .unwrap();
        wait(&monitor, |snapshot| snapshot.sample_age_ms.is_some());
        drop(monitor);
        assert_eq!(alive.load(Ordering::SeqCst), 0);
    }
}
