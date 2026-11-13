/* SPDX-License-Identifier: Apache-2.0 */
//! Server-side admission-to-output timing; this does not measure client receipt.
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
const WINDOW: Duration = Duration::from_secs(60);
const CAPACITY: usize = 4096;

/// HTTP admission to the first answer or reasoning text offered by the server.
/// Streaming records the first eligible body frame; nonstreaming records the
/// completed JSON body. This does not measure receipt or rendering by a client.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FirstResponseSnapshot {
    /// Median of retained observations, absent when no eligible output occurred.
    pub median_seconds: Option<f64>,
    /// Retained observations whose first offered text falls inside the window.
    pub samples: u64,
    /// Rolling retention interval, measured with a monotonic clock.
    pub window_seconds: u64,
    /// Maximum retained observations, keeping the most recent when exceeded.
    pub capacity: usize,
    /// Eligible samples were evicted before their window expired.
    pub truncated: bool,
}
#[derive(Default)]
struct Samples {
    values: VecDeque<(Instant, Duration)>,
    dropped_at: Option<Instant>,
}
#[derive(Default)]
pub(super) struct FirstResponseSamples(Mutex<Samples>);
impl Samples {
    fn expire(&mut self, now: Instant) {
        while self
            .values
            .front()
            .is_some_and(|(at, _)| now.saturating_duration_since(*at) >= WINDOW)
        {
            self.values.pop_front();
        }
        if self
            .dropped_at
            .is_some_and(|at| now.saturating_duration_since(at) >= WINDOW)
        {
            self.dropped_at = None;
        }
    }
    fn record(&mut self, start: Instant, now: Instant) {
        self.expire(now);
        if self.values.len() == CAPACITY {
            self.dropped_at = self.values.pop_front().map(|(at, _)| at);
        }
        self.values
            .push_back((now, now.saturating_duration_since(start)));
    }
}
impl FirstResponseSamples {
    pub(super) fn observe(self: &Arc<Self>, started: Instant) -> FirstResponse {
        FirstResponse {
            samples: self.clone(),
            started,
            recorded: false,
        }
    }
    pub(super) fn snapshot(&self) -> FirstResponseSnapshot {
        let (mut seconds, truncated) = {
            let mut samples = self.0.lock().unwrap_or_else(|e| e.into_inner());
            samples.expire(Instant::now());
            (
                samples
                    .values
                    .iter()
                    .map(|(_, latency)| latency.as_secs_f64())
                    .collect::<Vec<_>>(),
                samples.dropped_at.is_some(),
            )
        };
        seconds.sort_unstable_by(f64::total_cmp);
        let count = seconds.len();
        let median = if count == 0 {
            None
        } else if count % 2 == 0 {
            Some((seconds[count / 2 - 1] + seconds[count / 2]) / 2.0)
        } else {
            Some(seconds[count / 2])
        };
        FirstResponseSnapshot {
            median_seconds: median,
            samples: count as u64,
            window_seconds: WINDOW.as_secs(),
            capacity: CAPACITY,
            truncated,
        }
    }
}
pub(super) struct FirstResponse {
    samples: Arc<FirstResponseSamples>,
    started: Instant,
    recorded: bool,
}
impl FirstResponse {
    pub(super) fn output(&mut self) {
        if !self.recorded {
            let mut samples = self.samples.0.lock().unwrap_or_else(|e| e.into_inner());
            samples.record(self.started, Instant::now());
            self.recorded = true;
        }
    }
}

/// Only answer/reasoning text is eligible. Roles, keepalives, token observations,
/// usage, signatures and tool-only content do not represent visible text here.
pub(super) fn stream_text(value: &serde_json::Value) -> bool {
    let text = |v: &serde_json::Value| v.as_str().is_some_and(|s| !s.is_empty());
    if value["choices"].as_array().is_some_and(|choices| {
        choices.iter().any(|c| {
            text(&c["text"])
                || text(&c["delta"]["content"])
                || text(&c["delta"]["reasoning_content"])
                || text(&c["delta"]["reasoning"])
        })
    }) {
        return true;
    }
    match value["type"].as_str() {
        Some("content_block_delta") => {
            text(&value["delta"]["text"]) || text(&value["delta"]["thinking"])
        }
        Some("content_block_start") => {
            text(&value["content_block"]["text"]) || text(&value["content_block"]["thinking"])
        }
        Some(
            "response.output_text.delta"
            | "response.reasoning_text.delta"
            | "response.reasoning_summary_text.delta",
        ) => text(&value["delta"]),
        _ => false,
    }
}
pub(super) fn response_text(value: &serde_json::Value) -> bool {
    let text = |v: &serde_json::Value| v.as_str().is_some_and(|s| !s.is_empty());
    let parts = |v: &serde_json::Value| {
        v.as_array().is_some_and(|parts| {
            parts
                .iter()
                .any(|p| text(&p["text"]) || text(&p["thinking"]))
        })
    };
    value["choices"].as_array().is_some_and(|choices| {
        choices.iter().any(|c| {
            text(&c["text"])
                || text(&c["message"]["content"])
                || text(&c["message"]["reasoning_content"])
                || text(&c["message"]["reasoning"])
                || parts(&c["message"]["content"])
        })
    }) || parts(&value["content"])
        || value["output"]
            .as_array()
            .is_some_and(|items| items.iter().any(|item| parts(&item["content"])))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn first_response_median_expires_without_inventing_zero() {
        let samples = FirstResponseSamples::default();
        let empty = samples.snapshot();
        assert_eq!(empty.median_seconds, None);
        assert_eq!(empty.samples, 0);
        let now = Instant::now();
        {
            let mut values = samples.0.lock().unwrap();
            for seconds in [4, 1, 2, 3] {
                values.values.push_back((now, Duration::from_secs(seconds)));
            }
        }
        assert_eq!(samples.snapshot().median_seconds, Some(2.5));
        samples.0.lock().unwrap().expire(now + WINDOW);
        assert_eq!(samples.snapshot().median_seconds, None);
    }

    #[test]
    fn first_response_bounds_samples_and_reports_window_truncation() {
        let samples = FirstResponseSamples::default();
        let now = Instant::now();
        {
            let mut values = samples.0.lock().unwrap();
            for _ in 0..=CAPACITY {
                values.record(now, now);
            }
        }
        let snapshot = samples.snapshot();
        assert_eq!(snapshot.samples, CAPACITY as u64);
        assert_eq!(snapshot.window_seconds, 60);
        assert!(snapshot.truncated);
        samples.0.lock().unwrap().expire(now + WINDOW);
        let expired = samples.snapshot();
        assert_eq!(expired.samples, 0);
        assert!(!expired.truncated);
    }

    #[test]
    fn first_response_records_once_and_excludes_unoffered_output() {
        let samples = Arc::new(FirstResponseSamples::default());
        drop(samples.observe(Instant::now()));
        assert_eq!(samples.snapshot().samples, 0);
        let mut response = samples.observe(Instant::now());
        response.output();
        response.output();
        assert_eq!(samples.snapshot().samples, 1);
        assert!(samples.snapshot().median_seconds.is_some());
    }

    #[test]
    fn first_response_recognizes_answer_and_reasoning_across_protocols() {
        for frame in [
            json!({"choices":[{"delta":{"content":"answer"}}]}),
            json!({"choices":[{"delta":{"reasoning_content":"reason"}}]}),
            json!({"choices":[{"text":"answer"}]}),
            json!({"type":"content_block_delta","delta":{"thinking":"reason"}}),
            json!({"type":"content_block_start","content_block":{"text":"answer"}}),
            json!({"type":"response.output_text.delta","delta":"answer"}),
            json!({"type":"response.reasoning_summary_text.delta","delta":"reason"}),
        ] {
            assert!(stream_text(&frame), "{frame}");
        }
        for frame in [
            json!({"choices":[{"delta":{"role":"assistant","content":""}}]}),
            json!({"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{}"}}]}}]}),
            json!({"type":"content_block_delta","delta":{"signature":"signature"}}),
            json!({"usage":{"output_tokens":1}}),
        ] {
            assert!(!stream_text(&frame), "{frame}");
        }
        for response in [
            json!({"choices":[{"message":{"content":"answer"}}]}),
            json!({"choices":[{"message":{"reasoning_content":"reason"}}]}),
            json!({"content":[{"type":"thinking","thinking":"reason"}]}),
            json!({"output":[{"content":[{"text":"answer"}]}]}),
        ] {
            assert!(response_text(&response), "{response}");
        }
        assert!(!response_text(
            &json!({"choices":[{"message":{"content":""}}]})
        ));
    }
}
