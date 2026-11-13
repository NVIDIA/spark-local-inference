/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Expected serial continuation value, composed without joint outcome branches.

/// Expected isolated work and waiting, with times measured in milliseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Metrics {
    pub elapsed_ms: f64,
    pub prefill_tokens: f64,
    pub output_tokens: f64,
    pub prefill_wait_ms: f64,
    pub unfinished_wait_ms: f64,
    pub decode_age_ms2: f64,
}

/// Marginal moments of one request after a candidate execution.
///
/// A completed request contributes zeros. For uncertain EOS, each field is the
/// unconditional expectation over that request's possible completion records.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExpectedRow {
    pub weight: f64,
    pub isolated: Metrics,
    pub duration_second_ms2: f64,
    pub live_probability: f64,
    pub prefill_probability: f64,
    pub decode_probability: f64,
    pub decode_age_ms: f64,
    pub decode_first_ms: f64,
    pub age_until_first_ms2: f64,
}

/// Explicit values for progress, occupied GPU time, and waiting.
#[derive(Clone, Copy, Debug)]
pub struct Prices {
    pub output_token: f64,
    pub gpu_ms: f64,
    pub prefill_wait_ms: f64,
    pub unfinished_wait_ms: f64,
    pub decode_age_ms2: f64,
}

impl Prices {
    pub fn useful_work(&self, metrics: Metrics) -> f64 {
        metrics.prefill_tokens + self.output_token * metrics.output_tokens
    }

    pub fn net_work(&self, metrics: Metrics) -> f64 {
        self.useful_work(metrics)
            - self.prefill_wait_ms * metrics.prefill_wait_ms
            - self.unfinished_wait_ms * metrics.unfinished_wait_ms
            - self.decode_age_ms2 * metrics.decode_age_ms2
    }

    pub fn continuation(&self, metrics: Metrics) -> f64 {
        self.net_work(metrics) - self.gpu_ms * metrics.elapsed_ms
    }

    /// Expected net progress plus continuation change per occupied millisecond.
    /// Indeterminate graph durations require separate exploration selection.
    pub fn score(&self, step: Metrics, before: f64, after: f64) -> Option<f64> {
        if !step.elapsed_ms.is_finite() || step.elapsed_ms <= 0.0 {
            return None;
        }
        let score = (self.net_work(step) + after - before) / step.elapsed_ms;
        score.is_finite().then_some(score)
    }
}

/// Compose a serial reference in request order using independent row outcomes.
///
/// Timing variance contributes to stream age through the second moment of the
/// preceding requests' total duration. Replacing it with squared mean time
/// underestimates waiting whenever completion time is uncertain.
pub fn serial_metrics(rows: &[ExpectedRow]) -> Metrics {
    let mut total = Metrics::default();
    let mut preceding_mean = 0.0;
    let mut preceding_second = 0.0;
    for row in rows {
        let m = row.isolated;
        let w = row.weight;
        total.elapsed_ms += m.elapsed_ms;
        total.prefill_tokens += w * m.prefill_tokens;
        total.output_tokens += w * m.output_tokens;
        total.prefill_wait_ms += w * (m.prefill_wait_ms + row.prefill_probability * preceding_mean);
        total.unfinished_wait_ms +=
            w * (m.unfinished_wait_ms + row.live_probability * preceding_mean);
        total.decode_age_ms2 += w
            * (m.decode_age_ms2
                + row.age_until_first_ms2
                + (row.decode_age_ms + row.decode_first_ms) * preceding_mean
                + 0.5 * row.decode_probability * preceding_second);
        preceding_second += row.duration_second_ms2 + 2.0 * preceding_mean * m.elapsed_ms;
        preceding_mean += m.elapsed_ms;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prices() -> Prices {
        Prices {
            output_token: 2.0,
            gpu_ms: 1.0,
            prefill_wait_ms: 0.0,
            unfinished_wait_ms: 0.2,
            decode_age_ms2: 0.1,
        }
    }

    fn row(duration: f64, second: f64, age: f64) -> ExpectedRow {
        ExpectedRow {
            weight: 1.0,
            isolated: Metrics {
                elapsed_ms: duration,
                output_tokens: duration,
                unfinished_wait_ms: duration,
                decode_age_ms2: duration / 2.0,
                ..Default::default()
            },
            duration_second_ms2: second,
            live_probability: 1.0,
            decode_probability: 1.0,
            decode_age_ms: age,
            decode_first_ms: 1.0,
            age_until_first_ms2: age,
            ..Default::default()
        }
    }

    #[test]
    fn random_completion_time_preserves_its_variance_in_wait_cost() {
        let metrics = serial_metrics(&[row(2.0, 5.0, 0.0), row(1.0, 1.0, 4.0)]);
        assert_eq!(metrics.elapsed_ms, 3.0);
        assert_eq!(metrics.output_tokens, 3.0);
        assert_eq!(metrics.unfinished_wait_ms, 5.0);
        assert_eq!(metrics.decode_age_ms2, (10.5 + 25.5) / 2.0);
        assert!((prices().continuation(metrics) - 0.2).abs() < 1e-12);
    }

    #[test]
    fn eos_marginals_include_the_probability_of_no_remaining_work() {
        let mut uncertain = row(1.0, 2.0, 0.0);
        uncertain.live_probability = 0.5;
        uncertain.decode_probability = 0.5;
        uncertain.decode_first_ms = 0.5;
        let metrics = serial_metrics(&[uncertain, row(1.0, 1.0, 0.0)]);
        assert_eq!(metrics.elapsed_ms, 2.0);
        assert_eq!(metrics.output_tokens, 2.0);
        assert_eq!(metrics.unfinished_wait_ms, 3.0);
        assert_eq!(metrics.decode_age_ms2, (0.5 + 5.5) / 2.0);
    }

    #[test]
    fn prefill_wait_does_not_create_a_decode_age_clock() {
        let prefill = ExpectedRow {
            weight: 2.0,
            isolated: Metrics {
                elapsed_ms: 4.0,
                prefill_tokens: 64.0,
                output_tokens: 1.0,
                prefill_wait_ms: 4.0,
                unfinished_wait_ms: 4.0,
                ..Default::default()
            },
            duration_second_ms2: 16.0,
            live_probability: 1.0,
            prefill_probability: 1.0,
            ..Default::default()
        };
        let metrics = serial_metrics(&[row(1.0, 1.0, 0.0), prefill]);
        assert_eq!(metrics.prefill_tokens, 128.0);
        assert_eq!(metrics.prefill_wait_ms, 10.0);
        assert_eq!(metrics.decode_age_ms2, 0.5);
    }

    #[test]
    fn indeterminate_duration_is_not_divided_into_work() {
        assert_eq!(prices().score(Metrics::default(), 0.0, 0.0), None);
        let step = Metrics {
            elapsed_ms: 30.0,
            output_tokens: 8.0,
            ..Default::default()
        };
        assert_eq!(prices().score(step, 5.0, 10.0), Some(21.0 / 30.0));
    }

    #[test]
    fn sixteen_rows_require_only_a_linear_composition() {
        let rows = [row(1.0, 1.0, 0.0); 16];
        let metrics = serial_metrics(&rows);
        assert_eq!(metrics.elapsed_ms, 16.0);
        assert_eq!(metrics.output_tokens, 16.0);
        assert_eq!(metrics.unfinished_wait_ms, (1..=16).sum::<u32>() as f64);
        assert_eq!(
            metrics.decode_age_ms2,
            (1..=16).map(|i| i * i).sum::<u32>() as f64 / 2.0
        );
    }
}
