/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::{Error, ExecutionError, Result};

pub const MAX_READINESS_INTERVALS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadinessIntervals {
    pub(crate) intervals: [[u64; 2]; MAX_READINESS_INTERVALS],
    pub(crate) len: usize,
}

impl Default for ReadinessIntervals {
    fn default() -> Self {
        Self { intervals: [[0; 2]; MAX_READINESS_INTERVALS], len: 0 }
    }
}

impl ReadinessIntervals {
    pub(crate) fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.is_empty() || bytes.len() % 24 != 0 || bytes.len() / 24 > MAX_READINESS_INTERVALS {
            return Err(invalid("invalid readiness timestamp extent"));
        }
        let mut result = Self { len: bytes.len() / 24, ..Self::default() };
        for (output, sample) in result.intervals.iter_mut().zip(bytes.chunks_exact(24)) {
            let mut values = sample.chunks_exact(8).map(|value| u64::from_ne_bytes(value.try_into().unwrap()));
            let (before, ready, finished) = (values.next().unwrap(), values.next().unwrap(), values.next().unwrap());
            if before == 0 || before > ready || ready > finished {
                return Err(invalid("readiness timestamps must be positive and ordered"));
            }
            *output = [finished - before, finished - ready];
        }
        Ok(result)
    }

    pub(crate) fn merge(&mut self, peer: Self) -> Result<()> {
        if self.len != peer.len || self.len == 0 {
            return Err(invalid("rank readiness interval counts differ"));
        }
        for (local, remote) in self.intervals[..self.len].iter_mut().zip(&peer.intervals[..peer.len]) {
            local[0] = local[0].min(remote[0]);
            local[1] = local[1].min(remote[1]);
        }
        Ok(())
    }

    pub fn wait_ns(&self) -> u64 {
        // With a shared barrier endpoint E, min(E - arrival) - min(E - ready)
        // is max(ready) - max(arrival). Rank-local clocks cancel in each span.
        self.intervals[..self.len].iter()
            .fold(0u64, |total, pair| total.saturating_add(pair[0].saturating_sub(pair[1])))
    }
}

fn invalid(message: &str) -> Error {
    Error::execution(ExecutionError::Completion, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(values: &[[u64; 3]]) -> ReadinessIntervals {
        let bytes = values.iter().flatten().flat_map(|value| value.to_ne_bytes()).collect::<Vec<_>>();
        ReadinessIntervals::decode(&bytes).unwrap()
    }

    #[test]
    fn collective_wait_propagation_is_removed_without_comparing_rank_clocks() {
        // Rank clocks have independent offsets; both barriers complete at the same physical instant.
        let mut first = sample(&[[100, 110, 111]]);
        let second = sample(&[[1002, 1008, 1011]]);
        assert_eq!(first.wait_ns(), 10);
        assert_eq!(second.wait_ns(), 6);
        first.merge(second).unwrap();
        assert_eq!(first.wait_ns(), 8);
    }

    #[test]
    fn delivery_overlapped_by_peer_computation_is_not_charged_as_a_group_stall() {
        let mut first = sample(&[[100, 110, 121]]);
        first.merge(sample(&[[1020, 1020, 1021]])).unwrap();
        assert_eq!(first.wait_ns(), 0);
    }

    #[test]
    fn independent_delivery_intervals_accumulate_after_rank_aggregation() {
        let mut first = sample(&[[100, 110, 111], [200, 200, 207]]);
        first.merge(sample(&[[1002, 1008, 1011], [1202, 1206, 1207]])).unwrap();
        assert_eq!(first.wait_ns(), 12);
    }

    #[test]
    fn malformed_samples_and_rank_catalog_mismatches_are_rejected() {
        assert!(ReadinessIntervals::decode(&[]).is_err());
        assert!(ReadinessIntervals::decode(&[0; 24]).is_err());
        let mut first = sample(&[[100, 101, 102]]);
        assert!(first.merge(sample(&[[100, 101, 102], [200, 201, 202]])).is_err());
    }
}
