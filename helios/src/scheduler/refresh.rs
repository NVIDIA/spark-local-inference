/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::{Fence, ScheduledStep, cost};
use crate::{Result, execution::ExecutionTiming};

pub(super) const CREDIT_CAP_NS: f64 = 5_000_000.0;
const INCOME_RATE: f64 = 0.002 / 0.998;
const ACCEPTANCE_INCOME_RATE: f64 = 0.01 / 0.99;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum PlanOrigin {
    #[default]
    Ordinary,
    Discovery,
    RefreshStart(cost::Key),
    AcceptanceStart,
}

impl PlanOrigin {
    pub fn is_probe(self) -> bool {
        !matches!(self, Self::Ordinary | Self::Discovery)
    }

    pub fn is_acceptance(self) -> bool {
        matches!(self, Self::AcceptanceStart)
    }
}

#[derive(Default)]
pub(super) struct RefreshState {
    pub credit_ns: f64,
    pub acceptance_credit_ns: f64,
    active: bool,
}

impl RefreshState {
    #[cfg(test)]
    pub(super) fn active(&self) -> bool { self.active }

    pub fn ready(&self) -> bool {
        !self.active && self.credit_ns >= CREDIT_CAP_NS
    }

    pub fn acceptance_ready(&self) -> bool {
        !self.active && self.acceptance_credit_ns >= CREDIT_CAP_NS
    }

    pub fn validate(&self, origin: PlanOrigin) -> Result<()> {
        match origin {
            PlanOrigin::RefreshStart(_) if !self.ready() => Err("refresh requires full credit and no active episode".into()),
            PlanOrigin::AcceptanceStart if !self.acceptance_ready() => Err("acceptance learning requires full credit and no active episode".into()),
            _ => Ok(()),
        }
    }

    pub fn commit(&mut self, origin: PlanOrigin, _fence: Fence, _step: &ScheduledStep) {
        if origin.is_probe() { self.active = true; }
    }

    pub fn validate_completion(&self, origin: PlanOrigin, timing: ExecutionTiming) -> Result<()> {
        if origin.is_probe()
            && (timing.queued || (cost::observed_duration_ns(timing).is_none() && timing.wall_ns == 0)) {
            return Err("probe completion requires an isolated measured interval".into());
        }
        Ok(())
    }

    pub fn complete(&mut self, origin: PlanOrigin, timing: ExecutionTiming) {
        match origin {
            PlanOrigin::Ordinary => {
                if let Some(duration) = cost::observed_duration_ns(timing) {
                    self.credit_ns = CREDIT_CAP_NS.min(self.credit_ns + INCOME_RATE * duration as f64);
                    self.acceptance_credit_ns = CREDIT_CAP_NS.min(self.acceptance_credit_ns + ACCEPTANCE_INCOME_RATE * duration as f64);
                }
            }
            PlanOrigin::Discovery => {}
            _ => {
                let duration = cost::observed_duration_ns(timing).unwrap_or(timing.wall_ns);
                if origin.is_acceptance() { self.acceptance_credit_ns -= duration as f64; }
                else { self.credit_ns -= duration as f64; }
                self.active = false;
            }
        }
    }

    pub fn idle(&mut self) {
        self.credit_ns = self.credit_ns.min(0.0);
        self.acceptance_credit_ns = self.acceptance_credit_ns.min(0.0);
    }

    pub fn abort(&mut self) {
        self.active = false;
        self.idle();
    }
}
