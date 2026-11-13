/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared request-row validation and host staging for captured graph inputs.

use crate::{
    Result,
    execution::{GraphKind, OwnedUpload, REQUEST_LANES, Upload},
    model::{Admission, ModelConfig},
    scheduler::{LiveRequest, RequestScheduler, ScheduledStep},
};

pub struct InputStep<'a> {
    pub config: &'a ModelConfig,
    pub scheduler: &'a RequestScheduler,
    pub step: &'a ScheduledStep,
    pub admissions: &'a [Option<Admission>],
    pub slots: [i32; REQUEST_LANES],
    pub generations: [i64; REQUEST_LANES],
    pub lengths: [i32; REQUEST_LANES],
    pub samples: [i32; REQUEST_LANES],
}

/// Pack after the scheduler reserves every page referenced by the submission.
pub fn pack_inputs(
    inputs: &mut crate::model::Inputs,
    config: &ModelConfig,
    scheduler: &RequestScheduler,
    step: &ScheduledStep,
    admissions: &[Option<Admission>],
) -> Result<()> {
    step.validate_mixed()?;
    if let Some(mixed) = &step.mixed {
        if mixed.speculative != !config.speculative_widths.is_empty()
            || mixed.prefill_capacity > config.limits.prefill_chunk_size
        {
            return Err("mixed graph mode exceeds the loaded model configuration".into());
        }
    }
    if step.lanes.is_empty()
        || step.lanes.len() > step.batch_capacity
        || step.batch_capacity > config.limits.request_capacity()
    {
        return Err("invalid model lane count".into());
    }
    let mut frame = InputStep {
        config,
        scheduler,
        step,
        admissions,
        slots: [-1; REQUEST_LANES],
        generations: [0; REQUEST_LANES],
        lengths: [0; REQUEST_LANES],
        samples: [0; REQUEST_LANES],
    };
    let mut seen = [false; REQUEST_LANES];
    for (row, lane) in step.lanes.iter().enumerate() {
        let slot = lane.request.slot as usize;
        if slot >= config.limits.max_active
            || slot >= admissions.len()
            || std::mem::replace(&mut seen[slot], true)
        {
            return Err("model rows require unique physical slots".into());
        }
        admissions[slot]
            .as_ref()
            .filter(|admission| admission.handle == lane.request)
            .ok_or("missing or stale native request admission")?;
        let live = scheduler.request(lane.request)?;
        if step.kind == GraphKind::Mixed
            && live.phase
                != if step.lane_kind(row).is_prefill() {
                    crate::scheduler::RequestPhase::Prefill
                } else {
                    crate::scheduler::RequestPhase::Decode
                }
        {
            return Err("mixed row phase does not match its request".into());
        }
        frame.slots[row] = slot as i32;
        frame.generations[row] = lane.request.generation as i64;
        frame.lengths[row] =
            i32::try_from(lane.tokens).map_err(|_| "model row exceeds token capacity")?;
        frame.samples[row] = i32::from(
            !step.lane_kind(row).is_prefill()
                || lane.start_position.checked_add(lane.tokens as u64)
                    == Some(live.prompt_tokens as u64),
        );
    }
    inputs.pack(&frame)
}

pub fn append_pages(
    staging: &mut UploadBatch,
    counts: &mut [(u64, usize)],
    name: &str,
    request: &LiveRequest,
    capacity: usize,
    pages: std::ops::Range<usize>,
) -> Result<()> {
    let slot = request.handle.slot as usize;
    let previous = counts[slot];
    let count = if previous.0 == request.handle.generation {
        previous.1
    } else {
        0
    };
    let available = pages
        .end
        .checked_sub(pages.start)
        .ok_or("invalid page-table window")?;
    if available > capacity || available < count {
        return Err("page table does not match its request capacity".into());
    }
    if available > count {
        if pages.end > request.pages().count() {
            return Err("page-table window exceeds reserved pages".into());
        }
        staging.i32(
            name,
            (slot * capacity + count) * 4,
            request
                .pages()
                .skip(pages.start + count)
                .take(available - count)
                .map(|page| page.0 as i32),
        );
        counts[slot] = (request.handle.generation, available);
    }
    Ok(())
}

/// Graph execution copies active staging bytes before submit returns.
#[derive(Default)]
pub struct UploadBatch {
    uploads: Vec<OwnedUpload>,
    used: usize,
}

impl UploadBatch {
    pub fn clear(&mut self) {
        self.used = 0;
    }
    pub fn put(&mut self, name: &str, offset: usize, bytes: impl Iterator<Item = u8>) {
        if self.used == self.uploads.len() {
            self.uploads.push(OwnedUpload {
                buffer: String::new(),
                offset: 0,
                bytes: Vec::new(),
            });
        }
        let upload = &mut self.uploads[self.used];
        upload.buffer.clear();
        upload.buffer.push_str(name);
        upload.offset = offset;
        upload.bytes.clear();
        upload.bytes.extend(bytes);
        self.used += 1;
    }
    pub fn i32(&mut self, name: &str, offset: usize, values: impl IntoIterator<Item = i32>) {
        self.put(name, offset, values.into_iter().flat_map(i32::to_le_bytes));
    }
    pub fn i64(&mut self, name: &str, offset: usize, values: impl IntoIterator<Item = i64>) {
        self.put(name, offset, values.into_iter().flat_map(i64::to_le_bytes));
    }
    pub fn uploads(&self) -> Vec<Upload<'_>> {
        self.uploads[..self.used]
            .iter()
            .map(|upload| Upload {
                buffer: &upload.buffer,
                offset: upload.offset,
                bytes: &upload.bytes,
            })
            .collect()
    }
}
