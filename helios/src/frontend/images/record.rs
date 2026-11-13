/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Exact little-endian CPU tensor records for bounded prepared-input storage.

use super::{ImageGeometry, ProcessedImages};
use crate::Result;

impl ProcessedImages {
    /// Concatenates CPU image batches while preserving view sequence boundaries.
    pub fn append(&mut self, mut images: Self) {
        if self.cu_seqlens.is_empty() {
            self.cu_seqlens.push(0);
        }
        let offset = *self.cu_seqlens.last().unwrap();
        self.cu_seqlens
            .extend(images.cu_seqlens.into_iter().skip(1).map(|n| n + offset));
        self.geometry.append(&mut images.geometry);
        self.pixel_values.append(&mut images.pixel_values);
        self.position_ids.append(&mut images.position_ids);
        self.interpolation_indices
            .append(&mut images.interpolation_indices);
        self.interpolation_weights
            .append(&mut images.interpolation_weights);
    }

    pub fn encoded_len(&self) -> Result<usize> {
        [
            (self.geometry.len(), 56usize),
            (self.pixel_values.len(), 4),
            (self.position_ids.len(), 16),
            (self.interpolation_indices.len(), 32),
            (self.interpolation_weights.len(), 16),
            (self.cu_seqlens.len(), 4),
        ]
        .into_iter()
        .try_fold(9usize + 6 * 8, |total, (count, width)| {
            count
                .checked_mul(width)
                .and_then(|bytes| total.checked_add(bytes))
                .ok_or_else(|| "prepared image serialized extent overflows".into())
        })
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(self.encoded_len()?);
        self.write_encoded(|bytes| out.extend_from_slice(bytes));
        Ok(out)
    }

    fn write_encoded(&self, mut write: impl FnMut(&[u8])) {
        write(b"FLAREIMG\x01");
        let count = |write: &mut dyn FnMut(&[u8]), n: usize| write(&(n as u64).to_le_bytes());
        count(&mut write, self.geometry.len());
        for g in &self.geometry {
            for n in [g.height, g.width, g.patches, g.tokens] {
                count(&mut write, n);
            }
            for n in g.grid_thw {
                write(&n.to_le_bytes());
            }
        }
        count(&mut write, self.pixel_values.len());
        for n in &self.pixel_values {
            write(&n.to_le_bytes());
        }
        count(&mut write, self.position_ids.len());
        for n in self.position_ids.iter().flatten() {
            write(&n.to_le_bytes());
        }
        count(&mut write, self.interpolation_indices.len());
        for n in self.interpolation_indices.iter().flatten() {
            write(&n.to_le_bytes());
        }
        count(&mut write, self.interpolation_weights.len());
        for n in self.interpolation_weights.iter().flatten() {
            write(&n.to_le_bytes());
        }
        count(&mut write, self.cu_seqlens.len());
        for n in &self.cu_seqlens {
            write(&n.to_le_bytes());
        }
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader(bytes);
        if reader.take(9)? != b"FLAREIMG\x01" {
            return Err("invalid prepared image record version".into());
        }
        let count = reader.count(56)?;
        let mut geometry = Vec::with_capacity(count);
        for _ in 0..count {
            let [height, width, patches, tokens] = [
                reader.usize()?,
                reader.usize()?,
                reader.usize()?,
                reader.usize()?,
            ];
            let grid_thw = [reader.i64()?, reader.i64()?, reader.i64()?];
            geometry.push(ImageGeometry {
                height,
                width,
                patches,
                tokens,
                grid_thw,
            });
        }
        let count = reader.count(4)?;
        let mut pixel_values = Vec::with_capacity(count);
        for _ in 0..count {
            pixel_values.push(reader.f32()?);
        }
        let count = reader.count(16)?;
        let mut position_ids = Vec::with_capacity(count);
        for _ in 0..count {
            position_ids.push([reader.i64()?, reader.i64()?]);
        }
        let count = reader.count(32)?;
        let mut interpolation_indices = Vec::with_capacity(count);
        for _ in 0..count {
            interpolation_indices.push([
                reader.i64()?,
                reader.i64()?,
                reader.i64()?,
                reader.i64()?,
            ]);
        }
        let count = reader.count(16)?;
        let mut interpolation_weights = Vec::with_capacity(count);
        for _ in 0..count {
            interpolation_weights.push([
                reader.f32()?,
                reader.f32()?,
                reader.f32()?,
                reader.f32()?,
            ]);
        }
        let count = reader.count(4)?;
        let mut cu_seqlens = Vec::with_capacity(count);
        for _ in 0..count {
            cu_seqlens.push(i32::from_le_bytes(reader.take(4)?.try_into().unwrap()));
        }
        if !reader.0.is_empty() {
            return Err("trailing prepared image record bytes".into());
        }
        Ok(Self {
            geometry,
            pixel_values,
            position_ids,
            interpolation_indices,
            interpolation_weights,
            cu_seqlens,
        })
    }
}

struct Reader<'a>(&'a [u8]);
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let result = self.0.get(..n).ok_or("truncated prepared image record")?;
        self.0 = &self.0[n..];
        Ok(result)
    }
    fn usize(&mut self) -> Result<usize> {
        usize::try_from(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
            .map_err(|_| "prepared image extent exceeds address space".into())
    }
    fn count(&mut self, width: usize) -> Result<usize> {
        let n = self.usize()?;
        if n > self.0.len() / width {
            return Err("truncated prepared image extent".into());
        }
        Ok(n)
    }
    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}
