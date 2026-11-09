/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Image-layout tensors consumed through prepared semantic bindings.
use super::ProcessedImages;
use crate::{
    Result,
    execution::Dtype,
    model::{ModelConfig, contract::{ImageLayoutKind, ImagePreparationKind}, inputs::UploadBatch},
    scheduler::RequestHandle,
};

pub(crate) fn tensor_layout(
    kind: ImageLayoutKind,
    name: &str,
    patches: usize,
) -> Result<(Dtype, usize)> {
    use Dtype::{Float32, Int32, Int64};
    let layout = match (kind, name) {
        (ImageLayoutKind::Qwen | ImageLayoutKind::Mimo, "vision_pixel_values") => {
            (Float32, patches * 1536)
        }
        (
            ImageLayoutKind::Deepseek | ImageLayoutKind::DeepseekRowMajor | ImageLayoutKind::Moonvit | ImageLayoutKind::Step,
            "vision_pixel_values",
        ) => (Float32, patches * 588),
        (ImageLayoutKind::Bracketed, "vision_pixel_values") => (Float32, patches * 1176),
        (ImageLayoutKind::Radio, "vision_pixel_values") => (Float32, patches * 768),
        (ImageLayoutKind::Glm, "vision_pixel_values") => (Float32, patches * 1176),
        (_, "vision_position_ids") => (Int64, patches * 2),
        (ImageLayoutKind::Step, "vision_cu_seqlens") => (Int32, super::step::MAX_VIEWS + 1),
        (_, "vision_cu_seqlens") => (Int32, 2),
        (_, "vision_num_patches" | "vision_feature_offset") => (Int32, 1),
        (
            ImageLayoutKind::Bracketed | ImageLayoutKind::Mimo
            | ImageLayoutKind::Deepseek | ImageLayoutKind::DeepseekRowMajor
            | ImageLayoutKind::Moonvit
            | ImageLayoutKind::Step
            | ImageLayoutKind::Radio,
            "vision_num_features",
        ) => (Int32, 1),
        (
            ImageLayoutKind::Qwen
            | ImageLayoutKind::Moonvit
            | ImageLayoutKind::Step
            | ImageLayoutKind::Radio
            | ImageLayoutKind::Bracketed,
            "vision_interpolation_indices",
        ) => (Int64, patches * 4),
        (
            ImageLayoutKind::Qwen
            | ImageLayoutKind::Moonvit
            | ImageLayoutKind::Step
            | ImageLayoutKind::Radio
            | ImageLayoutKind::Bracketed,
            "vision_interpolation_weights",
        ) => (Float32, patches * 4),
        // Muse Glimmer's windowed bracketed layout: one segment per 32×32-patch cell, at most
        // `patches / 64 + 1` cells for an even-sided grid, plus the leading zero.
        (ImageLayoutKind::Bracketed, "vision_cu_window_seqlens") => (Int32, patches / 64 + 2),
        (ImageLayoutKind::Bracketed, "vision_reverse_index") => (Int64, patches),
        (ImageLayoutKind::Mimo, "vision_column_indices" | "vision_row_indices") => {
            (Int32, patches / 4)
        }
        (
            ImageLayoutKind::Deepseek | ImageLayoutKind::DeepseekRowMajor | ImageLayoutKind::Moonvit | ImageLayoutKind::Radio,
            "vision_merge_indices",
        ) => (Int32, patches),
        (ImageLayoutKind::Step, "vision_downsample1_indices") => (Int32, patches / 4 * 9),
        (ImageLayoutKind::Step, "vision_downsample2_indices") => (Int32, patches / 16 * 9),
        _ => return Err(format!("image layout has no tensor {name}").into()),
    };
    Ok(layout)
}

pub(crate) fn prepare(
    config: &ModelConfig,
    handle: RequestHandle,
    images: &ProcessedImages,
    image: usize,
    destination_feature_offset: usize,
) -> Result<UploadBatch> {
    let geometry = images.geometry.get(image).ok_or("missing scheduled image")?;
    if destination_feature_offset.checked_add(geometry.tokens)
        .is_none_or(|end| end > config.contract.vision.features_per_request) {
        return Err("invalid destination vision feature extent".into());
    }
    i32::try_from(destination_feature_offset).map_err(|_| "vision feature offset exceeds int32")?;
    match config.contract.frontend.image_layout
        .ok_or("image layout is not prepared")? {
        ImageLayoutKind::Bracketed
            if config.contract.frontend.image_preparation == Some(ImagePreparationKind::MuseGlimmer) =>
        {
            muse_glimmer(config, handle, images, image, destination_feature_offset)
        }
        ImageLayoutKind::Bracketed => bracketed(config, handle, images, image, destination_feature_offset),
        ImageLayoutKind::Qwen | ImageLayoutKind::Glm => qwen(config, handle, images, image, destination_feature_offset),
        ImageLayoutKind::Mimo => mimo(config, handle, images, image, destination_feature_offset),
        ImageLayoutKind::Deepseek | ImageLayoutKind::DeepseekRowMajor => deepseek(config, handle, images, image, destination_feature_offset),
        ImageLayoutKind::Moonvit | ImageLayoutKind::Radio => spatial_merge(config, handle, images, image, destination_feature_offset),
        ImageLayoutKind::Step => step(config, handle, images, image, destination_feature_offset),
    }
}

fn qwen(
    config: &ModelConfig,
    handle: RequestHandle,
    images: &ProcessedImages,
    image: usize,
    destination_feature_offset: usize,
) -> Result<UploadBatch> {
    let pixel_width = if matches!(
        config.contract.frontend.image_layout,
        Some(ImageLayoutKind::Moonvit | ImageLayoutKind::Step)
    ) {
        588
    } else if config.contract.frontend.image_layout == Some(ImageLayoutKind::Radio) {
        768
    } else if config.contract.frontend.image_layout == Some(ImageLayoutKind::Glm) {
        1176
    } else {
        1536
    };
    let mut staging = UploadBatch::default();
    let geometry = images
        .geometry
        .get(image)
        .ok_or("missing scheduled image")?;
    let start: usize = images.geometry[..image]
        .iter()
        .map(|geometry| geometry.patches)
        .sum();
    let end = start + geometry.patches;
    if geometry.patches == 0
        || geometry.patches
            > *config
                .contract
                .vision
                .patch_capacities
                .last()
                .ok_or("missing vision capacities")?
        || end / config.contract.vision.patches_per_feature
            > config.contract.vision.features_per_request
        || handle.slot as usize >= config.limits.max_active
        || handle.generation == 0
        || handle.generation > i64::MAX as u64
    {
        return Err("invalid vision graph input extent".into());
    }
    staging.put(
        "vision_pixel_values",
        0,
        images
            .pixel_values
            .get(start * pixel_width..end * pixel_width)
            .ok_or("truncated vision pixels")?
            .iter()
            .copied()
            .flat_map(f32::to_le_bytes),
    );
    if config.contract.frontend.image_layout != Some(ImageLayoutKind::Glm) {
        staging.i64(
            "vision_interpolation_indices",
            0,
            images
                .interpolation_indices
                .get(start..end)
                .ok_or("truncated vision interpolation indices")?
                .iter()
                .flatten()
                .copied(),
        );
        staging.put(
            "vision_interpolation_weights",
            0,
            images
                .interpolation_weights
                .get(start..end)
                .ok_or("truncated vision interpolation weights")?
                .iter()
                .flatten()
                .copied()
                .flat_map(f32::to_le_bytes),
        );
    }
    staging.i64(
        "vision_position_ids",
        0,
        images
            .position_ids
            .get(start..end)
            .ok_or("truncated vision positions")?
            .iter()
            .flatten()
            .copied(),
    );
    if config.contract.frontend.image_layout != Some(ImageLayoutKind::Step) {
        staging.i32("vision_cu_seqlens", 0, [0, geometry.patches as i32]);
    }
    staging.i32("vision_num_patches", 0, [geometry.patches as i32]);
    staging.i32(
        "vision_feature_offset",
        0,
        [destination_feature_offset as i32],
    );
    Ok(staging)
}

fn mimo(
    config: &ModelConfig,
    handle: RequestHandle,
    images: &ProcessedImages,
    image: usize,
    destination_feature_offset: usize,
) -> Result<UploadBatch> {
    let mut staging = UploadBatch::default();
    let geometry = images
        .geometry
        .get(image)
        .ok_or("missing scheduled image")?;
    let start: usize = images.geometry[..image].iter().map(|g| g.patches).sum();
    let end = start + geometry.patches;
    let h = usize::try_from(geometry.grid_thw[1]).map_err(|_| "invalid vision height")?;
    let w = usize::try_from(geometry.grid_thw[2]).map_err(|_| "invalid vision width")?;
    if h == 0
        || w == 0
        || h % 2 != 0
        || w % 2 != 0
        || geometry.grid_thw[0] != 1
        || geometry.patches > 16384
        || h.checked_mul(w) != Some(geometry.patches)
        || geometry.tokens != geometry.patches / 4
        || end / 4 > config.contract.vision.features_per_request
        || handle.slot as usize >= config.limits.max_active
        || handle.generation == 0
        || handle.generation > i64::MAX as u64
    {
        return Err("invalid MiMo vision graph extent".into());
    }
    staging.put(
        "vision_pixel_values",
        0,
        images
            .pixel_values
            .get(start * 1536..end * 1536)
            .ok_or("truncated vision pixels")?
            .iter()
            .copied()
            .flat_map(f32::to_le_bytes),
    );
    staging.i64(
        "vision_position_ids",
        0,
        images
            .position_ids
            .get(start..end)
            .ok_or("truncated vision positions")?
            .iter()
            .flatten()
            .copied(),
    );
    let (columns, rows) = column_permutation(h / 2, w / 2);
    staging.i32("vision_column_indices", 0, columns);
    staging.i32("vision_row_indices", 0, rows);
    staging.i32("vision_cu_seqlens", 0, [0, geometry.patches as i32]);
    staging.i32("vision_num_patches", 0, [geometry.patches as i32]);
    staging.i32("vision_num_features", 0, [geometry.tokens as i32]);
    staging.i32("vision_feature_offset", 0, [destination_feature_offset as i32]);
    Ok(staging)
}
fn column_permutation(height: usize, width: usize) -> (Vec<i32>, Vec<i32>) {
    let columns: Vec<_> = (0..width)
        .flat_map(|x| (0..height).map(move |y| (y * width + x) as i32))
        .collect();
    let mut rows = vec![0; columns.len()];
    for (index, column) in columns.iter().enumerate() {
        rows[*column as usize] = index as i32;
    }
    (columns, rows)
}

fn deepseek(
    config: &ModelConfig,
    handle: RequestHandle,
    images: &ProcessedImages,
    image: usize,
    destination_feature_offset: usize,
) -> Result<UploadBatch> {
    let mut staging = UploadBatch::default();
    let geometry = images
        .geometry
        .get(image)
        .ok_or("missing scheduled image")?;
    let start: usize = images.geometry[..image].iter().map(|g| g.patches).sum();
    let end = start + geometry.patches;
    let h = usize::try_from(geometry.grid_thw[1]).map_err(|_| "invalid vision height")?;
    let w = usize::try_from(geometry.grid_thw[2]).map_err(|_| "invalid vision width")?;
    let capacity = *config.contract.vision.patch_capacities.last().ok_or("missing vision capacities")?;
    let actual = h
        .checked_mul(w)
        .filter(|n| *n > 0 && *n <= capacity)
        .ok_or("invalid vision grid")?;
    if geometry.patches > capacity
        || geometry.patches != h.div_ceil(3) * w.div_ceil(3) * 9
        || geometry.tokens != geometry.patches / 9
        || end / 9 > config.contract.vision.features_per_request
        || handle.slot as usize >= config.limits.max_active
        || handle.generation == 0
        || handle.generation > i64::MAX as u64
    {
        return Err("invalid DeepSeek vision graph extent".into());
    }
    staging.put(
        "vision_pixel_values",
        0,
        images
            .pixel_values
            .get(start * 588..end * 588)
            .ok_or("truncated vision pixels")?
            .iter()
            .copied()
            .flat_map(f32::to_le_bytes),
    );
    staging.i64(
        "vision_position_ids",
        0,
        images
            .position_ids
            .get(start..end)
            .ok_or("truncated vision positions")?
            .iter()
            .flatten()
            .copied(),
    );
    let mut indices = Vec::with_capacity(geometry.patches);
    for row in 0..h.div_ceil(3) {
        for col in 0..w.div_ceil(3) {
            for dy in 0..3 {
                for dx in 0..3 {
                    let (y, x) = (row * 3 + dy, col * 3 + dx);
                    indices.push(if y < h && x < w {
                        (y * w + x) as i32
                    } else {
                        -1
                    });
                }
            }
        }
    }
    staging.i32("vision_merge_indices", 0, indices);
    staging.i32("vision_cu_seqlens", 0, [0, actual as i32]);
    staging.i32("vision_num_patches", 0, [actual as i32]);
    staging.i32("vision_num_features", 0, [geometry.tokens as i32]);
    staging.i32("vision_feature_offset", 0, [destination_feature_offset as i32]);
    Ok(staging)
}

fn spatial_merge(
    config: &ModelConfig,
    handle: RequestHandle,
    images: &ProcessedImages,
    image: usize,
    destination_feature_offset: usize,
) -> Result<UploadBatch> {
    let geometry = images
        .geometry
        .get(image)
        .ok_or("missing scheduled image")?;
    let h = usize::try_from(geometry.grid_thw[1]).map_err(|_| "invalid vision height")?;
    let w = usize::try_from(geometry.grid_thw[2]).map_err(|_| "invalid vision width")?;
    let max_side = if config.contract.frontend.image_layout == Some(ImageLayoutKind::Radio) {
        13312
    } else {
        512
    };
    if h == 0
        || w == 0
        || h > max_side
        || w > max_side
        || h % 2 != 0
        || w % 2 != 0
        || geometry.grid_thw[0] != 1
        || h.checked_mul(w) != Some(geometry.patches)
        || geometry.tokens != geometry.patches / 4
    {
        return Err("invalid spatial merge vision grid".into());
    }
    let mut staging = qwen(config, handle, images, image, destination_feature_offset)?;
    let mut indices = Vec::with_capacity(geometry.patches);
    for y in (0..h).step_by(2) {
        for x in (0..w).step_by(2) {
            for dy in 0..2 {
                for dx in 0..2 {
                    indices.push(((y + dy) * w + x + dx) as i32);
                }
            }
        }
    }
    staging.i32("vision_merge_indices", 0, indices);
    staging.i32("vision_num_features", 0, [geometry.tokens as i32]);
    Ok(staging)
}

fn convolution_indices(side: usize, base: usize) -> Vec<i32> {
    let mut indices = Vec::with_capacity(side * side / 4 * 9);
    for y in (0..side).step_by(2) {
        for x in (0..side).step_by(2) {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (y, x) = (y as isize + dy, x as isize + dx);
                    indices.push(
                        if y < 0 || x < 0 || y >= side as isize || x >= side as isize {
                            -1
                        } else {
                            (base + y as usize * side + x as usize) as i32
                        },
                    );
                }
            }
        }
    }
    indices
}

fn step(
    config: &ModelConfig,
    handle: RequestHandle,
    images: &ProcessedImages,
    image: usize,
    destination_feature_offset: usize,
) -> Result<UploadBatch> {
    let geometry = images
        .geometry
        .get(image)
        .ok_or("missing scheduled image")?;
    let ny = usize::try_from(geometry.grid_thw[1]).map_err(|_| "invalid crop rows")?;
    let nx = usize::try_from(geometry.grid_thw[2]).map_err(|_| "invalid crop columns")?;
    let local = ny
        .checked_mul(nx)
        .filter(|n| *n < super::step::MAX_VIEWS)
        .ok_or("too many image crops")?;
    if geometry.grid_thw[0] != 1
        || (nx == 0) != (ny == 0)
        || geometry.patches != local * 1296 + 2704
        || geometry.tokens != geometry.patches / 16
    {
        return Err("invalid Step image geometry".into());
    }
    let mut staging = qwen(config, handle, images, image, destination_feature_offset)?;
    let mut cu = vec![0];
    let (mut first, mut second) = (vec![], vec![]);
    let mut base = 0;
    for view in 0..=local {
        let side = if view == local { 52 } else { 36 };
        first.extend(convolution_indices(side, base));
        second.extend(convolution_indices(side / 2, base / 4));
        base += side * side;
        cu.push(base as i32);
    }
    cu.resize(super::step::MAX_VIEWS + 1, base as i32);
    staging.i32("vision_cu_seqlens", 0, cu);
    staging.i32("vision_downsample1_indices", 0, first);
    staging.i32("vision_downsample2_indices", 0, second);
    staging.i32("vision_num_features", 0, [geometry.tokens as i32]);
    Ok(staging)
}

/// Validates a 2×2-merged single-frame image against the contract and the request handle;
/// returns its geometry, its packed patch range, and the deepest vision capacity.
fn bracketed_extent<'a>(config: &ModelConfig, handle: RequestHandle, images: &'a ProcessedImages, image: usize, error: &'static str) -> Result<(&'a super::ImageGeometry, usize, usize, usize)> {
    let geometry = images.geometry.get(image).ok_or("missing scheduled image")?;
    let start: usize = images.geometry[..image].iter().map(|g| g.patches).sum();
    let end = start + geometry.patches;
    let capacities = &config.contract.vision.patch_capacities;
    let [t, h, w] = geometry.grid_thw;
    if t != 1 || h <= 0 || w <= 0 || h%2 != 0 || w%2 != 0
        || (h as usize).checked_mul(w as usize) != Some(geometry.patches)
        || geometry.tokens != geometry.patches/4 || config.contract.vision.patches_per_feature != 4
        || geometry.patches > *capacities.last().ok_or("missing vision capacities")?
        || handle.slot as usize >= config.limits.max_active || handle.generation == 0 || handle.generation > i64::MAX as u64 {
        return Err(error.into());
    }
    Ok((geometry, start, end, capacities[capacities.len() - 1]))
}

fn bracketed(config: &ModelConfig, handle: RequestHandle, images: &ProcessedImages, image: usize, destination_feature_offset: usize) -> Result<UploadBatch> {
    let (geometry, start, end, _) =
        bracketed_extent(config, handle, images, image, "invalid temporal patch-merge image extent")?;
    let mut staging = UploadBatch::default();
    staging.put("vision_pixel_values", 0, images.pixel_values.get(start*1176..end*1176).ok_or("truncated vision pixels")?.iter().copied().flat_map(f32::to_le_bytes));
    staging.i64("vision_position_ids", 0, images.position_ids.get(start..end).ok_or("truncated vision positions")?.iter().flatten().copied());
    staging.i32("vision_cu_seqlens", 0, [0, geometry.patches as i32]);
    staging.i32("vision_num_patches", 0, [geometry.patches as i32]);
    staging.i32("vision_num_features", 0, [geometry.tokens as i32]);
    staging.i32("vision_feature_offset", 0, [destination_feature_offset as i32]);
    Ok(staging)
}

/// Muse Glimmer's bracketed layout: window-ordered patches with position taps, per-cell
/// window segments padded to the bound width by repeating the patch count, and the
/// per-image reverse index (raster patch → packed row), each rebased to this image.
fn muse_glimmer(config: &ModelConfig, handle: RequestHandle, images: &ProcessedImages, image: usize, destination_feature_offset: usize) -> Result<UploadBatch> {
    let (geometry, start, end, capacity) =
        bracketed_extent(config, handle, images, image, "invalid Muse Glimmer image extent")?;
    let window_start: usize = images.geometry[..image].iter().map(super::muse_glimmer::windows).sum();
    let windows = super::muse_glimmer::windows(geometry);
    let cu_window = images
        .cu_window_seqlens
        .get(window_start..=window_start + windows)
        .ok_or("truncated vision window segments")?;
    if cu_window.first() != Some(&(start as i32)) || cu_window.last() != Some(&(end as i32)) {
        return Err("vision window segments differ from the image extent".into());
    }
    let padded = capacity / 64 + 2;
    let mut cu_window: Vec<i32> = cu_window.iter().map(|n| n - start as i32).collect();
    cu_window.resize(padded, geometry.patches as i32);
    let reverse_index = images
        .reverse_index
        .get(start..end)
        .ok_or("truncated vision reverse index")?;
    if reverse_index.iter().any(|packed| !(start as i64..end as i64).contains(packed)) {
        return Err("vision reverse index leaves the image extent".into());
    }
    let mut staging = UploadBatch::default();
    staging.put("vision_pixel_values", 0, images.pixel_values.get(start*1176..end*1176).ok_or("truncated vision pixels")?.iter().copied().flat_map(f32::to_le_bytes));
    staging.i64("vision_interpolation_indices", 0, images.interpolation_indices.get(start..end).ok_or("truncated vision interpolation indices")?.iter().flatten().copied());
    staging.put("vision_interpolation_weights", 0, images.interpolation_weights.get(start..end).ok_or("truncated vision interpolation weights")?.iter().flatten().copied().flat_map(f32::to_le_bytes));
    staging.i64("vision_position_ids", 0, images.position_ids.get(start..end).ok_or("truncated vision positions")?.iter().flatten().copied());
    staging.i32("vision_cu_seqlens", 0, [0, geometry.patches as i32]);
    staging.i32("vision_cu_window_seqlens", 0, cu_window);
    staging.i64("vision_reverse_index", 0, reverse_index.iter().map(|packed| packed - start as i64));
    staging.i32("vision_num_patches", 0, [geometry.patches as i32]);
    staging.i32("vision_feature_offset", 0, [destination_feature_offset as i32]);
    Ok(staging)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::images::{ImageLimits, ImageProcessor, ImageSettings, MuseGlimmerImageSettings, RgbImage};
    use crate::model::contract::ImagePreparationKind;

    fn glimmer_config() -> ModelConfig {
        let mut contract = crate::model::testing::metadata();
        contract.frontend.image_layout = Some(ImageLayoutKind::Bracketed);
        contract.frontend.image_preparation = Some(ImagePreparationKind::MuseGlimmer);
        contract.vision.patches_per_feature = 4;
        contract.vision.patch_capacities = vec![1024, 4096, 16384];
        contract.vision.features_per_request = 16384;
        ModelConfig {
            completion_token_capacity: 16,
            max_top_logprobs: 0,
            logprob_token_id_capacity: 0,
            prepared_sequence_tokens: 2048,
            contract,
            identity: [1; 32],
            model_id: "muse_glimmer".into(),
            limits: crate::scheduler::SchedulerLimits {
                request_slots: 4,
                max_active: 4,
                max_sequence_tokens: 2048,
                prefill_chunk_size: 512,
                pages: 64,
                snapshots: 40,
            },
            eos_token_ids: vec![1],
            sampling_ranges: Default::default(),
            sampling_defaults: crate::model::SamplingParameters::default(),
            max_new_tokens: 256,
            speculative_widths: Vec::new(),
            speculative_fixed_width: None,
            speculative_token_capacity: 512,
            decode_weight_bytes: 0,
            history_drafts: true,
            speculative_proposal_source: Default::default(),
        }
    }

    fn pattern(width: usize, height: usize) -> RgbImage {
        RgbImage::new(width, height, (0..width * height * 3).map(|i| ((i * 37 + 11) % 256) as u8).collect()).unwrap()
    }

    #[test]
    fn muse_glimmer_stages_window_tensors_rebased_per_image_and_padded_to_the_bound_width() {
        let config = glimmer_config();
        let processor = ImageProcessor::load(
            ImageSettings::MuseGlimmer(MuseGlimmerImageSettings {
                patch_size: 14,
                merge_size: 2,
                max_image_tokens: 4096,
                image_mean: [0.5; 3],
                image_std: [0.5; 3],
                image_token: "<|patch|>".into(),
                start_token: "<|image_start|>".into(),
                end_token: "<|image_end|>".into(),
            }),
            ImageLimits::default(),
        )
        .unwrap();
        // A 2×4-patch image first, then the ragged 70×38-patch image (532×980 px).
        let images = processor.process(&[pattern(56, 28), pattern(532, 980)]).unwrap();
        assert_eq!(images.geometry[1].grid_thw, [1, 70, 38]);
        let handle = RequestHandle { slot: 3, generation: 19 };
        let staging = prepare(&config, handle, &images, 1, 8).unwrap();
        let uploads = staging.uploads();
        let names: Vec<&str> = uploads.iter().map(|u| u.buffer).collect();
        assert_eq!(names, [
            "vision_pixel_values", "vision_interpolation_indices", "vision_interpolation_weights",
            "vision_position_ids", "vision_cu_seqlens", "vision_cu_window_seqlens", "vision_reverse_index",
            "vision_num_patches", "vision_feature_offset",
        ]);
        let bytes = |name: &str| uploads.iter().find(|u| u.buffer == name).unwrap().bytes;
        let i32s = |name: &str| bytes(name).chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
        let i64s = |name: &str| bytes(name).chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
        for name in names {
            let (dtype, minimum) = tensor_layout(ImageLayoutKind::Bracketed, name, 16384).unwrap();
            assert!(bytes(name).len() <= minimum * dtype.size(), "{name}");
            assert_eq!(uploads.iter().find(|u| u.buffer == name).unwrap().offset, 0);
        }
        assert_eq!(bytes("vision_pixel_values").len(), 2660 * 1176 * 4);
        assert_eq!(&bytes("vision_pixel_values")[..4], &images.pixel_values[8 * 1176].to_le_bytes());
        assert_eq!(i64s("vision_interpolation_indices").len(), 2660 * 4);
        assert_eq!(bytes("vision_interpolation_weights").len(), 2660 * 4 * 4);
        let positions = i64s("vision_position_ids");
        assert_eq!(positions.len(), 2660 * 2);
        assert_eq!(&positions[..4], &[0, 0, 0, 1]);
        assert_eq!(&positions[2 * 1024..2 * 1024 + 2], &[0, 32]);
        assert_eq!(i32s("vision_cu_seqlens"), [0, 2660]);
        let cu_window = i32s("vision_cu_window_seqlens");
        assert_eq!(cu_window.len(), 16384 / 64 + 2);
        assert_eq!(&cu_window[..7], &[0, 1024, 1216, 2240, 2432, 2624, 2660]);
        assert!(cu_window[7..].iter().all(|n| *n == 2660));
        let reverse = i64s("vision_reverse_index");
        assert_eq!(reverse.len(), 2660);
        assert_eq!(&reverse[..3], &[0, 1, 2]);
        assert_eq!(&reverse[32..34], &[1024, 1025]);
        assert_eq!(&reverse[38..40], &[32, 33]);
        for packed in 0..2660 {
            let raster = (positions[2 * packed] * 38 + positions[2 * packed + 1]) as usize;
            assert_eq!(reverse[raster], packed as i64);
        }
        assert_eq!(i32s("vision_num_patches"), [2660]);
        assert_eq!(i32s("vision_feature_offset"), [8]);
        let first = prepare(&config, handle, &images, 0, 0).unwrap();
        let first = first.uploads();
        let cu = first.iter().find(|u| u.buffer == "vision_cu_window_seqlens").unwrap();
        assert_eq!(&cu.bytes[..8], &[0, 0, 0, 0, 8, 0, 0, 0]);
        assert!(prepare(&config, RequestHandle { slot: 4, generation: 19 }, &images, 1, 8).is_err());
        assert!(prepare(&config, handle, &images, 1, 16384 - 100).is_err());
        let mut patch_merge = glimmer_config();
        patch_merge.contract.frontend.image_preparation = Some(ImagePreparationKind::PatchMerge);
        let plain = prepare(&patch_merge, handle, &images, 0, 0).unwrap();
        assert!(plain.uploads().iter().all(|u| u.buffer != "vision_cu_window_seqlens"));
        assert!(plain.uploads().iter().any(|u| u.buffer == "vision_num_features"));
    }
}
