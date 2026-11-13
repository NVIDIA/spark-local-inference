/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;

const PATCH: usize = 14;
const MERGE: usize = 3;
const FEATURES: usize = 3 * PATCH * PATCH;

fn grid_tokens(height: usize, width: usize) -> usize {
    let rows = (height / PATCH).div_ceil(MERGE).next_multiple_of(2);
    let columns = (width / PATCH).div_ceil(MERGE) + 1;
    rows * columns + 2 + (rows / 2 * columns % 2) * 2
}

pub(super) fn geometry(limits: ImageLimits, width: usize, height: usize) -> Result<ImageGeometry> {
    check_input_extent(limits, width, height)?;
    let mut width = width.min(height * 8);
    let mut height = height;
    if width * height < 147456 {
        let ratio = (147456. / (width * height) as f64).sqrt();
        width = (width as f64 * ratio) as usize;
        height = (height as f64 * ratio) as usize;
    }
    let (mut h, mut w) = (
        height.next_multiple_of(PATCH),
        width.next_multiple_of(PATCH),
    );
    let mut budget = 381usize;
    while grid_tokens(h, w) > 381 {
        let ratio = height as f64 / width as f64;
        let max_w = ((budget - 2) as f64 / ratio + 0.25).sqrt() - 0.5;
        let max_h = max_w * ratio;
        if max_w < 1. {
            w = PATCH * MERGE;
            h = ((budget - 2) / 2) / 2 * 2 * PATCH * MERGE;
        } else if max_h < 2. {
            h = 2 * PATCH * MERGE;
            w = ((budget - 2) / 2 - 1) * PATCH * MERGE;
        } else {
            let scale = ((max_w.floor() as usize * PATCH * MERGE) as f64 / width as f64)
                .min(((max_h.floor() as usize / 2 * 2) * PATCH * MERGE) as f64 / height as f64);
            w = (width as f64 * scale / PATCH as f64).floor() as usize * PATCH;
            h = (height as f64 * scale / PATCH as f64).floor() as usize * PATCH;
        }
        budget = budget.checked_sub(1).ok_or("invalid image resize budget")?;
    }
    let tokens = (h / PATCH).div_ceil(MERGE) * (w / PATCH).div_ceil(MERGE);
    let patches = tokens * MERGE * MERGE;
    check_image_patches(limits, patches)?;
    Ok(ImageGeometry {
        height: h,
        width: w,
        grid_thw: [1, (h / PATCH) as i64, (w / PATCH) as i64],
        patches,
        tokens,
    })
}

pub(super) fn resized_rgb(
    limits: ImageLimits,
    image: &RgbImage,
) -> Result<(ImageGeometry, Vec<u8>)> {
    let g = geometry(limits, image.width, image.height)?;
    if image.width >= 8 * image.height || image.width * g.height == image.height * g.width {
        return Ok((g, resize::pillow(image, g.width, g.height)));
    }
    pad(image, g)
}

/// Pillow ImageOps.pad after RGB conversion, including ties-to-even centering.
pub(super) fn pad(image: &RgbImage, g: ImageGeometry) -> Result<(ImageGeometry, Vec<u8>)> {
    // Match Pillow ImageOps.contain, then center its RGB result on a gray canvas.
    let source_ratio = image.width as f64 / image.height as f64;
    let target_ratio = g.width as f64 / g.height as f64;
    let (width, height) = if source_ratio > target_ratio {
        (
            g.width,
            (g.width as f64 / source_ratio).round_ties_even() as usize,
        )
    } else {
        (
            (g.height as f64 * source_ratio).round_ties_even() as usize,
            g.height,
        )
    };
    let resized = resize::pillow(image, width.max(1), height.max(1));
    let mut pixels = vec![127; g.width * g.height * 3];
    let x = ((g.width - width) as f64 / 2.).round_ties_even() as usize;
    let y = ((g.height - height) as f64 / 2.).round_ties_even() as usize;
    for row in 0..height {
        let start = ((y + row) * g.width + x) * 3;
        pixels[start..start + width * 3]
            .copy_from_slice(&resized[row * width * 3..(row + 1) * width * 3]);
    }
    Ok((g, pixels))
}

pub(super) fn process(limits: ImageLimits, images: &[RgbImage]) -> Result<ProcessedImages> {
    pack(
        limits,
        images,
        |image| geometry(limits, image.width, image.height),
        |image| resized_rgb(limits, image),
    )
}

/// Packs every image's patches after checking the whole batch against the request budget,
/// so a rejection reports the complete total rather than the images packed so far.
pub(super) fn pack(
    limits: ImageLimits, images: &[RgbImage],
    geometry: impl Fn(&RgbImage) -> Result<ImageGeometry>,
    resize: impl Fn(&RgbImage) -> Result<(ImageGeometry, Vec<u8>)>,
) -> Result<ProcessedImages> {
    limits.check_batch(images.len(), 0)?;
    let total = images
        .iter()
        .map(|image| geometry(image).map(|g| g.patches))
        .try_fold(0usize, |sum, patches| patches.map(|p| sum.saturating_add(p)))?;
    limits.check_batch(images.len(), total)?;
    let mut output = ProcessedImages {
        pixel_values: Vec::new(),
        geometry: Vec::new(),
        position_ids: Vec::new(),
        interpolation_indices: Vec::new(),
        interpolation_weights: Vec::new(),
        cu_seqlens: vec![0],
    };
    output.reserve_outputs(total, FEATURES, false, images.len(), images.len() + 1)?;
    for image in images {
        let (g, pixels) = resize(image)?;
        let h = g.height / PATCH;
        let w = g.width / PATCH;
        for row in 0..h {
            for column in 0..w {
                output.position_ids.push([row as i64, column as i64]);
                for channel in 0..3 {
                    for y in 0..PATCH {
                        for x in 0..PATCH {
                            let byte = pixels
                                [((row * PATCH + y) * g.width + column * PATCH + x) * 3 + channel];
                            output.pixel_values.push((byte as f32 / 255. - 0.5) / 0.5);
                        }
                    }
                }
            }
        }
        output
            .position_ids
            .extend(std::iter::repeat_n([0, 0], g.patches - h * w));
        output
            .pixel_values
            .resize(output.position_ids.len() * FEATURES, 0.);
        output
            .cu_seqlens
            .push(output.cu_seqlens.last().unwrap() + (h * w) as i32);
        output.geometry.push(g);
    }
    Ok(output)
}
