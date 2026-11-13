/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Resize a patch grid whose language tokens are start, raster rows, and end.

use super::*;

const PATCH: usize = 14;
const CELL: usize = PATCH * 3;

pub(super) fn validate(settings: &DeepseekRowMajorImageSettings) -> Result<()> {
    if settings.min_pixels == 0 || !(4..=1024).contains(&settings.max_tokens)
        || settings.min_pixels > 64 * 1024 * 1024
        || settings.max_width_height_ratio == Some(0)
    {
        return Err("invalid row-major image resize settings".into());
    }
    Ok(())
}

pub(super) fn geometry(
    settings: &DeepseekRowMajorImageSettings, limits: ImageLimits, width: usize, height: usize,
) -> Result<ImageGeometry> {
    check_input_extent(limits, width, height)?;
    let mut width = settings.max_width_height_ratio.map_or(width, |ratio| width.min(height.saturating_mul(ratio)));
    let mut height = height;
    if width * height < settings.min_pixels {
        let ratio = (settings.min_pixels as f64 / (width * height) as f64).sqrt();
        width = (width as f64 * ratio) as usize;
        height = (height as f64 * ratio) as usize;
    }
    let (mut h, mut w) = (height.next_multiple_of(PATCH), width.next_multiple_of(PATCH));
    let cost = |h: usize, w: usize| (h / PATCH).div_ceil(3) * ((w / PATCH).div_ceil(3) + 1) + 2;
    if cost(h, w) > settings.max_tokens {
        let ratio = height as f64 / width as f64;
        let max_w = ((settings.max_tokens - 2) as f64 / ratio + 0.25).sqrt() - 0.5;
        let max_h = max_w * ratio;
        if max_w < 1. {
            (h, w) = ((settings.max_tokens - 2) / 2 * CELL, CELL);
        } else if max_h < 1. {
            (h, w) = (CELL, (settings.max_tokens - 3) * CELL);
        } else {
            let scale = (max_w.floor() * CELL as f64 / width as f64)
                .min(max_h.floor() * CELL as f64 / height as f64);
            h = (height as f64 * scale / PATCH as f64).floor() as usize * PATCH;
            w = (width as f64 * scale / PATCH as f64).floor() as usize * PATCH;
        }
    }
    let tokens = (h / PATCH).div_ceil(3) * (w / PATCH).div_ceil(3);
    let patches = tokens * 9;
    check_image_patches(limits, patches)?;
    if h == 0 || w == 0 || cost(h, w) > settings.max_tokens {
        return Err("processed image exceeds the recipe's image token limit".into());
    }
    Ok(ImageGeometry { height: h, width: w, grid_thw: [1, (h / PATCH) as i64, (w / PATCH) as i64], patches, tokens })
}

pub(super) fn resized_rgb(
    settings: &DeepseekRowMajorImageSettings, limits: ImageLimits, image: &RgbImage,
) -> Result<(ImageGeometry, Vec<u8>)> {
    let g = geometry(settings, limits, image.width, image.height)?;
    if settings.max_width_height_ratio.is_some_and(|ratio| image.width >= image.height.saturating_mul(ratio))
        || image.width * g.height == image.height * g.width
    {
        return Ok((g, resize::pillow(image, g.width, g.height)));
    }
    super::deepseek4::pad(image, g)
}
