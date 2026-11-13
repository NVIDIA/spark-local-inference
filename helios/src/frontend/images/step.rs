/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Adaptive local crops followed by a global view for the Step perception encoder.

use super::{
    ImageGeometry, ImageLimits, ProcessedImages, RgbImage, check_image_patches,
    check_input_extent, resize,
};
use crate::Result;

pub(super) const MAX_VIEWS: usize = (65536 - 52 * 52) / (36 * 36) + 1;

struct Plan {
    width: usize,
    height: usize,
    crop_width: usize,
    crop_height: usize,
    window: usize,
    nx: usize,
    ny: usize,
    square: bool,
}

fn plan(mut width: usize, mut height: usize) -> Plan {
    let square = width.min(height) < 32 && width.max(height) > 4 * width.min(height);
    if square {
        width = width.max(height);
        height = width;
    }
    if width.max(height) > 3024 {
        let scale = 3024.0 / width.max(height) as f64;
        width = (width as f64 * scale) as usize;
        height = (height as f64 * scale) as usize;
    }
    let (short, long) = (width.min(height), width.max(height));
    let window = if long <= 728 {
        if 2 * long > 3 * short { short } else { 0 }
    } else if long > 4 * short {
        short.min(504)
    } else {
        504
    };
    let snap = |dim: usize| {
        if window == 0 || dim < window {
            dim
        } else {
            window * (dim / window + usize::from(5 * (dim % window) > window))
        }
    };
    let (crop_width, crop_height) = (snap(width), snap(height));
    let (nx, ny) = if window == 0 {
        (0, 0)
    } else {
        ((crop_width / window).max(1), (crop_height / window).max(1))
    };
    Plan {
        width,
        height,
        crop_width,
        crop_height,
        window,
        nx,
        ny,
        square,
    }
}

pub(super) fn geometry(limits: ImageLimits, width: usize, height: usize) -> Result<ImageGeometry> {
    check_input_extent(limits, width, height)?;
    let p = plan(width, height);
    if p.square {
        check_input_extent(limits, width.max(height), width.max(height))?;
    }
    let views = p.nx * p.ny;
    let patches = views * 36 * 36 + 52 * 52;
    check_image_patches(limits, patches)?;
    if views + 1 > MAX_VIEWS {
        return Err("image exceeds prepared Step crop capacity".into());
    }
    Ok(ImageGeometry {
        width: p.width,
        height: p.height,
        grid_thw: [1, p.ny as i64, p.nx as i64],
        patches,
        tokens: patches / 16,
    })
}

fn padded(image: &RgbImage, p: &Plan) -> Result<RgbImage> {
    let (w, h) = if p.square {
        let side = image.width.max(image.height);
        (side, side)
    } else {
        (image.width, image.height)
    };
    let mut pixels = vec![0; w * h * 3];
    for y in 0..image.height {
        pixels[y * w * 3..y * w * 3 + image.width * 3]
            .copy_from_slice(&image.pixels[y * image.width * 3..(y + 1) * image.width * 3]);
    }
    RgbImage::new(w, h, pixels)
}

fn overview(image: &RgbImage, p: &Plan) -> Result<Vec<u8>> {
    let global = RgbImage::new(
        p.width,
        p.height,
        resize::bilinear(image, p.width, p.height),
    )?;
    Ok(resize::bilinear(&global, 728, 728))
}

pub(super) fn resized_rgb(
    limits: ImageLimits,
    image: &RgbImage,
) -> Result<(ImageGeometry, Vec<u8>)> {
    let g = geometry(limits, image.width, image.height)?;
    let p = plan(image.width, image.height);
    Ok((g, overview(&padded(image, &p)?, &p)?))
}

fn append_view(result: &mut ProcessedImages, pixels: &[u8], side: usize) {
    const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
    const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];
    let grid = side / 14;
    let position = |i: usize| {
        let source = (52.0f32 / grid as f32 * (i as f32 + 0.5) - 0.5).max(0.0);
        let lo = source as i64;
        let fraction = source - lo as f32;
        ([lo, (lo + 1).min(51)], [1.0 - fraction, fraction])
    };
    for y in 0..grid {
        for x in 0..grid {
            result.position_ids.push([y as i64, x as i64]);
            let (yi, yw) = position(y);
            let (xi, xw) = position(x);
            result.interpolation_indices.push([
                yi[0] * 52 + xi[0],
                yi[0] * 52 + xi[1],
                yi[1] * 52 + xi[0],
                yi[1] * 52 + xi[1],
            ]);
            result.interpolation_weights.push([
                yw[0] * xw[0],
                yw[0] * xw[1],
                yw[1] * xw[0],
                yw[1] * xw[1],
            ]);
            for c in 0..3 {
                for dy in 0..14 {
                    for dx in 0..14 {
                        let v = pixels[((y * 14 + dy) * side + x * 14 + dx) * 3 + c];
                        result
                            .pixel_values
                            .push((f32::from(v) * (1.0 / 255.0) - MEAN[c]) / STD[c]);
                    }
                }
            }
        }
    }
    result.cu_seqlens.push(result.position_ids.len() as i32);
}

pub(super) fn process(limits: ImageLimits, images: &[RgbImage]) -> Result<ProcessedImages> {
    limits.check_batch(images.len(), 0)?;
    let geometry: Vec<_> = images
        .iter()
        .map(|image| geometry(limits, image.width, image.height))
        .collect::<Result<_>>()?;
    limits.check_batch(images.len(), geometry.iter().map(|g| g.patches).sum())?;
    let mut result = ProcessedImages {
        pixel_values: vec![],
        geometry,
        position_ids: vec![],
        interpolation_indices: vec![],
        interpolation_weights: vec![],
        cu_seqlens: vec![0],
    };
    let patches = result.geometry.iter().map(|g| g.patches).sum();
    let sequences = 1 + result.geometry.iter().map(|g| (g.grid_thw[1] * g.grid_thw[2]) as usize + 1).sum::<usize>();
    result.reserve_outputs(patches, 588, true, images.len(), sequences)?;
    for image in images {
        let p = plan(image.width, image.height);
        let padded = padded(image, &p)?;
        if p.window > 0 {
            let crop = resize::bilinear(&padded, p.crop_width, p.crop_height);
            for row in 0..p.ny {
                for col in 0..p.nx {
                    let w = p.window.min(p.crop_width - col * p.window);
                    let h = p.window.min(p.crop_height - row * p.window);
                    let pixels = (0..h)
                        .flat_map(|dy| {
                            let start = ((row * p.window + dy) * p.crop_width + col * p.window) * 3;
                            crop[start..start + w * 3].iter().copied()
                        })
                        .collect();
                    let view = RgbImage::new(w, h, pixels)?;
                    append_view(&mut result, &resize::bilinear(&view, 504, 504), 504);
                }
            }
        }
        append_view(&mut result, &overview(&padded, &p)?, 728);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_geometry_and_capacity() {
        let limits = ImageLimits {
            max_patches_per_image: 65536,
            max_total_patches: 65536,
            ..ImageLimits::default()
        };
        for (w, h, rows, cols) in [
            (500, 500, 0, 0),
            (700, 200, 1, 4),
            (200, 700, 4, 1),
            (1000, 900, 2, 2),
            (4000, 4000, 6, 6),
            (1000, 16, 2, 2),
        ] {
            let g = geometry(limits, w, h).unwrap();
            assert_eq!(g.grid_thw, [1, rows, cols]);
            assert_eq!(g.tokens, 169 + rows as usize * cols as usize * 81);
        }
        assert!(geometry(limits, 3024, 32).is_err());
    }
}
