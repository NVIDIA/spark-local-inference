/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::{
    ImageGeometry, ImageLimits, ProcessedImages, RgbImage, check_image_patches,
    check_input_extent, resize,
};
use crate::Result;

fn extent(width: usize, height: usize) -> (usize, usize) {
    let area = (width / 14).max(1) as f64 * (height / 14).max(1) as f64;
    let scale = 1.0f64
        .min((65536.0 / area).sqrt())
        .min(7168.0 / width as f64)
        .min(7168.0 / height as f64);
    (
        ((width as f64 * scale) as usize).clamp(1, 7168),
        ((height as f64 * scale) as usize).clamp(1, 7168),
    )
}

pub(super) fn geometry(limits: ImageLimits, width: usize, height: usize) -> Result<ImageGeometry> {
    check_input_extent(limits, width, height)?;
    let (w, h) = extent(width, height);
    let (width, height) = (w.div_ceil(28) * 28, h.div_ceil(28) * 28);
    let patches = width / 14 * (height / 14);
    check_image_patches(limits, patches)?;
    Ok(ImageGeometry {
        width,
        height,
        grid_thw: [1, (height / 14) as i64, (width / 14) as i64],
        patches,
        tokens: patches / 4,
    })
}

pub(super) fn resized_rgb(
    limits: ImageLimits,
    image: &RgbImage,
) -> Result<(ImageGeometry, Vec<u8>)> {
    let geometry = geometry(limits, image.width, image.height)?;
    let (width, height) = extent(image.width, image.height);
    let (mut pixels, alpha) = if let Some(cmyk) = &image.cmyk {
        let colors = RgbImage::new(
            image.width,
            image.height,
            cmyk.chunks_exact(4)
                .flat_map(|p| p[..3].iter().copied())
                .collect(),
        )?;
        let black = RgbImage::new(
            image.width,
            image.height,
            cmyk.chunks_exact(4).flat_map(|p| [p[3]; 3]).collect(),
        )?;
        let colors = resize::pillow(&colors, width, height);
        let black = resize::pillow(&black, width, height);
        (
            colors
                .into_iter()
                .zip(black)
                .map(|(c, k)| ((u32::from(255 - c) * u32::from(255 - k) + 127) / 255) as u8)
                .collect(),
            None,
        )
    } else if image.nearest_resize {
        let mut pixels = Vec::with_capacity(width * height * 3);
        let mut alpha = image
            .alpha
            .as_ref()
            .map(|_| Vec::with_capacity(width * height));
        for y in 0..height {
            let sy = (((y as f64 + 0.5) * image.height as f64 / height as f64) as usize)
                .min(image.height - 1);
            for x in 0..width {
                let sx = (((x as f64 + 0.5) * image.width as f64 / width as f64) as usize)
                    .min(image.width - 1);
                let i = sy * image.width + sx;
                pixels.extend_from_slice(&image.pixels[i * 3..i * 3 + 3]);
                if let (Some(out), Some(source)) = (&mut alpha, &image.alpha) {
                    out.push(source[i]);
                }
            }
        }
        (pixels, alpha)
    } else if let Some(alpha) = &image.alpha {
        if width == image.width && height == image.height {
            (image.pixels.clone(), Some(alpha.clone()))
        } else {
            // Pillow filters premultiplied RGBa, then converts back to RGBA.
            let premultiplied = image
                .pixels
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let product = u32::from(*v) * u32::from(alpha[i / 3]) + 128;
                    ((product + (product >> 8)) >> 8) as u8
                })
                .collect();
            let rgb = RgbImage::new(image.width, image.height, premultiplied)?;
            let a = RgbImage::new(
                image.width,
                image.height,
                alpha.iter().flat_map(|v| [*v; 3]).collect(),
            )?;
            let a = resize::pillow(&a, width, height)
                .into_iter()
                .step_by(3)
                .collect::<Vec<_>>();
            let pixels = resize::pillow(&rgb, width, height)
                .into_iter()
                .enumerate()
                .map(|(i, v)| {
                    let a = u32::from(a[i / 3]);
                    if a == 0 || a == 255 {
                        v
                    } else {
                        (255 * u32::from(v) / a).min(255) as u8
                    }
                })
                .collect();
            (pixels, Some(a))
        }
    } else {
        (resize::pillow(image, width, height), None)
    };
    if let Some(alpha) = alpha {
        for y in 0..height {
            for x in 0..width {
                let a = f32::from(alpha[y * width + x]) / 255.0;
                let background = if (y / 8 + x / 8) % 2 == 0 {
                    255.0
                } else {
                    180.0
                };
                for c in 0..3 {
                    let v = &mut pixels[(y * width + x) * 3 + c];
                    *v = (a * f32::from(*v) + (1.0 - a) * background) as u8;
                }
            }
        }
    }
    let mut padded = vec![0; geometry.width * geometry.height * 3];
    for y in 0..height {
        padded[y * geometry.width * 3..y * geometry.width * 3 + width * 3]
            .copy_from_slice(&pixels[y * width * 3..(y + 1) * width * 3]);
    }
    Ok((geometry, padded))
}

pub(super) fn process(limits: ImageLimits, images: &[RgbImage]) -> Result<ProcessedImages> {
    limits.check_batch(images.len(), 0)?;
    let mut result = ProcessedImages {
        pixel_values: vec![],
        geometry: vec![],
        position_ids: vec![],
        interpolation_indices: vec![],
        interpolation_weights: vec![],
        cu_seqlens: vec![0],
    };
    let geometries: Vec<_> = images
        .iter()
        .map(|image| geometry(limits, image.width, image.height))
        .collect::<Result<_>>()?;
    limits.check_batch(images.len(), geometries.iter().map(|g| g.patches).sum())?;
    result.reserve_outputs(geometries.iter().map(|g| g.patches).sum(), 588, true,
        images.len(), images.len() + 1)?;
    for (image, geometry) in images.iter().zip(geometries) {
        let (_, pixels) = resized_rgb(limits, image)?;
        let h = geometry.height / 14;
        let w = geometry.width / 14;
        for y in 0..h {
            for x in 0..w {
                result.position_ids.push([y as i64, x as i64]);
                let (yi, yw) = position(y, h);
                let (xi, xw) = position(x, w);
                result.interpolation_indices.push([
                    yi[0] * 64 + xi[0],
                    yi[0] * 64 + xi[1],
                    yi[1] * 64 + xi[0],
                    yi[1] * 64 + xi[1],
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
                            let v = pixels[((y * 14 + dy) * geometry.width + x * 14 + dx) * 3 + c];
                            result
                                .pixel_values
                                .push(((f64::from(v) / 255.0) as f32 - 0.5) * 2.0);
                        }
                    }
                }
            }
        }
        result.geometry.push(geometry);
        result.cu_seqlens.push(result.position_ids.len() as i32);
    }
    Ok(result)
}

fn position(index: usize, extent: usize) -> ([i64; 2], [f32; 2]) {
    let source = (64.0f32 / extent as f32 * (index as f32 + 0.5) - 0.5).max(0.0);
    let lo = source as i64;
    let fraction = source - lo as f32;
    ([lo, (lo + 1).min(63)], [1.0 - fraction, fraction])
}
