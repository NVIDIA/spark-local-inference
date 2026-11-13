/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::{ImageGeometry, ImageLimits, ProcessedImages, RgbImage, check_image_patches, check_input_extent, resize};
use crate::Result;

const MIN_PATCHES: usize = 1024;
const MAX_PATCHES: usize = 13312;
const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

pub(super) fn geometry(limits: ImageLimits, width: usize, height: usize) -> Result<ImageGeometry> {
    check_input_extent(limits, width, height)?;
    let h = (height as f64 / 16.0 + 0.5).round_ties_even() as usize;
    let w = (width as f64 / 16.0 + 0.5).round_ties_even() as usize;
    let scale = (MAX_PATCHES as f64 / (h * w) as f64).sqrt().min(1.0);
    let mut h = (scale * h as f64).floor() as usize;
    let mut w = (scale * w as f64).floor() as usize;
    if h == 0 || w == 0 {
        return Err("image aspect ratio produces an empty RADIO patch grid".into());
    }
    if h * w < MIN_PATCHES {
        let scale = (MIN_PATCHES as f64 / (h * w) as f64).sqrt();
        h = (scale * h as f64).ceil() as usize;
        w = (scale * w as f64).ceil() as usize;
    }
    if h % 2 != 0 {
        h = if (h + 1) * w <= MAX_PATCHES {
            h + 1
        } else {
            (h - 1).max(2)
        };
    }
    if w % 2 != 0 {
        w = if h * (w + 1) <= MAX_PATCHES {
            w + 1
        } else {
            (w - 1).max(2)
        };
    }
    let patches = h * w;
    check_image_patches(limits, patches)?;
    if patches > MAX_PATCHES {
        return Err("image exceeds prepared RADIO patch capacity".into());
    }
    Ok(ImageGeometry {
        width: w * 16,
        height: h * 16,
        grid_thw: [1, h as i64, w as i64],
        patches,
        tokens: patches / 4,
    })
}

fn position(index: usize, extent: usize) -> ([i64; 2], [f32; 2]) {
    let source = (128.0f32 / extent as f32 * (index as f32 + 0.5) - 0.5).max(0.0);
    let lo = source as i64;
    let fraction = source - lo as f32;
    ([lo, (lo + 1).min(127)], [1.0 - fraction, fraction])
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
    result.reserve_outputs(geometries.iter().map(|g| g.patches).sum(), 768, true,
        images.len(), images.len() + 1)?;
    for (image, geometry) in images.iter().zip(geometries) {
        let pixels = resize::float_bicubic(image, geometry.width, geometry.height);
        let (h, w) = (geometry.height / 16, geometry.width / 16);
        // RADIO interpolates to a square and crops its top-left rectangle.
        let side = h.max(w);
        for y in 0..h {
            for x in 0..w {
                result.position_ids.push([y as i64, x as i64]);
                let (yi, yw) = position(y, side);
                let (xi, xw) = position(x, side);
                result.interpolation_indices.push([
                    yi[0] * 128 + xi[0],
                    yi[0] * 128 + xi[1],
                    yi[1] * 128 + xi[0],
                    yi[1] * 128 + xi[1],
                ]);
                result.interpolation_weights.push([
                    yw[0] * xw[0],
                    yw[0] * xw[1],
                    yw[1] * xw[0],
                    yw[1] * xw[1],
                ]);
                for c in 0..3 {
                    for dy in 0..16 {
                        for dx in 0..16 {
                            let value =
                                pixels[((y * 16 + dy) * geometry.width + x * 16 + dx) * 3 + c];
                            result.pixel_values.push((value / 255.0 - MEAN[c]) / STD[c]);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> serde_json::Value {
        serde_json::from_str(include_str!("../../../../tests/frontend/radio.json")).unwrap()
    }

    fn pattern(width: usize, height: usize) -> RgbImage {
        RgbImage::new(
            width,
            height,
            (0..width * height * 3)
                .map(|i| ((i * 37 + 11) % 256) as u8)
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn geometry_matches_checkpoint_processor() {
        for case in fixture()["geometry"].as_array().unwrap() {
            let w = case["input"][0].as_u64().unwrap() as usize;
            let h = case["input"][1].as_u64().unwrap() as usize;
            let actual = geometry(ImageLimits::default(), w, h).unwrap();
            assert_eq!(
                serde_json::json!(&actual.grid_thw[1..]),
                case["grid"],
                "{w}x{h}"
            );
            assert_eq!(actual.tokens * 4, actual.patches);
        }
        let small = ImageLimits {
            max_patches_per_image: 512,
            ..ImageLimits::default()
        };
        assert!(geometry(small, 37, 29).is_err());
        assert!(geometry(ImageLimits::default(), 0, 512).is_err());
        assert!(geometry(ImageLimits::default(), usize::MAX, 512).is_err());
    }

    #[test]
    fn float_resize_matches_torch_antialiased_bicubic() {
        for case in fixture()["resize"].as_array().unwrap() {
            let dims = |field: &str| {
                (
                    case[field][0].as_u64().unwrap() as usize,
                    case[field][1].as_u64().unwrap() as usize,
                )
            };
            let (w, h) = dims("input");
            let (ow, oh) = dims("output");
            let actual = resize::float_bicubic(&pattern(w, h), ow, oh);
            for (a, e) in actual.iter().zip(case["pixels"].as_array().unwrap()) {
                let expected = e.as_f64().unwrap() as f32;
                assert!(
                    (a - expected).abs() < 0.00015,
                    "{w}x{h}->{ow}x{oh}: {a} != {expected}"
                );
            }
        }
    }

    #[test]
    fn normalization_and_patch_order_match_checkpoint_processor() {
        let f = fixture();
        let case = &f["normalization"];
        let image = pattern(
            case["input"][0].as_u64().unwrap() as usize,
            case["input"][1].as_u64().unwrap() as usize,
        );
        let result = process(ImageLimits::default(), &[image]).unwrap();
        assert_eq!(result.pixel_values.len(), result.geometry[0].patches * 768);
        for (i, e) in case["indices"]
            .as_array()
            .unwrap()
            .iter()
            .zip(case["pixels"].as_array().unwrap())
        {
            let a = result.pixel_values[i.as_u64().unwrap() as usize];
            let expected = e.as_f64().unwrap() as f32;
            assert!((a - expected).abs() < 0.000003, "{a} != {expected}");
        }
        for (i, w) in result
            .interpolation_indices
            .iter()
            .zip(&result.interpolation_weights)
        {
            assert!(i.iter().all(|i| (0..16384).contains(i)));
            assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
        let limited = ImageLimits {
            max_total_patches: 1024,
            ..ImageLimits::default()
        };
        assert!(process(limited, &[pattern(37, 29), pattern(37, 29)]).is_err());
    }
}
