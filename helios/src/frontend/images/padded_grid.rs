/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Unresized padded-grid preparation.
//!
//! Images keep their decoded size. Retained alpha is composited onto an 8-pixel
//! checkerboard of 204 and 153 as `(rgb·a + background·(255 − a) + 127) / 255`. The
//! image is padded on the bottom and right to whole 40-pixel token cells with −1 on the
//! 0–255 scale, and every value is normalized as `(v − 127.5) / 127.5`. Each cell holds
//! 2×2 native 20-pixel patches. Patches follow token raster order and, within a token,
//! merge-block row-major order; each patch has 1,200 features in pixel row, pixel column,
//! channel order. Grids count native patches. A learned 32×32 position table is sampled
//! with four bilinear taps at half-pixel centres (`align_corners=False`), clamped to the
//! table edge. Sequence offsets count one class token before each image's patches.

use std::borrow::Cow;

use super::{
    ImageGeometry, ImageLimits, ProcessedImages, RgbImage, check_image_patches,
    check_input_extent,
};
use crate::Result;

const PATCH: usize = 20;
const MERGE: usize = 2;
const CELL: usize = PATCH * MERGE;
const POSITION_SIDE: usize = 32;
pub(super) const FEATURES: usize = PATCH * PATCH * 3;
const CHECKER_SQUARE: usize = 8;
const CHECKER_LIGHT: u32 = 204;
const CHECKER_DARK: u32 = 153;
const PAD_VALUE: f32 = -1.0;

pub(super) fn geometry(limits: ImageLimits, width: usize, height: usize) -> Result<ImageGeometry> {
    check_input_extent(limits, width, height)?;
    let (rows, columns) = (height.div_ceil(CELL), width.div_ceil(CELL));
    let tokens = rows.checked_mul(columns).ok_or("image token count overflows")?;
    let patches = tokens
        .checked_mul(MERGE * MERGE)
        .ok_or("image patch count overflows")?;
    check_image_patches(limits, patches)?;
    Ok(ImageGeometry {
        height: rows * CELL,
        width: columns * CELL,
        grid_thw: [1, (rows * MERGE) as i64, (columns * MERGE) as i64],
        patches,
        tokens,
    })
}

/// RGB with retained alpha composited onto the checkerboard anchored at the top-left pixel.
fn composited(image: &RgbImage) -> Result<Cow<'_, [u8]>> {
    let Some(alpha) = &image.alpha else {
        return Ok(Cow::Borrowed(&image.pixels));
    };
    if alpha.len() != image.width * image.height {
        return Err("image alpha extent differs from its pixels".into());
    }
    let mut pixels = Vec::with_capacity(image.pixels.len());
    for (index, (rgb, alpha)) in image.pixels.chunks_exact(3).zip(alpha).enumerate() {
        let (y, x) = (index / image.width, index % image.width);
        let background = if (y / CHECKER_SQUARE + x / CHECKER_SQUARE) % 2 == 0 {
            CHECKER_LIGHT
        } else {
            CHECKER_DARK
        };
        let alpha = u32::from(*alpha);
        pixels.extend(
            rgb.iter()
                .map(|value| ((u32::from(*value) * alpha + background * (255 - alpha) + 127) / 255) as u8),
        );
    }
    Ok(Cow::Owned(pixels))
}

/// Two taps and weights along one axis of `size` native patches, in the float32 operation
/// order of Transformers `_interpolation_axis_taps_weights` with `align_corners=False`.
fn interpolation_axis(index: usize, size: usize) -> ([usize; 2], [f32; 2]) {
    let source = (index as f32 + 0.5) * POSITION_SIDE as f32 / size as f32 - 0.5;
    let floor = source.floor();
    let distance = source - floor;
    let tap = |offset: i64| (floor as i64 + offset).clamp(0, POSITION_SIDE as i64 - 1) as usize;
    (
        [tap(0), tap(1)],
        [(1.0 - distance).max(0.0), (1.0 - (distance - 1.0).abs()).max(0.0)],
    )
}

pub(super) fn process(limits: ImageLimits, images: &[RgbImage]) -> Result<ProcessedImages> {
    limits.check_batch(images.len(), 0)?;
    let geometry = images
        .iter()
        .map(|image| geometry(limits, image.width, image.height))
        .collect::<Result<Vec<_>>>()?;
    let patches = geometry
        .iter()
        .try_fold(0usize, |sum, image| sum.checked_add(image.patches))
        .ok_or("image patch count overflows")?;
    limits.check_batch(images.len(), patches)?;
    let mut result = ProcessedImages {
        geometry,
        cu_seqlens: vec![0],
        ..Default::default()
    };
    result.reserve_outputs(patches, FEATURES, true, images.len(), images.len() + 1)?;
    let normalized: [f32; 256] = std::array::from_fn(|value| (value as f32 - 127.5) / 127.5);
    let padding = (PAD_VALUE - 127.5) / 127.5;
    for (image, geometry) in images.iter().zip(&result.geometry) {
        let pixels = composited(image)?;
        let grid_h = geometry.height / PATCH;
        let grid_w = geometry.width / PATCH;
        for block_row in 0..grid_h / MERGE {
            for block_col in 0..grid_w / MERGE {
                for dy in 0..MERGE {
                    for dx in 0..MERGE {
                        let (row, col) = (block_row * MERGE + dy, block_col * MERGE + dx);
                        result.position_ids.push([row as i64, col as i64]);
                        let (rows, row_weights) = interpolation_axis(row, grid_h);
                        let (cols, col_weights) = interpolation_axis(col, grid_w);
                        result.interpolation_indices.push([
                            (rows[0] * POSITION_SIDE + cols[0]) as i64,
                            (rows[0] * POSITION_SIDE + cols[1]) as i64,
                            (rows[1] * POSITION_SIDE + cols[0]) as i64,
                            (rows[1] * POSITION_SIDE + cols[1]) as i64,
                        ]);
                        result.interpolation_weights.push([
                            row_weights[0] * col_weights[0],
                            row_weights[0] * col_weights[1],
                            row_weights[1] * col_weights[0],
                            row_weights[1] * col_weights[1],
                        ]);
                        for y in row * PATCH..(row + 1) * PATCH {
                            for x in col * PATCH..(col + 1) * PATCH {
                                if y < image.height && x < image.width {
                                    let offset = (y * image.width + x) * 3;
                                    result.pixel_values.extend(
                                        pixels[offset..offset + 3]
                                            .iter()
                                            .map(|value| normalized[*value as usize]),
                                    );
                                } else {
                                    result.pixel_values.extend([padding; 3]);
                                }
                            }
                        }
                    }
                }
            }
        }
        let sequence = i32::try_from(geometry.patches + 1)
            .ok()
            .and_then(|length| result.cu_seqlens.last().unwrap().checked_add(length))
            .ok_or("image sequence offsets exceed int32")?;
        result.cu_seqlens.push(sequence);
    }
    Ok(result)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::frontend::images::{ImageProcessor, ImageSettings};
    use serde_json::Value;
    use sha2::{Digest, Sha256};
    use std::{fs, path::PathBuf};

    pub(crate) fn fixture_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/frontend")
    }

    pub(crate) fn fixture() -> Value {
        serde_json::from_slice(&fs::read(fixture_root().join("inkling2_images.json")).unwrap()).unwrap()
    }

    pub(crate) fn processor() -> ImageProcessor {
        ImageProcessor::load(
            ImageSettings::PaddedGrid {},
            ImageLimits {
                max_patches_per_image: 42436,
                max_total_patches: 65536,
                patches_per_feature: 4,
                ..ImageLimits::default()
            },
        )
        .unwrap()
    }

    /// The xorshift byte sequence of `generate_inkling2_image_fixtures.py`.
    fn pattern(width: usize, height: usize, seed: u32) -> Vec<u8> {
        let mut state = seed;
        (0..width * height * 3)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as u8
            })
            .collect()
    }

    /// Images through the native decoders, or generated patterns. JPEG inputs are compared
    /// with Pillow's RGB within the recorded channel tolerance, then replaced by Pillow's RGB
    /// so preprocessing is checked exactly.
    pub(crate) fn inputs(processor: &ImageProcessor, case: &Value) -> Vec<RgbImage> {
        let root = fixture_root();
        let name = case["name"].as_str().unwrap();
        case["inputs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|input| {
                if let Some(pattern_case) = input.get("pattern") {
                    let [width, height, seed] = ["width", "height", "seed"]
                        .map(|key| pattern_case[key].as_u64().unwrap() as usize);
                    return RgbImage::new(width, height, pattern(width, height, seed as u32)).unwrap();
                }
                let bytes = fs::read(root.join(input["file"].as_str().unwrap())).unwrap();
                assert_eq!(format!("{:x}", Sha256::digest(&bytes)), input["sha256"], "{name}");
                let decoded = match input["format"].as_str().unwrap() {
                    "PNG" => processor.decode_png(&bytes),
                    "JPEG" => processor.decode_jpeg(&bytes),
                    "WEBP" => processor.decode_webp(&bytes),
                    format => panic!("unexpected fixture format {format}"),
                }
                .unwrap();
                let size = [input["width"].as_u64().unwrap() as usize, input["height"].as_u64().unwrap() as usize];
                assert_eq!([decoded.width(), decoded.height()], size, "{name} decoded orientation");
                let Some(rgb) = input.get("rgb") else {
                    return decoded;
                };
                let pillow = fs::read(root.join(rgb.as_str().unwrap())).unwrap();
                let error = decoded.pixels().iter().zip(&pillow).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
                assert_eq!(decoded.pixels().len(), pillow.len(), "{name}");
                assert!(u64::from(error) <= input["max_channel_error"].as_u64().unwrap(), "{name}: {error}");
                RgbImage::new(size[0], size[1], pillow).unwrap()
            })
            .collect()
    }

    pub(crate) fn hash(bytes: impl IntoIterator<Item = u8>) -> String {
        let mut digest = Sha256::new();
        digest.update(bytes.into_iter().collect::<Vec<_>>());
        format!("{:x}", digest.finalize())
    }

    #[test]
    fn reference_processor_tensors_match_byte_for_byte() {
        let fixture = fixture();
        let processor = processor();
        let cases = fixture["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 26);
        for case in cases {
            let name = case["name"].as_str().unwrap();
            let images = inputs(&processor, case);
            for (index, image) in images.iter().enumerate() {
                let rgb = composited(image).unwrap();
                assert_eq!(hash(rgb.iter().copied()), case["composited_rgb_sha256"][index], "{name} composite");
            }
            let processed = processor.process(&images).unwrap();
            let grids: Vec<_> = processed
                .geometry
                .iter()
                .map(|geometry| [1, geometry.grid_thw[1] / 2, geometry.grid_thw[2] / 2])
                .collect();
            assert_eq!(serde_json::json!(grids), case["grid_thw"], "{name}");
            let tokens: Vec<_> = processed.geometry.iter().map(|g| g.tokens).collect();
            assert_eq!(serde_json::json!(tokens), case["tokens"], "{name}");
            assert_eq!(serde_json::json!(processed.cu_seqlens), case["cu_seqlens"], "{name}");
            let hashes = &case["hashes"];
            assert_eq!(
                hash(processed.pixel_values.iter().flat_map(|v| v.to_le_bytes())),
                hashes["native_patches"],
                "{name} pixels"
            );
            assert_eq!(
                hash(processed.position_ids.iter().flatten().flat_map(|v| v.to_le_bytes())),
                hashes["native_positions"],
                "{name} positions"
            );
            assert_eq!(
                hash(processed.interpolation_indices.iter().flatten().flat_map(|v| v.to_le_bytes())),
                hashes["interpolation_indices"],
                "{name} interpolation indices"
            );
            assert_eq!(
                hash(processed.interpolation_weights.iter().flatten().flat_map(|v| v.to_le_bytes())),
                hashes["interpolation_weights"],
                "{name} interpolation weights"
            );
        }
    }

    #[test]
    fn transparency_and_orientation_follow_the_reference_decoder() {
        let root = fixture_root().join("inkling2_images-inputs");
        let processor = processor();
        for (file, alpha) in [("RGB.png", false), ("RGBA.png", true), ("LA.png", true), ("P.png", false),
            ("P-tRNS.png", true), ("RGB-tRNS.png", true), ("L-tRNS.png", true), ("RGBA-orientation-6.png", true)] {
            let image = processor.decode_png(&fs::read(root.join(file)).unwrap()).unwrap();
            assert_eq!(image.alpha.is_some(), alpha, "{file}");
            assert_eq!((image.width(), image.height()), (53, 47), "{file}");
        }
        let webp = processor.decode_webp(&fs::read(root.join("RGBA-lossless.webp")).unwrap()).unwrap();
        assert!(webp.alpha.is_some());
        let jpeg = processor.decode_jpeg(&fs::read(root.join("RGB-orientation-6.jpg")).unwrap()).unwrap();
        assert_eq!((jpeg.width(), jpeg.height()), (61, 29));
    }

    #[test]
    fn images_differing_only_in_alpha_have_distinct_identities() {
        let processor = processor();
        let mut image = RgbImage::new(9, 9, vec![80; 9 * 9 * 3]).unwrap();
        let opaque = processor.semantic_identity(&image);
        image.alpha = Some(vec![255; 81]);
        let alpha = processor.semantic_identity(&image);
        assert_ne!(opaque, alpha);
        image.alpha.as_mut().unwrap()[80] = 0;
        assert_ne!(alpha, processor.semantic_identity(&image));
        let transparent = processor.process(std::slice::from_ref(&image)).unwrap();
        image.alpha.as_mut().unwrap()[80] = 255;
        assert_ne!(transparent.pixel_values, processor.process(&[image]).unwrap().pixel_values);
    }

    #[test]
    fn size_is_bounded_by_decoded_pixels_and_prepared_tokens_without_resizing() {
        let processor = processor();
        let geometry = processor.geometry(4096, 4096).unwrap();
        assert_eq!((geometry.tokens, geometry.patches, geometry.grid_thw), (10609, 42436, [1, 206, 206]));
        let tall = processor.geometry(1, 41).unwrap();
        assert_eq!((tall.tokens, tall.width, tall.height), (2, 40, 80));
        assert_eq!(processor.geometry(8000, 1600).unwrap().tokens, 8000);
        let error = processor.geometry(4121, 4096).unwrap_err().to_string();
        assert!(error.contains("image needs 10712 image tokens") && error.contains("max_image_tokens_per_image is 10609"),
            "{error}");
        assert!(processor.geometry(0, 7).unwrap_err().to_string().contains("must be positive"));
        let small = ImageProcessor::load(ImageSettings::PaddedGrid {},
            ImageLimits { max_input_pixels: 1600, ..ImageLimits::default() }).unwrap();
        assert!(small.geometry(40, 40).is_ok());
        assert!(small.geometry(41, 40).unwrap_err().to_string().contains("max_image_pixels"));
        assert!(processor.resized_rgb(&RgbImage::new(1, 1, vec![0; 3]).unwrap()).is_err());
        let mut truncated = RgbImage::new(2, 1, vec![0; 6]).unwrap();
        truncated.alpha = Some(vec![0]);
        assert!(processor.process(&[truncated]).is_err());
    }
}
