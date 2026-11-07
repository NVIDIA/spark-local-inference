/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Muse Glimmer: token-capped aspect-preserving Lanczos resize, `(temporal, channel, y, x)`
//! patches packed in 32×32-patch window order with position-table interpolation taps.
//!
//! Mirrors Transformers' `MuseGlimmerImageProcessor` (`smart_resize`, LANCZOS, `patchify`) and
//! `MuseGlimmerVisionModel` precompute (`get_vision_window_index`,
//! `get_vision_interpolation_indices_and_weights`), both run with `spatial_merge_size = 1`:
//! windows, positions and taps are per patch; the 2×2 merge happens after the encoder in
//! `pixel_shuffle`, which gathers raster-ordered rows block by block.

use super::*;

const TEMPORAL: usize = 2;
/// Patches per window side: `pos_emb_height × patch_size / patch_size`.
const WINDOW: usize = 32;
/// Side of the learned position-embedding table (`pos_emb_height == pos_emb_width`).
const TABLE_SIDE: usize = 32;

#[derive(Clone)]
pub(super) struct Processor {
    limits: ImageLimits,
    patch_size: usize,
    merge_size: usize,
    max_image_tokens: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    image_token: String,
    start_token: String,
    end_token: String,
}

impl Processor {
    pub(super) fn load(settings: MuseGlimmerImageSettings, limits: ImageLimits) -> Result<Self> {
        if settings.patch_size != 14
            || settings.merge_size != 2
            || settings.max_image_tokens == 0
            || settings.image_mean.iter().any(|x| !x.is_finite())
            || settings.image_std.iter().any(|x| !x.is_finite() || *x <= 0.)
            || [&settings.image_token, &settings.start_token, &settings.end_token]
                .iter()
                .any(|s| s.is_empty())
            || settings.image_token == settings.start_token
            || settings.image_token == settings.end_token
            || settings.start_token == settings.end_token
        {
            return Err("invalid Muse Glimmer image settings".into());
        }
        Ok(Self {
            limits,
            patch_size: settings.patch_size,
            merge_size: settings.merge_size,
            max_image_tokens: settings.max_image_tokens,
            image_mean: settings.image_mean,
            image_std: settings.image_std,
            image_token: settings.image_token,
            start_token: settings.start_token,
            end_token: settings.end_token,
        })
    }

    /// `smart_resize`: the patch grid (at `patch × merge` granularity) closest to the input
    /// aspect ratio under `max_image_tokens`.
    pub(super) fn geometry(&self, width: usize, height: usize) -> Result<ImageGeometry> {
        check_input_extent(self.limits, width, height)?;
        let factor = (self.patch_size * self.merge_size) as f64;
        let max_tokens = self.max_image_tokens as f64;
        let (h, w) = (height as f64, width as f64);
        let mut ideal_h = h / factor;
        let mut ideal_w = w / factor;
        let ratio = ideal_w / ideal_h;
        if ideal_h * ideal_w > max_tokens {
            ideal_h = (max_tokens / ratio).sqrt();
            ideal_w = ideal_h * ratio;
        }
        let product = [
            (ideal_h.floor(), ideal_w.floor()),
            (ideal_h.floor(), ideal_w.ceil()),
            (ideal_h.ceil(), ideal_w.floor()),
            (ideal_h.ceil(), ideal_w.ceil()),
        ]
        .map(|(gh, gw)| (gh as u64, gw as u64));
        let aspect = h / w;
        let distance = |(gh, gw): (u64, u64)| (gh as f64 / gw as f64 - aspect).abs();
        let mut best: Option<(u64, u64)> = None;
        for candidate in cpython_set_order(product)
            .into_iter()
            .filter(|(gh, gw)| *gh >= 1 && *gw >= 1 && (gh * gw) as f64 <= max_tokens)
        {
            // Python's `min` keeps the first of equal keys in set iteration order.
            if best.is_none_or(|b| distance(candidate) < distance(b)) {
                best = Some(candidate);
            }
        }
        let (grid_h, grid_w) = best.unwrap_or((
            (ideal_h.round_ties_even() as u64).max(1),
            (ideal_w.round_ties_even() as u64).max(1),
        ));
        let rows = grid_h as usize * self.merge_size;
        let cols = grid_w as usize * self.merge_size;
        let patches = rows * cols;
        check_image_patches(self.limits, patches)?;
        Ok(ImageGeometry {
            height: rows * self.patch_size,
            width: cols * self.patch_size,
            grid_thw: [1, rows as i64, cols as i64],
            patches,
            tokens: patches / (self.merge_size * self.merge_size),
        })
    }

    pub(super) fn resized_rgb(&self, image: &RgbImage) -> Result<(ImageGeometry, Vec<u8>)> {
        let geometry = self.geometry(image.width, image.height)?;
        Ok((geometry, resize::lanczos(image, geometry.width, geometry.height)))
    }

    pub(super) fn process(&self, images: &[RgbImage]) -> Result<ProcessedImages> {
        self.limits.check_batch(images.len(), 0)?;
        let geometry: Vec<_> = images
            .iter()
            .map(|image| self.geometry(image.width, image.height))
            .collect::<Result<_>>()?;
        let total = geometry
            .iter()
            .try_fold(0usize, |sum, image| sum.checked_add(image.patches))
            .ok_or_else(|| Error::frontend(FrontendError::Input(Failure::new("patch count overflow"))))?;
        self.limits.check_batch(images.len(), total)?;
        let windows: usize = geometry.iter().map(windows).sum();
        let mut result = ProcessedImages {
            geometry,
            cu_seqlens: vec![0],
            cu_window_seqlens: vec![0],
            ..Default::default()
        };
        let pixel_width = TEMPORAL * 3 * self.patch_size * self.patch_size;
        result.reserve_outputs(total, pixel_width, true, images.len(), images.len() + 1)?;
        result.cu_window_seqlens.reserve_exact(windows);
        result.reverse_index.reserve_exact(total);
        let normalized: [[f32; 256]; 3] = std::array::from_fn(|channel| {
            std::array::from_fn(|value| {
                (value as f32 - self.image_mean[channel] * 255.) / (self.image_std[channel] * 255.)
            })
        });
        for (image, geometry) in images.iter().zip(&result.geometry) {
            let pixels = resize::lanczos(image, geometry.width, geometry.height);
            let (grid_h, grid_w) = (geometry.height / self.patch_size, geometry.width / self.patch_size);
            let base = result.reverse_index.len();
            result.reverse_index.resize(base + geometry.patches, 0);
            for window_y in 0..grid_h.div_ceil(WINDOW) {
                for window_x in 0..grid_w.div_ceil(WINDOW) {
                    for row in window_y * WINDOW..grid_h.min((window_y + 1) * WINDOW) {
                        for col in window_x * WINDOW..grid_w.min((window_x + 1) * WINDOW) {
                            result.reverse_index[base + row * grid_w + col] =
                                result.position_ids.len() as i64;
                            result.position_ids.push([row as i64, col as i64]);
                            let (yi, yw) = taps(row, grid_h);
                            let (xi, xw) = taps(col, grid_w);
                            result.interpolation_indices.push([
                                yi[0] * TABLE_SIDE as i64 + xi[0],
                                yi[0] * TABLE_SIDE as i64 + xi[1],
                                yi[1] * TABLE_SIDE as i64 + xi[0],
                                yi[1] * TABLE_SIDE as i64 + xi[1],
                            ]);
                            result.interpolation_weights.push([
                                yw[0] * xw[0],
                                yw[0] * xw[1],
                                yw[1] * xw[0],
                                yw[1] * xw[1],
                            ]);
                            for _temporal in 0..TEMPORAL {
                                for channel in 0..3 {
                                    for y in 0..self.patch_size {
                                        for x in 0..self.patch_size {
                                            let byte = pixels[((row * self.patch_size + y) * geometry.width
                                                + col * self.patch_size
                                                + x)
                                                * 3
                                                + channel];
                                            result.pixel_values.push(normalized[channel][byte as usize]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    result.cu_window_seqlens.push(result.position_ids.len() as i32);
                }
            }
            result.cu_seqlens.push(result.position_ids.len() as i32);
        }
        Ok(result)
    }

    pub(super) fn expand_prompt(
        &self,
        frontend: &Frontend,
        text: &str,
        images: &ProcessedImages,
    ) -> Result<ImagePrompt> {
        patch_merge::bracketed_prompt(
            frontend,
            text,
            images,
            &self.image_token,
            &self.start_token,
            &self.end_token,
        )
    }
}

/// Window segments of one image: one per 32×32-patch cell, border cells ragged.
pub(super) fn windows(geometry: &ImageGeometry) -> usize {
    (geometry.grid_thw[1] as usize).div_ceil(WINDOW) * (geometry.grid_thw[2] as usize).div_ceil(WINDOW)
}

/// Bilinear taps of patch `index` on an axis of `size` patches into the 32-entry table:
/// half-pixel centres (`align_corners=False`), out-of-range taps clamped with zero weight
/// (`F.grid_sample(padding_mode="zeros")`), computed in f32 as the reference does.
fn taps(index: usize, size: usize) -> ([i64; 2], [f32; 2]) {
    let source = (index as f32 + 0.5) * TABLE_SIDE as f32 / size as f32 - 0.5;
    let floor = source.floor();
    let mut indices = [0i64; 2];
    let mut weights = [0f32; 2];
    for (offset, (index, weight)) in indices.iter_mut().zip(&mut weights).enumerate() {
        let raw = floor as i64 + offset as i64;
        *index = raw.clamp(0, TABLE_SIDE as i64 - 1);
        *weight = if (0..TABLE_SIDE as i64).contains(&raw) {
            (1.0 - (source - floor - offset as f32).abs()).max(0.0)
        } else {
            0.0
        };
    }
    (indices, weights)
}

/// Iteration order of CPython 3.12's `set` built from these four `(int, int)` tuples: the
/// reference's `min(list(set(itertools.product(...))))` resolves equal aspect distances by
/// it, so the same grid needs the same order. Tuple hashing is CPython's xxHash-derived
/// `tuplehash`; the small table has eight slots probed by `i * 5 + 1 + (perturb >>= 5)`.
fn cpython_set_order(items: [(u64, u64); 4]) -> Vec<(u64, u64)> {
    const PRIME_1: u64 = 11_400_714_785_074_694_791;
    const PRIME_2: u64 = 14_029_467_366_897_019_727;
    const PRIME_5: u64 = 2_870_177_450_012_600_261;
    const MASK: u64 = 7;
    fn hash(item: (u64, u64)) -> u64 {
        let mut acc = PRIME_5;
        for lane in [item.0, item.1] {
            acc = acc.wrapping_add(lane.wrapping_mul(PRIME_2)).rotate_left(31).wrapping_mul(PRIME_1);
        }
        acc = acc.wrapping_add(2 ^ (PRIME_5 ^ 3_527_539));
        if acc == u64::MAX { 1_546_275_796 } else { acc }
    }
    let mut table: [Option<(u64, u64)>; 8] = [None; 8];
    for item in items {
        let hash = hash(item);
        let mut perturb = hash;
        let mut slot = (hash & MASK) as usize;
        loop {
            match table[slot] {
                None => {
                    table[slot] = Some(item);
                    break;
                }
                Some(present) if present == item => break,
                Some(_) => {
                    perturb >>= 5;
                    slot = ((slot as u64 * 5 + 1 + perturb) & MASK) as usize;
                }
            }
        }
    }
    table.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> MuseGlimmerImageSettings {
        MuseGlimmerImageSettings {
            patch_size: 14,
            merge_size: 2,
            max_image_tokens: 4096,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            image_token: "<|patch|>".into(),
            start_token: "<|image_start|>".into(),
            end_token: "<|image_end|>".into(),
        }
    }

    fn processor() -> ImageProcessor {
        ImageProcessor::load(ImageSettings::MuseGlimmer(settings()), ImageLimits::default()).unwrap()
    }

    fn pattern(width: usize, height: usize) -> RgbImage {
        RgbImage::new(width, height, (0..width * height * 3).map(|i| ((i * 37 + 11) % 256) as u8).collect())
            .unwrap()
    }

    #[test]
    fn load_rejects_other_patch_or_merge_sizes_and_bad_tokens() {
        for alter in [
            |s: &mut MuseGlimmerImageSettings| s.patch_size = 16,
            |s: &mut MuseGlimmerImageSettings| s.merge_size = 1,
            |s: &mut MuseGlimmerImageSettings| s.max_image_tokens = 0,
            |s: &mut MuseGlimmerImageSettings| s.image_std[1] = 0.,
            |s: &mut MuseGlimmerImageSettings| s.start_token = "<|patch|>".into(),
        ] {
            let mut settings = settings();
            alter(&mut settings);
            assert!(ImageProcessor::load(ImageSettings::MuseGlimmer(settings), ImageLimits::default()).is_err());
        }
    }

    /// `(width, height, resized width, resized height)` from the Transformers `smart_resize`
    /// (patch 28 = 14 × 2, `max_tokens` 4096).
    const GEOMETRY: [(usize, usize, usize, usize); 40] = [
        (1, 1, 28, 28),
        (8000, 10, 8008, 28),
        (10, 8000, 28, 8008),
        (28, 28, 28, 28),
        (27, 29, 28, 28),
        (29, 27, 28, 28),
        (14, 14, 28, 28),
        (13, 15, 28, 28),
        (100, 100, 112, 112),
        (1792, 1792, 1792, 1792),
        (1793, 1792, 1792, 1792),
        (1792, 1793, 1792, 1792),
        (3000, 2000, 2184, 1456),
        (2000, 3000, 1456, 2184),
        (4096, 4096, 1792, 1792),
        (1, 4096, 28, 4116),
        (4096, 1, 4116, 28),
        (500, 1, 504, 28),
        (1, 500, 28, 504),
        (640, 480, 644, 476),
        (480, 640, 476, 644),
        (1920, 1080, 1932, 1092),
        (1080, 1920, 1092, 1932),
        (56, 28, 56, 28),
        (57, 29, 56, 28),
        (300, 2000, 308, 2016),
        (2000, 300, 2016, 308),
        (777, 333, 784, 336),
        (333, 777, 336, 784),
        (2048, 2048, 1792, 1792),
        (2047, 2049, 1792, 1792),
        (12345, 678, 7644, 420),
        (678, 12345, 420, 7644),
        (31, 31, 28, 28),
        (29, 41, 28, 28),
        (1000, 1000, 1008, 1008),
        (3, 7, 28, 28),
        (7, 3, 28, 28),
        (5000, 5000, 1792, 1792),
        (16384, 16384, 1792, 1792),
    ];

    /// Sizes where two candidate grids tie on aspect distance; the reference's choice follows
    /// CPython set order, `(width, height, grid width, grid height)`.
    const TIES: [(usize, usize, usize, usize); 20] = [
        (32, 24, 2, 1),
        (36, 27, 2, 1),
        (120, 27, 4, 1),
        (29, 29, 1, 1),
        (38, 29, 1, 1),
        (20, 30, 1, 1),
        (40, 30, 1, 1),
        (41, 31, 1, 1),
        (29, 32, 1, 1),
        (24, 32, 1, 1),
        (27, 36, 1, 1),
        (27, 120, 1, 4),
        (84, 63, 3, 2),
        (63, 84, 2, 3),
        (112, 84, 4, 3),
        (105, 84, 4, 3),
        (147, 105, 6, 4),
        (210, 126, 8, 5),
        (2520, 1890, 73, 55),
        (4200, 1680, 101, 40),
    ];

    #[test]
    fn geometry_matches_transformers_smart_resize() {
        let processor = ImageProcessor::load(
            ImageSettings::MuseGlimmer(settings()),
            ImageLimits { max_input_pixels: 1 << 30, ..ImageLimits::default() },
        )
        .unwrap();
        for (width, height, resized_width, resized_height) in GEOMETRY {
            let geometry = processor.geometry(width, height).unwrap();
            assert_eq!((geometry.width, geometry.height), (resized_width, resized_height), "{width}x{height}");
            assert_eq!(geometry.grid_thw, [1, (resized_height / 14) as i64, (resized_width / 14) as i64]);
            assert_eq!(geometry.patches, resized_height / 14 * (resized_width / 14));
            assert_eq!(geometry.tokens, geometry.patches / 4);
            assert!(geometry.tokens <= 4096, "{width}x{height}");
        }
        for (width, height, grid_width, grid_height) in TIES {
            let geometry = processor.geometry(width, height).unwrap();
            assert_eq!((geometry.width, geometry.height), (grid_width * 28, grid_height * 28), "{width}x{height}");
        }
        assert_eq!(processor.geometry(1792, 1792).unwrap().tokens, 4096);
        let limited = ImageProcessor::load(
            ImageSettings::MuseGlimmer(settings()),
            ImageLimits { max_patches_per_image: 1024, ..ImageLimits::default() },
        )
        .unwrap();
        assert!(limited.geometry(1792, 1792).is_err());
        assert!(processor.geometry(0, 10).is_err());
    }

    #[test]
    fn patches_are_temporal_channel_y_x_in_raster_order_within_a_window() {
        let processor = processor();
        let image = pattern(56, 28);
        let (geometry, pixels) = processor.resized_rgb(&image).unwrap();
        assert_eq!((geometry.width, geometry.height), (56, 28));
        assert_eq!(pixels, image.pixels().to_vec());
        let images = processor.process(&[image]).unwrap();
        assert_eq!(images.pixel_values.len(), 8 * 1176);
        assert_eq!(images.position_ids, [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2], [1, 3]]);
        assert_eq!(images.cu_seqlens, [0, 8]);
        assert_eq!(images.cu_window_seqlens, [0, 8]);
        assert_eq!(images.reverse_index, (0..8).collect::<Vec<i64>>());
        for (patch, [row, col]) in images.pixel_values.chunks_exact(1176).zip(&images.position_ids) {
            let (frame_a, frame_b) = patch.split_at(588);
            assert_eq!(frame_a, frame_b);
            for channel in 0..3 {
                for y in 0..14 {
                    for x in 0..14 {
                        let byte = pixels[(((*row as usize) * 14 + y) * 56 + (*col as usize) * 14 + x) * 3 + channel];
                        let expected = (byte as f32 - 127.5) / 127.5;
                        assert_eq!(patch[channel * 196 + y * 14 + x], expected);
                    }
                }
            }
        }
    }

    #[test]
    fn window_order_and_reverse_index_follow_transformers_on_a_ragged_grid() {
        let processor = processor();
        // 532×980 px → 38×70 patches: windows of 32×32, 32×6, 6×32 and 6×6 patches.
        let (small, large) = (pattern(56, 28), pattern(532, 980));
        let images = processor.process(&[large, small]).unwrap();
        assert_eq!(images.geometry[0].grid_thw, [1, 70, 38]);
        assert_eq!(images.cu_seqlens, [0, 2660, 2668]);
        assert_eq!(images.cu_window_seqlens, [0, 1024, 1216, 2240, 2432, 2624, 2660, 2668]);
        let raster = |k: usize| (images.position_ids[k][0] * 38 + images.position_ids[k][1]) as usize;
        assert_eq!((0..6).map(raster).collect::<Vec<_>>(), [0, 1, 2, 3, 4, 5]);
        assert_eq!((32..34).map(raster).collect::<Vec<_>>(), [38, 39]);
        assert_eq!((1024..1027).map(raster).collect::<Vec<_>>(), [32, 33, 34]);
        assert_eq!((1216..1218).map(raster).collect::<Vec<_>>(), [1216, 1217]);
        assert_eq!((2657..2660).map(raster).collect::<Vec<_>>(), [2657, 2658, 2659]);
        assert_eq!(&images.reverse_index[..3], &[0, 1, 2]);
        assert_eq!(&images.reverse_index[32..34], &[1024, 1025]);
        assert_eq!(&images.reverse_index[38..40], &[32, 33]);
        assert_eq!(&images.reverse_index[2657..2660], &[2657, 2658, 2659]);
        assert_eq!(&images.reverse_index[2660..], &(2660..2668).collect::<Vec<i64>>());
        for packed in 0..2660 {
            assert_eq!(images.reverse_index[raster(packed)], packed as i64);
        }
        let mut seen = vec![false; 2668];
        for index in &images.reverse_index {
            assert!(!std::mem::replace(&mut seen[*index as usize], true));
        }
        assert_eq!(images.interpolation_indices.len(), 2668);
        assert_eq!(images.interpolation_weights.len(), 2668);
    }

    #[test]
    fn interpolation_taps_match_transformers_on_interior_and_border_patches() {
        let processor = processor();
        // 700×560 px → 50×40 patches (table 32 downsampled); 112×84 px → 8×6 (upsampled).
        let images = processor.process(&[pattern(700, 560), pattern(112, 84)]).unwrap();
        let find = |image: usize, row: i64, col: i64| {
            let start = images.cu_seqlens[image] as usize;
            let end = images.cu_seqlens[image + 1] as usize;
            (start..end).find(|k| images.position_ids[*k] == [row, col]).unwrap()
        };
        let expected: [(usize, i64, i64, [i64; 4], [f32; 4]); 9] = [
            (0, 20, 25, [495, 496, 527, 528], [0.018000099807977676, 0.08200028538703918, 0.16200020909309387, 0.737999439239502]),
            (0, 0, 0, [0, 0, 0, 0], [0.0, 0.0, 0.0, 0.7379999756813049]),
            (0, 39, 49, [1023, 1023, 1023, 1023], [0.737999439239502, 0.0, 0.0, 0.0]),
            (0, 0, 49, [31, 31, 31, 31], [0.0, 0.0, 0.737999677658081, 0.0]),
            (1, 0, 0, [65, 66, 97, 98], [0.4166666269302368, 0.4166666269302368, 0.08333337306976318, 0.08333337306976318]),
            (1, 0, 1, [69, 70, 101, 102], [0.4166666269302368, 0.4166666269302368, 0.08333337306976318, 0.08333337306976318]),
            (1, 1, 0, [225, 226, 257, 258], [0.25, 0.25, 0.25, 0.25]),
            (1, 2, 4, [401, 402, 433, 434], [0.08333349227905273, 0.08333349227905273, 0.41666650772094727, 0.41666650772094727]),
            (1, 5, 7, [925, 926, 957, 958], [0.08333301544189453, 0.08333301544189453, 0.41666698455810547, 0.41666698455810547]),
        ];
        for (image, row, col, indices, weights) in expected {
            let k = find(image, row, col);
            assert_eq!(images.interpolation_indices[k], indices, "image {image} patch ({row}, {col})");
            for (actual, expected) in images.interpolation_weights[k].iter().zip(weights) {
                assert!((actual - expected).abs() <= 1e-6, "image {image} patch ({row}, {col}): {actual} != {expected}");
            }
        }
        for (indices, weights) in images.interpolation_indices.iter().zip(&images.interpolation_weights) {
            assert!(indices.iter().all(|i| (0..1024).contains(i)));
            assert!(weights.iter().all(|w| (0.0..=1.0).contains(w)));
        }
    }

    #[test]
    fn lanczos_matches_torchvision_antialiased_uint8_resize() {
        // torchvision 0.29 `InterpolationMode.LANCZOS`, `antialias=True`, CPU uint8 tensors, on the
        // `pattern` images; PIL 12.3 `Image.LANCZOS` agrees on the first two and differs by 1 on two
        // of the 324 samples of the third.
        let cases: [(usize, usize, usize, usize, &[u8]); 3] = [
            (13, 9, 7, 5, &[80,126,157,165,83,130,154,165,92,111,159,125,110,124,122,122,119,142,140,123,103,148,139,101,127,158,123,93,130,171,161,102,118,155,136,86,102,122,141,99,122,123,102,123,157,123,109,130,155,119,113,130,165,121,103,114,174,158,79,120,158,157,81,144,95,125,120,150,104,119,122,151,121,123,126,151,124,127,128,165,123,123,120,172,127,98,143,114,125,104,137,117,126,103,140,122,136,105,144,127,138,107,142,125,152]),
            (5, 4, 11, 8, &[0,23,62,16,65,100,55,130,160,139,151,178,239,78,102,255,10,47,186,19,95,92,101,157,101,193,118,164,251,37,206,255,0,4,33,68,29,75,115,76,139,187,142,159,212,193,85,125,190,20,57,137,40,101,93,116,165,131,171,127,198,190,47,240,198,5,22,57,89,58,97,139,117,158,218,153,177,246,119,101,155,71,43,80,61,82,119,103,149,182,177,139,147,237,90,69,255,60,24,44,79,129,89,127,152,157,202,188,170,228,192,73,138,133,0,69,106,28,118,171,126,181,235,186,127,188,199,28,99,202,0,43,63,95,174,107,149,145,173,233,102,190,255,62,100,167,66,33,88,125,78,132,221,156,199,255,145,153,215,89,62,120,53,9,59,90,139,214,132,158,157,196,187,68,219,189,11,141,137,62,84,114,151,132,169,202,189,224,202,137,186,166,45,110,129,0,63,104,113,187,235,155,156,182,220,109,94,241,70,49,161,80,112,102,137,174,150,214,143,211,248,82,165,202,88,78,128,127,25,80,152,123,212,241,165,152,196,230,60,119,251,0,85,170,44,147,110,147,184,156,236,104,219,255,9,185,209,40,108,136,123,62,88,177]),
            (40, 30, 9, 12, &[121,133,128,140,117,131,125,127,120,131,126,131,131,140,118,117,122,133,124,124,123,130,136,135,143,112,132,123,142,116,125,122,131,135,135,118,116,129,125,128,122,128,122,136,136,134,118,129,126,125,121,132,124,136,136,136,120,121,133,140,125,119,126,128,134,132,120,120,134,133,121,118,127,125,135,137,133,120,119,131,124,122,119,126,127,132,127,134,125,137,136,123,118,124,132,129,138,120,126,123,136,117,125,123,126,131,134,133,126,131,138,138,116,123,121,128,125,134,131,129,128,139,122,119,123,135,131,134,121,117,129,138,123,117,122,126,122,124,129,126,133,134,139,116,117,124,130,130,128,128,122,134,139,138,117,126,124,129,127,132,126,133,139,130,121,118,131,121,128,122,128,123,137,136,126,118,129,131,126,121,131,125,137,136,137,118,116,133,126,125,119,127,129,136,129,116,122,136,130,122,119,128,126,135,138,126,123,119,134,120,130,124,128,128,140,134,131,122,141,137,120,120,123,132,130,138,116,127,124,133,117,125,125,127,134,137,127,116,124,138,128,125,118,124,128,128,134,128,132,129,140,117,120,124,135,132,134,119,117,130,135,123,119,124,126,134,132,135,118,130,134,141,118,119,126,125,132,128,129,121,134,138,132,117,128,126,130,127,129,121,135,133,120,118,120,129,127,128,122,124,125,136,135,118,117,128,129,131,125,136,127,137,137,132,120,120,132,125,133,126,126,127,135,114]),
        ];
        for (width, height, out_width, out_height, expected) in cases {
            let actual = resize::lanczos(&pattern(width, height), out_width, out_height);
            assert_eq!(actual.as_slice(), expected, "{width}x{height}->{out_width}x{out_height}");
        }
    }

    #[test]
    fn native_pixels_match_canonical_transformers_processor() {
        let python = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.venv/bin/python");
        if !python.is_file() {
            return;
        }
        let script = r#"
import sys, struct, torch
from transformers.models.muse_glimmer.image_processing_muse_glimmer import MuseGlimmerImageProcessor
image=((torch.arange(37*91*3,dtype=torch.int64)*37+11)%256).to(torch.uint8).view(37,91,3).permute(2,0,1)
result=MuseGlimmerImageProcessor(image_mean=[0.5,0.5,0.5],image_std=[0.5,0.5,0.5])(image,return_tensors='pt')
sys.stdout.buffer.write(struct.pack('<qqq',*result['image_grid_thw'][0].tolist()))
sys.stdout.buffer.write(result['pixel_values'].contiguous().numpy().tobytes())
"#;
        let reference = std::process::Command::new(python).args(["-I", "-c", script]).output().unwrap();
        assert!(reference.status.success(), "{}", String::from_utf8_lossy(&reference.stderr));
        let actual = processor().process(&[pattern(91, 37)]).unwrap();
        let grid: Vec<i64> = reference.stdout[..24]
            .chunks_exact(8)
            .map(|x| i64::from_le_bytes(x.try_into().unwrap()))
            .collect();
        assert_eq!(actual.geometry[0].grid_thw.as_slice(), grid.as_slice());
        let pixels: Vec<f32> = reference.stdout[24..]
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect();
        assert_eq!(actual.pixel_values.len(), pixels.len());
        // The reference packs raster order; `reverse_index[r]` is raster patch `r`'s packed row.
        for (raster, packed) in actual.reverse_index.iter().enumerate() {
            let packed = *packed as usize;
            let actual = &actual.pixel_values[packed * 1176..(packed + 1) * 1176];
            let expected = &pixels[raster * 1176..(raster + 1) * 1176];
            for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
                assert_eq!(*a, *b, "raster patch {raster} value {i}");
            }
        }
    }
}
