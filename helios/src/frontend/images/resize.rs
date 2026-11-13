/*
 * From PyTorch:
 *
 * Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
 * Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
 * Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
 * Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
 * Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
 * Copyright (c) 2011-2013 NYU                      (Clement Farabet)
 * Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
 * Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
 * Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
 *
 * From Caffe2:
 *
 * Copyright (c) 2016-present, Facebook Inc. All rights reserved.
 *
 * All contributions by Facebook:
 * Copyright (c) 2016 Facebook Inc.
 *
 * All contributions by Google:
 * Copyright (c) 2015 Google Inc.
 * All rights reserved.
 *
 * All contributions by Yangqing Jia:
 * Copyright (c) 2015 Yangqing Jia
 * All rights reserved.
 *
 * All contributions by Kakao Brain:
 * Copyright 2019-2020 Kakao Brain
 *
 * All contributions by Cruise LLC:
 * Copyright (c) 2022 Cruise LLC.
 * All rights reserved.
 *
 * All contributions by Tri Dao:
 * Copyright (c) 2024 Tri Dao.
 * All rights reserved.
 *
 * All contributions by Arm:
 * Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates
 *
 * All contributions from Caffe:
 * Copyright(c) 2013, 2014, 2015, the respective contributors
 * All rights reserved.
 *
 * All other contributions:
 * Copyright(c) 2015, 2016 the respective contributors
 * All rights reserved.
 *
 * Caffe2 uses a copyright model similar to Caffe: each contributor holds
 * copyright over their contributions to Caffe2. The project versioning records
 * all such contribution and copyright details. If a contributor wants to further
 * mark their specific copyright on a particular contribution, they should
 * indicate their copyright solely in the commit message of the change when it is
 * committed.
 *
 * All rights reserved.
 *
 * PyTorch license terms: helios/licenses/PyTorch.txt.
 */

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

// Coefficient generation follows PyTorch's uint8 antialiased bicubic kernel.
// See helios/licenses/PyTorch.txt and helios/README.md for source attribution.

use super::RgbImage;

struct Axis {
    taps: Vec<(usize, Vec<i32>)>,
    precision: u32,
}

fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        ((1.5 * x - 2.5) * x) * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}

fn axis(input: usize, output: usize, pillow: bool, linear: bool) -> Axis {
    let scale = input as f64 / output as f64;
    let support = (if linear { 1.0 } else { 2.0 }) * scale.max(1.0);
    let inverse_scale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let max_taps = support.ceil() as usize * 2 + 1;
    let mut maximum = 0.0f64;
    let floating: Vec<_> = (0..output)
        .map(|i| {
            let center = scale * (i as f64 + 0.5);
            let start = ((center - support + 0.5) as i64).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(input);
            let count = (end - start).min(max_taps);
            let mut weights: Vec<_> = (0..count)
                .map(|j| {
                    let x = (j as f64 + start as f64 - center + 0.5) * inverse_scale;
                    if linear {
                        (1.0 - x.abs()).max(0.0)
                    } else {
                        cubic(x)
                    }
                })
                .collect();
            let sum: f64 = weights.iter().sum();
            if sum != 0.0 {
                for weight in &mut weights {
                    *weight /= sum;
                    maximum = maximum.max(*weight);
                }
            }
            (start, weights)
        })
        .collect();
    let mut precision = 0;
    while precision < 22 {
        let next = (0.5 + maximum * f64::from(1 << (precision + 1))) as i32;
        if next >= 1 << 15 {
            break;
        }
        precision += 1;
    }
    if pillow {
        precision = 22;
    }
    let multiplier = f64::from(1 << precision);
    let taps = floating
        .into_iter()
        .map(|(start, weights)| {
            (
                start,
                weights
                    .into_iter()
                    .map(|weight| {
                        let scaled = weight * multiplier;
                        if scaled < 0.0 {
                            (scaled - 0.5) as i32
                        } else {
                            (scaled + 0.5) as i32
                        }
                    })
                    .collect(),
            )
        })
        .collect();
    Axis { taps, precision }
}

pub(super) fn resize(input: &RgbImage, width: usize, height: usize) -> Vec<u8> {
    resample(input, width, height, false, false)
}

pub(super) fn pillow(input: &RgbImage, width: usize, height: usize) -> Vec<u8> {
    // Pillow resizes vertically first only when shrinking an image taller than
    // 100 times its width. Each pass rounds to uint8, so the order affects RGB.
    if input.height > height && input.height > input.width * 100 {
        let pixels = resample(input, input.width, height, true, false);
        let intermediate = RgbImage::new(input.width, height, pixels).unwrap();
        return resample(&intermediate, width, height, true, false);
    }
    resample(input, width, height, true, false)
}

pub(super) fn bilinear(input: &RgbImage, width: usize, height: usize) -> Vec<u8> {
    resample(input, width, height, false, true)
}

fn resample(input: &RgbImage, width: usize, height: usize, pillow: bool, linear: bool) -> Vec<u8> {
    let source = &input.pixels;
    let horizontal = if width == input.width {
        None
    } else {
        let axis = axis(input.width, width, pillow, linear);
        let mut pixels = vec![0; input.height * width * 3];
        for y in 0..input.height {
            for (x, (start, weights)) in axis.taps.iter().enumerate() {
                for channel in 0..3 {
                    let mut value = 1i64 << (axis.precision - 1);
                    for (tap, weight) in weights.iter().enumerate() {
                        value += i64::from(source[(y * input.width + start + tap) * 3 + channel])
                            * i64::from(*weight);
                    }
                    pixels[(y * width + x) * 3 + channel] =
                        (value >> axis.precision).clamp(0, 255) as u8;
                }
            }
        }
        Some(pixels)
    };
    let source = horizontal.as_ref().unwrap_or(source);
    if height == input.height {
        return source.clone();
    }
    let axis = axis(input.height, height, pillow, linear);
    let mut pixels = vec![0; height * width * 3];
    for (y, (start, weights)) in axis.taps.iter().enumerate() {
        for x in 0..width * 3 {
            let mut value = 1i64 << (axis.precision - 1);
            for (tap, weight) in weights.iter().enumerate() {
                value += i64::from(source[(start + tap) * width * 3 + x]) * i64::from(*weight);
            }
            pixels[y * width * 3 + x] = (value >> axis.precision).clamp(0, 255) as u8;
        }
    }
    pixels
}

/// Float32 antialiased bicubic interpolation without integer rounding or clipping.
pub(super) fn float_bicubic(input: &RgbImage, width: usize, height: usize) -> Vec<f32> {
    fn weights(input: usize, output: usize) -> Vec<(usize, Vec<f32>)> {
        let scale = input as f32 / output as f32;
        let support = 2.0 * scale.max(1.0);
        let inverse = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
        (0..output)
            .map(|i| {
                let center = scale * (i as f32 + 0.5);
                let start = ((center - support + 0.5) as isize).max(0) as usize;
                let end = ((center + support + 0.5) as usize).min(input);
                let mut values: Vec<_> = (start..end)
                    .map(|j| {
                        let x = ((j as f32 - center + 0.5) * inverse).abs();
                        if x < 1.0 {
                            ((1.5 * x - 2.5) * x) * x + 1.0
                        } else if x < 2.0 {
                            ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let sum: f32 = values.iter().sum();
                for value in &mut values {
                    *value /= sum;
                }
                (start, values)
            })
            .collect()
    }
    let mut horizontal = vec![0.0; input.height * width * 3];
    if width == input.width {
        for (out, value) in horizontal.iter_mut().zip(&input.pixels) {
            *out = f32::from(*value);
        }
    } else {
        let axis = weights(input.width, width);
        for y in 0..input.height {
            for (x, (start, taps)) in axis.iter().enumerate() {
                for c in 0..3 {
                    let mut value = 0.0;
                    for (i, weight) in taps.iter().enumerate() {
                        value +=
                            f32::from(input.pixels[(y * input.width + start + i) * 3 + c]) * weight;
                    }
                    horizontal[(y * width + x) * 3 + c] = value;
                }
            }
        }
    }
    if height == input.height {
        return horizontal;
    }
    let mut output = vec![0.0; height * width * 3];
    let axis = weights(input.height, height);
    for (y, (start, taps)) in axis.iter().enumerate() {
        for x in 0..width * 3 {
            let mut value = 0.0;
            for (i, weight) in taps.iter().enumerate() {
                value += horizontal[(start + i) * width * 3 + x] * weight;
            }
            output[y * width * 3 + x] = value;
        }
    }
    output
}
