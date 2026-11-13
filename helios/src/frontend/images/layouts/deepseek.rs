/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::PromptData;
use crate::{
    Result,
    cache::ConditioningEvent,
    frontend::{
        Frontend, RenderedPrompt,
        images::{ImageProcessor, ImageDescription},
    },
    model::PreparedInput,
};

fn image_block(
    vocabulary: usize,
    height: usize,
    width: usize,
    position: usize,
    feature_offset: usize,
) -> (Vec<u32>, Vec<i32>, usize) {
    let padding = 3 - position % 4;
    let mut types = vec![1u32; padding];
    let mut indices = vec![-1; padding];
    types.push(0);
    indices.push(-1);
    let rows = height.next_multiple_of(2);
    for pair in 0..rows / 2 {
        for column in 0..=width {
            for row in pair * 2..pair * 2 + 2 {
                let feature = row < height && column < width;
                types.push(if row >= height {
                    1
                } else if feature {
                    2
                } else {
                    3
                });
                indices.push(if feature {
                    (feature_offset + row * width + column) as i32
                } else {
                    -1
                });
            }
        }
    }
    let last = rows / 2 * (width + 1) % 2 * 2;
    types.extend(std::iter::repeat_n(1, last));
    indices.extend(std::iter::repeat_n(-1, last));
    types.push(4);
    indices.push(-1);
    (
        types.into_iter().map(|t| vocabulary as u32 + t).collect(),
        indices,
        padding,
    )
}

pub(crate) fn prepare_input(
    vocabulary: usize, frontend: &Frontend, processor: Option<&ImageProcessor>,
    rendered: &RenderedPrompt, images: &[ImageDescription],
) -> Result<PreparedInput> {
    prepare(vocabulary, frontend, processor, rendered, images, image_block)
}

pub(crate) fn prepare_row_major_input(
    vocabulary: usize, frontend: &Frontend, processor: Option<&ImageProcessor>,
    rendered: &RenderedPrompt, images: &[ImageDescription],
) -> Result<PreparedInput> {
    prepare(vocabulary, frontend, processor, rendered, images, row_major_block)
}

fn row_major_block(
    vocabulary: usize, height: usize, width: usize, _position: usize, feature_offset: usize,
) -> (Vec<u32>, Vec<i32>, usize) {
    // Input-only IDs carry the type; the graph replaces every image embedding.
    let base = vocabulary as u32;
    let mut tokens = vec![base];
    let mut indices = vec![-1];
    for row in 0..height {
        for column in 0..width {
            tokens.push(base + 1);
            indices.push((feature_offset + row * width + column) as i32);
        }
        tokens.push(base + 2);
        indices.push(-1);
    }
    tokens.push(base + 3);
    indices.push(-1);
    (tokens, indices, 0)
}

fn prepare(
    vocabulary: usize,
    frontend: &Frontend,
    processor: Option<&ImageProcessor>,
    rendered: &RenderedPrompt,
    images: &[ImageDescription],
    block: fn(usize, usize, usize, usize, usize) -> (Vec<u32>, Vec<i32>, usize),
) -> Result<PreparedInput> {
    let placeholder = frontend.image_placeholder_token_id()?;
    if rendered.token_ids.iter().any(|t| *t as usize >= vocabulary) {
        return Err("invalid DeepSeek image placeholder or prompt vocabulary".into());
    }
    if rendered
        .token_ids
        .iter()
        .filter(|t| **t == placeholder)
        .count()
        != images.len()
    {
        return Err("image placeholder count does not match input images".into());
    }
    let processed = if images.is_empty() {
        None
    } else {
        Some(
            processor
                .ok_or("image processing is unavailable")?
                .describe_inputs(images)?,
        )
    };
    let mut tokens = Vec::new();
    let mut data = PromptData::default();
    let mut conditioning = Vec::new();
    let mut image_index = 0;
    let mut feature_offset = 0;
    let mut token_expansions = Vec::new();
    for (source_index, token) in rendered.token_ids.iter().enumerate() {
        let before = tokens.len();
        if *token != placeholder {
            tokens.push(*token);
            data.vision_token_indices.push(-1);
            continue;
        }
        let source = &images[image_index];
        let geometry = &processed.as_ref().unwrap().geometry[image_index];
        let start = tokens.len();
        let (block, indices, padding) = block(
            vocabulary,
            (geometry.grid_thw[1] as usize).div_ceil(3),
            (geometry.grid_thw[2] as usize).div_ceil(3),
            start,
            feature_offset,
        );
        data.spans.push(start + padding..start + block.len());
        tokens.extend(block);
        data.vision_token_indices.extend(indices);
        conditioning.push(ConditioningEvent {
            position: start as u64,
            identity: source.identity,
        });
        feature_offset += geometry.tokens;
        image_index += 1;
        token_expansions.push((source_index, tokens.len() - before - 1));
    }
    Ok(PreparedInput {
        tokens,
        token_expansions,
        data,
        conditioning,
        conditioned_inputs: vec![],
    })
}
