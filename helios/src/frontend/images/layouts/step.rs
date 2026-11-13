/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::{
    Result,
    cache::ConditioningEvent,
    frontend::{
        Frontend, RenderedPrompt,
        images::{ImageProcessor, ImageDescription},
    },
    model::{PreparedInput, PromptData},
};

pub(crate) fn prepare_input(
    frontend: &Frontend,
    processor: Option<&ImageProcessor>,
    rendered: &RenderedPrompt,
    images: &[ImageDescription],
) -> Result<PreparedInput> {
    let ids = frontend.encode("<im_patch>", false)?;
    let [image_token] = ids.as_slice() else {
        return Err("image placeholder must encode one token".into());
    };
    if rendered
        .token_ids
        .iter()
        .filter(|id| *id == image_token)
        .count()
        != images.len()
    {
        return Err("Step image placeholders do not match input images".into());
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
    let (mut tokens, mut data, mut conditioning) = (vec![], PromptData::default(), vec![]);
    let (mut image, mut offset) = (0, 0);
    let markers = [
        "<patch_start>",
        "<patch_end>",
        "<patch_newline>",
        "<im_start>",
        "<im_end>",
    ]
    .map(|text| frontend.encode(text, false))
    .into_iter()
    .collect::<Result<Vec<_>>>()?;
    let mut token_expansions = Vec::new();
    for (source_index, token) in rendered.token_ids.iter().enumerate() {
        let before = tokens.len();
        if token != image_token {
            tokens.push(*token);
            data.vision_token_indices.push(-1);
            continue;
        }
        let source = &images[image];
        let g = &processed.as_ref().unwrap().geometry[image];
        conditioning.push(ConditioningEvent {
            position: tokens.len() as u64,
            identity: source.identity,
        });
        // Every crop has 81 features; the global view follows all crop rows.
        for row in 0..g.grid_thw[1] {
            for _ in 0..g.grid_thw[2] {
                tokens.extend(&markers[0]);
                data.vision_token_indices
                    .extend(std::iter::repeat_n(-1, markers[0].len()));
                tokens.extend(std::iter::repeat_n(*image_token, 81));
                data.vision_token_indices
                    .extend((offset..offset + 81).map(|i| i as i32));
                offset += 81;
                data.vision_token_indices
                    .extend(std::iter::repeat_n(-1, markers[1].len()));
                tokens.extend(&markers[1]);
            }
            if row + 1 < g.grid_thw[1] {
                tokens.extend(&markers[2]);
                data.vision_token_indices
                    .extend(std::iter::repeat_n(-1, markers[2].len()));
            }
        }
        tokens.extend(&markers[3]);
        data.vision_token_indices
            .extend(std::iter::repeat_n(-1, markers[3].len()));
        tokens.extend(std::iter::repeat_n(*image_token, 169));
        data.vision_token_indices
            .extend((offset..offset + 169).map(|i| i as i32));
        offset += 169;
        tokens.extend(&markers[4]);
        data.vision_token_indices
            .extend(std::iter::repeat_n(-1, markers[4].len()));
        image += 1;
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
