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
    let placeholder = frontend.encode("<|media_pad|>", false)?;
    let [placeholder] = placeholder.as_slice() else {
        return Err("media placeholder must encode one token".into());
    };
    if rendered
        .token_ids
        .iter()
        .filter(|id| *id == placeholder)
        .count()
        != images.len()
    {
        return Err("MoonViT image placeholders do not match input images".into());
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
    let (mut image, mut offset) = (0, 0);
    let mut token_expansions = Vec::new();
    for (source_index, token) in rendered.token_ids.iter().enumerate() {
        let before = tokens.len();
        if token != placeholder {
            tokens.push(*token);
            data.vision_token_indices.push(-1);
            continue;
        }
        let source = &images[image];
        let count = processed.as_ref().unwrap().geometry[image].tokens;
        let prefix = frontend.encode(
            &format!(
                "<|media_begin|>image {}x{}<|media_content|>",
                source.width(),
                source.height()
            ),
            false,
        )?;
        tokens.extend(&prefix);
        data.vision_token_indices
            .extend(std::iter::repeat_n(-1, prefix.len()));
        conditioning.push(ConditioningEvent {
            position: tokens.len() as u64,
            identity: source.identity,
        });
        tokens.extend(std::iter::repeat_n(*placeholder, count));
        data.vision_token_indices
            .extend((offset..offset + count).map(|i| i as i32));
        let suffix = frontend.encode("<|media_end|>", false)?;
        data.vision_token_indices
            .extend(std::iter::repeat_n(-1, suffix.len()));
        tokens.extend(suffix);
        offset += count;
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
