/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Expand each rendered feature-token placeholder to one feature token per image token.

use crate::{
    Result,
    cache::ConditioningEvent,
    frontend::{
        RenderedPrompt,
        images::{ImageDescription, ImageProcessor},
    },
    model::{PreparedInput, PromptData},
};

/// `feature_token` is the prepared contract's image feature token; the renderer emits it once
/// per image, in input order, and text positions stay one-dimensional.
pub(crate) fn prepare_input(
    feature_token: u32,
    processor: Option<&ImageProcessor>,
    rendered: &RenderedPrompt,
    images: &[ImageDescription],
) -> Result<PreparedInput> {
    if rendered.token_ids.iter().filter(|id| **id == feature_token).count() != images.len() {
        return Err("image placeholders do not match input images".into());
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
    let mut tokens = Vec::with_capacity(rendered.token_ids.len());
    let mut data = PromptData::default();
    let mut conditioning = Vec::with_capacity(images.len());
    let mut token_expansions = Vec::with_capacity(images.len());
    let (mut image, mut feature) = (0, 0);
    for (source_index, token) in rendered.token_ids.iter().enumerate() {
        if *token != feature_token {
            tokens.push(*token);
            data.vision_token_indices.push(-1);
            continue;
        }
        let count = processed.as_ref().unwrap().geometry[image].tokens;
        conditioning.push(ConditioningEvent {
            position: tokens.len() as u64,
            identity: images[image].identity,
        });
        tokens.extend(std::iter::repeat_n(feature_token, count));
        data.vision_token_indices
            .extend((feature..feature + count).map(|index| index as i32));
        token_expansions.push((source_index, count - 1));
        feature += count;
        image += 1;
    }
    Ok(PreparedInput {
        tokens,
        token_expansions,
        data,
        conditioning,
        conditioned_inputs: vec![],
    })
}
