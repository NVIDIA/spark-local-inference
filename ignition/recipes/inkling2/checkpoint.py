# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The Inkling 2 Small checkpoint revision the recipe prepares, and its weight bytes.

The byte counts come from the tensor placements of `weights.checkpoint_layout` at
four ranks; this module imports no model code.
"""

from ignition.recipes.declaration import Checkpoint

# Stored payload of the 1770 indexed tensors (390,056,448,320 bytes, the index's
# total_size) plus the 84 drafter tensors of mtp.safetensors (3,423,420,768 bytes).
# Device bytes total the four ranks' weights: experts, attention and the drafter split
# across ranks, padded vocabulary-parallel embedding and unembedding (4 x 50,304 rows),
# and replicated router, normalization, convolution, residual-query, vision and Markov
# tensors. Each rank holds 99,086,416,544 bytes, of which the drafter is 946,159,968.
CHECKPOINT = Checkpoint(
    alias="inkling-2-small",
    repository="thinkingmachines/Inkling-2-Small",
    revision="b0e7afb8892a07897c78ce1c5b801a0d9cbd5e7b",
    architecture="Inkling2ForConditionalGeneration",
    stored_bytes=393_479_869_088,
    device_bytes=396_345_666_176,
    draft_bytes=3_784_639_872,
)
