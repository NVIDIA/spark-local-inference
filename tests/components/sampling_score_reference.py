# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Unconditional vocabulary scoring oracle for request-parameter branch checks."""

import triton
import triton.language as tl


@triton.jit
def scores_reference(Logits, Scores, Slots, Epochs, Live, Generation, Status, Counts, Params, Grammar, GrammarEnabled,
                     V: tl.constexpr, WORDS: tl.constexpr, BLOCK: tl.constexpr, SLOTS: tl.constexpr = 4):
    row, block = tl.program_id(0), tl.program_id(1)
    token = block * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(Slots + row)
    valid = (row < tl.load(Live)) & (slot >= 0) & (slot < SLOTS)
    valid &= tl.load(Generation + slot, valid, other=-1) == tl.load(Epochs + row)
    valid &= tl.load(Status + slot, valid, other=0) == 1
    counts = tl.load(Counts + slot * V + token, valid & (token < V), other=0)
    x = tl.load(Logits + row * V + token, token < V, other=-float("inf")).to(tl.float32)
    temperature = tl.load(Params + slot * 6, valid, other=1.)
    repetition = tl.load(Params + slot * 6 + 3, valid, other=1.)
    presence = tl.load(Params + slot * 6 + 4, valid, other=0.)
    frequency = tl.load(Params + slot * 6 + 5, valid, other=0.)
    x = tl.where(counts > 0, tl.where(x > 0., x / repetition, x * repetition) - presence, x)
    x = x - frequency * counts
    x = x / tl.where(valid, temperature, 1.)
    enabled = tl.load(GrammarEnabled + slot, valid, other=0)
    bits = tl.load(Grammar + slot * WORDS + token // 32, valid & (token < V), other=0).to(tl.uint32)
    allowed = (enabled == 0) | ((bits >> (token % 32)) & 1 != 0)
    tl.store(Scores + row * V + token, tl.where(valid & allowed, x, -float("inf")), token < V)
