# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP32 distributions and speculative block correction on vocabulary shards.

Each rank retains its probabilities and exchanges compact reduction records.
Proposal scores occupy their probability destination until normalization; only
reduction records need separate scratch. Correction tiles evaluate all prepared
positions together, sharing the selected-token prefix calculation.
"""

import torch
import triton
import triton.language as tl

from ignition.preparation.scratch import bind_scratch
from ignition.preparation.triton import launch_triton, prepare_triton
from ignition.preparation.persistent import persistent_copy
from ignition.preparation.tensor_metadata import record_numeric_tensor
from .collectives import PreparedAllGather
from .markov_sampling import _row_live, _token_noise, _transition_bias, RowScaledFP8MarkovWeight
from .speculation import SpeculativeAcceptance, TARGET_RNG_DOMAIN, _geometry, _metadata, _tensor

SUMMARY_WORDS = 32
_WORDS = tl.constexpr(SUMMARY_WORDS)
_TARGET_DOMAIN = tl.constexpr(TARGET_RNG_DOMAIN)


@triton.jit
def _score_tiles(
    VocabularyOffset,
    Logits,
    Probabilities,
    Partials,
    Embedding,
    Head,
    HeadScale,
    Pending,
    Drafts,
    Histogram,
    Slots,
    Generations,
    LiveRows,
    Generation,
    Status,
    Params,
    TopK,
    Grammar,
    GrammarEnabled,
    Seeds,
    Offsets,
    logit_row_stride,
    logit_step_stride,
    hist_stride,
    mask_stride,
    LOCAL: tl.constexpr,
    VOCAB: tl.constexpr,
    WIDTH: tl.constexpr,
    DRAFTS: tl.constexpr,
    PREFIX: tl.constexpr,
    BLOCKS: tl.constexpr,
    CAPACITY: tl.constexpr,
    BATCH: tl.constexpr,
    RUNNING: tl.constexpr,
    MARKOV: tl.constexpr,
    RANK: tl.constexpr,
    FP8: tl.constexpr,
    SAMPLE: tl.constexpr,
    STEP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile, lane_row = tl.program_id(0), tl.program_id(1)
    position = STEP if SAMPLE else lane_row % WIDTH
    request_row = lane_row if SAMPLE else lane_row // WIDTH
    local_id = tile * BLOCK + tl.arange(0, BLOCK)
    global_id = tl.load(VocabularyOffset) + local_id
    present = (local_id < LOCAL) & (global_id < VOCAB)
    value = tl.load(
        Logits + request_row * logit_row_stride + position * logit_step_stride + local_id, present, other=0
    ).to(tl.float32)
    if MARKOV:
        predecessor = tl.load(Pending + request_row) if STEP == 0 else tl.load(Drafts + request_row * DRAFTS + STEP - 1)
        predecessor = tl.minimum(tl.maximum(predecessor, 0), VOCAB - 1)
        value += _transition_bias(Embedding, Head, HeadScale, predecessor, local_id, present, RANK, 64, BLOCK, FP8)
    slot = tl.minimum(tl.maximum(tl.load(Slots + request_row).to(tl.int64), 0), CAPACITY - 1)
    temperature = tl.load(Params + slot * 6)
    repeat = tl.load(Params + slot * 6 + 3)
    presence = tl.load(Params + slot * 6 + 4)
    frequency = tl.load(Params + slot * 6 + 5)
    if (repeat != 1) | (presence.to(tl.int32, bitcast=True) != 0) | (frequency.to(tl.int32, bitcast=True) != 0):
        count = tl.load(Histogram + slot * hist_stride + global_id, present, other=0)
        for preceding in tl.static_range(PREFIX):
            if preceding < position:
                count += (global_id == tl.load(Drafts + request_row * DRAFTS + preceding)).to(tl.int32)
        repeated = tl.where(value > 0, tl.div_rn(value, repeat), value * repeat)
        value = tl.where(count > 0, repeated, value)
        value -= (count > 0).to(tl.float32) * presence
        value -= count.to(tl.float32) * frequency
    value = tl.div_rn(value, tl.where(temperature > 0, temperature, 1.0))
    if tl.load(GrammarEnabled + slot) != 0:
        allowed = tl.load(Grammar + slot * mask_stride + global_id // 32, present, other=0)
        present &= ((allowed >> (global_id % 32)) & 1) != 0
    value = tl.where(present, value, -float("inf"))
    invalid = tl.sum(((value != value) | (value == float("inf"))).to(tl.int32), 0) != 0
    if tile == 0:
        invalid |= ~_row_live(Slots, Generations, LiveRows, Generation, Status, request_row, CAPACITY, BATCH, RUNNING)
    if not SAMPLE:
        top_k = tl.load(TopK + slot)
        invalid |= (top_k > 0) & (top_k < VOCAB)
        invalid |= (tl.load(Params + slot * 6 + 1) != 1) | (tl.load(Params + slot * 6 + 2) != 0)
    maximum = tl.max(value, 0)
    stable_max = tl.where(tl.abs(maximum) < float("inf"), maximum, 0.0)
    mass = tl.sum(tl.exp(value - stable_max), 0)
    choice = value
    if SAMPLE:
        choice += _token_noise(tl.load(Seeds + request_row), tl.load(Offsets + request_row) + 4 * STEP, global_id)
    winner = tl.argmax(choice, 0, tie_break_left=True)
    # Scores are consumed before this same region becomes retained probabilities.
    tl.store(Probabilities + (request_row * WIDTH + position) * LOCAL + local_id, value, local_id < LOCAL)
    partial = Partials + lane_row * 5 * BLOCKS + tile
    tl.store(partial, maximum)
    tl.store(partial + BLOCKS, mass)
    tl.store(partial + 2 * BLOCKS, tl.max(choice, 0))
    tl.store(partial + 3 * BLOCKS, (tl.load(VocabularyOffset) + tile * BLOCK + winner).to(tl.float32))
    tl.store(partial + 4 * BLOCKS, invalid.to(tl.float32))


@triton.jit
def _softmax_records(Partials, Summary, BLOCKS: tl.constexpr, PAD: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.arange(0, PAD)
    base = Partials + row * 5 * BLOCKS + tile
    maxima = tl.load(base, tile < BLOCKS, other=-float("inf"))
    weights = tl.load(base + BLOCKS, tile < BLOCKS, other=0.0)
    maximum = tl.max(maxima, 0)
    origin = tl.where(tl.abs(maximum) < float("inf"), maximum, 0.0)
    mass = tl.sum(weights * tl.exp(maxima - origin), 0)
    candidates = tl.load(base + 2 * BLOCKS, tile < BLOCKS, other=-float("inf"))
    selected = tl.argmax(candidates, 0, tie_break_left=True)
    ids = tl.load(base + 3 * BLOCKS, tile < BLOCKS, other=0.0)
    identifier = tl.sum(tl.where(tile == selected, ids, 0.0), 0)
    bad = tl.sum(tl.load(base + 4 * BLOCKS, tile < BLOCKS, other=0.0), 0)
    word = tl.arange(0, _WORDS)
    record = tl.where(
        word == 0,
        maximum,
        tl.where(
            word == 1,
            mass,
            tl.where(word == 2, tl.max(candidates, 0), tl.where(word == 3, identifier, tl.where(word == 4, bad, 0.0))),
        ),
    )
    tl.store(Summary + row * _WORDS + word, record)


@triton.jit
def _finish_distribution(
    VocabularyOffset,
    Probabilities,
    Gathered,
    Valid,
    Tokens,
    Slots,
    Generations,
    LiveRows,
    Generation,
    Status,
    Params,
    LOCAL: tl.constexpr,
    VOCAB: tl.constexpr,
    WIDTH: tl.constexpr,
    STEP: tl.constexpr,
    WORLD: tl.constexpr,
    CAPACITY: tl.constexpr,
    BATCH: tl.constexpr,
    RUNNING: tl.constexpr,
    SAMPLE: tl.constexpr,
    BLOCK: tl.constexpr,
    RANK_PAD: tl.constexpr,
):
    tile, distribution = tl.program_id(0), tl.program_id(1)
    row = distribution if SAMPLE else distribution // WIDTH
    position = STEP if SAMPLE else distribution % WIDTH
    rank = tl.arange(0, RANK_PAD)
    records = Gathered + (distribution * WORLD + rank) * _WORDS
    maxima = tl.load(records, rank < WORLD, other=-float("inf"))
    masses = tl.load(records + 1, rank < WORLD, other=0.0)
    maximum = tl.max(maxima, 0)
    origin = tl.where(tl.abs(maximum) < float("inf"), maximum, 0.0)
    total = tl.sum(masses * tl.exp(maxima - origin), 0)
    invalid = tl.sum(tl.load(records + 4, rank < WORLD, other=0.0), 0)
    valid = (
        (tl.abs(maximum) < float("inf"))
        & (total > 0)
        & (total < float("inf"))
        & (invalid == 0)
        & _row_live(Slots, Generations, LiveRows, Generation, Status, row, CAPACITY, BATCH, RUNNING)
    )
    winners = tl.load(records + 2, rank < WORLD, other=-float("inf"))
    winning_rank = tl.argmax(winners, 0, tie_break_left=True)
    ids = tl.load(records + 3, rank < WORLD, other=0.0)
    token = tl.sum(tl.where(rank == winning_rank, ids, 0.0), 0).to(tl.int64)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    token_id = tl.load(VocabularyOffset) + col
    destination = Probabilities + (row * WIDTH + position) * LOCAL + col
    score = tl.load(destination, col < LOCAL, other=-float("inf"))
    probability = tl.exp(score - maximum) / total
    if not SAMPLE:
        slot = tl.minimum(tl.maximum(tl.load(Slots + row).to(tl.int64), 0), CAPACITY - 1)
        probability = tl.where(tl.load(Params + slot * 6) > 0, probability, (token_id == token).to(tl.float32))
    tl.store(destination, tl.where(valid, probability, (token_id == 0).to(tl.float32)), col < LOCAL)
    if tile == 0:
        tl.store(Valid + row * WIDTH + position, valid)
        if SAMPLE:
            tl.store(Tokens + row * WIDTH + position, tl.where(valid, token, 0))


@triton.jit
def _distribution_masses(
    P, Q, Partials, LOCAL: tl.constexpr, WIDTH: tl.constexpr, BLOCKS: tl.constexpr, BLOCK: tl.constexpr
):
    tile, distribution = tl.program_id(0), tl.program_id(1)
    row, step = distribution // WIDTH, distribution % WIDTH
    col = tile * BLOCK + tl.arange(0, BLOCK)
    p = tl.load(P + distribution * LOCAL + col, col < LOCAL, other=0.0)
    q = tl.full((BLOCK,), 0.0, tl.float32)
    if step < WIDTH - 1:
        q = tl.load(Q + (row * (WIDTH - 1) + step) * LOCAL + col, col < LOCAL, other=0.0)
    invalid = (
        tl.sum(((p < 0) | (q < 0) | (p != p) | (q != q) | (p == float("inf")) | (q == float("inf"))).to(tl.int32), 0)
        != 0
    )
    base = Partials + distribution * 3 * BLOCKS + tile
    tl.store(base, tl.where(invalid, float("nan"), tl.sum(p, 0)))
    tl.store(base + BLOCKS, tl.where(invalid, float("nan"), tl.sum(q, 0)))


@triton.jit
def _selected_records(
    VocabularyOffset,
    P,
    Q,
    Drafts,
    Partials,
    Summary,
    LOCAL: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCKS: tl.constexpr,
    PAD: tl.constexpr,
    WORDS: tl.constexpr,
):
    row = tl.program_id(0)
    word = tl.arange(0, WORDS)
    tl.store(Summary + row * WORDS + word, 0.0)
    tile = tl.arange(0, PAD)
    for step in tl.static_range(WIDTH):
        base = Partials + (row * WIDTH + step) * 3 * BLOCKS + tile
        pmass = tl.sum(tl.load(base, tile < BLOCKS, other=0.0), 0)
        qmass = tl.sum(tl.load(base + BLOCKS, tile < BLOCKS, other=0.0), 0)
        tl.store(Summary + row * WORDS + 2 * (WORDS // 4) + step, pmass)
        tl.store(Summary + row * WORDS + 3 * (WORDS // 4) + step, qmass)
        if step < WIDTH - 1:
            local_token = tl.load(Drafts + row * (WIDTH - 1) + step) - tl.load(VocabularyOffset)
            present = (local_token >= 0) & (local_token < LOCAL)
            p = tl.load(P + (row * WIDTH + step) * LOCAL + local_token, present, other=0.0)
            q = tl.load(Q + (row * (WIDTH - 1) + step) * LOCAL + local_token, present, other=0.0)
            tl.store(Summary + row * WORDS + step, p)
            tl.store(Summary + row * WORDS + WORDS // 4 + step, q)


@triton.jit
def _correction_tiles(
    VocabularyOffset,
    P,
    Q,
    Selected,
    Partials,
    Limits,
    Seeds,
    RngGeneration,
    RngOffset,
    Slots,
    Generations,
    CAPACITY: tl.constexpr,
    LOCAL: tl.constexpr,
    VOCAB: tl.constexpr,
    WIDTH: tl.constexpr,
    POS_PAD: tl.constexpr,
    WORLD: tl.constexpr,
    RANK_PAD: tl.constexpr,
    BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_LIMITS: tl.constexpr,
    WORDS: tl.constexpr,
):
    tile, row = tl.program_id(0), tl.program_id(1)
    rank = tl.arange(0, RANK_PAD)
    first_position = tl.program_id(2) * POS_PAD
    position = first_position + tl.arange(0, POS_PAD)
    selected = Selected + (row * WORLD + rank) * WORDS
    log_prefix = tl.full((), 0.0, tl.float32)
    prefix = tl.full((POS_PAD,), 0.0, tl.float32)
    for step in tl.static_range(WIDTH):
        if (step >= first_position) & (step < first_position + POS_PAD):
            prefix = tl.where(position == step, tl.exp(log_prefix), prefix)
        if step < WIDTH - 1:
            if step < first_position + POS_PAD - 1:
                p_chosen = tl.sum(tl.load(selected + step, rank < WORLD, other=0.0), 0)
                q_chosen = tl.sum(tl.load(selected + WORDS // 4 + step, rank < WORLD, other=0.0), 0)
                log_prefix = tl.minimum(log_prefix + tl.log(p_chosen) - tl.log(q_chosen), 0.0)
    local = tile * BLOCK + tl.arange(0, BLOCK)
    token = tl.load(VocabularyOffset) + local
    inside = (position[:, None] < WIDTH) & (local[None, :] < LOCAL) & (token[None, :] < VOCAB)
    p = tl.load(P + (row * WIDTH + position[:, None]) * LOCAL + local[None, :], inside, other=0.0)
    q = tl.load(
        Q + (row * (WIDTH - 1) + position[:, None]) * LOCAL + local[None, :],
        inside & (position[:, None] < WIDTH - 1),
        other=0.0,
    )
    limit = tl.load(Limits + row) if HAS_LIMITS else WIDTH
    correction = tl.where(position[:, None] < limit - 1, tl.maximum(prefix[:, None] * p - q, 0.0), p)
    slot = tl.minimum(tl.maximum(tl.load(Slots + row).to(tl.int64), 0), CAPACITY - 1)
    seed = tl.load(Seeds + slot).to(tl.int64) ^ _TARGET_DOMAIN
    offset = tl.where(tl.load(RngGeneration + slot) == tl.load(Generations + row), tl.load(RngOffset + slot), 0)
    score = tl.log(correction) + _token_noise(seed, offset + position[:, None], token[None, :])
    score = tl.where(inside & (correction > 0) & (score == score), score, -float("inf"))
    selected_local = tl.argmax(score, 1, tie_break_left=True)
    base = Partials + (row * WIDTH + position) * 3 * BLOCKS + tile
    tl.store(base, tl.sum(correction, 1), position < WIDTH)
    tl.store(base + BLOCKS, tl.max(score, 1), position < WIDTH)
    tl.store(
        base + 2 * BLOCKS, (tl.load(VocabularyOffset) + tile * BLOCK + selected_local).to(tl.float32), position < WIDTH
    )


@triton.jit
def _correction_records(Partials, Summary, BLOCKS: tl.constexpr, PAD: tl.constexpr, WORDS: tl.constexpr):
    distribution = tl.program_id(0)
    tile = tl.arange(0, PAD)
    base = Partials + distribution * 3 * BLOCKS + tile
    masses = tl.load(base, tile < BLOCKS, other=0.0)
    scores = tl.load(base + BLOCKS, tile < BLOCKS, other=-float("inf"))
    ids = tl.load(base + 2 * BLOCKS, tile < BLOCKS, other=0.0)
    winner = tl.argmax(scores, 0, tie_break_left=True)
    token = tl.sum(tl.where(tile == winner, ids, 0.0), 0)
    word = tl.arange(0, WORDS)
    record = tl.where(
        word == 0, tl.sum(masses, 0), tl.where(word == 1, tl.max(scores, 0), tl.where(word == 2, token, 0.0))
    )
    tl.store(Summary + distribution * WORDS + word, record)


@triton.jit
def _complete_block(
    Selected,
    Residual,
    Drafts,
    DraftValid,
    TargetValid,
    Limits,
    Slots,
    Generations,
    LiveRows,
    Generation,
    Status,
    Seeds,
    RngGeneration,
    RngOffset,
    Tokens,
    Count,
    Accepted,
    Offered,
    Valid,
    Increment,
    Survival,
    WIDTH: tl.constexpr,
    WORLD: tl.constexpr,
    RANK_PAD: tl.constexpr,
    CAPACITY: tl.constexpr,
    BATCH: tl.constexpr,
    VOCAB: tl.constexpr,
    RUNNING: tl.constexpr,
    RESERVATION: tl.constexpr,
    HAS_LIMITS: tl.constexpr,
    WORDS: tl.constexpr,
    SURVIVAL: tl.constexpr,
    POSITIONS: tl.constexpr,
):
    row = tl.program_id(0)
    rank = tl.arange(0, RANK_PAD)
    records = Selected + (row * WORLD + rank) * WORDS
    limit = tl.load(Limits + row) if HAS_LIMITS else WIDTH
    valid = (
        (limit >= 1)
        & (limit <= WIDTH)
        & _row_live(Slots, Generations, LiveRows, Generation, Status, row, CAPACITY, BATCH, RUNNING)
    )
    slot = tl.minimum(tl.maximum(tl.load(Slots + row).to(tl.int64), 0), CAPACITY - 1)
    seed = tl.load(Seeds + slot).to(tl.int64) ^ _TARGET_DOMAIN
    offset = tl.where(tl.load(RngGeneration + slot) == tl.load(Generations + row), tl.load(RngOffset + slot), 0)
    log_prefix, survival = tl.full((), 0.0, tl.float32), tl.full((), 1.0, tl.float32)
    accepted = tl.full((), 0, tl.int32)
    step_ids = tl.arange(0, POSITIONS)
    survivals = tl.full((POSITIONS,), -1.0, tl.float32)
    for step in tl.static_range(WIDTH):
        if step < limit:
            mass = tl.sum(tl.load(records + 2 * (WORDS // 4) + step, rank < WORLD, other=0.0), 0)
            valid &= (tl.abs(mass - 1) <= 1e-4) & tl.load(TargetValid + row * WIDTH + step)
            if step < WIDTH - 1:
                if step < limit - 1:
                    qmass = tl.sum(tl.load(records + 3 * (WORDS // 4) + step, rank < WORLD, other=0.0), 0)
                    token = tl.load(Drafts + row * (WIDTH - 1) + step)
                    q = tl.sum(tl.load(records + WORDS // 4 + step, rank < WORLD, other=0.0), 0)
                    p = tl.sum(tl.load(records + step, rank < WORLD, other=0.0), 0)
                    valid &= (
                        (tl.abs(qmass - 1) <= 1e-4)
                        & (q > 0)
                        & (token >= 0)
                        & (token < VOCAB)
                        & tl.load(DraftValid + row * (WIDTH - 1) + step)
                    )
                    log_prefix = tl.minimum(log_prefix + tl.log(p) - tl.log(q), 0.0)
                    prefix = tl.exp(log_prefix)
                    probability = prefix
                    if step < limit - 2:
                        next_record = Residual + ((row * WIDTH + step + 1) * WORLD + rank) * WORDS
                        mass = tl.sum(tl.load(next_record, rank < WORLD, other=0.0), 0)
                        denominator = mass + 1.0 - prefix
                        probability = tl.where(denominator > 0, mass / denominator, 1.0)
                    counter = (offset + step).to(tl.uint64)
                    bits, _, _, _ = tl.philox(seed, counter.to(tl.uint32), (counter >> 32).to(tl.uint32), 0, 0)
                    draw = (bits.to(tl.float32) + 0.5) * 2.3283064365386963e-10
                    accepted = tl.where(draw <= probability, step + 1, accepted)
                    survival *= tl.minimum(tl.div_rn(p, q), 1.0)
                    survivals = tl.where(step_ids == step, survival, survivals)
    correction = Residual + ((row * WIDTH + accepted) * WORLD + rank) * WORDS
    winners = tl.load(correction + 1, rank < WORLD, other=-float("inf"))
    selected_rank = tl.argmax(winners, 0, tie_break_left=True)
    ids = tl.load(correction + 2, rank < WORLD, other=0.0)
    next_token = tl.sum(tl.where(rank == selected_rank, ids, 0.0), 0).to(tl.int64)
    valid &= tl.max(winners, 0) > -float("inf")
    draft = tl.load(Drafts + row * (WIDTH - 1) + step_ids, step_ids < WIDTH - 1, other=-1)
    output = tl.where(step_ids < accepted, draft, next_token)
    tl.store(Tokens + row * WIDTH + step_ids, tl.where(valid & (step_ids <= accepted), output, -1), step_ids < WIDTH)
    tl.store(Survival + row * SURVIVAL + step_ids, tl.where(valid, survivals, -1.0), step_ids < SURVIVAL)
    tl.store(Valid + row, valid)
    tl.store(Offered + row, tl.minimum(tl.maximum(limit - 1, 0), WIDTH - 1))
    tl.store(Accepted + row, tl.where(valid, accepted, 0))
    tl.store(Count + row, tl.where(valid, accepted + 1, 0))
    tl.store(Increment + row, tl.where(valid, RESERVATION, 0))


class PreparedVocabularySoftmax:
    """Full-softmax or greedy target distributions, retaining only local FP32 p.

    Top-k, top-p and min-p filters must be disabled in the declared serving
    policy. Unexpected device filter state invalidates the row. Penalties,
    temperature and grammar use global token IDs and the committed histogram.
    """

    def __init__(
        self,
        logits,
        probabilities,
        valid,
        requests,
        slots,
        generations,
        live_rows,
        candidates,
        *,
        start,
        runtime,
        session,
        coordinator,
        name,
        workspace=None,
        embedding=None,
        head=None,
        pending=None,
        seeds=None,
        offsets=None,
        tokens=None,
    ):
        if logits.ndim != 3 or logits.dtype not in (torch.bfloat16, torch.float32) or logits.stride(-1) != 1:
            raise ValueError("vocabulary logits require BF16/FP32 [batch,width,local vocabulary]")
        b, w, local = logits.shape
        sample = tokens is not None
        _geometry(b, w + 1 if sample else w)
        world = 1 if runtime is None else runtime.world_size
        if (
            type(start) is not int
            or start < 0
            or start % local
            or start + local > local * world
            or not (local * (world - 1) < requests.vocab_size <= local * world)
            or requests.vocab_size > 2**24
        ):
            raise ValueError("vocabulary shards require equal contiguous partitions covering exactly the vocabulary")
        device = logits.device
        _metadata(requests, slots, generations, live_rows, b, device)
        _tensor(probabilities, (b, w, local), torch.float32, device, "local probabilities")
        _tensor(valid, (b, w), torch.bool, device, "distribution validity")
        _tensor(candidates, (b, w if sample else w - 1), torch.int64, device, "candidate prefix IDs")
        self.markov = embedding is not None
        fp8 = isinstance(head, RowScaledFP8MarkovWeight)
        values = head.values if fp8 else head
        scale = head.scale if fp8 else None
        if sample:
            _tensor(tokens, (b, w), torch.int64, device, "proposal tokens")
            _tensor(seeds, (b,), torch.int64, device, "proposal seeds")
            _tensor(offsets, (b,), torch.int64, device, "proposal offsets")
            if self.markov:
                if (
                    embedding.dtype != torch.bfloat16
                    or embedding.ndim != 2
                    or embedding.shape[0] != requests.vocab_size
                    or not embedding.is_contiguous()
                    or values.shape != (local, embedding.shape[1])
                    or not values.is_contiguous()
                    or values.dtype != (torch.float8_e4m3fn if fp8 else torch.bfloat16)
                    or embedding.shape[1] % 64
                    or values.device != device
                    or embedding.device != device
                ):
                    raise ValueError(
                        "Markov proposals require replicated BF16 embeddings and a contiguous BF16/E4M3 head shard"
                    )
                _tensor(pending, (b,), torch.int64, device, "pending tokens")
                if fp8:
                    record_numeric_tensor(values)
                    record_numeric_tensor(scale)
        elif self.markov:
            raise ValueError("target softmax cannot attach proposal Markov weights")
        self.logits, self.probabilities, self.valid = logits, probabilities, valid
        self.state, self.slots, self.generations, self.live_rows = requests, slots, generations, live_rows
        self.candidates, self.start, self.world, self.batch, self.width, self.local = (
            candidates,
            start,
            world,
            b,
            w,
            local,
        )
        # Rank-local vocabulary placement is a binding, so TP peers reuse the
        # same compiled computation while retaining their own global token IDs.
        self.vocabulary_offset = persistent_copy(torch.tensor([start], dtype=torch.int32, device=device))
        self.embedding, self.head, self.scale, self.pending = embedding, values, scale, pending
        self.seeds, self.offsets, self.tokens, self.sample = seeds, offsets, tokens, sample
        summary_rows = b if sample else b * w
        self.scratch = bind_scratch(
            dict(
                partials=((summary_rows, 5, triton.cdiv(local, 256)), torch.float32),
                summary=((summary_rows, SUMMARY_WORDS), torch.float32),
                gathered=((summary_rows, world * SUMMARY_WORDS), torch.float32),
            ),
            device=device,
            workspace=workspace,
        )
        self.gather = PreparedAllGather(
            self.scratch["summary"],
            self.scratch["gathered"],
            runtime=runtime,
            session=session,
            coordinator=coordinator,
            name=name + ".statistics",
        )
        self.steps = None

    def prepare(self):
        from .sampling import RUNNING

        b, w, local, s, state = self.batch, self.width, self.local, self.scratch, self.state
        steps = []
        for step in range(w if self.sample else 1):
            summary_rows = b if self.sample else b * w
            score = prepare_triton(
                _score_tiles,
                (
                    self.vocabulary_offset,
                    self.logits,
                    self.probabilities,
                    s["partials"],
                    self.embedding,
                    self.head,
                    self.scale,
                    self.pending,
                    self.candidates,
                    state.histogram,
                    self.slots,
                    self.generations,
                    self.live_rows,
                    state.generation,
                    state.status,
                    state.parameters,
                    state.top_k,
                    state.grammar_mask,
                    state.grammar_enabled,
                    self.seeds,
                    self.offsets,
                    self.logits.stride(0),
                    self.logits.stride(1),
                    state.histogram.stride(0),
                    state.grammar_mask.stride(0),
                ),
                (triton.cdiv(local, 256), summary_rows, 1),
                LOCAL=local,
                VOCAB=state.vocab_size,
                WIDTH=w,
                DRAFTS=self.candidates.shape[1],
                PREFIX=step if self.sample else w - 1,
                BLOCKS=triton.cdiv(local, 256),
                CAPACITY=state.capacity,
                BATCH=b,
                RUNNING=RUNNING,
                MARKOV=self.markov,
                RANK=64 if self.head is None else self.head.shape[1],
                FP8=self.scale is not None,
                SAMPLE=self.sample,
                STEP=step,
                BLOCK=256,
                num_warps=4,
                enable_fp_fusion=False,
            )
            summary = prepare_triton(
                _softmax_records,
                (s["partials"], s["summary"]),
                (summary_rows, 1, 1),
                BLOCKS=triton.cdiv(local, 256),
                PAD=triton.next_power_of_2(triton.cdiv(local, 256)),
                num_warps=4,
            )
            normalize = prepare_triton(
                _finish_distribution,
                (
                    self.vocabulary_offset,
                    self.probabilities,
                    s["gathered"],
                    self.valid,
                    self.tokens,
                    self.slots,
                    self.generations,
                    self.live_rows,
                    state.generation,
                    state.status,
                    state.parameters,
                ),
                (triton.cdiv(local, 256), summary_rows, 1),
                LOCAL=local,
                VOCAB=state.vocab_size,
                WIDTH=w,
                STEP=step,
                WORLD=self.world,
                CAPACITY=state.capacity,
                BATCH=b,
                RUNNING=RUNNING,
                SAMPLE=self.sample,
                BLOCK=256,
                RANK_PAD=triton.next_power_of_2(self.world),
                num_warps=4,
                enable_fp_fusion=False,
            )
            steps.append((score, summary, normalize))
        self.steps = tuple(steps)

    def functional(self, *, session):
        if self.steps is None:
            raise RuntimeError("vocabulary distributions require preparation before functional conversion")
        values = dict(self.scratch, logits=self.logits, probabilities=self.probabilities,
            valid=self.valid, candidates=self.candidates, pending=self.pending,
            seeds=self.seeds, offsets=self.offsets, tokens=self.tokens)
        self.local_layout = {name: (tuple(value.shape), value.dtype) for name, value in values.items()
                             if value is not None}
        names = {id(value): name for name, value in values.items() if value is not None}
        for operations in self.steps:
            for operation in operations:
                operation.local_slots = tuple((index, names[id(value)]) for index, value in enumerate(operation.args)
                                              if id(value) in names)
                operation.args = tuple(None if id(value) in names else value for value in operation.args)
        self.gather.functional(session=session)
        self.logits = self.probabilities = self.valid = self.candidates = self.pending = None
        self.seeds = self.offsets = self.tokens = self.scratch = None
        return self

    def run(self, logits, candidates, *, pending=None, seeds=None, offsets=None,
            probabilities=None, valid=None, tokens=None):
        local = {name: torch.empty(shape, dtype=dtype, device=logits.device)
                 for name, (shape, dtype) in self.local_layout.items()
                 if name not in ("logits", "candidates", "pending", "seeds", "offsets")}
        local.update(logits=logits, candidates=candidates, pending=pending, seeds=seeds, offsets=offsets)
        for name, value in (("probabilities", probabilities), ("valid", valid), ("tokens", tokens)):
            if value is not None:
                local[name] = value
        for score, summary, normalize in self.steps:
            for operation in (score, summary):
                args = list(operation.args)
                for index, name in operation.local_slots:
                    args[index] = local[name]
                launch_triton(operation.kernel, operation.compiled, tuple(args), operation.grid, **operation.options)
            local["gathered"] = self.gather(local["summary"])
            args = list(normalize.args)
            for index, name in normalize.local_slots:
                args[index] = local[name]
            launch_triton(normalize.kernel, normalize.compiled, tuple(args), normalize.grid, **normalize.options)
        return local["probabilities"], local["valid"], local.get("tokens")

    def __call__(self):
        if self.steps is None:
            raise RuntimeError("vocabulary distributions require preparation")
        for score, summary, normalize in self.steps:
            score()
            summary()
            self.gather(self.scratch["summary"])
            normalize()
        return self.probabilities


class PreparedVocabularyBlockVerification:
    """Speculative block acceptance with local FP32 proposal and target probabilities.

    For target probabilities p_i and the actual conditional proposals q_i, the
    block recurrence starts with r_0=1 and updates
    r_(i+1)=min(r_i*p_i(x_i)/q_i(x_i),1). Correction i has unnormalized mass
    d_i(x)=max(r_i*p_i(x)-q_i(x),0). The decision after draft i uses
    h_i=sum(d_(i+1))/(sum(d_(i+1))+1-r_(i+1)) when another draft follows,
    and h_i=r_(i+1) otherwise. The last successful decision selects the emitted
    prefix; its correction is sampled from d_i, or from p at the bonus position.
    The last successful decision determines the prefix; stopping at its first failed decision would
    implement a different algorithm. Log-domain recurrence preserves zero
    prefixes without overflow in a later p/q ratio.

    Normalized p and q remain on their vocabulary owner. Selected-token
    probabilities and residual winners cross ranks in fixed-size summaries.
    Scalar acceptance and token-wise residual draws use the request's target
    seed domain and complete 64-bit offsets; no batch coordinate enters RNG.
    The seeded trace differs from FlashInfer's serial-chain CDF sampling.
    """

    def __init__(
        self,
        draft_probs,
        drafts,
        target_probs,
        requests,
        slots,
        generations,
        live_rows,
        *,
        start,
        runtime,
        session,
        coordinator,
        name,
        rng,
        draft_valid,
        target_valid,
        token_limits=None,
        output=None,
        workspace=None,
    ):
        b, w, local = target_probs.shape
        _geometry(b, w)
        device = target_probs.device
        world = 1 if runtime is None else runtime.world_size
        if (
            type(start) is not int
            or start < 0
            or start % local
            or start + local > local * world
            or not local * (world - 1) < requests.vocab_size <= local * world
        ):
            raise ValueError("block verification requires equal contiguous vocabulary partitions")
        for value, shape, dtype, label in (
            (draft_probs, (b, w - 1, local), torch.float32, "proposal probabilities"),
            (target_probs, (b, w, local), torch.float32, "target probabilities"),
            (drafts, (b, w - 1), torch.int64, "draft tokens"),
            (draft_valid, (b, w - 1), torch.bool, "draft validity"),
            (target_valid, (b, w), torch.bool, "target validity"),
            (rng.generation, (requests.capacity,), torch.int64, "RNG generation"),
            (rng.target_offset, (requests.capacity,), torch.int64, "RNG offset"),
        ):
            _tensor(value, shape, dtype, device, label)
        _metadata(requests, slots, generations, live_rows, b, device)
        if token_limits is not None:
            _tensor(token_limits, (b,), torch.int32, device, "token limits")
        self.p, self.q, self.drafts = target_probs, draft_probs, drafts
        self.state, self.slots, self.generations, self.live_rows, self.rng = (
            requests,
            slots,
            generations,
            live_rows,
            rng,
        )
        self.draft_valid, self.target_valid, self.limits = draft_valid, target_valid, token_limits
        self.has_token_limits = token_limits is not None
        self.batch, self.width, self.local, self.start, self.world = b, w, local, start, world
        # Rank-local vocabulary placement is a binding, so TP peers reuse the
        # same compiled computation while retaining their own global token IDs.
        self.vocabulary_offset = persistent_copy(torch.tensor([start], dtype=torch.int32, device=device))
        self.output = SpeculativeAcceptance.allocate(b, w, device) if output is None else output
        self.output.validate(b, w, device)
        self.draw_reservation = 4 * ((w + 1 + 3) // 4)
        blocks = triton.cdiv(local, 256)
        self.summary_words = max(SUMMARY_WORDS, 4 * triton.next_power_of_2(w))
        self.scratch = bind_scratch(
            dict(
                partials=((b, w, 3, blocks), torch.float32),
                selected_summary=((b, self.summary_words), torch.float32),
                selected=((b, world * self.summary_words), torch.float32),
                residual_summary=((b * w, self.summary_words), torch.float32),
                residual=((b * w, world * self.summary_words), torch.float32),
            ),
            device=device,
            workspace=workspace,
        )
        self.selected_gather = PreparedAllGather(
            self.scratch["selected_summary"],
            self.scratch["selected"],
            runtime=runtime,
            session=session,
            coordinator=coordinator,
            name=name + ".selected",
        )
        self.residual_gather = PreparedAllGather(
            self.scratch["residual_summary"],
            self.scratch["residual"],
            runtime=runtime,
            session=session,
            coordinator=coordinator,
            name=name + ".residual",
        )
        self.kernels = None

    def prepare(self):
        from .sampling import RUNNING

        b, w, local, state, s, out = self.batch, self.width, self.local, self.state, self.scratch, self.output
        blocks = triton.cdiv(local, 256)
        partials = prepare_triton(
            _distribution_masses,
            (self.p, self.q, s["partials"]),
            (blocks, b * w, 1),
            LOCAL=local,
            WIDTH=w,
            BLOCKS=blocks,
            BLOCK=256,
            num_warps=4,
        )
        selected = prepare_triton(
            _selected_records,
            (self.vocabulary_offset, self.p, self.q, self.drafts, s["partials"], s["selected_summary"]),
            (b, 1, 1),
            LOCAL=local,
            WIDTH=w,
            BLOCKS=blocks,
            PAD=triton.next_power_of_2(blocks),
            WORDS=self.summary_words,
            num_warps=4,
        )
        # Two positions share prefix work while retaining enough independent
        # vocabulary tiles to occupy the GPU for a single request.
        positions = min(w, 2)
        residual = prepare_triton(
            _correction_tiles,
            (
                self.vocabulary_offset,
                self.p,
                self.q,
                s["selected"],
                s["partials"],
                self.limits,
                state.seed,
                self.rng.generation,
                self.rng.target_offset,
                self.slots,
                self.generations,
            ),
            (blocks, b, triton.cdiv(w, positions)),
            CAPACITY=state.capacity,
            LOCAL=local,
            VOCAB=state.vocab_size,
            WIDTH=w,
            POS_PAD=positions,
            WORLD=self.world,
            RANK_PAD=triton.next_power_of_2(self.world),
            BLOCKS=blocks,
            BLOCK=256,
            HAS_LIMITS=self.limits is not None,
            WORDS=self.summary_words,
            num_warps=4,
            enable_fp_fusion=False,
        )
        summary = prepare_triton(
            _correction_records,
            (s["partials"], s["residual_summary"]),
            (b * w, 1, 1),
            BLOCKS=blocks,
            PAD=triton.next_power_of_2(blocks),
            WORDS=self.summary_words,
            num_warps=4,
        )
        finish = prepare_triton(
            _complete_block,
            (
                s["selected"],
                s["residual"],
                self.drafts,
                self.draft_valid,
                self.target_valid,
                self.limits,
                self.slots,
                self.generations,
                self.live_rows,
                state.generation,
                state.status,
                state.seed,
                self.rng.generation,
                self.rng.target_offset,
                out.token_ids,
                out.proposed_count,
                out.accepted_drafts,
                out.offered_drafts,
                out.valid,
                out.rng_increment,
                out.survival,
            ),
            (b, 1, 1),
            WIDTH=w,
            WORLD=self.world,
            RANK_PAD=triton.next_power_of_2(self.world),
            CAPACITY=state.capacity,
            BATCH=b,
            VOCAB=state.vocab_size,
            RUNNING=RUNNING,
            RESERVATION=self.draw_reservation,
            HAS_LIMITS=self.limits is not None,
            WORDS=self.summary_words,
            SURVIVAL=out.survival.shape[1],
            POSITIONS=triton.next_power_of_2(max(w, out.survival.shape[1])),
            num_warps=4,
            enable_fp_fusion=False,
        )
        self.kernels = partials, selected, residual, summary, finish

    def functional(self, *, session):
        if self.kernels is None:
            raise RuntimeError("vocabulary verification requires preparation before functional conversion")
        values = dict(self.scratch, p=self.p, q=self.q, drafts=self.drafts,
            draft_valid=self.draft_valid, target_valid=self.target_valid)
        if self.has_token_limits:
            values["limits"] = self.limits
        values.update({"output." + name: getattr(self.output, name) for name in self.output.__dataclass_fields__})
        self.local_layout = {name: (tuple(value.shape), value.dtype) for name, value in values.items()}
        names = {id(value): name for name, value in values.items()}
        for operation in self.kernels:
            operation.local_slots = tuple((index, names[id(value)]) for index, value in enumerate(operation.args)
                                          if id(value) in names)
            operation.args = tuple(None if id(value) in names else value for value in operation.args)
        self.selected_gather.functional(session=session)
        self.residual_gather.functional(session=session)
        self.p = self.q = self.drafts = self.draft_valid = self.target_valid = self.output = self.scratch = self.limits = None
        return self

    def run(self, draft_probs, drafts, target_probs, draft_valid, target_valid, *, token_limits=None):
        if (token_limits is not None) != self.has_token_limits:
            raise ValueError("vocabulary verification token limits must match the prepared mask contract")
        if token_limits is not None:
            _tensor(token_limits, (self.batch,), torch.int32, target_probs.device, "token limits")
        out = SpeculativeAcceptance.allocate(self.batch, self.width, target_probs.device)
        local = {name: torch.empty(shape, dtype=dtype, device=target_probs.device)
                 for name, (shape, dtype) in self.local_layout.items()
                 if name not in ("p", "q", "drafts", "draft_valid", "target_valid", "limits") and not name.startswith("output.")}
        local.update(p=target_probs, q=draft_probs, drafts=drafts,
                     draft_valid=draft_valid, target_valid=target_valid, limits=token_limits)
        local.update({"output." + name: getattr(out, name) for name in out.__dataclass_fields__})
        partials, selected, residual, summary, finish = self.kernels
        for operation in (partials, selected, residual, summary, finish):
            if operation is residual:
                local["selected"] = self.selected_gather(local["selected_summary"])
            if operation is finish:
                local["residual"] = self.residual_gather(local["residual_summary"])
            args = list(operation.args)
            for index, name in operation.local_slots:
                args[index] = local[name]
            launch_triton(operation.kernel, operation.compiled, tuple(args), operation.grid, **operation.options)
        return out

    def __call__(self):
        if self.kernels is None:
            raise RuntimeError("vocabulary block verification requires preparation")
        partials, selected, residual, summary, finish = self.kernels
        partials()
        selected()
        self.selected_gather(self.scratch["selected_summary"])
        residual()
        summary()
        self.residual_gather(self.scratch["residual_summary"])
        finish()
        return self.output
