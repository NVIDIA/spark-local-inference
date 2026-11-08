# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded top-k draft proposals: filtering and sampling over at most DRAFT_CANDIDATES tokens.

Proposals follow the request policy after penalties, grammar and temperature:
top-k, softmax, top-p and min-p, each retaining boundary ties among the
candidates, then renormalization. A request whose top-k exceeds
DRAFT_CANDIDATES, including disabled top-k, proposes from the
DRAFT_CANDIDATES highest scores. Chain verification receives exactly the
distribution each draft was drawn from, so verified output is unchanged;
only acceptance depends on how closely proposals follow the target policy.

The first kernel writes zeros over the dense proposal row and keeps each
block's highest scores. The second selects the row's candidates, applies the
policy, draws one Philox uniform, and scatters the surviving probabilities.
"""

import torch
import triton
import triton.language as tl

from ignition.components.speculation import PreparedSpeculativeDistributions

DRAFT_CANDIDATES = 64
BLOCK = 4096


@triton.jit
def _orderable(scores):
    # Signed integers ordered like the scores; NaN orders above infinity and is rejected later.
    bits = scores.to(tl.int32, bitcast=True)
    # IEEE signed zeros are equal scores and share the token-index tie order.
    bits = tl.where(scores == 0., 0, bits)
    return tl.where(bits < 0, bits ^ 2147483647, bits)


@triton.jit
def _kth_largest(ordered, k):
    # Largest t with at least k values >= t, by bisection over the 32-bit order.
    low = tl.full((), -2147483648, tl.int64)
    high = tl.full((), 2147483647, tl.int64)
    wide = ordered.to(tl.int64)
    for _ in range(32):
        middle = (low + high + 1) >> 1
        enough = tl.sum((wide >= middle).to(tl.int32), 0) >= k
        low = tl.where(enough, middle, low)
        high = tl.where(enough, high, middle - 1)
    return low


@triton.jit
def _candidate_positions(ordered, pivot, present, K: tl.constexpr):
    # Cutoff ties may exceed the bounded buffer. Every strictly higher score
    # must survive before ties consume its remaining slots.
    above = present & (ordered.to(tl.int64) > pivot)
    tied = present & (ordered.to(tl.int64) == pivot)
    higher_count = tl.sum(above.to(tl.int32), 0)
    position = tl.where(above, tl.cumsum(above.to(tl.int32), 0) - 1,
                        higher_count + tl.cumsum(tied.to(tl.int32), 0) - 1)
    take = (above | tied) & (position < K)
    count = tl.minimum(higher_count + tl.sum(tied.to(tl.int32), 0), K)
    return position, take, count


@triton.jit
def _block_candidates(Scores, Probabilities, TopK, Candidates, V: tl.constexpr, BLOCKS: tl.constexpr,
                      BLOCK: tl.constexpr, K: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    tokens = block * BLOCK + tl.arange(0, BLOCK)
    inside = tokens < V
    scores = tl.load(Scores + row * V + tokens, inside, other=-float("inf"))
    tl.store(Probabilities + row * V + tokens, tl.zeros((BLOCK,), tl.float32), inside)
    k = tl.minimum(tl.maximum(tl.load(TopK + row), 1), K)
    ordered = tl.where(inside, _orderable(scores), -2147483648)
    position, take, count = _candidate_positions(ordered, _kth_largest(ordered, k), inside, K)
    keys = (ordered.to(tl.int64) << 32) | (2147483647 - tokens).to(tl.int64)
    base = Candidates + (row * BLOCKS + block) * K
    slots = tl.arange(0, K)
    tl.store(base + slots, tl.full((K,), -9223372036854775808, tl.int64), slots >= count)
    tl.store(base + position, keys, take)


@triton.jit
def _shortlist_candidates(Scores, Ids, Candidates, id_stride, V: tl.constexpr,
                          SUPPORT: tl.constexpr, POOL: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0)
    index = tl.arange(0, POOL)
    # Padding keeps top-k cutoffs above the support size valid without duplicate token IDs.
    padding = index - SUPPORT
    token = tl.load(Ids + row * id_stride + index, index < SUPPORT, other=0)
    token = tl.where(index < SUPPORT, token, padding)
    present = (index < SUPPORT) | ((padding >= 0) & (padding < K + SUPPORT) & (padding < V))
    for entry in tl.static_range(SUPPORT):
        support_id = tl.load(Ids + row * id_stride + entry)
        present &= (index < SUPPORT) | (token != support_id)
    present &= (token >= 0) & (token < V)
    scores = tl.load(Scores + row * V + token, present, other=-float("inf"))
    keys = (_orderable(scores).to(tl.int64) << 32) | (2147483647 - token).to(tl.int64)
    tl.store(Candidates + row * POOL + index, tl.where(present, keys, -9223372036854775808))


@triton.jit
def _select_and_sample(Candidates, Selected, Probabilities, Valid, TopK, TopP, Params, Slots, Seed,
                       Offset, Sampled, V: tl.constexpr, BLOCKS: tl.constexpr, POOL: tl.constexpr, K: tl.constexpr,
                       SLOTS: tl.constexpr, SupportIds, SupportProbabilities, WRITE_SUPPORT: tl.constexpr):
    row = tl.program_id(0)
    index = tl.arange(0, POOL)
    pool = tl.load(Candidates + row * BLOCKS * K + index, index < BLOCKS * K, other=-9223372036854775808)
    top_k = tl.minimum(tl.maximum(tl.load(TopK + row), 1), K)
    present = pool != -9223372036854775808
    ordered = (pool >> 32).to(tl.int32)
    position, take, _ = _candidate_positions(ordered, _kth_largest(ordered, top_k), present, K)
    rank = tl.arange(0, K)
    tl.store(Selected + row * K + rank, tl.full((K,), -9223372036854775808, tl.int64))
    tl.debug_barrier()
    tl.store(Selected + row * K + position, pool, take)
    tl.debug_barrier()
    keys = tl.sort(tl.load(Selected + row * K + rank), descending=True)
    ordered = (keys >> 32).to(tl.int32)
    scores = tl.where(ordered < 0, ordered ^ 2147483647, ordered).to(tl.float32, bitcast=True)
    tokens = 2147483647 - (keys & 2147483647).to(tl.int32)
    present = keys != -9223372036854775808
    finite = present & (scores == scores) & (scores != float("inf")) & (scores != -float("inf"))  # noqa: PLR0124

    slot = tl.minimum(tl.maximum(tl.load(Slots + row), 0), SLOTS - 1)
    min_p = tl.load(Params + slot * 6 + 2)
    top_p = tl.load(TopP + row)

    # Top-k keeps every candidate tied with the k-th score.
    pivot = tl.max(tl.where(rank == top_k - 1, scores, -float("inf")), 0)
    kept = finite & (scores >= pivot)
    maximum = tl.max(tl.where(kept, scores, -float("inf")), 0)
    weights = tl.where(kept, tl.exp(scores - maximum), 0.)
    probabilities = weights / tl.sum(weights, 0)
    # Top-p keeps the shortest descending prefix reaching top_p and ties with its last member.
    before = tl.cumsum(probabilities, 0) - probabilities
    cutoff = tl.min(tl.where(kept & (before < top_p), probabilities, float("inf")), 0)
    kept &= probabilities >= cutoff
    probabilities = tl.where(kept, probabilities, 0.)
    probabilities = probabilities / tl.sum(probabilities, 0)
    # Min-p against the largest remaining probability, then renormalize.
    kept &= (probabilities > 0.) & (probabilities >= tl.max(probabilities, 0) * min_p)
    probabilities = tl.where(kept, probabilities, 0.)
    mass = tl.sum(probabilities, 0)
    valid = (tl.load(Valid + row) != 0) & (mass > 0.) & (mass == mass) & (mass != float("inf"))  # noqa: PLR0124
    probabilities = probabilities / tl.where(mass > 0., mass, 1.)

    seed = tl.load(Seed + row)
    offset = (tl.load(Offset + row) & 4294967295).to(tl.uint32)
    uniform = tl.rand(seed, offset)
    cumulative = tl.cumsum(probabilities, 0)
    chosen = tl.sum(((cumulative <= uniform) & kept).to(tl.int32), 0)
    last = tl.max(tl.where(kept, rank, 0), 0)
    chosen = tl.minimum(chosen, last)
    sampled = tl.sum(tl.where(rank == chosen, tokens, 0), 0)
    sampled = tl.where(valid, sampled, 0)

    tl.store(Probabilities + row * V + tokens, probabilities, kept & valid)
    tl.store(Probabilities + row * V + sampled, 1., ~valid)
    tl.store(Valid + row, valid)
    tl.store(Sampled + row, sampled.to(tl.int64))
    if WRITE_SUPPORT:
        # Keep the original Selected order and exact dense-q bits, including padding at token zero.
        tl.debug_barrier()
        local_keys = tl.load(Selected + row * K + rank)
        support_ids = tl.where(local_keys != -9223372036854775808,
                               2147483647 - (local_keys & 2147483647).to(tl.int32), 0)
        support_probabilities = tl.load(Probabilities + row * V + support_ids)
        tl.store(SupportIds + row * K + rank, tl.where(valid, support_ids, 0))
        tl.store(SupportProbabilities + row * K + rank, tl.where(valid, support_probabilities, 1.))


@triton.jit
def _restore_draft_support(Selected, SupportIds, SupportProbabilities, Probabilities,
                           V: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0)
    rank = tl.arange(0, K)
    keys = tl.load(Selected + row * K + rank)
    local_ids = 2147483647 - (keys & 2147483647).to(tl.int32)
    # Candidate generation zeroed the dense row; selection only wrote its local support or token zero.
    tl.store(Probabilities + row * V + local_ids, 0.,
             (keys != -9223372036854775808) & (local_ids > 0) & (local_ids < V))
    tl.store(Probabilities + row * V, 0.)
    tl.debug_barrier()
    ids = tl.load(SupportIds + row * K + rank)
    values = tl.load(SupportProbabilities + row * K + rank)
    # Padding repeats zero with identical bits. Write it once without floating-point arithmetic.
    zero_bits = tl.max(tl.where(ids == 0, values.to(tl.uint32, bitcast=True), 0), 0)
    tl.store(Probabilities + row * V, zero_bits.to(tl.float32, bitcast=True))
    tl.store(Probabilities + row * V + ids, values, ids != 0)


@triton.jit
def _draft_score_values(Logits, Bias, Histogram, Candidates, Slots, Params, GrammarMask, GrammarEnabled,
                        hist_stride, mask_stride, candidate_stride, logit_stride, bias_stride,
                        ADD_BIAS: tl.constexpr, CAPACITY: tl.constexpr, V: tl.constexpr,
                        PREFIX: tl.constexpr, BLOCK: tl.constexpr):
    # One vocabulary block of one row: the committed histogram plus the row's earlier drafts give the penalty
    # counts; repetition, presence and frequency penalties, temperature and the grammar mask follow the
    # unfused order of operations. The caller reduces each block's maximum.
    row, block = tl.program_id(0), tl.program_id(1)
    safe = tl.minimum(tl.maximum(tl.load(Slots + row).to(tl.int64), 0), CAPACITY - 1)
    token = block * BLOCK + tl.arange(0, BLOCK)
    inside = token < V
    score = tl.load(Logits + row * logit_stride + token, mask=inside, other=0.).to(tl.float32)
    if ADD_BIAS:
        score = score + tl.load(Bias + row * bias_stride + token, mask=inside, other=0.).to(tl.float32)
    temperature = tl.load(Params + safe * 6)
    repetition = tl.load(Params + safe * 6 + 3)
    presence = tl.load(Params + safe * 6 + 4)
    frequency = tl.load(Params + safe * 6 + 5)
    # Nonnegative counts and positive-zero penalties leave signed logits unchanged.
    # Prefix validity remains a separate check in _draft_scores_finish.
    neutral = ((repetition == 1.) & (presence.to(tl.int32, bitcast=True) == 0)
        & (frequency.to(tl.int32, bitcast=True) == 0))
    if not neutral:
        count = tl.load(Histogram + safe * hist_stride + token, mask=inside, other=0)
        for index in tl.static_range(PREFIX):
            draft = tl.load(Candidates + row * candidate_stride + index)
            count += ((token == draft) & (draft >= 0) & (draft < V)).to(tl.int32)
        # IEEE division and no FMA contraction preserve the unfused arithmetic.
        adjusted = tl.where(score > 0, tl.math.div_rn(score, repetition), score * repetition)
        score = tl.where(count > 0, adjusted, score)
        score = score - (count > 0).to(tl.float32) * presence - count.to(tl.float32) * frequency
    score = tl.math.div_rn(score, tl.where(temperature > 0, temperature, 1.))
    if tl.load(GrammarEnabled + safe) != 0:
        word = tl.load(GrammarMask + safe * mask_stride + token // 32, mask=inside, other=0)
        score = tl.where(((word >> (token % 32)) & 1) != 0, score, -float("inf"))
    return token, inside, score


@triton.jit
def _draft_scores(Logits, Histogram, Candidates, Slots, Params, GrammarMask, GrammarEnabled, Scores, BlockMax,
                  hist_stride, mask_stride, candidate_stride, CAPACITY: tl.constexpr, V: tl.constexpr,
                  PREFIX: tl.constexpr, BLOCK: tl.constexpr, BLOCKS: tl.constexpr):
    token, inside, score = _draft_score_values(Logits, Logits, Histogram, Candidates, Slots, Params,
        GrammarMask, GrammarEnabled, hist_stride, mask_stride, candidate_stride, V, V,
        False, CAPACITY, V, PREFIX, BLOCK)
    row, block = tl.program_id(0), tl.program_id(1)
    tl.store(Scores + row * V + token, score, mask=inside)
    score = tl.where(inside, score, -float("inf"))
    maximum = tl.max(score, 0)
    tl.store(BlockMax + row * BLOCKS + block, maximum)


@triton.jit
def _draft_score_candidates(Logits, Bias, Histogram, PrefixIds, Slots, Params, GrammarMask, GrammarEnabled,
                            TopK, Probabilities, LocalCandidates, BlockMax, hist_stride, mask_stride,
                            candidate_stride, logit_stride, bias_stride, ADD_BIAS: tl.constexpr,
                            CAPACITY: tl.constexpr, V: tl.constexpr, PREFIX: tl.constexpr,
                            BLOCK: tl.constexpr, BLOCKS: tl.constexpr, K: tl.constexpr):
    tokens, inside, scores = _draft_score_values(Logits, Bias, Histogram, PrefixIds, Slots, Params,
        GrammarMask, GrammarEnabled, hist_stride, mask_stride, candidate_stride, logit_stride,
        bias_stride, ADD_BIAS, CAPACITY, V, PREFIX, BLOCK)
    row, block = tl.program_id(0), tl.program_id(1)
    # Validity depends on every score, including nonfinite entries outside the selected support.
    tl.store(BlockMax + row * BLOCKS + block, tl.max(tl.where(inside, scores, -float("inf")), 0))
    tl.store(Probabilities + row * V + tokens, tl.zeros((BLOCK,), tl.float32), inside)
    safe = tl.minimum(tl.maximum(tl.load(Slots + row).to(tl.int64), 0), CAPACITY - 1)
    top_k = tl.load(TopK + safe)
    k = tl.minimum(tl.maximum(tl.where(top_k > 0, top_k, V), 1), K)
    ordered = tl.where(inside, _orderable(scores), -2147483648)
    position, take, count = _candidate_positions(ordered, _kth_largest(ordered, k), inside, K)
    keys = (ordered.to(tl.int64) << 32) | (2147483647 - tokens).to(tl.int64)
    base = LocalCandidates + (row * BLOCKS + block) * K
    slots = tl.arange(0, K)
    tl.store(base + slots, tl.full((K,), -9223372036854775808, tl.int64), slots >= count)
    tl.store(base + position, keys, take)


@triton.jit
def _selected_row(Slots, Generations, Generation, live_rows, row, CAPACITY: tl.constexpr, BATCH: tl.constexpr):
    slot = tl.load(Slots + row).to(tl.int64)
    safe = tl.minimum(tl.maximum(slot, 0), CAPACITY - 1)
    selected = (row < live_rows) & (live_rows >= 0) & (live_rows <= BATCH) & (slot >= 0) & (slot < CAPACITY)
    return selected & (tl.load(Generation + safe) == tl.load(Generations + row)), slot, safe


@triton.jit
def _draft_scores_finish(BlockMax, Candidates, Slots, Generations, LiveRows, Generation, Status, TopK,
                         Params, Maximum, Valid, PackedTopK, PackedTopP, candidate_stride,
                         CAPACITY: tl.constexpr, BATCH: tl.constexpr, V: tl.constexpr, PREFIX: tl.constexpr,
                         BLOCKS: tl.constexpr, BLOCK_COUNT: tl.constexpr, RUNNING: tl.constexpr):
    # The row maximum, then the row's liveness (as speculation's _live), validity and filters.
    row = tl.program_id(0)
    block = tl.arange(0, BLOCK_COUNT)
    present = block < BLOCKS
    maxima = tl.load(BlockMax + row * BLOCKS + block, mask=present, other=-float("inf"))
    maximum = tl.max(maxima, 0)
    live_rows = tl.load(LiveRows)
    selected, slot, safe = _selected_row(Slots, Generations, Generation, live_rows, row, CAPACITY, BATCH)
    sharing = 0
    for other in tl.static_range(BATCH):
        other_selected, other_slot, _ = _selected_row(Slots, Generations, Generation, live_rows, other, CAPACITY, BATCH)
        sharing += (other_selected & (other_slot == slot)).to(tl.int32)
    live = selected & (sharing <= 1) & (tl.load(Status + safe) == RUNNING)
    prefix_valid = True
    for index in tl.static_range(PREFIX):
        draft = tl.load(Candidates + row * candidate_stride + index)
        prefix_valid &= (draft >= 0) & (draft < V)
    finite = (maximum == maximum) & (tl.abs(maximum) < float("inf"))
    tl.store(Maximum + row, maximum)
    tl.store(Valid + row, live & prefix_valid & finite)
    top_k = tl.load(TopK + safe)
    tl.store(PackedTopK + row, tl.where(top_k > 0, top_k, V))
    tl.store(PackedTopP + row, tl.load(Params + safe * 6 + 1))


_SCORE_BLOCK = BLOCK


class PreparedDraftProposal:
    """Captured candidate selection over explicit request-policy inputs for one draft position."""

    def __init__(self, scores, probabilities, valid, top_k, top_p, params, slots, seed, offset, sampled,
                 *, capacity, candidates=None, selected=None):
        rows, vocab = scores.shape[0], scores.shape[-1]
        self.blocks = triton.cdiv(vocab, BLOCK)
        self.candidates = (torch.empty((rows, self.blocks, DRAFT_CANDIDATES), dtype=torch.int64, device=scores.device)
                           if candidates is None else candidates)
        self.selected = (torch.empty((rows, DRAFT_CANDIDATES), dtype=torch.int64, device=scores.device)
                         if selected is None else selected)
        for tensor, shape in ((self.candidates, (rows, self.blocks, DRAFT_CANDIDATES)),
                              (self.selected, (rows, DRAFT_CANDIDATES))):
            if (tensor.shape != shape or tensor.dtype != torch.int64
                    or tensor.device != scores.device or not tensor.is_contiguous()):
                raise ValueError("draft proposal scratch requires contiguous Int64 tensors on the score device")
        pool = triton.next_power_of_2(self.blocks * DRAFT_CANDIDATES)
        self._block_args = (scores, probabilities, top_k, self.candidates, vocab, self.blocks, BLOCK, DRAFT_CANDIDATES)
        self._select_args = (self.selected, probabilities, valid, top_k, top_p, params, slots,
                             seed, offset, sampled, vocab)
        self._candidate_source = self.candidates
        self._candidate_geometry = (self.blocks, pool, DRAFT_CANDIDATES, capacity)
        self._shortlist_args = None
        self._grids = ((rows, self.blocks, 1), (rows, 1, 1))
        self._kernels = None
        self._support = None

    def bind_shortlist(self, ids, candidates):
        if self._kernels is not None:
            raise RuntimeError("draft shortlist must be bound before prepare()")
        rows, vocab = self._block_args[0].shape[0], self._block_args[4]
        if (ids.ndim != 2 or ids.shape[0] != rows or not 0 < ids.shape[1] <= DRAFT_CANDIDATES
                or ids.dtype != torch.int64 or ids.device != self.selected.device or ids.stride(-1) != 1):
            raise ValueError("draft shortlist requires Int64 token rows with at most 64 candidates")
        shape = (rows, 4, DRAFT_CANDIDATES)
        if (candidates.shape != shape or candidates.dtype != torch.int64
                or candidates.device != ids.device or not candidates.is_contiguous()):
            raise ValueError("draft shortlist scratch requires contiguous Int64 candidate storage")
        self._candidate_source = candidates
        self._candidate_geometry = (4, 4 * DRAFT_CANDIDATES, DRAFT_CANDIDATES, self._candidate_geometry[-1])
        self._shortlist_args = (self._block_args[0], ids, candidates, ids.stride(0),
                               vocab, ids.shape[1], 4 * DRAFT_CANDIDATES, DRAFT_CANDIDATES)

    def bind_support(self, ids, probabilities):
        """Bind canonical broadcast payloads before graph capture, preserving candidate order."""
        shape = (self.selected.shape[0], DRAFT_CANDIDATES)
        for tensor, dtype in ((ids, torch.int64), (probabilities, torch.float32)):
            if (tensor.shape != shape or tensor.dtype != dtype or tensor.device != self.selected.device
                    or not tensor.is_contiguous()):
                raise ValueError("draft support requires contiguous Int64 IDs and FP32 probabilities")
        self._support = ids, probabilities
        if self._kernels is not None:
            self._kernels = self._kernels[:-1] + (self._prepare_select(),)

    def _prepare_select(self):
        from ignition.preparation.triton import prepare_triton

        support = (self.selected, self._block_args[1]) if self._support is None else self._support
        return prepare_triton(_select_and_sample,
                              (self._candidate_source, *self._select_args, *self._candidate_geometry, *support), self._grids[1],
                              WRITE_SUPPORT=self._support is not None, num_warps=4)

    def prepare(self, *, candidates_prepared=False):
        from ignition.preparation.triton import prepare_triton

        if self._shortlist_args is not None:
            self._kernels = (prepare_triton(_shortlist_candidates, self._shortlist_args,
                                           self._grids[1], num_warps=4), self._prepare_select())
        else:
            self._kernels = (() if candidates_prepared else (
                prepare_triton(_block_candidates, self._block_args, self._grids[0], num_warps=8),
            )) + (self._prepare_select(),)
        self()

    def __call__(self):
        if self._kernels is None:
            raise RuntimeError("draft proposal prepare() must complete before capture")
        if self._shortlist_args is not None:
            self._block_args[1].zero_()
        for kernel in self._kernels:
            kernel()



class PreparedDraftDistribution(PreparedSpeculativeDistributions):
    """A one-position conditional proposal whose filtering and draw use PreparedDraftProposal."""

    @staticmethod
    def broadcast_layout(batch):
        return dict(support_ids=((batch, DRAFT_CANDIDATES), torch.int64),
                    support_probabilities=((batch, DRAFT_CANDIDATES), torch.float32),
                    sampled=((batch,), torch.int64), valid=((batch, 1), torch.bool))

    @staticmethod
    def scratch_layout(batch, width, vocab):
        return {**PreparedSpeculativeDistributions.scratch_layout(batch, width, vocab),
            "block_max": ((batch, triton.cdiv(vocab, _SCORE_BLOCK)), torch.float32),
            "proposal_candidates": ((batch, triton.cdiv(vocab, BLOCK), DRAFT_CANDIDATES), torch.int64),
            "proposal_selected": ((batch, DRAFT_CANDIDATES), torch.int64),
            "shortlist_candidates": ((batch, 4, DRAFT_CANDIDATES), torch.int64)}

    def __init__(self, logits, requests, slots, generations, live_rows, candidate_prefix_ids, *,
                 seed, offset, sampled, prefix_offset=0, workspace=None):
        super().__init__(logits, requests, slots, generations, live_rows, candidate_prefix_ids,
                         prefix_offset=prefix_offset, workspace=workspace)
        del self.vocab_ids
        if logits.shape[1] != 1:
            raise ValueError("draft proposals cover one position")
        # Fused scoring: per-block maxima of each row before the row reduction.
        self._score_kernels = None
        # Contiguous [B,1,...] storage is addressed by row; kernels receive the base tensors.
        self.proposal = PreparedDraftProposal(
            self.scores, self.probabilities, self.valid, self.packed_top_k, self.packed_top_p,
            requests.parameters, slots, seed, offset, sampled, capacity=requests.capacity,
            candidates=self.proposal_candidates, selected=self.proposal_selected)
        self.sampled = sampled
        self.broadcast = None
        self.source_logits, self.source_bias = None, None
        self._candidate_kernels = None
        self._restore_support = None

    def bind_shortlist(self, ids):
        """Bind unique IDs covering every logit other than negative infinity."""
        if self._prepared or self.source_logits is not None:
            raise RuntimeError("draft shortlist must be bound before preparation without bound logits")
        self.proposal.bind_shortlist(ids, self.shortlist_candidates)

    def bind_logits(self, logits, bias=None):
        """Read strided BF16/FP32 head rows and bias with FP32 addition, without staging logits."""
        if self._prepared or self.proposal._shortlist_args is not None:
            raise RuntimeError("draft logits must be bound before preparation without a shortlist")
        shape = (self.logits.shape[0], self.logits.shape[-1])
        if (self.logits.dtype != torch.float32 or logits.shape != shape
                or logits.device != self.logits.device or logits.dtype not in (torch.bfloat16, torch.float32)
                or logits.stride(-1) != 1 or logits.stride(0) < shape[1]):
            raise ValueError("bound draft logits require FP32 staging and BF16/FP32 contiguous vocabulary rows")
        if bias is not None and (bias.shape != shape or bias.dtype not in (torch.bfloat16, torch.float32)
                or bias.device != logits.device or bias.stride(-1) != 1 or bias.stride(0) < shape[1]):
            raise ValueError("draft bias requires BF16/FP32 contiguous vocabulary rows on the logit device")
        self.source_logits, self.source_bias = logits, bias

    def prepare_broadcast(self, *, runtime, session, coordinator, name, buffers=None):
        if runtime is None:
            return
        from .collectives import PreparedBroadcast

        def allocate(shape, dtype):
            return (torch.zeros(shape, dtype=dtype, device=self.logits.device) if buffers is None
                    else buffers.zeros(shape, dtype))

        layout = self.broadcast_layout(self.logits.shape[0])
        self.support_ids = allocate(*layout["support_ids"])
        self.support_probabilities = allocate(*layout["support_probabilities"])
        self.proposal.bind_support(self.support_ids, self.support_probabilities)
        self.broadcast = PreparedBroadcast(
            (self.support_ids, self.support_probabilities, self.sampled, self.valid),
            runtime=runtime, session=session, coordinator=coordinator, name=name, buffers=buffers)
        if self._prepared:
            self._prepare_support_restore()

    def _prepare_support_restore(self):
        from ignition.preparation.triton import prepare_triton

        self._restore_support = prepare_triton(_restore_draft_support,
            (self.proposal_selected, self.support_ids, self.support_probabilities, self.probabilities),
            (self.logits.shape[0], 1, 1), V=self.logits.shape[-1], K=DRAFT_CANDIDATES, num_warps=4)

    def prepare(self):
        if self.logits.device.type != "cuda":
            raise ValueError("draft proposals require CUDA")
        from ignition.components.sampling import RUNNING
        from ignition.preparation.triton import prepare_triton

        state = self.state
        batch, _, vocab = self.logits.shape
        blocks = self.block_max.shape[1]
        self._score_kernels = (
            prepare_triton(_draft_scores, (self.logits, state.histogram, self.candidates, self.slots,
                state.parameters, state.grammar_mask, state.grammar_enabled, self.scores, self.block_max,
                state.histogram.stride(0), state.grammar_mask.stride(0), self.candidates.stride(0)),
                (batch, blocks, 1), CAPACITY=state.capacity, V=vocab, PREFIX=self.prefix_offset, BLOCK=_SCORE_BLOCK,
                BLOCKS=blocks, num_warps=8, enable_fp_fusion=False),
            prepare_triton(_draft_scores_finish, (self.block_max, self.candidates, self.slots,
                self.generations, self.live_rows, state.generation, state.status, state.top_k, state.parameters,
                self.maximum, self.valid, self.packed_top_k,
                self.packed_top_p, self.candidates.stride(0)), (batch, 1, 1), CAPACITY=state.capacity, BATCH=batch,
                V=vocab, PREFIX=self.prefix_offset, BLOCKS=blocks, BLOCK_COUNT=triton.next_power_of_2(blocks),
                RUNNING=RUNNING, num_warps=1))
        if self.source_logits is not None:
            bias = self.source_logits if self.source_bias is None else self.source_bias
            self._candidate_kernels = (
                prepare_triton(_draft_score_candidates, (self.source_logits, bias, state.histogram,
                    self.candidates, self.slots, state.parameters, state.grammar_mask, state.grammar_enabled,
                    state.top_k, self.probabilities, self.proposal_candidates, self.block_max,
                    state.histogram.stride(0), state.grammar_mask.stride(0), self.candidates.stride(0),
                    self.source_logits.stride(0), bias.stride(0)), (batch, blocks, 1),
                    ADD_BIAS=self.source_bias is not None, CAPACITY=state.capacity, V=vocab,
                    PREFIX=self.prefix_offset, BLOCK=BLOCK, BLOCKS=blocks, K=DRAFT_CANDIDATES,
                    num_warps=8, enable_fp_fusion=False), self._score_kernels[1])
        self._prepared = True
        self.prepare_scores()
        if self._candidate_kernels is not None:
            for kernel in self._candidate_kernels:
                kernel()
        self.proposal.prepare(candidates_prepared=self._candidate_kernels is not None)
        if self.broadcast is not None:
            self._prepare_support_restore()

    def prepare_scores(self):
        """One-position scores in two kernels, without materialized penalty counts or a separate maximum."""
        if self.source_logits is not None:
            # Explicit score inspection retains the ordinary materialized-logit contract.
            self.logits[:, 0].copy_(self.source_logits.float())
            if self.source_bias is not None:
                self.logits[:, 0].add_(self.source_bias)
        for kernel in self._score_kernels:
            kernel()

    def __call__(self):
        if not self._prepared:
            raise RuntimeError("draft proposal requires prepare() before capture")
        if self._candidate_kernels is None:
            self.prepare_scores()
        else:
            for kernel in self._candidate_kernels:
                kernel()
        self.proposal()
        if self.broadcast is not None:
            self.broadcast()
            self._restore_support()
        return self.probabilities


class PreparedDraftBlockBroadcast:
    """Canonicalize a replicated proposal chain after its final conditional draw.

    Every rank may execute a different local chain, but only rank zero's tokens,
    exact proposal probabilities and validity survive. No peer-dependent operation
    may consume a local proposal before finish(). Subsequent confidence features
    must be reconstructed from the canonical preceding tokens.
    """

    @staticmethod
    def broadcast_layout(batch, depth):
        return dict(support_ids=((depth, batch, DRAFT_CANDIDATES), torch.int64),
                    support_probabilities=((depth, batch, DRAFT_CANDIDATES), torch.float32),
                    sampled=((depth, batch), torch.int64), valid=((depth, batch), torch.bool))

    def __init__(self, proposals, drafts, probabilities, valid, sampled, *, runtime,
                 session, coordinator, name, buffers=None):
        from .collectives import PreparedBroadcast

        batch, depth = drafts.shape
        if not proposals or len(proposals) != depth or any(p.broadcast is not None for p in proposals):
            raise ValueError("block proposal broadcast requires an unbroadcast complete proposal chain")
        if (probabilities.shape[:2] != (batch, depth) or valid.shape != drafts.shape
                or sampled.shape != (batch,)):
            raise ValueError("block proposal outputs must cover every request and draft position")
        layout = self.broadcast_layout(batch, depth)
        def allocate(shape, dtype):
            return (torch.zeros(shape, dtype=dtype, device=drafts.device) if buffers is None
                    else buffers.zeros(shape, dtype))
        self.support_ids, self.support_probabilities, self.sampled, self.valid = (
            allocate(*field) for field in layout.values())
        self.drafts, self.probabilities, self.output_valid = drafts, probabilities, valid
        self.output_sampled = sampled
        self.positions = torch.arange(depth, device=drafts.device)
        for index, proposal in enumerate(proposals):
            proposal.proposal.bind_support(self.support_ids[index], self.support_probabilities[index])
        self.broadcast = PreparedBroadcast(
            (self.support_ids, self.support_probabilities, self.sampled, self.valid),
            runtime=runtime, session=session, coordinator=coordinator, name=name, buffers=buffers)

    def retain(self, position, proposal):
        self.sampled[position].copy_(proposal.sampled)
        self.valid[position].copy_(proposal.valid[:, 0])

    def finish(self, active, limits):
        self.broadcast()
        self.drafts.copy_(self.sampled.T)
        self.output_sampled.copy_(self.sampled[-1])
        self.output_valid.copy_(self.valid.T & active[:, None]
                                & (self.positions[None, :] < limits[:, None] - 1))
        # Local supports can differ at every position. Clear the complete retained
        # rows before restoring root q; padding repeats token zero with identical bits.
        self.probabilities.zero_()
        self.probabilities.scatter_(2, self.support_ids.permute(1, 0, 2),
                                   self.support_probabilities.permute(1, 0, 2))
