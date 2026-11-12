# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel DeepSeek text stack with lagged hyperconnections."""

from dataclasses import dataclass, fields, is_dataclass, replace

import torch

from ignition.components.collectives import (PreparedAllReduce, PreparedChunkedGatherShards,
    PreparedGatherShards, PreparedReduceScatter)
from ignition.preparation.persistent import persistent_zeros
from ignition.components.embedding import PreparedEmbedding
from ignition.components.expert_overlap import ConcurrentSharedExpert
from ignition.components.ring_commit import PreparedRingCommit
from ignition.components.linear import UnquantizedProjection
from ignition.components.mhc import MHCWeights, MHCScratch, PreparedMHC, PreparedMHCPost
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.moe import RouterScratch, PreparedRouter, PreparedRoutedExperts
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.tuning import register_finalizer

from .attention import AttentionIndices, AttentionScratch, PreparedAttention, PreparedGlobalKV, Rotary, normalize
from .decoder_prefill import PreparedDecoderSuffix
from .engram import EngramScratch, PreparedEngram, PreparedTableRequests


# Row-shard eligibility uses 512-row blocks. This constant does not bound mHC calls;
# callers select bounded calls only by passing max_rows_per_call to PreparedMHC.
MHC_CALL_ROWS = 512


@dataclass(frozen=True)
class RowShard:
    """One rank's contiguous rows of a row-sharded prefill decoder.

    Every rank declares the same shard sizes, so a traced forward keeps one layout and the ranks
    differ only in the values the collectives exchange. These tensors are declaration surfaces:
    they size the prepared reductions, gathers and mHC calls, and `functional()` leaves the
    graph's own copies to the traced forward, where compilation plans them.
    """

    world: int
    rank: int
    rows: int
    start: int
    rows_index: torch.Tensor
    attention_output: torch.Tensor
    mlp_output: torch.Tensor
    residual: torch.Tensor
    pre_mix: torch.Tensor
    attn_mix: torch.Tensor
    attention_hc: MHCScratch
    ffn_hc: MHCScratch
    # Row-sharded attention: the shard's own attention scratch, the full-row KV latent the window
    # writer stores, and the shard's router logits before they are gathered to every row.
    attention: AttentionScratch | None = None
    window_values: torch.Tensor | None = None
    router_logits: torch.Tensor | None = None
    # Every graph's `rows_index` is a view of this table, one segment per shard row count (see
    # `_row_table_layout`). One compiled forward binds every rank, and binding compares storage
    # offsets, so a segment sits at the same offset on every rank and only its contents, this rank's
    # row numbers, differ.
    rows_table: torch.Tensor | None = None

    @classmethod
    def allocate(cls, tokens, world, rank, device, *, candidate_tokens=None, attention=False,
                 experts=384):
        """This rank's shard declarations for a decoder of `tokens` rows, or None when rows do not divide."""
        if world < 2 or tokens % world or tokens // world % MHC_CALL_ROWS:
            return None
        rows, start = tokens // world, rank * (tokens // world)
        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)
        def scratch():
            return MHCScratch.allocate(rows, 5120, device)
        layout = _row_table_layout(rows)
        rows_values = torch.cat([torch.arange(rank * count, (rank + 1) * count, dtype=torch.int64)
                                 for count, _ in layout])
        # The table is snapshot storage: prefill scratch is exported as zeros, and a captured graph
        # reads these row numbers as data. `persistent_zeros` routes to the core-owned arena while a
        # preparation exports, and to ordinary storage on a CPU test.
        rows_table = persistent_zeros(rows_values.shape, rows_values.dtype, device).copy_(rows_values)
        first = layout[-1][1]
        return cls(world, rank, rows, start, rows_table[first:first + rows], zeros(rows, 5120),
                   zeros(rows, 5120), zeros(rows, 4, 5120), zeros(rows, 4, dtype=torch.float32),
                   zeros(rows, 4, dtype=torch.float32), scratch(), scratch(),
                   (AttentionScratch.allocate(tokens, world, device, candidate_tokens=candidate_tokens,
                                              shard_rows=rows) if attention else None),
                   zeros(tokens, 512) if attention else None,
                   zeros(rows, experts, dtype=torch.float32) if attention else None,
                   rows_table)

    def graph(self, tokens):
        """The shard a decoder of `tokens` rows uses, or None when its rows do not divide."""
        rows = tokens // self.world
        if tokens % self.world or rows % MHC_CALL_ROWS or rows > self.rows:
            return None
        # A narrower graph holds fewer rows per rank, so this rank's block begins earlier than it
        # does in the captured layout: the graph reads its row count's segment of the table.
        start = self.rank * rows
        layout = _row_table_layout(self.rows)
        index = [count for count, _ in layout].index(rows)
        first = layout[index][1]
        return replace(self, rows=rows, start=start,
                       rows_index=self.rows_table[first:first + rows],
                       attention_output=self.attention_output[:rows], mlp_output=self.mlp_output[:rows],
                       residual=self.residual[:rows], pre_mix=self.pre_mix[:rows], attn_mix=self.attn_mix[:rows],
                       attention_hc=self.attention_hc.rows(rows), ffn_hc=self.ffn_hc.rows(rows),
                       attention=None if self.attention is None else self.attention.rows(rows, tokens),
                       window_values=None if self.window_values is None else self.window_values[:tokens],
                       router_logits=None if self.router_logits is None else self.router_logits[:rows])


def _row_table_layout(rows):
    """Each shard row count up to `rows` a graph can use, with its segment's first table element.

    The layout depends only on `rows`, which every rank shares, so a segment's storage offset is the
    same on every rank.
    """
    layout, first = [], 0
    for count in range(1, rows + 1):
        if count % MHC_CALL_ROWS == 0:
            layout.append((count, first))
            first += count
    return layout


def row_shard_ready(scratch, shard, runtime):
    """True when a runtime serves every row-sharded reduction and gather over this scratch."""
    return (PreparedReduceScatter.fits(runtime, scratch.attention.output, shard.attention_output)
            and PreparedReduceScatter.fits(runtime, scratch.mlp_partial, shard.mlp_output)
            and PreparedGatherShards.fits(runtime, shard.attention_hc.normalized,
                                          scratch.attention_hc.normalized)
            and PreparedGatherShards.fits(runtime, shard.ffn_hc.normalized, scratch.ffn_hc.normalized)
            and PreparedGatherShards.fits(runtime, shard.pre_mix.view(torch.bfloat16),
                                          scratch.pre_mix.view(torch.bfloat16))
            and PreparedChunkedGatherShards.fits(runtime, shard.residual, scratch.residual))


def attention_shard_ready(scratch, shard, runtime):
    """True when a runtime serves the gathers row-sharded attention adds to the same scratch.

    The KV latent, the router logits and the carried selection are gathered whole. A selected row is
    the widest shard-row operand a shard gathers, so proving it proves the compressor's two input
    projections, which are gathered whole at the same row width and no wider.
    """
    if shard is None or shard.attention is None:
        return False
    return (PreparedGatherShards.fits(runtime, shard.attention.window_values, shard.window_values)
            and PreparedGatherShards.fits(runtime, shard.router_logits.view(torch.bfloat16),
                                          scratch.router_logits.view(torch.bfloat16))
            and PreparedGatherShards.fits(runtime, shard.attention.selected.view(torch.bfloat16),
                                          scratch.attention.selected.view(torch.bfloat16)))


def shard_schedule(config, stop, sharded, aux_layers=()):
    """Per-layer `(shard, defer, reseed)` for a prefill decoder whose shard exists.

    The layers `config.shardable_attention_layers` names keep this rank's rows. An Engram layer
    reads and writes full rows, so it runs full-row, and the layers on either side of it change row
    dimension: a layer that lands on the far side of such a change runs its own post, so the
    full-row residual its consumer reads is complete, and the first sharded layer after a full-row
    run reseeds the shard from the rows it owns.
    """
    shardable = set(config.shardable_attention_layers(stop, aux_layers))
    engram = set(config.text["engram_layer_ids"])
    def sharded_at(index):
        return sharded is not None and index in shardable
    schedule = []
    for index in range(config.layers):
        shard = sharded_at(index)
        # Deferred post hands this layer's four streams to the next layer's attention mHC, which
        # reads them only when both layers hold the same rows.
        defer = (index + 1 < config.layers and index + 1 not in engram
                 and index + 1 not in aux_layers and sharded_at(index + 1) == shard)
        schedule.append((shard, defer, shard and index > 0 and not sharded_at(index - 1)))
    return schedule


@dataclass(frozen=True)
class SelectionMove:
    """One attention layer's move of the selection state it reads into the rows it runs in.

    `selected` and `candidates` name a direction each, because the sparse indices and the candidate
    shortlist are written by different layers and so carry different row representations.
    """

    source: int
    selected: str | None = None
    candidates: str | None = None


def selection_moves(config, schedule, *, attention_shard):
    """Per-layer moves of the selection state an attention-sharded prefill decoder carries.

    Attention reads the sparse indices an index source wrote against that source's latent pages, so
    its consumer must hold that source's selection in the rows it runs in. The decoder carries one
    selection value from layer to layer, so the value holds the rows of the layer that last wrote
    it: a layer that runs in the other representation moves it before its attention reads it. The
    candidate shortlist has its own producer and its own consumers, so it is tracked apart.
    """
    if not attention_shard:
        return (None,) * len(schedule)
    candidate = config.text["candidate_source_layer_id"]
    rows = ["shard" if sharded else "full" for sharded, _, _ in schedule]
    def direction(held, index):
        if held == rows[index]:
            return None
        return "full_to_shard" if held == "full" else "shard_to_full"
    moves, held, shortlist = [], rows[0], None
    for index in range(len(rows)):
        source = config.index_source(index)
        selected = candidates = None
        if source == index:
            # An index source writes its own selection, so it reads no carried one.
            held = rows[index]
        elif source is not None:
            selected, held = direction(held, index), rows[index]
        if index == candidate:
            shortlist = rows[index]
        elif source == index and index > candidate:
            candidates, shortlist = direction(shortlist, index), rows[index]
        moves.append(None if selected is None and candidates is None
                     else SelectionMove(source if source is not None else candidate, selected, candidates))
    return tuple(moves)


class PreparedSelectionRefill:
    """Move the carried selection state into the rows the next attention layer reads.

    A full-to-shard move copies the rows this rank owns out of the replicated full state, and the
    candidate shortlist and its counts move with the same rows. A shard-to-full move gathers every
    rank's selected rows through a BF16 view of the same words. Both run before the layer's selectors
    or attention read the state.
    """

    def __init__(self, move, *, full, shard, rows_index, runtime, session, coordinator, name):
        self.move, self.rows_index, self.gather = move, rows_index, None
        if move.candidates == "shard_to_full":
            # The shard gather moves whole 16-byte rows; a count row is four bytes.
            raise ValueError(f"layer {move.source} reads index source {move.source}'s candidate"
                             " shortlist, which only a shard produced; its counts cannot be gathered"
                             " with the shard-row gather")
        if move.selected == "shard_to_full":
            self.gather = PreparedGatherShards(shard.selected.view(torch.bfloat16),
                full.selected.view(torch.bfloat16), runtime=runtime, session=session,
                coordinator=coordinator, name=name + ".selected")

    def functional(self, *, session):
        if self.gather is not None:
            self.gather.functional(session=session)

    def __call__(self, selection):
        selected, candidates, counts = selection
        if self.move.selected == "full_to_shard":
            selected = selected.index_select(0, self.rows_index)
        elif self.move.selected == "shard_to_full":
            selected = self.gather(selected.view(torch.bfloat16)).view(torch.int32)
        if self.move.candidates == "full_to_shard":
            candidates = None if candidates is None else candidates.index_select(0, self.rows_index)
            counts = None if counts is None else counts.index_select(0, self.rows_index)
        return selected, candidates, counts


@dataclass(frozen=True)
class Scratch:
    """BF16 decoder activations and FP32 target vocabulary logits."""

    logits_dtype = torch.float32

    embedding: torch.Tensor
    residual: torch.Tensor
    engram_residual: torch.Tensor
    engram: EngramScratch
    pre_mix: torch.Tensor
    attn_mix: torch.Tensor
    attention_hc: MHCScratch
    ffn_hc: MHCScratch
    attention: AttentionScratch
    attention_output: torch.Tensor
    shared: MLPScratch
    router_logits: torch.Tensor
    router: RouterScratch
    routed: torch.Tensor
    mlp_partial: torch.Tensor
    mlp_output: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor
    row_shard: RowShard | None = None

    def rows(self, tokens, samples):
        def select(value, count):
            if isinstance(value, torch.Tensor):
                return value[:count]
            if is_dataclass(value):
                return type(value)(*(select(getattr(value, field.name), count) for field in fields(value)))
            raise TypeError("decoder scratch must contain tensors and tensor dataclasses")
        def field_value(name):
            if name == "row_shard":
                return None if self.row_shard is None else self.row_shard.graph(tokens)
            return select(getattr(self, name), samples if name in
                          ("hidden", "local_logits", "logits") else tokens)
        return type(self)(*(field_value(field.name) for field in fields(self)))

    @classmethod
    def allocate(cls, tokens, samples, world, device, *, experts=384, top_k=6,
                 candidate_tokens=None, row_shard=None, attention_shard=False):
        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)
        shard = (None if row_shard is None else RowShard.allocate(tokens, world, row_shard.rank, device,
            candidate_tokens=candidate_tokens, attention=attention_shard, experts=experts))
        residual, engram_residual = zeros(tokens, 4, 5120), zeros(tokens, 4, 5120)
        pre_mix, attn_mix = zeros(tokens, 4, dtype=torch.float32), zeros(tokens, 4, dtype=torch.float32)
        attention_hc = MHCScratch.allocate(tokens, 5120, device)
        ffn_hc = MHCScratch.allocate(tokens, 5120, device)
        attention = AttentionScratch.allocate(tokens, world, device, candidate_tokens=candidate_tokens)
        attention_output, mlp_output = zeros(tokens, 5120), zeros(tokens, 5120)
        _, vocabulary = TensorParallelShard(0, world).padded_partition(129280, alignment=64)
        hidden = zeros(samples, 5120)
        local_logits = zeros(samples, vocabulary, dtype=cls.logits_dtype)
        logits = zeros(samples, 129280, dtype=cls.logits_dtype)
        embedding = zeros(tokens, 5120)
        engram = EngramScratch.allocate(tokens, device, world=world)
        shared = MLPScratch.allocate(tokens, 5120, 2304 // world, device)
        router_logits = zeros(tokens, experts, dtype=torch.float32)
        router = RouterScratch.allocate(tokens, device, experts=experts, top_k=top_k)
        routed, partial = zeros(tokens, 5120), zeros(tokens, 5120)
        return cls(embedding, residual, engram_residual, engram, pre_mix, attn_mix, attention_hc, ffn_hc,
            attention, attention_output, shared, router_logits, router, routed, partial, mlp_output,
            hidden, local_logits, logits, shard)



class DraftScratch(Scratch):
    """Retain BF16 proposal logits independently of FP32 target logits."""

    logits_dtype = torch.bfloat16


def _mhc_weights(weights, kind, norm):
    residual = weights.residual
    return MHCWeights(residual[f"hc_{kind}_fn"], residual[f"hc_{kind}_scale"],
                      residual[f"hc_{kind}_base"], residual[norm + ".weight"])


class PreparedDecoderBoundary:
    """Produce complete global KV, then gather the decoder's bounded replay rows."""

    def __init__(self, index, weights, source, target, packing, pool, rotary, *, carry,
                 session, name, workspace, window_metadata, source_indices):
        self.packing = packing
        incoming = ((source.mlp_output, source.ffn_hc.residual, source.ffn_hc.post, source.ffn_hc.comb)
                    if carry else (source.residual,))
        text = pool.config.text
        self.mhc = PreparedMHC(incoming, _mhc_weights(weights, "attn", "attn_norm"), source.attention_hc,
            session=session, name=name + ".attn_hc", workspace=workspace,
            pre_mix=source.pre_mix, pre_out=source.attn_mix, rms_eps=text["rms_norm_eps"],
            norm_eps=text["rms_norm_eps"], hc_eps=text["hc_eps"], sinkhorn_iters=text["hc_sinkhorn_iters"]).functional()
        self.global_kv = PreparedGlobalKV(index, weights.attention, source.attention_hc.normalized, source.attention,
            source_indices.metadata, pool, rotary, session=session, name=name + ".global_kv",
            indices=source_indices)

    def __call__(self, incoming, pre_mix, metadata, indices, indexer):
        suffix = self.packing.begin(metadata)
        attn_mix = torch.zeros_like(pre_mix)
        residual, post, comb, normalized = self.mhc(*incoming, pre_mix=pre_mix, pre_out=attn_mix)
        self.global_kv(normalized, indexer, metadata, indices.compressed(self.global_kv.ratio))
        return suffix, tuple(suffix.gather(value) for value in (residual, post, comb, normalized)), suffix.gather(attn_mix)


class PreparedLayer:
    def __init__(self, index, weights, tables, scratch, metadata, pool, rotary, *, shard, mode,
                 runtime, coordinator, session, name, workspace, carry, defer_post,
                 boundary=None, replay_start=None, window_staging=None, mla_compute_mode="auto",
                 window_metadata=None, examples=None, row_shard=None, attention_shard=False,
                 selection_move=None, reseed=False):
        s, text = scratch, pool.config.text
        self.scratch, self.carry, self.row_shard = s, carry, row_shard
        self.attention_shard = attention_shard
        # A full-row layer owns the rows since this shard last ran, so the shard is refilled here.
        self.reseed = reseed and row_shard is not None
        self.metadata = metadata
        self.image_mask = (torch.zeros(metadata.token_ids.shape, dtype=torch.bool, device=metadata.token_ids.device)
                           if mode == "prefill" and weights.experts.vision_correction is not None else None)
        self.boundary = boundary
        self.engram = None
        if weights.engram is not None:
            self.engram = PreparedEngram(index, weights.engram, tables, s.residual, s.engram_residual,
                metadata, pool,
                scratch=s.engram, session=session, name=name + ".engram", workspace=workspace,
                runtime=runtime, coordinator=coordinator, examples=examples)
        # A row shard reduces into this rank's rows and normalizes them there; the full-size scratch
        # holds the branch input every consumer reads, which a gather fills before that consumer runs.
        attention_scratch = s.attention_hc if row_shard is None else row_shard.attention_hc
        ffn_scratch = s.ffn_hc if row_shard is None else row_shard.ffn_hc
        pre_mix = s.pre_mix if row_shard is None else row_shard.pre_mix
        attn_mix = s.attn_mix if row_shard is None else row_shard.attn_mix
        residual = s.residual if row_shard is None else row_shard.residual
        attention_output = s.attention_output if row_shard is None else row_shard.attention_output
        mlp_output = s.mlp_output if row_shard is None else row_shard.mlp_output
        source = s.engram_residual if self.engram is not None else residual
        options = dict(rms_eps=text["rms_norm_eps"], norm_eps=text["rms_norm_eps"], hc_eps=text["hc_eps"],
                       sinkhorn_iters=text["hc_sinkhorn_iters"])
        incoming = ((mlp_output, ffn_scratch.residual, ffn_scratch.post, ffn_scratch.comb) if carry else (source,))
        self.attention_hc = None if boundary is not None else PreparedMHC(
            incoming, _mhc_weights(weights, "attn", "attn_norm"), attention_scratch,
            session=session, name=name + ".attn_hc", workspace=workspace, pre_mix=pre_mix, pre_out=attn_mix, **options)
        # Row-sharded attention reads and writes only this rank's rows, computes every head and
        # output group, and therefore needs neither the branch-input gather nor a reduction.
        attention_source = attention_scratch.normalized if attention_shard else s.attention_hc.normalized
        attention_buffer = row_shard.attention if attention_shard else s.attention
        self.attention = PreparedAttention(index, weights.attention, attention_source, attention_buffer,
            metadata, pool, rotary, mode=mode, world=shard.world_size,
            session=session, name=name + ".attention", workspace=workspace, replay_start=replay_start,
            window_staging=window_staging, mla_compute_mode=mla_compute_mode,
            window_metadata=window_metadata, examples=examples, runtime=runtime, coordinator=coordinator,
            row_shard=row_shard if attention_shard else None,
            window_target=row_shard.window_values if attention_shard else None)
        # A layer on one side of a row transition reads a selection the other side wrote, so its
        # forward moves that state into the rows its attention reads before it runs.
        self.selection_move = None
        if selection_move is not None:
            # The layer that crosses back to full rows runs full-row, so it holds no shard of its
            # own and reads the carried selection through the decoder's shard instead.
            carried = row_shard if row_shard is not None else s.row_shard
            if carried is None or carried.attention is None:
                raise ValueError("a selection move requires the row shard's attention scratch")
            self.selection_move = PreparedSelectionRefill(selection_move, full=s.attention,
                shard=carried.attention, rows_index=carried.rows_index, runtime=runtime,
                session=session, coordinator=coordinator, name=name + ".selection")
        self.attention_gather = None if row_shard is None or attention_shard else PreparedGatherShards(
            attention_scratch.normalized, s.attention_hc.normalized, runtime=runtime, coordinator=coordinator,
            session=session, name=name + ".attention_gather")
        if attention_shard:
            self.attention_reduce = None
        elif row_shard is None:
            self.attention_reduce = PreparedAllReduce(s.attention.output, s.attention_output, runtime=runtime,
                coordinator=coordinator, session=session, name=name + ".attention_reduce")
        else:
            self.attention_reduce = PreparedReduceScatter(s.attention.output, attention_output, runtime=runtime,
                coordinator=coordinator, session=session, name=name + ".attention_reduce")
        self.ffn_hc = PreparedMHC((attention_output, attention_scratch.residual, attention_scratch.post,
            attention_scratch.comb), _mhc_weights(weights, "ffn", "ffn_norm"), ffn_scratch, session=session,
            name=name + ".ffn_hc", workspace=workspace, pre_mix=attn_mix, pre_out=pre_mix, **options)
        self.ffn_gather = None if row_shard is None else PreparedGatherShards(
            ffn_scratch.normalized, s.ffn_hc.normalized, runtime=runtime, coordinator=coordinator,
            session=session, name=name + ".ffn_gather")
        # Top-k is row-local, so a shard projects the logits it owns and one gather hands the
        # router every row. The routed experts keep all rows.
        router_source = ffn_scratch.normalized if attention_shard else s.ffn_hc.normalized
        router_logits = s.router_logits if not attention_shard else row_shard.router_logits
        self.router_projection = UnquantizedProjection(router_source, weights.experts.router, router_logits,
            session=session, name=name + ".router")
        self.router_gather = None if not attention_shard else PreparedGatherShards(
            router_logits.view(torch.bfloat16), s.router_logits.view(torch.bfloat16), runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".router_gather")
        self.router = PreparedRouter(s.router_logits, s.router, session=session, name=name + ".topk",
            score_func=text["scoring_func"], renormalize=text["norm_topk_prob"],
            routed_scaling_factor=text["routed_scaling_factor"], correction_bias=weights.experts.correction,
            image_correction_bias=weights.experts.vision_correction if self.image_mask is not None else None,
            image_mask=self.image_mask)
        self.experts = PreparedRoutedExperts(s.ffn_hc.normalized, weights.experts.experts, s.router.ids,
            s.router.weights, s.routed, session=session, name=name + ".experts", workspace=workspace,
            router_plan=self.router.plan)
        self.shared = PreparedMLP(s.ffn_hc.normalized, weights.experts.shared, s.shared.gate_up,
            s.shared.intermediate, s.shared.output, session=session, name=name + ".shared", workspace=workspace,
            swiglu_limit=text["swiglu_limit"], round_silu=False, activation_mode="auto")
        self.concurrent_shared = None
        if mode in ("decode", "verify", "draft") and s.ffn_hc.normalized.shape[0] <= 128 and workspace.overlap:
            self.concurrent_shared = ConcurrentSharedExpert(self.shared,
                parallel_plans=(self.router_projection.plan, self.router.plan, self.experts.plan),
                session=session, workspace=workspace,
                defer_launch=s.ffn_hc.normalized.shape[0] > 4)
        if row_shard is None:
            self.mlp_reduce = PreparedAllReduce(s.mlp_partial, s.mlp_output, runtime=runtime,
                coordinator=coordinator, session=session, name=name + ".mlp_reduce")
        else:
            self.mlp_reduce = PreparedReduceScatter(s.mlp_partial, mlp_output, runtime=runtime,
                coordinator=coordinator, session=session, name=name + ".mlp_reduce")
        self.post = None if defer_post else PreparedMHCPost(mlp_output, ffn_scratch.residual,
            ffn_scratch.post, ffn_scratch.comb, residual, session=session, name=name + ".post")
        # A deferred post runs in the next layer's attention mHC over the same rows, so only a layer
        # that runs its own post widens this rank's four-stream residual back to every row.
        self.residual_gather = self.mix_gather = None
        if row_shard is not None and not defer_post:
            self.residual_gather = PreparedChunkedGatherShards(row_shard.residual, s.residual, runtime=runtime,
                session=session, coordinator=coordinator, name=name + ".residual_gather")
            # A BF16 view gathers the FP32 lagged mix through the same prepared surface.
            self.mix_gather = PreparedGatherShards(row_shard.pre_mix.view(torch.bfloat16),
                s.pre_mix.view(torch.bfloat16), runtime=runtime, coordinator=coordinator,
                session=session, name=name + ".mix_gather")

        if self.attention_hc is not None:
            self.attention_hc.functional()
        if self.selection_move is not None:
            self.selection_move.functional(session=session)
        if self.attention_gather is not None:
            self.attention_gather.functional(session=session)
        if self.attention_reduce is not None:
            self.attention_reduce.functional(session=session)
        self.ffn_hc.functional()
        if self.ffn_gather is not None:
            self.ffn_gather.functional(session=session)
        self.router_projection.functional(session=session)
        if self.router_gather is not None:
            self.router_gather.functional(session=session)
        self.router.functional(session=session)
        self.experts.functional(session=session)
        if self.concurrent_shared is None:
            self.shared.functional(session=session)
        else:
            self.concurrent_shared.functional(session=session)
        self.mlp_reduce.functional(session=session)
        if self.post is not None:
            self.post.functional()
        if self.residual_gather is not None:
            self.residual_gather.functional(session=session)
            self.mix_gather.functional(session=session)
        self.has_image_mask = self.image_mask is not None
        register_finalizer(session, self.release_examples)

    def release_examples(self):
        self.scratch = self.metadata = self.image_mask = None
        if self.row_shard is not None:
            self.row_shard = replace(self.row_shard, attention_output=None, mlp_output=None,
                residual=None, pre_mix=None, attn_mix=None, attention_hc=None, ffn_hc=None,
                attention=None, window_values=None, router_logits=None)
        if self.concurrent_shared is not None:
            self.shared = None

    def __call__(self, incoming, pre_mix, metadata, indices, selection, *, boundary=None, window_cache=None):
        incoming, next_mix, records, selection, _ = self._run(incoming, pre_mix, metadata, indices,
            selection, boundary=boundary, window_cache=window_cache)
        return incoming, next_mix, records, selection

    def _run(self, incoming, pre_mix, metadata, indices, selection, *, boundary=None, window_cache=None):
        shard = self.row_shard
        if self.reseed:
            # The full-row layers since this shard last ran own every row, so take this rank's rows
            # out of them through the row numbers the graph reads.
            incoming = tuple(value.index_select(0, shard.rows_index) for value in incoming)
            pre_mix = pre_mix.index_select(0, shard.rows_index)
        if self.engram is not None:
            incoming = (self.engram(incoming[0], metadata),)
        if boundary is None:
            attn_mix = torch.zeros_like(pre_mix)
            residual, post, comb, normalized = self.attention_hc(*incoming, pre_mix=pre_mix, pre_out=attn_mix)
        else:
            (residual, post, comb, normalized), attn_mix = boundary
        if self.attention_gather is not None:
            normalized = self.attention_gather(normalized)
        attention_normalized = normalized
        if self.selection_move is not None:
            selection = self.selection_move(selection)
        attended, records, selection, observations = self.attention._run(normalized, metadata, selection,
            indices=indices, window_cache=window_cache)
        # A shard's own rows are complete once its output projection has summed all groups.
        attention_output = attended if self.attention_reduce is None else self.attention_reduce(attended)
        next_mix = torch.zeros_like(attn_mix)
        residual, post, comb, normalized = self.ffn_hc(attention_output, residual, post, comb,
            pre_mix=attn_mix, pre_out=next_mix)
        router_source = normalized
        if self.ffn_gather is not None:
            normalized = self.ffn_gather(normalized)
        pending = None if self.concurrent_shared is None else self.concurrent_shared.start(normalized)
        image_mask = metadata.token_ids >= 129280 if self.has_image_mask else None
        router_logits = self.router_projection(router_source if self.attention_shard else normalized)
        if self.router_gather is not None:
            # The gather's surface moves BF16 packs of the FP32 logits it was prepared over.
            router_logits = self.router_gather(router_logits.view(torch.bfloat16)).view(torch.float32)
        route = self.router(router_logits, image_mask=image_mask)
        routed = self.experts(normalized, route.ids, route.weights, router_logits=router_logits,
                              selected_logits=route.selected_logits, image_mask=image_mask)
        shared = (self.shared(normalized) if self.concurrent_shared is None
                  else self.concurrent_shared.finish(pending, routed))
        output = self.mlp_reduce(routed + shared)
        incoming = ((self.post(output, residual, post, comb),) if self.post is not None
                    else (output, residual, post, comb))
        if self.residual_gather is not None:
            # Every row's consumer follows, so widen this rank's rows back out: the four-stream
            # residual and the lagged mix the next layer's mHC reads.
            residual, = incoming
            full_mix = torch.empty((metadata.token_ids.numel(), 4), dtype=next_mix.dtype,
                                   device=next_mix.device)
            self.mix_gather(next_mix.view(torch.bfloat16), full_mix.view(torch.bfloat16))
            incoming, next_mix = (self.residual_gather(residual),), full_mix
        observations = {"attention." + key: value for key, value in observations.items()}
        observations.update({"attention_hc.normalized": attention_normalized, "router.ids": route.ids,
                             "output": incoming[0]})
        return incoming, next_mix, records, selection, observations


class PreparedDecoder:
    def __init__(self, weights, tables, scratch, metadata, pool, *, mode, runtime,
                 vocabulary_runtime, coordinator, session, name, workspace,
                 vision=None, mla_compute_mode="auto", scoring=None, vocabulary_parallel=False, examples=None,
                 row_shard_attention=False):
        if scratch.local_logits.dtype != torch.float32 or scratch.logits.dtype != torch.float32:
            raise ValueError("DeepSeek target vocabulary logits require FP32 storage")
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        self.mode = mode
        self.vision = vision if mode == "prefill" else None
        self.table_requests = PreparedTableRequests(tables, metadata, pool, session=session, name=name + ".table_requests")
        decoder_start = weights.config.decoder_start
        self.packing = None
        if mode == "prefill" and metadata.positions.numel() > 128 and decoder_start is not None and scoring is None:
            if (any(layer.engram is not None for layer in weights.layers[decoder_start:])
                    or any(source.layer > decoder_start for source in weights.config.sources)):
                raise ValueError("decoder suffix replay requires Engram and compressed KV sources before its boundary")
            self.packing = PreparedDecoderSuffix(metadata)
        self.output_metadata = metadata if self.packing is None else self.packing.metadata
        self.output_scratch = scratch if self.packing is None else Scratch.allocate(
            128, scratch.hidden.shape[0], weights.shard.world_size, scratch.embedding.device)
        self.aux_layers = () if weights.draft is None else weights.config.draft.target_layers
        if self.packing is not None and any(layer <= decoder_start for layer in self.aux_layers):
            raise ValueError("bounded decoder feedback requires draft target layers after its global-KV boundary")
        # Layers below the prefill's stop run row-sharded; the layer at `stop - 1` materializes
        # every row again so the decoder suffix keeps its full-row boundary and replay gather.
        stop = weights.config.prefill_stop(self.packing is None)
        self.row_shard = scratch.row_shard
        if (self.row_shard is None or mode != "prefill" or stop < 1
                or not row_shard_ready(scratch, self.row_shard, runtime)):
            self.row_shard = None
        # Attention joins the shard only when the bootstrap asked and its gathers are served too.
        self.attention_shard = (self.row_shard is not None and mode == "prefill" and row_shard_attention
                                and attention_shard_ready(scratch, self.row_shard, runtime))
        schedule = shard_schedule(weights.config, stop, self.row_shard, self.aux_layers)
        moves = selection_moves(weights.config, schedule, attention_shard=self.attention_shard)
        self.auxiliary_shape = (self.output_metadata.token_ids.numel(), len(self.aux_layers) * 5120)
        self.auxiliary = torch.zeros(self.auxiliary_shape, dtype=torch.bfloat16, device=metadata.positions.device)
        self.safe_tokens = torch.zeros_like(metadata.token_ids) if self.vision is not None else metadata.token_ids
        self.embedding = PreparedEmbedding(weights.embedding, self.safe_tokens, scratch.embedding,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding").functional(session=session, dynamic_inputs=True)
        self.rotary = Rotary(weights.config, scratch.embedding.device)
        self.window_commit = None
        if mode == "verify":
            self.window_commit = PreparedRingCommit(pool.reserved_backing,
                [pool.layout.region(f"layer{index}.window").offset for index in range(len(weights.layers))],
                metadata.state_rows, metadata.starts,
                token_capacity=metadata.max_query_tokens,
                window_tokens=128, record_bytes=528)
        ratios = weights.config.text["compress_ratios"][:len(weights.layers)]
        prefix_ratios = ratios if self.packing is None else ratios[:decoder_start + 1]
        self.window_metadata = AttentionIndices(metadata, mode=mode,
            tokens=metadata.positions.numel(),
            page_size=128 if mode == "decode" else 128 + metadata.max_query_tokens, width=128,
            ratios=set(prefix_ratios) - {0})
        self.suffix_window_metadata = None
        if self.packing is not None:
            self.suffix_window_metadata = AttentionIndices(self.output_metadata, mode=mode,
                tokens=self.output_metadata.positions.numel(),
                page_size=128 + self.output_metadata.max_query_tokens, width=128,
                replay_start=self.packing.replay_start, ratios=set(ratios[decoder_start:]) - {0})
        self.layers, carry = [], False
        for index, layer in enumerate(weights.layers):
            boundary = replay_start = None
            layer_scratch, layer_metadata = scratch, metadata
            if self.packing is not None and index >= decoder_start:
                layer_scratch, layer_metadata = self.output_scratch, self.output_metadata
                replay_start = self.packing.replay_start
                if index == decoder_start:
                    boundary = PreparedDecoderBoundary(index, layer, scratch, layer_scratch, self.packing,
                        pool, self.rotary, carry=carry, session=session, name=f"{name}.layer{index}", workspace=workspace,
                        window_metadata=self.suffix_window_metadata,
                        source_indices=self.window_metadata)
                    layer = replace(layer, attention={key: value for key, value in layer.attention.items()
                                                      if key not in ("compressor", "index_k", "index_norm")})
            sharded, defer, reseed = schedule[index]
            sharded = self.row_shard if sharded else None
            self.layers.append(PreparedLayer(index, layer, tables, layer_scratch, layer_metadata, pool, self.rotary,
                shard=weights.shard, mode=mode, runtime=runtime,
                coordinator=coordinator, session=session, name=f"{name}.layer{index}", workspace=workspace,
                carry=carry, defer_post=defer, boundary=boundary, replay_start=replay_start,
                window_staging=None if self.window_commit is None else self.window_commit.staging[index],
                mla_compute_mode=mla_compute_mode,
                window_metadata=(self.window_metadata if layer_metadata is metadata else self.suffix_window_metadata),
                examples=examples, row_shard=sharded, attention_shard=self.attention_shard and sharded is not None,
                selection_move=moves[index], reseed=reseed))
            carry = defer
        self.scoring = None if scoring is None else scoring.prepare(scratch.embedding, weights.head,
            vocabulary=129280, session=session, coordinator=coordinator, name=name + ".prompt",
            logits_dtype=torch.float32)
        self._prepare_head(vocabulary_parallel=vocabulary_parallel, vocabulary_runtime=vocabulary_runtime,
            coordinator=coordinator, session=session, name=name, workspace=workspace)
        self.window_metadata.functional()
        if self.suffix_window_metadata is not None:
            self.suffix_window_metadata.functional()
            self.packing.functional()
        self.staging_shape = None if self.window_commit is None else tuple(self.window_commit.staging.shape)
        if self.window_commit is not None:
            self.window_commit.functional()
        register_finalizer(session, self.release_outputs)

    def _prepare_head(self, *, vocabulary_parallel, vocabulary_runtime, coordinator, session, name, workspace=None):
        scratch, weights = self.scratch, self.weights
        self.compact_head = scratch.hidden.shape[0] > self.output_metadata.token_ids.numel()
        head_rows = self.output_metadata.token_ids.numel() if self.compact_head else scratch.hidden.shape[0]
        self.head_hidden = scratch.hidden[:head_rows]
        head_output = scratch.logits
        if vocabulary_parallel and self.compact_head:
            # Declare expanded local logits independently of the physical projection rows.
            head_output = scratch.logits.view(-1)[:scratch.local_logits.numel()].view_as(scratch.local_logits)
        self.head = PreparedVocabularyProjection(self.head_hidden, weights.head, scratch.local_logits[:head_rows],
            head_output, runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head",
            row_indices=self.output_metadata.sample_indices if self.compact_head else None, gather=not vocabulary_parallel,
            workspace=workspace)

        self.local_logits = self.head.output if vocabulary_parallel else scratch.local_logits
        self.head.functional(session=session, dynamic_indices=True)

    def release_outputs(self):
        self.scratch = self.metadata = self.output_scratch = self.output_metadata = None
        if self.row_shard is not None:
            self.row_shard = replace(self.row_shard, attention_output=None, mlp_output=None,
                residual=None, pre_mix=None, attn_mix=None, attention_hc=None, ffn_hc=None,
                attention=None, window_values=None, router_logits=None)
        self.auxiliary = self.safe_tokens = self.head_hidden = self.local_logits = None

    def __call__(self, metadata):
        source_metadata = metadata
        raw_history = self.table_requests(metadata)
        tokens = (metadata.token_ids.clamp(0, self.weights.embedding.shape[0] - 1)
                  if self.vision is not None else metadata.token_ids)
        embedding = self.embedding(token_ids=tokens, num_rows=metadata.live_counts[:1])
        if self.vision is not None:
            self.vision.inject(metadata, embedding)
        residual = embedding[:, None, :].expand(-1, 4, -1).contiguous()
        pre_mix = torch.zeros((embedding.shape[0], 4), dtype=torch.float32, device=embedding.device)
        pre_mix[:, 0].fill_(1)
        incoming = (residual,)
        if self.row_shard is not None:
            # The first layer keeps this rank's rows, so enter with them.
            incoming = (residual.index_select(0, self.row_shard.rows_index),)
            pre_mix = pre_mix.index_select(0, self.row_shard.rows_index)
        indices = self.window_metadata(metadata)
        # The carried selection holds this rank's rows from the first layer on; every later change
        # of row representation moves it in the layer that crosses over.
        selection_rows = self.row_shard.rows if self.attention_shard else embedding.shape[0]
        selection = (torch.full((selection_rows, 512), -1, dtype=torch.int32, device=embedding.device), None, None)
        staging = (None if self.staging_shape is None else torch.zeros(
            self.staging_shape, dtype=torch.uint8, device=embedding.device))
        auxiliary = torch.zeros(self.auxiliary_shape, dtype=torch.bfloat16, device=embedding.device)
        suffix, records = None, []
        for index, layer in enumerate(self.layers):
            boundary = None
            if layer.boundary is not None:
                suffix, state, attn_mix = layer.boundary(incoming, pre_mix, metadata, indices,
                                                        layer.attention.selectors[0])
                metadata = suffix.metadata
                indices = self.suffix_window_metadata(metadata, replay_start=suffix.replay_start)
                boundary = state, attn_mix
                selection = (torch.full((metadata.positions.numel(), 512), -1,
                                       dtype=torch.int32, device=embedding.device), None, None)
            if index in self.aux_layers:
                feedback = incoming[0].float().mean(1)
                if suffix is not None:
                    feedback = torch.where(suffix.live[:, None], feedback, 0)
                column = self.aux_layers.index(index)
                auxiliary[:, column * 5120:(column + 1) * 5120].copy_(feedback)
            incoming, pre_mix, record, selection = layer(incoming, pre_mix, metadata, indices, selection,
                boundary=boundary, window_cache=None if staging is None else staging[index])
            records.append(record)
        residual, = incoming
        samples = metadata.sample_indices
        head_residual = residual if self.compact_head else residual[samples]
        head_mix = pre_mix if self.compact_head else pre_mix[samples]
        collapsed = (head_residual.float() * head_mix[:, :, None]).sum(1).bfloat16()
        if self.scoring is not None:
            collapsed = (residual.float() * pre_mix[:, :, None]).sum(1).bfloat16()
            normalized = torch.zeros_like(collapsed)
            normalize(collapsed, self.weights.norm, normalized, self.weights.config.text["rms_norm_eps"])
            self.scoring(normalized)
            hidden = normalized[samples]
        else:
            hidden = torch.zeros_like(collapsed)
            normalize(collapsed, self.weights.norm, hidden, self.weights.config.text["rms_norm_eps"])
        logits = self.head(hidden, row_indices=samples if self.compact_head else None)
        if self.compact_head:
            hidden = hidden[samples]
        if self.mode != "verify":
            self.table_requests.commit(source_metadata.lengths, raw_history, source_metadata)
        return hidden, logits, auxiliary, (staging, tuple(records), raw_history)

    def commit(self, counts, records, metadata):
        if self.window_commit is None:
            raise ValueError("accepted decoder commits require verification staging")
        staging, layers, raw_history = records
        self.window_commit(counts, staging=staging, rows=metadata.state_rows, starts=metadata.starts)
        for layer, record in zip(self.layers, layers, strict=True):
            layer.attention.commit_pending(counts, record, metadata)
        self.table_requests.commit(counts, raw_history, metadata)
