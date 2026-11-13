# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Three causal MTP predictors with accepted-prefix feedback and private proposal KV."""

import torch

from ignition.components.collectives import PreparedAllReduceRMSNorm
from ignition.components.embedding import PreparedShardedEmbedding
from ignition.components.linear import UnquantizedProjection
from ignition.components.paged_batch import Metadata
from ignition.components.rms_norm import PreparedPairedRMSNorm, PreparedRMSNorm
from ignition.components.state import gather_rows, save_hidden
from ignition.components.vocabulary import PreparedVocabularyProjection

from .attention import PreparedAttention
from .decoder import PreparedDecoderLayer, PreparedFeedForward, Scratch


class PreparedPredictor:
    def __init__(self, index, weights, scratch, metadata, pool, output, *, mode, runtime, embedding_runtime,
                 vocabulary_runtime, coordinator, session, workspace, name, proposal=None, logits=None, vision=None,
                 embedded=None, cache_only=False, buffers=None, joined=None):
        self.metadata, self.output, self.vision = metadata, output, vision
        self.external_embedding, self.cache_only, self.scratch = embedded, cache_only, scratch
        w, c = weights.draft[index], weights.config
        self.embedding = None if embedded is not None else PreparedShardedEmbedding(weights.embedding, metadata.token_ids,
            scratch.local_embedding, scratch.embedding, runtime=embedding_runtime, coordinator=coordinator,
            session=session, name=name + ".embedding")
        def zeros(shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=output.device) if buffers is None else buffers.zeros(shape, dtype)
        self.joined = zeros((output.shape[0], 2 * c.hidden)) if joined is None else joined
        self.join = PreparedPairedRMSNorm(output, output, (w["enorm"], w["hnorm"]), self.joined,
            eps=c.text["rms_norm_eps"], session=session, name=name + ".norm_concat", zero_centered=True)
        self.project = UnquantizedProjection(self.joined, w["projection"], scratch.embedding,
                                              session=session, name=name + ".eh")
        layer = c.layers + index
        common = {"runtime": runtime, "coordinator": coordinator, "session": session, "workspace": workspace}
        fused = PreparedAllReduceRMSNorm.fits(runtime, scratch.partial)
        attention = PreparedAttention(layer, w["layer"], c, weights.shard, scratch.normalized, scratch.attention,
            metadata, pool, mode=mode, name=name + ".attention", first_position=index + 1, proposal=proposal,
            reduce_output=not fused, buffers=buffers, **common)
        # FP8 activation rounding compounds through the predictor feedback chain.
        mlp = PreparedFeedForward(layer, w["layer"], c, scratch, name=name + ".mlp", overlap=False,
                                 activation_mode="a16", **common)
        self.layer = PreparedDecoderLayer(w["layer"], attention, mlp, scratch, c.text["rms_norm_eps"],
                                         session=session, name=name + ".layer", runtime=runtime, coordinator=coordinator)
        self.head = None
        if logits is not None:
            self.sampled = zeros((logits.shape[0], c.hidden), output.dtype)
            self.normalized = zeros(self.sampled.shape, self.sampled.dtype)
            self.norm = PreparedRMSNorm(self.sampled, w["norm"], self.normalized, eps=c.text["rms_norm_eps"],
                session=session, name=name + ".norm", zero_centered=True)
            self.local_logits = zeros((logits.shape[0], w["head"].shape[0]), logits.dtype)
            self.head = PreparedVocabularyProjection(self.normalized, w["head"], self.local_logits, logits,
                runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head", buffers=buffers)

    def __call__(self, previous):
        embedded = self.external_embedding
        if embedded is None:
            local = self.embedding.lookup()
            if self.vision is not None:
                self.vision.inject(self.metadata, local)
            embedded = self.embedding.gather(local)
        self.join(embedded, previous)
        residual = self.scratch.residual
        residual.copy_(self.project(self.joined))
        if self.cache_only:
            self.layer.attention(self.layer.input_norm(residual), cache_only=True)
            return None
        residual.add_(self.layer(residual))
        self.output.copy_(torch.where((self.metadata.positions >= 0)[:, None], residual, 0))
        if self.head is not None:
            self.sampled.copy_(self.output[self.metadata.sample_indices])
            self.norm(self.sampled)
            self.head(self.normalized)
        return self.output

    def scratch_tensors(self):
        return (self.joined, self.output, self.layer.attention.scratch_tensors(),
                () if self.head is None else (self.sampled, self.normalized, self.local_logits, self.head.gathered))


class PreparedFeedback:
    """Teacher-force all predictors using each preceding level's shifted hidden states."""

    def __init__(self, weights, auxiliary, metadata, pool, *, scratch, runtime, embedding_runtime,
                 vocabulary_runtime, coordinator, session, name, workspace, vision=None, buffers=None):
        self.source, self.pool = metadata, pool
        self.flush = PreparedProposals(weights, pool, scratch, metadata.starts.numel(), 0,
            runtime=runtime, embedding_runtime=embedding_runtime, vocabulary_runtime=vocabulary_runtime,
            coordinator=coordinator, session=session, workspace=workspace, name=name + ".pending", buffers=buffers)
        self.metadata = m = Metadata.allocate(auxiliary.shape[0], metadata.starts.numel(), metadata.page_table.shape[1],
                                              auxiliary.device, samples=auxiliary.shape[0], buffers=buffers)
        def zeros(shape):
            return (torch.zeros(shape, dtype=torch.bfloat16, device=auxiliary.device) if buffers is None
                    else buffers.zeros(shape, torch.bfloat16))
        self.previous, self.output = zeros(auxiliary.shape), zeros(auxiliary.shape)
        joined = zeros((auxiliary.shape[0], 2 * weights.config.hidden))
        self.indices = torch.arange(auxiliary.shape[0], device=auxiliary.device)
        self.rows = torch.arange(metadata.starts.numel(), device=auxiliary.device)
        self.histories = (pool.reserved_region("draft.target_hidden"),
                          *(pool.reserved_region(f"draft{i}.hidden") for i in range(len(weights.draft) - 1)))
        self.saved = tuple(zeros((metadata.starts.numel(), weights.config.hidden)) for _ in weights.draft)
        self.predictors = tuple(PreparedPredictor(i, weights, scratch, m, pool, self.output,
            mode="prefill" if metadata.starts.numel() == 1 and auxiliary.shape[0] > 1 else "decode",
            runtime=runtime, embedding_runtime=embedding_runtime, vocabulary_runtime=vocabulary_runtime,
            coordinator=coordinator, session=session, workspace=workspace, name=f"{name}.{i}", vision=vision, cache_only=i == len(weights.draft) - 1,
            buffers=buffers, joined=joined)
            for i in range(len(weights.draft)))

    def __call__(self, auxiliary, counts):
        source, m = self.source, self.metadata
        active = counts > 0
        self.flush.begin(torch.where(active, source.state_rows, 0))
        self.flush.finish(source.starts, torch.where(active, source.state_rows, 0), active)
        for field in ("token_ids", "requests", "state_rows", "starts", "query_start_loc", "live_counts"):
            getattr(m, field).copy_(getattr(source, field))
        m.lengths.copy_(counts)
        offset = self.indices - source.query_start_loc[source.requests]
        active = (source.positions >= 0) & (offset < counts[source.requests])
        last = (source.query_start_loc[:-1] + counts - 1).clamp(0, self.indices.numel() - 1).long()
        for saved, history in zip(self.saved, self.histories, strict=False):
            gather_rows(history, m.state_rows, saved)
        previous = auxiliary
        for index, predictor in enumerate(self.predictors):
            m.positions.copy_(torch.where(active & (source.positions >= index + 1), source.positions, -1))
            self.previous.copy_(torch.where((offset == 0)[:, None], self.saved[index][m.requests],
                                           previous[(self.indices - 1).clamp_min(0)]))
            previous = predictor(self.previous)
            valid = (counts > 0) & (source.starts + counts > index + 1)
            if index + 1 < len(self.histories):
                save_hidden(previous[last], self.histories[index + 1], m.state_rows, valid)
        save_hidden(auxiliary[last], self.histories[0], m.state_rows, counts > 0)

    def scratch_tensors(self):
        return (self.metadata, self.previous, self.output, self.saved, self.flush.scratch_tensors(),
                *(predictor.scratch_tensors() for predictor in self.predictors))


class PreparedProposals:
    """Combine accepted teacher forcing and proposals in one packed pass per predictor.

    Continuations retain the accepted target rows not yet consumed by the draft.
    Each pass commits only those rows; candidate KV remains private until target
    verification supplies the accepted target states for the following round.
    """

    def __init__(self, weights, pool, scratch, batch, depth, *, runtime, embedding_runtime,
                 vocabulary_runtime, coordinator, session, workspace, name, buffers=None):
        self.batch, self.depth, self.pool = batch, depth, pool
        c, device = weights.config, pool.backing.device
        self.pending_width = len(weights.draft) + 1
        self.pitch = self.pending_width + depth
        tokens = batch * self.pitch
        if scratch.embedding.shape[0] < tokens:
            scratch = Scratch.allocate(c, weights.shard, tokens, tokens, 0, device, buffers=buffers)
        else:
            scratch = scratch.rows(tokens, tokens)
        self.scratch = scratch
        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device) if buffers is None else buffers.zeros(shape, dtype)
        self.previous, self.output, self.embedded = (zeros(tokens, c.hidden) for _ in range(3))
        self.logits = zeros(batch, c.vocabulary)
        self.pending_count = zeros(batch, 1, dtype=torch.int32)
        self.pending_tokens = zeros(batch, self.pending_width, dtype=torch.int64)
        self.pending_hidden, self.pending_embedding = (zeros(batch, self.pending_width, c.hidden) for _ in range(2))
        self.ids = zeros(batch, self.pitch, dtype=torch.int64)
        self.last_tokens = zeros(batch, dtype=torch.int64)
        self.local_embedding = zeros(batch, c.hidden // weights.shard.world_size)
        self.last_embedding = zeros(batch, c.hidden)
        self.embedding = PreparedShardedEmbedding(weights.embedding, self.last_tokens, self.local_embedding,
            self.last_embedding, runtime=embedding_runtime, coordinator=coordinator, session=session, name=name + ".embedding")
        self.histories = (pool.reserved_region("draft.target_hidden"),
                          *(pool.reserved_region(f"draft{i}.hidden") for i in range(len(weights.draft) - 1)))
        self.saved = tuple(zeros(batch, c.hidden) for _ in self.histories)
        self.rows = torch.arange(batch, device=device)
        self.columns = torch.arange(self.pitch, device=device)
        joined = zeros(tokens, 2 * c.hidden)
        self.predictors = []
        for i in range(len(weights.draft)):
            m = Metadata.allocate(tokens, batch, 1, device, samples=batch, buffers=buffers)
            predictor = PreparedPredictor(i, weights, scratch, m, pool, self.output, mode="verify",
                runtime=runtime, embedding_runtime=embedding_runtime, vocabulary_runtime=vocabulary_runtime,
                coordinator=coordinator, session=session, workspace=workspace, name=f"{name}.{i}",
                logits=self.logits if i < depth else None, embedded=self.embedded,
                cache_only=i == len(weights.draft) - 1 and i >= depth, buffers=buffers, joined=joined)
            self.predictors.append((m, predictor))

    def begin(self, states):
        for destination, name in ((self.pending_count, "count"), (self.pending_tokens, "tokens"),
                                  (self.pending_hidden, "hidden"), (self.pending_embedding, "embedding")):
            gather_rows(self.pool.reserved_region("draft.pending_" + name), states, destination)
        for saved, history in zip(self.saved, self.histories, strict=True):
            gather_rows(history, states, saved)
        self.ids.zero_()
        self.ids[:, :self.pending_width].copy_(self.pending_tokens)
        self.embedded.zero_()
        self.embedded.view(self.batch, self.pitch, -1)[:, :self.pending_width].copy_(self.pending_embedding)

    def _pass(self, index, before, states, active, proposing):
        m, predictor = self.predictors[index]
        count = torch.where(active, self.pending_count[:, 0], 0)
        starts = before - count
        lengths = count + torch.where(proposing, index + 1, 0)
        positions = starts[:, None] + self.columns
        valid = (self.columns[None] < lengths[:, None]) & (positions >= index + 1)
        m.token_ids.copy_(self.ids.flatten())
        m.starts.copy_(starts)
        m.lengths.copy_(lengths)
        m.state_rows.copy_(torch.where(active, states, 0))
        m.requests.copy_(self.rows[:, None].expand(-1, self.pitch).flatten())
        m.query_start_loc.copy_(torch.arange(self.batch + 1, device=before.device) * self.pitch)
        m.positions.copy_(torch.where(valid, positions, -1).flatten())
        m.sample_indices.copy_(self.rows * self.pitch + (count + index).clamp_max(self.pitch - 1))
        preceding = self.pending_hidden if index == 0 else self.output.view(self.batch, self.pitch, -1)
        previous = self.previous.view(self.batch, self.pitch, -1)
        previous[:, 0].copy_(self.saved[index])
        if index == 0:
            previous[:, 1:self.pending_width + 1].copy_(preceding[:, :self.pitch - 1])
        else:
            previous[:, 1:].copy_(preceding[:, :-1])
        predictor(self.previous)
        if not predictor.cache_only:
            predictor.layer.attention.commit(count)
        if index + 1 < len(self.histories):
            last = self.output.view(self.batch, self.pitch, -1)[self.rows, (count - 1).clamp_min(0).long()]
            save_hidden(last, self.histories[index + 1], states, (count > 0) & (before > index + 1))
        if index == len(self.predictors) - 1:
            target = self.pending_hidden[self.rows, (count - 1).clamp_min(0).long()]
            save_hidden(target, self.histories[0], states, count > 0)
            save_hidden(torch.zeros_like(self.pending_count), self.pool.reserved_region("draft.pending_count"), states, active)
        return self.logits

    def step(self, index, tokens, before, states, active, proposing=None):
        proposing = active if proposing is None else proposing
        column = (self.pending_count[:, 0] + index).long()
        self.last_tokens.copy_(torch.where(proposing, tokens, 0))
        embedded = self.embedding.gather(self.embedding.lookup())
        self.ids[self.rows, column] = self.last_tokens
        self.embedded.view(self.batch, self.pitch, -1)[self.rows, column] = embedded
        return self._pass(index, before, states, active, proposing)

    def finish(self, before, states, active):
        for index in range(self.depth, len(self.predictors)):
            self._pass(index, before, states, active, torch.zeros_like(active))

    def scratch_tensors(self):
        return (self.scratch, self.previous, self.output, self.embedded, self.logits, self.pending_count,
                self.pending_tokens, self.pending_hidden, self.pending_embedding, self.ids, self.saved,
                self.local_embedding, self.last_embedding, self.last_tokens,
                *((m, p.scratch_tensors()) for m, p in self.predictors))


class PreparedAcceptedContext:
    """Save accepted target rows for the next packed predictor passes."""

    def __init__(self, weights, metadata, pool):
        self.metadata, self.pool = metadata, pool
        self.width = len(weights.draft) + 1
        self.columns = torch.arange(self.width, device=metadata.token_ids.device)

    def __call__(self, hidden, embedded, counts):
        m = self.metadata
        indices = (m.query_start_loc[:-1, None] + self.columns).clamp_max(hidden.shape[0] - 1).long()
        valid = self.columns[None] < counts[:, None]
        for name, source in (("tokens", m.token_ids), ("hidden", hidden), ("embedding", embedded)):
            values = source[indices]
            values = torch.where(valid if name == "tokens" else valid[..., None], values, 0).flatten(1)
            destination = self.pool.reserved_region("draft.pending_" + name)
            save_hidden(values, destination.view(destination.shape[0], -1), m.state_rows, counts > 0)
        save_hidden(counts[:, None], self.pool.reserved_region("draft.pending_count"), m.state_rows, counts > 0)

    def scratch_tensors(self):
        return self.metadata
