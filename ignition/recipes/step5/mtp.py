# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Three causal MTP predictors with accepted-prefix feedback and private proposal KV."""

from dataclasses import dataclass, replace

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
                 embedded=None, cache_only=False, joined=None):
        self.metadata, self.output, self.vision = metadata, output, vision
        self.external_embedding, self.cache_only, self.scratch = embedded, cache_only, scratch
        w, c = weights.draft[index], weights.config
        self.embedding = None if embedded is not None else PreparedShardedEmbedding(weights.embedding, metadata.token_ids,
            scratch.local_embedding, scratch.embedding, runtime=embedding_runtime, coordinator=coordinator,
            session=session, name=name + ".embedding").functional(session=session, dynamic_inputs=True)
        def zeros(shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=output.device)
        self.joined = zeros((output.shape[0], 2 * c.hidden)) if joined is None else joined
        self.join = PreparedPairedRMSNorm(output, output, (w["enorm"], w["hnorm"]), self.joined,
            eps=c.text["rms_norm_eps"], session=session, name=name + ".norm_concat", zero_centered=True).functional()
        self.project = UnquantizedProjection(self.joined, w["projection"], scratch.embedding,
                                              session=session, name=name + ".eh").functional(session=session)
        layer = c.layers + index
        common = {"runtime": runtime, "coordinator": coordinator, "session": session, "workspace": workspace}
        fused = PreparedAllReduceRMSNorm.fits(runtime, scratch.partial)
        attention = PreparedAttention(layer, w["layer"], c, weights.shard, scratch.normalized, scratch.attention,
            metadata, pool, mode=mode, name=name + ".attention", first_position=index + 1, proposal=proposal,
            reduce_output=not fused,  **common)
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
                session=session, name=name + ".norm", zero_centered=True).functional()
            self.local_logits = zeros((logits.shape[0], w["head"].shape[0]), logits.dtype)
            self.head = PreparedVocabularyProjection(self.normalized, w["head"], self.local_logits, logits,
                runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head",  workspace=workspace).functional(session=session)

    def __call__(self, previous, metadata, *, embedded=None, proposal=None):
        if embedded is None:
            local = self.embedding.lookup(token_ids=metadata.token_ids)
            if self.vision is not None:
                self.vision.inject(metadata, local)
            embedded = self.embedding.gather(local)
        joined = self.join(embedded, previous)
        residual = self.project(joined).float()
        if self.cache_only:
            self.layer.attention(self.layer.input_norm(residual), metadata=metadata, cache_only=True, proposal=proposal)
            return None, None, None
        update, records = self.layer(residual, metadata=metadata, proposal=proposal)
        residual.add_(update)
        output = torch.where((metadata.positions >= 0)[:, None], residual, 0).bfloat16()
        logits = None if self.head is None else self.head(self.norm(output[metadata.sample_indices]))
        return output, logits, records

    def release_outputs(self):
        self.metadata = self.output = self.external_embedding = self.scratch = self.joined = None
        for name in ("sampled", "normalized", "local_logits"):
            if hasattr(self, name):
                setattr(self, name, None)
        self.layer.feed_forward.scratch = None
        self.layer.attention.release_examples()

    def scratch_tensors(self):
        return ()


class PreparedFeedback:
    """Teacher-force all predictors using each preceding level's shifted hidden states."""

    def __init__(self, weights, auxiliary, metadata, pool, *, scratch, runtime, embedding_runtime,
                 vocabulary_runtime, coordinator, session, name, workspace, vision=None):
        self.source, self.pool = metadata, pool
        self.flush = PreparedProposals(weights, pool, scratch, metadata.starts.numel(), 0,
            runtime=runtime, embedding_runtime=embedding_runtime, vocabulary_runtime=vocabulary_runtime,
            coordinator=coordinator, session=session, workspace=workspace, name=name + ".pending")
        self.metadata = m = Metadata.allocate(auxiliary.shape[0], metadata.starts.numel(), metadata.page_table.shape[1],
                                              auxiliary.device, samples=auxiliary.shape[0])
        def zeros(shape):
            return (torch.zeros(shape, dtype=torch.bfloat16, device=auxiliary.device))
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
             joined=joined)
            for i in range(len(weights.draft)))

    def __call__(self, auxiliary, counts, metadata):
        hidden, _ = auxiliary
        active = counts > 0
        states = torch.where(active, metadata.state_rows, 0)
        pending = self.flush.begin(states)
        self.flush.finish(pending, metadata.starts, states, active)
        indices = torch.arange(hidden.shape[0], device=hidden.device)
        offset = indices - metadata.query_start_loc[metadata.requests]
        live = (metadata.positions >= 0) & (offset < counts[metadata.requests])
        last = (metadata.query_start_loc[:-1] + counts - 1).clamp(0, hidden.shape[0] - 1).long()
        saved = tuple(torch.empty((counts.numel(), hidden.shape[1]), dtype=hidden.dtype, device=hidden.device)
                      for _ in self.histories)
        for destination, history in zip(saved, self.histories, strict=True):
            gather_rows(history, metadata.state_rows, destination)
        previous = hidden
        for index, predictor in enumerate(self.predictors):
            positions = torch.where(live & (metadata.positions >= index + 1), metadata.positions, -1)
            m = replace(metadata, positions=positions, lengths=counts.to(torch.int32))
            source = torch.where((offset == 0)[:, None], saved[index][m.requests],
                                 previous[(indices - 1).clamp_min(0)])
            previous, _, _ = predictor(source, m)
            valid = (counts > 0) & (metadata.starts + counts > index + 1)
            if index + 1 < len(self.histories):
                save_hidden(previous[last], self.histories[index + 1], m.state_rows, valid)
        save_hidden(hidden[last], self.histories[0], metadata.state_rows, counts > 0)

    def functional(self, *, session=None):
        self.release_outputs()
        return self

    def release_outputs(self):
        self.source = self.metadata = self.previous = self.output = self.saved = self.indices = self.rows = None
        self.flush.release_outputs()
        for predictor in self.predictors:
            predictor.release_outputs()

    def scratch_tensors(self):
        return ()


@dataclass(frozen=True)
class ProposalState:
    pending_count: torch.Tensor
    pending_tokens: torch.Tensor
    pending_hidden: torch.Tensor
    pending_embedding: torch.Tensor
    saved: tuple[torch.Tensor, ...]
    ids: torch.Tensor
    embedded: torch.Tensor
    output: torch.Tensor | None = None


class PreparedProposals:
    """Combine accepted teacher forcing and proposals in one packed pass per predictor.

    Continuations retain the accepted target rows not yet consumed by the draft.
    Each pass commits only those rows; candidate KV remains private until target
    verification supplies the accepted target states for the following round.
    """

    def __init__(self, weights, pool, scratch, batch, depth, *, runtime, embedding_runtime,
                 vocabulary_runtime, coordinator, session, workspace, name):
        self.batch, self.depth, self.pool = batch, depth, pool
        c, device = weights.config, pool.backing.device
        self.pending_width = len(weights.draft) + 1
        self.pitch = self.pending_width + depth
        tokens = batch * self.pitch
        if scratch.embedding.shape[0] < tokens:
            scratch = Scratch.allocate(c, weights.shard, tokens, tokens, 0, device)
        else:
            scratch = scratch.rows(tokens, tokens)
        self.scratch = scratch
        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)
        self.previous, self.output, self.embedded = (zeros(tokens, c.hidden) for _ in range(3))
        self.logits = zeros(batch, c.vocabulary)
        self.logit_shape, self.logit_dtype = tuple(self.logits.shape), self.logits.dtype
        self.pending_count = zeros(batch, 1, dtype=torch.int32)
        self.pending_tokens = zeros(batch, self.pending_width, dtype=torch.int64)
        self.pending_hidden, self.pending_embedding = (zeros(batch, self.pending_width, c.hidden) for _ in range(2))
        self.ids = zeros(batch, self.pitch, dtype=torch.int64)
        self.last_tokens = zeros(batch, dtype=torch.int64)
        self.local_embedding = zeros(batch, c.hidden // weights.shard.world_size)
        self.last_embedding = zeros(batch, c.hidden)
        self.embedding = PreparedShardedEmbedding(weights.embedding, self.last_tokens, self.local_embedding,
            self.last_embedding, runtime=embedding_runtime, coordinator=coordinator, session=session, name=name + ".embedding").functional(session=session, dynamic_inputs=True)
        self.histories = (pool.reserved_region("draft.target_hidden"),
                          *(pool.reserved_region(f"draft{i}.hidden") for i in range(len(weights.draft) - 1)))
        self.saved = tuple(zeros(batch, c.hidden) for _ in self.histories)
        self.rows = torch.arange(batch, device=device)
        self.columns = torch.arange(self.pitch, device=device)
        joined = zeros(tokens, 2 * c.hidden)
        self.predictors = []
        for i in range(len(weights.draft)):
            m = Metadata.allocate(tokens, batch, 1, device, samples=batch)
            predictor = PreparedPredictor(i, weights, scratch, m, pool, self.output, mode="verify",
                runtime=runtime, embedding_runtime=embedding_runtime, vocabulary_runtime=vocabulary_runtime,
                coordinator=coordinator, session=session, workspace=workspace, name=f"{name}.{i}",
                logits=self.logits if i < depth else None, embedded=self.embedded,
                cache_only=i == len(weights.draft) - 1 and i >= depth,  joined=joined)
            self.predictors.append((m, predictor))

    def begin(self, states):
        hidden = self.histories[0].shape[-1]
        def zeros(shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=states.device)
        count = zeros((self.batch, 1), torch.int32)
        tokens = zeros((self.batch, self.pending_width), torch.int64)
        pending_hidden, pending_embedding = (zeros((self.batch, self.pending_width, hidden)) for _ in range(2))
        saved = tuple(zeros((self.batch, hidden)) for _ in self.histories)
        for destination, name in ((count, "count"), (tokens, "tokens"),
                                  (pending_hidden, "hidden"), (pending_embedding, "embedding")):
            gather_rows(self.pool.reserved_region("draft.pending_" + name), states, destination)
        for destination, history in zip(saved, self.histories, strict=True):
            gather_rows(history, states, destination)
        ids = zeros((self.batch, self.pitch), torch.int64)
        ids[:, :self.pending_width].copy_(tokens)
        embedded = zeros((self.batch, self.pitch, hidden))
        embedded[:, :self.pending_width].copy_(pending_embedding)
        return ProposalState(count, tokens, pending_hidden, pending_embedding, saved, ids, embedded)

    def _pass(self, state, index, before, states, active, proposing):
        predictor = self.predictors[index][1]
        count = torch.where(active, state.pending_count[:, 0], 0)
        starts = before - count
        lengths = count + torch.where(proposing, index + 1, 0)
        columns = torch.arange(self.pitch, device=before.device)
        rows = torch.arange(self.batch, device=before.device)
        positions = starts[:, None] + columns
        valid = (columns[None] < lengths[:, None]) & (positions >= index + 1)
        m = Metadata(
            state.ids.flatten(), rows * self.pitch + (count + index).clamp_max(self.pitch - 1),
            torch.arange(self.batch + 1, dtype=torch.int32, device=before.device) * self.pitch,
            torch.where(active, states, 0), starts, lengths.to(torch.int32),
            rows[:, None].expand(-1, self.pitch).flatten().contiguous(), torch.where(valid, positions, -1).flatten(),
            torch.full((self.batch, 1), -1, dtype=torch.int32, device=before.device),
            torch.full((self.batch * self.pitch, 1), -1, dtype=torch.int32, device=before.device),
            torch.zeros(2, dtype=torch.int32, device=before.device), self.pitch)
        preceding = state.pending_hidden if index == 0 else state.output.view(self.batch, self.pitch, -1)
        previous = torch.zeros_like(state.embedded)
        previous[:, 0].copy_(state.saved[index])
        if index == 0:
            previous[:, 1:self.pending_width + 1].copy_(preceding[:, :self.pitch - 1])
        else:
            previous[:, 1:].copy_(preceding[:, :-1])
        output, logits, records = predictor(previous.flatten(0, 1), m, embedded=state.embedded.flatten(0, 1))
        if not predictor.cache_only:
            predictor.layer.attention.commit(count, records, m)
        if index + 1 < len(self.histories):
            last = output.view(self.batch, self.pitch, -1)[rows, (count - 1).clamp_min(0).long()]
            save_hidden(last, self.histories[index + 1], states, (count > 0) & (before > index + 1))
        if index == len(self.predictors) - 1:
            target = state.pending_hidden[rows, (count - 1).clamp_min(0).long()]
            save_hidden(target, self.histories[0], states, count > 0)
            save_hidden(torch.zeros_like(state.pending_count), self.pool.reserved_region("draft.pending_count"), states, active)
        return logits, replace(state, output=output)

    def step(self, state, index, tokens, before, states, active, proposing=None):
        proposing = active if proposing is None else proposing
        column = (state.pending_count[:, 0] + index).long()
        rows = torch.arange(self.batch, device=tokens.device)
        selected = torch.where(proposing, tokens, 0)
        embedded = self.embedding(token_ids=selected)
        state.ids[rows, column] = selected
        state.embedded[rows, column] = embedded
        return self._pass(state, index, before, states, active, proposing)

    def finish(self, state, before, states, active):
        for index in range(self.depth, len(self.predictors)):
            _, state = self._pass(state, index, before, states, active, torch.zeros_like(active))
        return state

    def release_outputs(self):
        for name in ("scratch", "previous", "output", "embedded", "logits", "pending_count", "pending_tokens",
                     "pending_hidden", "pending_embedding", "ids", "last_tokens", "local_embedding", "last_embedding",
                     "saved", "rows", "columns"):
            setattr(self, name, None)
        for _, predictor in self.predictors:
            predictor.release_outputs()
        self.predictors = tuple((None, predictor) for _, predictor in self.predictors)

    def scratch_tensors(self):
        return ()


class PreparedAcceptedContext:
    """Save accepted target rows for the next packed predictor passes."""

    def __init__(self, weights, metadata, pool):
        self.metadata, self.pool = metadata, pool
        self.width = len(weights.draft) + 1
        self.columns = torch.arange(self.width, device=metadata.token_ids.device)

    def __call__(self, auxiliary, counts, metadata):
        hidden, embedded = auxiliary
        columns = torch.arange(self.width, device=hidden.device)
        indices = (metadata.query_start_loc[:-1, None] + columns).clamp_max(hidden.shape[0] - 1).long()
        valid = columns[None] < counts[:, None]
        for name, source in (("tokens", metadata.token_ids), ("hidden", hidden), ("embedding", embedded)):
            values = source[indices]
            values = torch.where(valid if name == "tokens" else valid[..., None], values, 0).flatten(1)
            destination = self.pool.reserved_region("draft.pending_" + name)
            save_hidden(values, destination.view(destination.shape[0], -1), metadata.state_rows, counts > 0)
        save_hidden(counts[:, None], self.pool.reserved_region("draft.pending_count"), metadata.state_rows, counts > 0)

    def functional(self, *, session=None):
        self.release_outputs()
        return self

    def release_outputs(self):
        self.metadata = self.columns = None

    def scratch_tensors(self):
        return ()
