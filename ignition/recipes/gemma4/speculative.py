# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 Assistant frozen-KV multi-token proposals and stochastic verification."""

import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.draft_sampling import PreparedDraftDistribution
from ignition.components.embedding import PreparedEmbedding
from ignition.components.paged_batch import Metadata, PreparedBatch
from ignition.components.parallel_draft import prepare_sampling_broadcasts
from ignition.components.speculation import (
    DRAFT_RNG_DOMAIN,
    PreparedSpeculativeCommit,
    PreparedSpeculativeDistributions,
    PreparedStochasticVerification,
)
from ignition.components.state import gather_rows, save_hidden
from ignition.components.vocabulary import PreparedVocabularyProjection

from .attention import attention_descales, cache_bindings, declaration_tensor, prepare_projection, rms_norm
from ignition.preparation.persistent import persistent_copy
from .decoder import PreparedDecoder, PreparedLayer, Scratch, softcap_logits


class PreparedContext:
    """No-op draft context handler since PreparedGeneration already saves pool.last_hidden."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return None

    def functional(self):
        return self

    def scratch_tensors(self):
        return ()


class PreparedAssistant:
    """Single-position recurrent Gemma 4 Assistant reading from the frozen target KV cache."""

    def __init__(
        self,
        weights,
        pool,
        batch: int,
        blocks: int,
        *,
        runtime=None,
        vocabulary_runtime=None,
        coordinator=None,
        session,
        name: str,
        workspace,
        examples,
    ):
        self.weights = weights
        self.pool = pool
        self.batch = batch
        target_config = weights.config
        draft_config = weights.config.draft
        draft = weights.draft
        self.target_config = target_config
        self.draft_config = draft_config
        self.eps = draft_config.rms_norm_eps
        self.softcap = draft_config.final_logit_softcapping
        device = pool.backing.device

        self.metadata = Metadata.allocate(batch, batch, blocks, device, samples=batch)
        self.cache_metadata = cache_bindings(self.metadata)

        def zeros(role, shape, dtype=torch.bfloat16):
            return declaration_tensor(examples, "assistant." + role, shape, dtype, device)

        self.token_embed = zeros("token_embed", (batch, target_config.hidden))
        self.embedding = PreparedEmbedding(
            weights.embedding,
            self.metadata.token_ids,
            self.token_embed,
            num_rows=self.metadata.live_counts[:1],
            session=session,
            name=name + ".embedding",
        )
        self.embed_scale = persistent_copy(torch.tensor(target_config.hidden**0.5, dtype=torch.bfloat16, device=device))

        self.concat_in = zeros("concat_in", (batch, 2 * target_config.hidden))
        self.hidden_states = zeros("hidden_states", (batch, draft_config.hidden))
        self.normed = zeros("normed", (batch, draft_config.hidden))
        self.sampled_normed = zeros("sampled_normed", (batch, draft_config.hidden))
        self.projected = zeros("projected", (batch, target_config.hidden))

        self.pre_projection = prepare_projection(
            self.concat_in,
            draft.pre_projection,
            self.hidden_states,
            session=session,
            name=name + ".pre_projection",
            workspace=workspace,
        )
        self.post_projection = prepare_projection(
            self.sampled_normed,
            draft.post_projection,
            self.projected,
            session=session,
            name=name + ".post_projection",
            workspace=workspace,
        )

        self.sliding_inv_freq = persistent_copy(draft_config.rotary_inverse("sliding_attention", device))
        self.full_inv_freq = persistent_copy(draft_config.rotary_inverse("full_attention", device))
        unit_scale, sliding_descale, full_descale = attention_descales(draft_config, device)

        self.layers = []
        for index, layer_weights in enumerate(draft.layers):
            descale = sliding_descale if draft_config.is_sliding(index) else full_descale
            self.layers.append(
                PreparedLayer(
                    index,
                    layer_weights,
                    draft_config,
                    self.normed,
                    self.metadata,
                    pool,
                    shard=weights.shard,
                    runtime=runtime,
                    coordinator=coordinator,
                    mode="decode",
                    session=session,
                    name=f"{name}.layer{index}",
                    workspace=workspace,
                    examples=examples,
                    cache_metadata=self.cache_metadata,
                    unit_scale=unit_scale,
                    k_descale=descale,
                )
            )

        self.logits = zeros("logits", (batch, target_config.vocabulary))
        if draft_config.use_ordered_embeddings:
            self.head = None
            self.local_logits = None
            self.centroid_logits = zeros("centroid_logits", (batch, draft_config.num_centroids))
            self.centroid_proj = prepare_projection(
                self.sampled_normed,
                draft.centroids,
                self.centroid_logits,
                session=session,
                name=name + ".centroids",
                workspace=workspace,
            )
            self.top_k = draft_config.centroid_intermediate_top_k
            self.vocab_per_centroid = target_config.vocabulary // draft_config.num_centroids
            self.canonical_table = draft.token_ordering.view(
                draft_config.num_centroids, self.vocab_per_centroid
            )
            self.head_weight = draft.head
        else:
            self.centroid_proj = None
            local_vocab = target_config.vocabulary_partition(weights.shard)[2]
            self.local_logits = zeros("local_logits", (batch, local_vocab))
            self.head = PreparedVocabularyProjection(
                self.sampled_normed,
                draft.head,
                self.local_logits,
                self.logits,
                runtime=vocabulary_runtime,
                coordinator=coordinator,
                session=session,
                name=name + ".head",
                activation_mode="auto",
                workspace=workspace,
            )

        self.embedding.functional(session=session, dynamic_inputs=True)
        self.pre_projection.functional(session=session)
        self.post_projection.functional(session=session)
        if self.head is not None:
            self.head.functional(session=session)
        if self.centroid_proj is not None:
            self.centroid_proj.functional(session=session)
        self.metadata = self.cache_metadata = None

    def release_examples(self):
        for name in ("token_embed", "concat_in", "hidden_states", "normed", "sampled_normed", "projected",
                     "logits", "local_logits", "centroid_logits"):
            if hasattr(self, name):
                setattr(self, name, None)

    def step(self, step_index, tokens, prev_hidden, before, states, pages, live_rows, active):
        starts = (before - 1).clamp_min(0)
        lengths = active.int()
        safe_pages = torch.where(active[:, None], pages, -1)
        m = Metadata.pack(starts, lengths, states, safe_pages, tokens[:, None], live_rows,
                          token_capacity=self.batch, sample_count=self.batch)
        cache_metadata = cache_bindings(m)
        pos = torch.where(m.positions >= 0, m.positions + 1 + step_index, 0).float()[:, None]
        s_freq = pos * self.sliding_inv_freq[None, :]
        s_emb = torch.cat((s_freq, s_freq), dim=-1)
        sliding_cos, sliding_sin = s_emb.cos().bfloat16(), s_emb.sin().bfloat16()
        f_freq = pos * self.full_inv_freq[None, :]
        f_emb = torch.cat((f_freq, f_freq), dim=-1)
        full_cos, full_sin = f_emb.cos().bfloat16(), f_emb.sin().bfloat16()
        token_embed = self.embedding(token_ids=m.token_ids, num_rows=m.live_counts[:1]) * self.embed_scale
        hidden = self.pre_projection(torch.cat((token_embed, prev_hidden[m.requests]), dim=-1))
        for layer in self.layers:
            cos, sin = (sliding_cos, sliding_sin) if layer.is_sliding else (full_cos, full_sin)
            hidden = layer(hidden, cos, sin, metadata=m, cache_metadata=cache_metadata)
        sampled = rms_norm(hidden, self.weights.draft.norm, self.eps)[m.sample_indices]
        projected = self.post_projection(sampled)
        if self.head is not None:
            logits = self.head(sampled)
        else:
            b = self.batch
            centroid_logits = self.centroid_proj(sampled)
            _, top_k_indices = torch.topk(centroid_logits, k=self.top_k, dim=-1)
            canonical = self.canonical_table[top_k_indices].reshape(b, self.top_k * self.vocab_per_centroid)
            selected_embeds = self.head_weight[canonical.reshape(-1)].view(
                b, self.top_k * self.vocab_per_centroid, self.draft_config.hidden)
            selected_logits = torch.bmm(sampled.unsqueeze(1), selected_embeds.transpose(1, 2)).squeeze(1)
            mask_value = selected_logits.amin(dim=-1, keepdim=True) - 1.0
            logits = mask_value.expand(b, self.target_config.vocabulary).clone()
            logits.scatter_(dim=-1, index=canonical, src=selected_logits)
        softcap_logits(logits, self.softcap)
        return logits, projected

    def scratch_tensors(self):
        return ()


class PreparedRound:
    """Multi-step Assistant proposal loop followed by parallel target verification and commit."""

    def __init__(self, owner, batch: int, width: int, name: str):
        o, b, w = owner, batch, width
        self.batch, self.width = b, w
        self.collectives, self.session, self.name = o.collectives, o.session, name
        self.inputs, self.requests, self.pool = o.inputs, o.requests, o.pool
        self.completion, self.frame, self.sequence = o.completion, o.frame, o.sequence
        self.rng, self.draft_offset, self.draft_generation = o.rng, o.offset, o.generation
        self.hidden = o.weights.config.hidden
        device, blocks = o.pool.backing.device, o.inputs.page_tables.shape[1]
        self.rows = torch.arange(b, device=device)
        self.metadata = Metadata.allocate(b * w, b, blocks, device, samples=b * w)
        self.validator = PreparedBatch(
            o.inputs,
            o.requests,
            Metadata.allocate(b, b, blocks, device),
            mode="decode",
            shard=o.weights.shard,
            pool=o.pool,
            verification_width=w,
        )
        self.validation = PreparedAllReduce(
            self.validator.validation_input,
            self.validator.validation_sum,
            runtime=o.collectives["runtime"],
            coordinator=o.collectives["coordinator"],
            session=o.session,
            name=name + ".validation",
        )
        scratch = Scratch.allocate(b * w, b * w, o.weights.config, shard=o.weights.shard, device=device, examples=o.examples)
        self.decoder = PreparedDecoder(
            o.weights,
            scratch,
            self.metadata,
            o.pool,
            mode="decode" if w == 1 else "verify",
            session=o.session,
            name=name + ".target",
            workspace=o.workspace,
            examples=o.examples,
            **o.collectives,
        )
        self.draft = (
            None
            if w == 1
            else PreparedAssistant(
                o.weights,
                o.pool,
                b,
                blocks,
                session=o.session,
                name=name + ".draft",
                workspace=o.workspace,
                examples=o.examples,
                **o.collectives,
            )
        )

        def zeros(role, shape, dtype=torch.int64):
            return declaration_tensor(o.examples, "round." + role, shape, dtype, device)

        self.previous = zeros("previous", (b, self.hidden), torch.bfloat16)
        self.before = zeros("before", (b,))
        self.active = zeros("active", (b,), torch.bool)
        self.candidates = zeros("candidates", (b, w))
        self.drafts = zeros("drafts", (b, w - 1))
        self.probabilities = zeros("probabilities", (b, w - 1, o.requests.vocab_size), torch.float32)
        self.valid = zeros("valid", (b, w - 1), torch.bool)
        self.seed, self.offset, self.base_offset, self.sampled = (zeros(role, (b,)) for role in ("seed", "offset", "base_offset", "sampled"))
        self.operations = None

    def prepare(self):
        if self.operations is not None:
            return
        b, w = self.batch, self.width
        args = (
            self.requests,
            self.inputs.row_slot_ids[:b],
            self.inputs.row_generations[:b],
            self.inputs.live_rows,
        )
        target = PreparedSpeculativeDistributions(
            self.decoder.scratch.logits.view(b, w, -1),
            *args,
            self.drafts,
        )
        proposals = (
            ()
            if w == 1
            else tuple(
                PreparedDraftDistribution(
                    self.draft.logits.view(b, 1, -1),
                    *args,
                    self.drafts,
                    prefix_offset=i,
                    seed=self.seed,
                    offset=self.offset,
                    sampled=self.sampled,
                )
                for i in range(w - 1)
            )
        )
        target.prepare()
        for proposal in proposals:
            proposal.prepare()
        verify = PreparedStochasticVerification(
            self.probabilities,
            self.drafts,
            target.probabilities,
            *args,
            rng=self.rng,
            draft_valid=self.valid,
            target_valid=target.valid,
            token_limits=self.inputs.query_lengths[:b],
        )
        verify.prepare()
        prepare_sampling_broadcasts(
            proposals,
            verify,
            collectives=self.collectives,
            session=self.session,
            name=self.name,
        )
        commit = PreparedSpeculativeCommit(
            self.requests,
            verify.output,
            self.candidates,
            self.inputs.row_slot_ids[:b],
            self.inputs.row_generations[:b],
            self.inputs.live_rows,
            self.completion,
            self.frame,
            self.sequence,
            logits=self.decoder.scratch.logits.view(b, w, -1),
            rng=self.rng,
            token_limits=self.inputs.query_lengths[:b],
            draft_context_lag=0,
        )
        commit.prepare()
        self.operations = target, proposals, verify, commit
        self.validation.functional(session=self.session)
        self.validator.functional()
        for operation in (target, *proposals, verify, commit):
            operation.functional(session=self.session)
        self.decoder.release_outputs()
        if self.draft is not None:
            self.draft.release_examples()
        self.metadata = self.rows = None
        for name in ("previous", "before", "active", "candidates", "drafts", "probabilities", "valid",
                     "seed", "offset", "base_offset", "sampled"):
            setattr(self, name, None)

    def __call__(self):
        b, w = self.batch, self.width
        target, proposals, verify, commit = self.operations
        validation, present = self.validator.validate()
        validated = self.validator.begin(self.validation(validation), present=present)
        slots = self.inputs.row_slot_ids[:b].long()
        safe = slots.clamp(0, self.requests.capacity - 1)
        active = validated.lengths > 0
        before = self.requests.consumed_tokens[safe]
        limits = torch.where(active, self.inputs.query_lengths[:b], 0)
        states = self.inputs.request_state_rows[safe, 0]
        pages = self.inputs.page_tables[safe]
        candidates = torch.zeros((b, w), dtype=torch.int64, device=slots.device)
        drafts = torch.zeros((b, w - 1), dtype=torch.int64, device=slots.device)
        probabilities = torch.zeros((b, w - 1, self.requests.vocab_size), dtype=torch.float32, device=slots.device)
        valid = torch.zeros((b, w - 1), dtype=torch.bool, device=slots.device)
        candidates[:, 0].copy_(self.requests.pending_token[safe])
        base_offset = torch.where(self.draft_generation[safe] == self.requests.generation[safe], self.draft_offset[safe], 0)
        seed = self.requests.seed.view(torch.int64)[safe] ^ DRAFT_RNG_DOMAIN
        if w > 1:
            previous = torch.empty((b, self.hidden), dtype=torch.bfloat16, device=slots.device)
            gather_rows(self.pool.last_hidden, torch.where(active, states, 0), previous)
            for i, proposal in enumerate(proposals):
                tokens = candidates[:, 0] if i == 0 else drafts[:, i - 1]
                proposing = active & (i < limits - 1)
                logits, previous = self.draft.step(i, tokens, previous, before, states, pages, self.inputs.live_rows, active)
                scratch = {name: torch.empty(shape, dtype=dtype, device=slots.device)
                           for name, (shape, dtype) in proposal.layout.items()}
                sampled = torch.empty((b,), dtype=torch.int64, device=slots.device)
                proposal.run(scratch, logits.view(b, 1, -1), candidates=drafts,
                             seed=seed, offset=base_offset + 4 * i, sampled=sampled)
                drafts[:, i].copy_(sampled)
                probabilities[:, i].copy_(scratch["probabilities"][:, 0])
                valid[:, i].copy_(scratch["valid"][:, 0] & proposing)
            matches = active[None] & (slots[None] == torch.arange(self.requests.capacity, device=slots.device)[:, None])
            base = torch.where(matches, base_offset[None], 0).sum(1)
            self.draft_offset.copy_(torch.where(matches.any(1), base + 4 * (w - 1), self.draft_offset))
            self.draft_generation.copy_(torch.where(matches.any(1), self.requests.generation, self.draft_generation))
        candidates[:, 1:].copy_(drafts)
        metadata = Metadata.pack(before, limits, states, pages, candidates, self.inputs.live_rows,
                                 token_capacity=b * w, sample_count=b * w, max_query_tokens=w)
        hidden, logits, _, _ = self.decoder(metadata)
        target_scratch = {name: torch.empty(shape, dtype=dtype, device=slots.device)
                          for name, (shape, dtype) in target.layout.items()}
        target.run(target_scratch, logits.view(b, w, -1), candidates=drafts)
        verify_scratch = {name: torch.empty(shape, dtype=dtype, device=slots.device)
                          for name, (shape, dtype) in verify.layout.items()}
        acceptance = verify.run(verify_scratch, probabilities, target_scratch["probabilities"],
                                target_scratch["valid"], disjoint=True, drafts=drafts, draft_valid=valid)
        committed = commit(logits.view(b, w, -1), acceptance=acceptance, inputs=candidates)
        counts = committed.emitted_count
        selected = hidden.view(b, w, self.hidden)[torch.arange(b, device=slots.device), (counts - 1).long().clamp_min(0)]
        save_hidden(selected, self.pool.last_hidden, metadata.state_rows, counts > 0)

    def scratch_tensors(self):
        return ()
