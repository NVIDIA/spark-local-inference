# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Block drafts conditioned on accepted target states, with sequential Markov logits."""

import torch

from ignition.components.embedding import PreparedEmbedding
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.preparation.tuning import register_finalizer

from .attention import AttentionIndices, Rotary, normalize
from .decoder import PreparedLayer


class PreparedDraft:
    def __init__(self, weights, scratch, metadata, pool, *, runtime, vocabulary_runtime,
                 coordinator, session, name, workspace, mla_compute_mode="auto", vocabulary_parallel=False, examples=None):
        if scratch.local_logits.dtype != torch.bfloat16 or scratch.logits.dtype != torch.bfloat16:
            raise ValueError("DSpark proposal vocabulary logits require BF16 storage")
        self.weights, self.scratch, self.metadata = weights, scratch, metadata
        self.embedding = PreparedEmbedding(weights.embedding, metadata.token_ids, scratch.embedding,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding").functional(session=session, dynamic_inputs=True)
        rotary = Rotary(weights.config, scratch.embedding.device)
        page_size = 128 + metadata.max_query_tokens
        self.window_metadata = AttentionIndices(metadata, mode="draft",
            tokens=metadata.positions.numel(), page_size=page_size, width=page_size)
        self.layers = tuple(PreparedLayer(i, layer, None, scratch, metadata, pool, rotary,
            shard=weights.shard, mode="draft", runtime=runtime, coordinator=coordinator,
            session=session, name=f"{name}.layer{i}", workspace=workspace, carry=i > 0,
            defer_post=i + 1 < len(weights.draft.layers), mla_compute_mode=mla_compute_mode,
            window_metadata=self.window_metadata, examples=examples)
            for i, layer in enumerate(weights.draft.layers))
        self.head_hidden = torch.zeros_like(scratch.hidden)
        self.head_hidden_shape = tuple(scratch.hidden.shape)
        self.logit_shape, self.logit_dtype = tuple(scratch.logits.shape), scratch.logits.dtype
        self.local_logit_shape, self.local_logit_dtype = tuple(scratch.local_logits.shape), scratch.local_logits.dtype
        head = weights.head if weights.draft.draft_head is None else weights.draft.draft_head
        self.head = PreparedVocabularyProjection(scratch.hidden, head, scratch.local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head", workspace=workspace, gather=not vocabulary_parallel).functional(session=session)
        self.window_metadata.functional()
        register_finalizer(session, self.release_outputs)

    def release_outputs(self):
        self.scratch = self.metadata = self.head_hidden = None

    def __call__(self, metadata):
        embedding = self.embedding(token_ids=metadata.token_ids, num_rows=metadata.live_counts[:1])
        incoming = (embedding[:, None, :].expand(-1, 4, -1).contiguous(),)
        pre_mix = torch.zeros((embedding.shape[0], 4), dtype=torch.float32, device=embedding.device)
        pre_mix[:, 0].fill_(1)
        indices = self.window_metadata(metadata)
        selection = (torch.full((embedding.shape[0], 512), -1, dtype=torch.int32, device=embedding.device), None, None)
        for layer in self.layers:
            incoming, pre_mix, _, selection = layer(incoming, pre_mix, metadata, indices, selection)
        residual, = incoming
        head_hidden = (residual[metadata.sample_indices].float()
                       * pre_mix[metadata.sample_indices, :, None]).sum(1).bfloat16()
        hidden = torch.zeros_like(head_hidden)
        normalize(head_hidden, self.weights.draft.norm, hidden, self.weights.config.text["rms_norm_eps"])
        features = {"head_hidden": head_hidden.view(metadata.starts.numel(), -1, head_hidden.shape[-1])}
        return self.head(hidden), None, features
