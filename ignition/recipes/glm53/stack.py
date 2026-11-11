# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Complete GLM text forwards with invocation-local tensors and sharded vocabulary weights."""

from dataclasses import dataclass

import torch
from flashinfer.triton.norm import rms_norm

from ignition.components.embedding import PreparedEmbedding
from ignition.components.mhc import MHCScratch
from ignition.components.mlp import MLPScratch
from ignition.components.parallel import TensorParallelShard
from ignition.components.vocabulary import PreparedVocabularyProjection

from .decoder import LayerScratch, PreparedGlmLayer
from .kda import KDAScratch
from .mla import MLAScratch
from .moe import MoeScratch
from .state import DecoderMetadata, DecoderState
from .weights import ModelWeights, MoEWeights


@dataclass(frozen=True)
class DecoderScratch:
    embedding: torch.Tensor
    residual: torch.Tensor
    selected_residual: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor
    attention_hc: MHCScratch
    mlp_hc: MHCScratch
    kda: KDAScratch
    mla: MLAScratch
    dense: MLPScratch
    moe: MoeScratch
    attention_output: torch.Tensor
    mlp_output: torch.Tensor

    @classmethod
    def allocate(cls, tokens, blocks, mode, device, *, shard: TensorParallelShard, sample_rows, context_ranks=1,
                 remote=False):
        """Allocate declaration examples for each operator's layouts and tuning."""
        if type(sample_rows) is not int or not 0 < sample_rows <= tokens:
            raise ValueError("GLM sampling capacity must be positive and fit the token capacity")
        def buffer(*shape):
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)
        layers = (KDAScratch.allocate(tokens, 64 // shard.world_size, device),
            MLAScratch.allocate(tokens, 64 // shard.world_size, blocks, mode, device, context_ranks=context_ranks,
                                remote=remote),
            MLPScratch.allocate(tokens, 4096, 12288 // shard.world_size, device),
            MoeScratch.allocate(tokens, shard.world_size, device))
        return cls(buffer(tokens, 4096), buffer(tokens, 4, 4096), buffer(sample_rows, 4, 4096),
            buffer(sample_rows, 4096), buffer(sample_rows, 154880 // shard.world_size), buffer(sample_rows, 154880),
            MHCScratch.allocate(tokens, 4096, device), MHCScratch.allocate(tokens, 4096, device),
            *layers, buffer(tokens, 4096), buffer(tokens, 4096))

    def rows(self, tokens, mode, *, sample_rows):
        """Select declaration examples with the graph variant's shapes and strides."""
        if type(tokens) is not int or not 0 < tokens <= self.embedding.shape[0]:
            raise ValueError("GLM decoder rows exceed scratch capacity")
        if type(sample_rows) is not int or not 0 < sample_rows <= min(tokens, self.hidden.shape[0]):
            raise ValueError("GLM sample rows exceed scratch capacity")
        return DecoderScratch(self.embedding[:tokens], self.residual[:tokens],
            self.selected_residual[:sample_rows], self.hidden[:sample_rows],
            self.local_logits[:sample_rows], self.logits[:sample_rows],
            self.attention_hc.rows(tokens), self.mlp_hc.rows(tokens), self.kda.rows(tokens),
            self.mla.rows(tokens, mode), self.dense.rows(tokens), self.moe.rows(tokens),
            self.attention_output[:tokens], self.mlp_output[:tokens])

    def layer(self, kind, *, sparse):
        if kind not in ("linear_attention", "indexed_attention"):
            raise ValueError(f"unsupported canonical GLM attention type: {kind}")
        return LayerScratch(self.attention_hc, self.mlp_hc,
            self.kda if kind == "linear_attention" else self.mla, self.attention_output,
            self.moe if sparse else self.dense, self.mlp_output)


class PreparedDecoder:
    """Prepare all 45 layers into a serial forward owned by one TP rank.

    Layers carry MLP output and mixing coefficients into the following fused
    mHC post/pre operation. Inductor manages invocation-local intermediates.
    Each layer's cache and recurrent state have separate owners. Canonical final stream
    averaging and RMSNorm operate only on selected output rows. Vocabulary
    projection shards run on every rank before gathering logits for sampling.
    This object executes in Python only during preparation and CUDA graph capture.
    """

    def __init__(self, weights: ModelWeights, scratch: DecoderScratch, state: DecoderState,
                 metadata: DecoderMetadata, *, mode, runtime, vocabulary_runtime, coordinator, session, name,
                 workspace, progress=None, feedback=None, max_cache_pages=None, context=None, kda_records=None, scoring=None, vision=None):
        language = weights.tree.model.language_model
        if len(weights.layers) != 45 or len(language.layers) != 45 or len(state.layers) != 45:
            raise ValueError("GLM text execution requires all 45 decoder layers")
        if metadata.token_ids.shape != scratch.embedding.shape[:1]:
            raise ValueError("GLM token metadata and scratch capacities differ")
        if metadata.sample_indices.shape != scratch.hidden.shape[:1]:
            raise ValueError("GLM sample indices and hidden-output capacities differ")
        self.weights, self.scratch, self.state, self.metadata = weights, scratch, state, metadata
        self.tokens, self.blocks, self.sample_rows = (scratch.embedding.shape[0],
            scratch.mla.index.page_table.shape[1], scratch.hidden.shape[0])
        self.mode = mode
        self.device, self.context, self.kda_records = metadata.token_ids.device, context, kda_records
        self.full_feedback = feedback is not None
        self.vision = vision if mode == "prefill" else None
        self.scoring = None if scoring is None else scoring.prepare(scratch.embedding, weights.head,
            vocabulary=154880, session=session, coordinator=coordinator, name=name + ".prompt")
        self.feedback = feedback
        self.embedding = PreparedEmbedding(language.embed_tokens.weight, metadata.token_ids, scratch.embedding,
            num_rows=metadata.recurrent.num_tokens, session=session, name=f"{name}.embedding")
        self.layers = []
        for index, (module, layer_weights, layer_state) in enumerate(zip(
                language.layers, weights.layers, state.layers, strict=True)):
            indexed = module.block_type == "indexed_attention"
            layer = PreparedGlmLayer(module, layer_weights.attention, layer_weights.mlp, layer_weights.residual,
                scratch.residual if index == 0 else scratch.mlp_output, scratch.residual,
                scratch.layer(module.block_type, sparse=isinstance(layer_weights.mlp, MoEWeights)), layer_state,
                metadata.indexed if indexed else metadata.recurrent, mode=mode, runtime=runtime,
                coordinator=coordinator, session=session, name=f"{name}.layer{index}", workspace=workspace,
                carry=None if index == 0 else scratch.mlp_hc, defer_post=index < len(language.layers) - 1,
                max_cache_pages=max_cache_pages, context=context,
                kda_records=None if kda_records is None or indexed else kda_records.layer(index))
            self.layers.append(layer)
            if progress is not None:
                progress(f"declared {mode} layer {index + 1}/45")
        self.lm_head = PreparedVocabularyProjection(scratch.hidden, weights.head,
            scratch.local_logits, scratch.logits, runtime=vocabulary_runtime, coordinator=coordinator,
            session=session, name=f"{name}.lm_head", workspace=workspace)

    def functional(self, *, session):
        self.embedding.functional(session=session, dynamic_inputs=True)
        for layer in self.layers:
            layer.release_outputs(session=session)
        self.lm_head.functional(session=session)
        self.scratch = self.metadata = self.feedback = self.kda_records = None
        return self

    def rebind_pages(self):
        """Rebind every sparse-attention layer to the continuation pool's current page pools."""
        for layer in self.layers:
            if hasattr(layer.attention, "rebind_pages"):
                layer.attention.rebind_pages()

    def __call__(self, metadata, *, records=None):
        language = self.weights.tree.model.language_model
        embedded = self.embedding(token_ids=metadata.token_ids, num_rows=metadata.recurrent.num_tokens)
        if self.vision is not None:
            self.vision.inject(metadata, embedded)
        value, carry = embedded[:, None, :].expand(-1, 4, -1).contiguous(), None
        for index, layer in enumerate(self.layers):
            layer_records = None if records is None or index not in records.layers else records.layer(index)
            attention_metadata = metadata.indexed if layer.module.block_type == "indexed_attention" else metadata.recurrent
            value, carry = layer(value, attention_metadata, carry, records=layer_records)
        auxiliary = None
        if not self.full_feedback and self.scoring is None:
            selected = value[metadata.sample_indices]
            collapsed = language.hc_head(selected[:, None]).squeeze(1)
            hidden = torch.empty_like(collapsed)
            rms_norm(collapsed, language.norm.weight, hidden, language.norm.variance_epsilon,
                round_before_weight=True)
        else:
            collapsed = language.hc_head(value[:, None]).squeeze(1)
            normalized = torch.empty_like(collapsed)
            rms_norm(collapsed, language.norm.weight, normalized, language.norm.variance_epsilon,
                round_before_weight=True)
            auxiliary = normalized if self.full_feedback or self.scoring is not None else None
            hidden = normalized[metadata.sample_indices]
        if self.scoring is not None:
            self.scoring(auxiliary)
        logits = self.lm_head(hidden)
        return hidden, logits, auxiliary, records
