# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron-H heterogeneous residual blocks and vocabulary projection."""

from dataclasses import dataclass

import torch
import triton

from ignition.components.embedding import PreparedEmbedding
from ignition.components.mamba_speculation import PreparedMambaReplay
from ignition.components.recurrent import DeltaMetadata, _commit_convolution
from ignition.components.rms_norm import PreparedRMSNorm
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.preparation.persistent import persistent_copy
from ignition.preparation.examples import declaration_tensor

from .attention import PreparedAttention
from .mamba import PreparedMamba
from .moe import PreparedLatentMoE
from .pool import REPLAY_RING


def mamba_metadata(metadata):
    rows, device = metadata.state_rows.numel(), metadata.state_rows.device
    indices = torch.zeros((rows, metadata.max_query_tokens), dtype=torch.int64, device=device)
    indices[:, 0].copy_(metadata.state_rows)
    return DeltaMetadata(metadata.query_start_loc,
        torch.where(metadata.lengths > 0, metadata.state_rows, -1), metadata.starts == 0,
        torch.ones_like(metadata.lengths), indices,
        torch.where(metadata.starts > 0, metadata.state_rows, 0), metadata.state_rows,
        torch.full_like(metadata.state_rows, -1), torch.zeros_like(metadata.lengths),
        metadata.live_counts[1:], metadata.live_counts[:1])


@dataclass(frozen=True)
class Scratch:
    activations: torch.Tensor
    normalized: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, tokens, rows, config, device, *, examples=None, role="nemotron.target"):
        def empty(field, shape):
            return declaration_tensor(examples, role + "." + field, shape, torch.bfloat16, device)
        return cls(empty("activations", (tokens, config.hidden)), empty("normalized", (tokens, config.hidden)),
            empty("hidden", (rows, config.hidden)), empty("local_logits", (rows, config.vocabulary)),
            empty("logits", (rows, config.vocabulary)))

    def rows(self, tokens, rows):
        return type(self)(self.activations[:tokens], self.normalized[:tokens], self.hidden[:rows],
                          self.local_logits[:rows], self.logits[:rows])


class PreparedDecoder:
    def __init__(self, weights, scratch, metadata, pool, *, mode, session, name, workspace,
                 coordinator=None, vocabulary_runtime=None, vision_state=None, scoring=None, examples=None):
        self.weights, self.scratch, self.pool, self.session = weights, scratch, pool, session
        self.vision_state = vision_state if mode == "prefill" else None
        config, source = weights.config, scratch.normalized
        self.embedding = PreparedEmbedding(weights.embedding, metadata.token_ids, scratch.activations,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        self.mode = mode
        self.kv_scale = persistent_copy(torch.ones(1, device=source.device, dtype=torch.float32))
        self.target_layers = () if weights.draft is None else config.draft.target_layers
        self.auxiliary = None if not self.target_layers else declaration_tensor(examples, "nemotron.target.auxiliary",
            (source.shape[0], len(self.target_layers) * config.hidden), torch.bfloat16, source.device)
        self.width = metadata.max_query_tokens if mode == "verify" else 1
        self.replay = None
        if weights.draft is not None:
            indices = tuple(i for i, kind in enumerate(config.layers) if kind == "mamba")
            regions = [tuple(pool.layout.region(f"layer{i}.{field}") for field in
                             ("recurrent", "replay_x", "replay_b", "replay_dt")) for i in indices]
            self.replay = PreparedMambaReplay(pool.reserved_backing, pool.layout, regions,
                persistent_copy(torch.stack([weights.layers[i]["A"] for i in indices])),
                ring_region=pool.layout.region("mamba_ring_start"), count_region=pool.layout.region("mamba_pending_count"),
                ring_length=REPLAY_RING)
            convolutions = [pool.layout.region(f"layer{i}.convolution") for i in indices]
            self.convolution_offsets = persistent_copy(torch.tensor([r.offset // 2 for r in convolutions], dtype=torch.int64, device=source.device))
            self.convolution_shape = convolutions[0].shape
        self.layers = []
        previous = None
        for index, w in enumerate(weights.layers):
            prefix, kind = f"{name}.layer{index}", config.layers[index]
            norm = PreparedRMSNorm(scratch.activations, w["norm"], source,
                eps=config.text["layer_norm_epsilon"], residual=previous, session=session, name=prefix + ".norm")
            kwargs = dict(session=session, name=prefix, workspace=workspace, examples=examples)
            records = pool.replay_records(index) if mode == "verify" and kind == "mamba" else None
            if kind == "mamba":
                block = PreparedMamba(w, source, pool.recurrent(index), mamba_metadata(metadata), config, mode=mode,
                    replay=self.replay, records=records,
                    verification_width=self.width, **kwargs)
            elif kind == "attention":
                key, value = pool.attention(index, config)
                block = PreparedAttention(w, source, metadata, key, value, config, mode=mode,
                    tuning_page_count=int(pool.memory_capacity[1].item()),
                    key_scale=self.kv_scale, value_scale=self.kv_scale, **kwargs)
            else:
                block = PreparedLatentMoE(w, source, config, index,
                    num_valid_tokens=metadata.live_counts[:1], **kwargs)
            # Every mixer returns a contiguous hidden-width matrix consumed by the following norm.
            previous = source
            self.layers.append((norm, block, index, kind, records))
        self.final = PreparedRMSNorm(scratch.activations, weights.norm, source,
            eps=config.text["layer_norm_epsilon"], residual=previous, session=session, name=name + ".norm_f")
        self.scoring = None if scoring is None else scoring.prepare(scratch.normalized, weights.head,
            vocabulary=weights.config.text["vocab_size"], session=session, coordinator=coordinator,
            name=name + ".prompt")
        self.head = PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head", activation_mode="auto", workspace=workspace)

    def release_outputs(self):
        self.embedding.functional(session=self.session, dynamic_inputs=True)
        self.layers = tuple((norm.functional(), block.functional(session=self.session), index, kind, records)
                            for norm, block, index, kind, records in self.layers)
        self.final = self.final.functional()
        self.head.functional(session=self.session)
        self.scratch = None
        self.auxiliary = None

    def commit(self, counts, records, metadata):
        if self.replay is not None:
            ring_start, pending_count = records["mamba_ring"]
            self.replay.commit(counts, metadata.state_rows, metadata.lengths, ring_start, pending_count,
                               width=self.width)
        channels, columns = self.convolution_shape
        torch.library.wrap_triton(_commit_convolution)[
            (self.convolution_offsets.numel(), counts.numel(), triton.cdiv(channels, 128))](
            self.pool.backing.view(torch.bfloat16), self.convolution_offsets, metadata.state_rows, counts,
            self.pool.layout.row_bytes // 2, channels, columns, 128)

    def __call__(self, metadata):
        m = metadata
        replay_state = None
        records = {}
        recurrent_metadata = mamba_metadata(m)
        if self.replay is not None:
            replay_state = self.replay.prepare(m.state_rows, m.lengths, width=self.width)
            records["mamba_ring"] = replay_state
        residual, output = self.embedding(token_ids=m.token_ids, num_rows=m.live_counts[:1]), None
        auxiliary = (torch.empty((residual.shape[0], len(self.target_layers) * residual.shape[1]),
                      dtype=residual.dtype, device=residual.device) if self.target_layers else None)
        if self.vision_state is not None:
            self.vision_state.inject(m, residual)
        for index, (norm, block, layer_index, kind, layer_records) in enumerate(self.layers):
            normalized = norm(residual, output)
            if index - 1 in self.target_layers:
                offset = self.target_layers.index(index - 1) * residual.shape[1]
                auxiliary[:, offset:offset + residual.shape[1]].copy_(residual)
            if kind == "mamba":
                state = self.pool.recurrent(layer_index)
                recurrent_records = None if layer_records is None else layer_records
                output = block(normalized, recurrent_metadata, state, records=recurrent_records, replay_state=replay_state)
            elif kind == "attention":
                key, value = self.pool.attention(layer_index, self.weights.config)
                output = block(normalized, m, key, value)
            else:
                output = block(normalized, m)
        normalized = self.final(residual, output)
        if len(self.layers) - 1 in self.target_layers:
            offset = self.target_layers.index(len(self.layers) - 1) * residual.shape[1]
            auxiliary[:, offset:offset + residual.shape[1]].copy_(residual)
        if self.scoring is not None:
            self.scoring(normalized)
        hidden = normalized[m.sample_indices]
        logits = self.head(hidden)
        return hidden, logits, auxiliary, records
