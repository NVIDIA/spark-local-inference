# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3.1 DSpark over accepted target feedback and causal latent attention.

Accepted target states populate paged draft history. Proposal records stay in a
private block until target verification decides which target states to retain.
"""

from dataclasses import dataclass
import math

import torch
from b12x.attention import dense_mla
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from ignition.components.collectives import PreparedAllGather, PreparedAllReduce, PreparedChunkedAllGather
from ignition.components.collectives import PreparedLseReduceScatter
from ignition.components.dense_mla import PreparedDenseMLA
from ignition.components.embedding import PreparedShardedEmbedding
from ignition.components.linear import UnquantizedProjection
from ignition.components.mlp import MLPScratch
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.recipes.kimi3.attention import PreparedMLAProjection
from ignition.recipes.kimi3.config import PAGE_TOKENS
from ignition.recipes.kimi3.dcp import RECORD_BYTES, write_records
from ignition.recipes.kimi3.dspark import DraftMLP, canonical_norm, projection
from ignition.recipes.kimi3.pointwise import rms_norm_columns

def inverse_frequencies(config, device):
    """YaRN frequencies with the checkpoint's original position range and correction bands."""
    text = config.draft.text
    rope, width = text["rope_parameters"], text["qk_rope_head_dim"]
    theta, factor = rope["rope_theta"], rope["factor"]
    original = rope["original_max_position_embeddings"]

    def correction(rotations):
        return width * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(theta))

    low = max(math.floor(correction(rope["beta_fast"])), 0)
    high = min(math.ceil(correction(rope["beta_slow"])), width - 1)
    if low == high:
        high += .001
    ramp = ((torch.arange(width // 2, dtype=torch.float32, device=device) - low) / (high - low)).clamp(0, 1)
    frequencies = theta ** (torch.arange(0, width, 2, dtype=torch.float32, device=device) / width)
    extrapolation = 1. - ramp
    return (1. / (factor * frequencies)) * (1. - extrapolation) + (1. / frequencies) * extrapolation


def rotary_gain(config):
    rope = config.draft.text["rope_parameters"]

    def magnitude(mscale):
        return 1. if rope["factor"] <= 1 else 1. + .1 * mscale * math.log(rope["factor"])

    return magnitude(rope["mscale"]) / magnitude(rope["mscale_all_dim"]) * rope["attn_factor"]


def rotate(values, positions, inverse, gain=1.):
    """Interleaved RoPE with FP32 angles and arithmetic, rounded once to the input dtype."""
    angles = positions.clamp_min(0).float()[:, None] * inverse[None]
    shape = (values.shape[0],) + (1,) * (values.ndim - 2) + (inverse.numel(),)
    cos, sin = (angles.cos() * gain).view(shape), (angles.sin() * gain).view(shape)
    even, odd = values[..., ::2].float(), values[..., 1::2].float()
    return torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1).flatten(-2).to(values.dtype)


class Rotary:
    def __init__(self, config, device, *, inverse=None):
        self.inverse = inverse_frequencies(config, device) if inverse is None else inverse
        self.gain = rotary_gain(config)

    def __call__(self, query, key, positions):
        query.copy_(rotate(query, positions, self.inverse, self.gain))
        key.copy_(rotate(key, positions, self.inverse, self.gain))

    def key(self, values, positions):
        values.copy_(rotate(values, positions, self.inverse, self.gain))


def merge_attention(history, history_lse, current, current_lse, history_lengths, current_lengths, output, lse):
    """Combine disjoint attention partitions, including either or both partitions being empty."""
    history_live, current_live = history_lengths[:, None] > 0, current_lengths[:, None] > 0
    kept_lse = torch.where(history_live, history_lse, -torch.inf)
    added_lse = torch.where(current_live, current_lse, -torch.inf)
    peak = torch.maximum(kept_lse, added_lse).clamp_min(-1e30)
    kept, added = torch.exp(kept_lse - peak), torch.exp(added_lse - peak)
    total = (kept + added).clamp_min(1e-30)
    numerator = (torch.where(history_live[..., None], history.float(), 0.) * kept[..., None]
                 + torch.where(current_live[..., None], current.float(), 0.) * added[..., None])
    output.copy_(numerator / total[..., None])
    lse.copy_(torch.where(history_live | current_live, peak + torch.log(total), -torch.inf))


def proposal_metadata(metadata, shards):
    """Address private proposal pages while limiting persistent history to the accepted prefix."""
    m = metadata
    first = shards.local_tokens(m.starts[m.requests])
    live = m.positions >= 0
    offset = m.positions.clamp_min(0) // shards.world - first
    owned = live & (m.positions % shards.world == shards.owner) & (offset >= 0) & (offset < PAGE_TOKENS)
    slots = torch.where(owned, m.requests * PAGE_TOKENS + offset, -1)
    history = torch.where(live, first, 0).to(torch.int32)
    current = torch.where(live, shards.local_tokens(m.positions + 1) - first, 0).to(torch.int32)
    return slots, history, current


def accepted_slots(metadata, counts, shards):
    """Map only accepted target rows to rank-owned persistent draft records."""
    m = metadata
    offset = m.positions - m.starts[m.requests]
    accepted = (m.positions >= 0) & (offset >= 0) & (offset < counts[m.requests])
    return shards.token_slots(torch.where(accepted, m.positions, -1), m.token_page_table)


class PreparedContext:
    """Project accepted target output taps into each draft layer's paged latent history."""

    def __init__(self, weights, auxiliary, metadata, pool, *, session, name, workspace, runtime, coordinator,
                 buffers, context, allocator=None):
        self.metadata, self.shards = metadata, context.shards
        config, draft = weights.config, weights.draft
        tokens, device = auxiliary.shape[0], auxiliary.device
        self.eps = config.draft.text["rms_norm_eps"]

        def buffer(key, *shape, dtype=torch.bfloat16):
            if key not in buffers:
                buffers[key] = (torch.zeros((tokens, *shape), dtype=dtype, device=device) if allocator is None
                                else allocator.zeros((tokens, *shape), dtype))
            if buffers[key].shape[0] < tokens:
                raise ValueError("draft context storage must be allocated for the largest capacity first")
            return buffers[key][:tokens]

        self.local = buffer("local", config.hidden // weights.shard.world_size)
        self.projected = buffer("projected", config.hidden)
        self.normalized = buffer("normalized", config.hidden)
        self.kv = buffer("kv", draft.context_kv.shape[0])
        self.latent = buffer("latent", RECORD_BYTES)
        self.slots = buffer("slots", dtype=torch.int64)
        self.hidden_norm = draft.hidden_norm
        self.project = projection(auxiliary, draft.projection, self.local, session=session,
                                  name=name + ".project", workspace=workspace)
        self.gather = PreparedChunkedAllGather(self.local, self.projected, runtime=runtime, coordinator=coordinator,
                                               session=session, name=name + ".gather")
        self.project_kv = UnquantizedProjection(self.normalized, draft.context_kv, self.kv,
                                                session=session, name=name + ".kv")
        self.rotary = Rotary(config, device)
        self.stages = tuple((layer.attention["kv_norm"], pool.draft_attention(index))
                            for index, layer in enumerate(draft.layers))

    def __call__(self, auxiliary, counts):
        m = self.metadata
        self.slots.copy_(accepted_slots(m, counts, self.shards))
        self.gather(self.project(auxiliary))
        rms_norm_columns(self.projected, self.hidden_norm, self.normalized, self.eps)
        self.project_kv(self.normalized)
        chunks = self.kv.view(self.kv.shape[0], len(self.stages), RECORD_BYTES)
        for index, (norm, cache) in enumerate(self.stages):
            rms_norm_columns(chunks[:, index, :512], norm, self.latent[:, :512], self.eps)
            self.latent[:, 512:].copy_(chunks[:, index, 512:])
            self.rotary.key(self.latent[:, 512:], m.positions)
            write_records(self.latent, self.slots, cache)

    def scratch_tensors(self):
        return (self.local, self.projected, self.normalized, self.kv, self.latent, self.slots)


@dataclass(frozen=True)
class DraftScratch:
    local_embedding: torch.Tensor
    embedding: torch.Tensor
    output: torch.Tensor
    mlp: MLPScratch
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor
    projection: dict
    attention: dict
    constants: dict

    @classmethod
    def allocate(cls, config, shard, tokens, block, device):
        d, text = config.draft, config.draft.text
        hidden, world = config.hidden, shard.world_size
        heads = d.heads // world
        if tokens % block or not 0 < block <= PAGE_TOKENS:
            raise ValueError("DSpark scratch requires complete blocks no wider than one private page")

        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)

        latent = text["q_lora_rank"] + RECORD_BYTES
        local = -(-(latent // world) // 8) * 8
        projection_buffers = {key: zeros(tokens, *shape) for key, shape in {
            "query_norm": (text["q_lora_rank"],), "query": (heads * text["qk_head_dim"],),
            "latent": (RECORD_BYTES,), "absorbed": (heads, RECORD_BYTES), "attended": (heads, 512),
            "expanded": (heads * text["v_head_dim"],), "output": (hidden,), "local": (local,),
            "input": (latent,), "send": (local,), "gathered_latents": (world * local,),
        }.items()}
        attention_buffers = {
            "block": zeros(tokens // block, PAGE_TOKENS, RECORD_BYTES),
            "query": zeros(tokens, heads, RECORD_BYTES),
            "gathered": zeros(tokens, d.heads, RECORD_BYTES),
            "history": zeros(tokens, d.heads, 512), "current": zeros(tokens, d.heads, 512),
            "history_lse": zeros(tokens, d.heads, dtype=torch.float32),
            "current_lse": zeros(tokens, d.heads, dtype=torch.float32),
            "slots": zeros(tokens, dtype=torch.int64),
            "current_pages": zeros(tokens, 1, dtype=torch.int32),
            "history_lengths": zeros(tokens, dtype=torch.int32),
            "current_lengths": zeros(tokens, dtype=torch.int32),
        }
        constants = {"starts": torch.arange(tokens + 1, dtype=torch.int32, device=device),
                     "inverse": inverse_frequencies(config, device)}
        return cls(zeros(tokens, hidden // world), zeros(tokens, hidden), zeros(tokens, hidden),
                   MLPScratch.allocate(tokens, hidden, text["intermediate_size"] // world, device),
                   zeros(tokens, hidden), zeros(tokens, config.vocabulary // world),
                   zeros(tokens, config.vocabulary), projection_buffers, attention_buffers, constants)

    def rows(self, tokens, samples):
        if samples != tokens or not 0 < tokens <= self.embedding.shape[0]:
            raise ValueError("DSpark projects every block position within its scratch capacity")
        return type(self)(self.local_embedding[:tokens], self.embedding[:tokens], self.output[:tokens],
                          self.mlp.rows(tokens), self.hidden[:tokens], self.local_logits[:tokens], self.logits[:tokens],
                          self.projection, self.attention, self.constants)

    def scratch_tensors(self):
        return (self.local_embedding, self.embedding, self.output, self.mlp, self.hidden, self.local_logits,
                self.logits, self.projection, self.attention)


class DraftAttention(torch.nn.Module):
    """Causal MLA over accepted history and a separate proposal block on each context rank."""

    def __init__(self, index, weights, config, scratch, metadata, pool, block, *, runtime, coordinator, session,
                 name, workspace, context, latents):
        super().__init__()
        self.metadata, self.shards = metadata, context.shards
        self.cache = pool.draft_attention(index)
        self.projected = PreparedMLAProjection(weights.attention, scratch.embedding, metadata, self.cache,
            eps=config.draft.text["rms_norm_eps"], mode="verify", session=session, name=name,
            workspace=workspace, buffers=scratch.projection, latents=latents, coordinator=coordinator,
            context=context, gated=False,
            rotary=Rotary(config, scratch.embedding.device, inverse=scratch.constants["inverse"]))
        tokens, requests = metadata.positions.numel(), metadata.starts.numel()
        b = scratch.attention
        self.block = b["block"][:requests]
        self.query, self.gathered = b["query"][:tokens], b["gathered"][:tokens]
        self.history, self.current = b["history"][:tokens], b["current"][:tokens]
        self.history_lse, self.current_lse = b["history_lse"][:tokens], b["current_lse"][:tokens]
        device, width = scratch.embedding.device, metadata.page_table.shape[1]
        if "history_pages" not in b:
            b["history_pages"] = torch.zeros((b["query"].shape[0], width), dtype=torch.int32, device=device)
        self.slots, self.history_pages = b["slots"][:tokens], b["history_pages"][:tokens]
        self.current_pages = b["current_pages"][:tokens]
        self.history_lengths, self.current_lengths = b["history_lengths"][:tokens], b["current_lengths"][:tokens]
        self.starts = scratch.constants["starts"][:tokens + 1]
        self.gather = PreparedAllGather(self.query.view(tokens, -1), self.gathered.view(tokens, -1),
            runtime=context.heads, session=session, coordinator=coordinator, name=name + ".gather_heads",
            dependencies=(self.cache, self.block))
        common = dict(device=device, mode="decode", q_dtype=torch.bfloat16, kv_dtype=torch.bfloat16, page_size=PAGE_TOKENS,
                      use_cuda_graph=True, partial_dtype=torch.float32, num_q_heads=config.draft.heads,
                      max_total_q=tokens, max_batch=tokens)
        scale = config.draft.text["qk_head_dim"] ** -.5
        history_caps = dense_mla.Caps(max_cache_tokens=width * PAGE_TOKENS, max_page_table_width=width,
                                      num_cache_pages=self.cache.shape[0], **common)
        self.history_attention = PreparedDenseMLA(self.gathered, self.cache, self.history, self.history_pages,
            self.history_lengths, self.starts, caps=history_caps, scale=scale, session=session,
            name=name + ".history", workspace=workspace, lse=self.history_lse)
        current_caps = dense_mla.Caps(max_cache_tokens=PAGE_TOKENS, max_page_table_width=1,
                                      num_cache_pages=requests, **common)
        self.current_attention = PreparedDenseMLA(self.gathered, self.block, self.current, self.current_pages,
            self.current_lengths, self.starts, caps=current_caps, scale=scale, session=session,
            name=name + ".block", workspace=workspace, lse=self.current_lse)
        self.combine = PreparedLseReduceScatter(self.history, self.history_lse, self.projected.attended,
            runtime=context.heads, session=session, coordinator=coordinator, name=name + ".combine", threads=512, block_limit=8)
        self.reduce = PreparedAllReduce(self.projected.projections["output"].output, scratch.output,
            runtime=runtime, session=session, coordinator=coordinator, name=name + ".reduce")

    def forward(self, hidden_states, **kwargs):
        m, shards, p = self.metadata, self.shards, self.projected
        p.project(hidden_states)
        live = m.positions >= 0
        slots, history, current = proposal_metadata(m, shards)
        self.slots.copy_(slots)
        write_records(p.latent, self.slots, self.block)
        self.history_pages.copy_(m.token_page_table)
        self.current_pages[:, 0].copy_(m.requests)
        self.history_lengths.copy_(history)
        self.current_lengths.copy_(current)
        self.query.copy_(p.absorbed)
        self.gather(self.query.view(m.positions.numel(), -1))
        self.history_attention()
        self.current_attention()
        merge_attention(self.history, self.history_lse, self.current, self.current_lse,
                        self.history_lengths, self.current_lengths, self.history, self.history_lse)
        self.combine(self.history, self.history_lse)
        p.attended.copy_(torch.where(live[:, None, None], p.attended, 0.))
        return self.reduce(p.finish()), None


class PreparedDraft:
    """Run the causal MLA stack and project every anchor or mask position to target vocabulary logits."""

    def __init__(self, weights, scratch, metadata, pool, *, block, runtime, vocabulary_runtime, embedding_runtime,
                 coordinator, session, name, workspace, context, latents):
        self.weights, self.scratch, self.metadata = weights, scratch, metadata
        self.block = block
        config, draft = weights.config, weights.draft
        eps = config.draft.text["rms_norm_eps"]
        self.embedding = PreparedShardedEmbedding(draft.embedding, metadata.token_ids, scratch.local_embedding,
            scratch.embedding, runtime=embedding_runtime, coordinator=coordinator,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        self.layers = []
        for index, layer_weights in enumerate(draft.layers):
            layer = Qwen3DecoderLayer.__new__(Qwen3DecoderLayer)
            torch.nn.Module.__init__(layer)
            layer.self_attn = DraftAttention(index, layer_weights, config, scratch, metadata, pool, block,
                runtime=runtime, coordinator=coordinator, session=session, name=f"{name}.layer{index}.attention",
                workspace=workspace, context=context, latents=latents)
            layer.mlp = DraftMLP(layer_weights.mlp, scratch, runtime=runtime, coordinator=coordinator,
                                 session=session, name=f"{name}.layer{index}.mlp", workspace=workspace)
            layer.input_layernorm = canonical_norm(layer_weights.input_norm, eps)
            layer.post_attention_layernorm = canonical_norm(layer_weights.post_attention_norm, eps)
            self.layers.append(layer.eval())
        self.norm = canonical_norm(draft.norm, eps)
        head = weights.head if draft.head is None else draft.head
        self.head = PreparedVocabularyProjection(scratch.hidden, head, scratch.local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head")

    def __call__(self):
        hidden = self.embedding.gather(self.embedding.lookup())
        for layer in self.layers:
            hidden = layer(hidden)
        live = (self.metadata.lengths > 0).repeat_interleave(self.block)
        self.scratch.hidden.copy_(torch.where(live[:, None], self.norm(hidden[self.metadata.sample_indices]), 0.))
        self.head(self.scratch.hidden)
