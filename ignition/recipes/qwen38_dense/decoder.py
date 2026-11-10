# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Canonical Qwen dense layers with prepared projections and persistent GPU caches."""

from copy import copy
from dataclasses import dataclass

import torch
import triton

from ignition.components.embedding import PreparedEmbedding
from ignition.components.dflash2 import PreparedContextCapture
from ignition.components.gated_delta import GDNScratch, PreparedGDN
from ignition.components.head_gate import PreparedHeadGate
from ignition.components.linear import QuantizedProjection, prepare_projection
from ignition.components.gdn_recovery import GDNLayerRecords, PreparedGDNCommit
from ignition.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter
from ignition.components.recurrent import DeltaMetadata, _commit_convolution
from ignition.components.rms_norm import PreparedRMSNorm
from ignition.components.head_norm_rope import head_norm_rope
from ignition.components.tensor_fp8 import TensorFP8Projection, TensorFP8Weight
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.preparation.quantization import linear_format

from .config import PAGE_TOKENS


@dataclass(frozen=True)
class Scratch:
    """Declaration and tuning examples released before forward compilation."""

    activations: torch.Tensor
    hidden: torch.Tensor
    logits: torch.Tensor
    gdn: GDNScratch
    quantized: torch.Tensor

    @classmethod
    def allocate(cls, tokens, rows, config, device):
        def zeros(shape):
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)
        quantized_width = (max(config.hidden, config.text["num_attention_heads"] * config.text["head_dim"], 6144)
            if any(linear_format(config.quantization, name) == "tensor_fp8" for name in config.quantization) else 0)
        quantized_shape = (tokens * quantized_width,)
        quantized = torch.zeros(quantized_shape, dtype=torch.uint8, device=device)
        return cls(zeros((tokens, config.hidden)), zeros((rows, config.hidden)),
                   zeros((rows, config.vocabulary)),
                   GDNScratch.allocate(tokens, device, hidden_size=config.hidden,
                                       pack_ba=config.pack_gdn_inputs), quantized)

    def rows(self, tokens, rows):
        return Scratch(self.activations[:tokens], self.hidden[:rows], self.logits[:rows], self.gdn.rows(tokens), self.quantized)


class Projection(torch.nn.Module):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def forward(self, source):
        return self.operation(source)


def _uses_fused_swiglu(activation, down_weight, device):
    from b12x.gemm.blockscaled import NVFP4LinearWeight
    from transformers.activations import SiLUActivation
    from ignition.components import sm103_projection

    canonical_silu = (type(activation) is SiLUActivation
                      or type(activation) is torch.nn.SiLU and not activation.inplace)
    return canonical_silu and isinstance(down_weight, NVFP4LinearWeight) and not sm103_projection.is_sm103(device)


def _allocate_mlp_intermediate(weights, source):
    for index, layer in enumerate(weights.tree.layers):
        prefix = f"layers.{index}.mlp."
        if (prefix + "gate_up" not in weights.projections
                or not _uses_fused_swiglu(layer.mlp.act_fn, weights.projections[prefix + "down_proj"], source.device)):
            shape = (source.shape[0], weights.config.text["intermediate_size"])
            return torch.zeros(shape, dtype=torch.bfloat16, device=source.device)
    return None


class GroupedMLP(torch.nn.Module):
    """Concatenated gate/up projection with the canonical BF16 activation."""

    def __init__(self, canonical, source, gate_up_weight, down_weight, gate_up, intermediate, output,
                 *, session, name, workspace):
        super().__init__()
        self.activation = canonical.act_fn
        self.gate_up = prepare_projection(source, gate_up_weight, gate_up,
            activation_mode="auto", session=session, workspace=workspace, name=name + ".gate_up").functional(session=session)
        self.down_source_is_gate_up = _uses_fused_swiglu(self.activation, down_weight, source.device)
        if self.down_source_is_gate_up:
            self.down = QuantizedProjection(gate_up, down_weight, output,
                source_activation="swiglu_bf16", activation_mode="auto", session=session,
                workspace=workspace, name=name + ".down").functional(session=session)
        else:
            self.down = prepare_projection(intermediate, down_weight, output,
                activation_mode="auto", session=session, workspace=workspace, name=name + ".down").functional(session=session)

    def forward(self, source):
        projected = self.gate_up(source)
        if self.down_source_is_gate_up:
            return self.down(projected)
        gate, up = projected.chunk(2, dim=-1)
        return self.down(self.activation(gate) * up)


class RMSNorm(torch.nn.Module):
    def __init__(self, canonical, source, output, *, session, name, residual=None):
        super().__init__()
        self.operation = PreparedRMSNorm(source, canonical.weight, output, eps=canonical.eps,
            zero_centered=True, session=session, name=name, residual=residual).functional()

    def forward(self, source, residual=None):
        return self.operation(source, residual)


class ResidualNormalizedLayer(torch.nn.Module):
    """Compute attention and MLP branches with prepared residual normalization."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.next_input_layernorm = None

    def forward(self, hidden_states, normalized, position_embeddings, records=None, *, metadata, delta):
        layer = self.layer
        if layer.block_type == "linear_attention":
            attended = layer.linear_attn(normalized, records=records, metadata=delta)
        else:
            attended, _ = layer.self_attn(normalized, position_embeddings=position_embeddings, metadata=metadata)
        normalized = layer.post_attention_layernorm(attended, hidden_states)
        projected = layer.mlp(normalized)
        if self.next_input_layernorm is None:
            return attended + projected, None
        return projected, self.next_input_layernorm(projected, attended)


def module_copy(module):
    result = copy(module)
    result._modules = module._modules.copy()
    return result


class RecurrentAttention(torch.nn.Module):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def forward(self, hidden_states, records=None, *, metadata, **kwargs):
        return self.operation(hidden_states, records=records, metadata=metadata)


def _attention_query_dtype(source, metadata, mode, head_dim):
    if (mode == "prefill" and source.shape[0] >= 128 and head_dim == 256
            and metadata.page_table.shape[0] == 1
            and torch.cuda.get_device_capability(source.device) in ((12, 0), (12, 1))):
        return torch.float8_e4m3fn
    return torch.bfloat16


class FullAttention(torch.nn.Module):
    def __init__(self, module, weights, source, metadata, pool, layer, *, mode, session, workspace, name, buffers,
                 projection_staging=None):
        super().__init__()
        self.canonical = module
        config = weights.config
        self.heads, self.kv_heads = config.text["num_attention_heads"], config.text["num_key_value_heads"]
        self.dim = config.text["head_dim"]
        tokens, device = source.shape[0], source.device
        self.tokens = tokens

        def empty(field, width, dtype=torch.bfloat16):
            if field not in buffers:
                buffers[field] = torch.zeros((tokens, width), dtype=dtype, device=device)
            return buffers[field][:tokens]

        prefix = f"layers.{layer}.self_attn."
        packed = prefix + "qgkv_proj" in weights.projections
        self.packed = packed
        if packed:
            q_width, kv_width = self.heads * self.dim, self.kv_heads * self.dim
            qgkv = empty("qgkv", 2 * q_width + 2 * kv_width)
            query, gate, key, value = qgkv.split((q_width, q_width, kv_width, kv_width), dim=-1)
            query, gate = (part.view(tokens, self.heads, self.dim) for part in (query, gate))
            inputs = (("qgkv_proj", source, qgkv),)
        else:
            qg = empty("qg", 2 * self.heads * self.dim)
            query, gate = qg.view(tokens, self.heads, 2 * self.dim).chunk(2, dim=-1)
            if prefix + "kv_proj" in weights.projections:
                kv = empty("kv", 2 * self.kv_heads * self.dim)
                key, value = kv.chunk(2, dim=-1)
                inputs = (("kv_proj", source, kv),)
            else:
                key, value = (empty(field, self.kv_heads * self.dim) for field in ("k", "v"))
                inputs = (("k_proj", source, key), ("v_proj", source, value))
            inputs = (("q_proj", source, qg), *inputs)
        query_dtype = _attention_query_dtype(source, metadata, mode, self.dim)
        self.query_dtype = query_dtype
        q = empty("q", self.heads * self.dim, query_dtype).view(tokens, self.heads, self.dim)
        normalized_key = empty("key", self.kv_heads * self.dim).view(tokens, self.kv_heads, self.dim)
        attended, output = empty("attended", self.heads * self.dim), empty("output", config.hidden)
        def projection(field, inputs, output):
            weight = weights.projections[prefix + field]
            if isinstance(weight, TensorFP8Weight):
                return TensorFP8Projection(inputs, weight, output, session=session, name=name + "." + field,
                                           staging=projection_staging).functional(session=session)
            return prepare_projection(inputs, weight, output, session=session, name=name + "." + field,
                                      workspace=workspace).functional(session=session)
        operands = (*inputs, ("o_proj", attended, output))
        self.input_projections = tuple(field for field, _, _ in operands[:-1])
        self.projections = {
            field: projection(field, inputs, output)
            for field, inputs, output in operands
        }
        slots = torch.full((tokens,), -1, dtype=torch.int64, device=device)
        self.lengths = torch.zeros_like(metadata.lengths)
        key_cache, value_cache = pool.attention(layer, config)
        self.writer = PreparedPagedKVWriter(normalized_key, value.view(tokens, self.kv_heads, self.dim),
            key_cache, value_cache, slots, session=session, name=name + ".write", caller_owned_placeholders=True).functional()
        self.attention = PreparedPagedAttention(q, key_cache, value_cache, metadata.page_table,
            self.lengths, metadata.query_start_loc, attended.view(tokens, self.heads, self.dim),
            output_gate=gate if mode == "prefill" else None,
            tuning_page_count=int(pool.memory_capacity[1].item()),
            mode=mode, width=metadata.max_query_tokens if mode == "verify" else 1,
            session=session, workspace=workspace, name=name + ".attention", caller_owned_placeholders=True)
        if query_dtype == torch.float8_e4m3fn and self.attention.tma is None:
            raise ValueError("FP8 Qwen prefill queries require the prepared TMA attention path")
        self.attention = self.attention.functional(session=session, cache_seqlens=None,
                                                    page_table=None, cu_seqlens_q=None)
        self.lengths = None
        self.output_gate = None
        if mode != "prefill":
            values = (attended.view(tokens, self.heads * self.dim, 1) if packed
                      else attended.view(tokens * self.heads, self.dim, 1))
            gates = (gate.view(tokens, self.heads * self.dim) if packed
                     else gate.view(tokens * self.heads, self.dim))
            self.output_gate = PreparedHeadGate(values, gates, session=session, name=name + ".output_gate").functional(session=session)

    def write_cache(self, hidden_states, position_embeddings, metadata):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

        projected = {field: self.projections[field](hidden_states) for field in self.input_projections}
        tokens, heads, kv_heads, dim = self.tokens, self.heads, self.kv_heads, self.dim
        if self.packed:
            query, gate, key, value = projected["qgkv_proj"].split(
                (heads * dim, heads * dim, kv_heads * dim, kv_heads * dim), dim=-1)
            query, gate = (part.view(tokens, heads, dim) for part in (query, gate))
        else:
            query, gate = projected["q_proj"].view(tokens, heads, 2 * dim).chunk(2, dim=-1)
            if "kv_proj" in projected:
                key, value = projected["kv_proj"].chunk(2, dim=-1)
            else:
                key, value = projected["k_proj"], projected["v_proj"]
        q = torch.empty((tokens, heads, dim), dtype=self.query_dtype, device=hidden_states.device)
        normalized_key = torch.empty((tokens, kv_heads, dim), dtype=hidden_states.dtype, device=hidden_states.device)
        if self.query_dtype == torch.float8_e4m3fn:
            cosine, sine = position_embeddings
            head_norm_rope(query, self.canonical.q_norm.weight, cosine, sine, q,
                           eps=self.canonical.q_norm.eps)
            head_norm_rope(key.view(-1, kv_heads, dim), self.canonical.k_norm.weight,
                           cosine, sine, normalized_key, eps=self.canonical.k_norm.eps)
        else:
            rotated_query, rotated_key = apply_rotary_pos_emb(self.canonical.q_norm(query),
                self.canonical.k_norm(key.view(-1, kv_heads, dim)), *position_embeddings)
            q.copy_(rotated_query)
            normalized_key.copy_(rotated_key)
        m = metadata
        page = m.token_page_table.gather(1, (m.positions.clamp(min=0) // PAGE_TOKENS)[:, None]).squeeze(1)
        slots = torch.where(m.positions >= 0, page.long() * PAGE_TOKENS + m.positions % PAGE_TOKENS, -1)
        lengths = torch.where(m.lengths > 0, m.starts + m.lengths, 0).to(m.lengths.dtype)
        self.writer(normalized_key, value.view(tokens, kv_heads, dim), slots)
        return q, gate, lengths

    def forward(self, hidden_states, position_embeddings, *, metadata, **kwargs):
        query, gate, lengths = self.write_cache(hidden_states, position_embeddings, metadata)
        attended = self.attention(query, output_gate=gate if self.attention.gated else None,
                                  page_table=metadata.page_table, cu_seqlens_q=metadata.query_start_loc,
                                  cache_seqlens=lengths)
        if not self.attention.gated:
            if self.output_gate is None:
                attended.mul_(gate.reshape_as(attended).sigmoid())
            else:
                values = (attended.view(self.tokens, self.heads * self.dim, 1) if self.packed
                          else attended.view(self.tokens * self.heads, self.dim, 1))
                gates = (gate.view(self.tokens, self.heads * self.dim) if self.packed
                         else gate.view(self.tokens * self.heads, self.dim))
                self.output_gate(values, gates)
        return self.projections["o_proj"](attended.flatten(1)), None


class PreparedDecoder:
    def __init__(self, weights, scratch, metadata, pool, *, mode, vocabulary_runtime,
                 coordinator, session, name, workspace, scoring=None, records=None):
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        self.embedding = PreparedEmbedding(weights.tree.embed_tokens.weight, metadata.token_ids,
            scratch.activations, num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding").functional(session=session, dynamic_inputs=True)
        self.delta = DeltaMetadata.allocate(metadata.starts.numel(), scratch.activations.device,
            verification_columns=metadata.max_query_tokens if mode == "verify" else 1)
        config, source = weights.config, scratch.activations
        def zeros(shape, dtype=None):
            dtype = source.dtype if dtype is None else dtype
            return torch.zeros(shape, dtype=dtype, device=source.device)
        query_dtype = _attention_query_dtype(source, metadata, mode, config.text["head_dim"])
        attention_buffers = {field: zeros((source.shape[0], width), query_dtype if field == "q" else None)
                             for field, width in (
            ("q", config.text["num_attention_heads"] * config.text["head_dim"]),
            ("attended", config.text["num_attention_heads"] * config.text["head_dim"]),
            ("key", config.text["num_key_value_heads"] * config.text["head_dim"]), ("output", config.hidden))}
        attention_prefixes = tuple(f"layers.{i}.self_attn." for i, kind in enumerate(config.layers)
                                   if kind == "full_attention")
        packed = tuple(prefix + "qgkv_proj" in weights.projections for prefix in attention_prefixes)
        q_width = config.text["num_attention_heads"] * config.text["head_dim"]
        kv_width = config.text["num_key_value_heads"] * config.text["head_dim"]
        if any(packed):
            attention_buffers["qgkv"] = zeros((source.shape[0], 2 * q_width + 2 * kv_width))
        if not all(packed):
            attention_buffers["qg"] = zeros((source.shape[0], 2 * q_width))
        separate = tuple(prefix for prefix, is_packed in zip(attention_prefixes, packed, strict=True) if not is_packed)
        if any(prefix + "kv_proj" in weights.projections for prefix in separate):
            attention_buffers["kv"] = zeros((source.shape[0], 2 * kv_width))
        if any(prefix + "kv_proj" not in weights.projections for prefix in separate):
            for field in ("k", "v"):
                attention_buffers[field] = zeros((source.shape[0], kv_width))
        intermediate = config.text["intermediate_size"]
        def mlp_candidate(width):
            return torch.empty((source.shape[0], width), dtype=torch.bfloat16, device=source.device)
        mlp_buffers = {"down_proj": mlp_candidate(config.hidden)}
        grouped = tuple(f"layers.{i}.mlp.gate_up" in weights.projections for i in range(len(weights.tree.layers)))
        if any(grouped):
            mlp_buffers["gate_up"] = mlp_candidate(2 * intermediate)
        if not all(grouped):
            for field in ("gate_proj", "up_proj"):
                mlp_buffers[field] = mlp_candidate(intermediate)
        mlp_input = _allocate_mlp_intermediate(weights, source)
        normalized = mlp_candidate(config.hidden)
        self.layers = []
        previous_attended = None
        for index, original in enumerate(weights.tree.layers):
            layer = module_copy(original)
            prefix = f"layers.{index}."
            layer.input_layernorm = RMSNorm(original.input_layernorm,
                mlp_buffers["down_proj"] if index else source, normalized,
                session=session, name=name + "." + prefix + "input_layernorm",
                residual=previous_attended if index else None)
            if original.block_type == "linear_attention":
                inputs = (("in_proj_qkvzba",) if prefix + "linear_attn.in_proj_qkvzba" in weights.projections
                          else ("in_proj_qkvz", "in_proj_ba") if prefix + "linear_attn.in_proj_qkvz" in weights.projections
                          else ("in_proj_qkv", "in_proj_z", "in_proj_ba"))
                projections = {field: weights.projections[prefix + "linear_attn." + field]
                               for field in (*inputs, "out_proj")}
                layer.linear_attn = RecurrentAttention(PreparedGDN(original.linear_attn, projections, source,
                    scratch.gdn, pool.recurrent(index), self.delta,
                    mode="decode" if mode == "verify" else mode, session=session,
                    records=records[index] if mode == "verify" else None,
                    name=name + "." + prefix + "gdn", workspace=workspace, projection_workspace=workspace,
                    projection_staging=scratch.quantized,
                    gdn_buffers=None).functional(session=session, dynamic_metadata=True))
            else:
                layer.self_attn = FullAttention(original.self_attn, weights, source, metadata, pool, index,
                    mode=mode, session=session, workspace=workspace, name=name + "." + prefix + "attention",
                    buffers=attention_buffers, projection_staging=scratch.quantized)
            attended = (scratch.gdn.output if original.block_type == "linear_attention"
                        else attention_buffers["output"])
            layer.post_attention_layernorm = RMSNorm(original.post_attention_layernorm,
                attended, normalized, session=session,
                name=name + "." + prefix + "post_attention_layernorm",
                residual=source)
            if grouped[index]:
                layer.mlp = GroupedMLP(original.mlp, source,
                    weights.projections[prefix + "mlp.gate_up"], weights.projections[prefix + "mlp.down_proj"],
                    mlp_buffers["gate_up"], mlp_input, mlp_buffers["down_proj"], session=session,
                    name=name + "." + prefix + "mlp", workspace=workspace)
            else:
                layer.mlp = module_copy(original.mlp)
                for field in ("gate_proj", "up_proj", "down_proj"):
                    weight = weights.projections[prefix + "mlp." + field]
                    inputs = source if field != "down_proj" else mlp_input
                    output = mlp_buffers[field]
                    setattr(layer.mlp, field, Projection(prepare_projection(inputs, weight, output,
                        activation_mode="auto", session=session, workspace=workspace,
                        name=name + "." + prefix + field).functional(session=session)))
            self.layers.append(ResidualNormalizedLayer(layer))
            previous_attended = attended
        for left, right in zip(self.layers, self.layers[1:]):
            left.next_input_layernorm = right.layer.input_layernorm
        rotary = config.text["rope_parameters"]
        dim = int(config.text["head_dim"] * rotary["partial_rotary_factor"])
        from ignition.preparation.persistent import persistent_copy, persistent_empty

        self.inverse_frequency = persistent_copy(1.0 / (rotary["rope_theta"] ** (
            torch.arange(0, dim, 2, device=source.device, dtype=torch.float32) / dim)))
        self.scoring = None if scoring is None else scoring.prepare(scratch.activations, weights.head,
            vocabulary=weights.config.text["vocab_size"], session=session, coordinator=coordinator,
            name=name + ".prompt", functional=True)
        self.head = PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head", activation_mode="auto", workspace=workspace).functional(session=session)
        self.context_capture = (PreparedContextCapture(metadata,
            config.hidden * len(config.draft.target_layers), config.draft.attention.window,
            compact_prefill=mode == "prefill").functional() if weights.draft is not None else None)
        self.auxiliary = (None if self.context_capture is None else torch.empty(
            self.context_capture.shape, dtype=torch.bfloat16, device=source.device))
        self.replay = None
        self.record_layers = ()
        if mode == "verify":
            self.record_layers = records.layers
            self.record_rows, self.record_width = records.delta.shape[1], records.delta.shape[3]
            counts = torch.zeros(metadata.starts.numel(), dtype=torch.int32, device=source.device)
            states = {i: pool.region(f"layer{i}.recurrent") for i in records.layers}
            self.replay = PreparedGDNCommit(pool.backing, states, records, self.delta, counts,
                key_heads=16, value_heads=48).functional(session=session, dynamic_metadata=True)
            regions = [pool.layout.region(f"layer{i}.convolution") for i in records.layers]
            self.convolution_offsets = persistent_empty((len(regions),), torch.int64, source.device)
            self.convolution_offsets.copy_(torch.tensor([r.offset // 2 for r in regions], dtype=torch.int64))
            self.convolution_shape = regions[0].shape
        self.verification_columns = metadata.max_query_tokens if mode == "verify" else 1
        self.metadata = self.delta = None

    def release_outputs(self):
        self.scratch = self.auxiliary = None

    def _delta_metadata(self, metadata):
        sequences = metadata.starts.numel()
        active = metadata.lengths > 0
        state_indices = torch.zeros((sequences, self.verification_columns), dtype=torch.int64,
                                    device=metadata.starts.device)
        state_indices[:, 0].copy_(metadata.state_rows)
        return DeltaMetadata(
            metadata.query_start_loc,
            torch.where(active, metadata.state_rows, -1),
            metadata.starts == 0,
            torch.ones((sequences,), dtype=torch.int32, device=metadata.starts.device),
            state_indices,
            torch.where(metadata.starts > 0, metadata.state_rows, 0),
            metadata.state_rows,
            torch.zeros((sequences,), dtype=torch.int64, device=metadata.starts.device),
            torch.zeros((sequences,), dtype=torch.int32, device=metadata.starts.device),
            metadata.live_counts[1:], metadata.live_counts[:1],
        )

    def commit(self, counts, records, metadata=None):
        if metadata is None:
            raise ValueError("dense decoder commit requires metadata")
        counts = counts.to(torch.int32)
        self.replay(counts, records=records, metadata=self._delta_metadata(metadata))
        channels, columns = self.convolution_shape
        torch.library.wrap_triton(_commit_convolution)[
            (self.convolution_offsets.numel(), counts.numel(), triton.cdiv(channels, 128))](
            self.pool.backing.view(torch.bfloat16), self.convolution_offsets, metadata.state_rows, counts,
            self.pool.layout.row_bytes // 2, channels, columns, 128)
        return counts

    def __call__(self, metadata=None):
        if metadata is None:
            raise ValueError("dense decoder requires metadata")
        m, d = metadata, self._delta_metadata(metadata)
        frequency = m.positions.float()[:, None] * self.inverse_frequency[None, :]
        embedding = torch.cat((frequency, frequency), dim=-1)
        position_embeddings = embedding.cos().bfloat16(), embedding.sin().bfloat16()
        hidden = self.embedding(token_ids=m.token_ids, num_rows=m.live_counts[:1])
        records = (GDNLayerRecords.allocate(self.record_layers, self.record_rows, hidden.device,
            key_heads=16, value_heads=48, width=self.record_width) if self.record_layers else None)
        if self.context_capture is not None:
            capture = self.context_capture.begin(m)
        taps = () if self.context_capture is None else self.weights.config.draft.target_layers
        normalized = self.layers[0].layer.input_layernorm(hidden)
        for index, layer in enumerate(self.layers):
            layer_records = None if records is None or index not in records.layers else records[index]
            hidden, normalized = layer(hidden, normalized, position_embeddings=position_embeddings, records=layer_records,
                                       metadata=m, delta=d)
            if index in taps:
                column = taps.index(index) * hidden.shape[-1]
                self.context_capture(capture, hidden, column)
        if self.scoring is not None:
            normalized_hidden = self.weights.tree.norm(hidden)
            self.scoring(normalized_hidden)
            sampled_hidden = normalized_hidden[m.sample_indices]
        else:
            sampled_hidden = self.weights.tree.norm(hidden[m.sample_indices])
        return sampled_hidden, self.head(sampled_hidden), None if self.context_capture is None else capture[0], records
