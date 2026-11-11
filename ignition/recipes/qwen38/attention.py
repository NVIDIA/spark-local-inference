# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared Qwen sparse attention with paged KV and selector state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial

import torch
from b12x.attention import paged_kv_write, qsa
from b12x.preparation import PreparationSession, PreparedCall
from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

from ignition.components import sm103
from ignition.components.sm103_sparse_attention import PARTIAL_ROWS, SPLIT_ROWS, PreparedSparseAttention
from ignition.components.linear import QuantizedProjection, UnquantizedProjection
from ignition.preparation.fixtures import qsa as qsa_fixtures
from ignition.preparation.scratch import ScratchArena, planned_scratch, prime_owners
from ignition.preparation.tuning import declare, register_finalizer

from .attention_overlap import ConcurrentAttention, FunctionalQSA
from ignition.preparation.persistent import persistent_copy, persistent_empty, persistent_zeros


# b12x attends at most this many rows with its split schedule: it splits each row's selected
# positions across programs, which verification rounds (at most batch x width rows) need. Larger
# prefill chunks use the row-per-program Triton kernel; at 128 rows it would leave the GPU nearly idle.
_SPARSE_ATTENTION_MIN_ROWS = 512


@dataclass(frozen=True)
class RotaryTables:
    cos: torch.Tensor
    sin: torch.Tensor
    axes: torch.Tensor

    @classmethod
    def allocate(cls, config, capacity: int, device: torch.device) -> RotaryTables:
        if capacity <= 0:
            raise ValueError("rotary position capacity must be positive")
        rope = config.rope_parameters
        if (config.head_dim != 256 or rope["partial_rotary_factor"] != .25
                or tuple(rope["mrope_section"]) != (11, 11, 10)
                or rope["rope_theta"] != 10000000):
            raise ValueError("Qwen rotary geometry differs from the checkpoint contract")
        inverse = (1.0 / (rope["rope_theta"] ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))).to(device)
        frequency = torch.arange(capacity, device=device, dtype=torch.float32)[:, None] * inverse
        axes = persistent_zeros((32,), dtype=torch.int64, device=device)
        axes[1:33:3] = 1
        axes[2:30:3] = 2
        cos = persistent_empty(frequency.shape, dtype=torch.float32, device=device)
        sin = persistent_empty(frequency.shape, dtype=torch.float32, device=device)
        torch.cos(frequency, out=cos)
        torch.sin(frequency, out=sin)
        return cls(cos, sin, axes)

    def gather(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Negative metadata marks inactive rows; their arithmetic uses position zero.
        rows = positions.clamp_min(0).index_select(1, self.axes)
        columns = torch.arange(32, device=positions.device)
        cos, sin = self.cos[rows, columns], self.sin[rows, columns]
        return torch.cat((cos, cos), -1).bfloat16(), torch.cat((sin, sin), -1).bfloat16()


@dataclass(frozen=True)
class QSAState:
    main_k_cache: torch.Tensor
    main_v_cache: torch.Tensor
    compressed_k_cache: torch.Tensor
    raw_k_ring: torch.Tensor
    raw_logical_positions: torch.Tensor
    raw_rope_positions: torch.Tensor
    raw_interval_start_positions: torch.Tensor

    @classmethod
    def allocate(cls, *, pages: int, slots: int, device: torch.device,
                 page_size: int = 64, draft_tokens: int = 0, kv_heads: int = 2) -> QSAState:
        if pages <= 0 or slots <= 0:
            raise ValueError("QSA page and state capacities must be positive")
        geometry = qsa.cache_requirements(main_page_size=page_size, position_axes=3, kv_heads=kv_heads,
                                         max_speculative_tokens=draft_tokens, kv_dtype=torch.float8_e4m3fn)
        def zeros(shape, dtype=torch.bfloat16):
            return torch.zeros(shape, device=device, dtype=dtype)
        def invalid(shape):
            return torch.full(shape, -1, device=device, dtype=torch.int64)
        return cls(zeros((pages, *geometry.main_k_page_shape), geometry.kv_dtype),
                   zeros((pages, *geometry.main_v_page_shape), geometry.kv_dtype),
                   zeros((pages, *geometry.compressed_page_shape)),
                   zeros((slots, *geometry.raw_k_ring_shape)),
                   invalid((slots, *geometry.raw_logical_positions_shape)),
                   invalid((slots, *geometry.raw_rope_positions_shape)), invalid((slots,)))

    def resize_pages_for_preparation(self, pages: int, *, metadata_mode=None, storage=None, prefix="", backed_pages=None) -> QSAState:
        """Allocate final page pools while retaining fixed raw selector slots.

        No graph referencing the returned page pools may have been handed to
        native execution. Rebind each prepared attention operator and recapture
        its graphs before use. Page contents are empty after this operation.
        """
        if type(pages) is not int or pages <= 0:
            raise ValueError("QSA page capacity must be positive")
        def pages_like(value, name):
            if metadata_mode is not None:
                from ignition.preparation.deferred import deferred_tensor

                return deferred_tensor((pages, *value.shape[1:]), dtype=value.dtype,
                                       device=value.device, mode=metadata_mode)
            if storage is not None:
                return storage.device_allocate(prefix + name, (pages, *value.shape[1:]), value.dtype,
                    group="KV_PAGES", maximum=pages, units=backed_pages)
            return torch.zeros((pages, *value.shape[1:]), dtype=value.dtype, device=value.device)
        return type(self)(pages_like(self.main_k_cache, "key"), pages_like(self.main_v_cache, "value"),
            pages_like(self.compressed_k_cache, "selector"), self.raw_k_ring, self.raw_logical_positions,
            self.raw_rope_positions, self.raw_interval_start_positions)


@dataclass(frozen=True)
class QSAMetadata:
    main_block_table: torch.Tensor
    compressed_block_table: torch.Tensor
    raw_state_slot_ids: torch.Tensor
    request_ids: torch.Tensor
    query_positions: torch.Tensor
    rope_positions: torch.Tensor
    sequence_lengths: torch.Tensor
    query_start_loc: torch.Tensor
    num_accepted_tokens: torch.Tensor
    is_prefilling: torch.Tensor
    slot_mapping: torch.Tensor

    @classmethod
    def allocate(cls, *, tokens: int, batch: int, context: int,
                 device: torch.device, page_size: int = 64) -> QSAMetadata:
        if tokens < batch or min(batch, context) <= 0 or context % page_size:
            raise ValueError("QSA requires positive, page-aligned capacity and tokens >= batch")
        def full(shape, value=-1, dtype=torch.int64):
            return torch.full(shape, value, dtype=dtype, device=device)
        return cls(full((batch, context // page_size), dtype=torch.int32),
                   full((batch, context // page_size), dtype=torch.int32), full((batch,)),
                   full((tokens,)), full((tokens,)), full((tokens, 3)),
                   full((batch,), 0, torch.int32), full((batch + 1,), 0, torch.int32),
                   full((batch,), 0, torch.int32), full((batch,), False, torch.bool), full((tokens,)))


@dataclass(frozen=True)
class QSAScratch:
    projected_q: torch.Tensor
    projected_k: torch.Tensor
    projected_v: torch.Tensor
    projected_index: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    attended: torch.Tensor
    selected_positions: torch.Tensor
    gated: torch.Tensor
    output: torch.Tensor
    projected_qkv: torch.Tensor
    # Split attention's FP32 partial rows; every row view keeps them whole.
    partial: torch.Tensor
    partial_lse: torch.Tensor

    @classmethod
    def allocate(cls, tokens: int, device: torch.device, *, world_size=1, output=None) -> QSAScratch:
        if tokens <= 0:
            raise ValueError("QSA token capacity must be positive")
        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, device=device, dtype=dtype)
        q_heads, kv_heads = 24 // world_size, max(1, 2 // world_size)
        q_width, kv_width = q_heads * 512, kv_heads * 256
        qkv = zeros(tokens, q_width + 2 * kv_width)
        partial_rows = PARTIAL_ROWS if sm103.is_sm103(device) else 0
        return cls(qkv[:, :q_width], qkv[:, q_width:q_width + kv_width], qkv[:, q_width + kv_width:],
                   zeros(tokens, 640), zeros(tokens, q_heads, 256), zeros(tokens, kv_heads, 256),
                   zeros(tokens, q_heads, 256),
                   zeros(tokens, 2051, dtype=torch.int32).fill_(-1),
                   zeros(tokens, q_heads * 256), zeros(tokens, 2560) if output is None else output, qkv,
                   zeros(partial_rows, q_heads, 256, dtype=torch.float32),
                   zeros(partial_rows, q_heads, dtype=torch.float32))

    def rows(self, tokens: int) -> QSAScratch:
        whole = ("partial", "partial_lse")
        return QSAScratch(*(getattr(self, field)[:None if field in whole else tokens]
                            for field in self.__dataclass_fields__))


class PreparedAttention:
    """All Python tensor composition executes during warmup and graph capture.

    Metadata begins inactive during construction. The scheduler supplies dense
    request intervals, page mappings, and accepted-token counts before replay.
    It initializes each raw interval anchor to consumed length minus accepted
    count on restore and invalidates raw tags at aligned checkpoints.
    """

    def __init__(self, module: torch.nn.Module, weights: Mapping, source: torch.Tensor,
                 scratch: QSAScratch, state: QSAState, metadata: QSAMetadata,
                 rotary: RotaryTables, *, session: PreparationSession, name: str,
                 draft_tokens: int = 0, workspace: ScratchArena | torch.Tensor | None = None,
                 projection_workspace: ScratchArena | torch.Tensor | None = None,
                 max_cache_pages: int | None = None, cache_only: bool = False) -> None:
        self.q_heads, self.kv_heads = scratch.query.shape[1], scratch.key.shape[1]
        self.cache_only = cache_only
        self.module, self.scratch, self.state = module, scratch, state
        self.metadata, self.rotary = metadata, rotary
        self.tokens = source.shape[0]
        if source.shape != scratch.output.shape or source.dtype != torch.bfloat16:
            raise ValueError("QSA source must match BF16 [token_capacity, 2560] output")
        if bool((metadata.request_ids != -1).any().item()) or bool((metadata.slot_mapping != -1).any().item()):
            raise ValueError("QSA preparation requires inactive request and cache-write rows")
        batch, table_width = metadata.main_block_table.shape
        page_size = state.main_k_cache.shape[1]
        context = table_width * page_size
        cache_pages = state.main_k_cache.shape[0] if max_cache_pages is None else max_cache_pages
        if type(cache_pages) is not int or cache_pages < state.main_k_cache.shape[0]:
            raise ValueError("planned QSA page capacity must cover the bound page pools")
        self.cache_page_capacity = cache_pages
        # On SM103 one score workspace covers every representative group of a
        # prefill chunk, so selection makes a single pass instead of one per 128 MiB.
        score_workspace = self.tokens * (context // 4) * 4 if sm103.is_sm103(source.device) else None
        caps = qsa.Caps(device=source.device, max_batch=batch, kv_dtype=state.main_k_cache.dtype,
                        max_raw_state_slots=state.raw_k_ring.shape[0], max_q_rows=self.tokens,
                        max_seq_len=context, num_main_cache_pages=cache_pages,
                        num_compressed_cache_pages=cache_pages,
                        main_page_size=page_size, compressed_page_size=page_size // 4,
                        max_speculative_tokens=draft_tokens, position_axes=3,
                        q_heads=self.q_heads, kv_heads=self.kv_heads,
                        mrope_sections=(11, 11, 10), mrope_interleaved=True,
                        rms_norm_eps=module.config.rms_norm_eps,
                        score_workspace_bytes=max(score_workspace, 128 << 20) if score_workspace else None)
        if rotary.cos.shape[0] < context:
            raise ValueError("rotary table capacity must cover the admitted context")
        self.projections = {}
        def projection(source, weight, output, key):
            if isinstance(weight, torch.Tensor):
                return UnquantizedProjection(source, weight, output, session=session, name=f"{name}.{key}")
            return QuantizedProjection(source, weight, output, session=session,
                                       name=f"{name}.{key}", activation_mode="auto", workspace=projection_workspace)

        for key, output in (("qkv_proj", scratch.projected_qkv),
                            ("indexer.index_qk_proj", scratch.projected_index)):
            self.projections[key] = projection(source, weights[key], output, key)
        if not cache_only:
            self.projections["o_proj"] = projection(scratch.gated, weights["o_proj"], scratch.output, "o_proj")
        self.index_query = scratch.projected_index[:, :512].view(self.tokens, 4, 128)
        self.index_key = scratch.projected_index[:, 512:]
        self.binding_kwargs = {
            **{field: getattr(state, field) for field in state.__dataclass_fields__},
            **{field: getattr(metadata, field) for field in
               ("main_block_table", "compressed_block_table", "raw_state_slot_ids")},
            "index_q_norm_weight": persistent_copy(module.indexer.q_layernorm.weight.detach().float()),
            "index_k_norm_weight": persistent_copy(module.indexer.k_layernorm.weight.detach().float()),
            "rope_cos": rotary.cos, "rope_sin": rotary.sin,
        }
        operands = {**self.binding_kwargs, "request_ids": metadata.request_ids,
                    "rope_positions": metadata.rope_positions, "index_query": self.index_query,
                    "raw_index_key": self.index_key}
        self.plan = qsa.plan(caps, invocation=qsa.invocation_from_tensors(caps, **operands))
        self.binding_kwargs.update(output=scratch.attended, selected_positions=scratch.selected_positions,
            k_descale=persistent_copy(torch.ones(1, dtype=torch.float32, device=source.device)),
            v_descale=persistent_copy(torch.ones(1, dtype=torch.float32, device=source.device)))
        self.workspace = workspace

        def prepare(materialized):
            storage = planned_scratch(materialized.scratch_specs(), workspace)
            binding = materialized.bind_for_preparation(scratch=storage, **self.binding_kwargs)
            self.workspace = storage
            return PreparedCall(run=lambda: materialized.run_for_preparation(binding, **self.dynamic()),
                                owners=prime_owners(workspace, storage, binding))

        benchmark_inputs = qsa_fixtures.input_factory(caps, self.binding_kwargs, self.dynamic(),
            storage=getattr(session, "benchmark_storage", None), session=session)
        declare(session, (self.plan.request(name=f"{name}.qsa", prepare_call=prepare,
            benchmark_call=partial(qsa_fixtures.benchmark_call, inputs=benchmark_inputs, workspace=workspace)),))
        def refresh():
            self.workspace = planned_scratch(self.plan.scratch_specs(), workspace)
            self.binding = qsa.bind(self.plan, scratch=self.workspace, **self.binding_kwargs)

        register_finalizer(session, refresh, workspace=workspace)
        self.writer = paged_kv_write.plan(paged_kv_write.Caps(device=source.device,
            num_kv_heads=self.kv_heads, head_dim_k=256, head_dim_v=256, page_size=page_size,
            kv_dtype=state.main_k_cache.dtype))
        self.write_kwargs = {"key": scratch.key, "value": scratch.projected_v.view(self.tokens, self.kv_heads, 256),
                             "k_cache": state.main_k_cache, "v_cache": state.main_v_cache,
                             "slot_mapping": metadata.slot_mapping,
                             "k_scale": self.binding_kwargs["k_descale"],
                             "v_scale": self.binding_kwargs["v_descale"]}
        declare(session, (self.writer.request(name=f"{name}.kv_write", prepare_call=lambda materialized:
            PreparedCall(run=lambda: materialized.run(**self.write_kwargs))),))
        concurrent = (not cache_only and self.tokens <= 16 and source.is_cuda and isinstance(workspace, ScratchArena)
                      and workspace is projection_workspace
                      and all(isinstance(self.projections[key], QuantizedProjection)
                              for key in ("qkv_proj", "indexer.index_qk_proj")))
        # SM103 attends QSA's selection with Triton kernels: prefill chunks in one pass and
        # decode-sized rows split across programs. Rows between, and rows whose selection
        # overlaps QKV production, keep b12x's attention. Without stream overlap those rows
        # run the same kernels serially.
        self.sparse = None
        if (not cache_only and not concurrent and sm103.is_sm103(source.device)
                and (self.tokens > _SPARSE_ATTENTION_MIN_ROWS or self.tokens <= SPLIT_ROWS)):
            self.sparse = PreparedSparseAttention(scratch.query, state.main_k_cache, state.main_v_cache,
                scratch.selected_positions, metadata.main_block_table, metadata.request_ids,
                metadata.query_positions, scratch.attended, partial=scratch.partial,
                partial_lse=scratch.partial_lse)
        self.concurrent = (ConcurrentAttention(self, session=session, workspace=workspace)
                           if concurrent and workspace.overlap else None)

    def dynamic(self) -> dict:
        return {"query": self.scratch.query, "index_query": self.index_query,
                "raw_index_key": self.index_key,
                **{field: getattr(self.metadata, field) for field in
                   ("request_ids", "query_positions", "rope_positions", "sequence_lengths",
                    "query_start_loc", "num_accepted_tokens", "is_prefilling")}}

    def rebind_pages_for_preparation(self, state: QSAState) -> None:
        """Replace only cache page pools before final CUDA graph capture."""
        if state.main_k_cache.shape[0] > self.cache_page_capacity:
            raise ValueError("replacement cache exceeds prepared QSA page capacity")
        for name in ("raw_k_ring", "raw_logical_positions", "raw_rope_positions", "raw_interval_start_positions"):
            if getattr(state, name).data_ptr() != getattr(self.state, name).data_ptr():
                raise ValueError("page rebinding must preserve raw selector state storage")
        for name in ("main_k_cache", "main_v_cache", "compressed_k_cache"):
            before, after = getattr(self.state, name), getattr(state, name)
            if (before.shape[1:] != after.shape[1:] or before.stride() != after.stride()
                    or before.dtype != after.dtype or before.device != after.device):
                raise ValueError("page rebinding must preserve the prepared cache ABI")
            if self.binding_kwargs is None:
                self.constants[name] = after
            else:
                self.binding_kwargs[name] = after
        if self.binding_kwargs is None:
            self.state = state
            return
        self.binding = qsa.bind(self.plan, scratch=self.workspace, **self.binding_kwargs)
        self.write_kwargs.update(k_cache=state.main_k_cache, v_cache=state.main_v_cache)
        self.state = state

    def functional(self, *, session):
        self.functional_qsa = FunctionalQSA(self)
        for projection in self.projections.values():
            projection.functional(session=session)
        if self.concurrent is not None:
            self.concurrent.functional(session=session)
        if self.sparse is not None:
            self.sparse.functional(session=session)
        register_finalizer(session, self._finalize_functional)
        return self

    def _finalize_functional(self):
        self.workspace_bytes = self.workspace.numel()
        self.constants = {name: value for name, value in self.binding_kwargs.items()
            if name not in ("main_block_table", "compressed_block_table", "raw_state_slot_ids",
                            "output", "selected_positions")}
        self.scratch = self.metadata = self.index_query = self.index_key = None
        self.binding = self.binding_kwargs = self.write_kwargs = self.workspace = None

    def __call__(self, source: torch.Tensor, metadata) -> torch.Tensor | None:
        if self.concurrent is None:
            projected_index = self.projections["indexer.index_qk_proj"](source)
            projected_qkv = self.projections["qkv_proj"](source)
            projection_workspaces = None
        else:
            (projected_qkv, projected_index), projection_workspaces = self.concurrent.project_runtime(source)
        q_width = self.q_heads * 512
        kv_width = self.kv_heads * 256
        projected_q = projected_qkv[:, :q_width].view(self.tokens, self.q_heads, 512)
        query, gate = projected_q[..., :256], projected_q[..., 256:]
        key = projected_qkv[:, q_width:q_width + kv_width].view(self.tokens, self.kv_heads, 256)
        value = projected_qkv[:, q_width + kv_width:].view(self.tokens, self.kv_heads, 256)
        cos, sin = self.rotary.gather(metadata.rope_positions)
        query, key = apply_rotary_pos_emb(self.module.q_norm(query), self.module.k_norm(key), cos, sin,
                                         unsqueeze_dim=1)
        query, key = query.contiguous(), key.contiguous()
        paged_kv_write.run(self.writer, key=key, value=value, k_cache=self.state.main_k_cache,
            v_cache=self.state.main_v_cache, slot_mapping=metadata.slot_mapping,
            k_scale=self.constants["k_descale"], v_scale=self.constants["v_descale"])
        attended = torch.empty((self.tokens, self.q_heads, 256), dtype=source.dtype, device=source.device)
        selected = torch.empty((self.tokens, 2051), dtype=torch.int32, device=source.device)
        bindings = {**self.constants, "main_block_table": metadata.main_block_table,
            "compressed_block_table": metadata.compressed_block_table,
            "raw_state_slot_ids": metadata.raw_state_slot_ids, "output": attended,
            "selected_positions": selected}
        dynamic = {"query": query, "index_query": projected_index[:, :512].view(self.tokens, 4, 128),
            "raw_index_key": projected_index[:, 512:], **{name: getattr(metadata, name) for name in
                ("request_ids", "query_positions", "rope_positions", "sequence_lengths", "query_start_loc",
                 "num_accepted_tokens", "is_prefilling")}}
        if self.concurrent is None:
            workspace = torch.empty(self.workspace_bytes, dtype=torch.uint8, device=source.device)
            self.functional_qsa(source, bindings, dynamic, workspace)
        else:
            self.concurrent.attend_runtime(source, bindings, dynamic, projection_workspaces)
        if self.cache_only:
            return None
        if self.sparse is not None:
            attended = self.sparse(self.state.main_k_cache, self.state.main_v_cache, query=query,
                selected=selected, block_table=metadata.main_block_table,
                requests=metadata.request_ids, positions=metadata.query_positions)
        gated = attended.flatten(1) * torch.sigmoid(gate.reshape(self.tokens, self.q_heads * 256))
        output = self.projections["o_proj"](gated)
        output.masked_fill_((metadata.request_ids < 0)[:, None], 0)
        return output
