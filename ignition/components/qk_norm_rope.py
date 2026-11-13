# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared packed Q/K normalization, rotary embedding, and optional KV append."""

from dataclasses import dataclass, replace

import torch
import triton
from b12x._lib.compile_plan import (
    attach_programs,
    compile_only_launches,
    launch_triton,
    load_programs,
)
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.attention.qsa._kernels import _prepare_index_query_kernel
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.tuning import benchmark_resource, declare, register_finalizer

from .native_operator import native_contract, native_plan
from .selected_attention import metadata_tensor, tensor_facts


@dataclass(frozen=True)
class Query:
    operands: tuple
    epsilon: float
    key_layer_norm: bool
    append: bool
    copy_value: bool


CONTRACT = native_contract(component_id="flare.qk_norm_rope", query_type=Query, tactics=lambda q: (4, 1, 2, 8))


def _arguments(tensors, q):
    projected, qw, kw, bias, inverse, requests, positions, query, key, value, kc, vc, table = tensors
    heads, dim = query.shape[1:]
    groups = key.shape[1]
    return (projected, requests, qw, positions, inverse, inverse, query, 0, q.epsilon,
            positions.stride(0), 0, 0, 0, projected.stride(0)), {
        "INDEX_HEADS": heads + groups, "HEAD_DIM": dim, "ROTARY_DIM": 2 * inverse.numel(),
        "POSITION_AXES": 1, "MROPE_INTERLEAVED": False, "MROPE_SECTION_0": 0, "MROPE_SECTION_1": 0,
        "ROPE_IS_BF16": False, "BLOCK_D": triton.next_power_of_2(dim),
        "key_weight": kw, "key_bias": bias, "key_output": key, "value_output": value,
        "key_cache": kc, "value_cache": vc, "page_table": table,
        "cache_page_stride": kc.stride(0), "table_stride": table.stride(0),
        "PAGE_SIZE": kc.shape[1], "KEY_HEADS": groups, "KEY_LAYER_NORM": q.key_layer_norm,
        "COPY_VALUE": q.copy_value, "APPEND_CACHE": q.append, "INVERSE_ROPE": True, "ROUND_OUTPUT": True}


def _launch(tensors, q, warps):
    args, options = _arguments(tensors, q)
    query, key = tensors[7:9]
    positional = ("INDEX_HEADS", "HEAD_DIM", "ROTARY_DIM", "POSITION_AXES", "MROPE_INTERLEAVED",
                  "MROPE_SECTION_0", "MROPE_SECTION_1", "ROPE_IS_BF16", "BLOCK_D", "key_weight",
                  "key_bias", "key_output", "value_output", "key_cache", "value_cache", "page_table")
    args += tuple(options.pop(name) for name in positional)
    return launch_triton(_prepare_index_query_kernel, (query.shape[0] * (query.shape[1] + key.shape[1]),),
                        *args, **options, num_warps=warps, enable_fp_fusion=False)


@program_cache(scope="preparation")
def compile_norm(payload, warps, ordinal):
    q = Query(**dict(payload))
    with torch.cuda.device(ordinal), compile_only_launches():
        return _launch(tuple(map(metadata_tensor, q.operands)), q, warps)


@dataclass(frozen=True)
class State:
    query: Query
    warps: int
    program: object

    def run(self, tensors):
        args, options = _arguments(tensors, self.query)
        # Triton's launcher keeps constexpr argument positions before runtime pointers.
        query, key = tensors[7:9]
        self.program[(query.shape[0] * (query.shape[1] + key.shape[1]), 1, 1)](*args, *options.values())


@torch.library.custom_op("flare::qk_norm_rope", mutates_args=("query", "key", "value", "key_cache", "value_cache"))
def _execute(projected: torch.Tensor, qw: torch.Tensor, kw: torch.Tensor, bias: torch.Tensor,
             inverse: torch.Tensor, requests: torch.Tensor, positions: torch.Tensor, query: torch.Tensor,
             key: torch.Tensor, value: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
             table: torch.Tensor, handle: int) -> None:
    plan = plan_from_handle(handle)
    state = require_prepared(plan, plan.component_id, projected.device)
    state.run((projected, qw, kw, bias, inverse, requests, positions, query, key, value, key_cache, value_cache, table))


@_execute.register_fake
def _fake(projected, qw, kw, bias, inverse, requests, positions, query, key, value, key_cache, value_cache, table, handle):
    pass


class PreparedQKNormRoPE:
    def __init__(self, projected, qw, kw, inverse, metadata, query, key, *, epsilon, session, name,
                 value=None, key_cache=None, value_cache=None, bias=None):
        self.output = query
        if (projected.ndim != 2 or projected.stride(1) != 1 or projected.dtype != torch.bfloat16
                or query.ndim != 3 or key.ndim != 3 or query.shape[0] != projected.shape[0]
                or query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]
                or not query.is_contiguous() or not key.is_contiguous()
                or query.dtype not in (torch.bfloat16, torch.float8_e4m3fn) or key.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
                or qw.shape != query.shape[-1:] or kw.shape != qw.shape
                or any(w.dtype not in (torch.bfloat16, torch.float32) or not w.is_contiguous() for w in (qw, kw))
                or inverse.dtype != torch.float32 or not 0 < 2 * inverse.numel() <= query.shape[-1]
                or not inverse.is_contiguous()):
            raise ValueError("packed Q/K normalization requires matching contiguous heads and rotary geometry")
        if bias is not None and (bias.shape != kw.shape or bias.dtype != kw.dtype or not bias.is_contiguous()):
            raise ValueError("key LayerNorm requires matching contiguous affine parameters")
        append, copy_value = key_cache is not None, value is not None
        if projected.shape[1] < (query.shape[1] + key.shape[1] * (2 if copy_value else 1)) * query.shape[-1]:
            raise ValueError("packed projection does not contain the declared Q/K/V heads")
        if value is not None and (value.shape != key.shape or value.dtype != key.dtype or not value.is_contiguous()):
            raise ValueError("value output must have contiguous key-head geometry and dtype")
        if (value_cache is not None) != append:
            raise ValueError("key and value caches must be supplied together")
        if append and (not copy_value or key_cache.ndim != 4 or key_cache.shape != value_cache.shape
                or key_cache.shape[2:] != key.shape[1:] or key_cache.stride() != value_cache.stride()
                or key_cache.stride()[1:] != (key.shape[1] * key.shape[2], key.shape[2], 1)
                or key_cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn) or value_cache.dtype != key_cache.dtype):
            raise ValueError("KV append requires matching BF16 or FP8 pages with contiguous token/head dimensions")
        # Unused output operands remain disjoint to preserve custom-op alias contracts.
        self.empty = torch.empty(0, dtype=torch.bfloat16, device=projected.device)
        if value is None:
            value = self.empty
        if key_cache is None:
            key_cache = torch.empty((0, 128, key.shape[1], key.shape[2]), dtype=torch.bfloat16, device=key.device)
            value_cache = torch.empty_like(key_cache)
        self.arguments = (projected, qw, kw, qw if bias is None else bias, inverse, metadata.requests,
                          metadata.positions, query, key, value, key_cache, value_cache, metadata.page_table)
        q = Query(tuple(tensor_facts(t) for t in self.arguments), float(epsilon), bias is not None, append, copy_value)
        payload = CONTRACT.encode_query(q)

        def materialize(warps):
            program = compile_norm(payload, warps, projected.device.index)
            load_programs(program)
            return attach_programs(State(q, warps, program), program)

        self.plan = native_plan(contract=CONTRACT, query=q, device=projected.device,
            scratch_bytes=lambda t: 0, materialize=materialize,
            compile_jobs=lambda t, d: (CompileJob.create(
                "ignition.components.qk_norm_rope:compile_norm", payload, t, d.ordinal),))

        def benchmark(state):
            def inputs():
                tensors = []
                for i, tensor in enumerate(self.arguments):
                    if i in (1, 2, 3, 4):
                        tensors.append(tensor)
                        continue
                    shape = tensor.shape
                    # Benchmark metadata addresses only page zero, even for virtual caches.
                    if i in (10, 11):
                        shape = (min(1, shape[0]), *shape[1:])
                    tensors.append(torch.empty_strided(shape, tensor.stride(),
                        dtype=tensor.dtype, device=tensor.device))
                tensors = tuple(tensors)
                tensors[0].normal_(std=.2)
                tensors[5].zero_()
                tensors[6].zero_()
                tensors[-1].zero_()
                return tensors, tensors[0].clone()
            tensors, seed = benchmark_resource(session, ("qk_norm_rope", name), inputs)
            return PreparedCall(run=lambda: state.run(tensors), output=tensors[7],
                                produce=lambda: tensors[0].copy_(seed), owners=(tensors, seed))
        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: replace(benchmark(state), owners=()),
            benchmark_call=benchmark),))

    def functional(self, *, session):
        register_finalizer(session, self._finalize_functional)
        return self

    def _finalize_functional(self):
        args = self.arguments
        self.constants = args[1:5] + args[10:12]
        self.layouts = tuple((tuple(t.shape), tuple(t.stride()), t.dtype) for t in args[7:10])
        self.copy_value = bool(args[9].numel())
        self.arguments = self.output = self.empty = None

    def __call__(self, projected=None, metadata=None):
        if self.arguments is not None:
            _execute(*self.arguments, self.plan.handle)
            return self.output
        qw, kw, bias, inverse, key_cache, value_cache = self.constants
        query, key, value = (torch.empty_strided(shape, stride, dtype=dtype, device=projected.device)
                             for shape, stride, dtype in self.layouts)
        _execute(projected, qw, kw, bias, inverse, metadata.requests, metadata.positions,
                 query, key, value, key_cache, value_cache, metadata.page_table, self.plan.handle)
        return query, key, value if self.copy_value else None
