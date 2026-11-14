# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Routed experts on SM103 (GB300/B300) through FlashInfer's SM100-family MoE kernels.

b12x executes routed experts on SM120/SM121 only. On compute capability 10.3
the same checkpoint storage runs through FlashInfer:

* NVFP4 experts with NVFP4 activations, including CSF-encoded scales, use
  trtllm-gen's routed FP4 MoE. Compressed scales are expanded losslessly once
  during loading and stay resident per layer. FC1 inputs retain each expert's
  checkpoint scale, quantizing separate routed rows when scales differ. FC2 inputs are quantized per
  expert in trtllm-gen's FC1 epilogue with each expert's FC2 input scale.
* W4A16 NVFP4 experts use FlashInfer's CuTe-DSL NVFP4xBF16 grouped GEMMs.

Weights are re-laid out once, in place. Routing remains b12x's portable
``route_topk``. Native kernels allocate their graph-local intermediates from
PyTorch's caching allocator, which the capture's graph pool owns; caller-owned
scratch holds the quantized activations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, partial

import torch
import triton
import triton.language as tl
from b12x._lib.compile_plan import _NATIVE_JITS, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.persistent import persistent_empty, persistent_zeros
from ignition.preparation.scratch import ScratchArena, planned_scratch, prime_owners
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import benchmark_resource, declare, register_finalizer

from .native_operator import native_contract, native_plan

_CHUNK = 64


@cache
def _trtllm():
    from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module

    return get_trtllm_moe_sm100_module().moe_op


@cache
def _fp4_quantizer():
    from flashinfer.jit.fp4_quantization import gen_fp4_quantization_sm103_module

    return gen_fp4_quantization_sm103_module().build_and_load()


def _vector(values, count, device, name):
    values = torch.as_tensor(values).reshape(-1).to("cpu", torch.float64)
    if values.shape != (count,) or not bool((torch.isfinite(values) & (values > 0)).all()):
        raise ValueError(f"{name} must be {count} positive finite values")
    return values


class Sm103Experts:
    """Routed experts prepared for a FlashInfer-native SM103 kernel."""

    has_csf_blocks = False

    @property
    def num_experts(self) -> int:
        return self.w13.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.w2.shape[1]

    @property
    def intermediate_size(self) -> int:
        return self.w2.shape[2] * 2

    @property
    def device(self) -> torch.device:
        return self.w13.device


@dataclass(frozen=True, eq=False)
class TrtllmNvfp4Experts(Sm103Experts):
    """NVFP4 x NVFP4 SwiGLU experts in trtllm-gen's shuffled MajorK layout.

    FC1 rows are ``[up; gate]`` reordered for the gated epilogue; block scales
    follow the same rows and the 128x4 interleave. Scalars follow trtllm-gen:
    ``output1_gate_scale`` dequantizes FC1, ``output1_scale`` additionally
    applies each expert's FC2 input global scale, and ``output2_scale``
    dequantizes FC2.
    """

    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    input_global_scale: torch.Tensor
    output1_scale: torch.Tensor
    output1_gate_scale: torch.Tensor
    output2_scale: torch.Tensor
    gate_alpha: torch.Tensor
    # Clamped SwiGLU (min(gate, limit) and up within +-limit) per expert, in raw FC1 accumulator units
    # as trtllm-gen applies it, or None for plain SwiGLU.
    gate_beta: torch.Tensor | None = None
    clamp_limit: torch.Tensor | None = None
    kind = "trtllm_nvfp4"


@dataclass(frozen=True, eq=False)
class CuteW4A16Experts(Sm103Experts):
    """NVFP4 weights for BF16 activations in FlashInfer's CuTe-DSL grouped GEMMs.

    FC1 rows interleave 64-row ``up`` and ``gate`` groups. Scale storage is the
    128x4-interleaved buffer; the ``*_scale`` fields are its MMA-layout views.
    ``*_alpha`` are the per-expert weight global scales.
    """

    w13: torch.Tensor
    w13_scale: torch.Tensor
    w13_alpha: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    w2_alpha: torch.Tensor
    scale_storage: tuple[torch.Tensor, torch.Tensor]
    # SwiGLU clamp: min(gate, limit) and up within +-limit; None for plain SwiGLU.
    swiglu_limit: float | None = None
    kind = "cute_w4a16"


def _check_packed(w13, s13, w2, s2):
    if w13.dtype != torch.uint8 or w2.dtype != torch.uint8 or w13.ndim != 3 or w2.ndim != 3:
        raise ValueError("packed NVFP4 experts must be rank-three uint8 tensors")
    experts, rows, half = w13.shape
    intermediate, hidden = rows // 2, half * 2
    if rows % 2 or w2.shape != (experts, hidden, intermediate // 2):
        raise ValueError("FC1 must hold [up; gate] rows matching FC2 geometry")
    for scales, shape in ((s13, (experts, rows, hidden // 16)), (s2, (experts, hidden, intermediate // 16))):
        if scales.dtype not in (torch.uint8, torch.float8_e4m3fn) or tuple(scales.shape) != shape:
            raise ValueError("NVFP4 block scales must be row-major E4M3 with one scale per 16 values")
    tensors = (w13, s13, w2, s2)
    if any(t.device != w13.device or not t.is_contiguous() for t in tensors):
        raise ValueError("expert tensors must be contiguous on one CUDA device")
    return experts, hidden, intermediate


def _permute_rows(tensor, permutation):
    """Reorder dimension one of every expert in bounded chunks, in place."""
    for start in range(0, tensor.shape[0], _CHUNK):
        part = tensor[start:start + _CHUNK]
        part.copy_(part.index_select(1, permutation))


def _interleaved_scales(scales, permutation):
    """Row-permuted, 128x4-interleaved E4M3 scales with the source's logical shape."""
    from flashinfer.fp4_quantization import block_scale_interleave

    experts, rows, columns = scales.shape
    output = persistent_empty((experts, rows, columns), dtype=torch.uint8, device=scales.device)
    flat = output.view(experts, -1)
    for start in range(0, experts, _CHUNK):
        part = scales[start:start + _CHUNK].view(torch.uint8)
        if permutation is not None:
            part = part.index_select(1, permutation)
        flat[start:start + part.shape[0]].copy_(
            block_scale_interleave(part.contiguous()).view(part.shape[0], -1))
    return output


def prepare_trtllm_nvfp4_experts(w13, s13, w2, s2, scale_vectors, *, swiglu_limit=None) -> TrtllmNvfp4Experts:
    """Re-lay out packed NVFP4 experts for trtllm-gen; ``w13`` and ``w2`` change in place.

    ``s13``/``s2`` are expanded row-major E4M3 scales. ``scale_vectors`` holds
    the FC1 and FC2 weight global scales followed by the FC1 and FC2
    activation global scales (reciprocal checkpoint input scales). FC1 inputs
    are quantized once for shared scales, or once per route for distinct scales.
    """
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    experts, hidden, intermediate = _check_packed(w13, s13, w2, s2)
    if (2 * intermediate) % 128 or hidden % 128 or (hidden // 16) % 4 or (intermediate // 16) % 4:
        raise ValueError("trtllm-gen NVFP4 experts require 128-row tiles and 4-aligned scale columns")
    if len(scale_vectors) != 4:
        raise ValueError("NVFP4 A4 experts require weight and activation global scales")
    w13_global, w2_global, a1, a2 = (
        _vector(v, experts, w13.device, name) for v, name in zip(scale_vectors, (
            "FC1 weight global scales", "FC2 weight global scales",
            "FC1 activation global scales", "FC2 activation global scales"), strict=True))
    shared_input_scale = bool((a1 == a1[0]).all())
    cache = {}
    w13_rows = _maybe_get_cached_w3_w1_permute_indices(cache, w13[0], 128, is_gated_act_gemm=True)
    s13_rows = _maybe_get_cached_w3_w1_permute_indices(
        cache, s13[0].view(torch.uint8), 128, num_elts_per_sf=16, is_gated_act_gemm=True)
    w2_rows = get_w2_permute_indices_with_cache(cache, w2[0], 128, is_gated_act_gemm=True)
    s2_rows = get_w2_permute_indices_with_cache(
        cache, s2[0].view(torch.uint8), 128, num_elts_per_sf=16, is_gated_act_gemm=True)
    _permute_rows(w13, w13_rows.to(w13.device))
    _permute_rows(w2, w2_rows.to(w2.device))
    scales13 = _interleaved_scales(s13, s13_rows.to(w13.device)).view(torch.float8_e4m3fn)
    scales2 = _interleaved_scales(s2, s2_rows.to(w13.device)).view(torch.float8_e4m3fn)
    gate = w13_global / a1
    device = w13.device
    def as_device(values):
        values = torch.as_tensor(values, dtype=torch.float32)
        return persistent_empty(values.shape, dtype=values.dtype, device=device).copy_(values)
    return TrtllmNvfp4Experts(
        w13=w13, w13_scale=scales13, w2=w2, w2_scale=scales2,
        input_global_scale=as_device(a1[:1] if shared_input_scale else a1),
        output1_scale=as_device(a2 * gate),
        output1_gate_scale=as_device(gate),
        output2_scale=as_device(w2_global / a2),
        gate_alpha=persistent_empty((experts,), dtype=torch.float32, device=device).fill_(1),
        gate_beta=None if swiglu_limit is None else persistent_zeros((experts,), dtype=torch.float32, device=device),
        # trtllm-gen clamps raw FC1 accumulators, before the gate dequantization scale.
        clamp_limit=None if swiglu_limit is None else as_device(float(swiglu_limit) / gate),
    )


def prepare_cute_w4a16_experts(w13, s13, w2, s2, w13_global, w2_global, *, swiglu_limit=None) -> CuteW4A16Experts:
    """Re-lay out packed NVFP4 experts for BF16-activation CuTe-DSL GEMMs, in place."""
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    experts, hidden, intermediate = _check_packed(w13, s13, w2, s2)
    if intermediate % 64 or hidden % 16:
        raise ValueError("CuTe-DSL W4A16 experts require 64-row gate/up groups")
    rows = torch.arange(2 * intermediate, device=w13.device).view(2, intermediate // 64, 64)
    rows = rows.transpose(0, 1).reshape(-1)
    _permute_rows(w13, rows)
    storage13 = _interleaved_scales(s13, rows)
    storage2 = _interleaved_scales(s2, None)
    device = w13.device
    alphas = tuple(persistent_empty((experts,), dtype=torch.float32, device=device).copy_(
                       _vector(v, experts, device, name).to(torch.float32))
                   for v, name in ((w13_global, "FC1 weight global scales"),
                                   (w2_global, "FC2 weight global scales")))
    return CuteW4A16Experts(
        w13=w13, w13_scale=convert_sf_to_mma_layout(
            storage13.view(-1), m=2 * intermediate, k=hidden, num_groups=experts, sf_vec_size=16),
        w13_alpha=alphas[0], w2=w2,
        w2_scale=convert_sf_to_mma_layout(
            storage2.view(-1), m=hidden, k=intermediate, num_groups=experts, sf_vec_size=16),
        w2_alpha=alphas[1], scale_storage=(storage13, storage2),
        swiglu_limit=None if swiglu_limit is None else float(swiglu_limit))


@dataclass(frozen=True)
class RoutedExpertsQuery:
    """Exact routed-expert invocation; tactics depend only on these values."""

    kind: str
    num_tokens: int
    num_experts: int
    hidden_size: int
    intermediate_size: int
    top_k: int
    routed_inputs: bool = False


@cache
def _trtllm_tactics(query: RoutedExpertsQuery):
    from flashinfer.tllm_enums import ActivationType, DtypeTrtllmGen, Fp8QuantizationType, WeightLayout

    configs = _trtllm().trtllm_get_valid_moe_configs(
        int(DtypeTrtllmGen.E2m1), int(DtypeTrtllmGen.E2m1), int(Fp8QuantizationType.NoneFp8),
        1 if query.routed_inputs else query.top_k, query.hidden_size, query.hidden_size, query.intermediate_size,
        query.num_experts, int(ActivationType.Swiglu), True, int(WeightLayout.MajorK), False,
        query.num_tokens * (query.top_k if query.routed_inputs else 1), False)
    return tuple((int(tile), int(config)) for tile, config in configs)


@cache
def _w4a16_tactics(query: RoutedExpertsQuery):
    import cutlass
    from flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16_kernel import Sm100W4A16GroupedGemmKernel
    from flashinfer.fused_moe.cute_dsl.moe_utils import get_max_num_permuted_tokens
    from flashinfer.fused_moe.cute_dsl.tuner import W4A16_MOE_TACTICS

    def implementable(m, k, slots, tactic):
        tiler, cluster, _ = tactic
        return Sm100W4A16GroupedGemmKernel.can_implement(
            mnkl=(m, slots, k, query.num_experts), a_dtype=cutlass.Float4E2M1FN,
            b_dtype=cutlass.BFloat16, c_dtype=cutlass.BFloat16, a_major="k", b_major="k",
            c_major="m", mma_tiler=tiler, cluster_shape_mn=cluster, use_2cta_instrs=tiler[0] == 256)

    hidden, intermediate = query.hidden_size, query.intermediate_size
    tactics = []
    for fc1, fc2 in W4A16_MOE_TACTICS:
        slots = get_max_num_permuted_tokens(query.num_tokens, query.top_k, query.num_experts, fc1[0][1])
        if implementable(2 * intermediate, hidden, slots, fc1) and implementable(hidden, intermediate, slots, fc2):
            tactics.append((fc1, fc2))
    return tuple(tactics)


TRTLLM_CONTRACT = native_contract(component_id="flare.moe.sm103_trtllm_nvfp4",
                                  query_type=RoutedExpertsQuery, tactics=_trtllm_tactics,
                                  semantic_version=3)
# Bump a contract's semantic version whenever its tactic space or launch semantics change,
# so cached selections from earlier spaces are never looked up.
W4A16_CONTRACT = native_contract(component_id="flare.moe.sm103_cute_w4a16",
                                 query_type=RoutedExpertsQuery, tactics=_w4a16_tactics,
                                 semantic_version=3)


def _aligned(nbytes):
    return (nbytes + 255) // 256 * 256


def _scratch_bytes(query):
    """NVFP4 activations and their row-major scales, padded to 16 rows as FlashInfer writes them."""
    if query.kind != "trtllm_nvfp4":
        return 0
    tokens, hidden = query.num_tokens * (query.top_k if query.routed_inputs else 1), query.hidden_size
    return _aligned(tokens * hidden // 2) + _aligned((tokens + 15) // 16 * 16 * hidden // 16)


_REDUCE_BLOCK = 512


@triton.jit
def _quantize_routes(Source, Ids, Globals, Values, Scales, ROWS: tl.constexpr,
                     HIDDEN: tl.constexpr, TOP_K: tl.constexpr):
    """NVFP4 rows in token/route order, preserving each routed expert's FC1 scale."""
    rows = tl.program_id(0) * 4 + tl.arange(0, 4)
    columns = tl.program_id(1) * 512 + tl.arange(0, 512)
    valid = rows < ROWS
    expert = tl.load(Ids + rows, valid, 0)
    global_scale = tl.load(Globals + expert)
    reciprocal = tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", constraints="=f,f",
        args=[global_scale], dtype=tl.float32, is_pure=True, pack=1)
    a = tl.load(Source + (rows // TOP_K)[:, None] * HIDDEN + columns[None, :] * 2,
        valid[:, None] & (columns[None, :] * 2 < HIDDEN), 0).to(tl.float32).reshape(4, 64, 8)
    b = tl.load(Source + (rows // TOP_K)[:, None] * HIDDEN + columns[None, :] * 2 + 1,
        valid[:, None] & (columns[None, :] * 2 < HIDDEN), 0).to(tl.float32).reshape(4, 64, 8)
    maximum = tl.max(tl.maximum(tl.abs(a), tl.abs(b)), 2)
    scale = tl.minimum(global_scale[:, None] * (maximum * (1.0 / 6.0)), 448.).to(tl.float8e4nv)
    effective = scale.to(tl.float32) * reciprocal[:, None]
    multiplier = tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", constraints="=f,f",
        args=[effective], dtype=tl.float32, is_pure=True, pack=1)
    multiplier = tl.where(maximum == 0, 0., multiplier)
    a, b = (a * multiplier[:, :, None]).reshape(4, 512), (b * multiplier[:, :, None]).reshape(4, 512)
    packed = tl.inline_asm_elementwise(
        "{ .reg .b8 v; cvt.rn.satfinite.e2m1x2.f32 v, $2, $1; cvt.u32.u8 $0, v; }",
        constraints="=r,f,f", args=[a, b], dtype=tl.uint32, is_pure=True, pack=1)
    tl.store(Values + rows[:, None] * (HIDDEN // 2) + columns[None, :], packed.to(tl.uint8),
        valid[:, None] & (columns[None, :] * 2 < HIDDEN))
    groups = tl.program_id(1) * 64 + tl.arange(0, 64)
    tl.store(Scales + rows[:, None] * (HIDDEN // 16) + groups[None, :],
        scale.to(tl.uint8, bitcast=True), valid[:, None] & (groups[None, :] < HIDDEN // 16))


_NATIVE_JITS.add(_quantize_routes)


@program_cache(scope="preparation")
def compile_route_quantization(tokens, hidden, top_k, ordinal):
    with torch.cuda.device(ordinal):
        return _quantize_routes.warmup(torch.bfloat16, torch.int32, torch.float32,
            torch.uint8, torch.uint8, ROWS=tokens * top_k, HIDDEN=hidden, TOP_K=top_k,
            grid=(1, 1, 1), num_warps=4)


@triton.jit
def _reduce_routes(Rows, Permuted, Weights, Shared, Out, HIDDEN: tl.constexpr, TOP_K: tl.constexpr,
                   BLOCK: tl.constexpr, ADD_SHARED: tl.constexpr):
    """Out = BF16(BF16(sum_k weight * FC2 row) + shared), routes summed in trtllm-gen's order."""
    token = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    total = tl.zeros((BLOCK,), dtype=tl.float32)
    for route in tl.static_range(TOP_K):
        row = tl.load(Permuted + token * TOP_K + route).to(tl.int64)
        weight = tl.load(Weights + token * TOP_K + route)
        total += weight * tl.load(Rows + row * HIDDEN + columns).to(tl.float32)
    routed = total.to(tl.bfloat16).to(tl.float32)
    if ADD_SHARED:
        routed += tl.load(Shared + token * HIDDEN + columns).to(tl.float32)
    tl.store(Out + token * HIDDEN + columns, routed.to(tl.bfloat16))


# b12x evicts compile-planning placeholders only from the Triton functions it tracks, which by
# default are its own; tracking this one lets in-process compilation (compile_workers=0) lower it.
_NATIVE_JITS.add(_reduce_routes)


@program_cache(scope="preparation")
def compile_route_reduction(tokens, hidden, top_k, ordinal, add_shared=True):
    """Warm the routed-row reduction that folds trtllm-gen's finalize into the shared-expert sum."""
    with torch.cuda.device(ordinal):
        return _reduce_routes.warmup(torch.bfloat16, torch.int32, torch.float32, torch.bfloat16,
                                     torch.bfloat16, HIDDEN=hidden, TOP_K=top_k, BLOCK=_REDUCE_BLOCK,
                                     ADD_SHARED=add_shared, grid=(tokens, hidden // _REDUCE_BLOCK), num_warps=4)


def _route_reduction_job(query, ordinal):
    return CompileJob.create("ignition.components.moe_sm103:compile_route_reduction",
                             query.num_tokens, query.hidden_size, query.top_k, ordinal)


class _TrtllmNvfp4State:
    """One trtllm-gen tactic over NVFP4 activations quantized into caller scratch.

    ``run`` finalizes inside trtllm-gen. ``stage_rows`` leaves FC2's permuted
    rows in caller-owned body buffers. ``reduce`` consumes those rows and their
    permutation to fold the routed sum into the shared expert's output in
    trtllm-gen's summation order. Bound calls retain the staged buffers until
    reduction; functional calls carry them explicitly between operations.
    """

    implementation = "trtllm_nvfp4"

    def __init__(self, experts: TrtllmNvfp4Experts, query: RoutedExpertsQuery, tactic):
        from flashinfer.tllm_enums import ActivationType, RoutingInputMode, RoutingMethodType

        self.experts, self.query, self.tactic = experts, query, [int(tactic[0]), int(tactic[1])]
        self.quantize = _fp4_quantizer().fp4_quantize
        self.moe = _trtllm().trtllm_fp4_block_scale_moe
        self.enums = (int(RoutingInputMode.UnpackedPrecomputed), int(RoutingMethodType.Renormalize),
                      int(ActivationType.Swiglu))
        ordinal = experts.device.index if experts.device.index is not None else torch.cuda.current_device()
        # Load the compiled reduction while preparing; serving freezes kernel resolution.
        self.reduction = load_programs(compile_route_reduction(query.num_tokens, query.hidden_size,
                                                               query.top_k, ordinal))
        self.routed_quantize = self.routed_reduction = None
        tokens = query.num_tokens
        if query.routed_inputs:
            tokens *= query.top_k
            self.routed_quantize = load_programs(compile_route_quantization(
                query.num_tokens, query.hidden_size, query.top_k, ordinal))
            self.routed_reduction = load_programs(compile_route_reduction(
                query.num_tokens, query.hidden_size, query.top_k, ordinal, False))
        self.staged = None
        self._prepare_buffer_layout(tokens)

    def _native(self, values, scales, ids, weights, output, finalize, metadata=(), body=(), prepare=False):
        e, q = self.experts, self.query
        routing, method, activation = self.enums
        return self.moe(routing, None, ids, weights, None, values, scales,
            e.w13, e.w13_scale, None, None, e.gate_alpha, e.gate_beta, e.clamp_limit,
            e.w2, e.w2_scale, None, e.output1_scale, e.output1_gate_scale, e.output2_scale,
            None, q.num_experts, 1 if q.routed_inputs else q.top_k, 0, None, None, q.intermediate_size, 0,
            q.num_experts, None, method, finalize, True, activation, output, self.tactic,
            True, None, list(metadata), list(body), prepare, None, None)

    def _prepare_buffer_layout(self, tokens):
        from flashinfer.fused_moe.core import trtllm_moe_allocate_routing_metadata
        from flashinfer.tllm_enums import RoutingInputMode

        q, device = self.query, self.experts.device
        top_k = 1 if q.routed_inputs else q.top_k
        ids = torch.empty((tokens, top_k), dtype=torch.int32, device=device)
        weights = torch.ones((tokens, top_k), dtype=torch.float32, device=device)
        metadata = trtllm_moe_allocate_routing_metadata(ids, num_experts=q.num_experts, top_k=top_k,
            local_expert_offset=0, num_local_experts=q.num_experts, tile_n=self.tactic[0],
            routing_input_mode=RoutingInputMode.UnpackedPrecomputed, topk_weights=weights)
        values = torch.empty((tokens, q.hidden_size // 2), dtype=torch.uint8, device=device)
        scales = torch.empty((tokens, q.hidden_size // 16), dtype=torch.float8_e4m3fn, device=device)
        unfinalized = torch.empty((tokens, 0), dtype=torch.bfloat16, device=device)
        body = self._native(values, scales, ids, weights, unfinalized, False, metadata.tensors(), prepare=True)
        body = [item if isinstance(item, torch.Tensor) else torch.from_dlpack(item) for item in body]
        layouts = [(tuple(t.shape), t.dtype) for t in (*metadata.tensors(), *body,
            weights if q.routed_inputs else weights[:0], unfinalized)]
        layouts[3] = ((0,), torch.float32)
        self.buffer_layouts = tuple(layouts)

    def allocate_buffers(self, device):
        buffers = [torch.empty(shape, dtype=dtype, device=device) for shape, dtype in self.buffer_layouts]
        if self.query.routed_inputs:
            buffers[16].fill_(1)
        return buffers

    def scratch_bytes(self):
        return _scratch_bytes(self.query)

    def _launch(self, a, topk_ids, topk_weights, output, scratch, finalize, buffers):
        e, q = self.experts, self.query
        tokens, hidden = a.shape
        if q.routed_inputs:
            tokens *= q.top_k
        values = scratch[:tokens * hidden // 2].view(tokens, hidden // 2)
        offset = _aligned(tokens * hidden // 2)
        scales = scratch[offset:offset + (tokens + 15) // 16 * 16 * hidden // 16]
        if q.routed_inputs:
            self.routed_quantize[(triton.cdiv(tokens, 4), triton.cdiv(hidden, 1024), 1)](
                a, topk_ids, e.input_global_scale, values, scales, tokens, hidden, q.top_k)
            topk_ids, topk_weights = topk_ids.view(tokens, 1), buffers[16]
        else:
            self.quantize(a, e.input_global_scale, values, scales, 16, False, False, False, False, True, 0)
        scales = scales[:tokens * hidden // 16].view(torch.float8_e4m3fn).view(tokens, hidden // 16)
        metadata = list(buffers[:9])
        metadata[3] = topk_weights
        _trtllm().trtllm_moe_populate_routing_metadata_multi_tile(topk_ids, q.num_experts,
            1 if q.routed_inputs else q.top_k, 0, q.num_experts, [self.tactic[0]], metadata,
            self.enums[0], topk_weights)
        return self._native(values, scales, topk_ids, topk_weights, output, finalize, metadata, buffers[9:16])

    def run(self, a, topk_ids, topk_weights, output, scratch, buffers=None):
        buffers = self.allocate_buffers(a.device) if buffers is None else buffers
        if not self.query.routed_inputs:
            self._launch(a, topk_ids, topk_weights, output, scratch, True, buffers)
            return
        rows, permuted = self.stage_rows(a, topk_ids, topk_weights, scratch, buffers)
        tokens, hidden = output.shape
        self.routed_reduction[(tokens, hidden // _REDUCE_BLOCK, 1)](
            rows, permuted, topk_weights, output, output, hidden, self.query.top_k, _REDUCE_BLOCK, False)

    def stage_rows(self, a, topk_ids, topk_weights, scratch, buffers=None):
        """Run quantization, routing, FC1 and FC2, keeping the unreduced FC2 rows for ``reduce``."""
        bound = buffers is None
        if bound and self.staged is not None:
            raise RuntimeError("SM103 routed rows were staged twice without a reduction")
        buffers = self.allocate_buffers(a.device) if buffers is None else buffers
        result = self._launch(a, topk_ids, topk_weights, buffers[17], scratch, False, buffers)
        rows, permuted = (t if isinstance(t, torch.Tensor) else torch.from_dlpack(t)
                          for t in (result[0], result[2]))
        if (rows.ndim != 2 or rows.shape[1] != self.query.hidden_size or not rows.is_contiguous()
                or permuted.numel() != topk_ids.numel()):
            raise RuntimeError("trtllm-gen FC2 rows differ from their planned geometry")
        if bound:
            self.staged = rows, permuted.view(-1), buffers
        return rows, permuted.view(-1)

    def reduce(self, topk_weights, shared, output, *, rows=None, permuted=None):
        if rows is None:
            if self.staged is None:
                raise RuntimeError("SM103 routed reduction requires staged FC2 rows")
            (rows, permuted, _), self.staged = self.staged, None
        tokens, hidden = output.shape
        self.reduction[(tokens, hidden // _REDUCE_BLOCK, 1)](
            rows, permuted, topk_weights, shared, output, hidden, self.query.top_k, _REDUCE_BLOCK, True)


class _CuteW4A16State:
    """One CuTe-DSL W4A16 tactic over BF16 activations.

    FC2 writes permuted rows that ``moe_unpermute`` reduces deterministically;
    the fused finalize's BF16 atomic reduction is not used.
    """

    implementation = "cute_w4a16"

    def __init__(self, experts: CuteW4A16Experts, query: RoutedExpertsQuery, tactic):
        from flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16 import launch_w4a16_moe
        from flashinfer.tllm_enums import ActivationType

        self.experts, self.query, self.tactic = experts, query, tuple(tactic)
        self.launch = partial(launch_w4a16_moe, activation_type=ActivationType.Swiglu)
        from flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16 import _get_workspace

        example = torch.empty((query.num_tokens, query.hidden_size), dtype=torch.bfloat16, device=experts.device)
        workspace = _get_workspace(example, query.top_k, query.num_experts, query.num_experts,
                                    query.intermediate_size, self.tactic[0][0][1])
        self.sort_names = tuple(workspace.moe_sort_buffers)
        self.buffer_layouts = tuple((tuple(t.shape), t.dtype) for t in
            (*workspace.moe_sort_buffers.values(), workspace.hidden_workspace, workspace.intermediate))

    def scratch_bytes(self):
        return _scratch_bytes(self.query)

    def run(self, a, topk_ids, topk_weights, output, scratch, buffers=None):
        e, q = self.experts, self.query
        clamp = {} if e.swiglu_limit is None else {"swiglu_limit": e.swiglu_limit}
        options = {}
        if buffers is not None:
            from flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16 import _W4A16Workspace
            options["workspace"] = _W4A16Workspace(dict(zip(self.sort_names, buffers[:-2], strict=True)),
                                                  buffers[-2], buffers[-1])
        self.launch(a, topk_ids, topk_weights, e.w13, e.w13_scale, e.w13_alpha,
                    e.w2, e.w2_scale, e.w2_alpha, q.num_experts, q.num_experts, 0, output,
                    use_fused_finalize=False, enable_pdl=True, tactic=self.tactic, **clamp, **options)


def _state_type(experts):
    if isinstance(experts, TrtllmNvfp4Experts):
        return TRTLLM_CONTRACT, _TrtllmNvfp4State
    if isinstance(experts, CuteW4A16Experts):
        return W4A16_CONTRACT, _CuteW4A16State
    raise TypeError("SM103 routed experts require FlashInfer-prepared expert storage")


def plan_routed_experts(experts: Sm103Experts, *, tokens: int, top_k: int):
    """Declare one exact-shape native routed-expert operator for ``experts``."""
    contract, state_type = _state_type(experts)
    query = RoutedExpertsQuery(experts.kind, int(tokens), experts.num_experts, experts.hidden_size,
                               experts.intermediate_size, int(top_k),
                               isinstance(experts, TrtllmNvfp4Experts) and experts.input_global_scale.numel() > 1)
    jobs = ((lambda tactic, detected: (_route_reduction_job(query, detected.ordinal),))
            if state_type is _TrtllmNvfp4State else (lambda tactic, detected: ()))
    if query.routed_inputs:
        jobs = lambda tactic, detected: (
            _route_reduction_job(query, detected.ordinal),
            CompileJob.create("ignition.components.moe_sm103:compile_route_reduction",
                query.num_tokens, query.hidden_size, query.top_k, detected.ordinal, False),
            CompileJob.create("ignition.components.moe_sm103:compile_route_quantization",
                query.num_tokens, query.hidden_size, query.top_k, detected.ordinal))
    return native_plan(contract=contract, query=query, device=experts.device,
                       scratch_bytes=lambda tactic: _scratch_bytes(query),
                       materialize=lambda tactic: state_type(experts, query, tactic), compile_jobs=jobs)


@torch.library.custom_op("flare::sm103_routed_experts", mutates_args=("output", "scratch"))
def _sm103_routed_experts(a: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor,
                          output: torch.Tensor, scratch: torch.Tensor, plan_handle: int) -> None:
    plan = plan_from_handle(plan_handle)
    require_prepared(plan, plan.component_id, a.device).run(a, topk_ids, topk_weights, output, scratch)


@_sm103_routed_experts.register_fake
def _fake_sm103_routed_experts(a, topk_ids, topk_weights, output, scratch, plan_handle) -> None:
    pass


@torch.library.custom_op("flare::sm103_stage_routed_rows", mutates_args=("scratch",))
def _sm103_stage_routed_rows(a: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor,
                             scratch: torch.Tensor, plan_handle: int) -> None:
    plan = plan_from_handle(plan_handle)
    require_prepared(plan, plan.component_id, a.device).stage_rows(a, topk_ids, topk_weights, scratch)


@_sm103_stage_routed_rows.register_fake
def _fake_sm103_stage_routed_rows(a, topk_ids, topk_weights, scratch, plan_handle) -> None:
    pass


@torch.library.custom_op("flare::sm103_reduce_routes", mutates_args=("output", "scratch"))
def _sm103_reduce_routes(topk_weights: torch.Tensor, shared: torch.Tensor, output: torch.Tensor,
                         scratch: torch.Tensor, plan_handle: int) -> None:
    plan = plan_from_handle(plan_handle)
    require_prepared(plan, plan.component_id, output.device).reduce(topk_weights, shared, output)


@_sm103_reduce_routes.register_fake
def _fake_sm103_reduce_routes(topk_weights, shared, output, scratch, plan_handle) -> None:
    pass


@torch.library.custom_op("flare::sm103_experts_out", mutates_args=("output", "scratch", "buffers"))
def _experts_out(a: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, output: torch.Tensor,
                 scratch: torch.Tensor, buffers: list[torch.Tensor], plan_handle: int, stage: bool) -> None:
    plan = plan_from_handle(plan_handle)
    state = require_prepared(plan, plan.component_id, a.device)
    if stage:
        state.stage_rows(a, ids, weights, scratch, buffers)
    else:
        state.run(a, ids, weights, output, scratch, buffers)


@_experts_out.register_fake
def _fake_experts_out(a, ids, weights, output, scratch, buffers, plan_handle, stage):
    pass


@torch.library.custom_op("flare::sm103_reduce_rows", mutates_args=("output",))
def _reduce_rows(rows: torch.Tensor, permutation: torch.Tensor, weights: torch.Tensor,
                 shared: torch.Tensor, output: torch.Tensor, plan_handle: int) -> None:
    plan = plan_from_handle(plan_handle)
    require_prepared(plan, plan.component_id, output.device).reduce(
        weights, shared, output, rows=rows, permuted=permutation)


@_reduce_rows.register_fake
def _fake_reduce_rows(rows, permutation, weights, shared, output, plan_handle):
    pass


@dataclass(frozen=True)
class StagedRoutedExperts:
    source: torch.Tensor
    ids: torch.Tensor
    weights: torch.Tensor
    scratch: torch.Tensor
    buffers: list[torch.Tensor]

    @property
    def rows(self):
        return self.buffers[13]

    @property
    def permutation(self):
        return self.buffers[1]


@dataclass(frozen=True, eq=False)
class Sm103Binding:
    """Exact tensors of one prepared native routed-expert launch.

    Eager calls launch the bound state directly. Traced calls record the
    compiler-visible custom op, which resolves the prepared plan by handle.
    """

    plan: object
    state: object
    a: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    output: torch.Tensor
    scratch: torch.Tensor
    implementation: str

    def run(self):
        if _tracing():
            torch.ops.flare.sm103_routed_experts(self.a, self.topk_ids, self.topk_weights,
                                                 self.output, self.scratch, self.plan._handle)
        else:
            self.state.run(self.a, self.topk_ids, self.topk_weights, self.output, self.scratch)
        return self.output

    def stage_rows(self):
        """Launch the routed experts without their reduction; ``reduce`` must follow."""
        if _tracing():
            torch.ops.flare.sm103_stage_routed_rows(self.a, self.topk_ids, self.topk_weights,
                                                    self.scratch, self.plan._handle)
        else:
            self.state.stage_rows(self.a, self.topk_ids, self.topk_weights, self.scratch)
        return self.scratch

    def reduce(self, shared, output):
        """output = routed sum + shared, rounding the routed sum to BF16 first as finalize does."""
        if shared.shape != output.shape or shared.dtype != torch.bfloat16 or not shared.is_contiguous():
            raise ValueError("the shared-expert output must be contiguous BF16 matching the routed output")
        if _tracing():
            torch.ops.flare.sm103_reduce_routes(self.topk_weights, shared, output, self.scratch,
                                                self.plan._handle)
        else:
            self.state.reduce(self.topk_weights, shared, output)
        return output


def _tracing():
    from torch.fx.experimental.proxy_tensor import get_proxy_mode

    return torch.compiler.is_compiling() or get_proxy_mode() is not None


def _bind(plan, state, a, topk_ids, topk_weights, output, scratch):
    tokens, hidden = a.shape
    query = plan.query
    if (a.dtype != torch.bfloat16 or (tokens, hidden) != (query.num_tokens, query.hidden_size)
            or not a.is_contiguous()):
        raise ValueError("SM103 routed experts require the prepared contiguous BF16 source")
    for tensor, dtype in ((topk_ids, torch.int32), (topk_weights, torch.float32)):
        if tensor.dtype != dtype or tensor.shape != (tokens, query.top_k) or not tensor.is_contiguous():
            raise ValueError("SM103 routes require contiguous int32 IDs and FP32 weights per token")
    if output.dtype != torch.bfloat16 or output.shape != a.shape or not output.is_contiguous():
        raise ValueError("SM103 routed output must be contiguous BF16 matching the source")
    if scratch.dtype != torch.uint8 or scratch.ndim != 1 or scratch.numel() < state.scratch_bytes():
        raise ValueError("SM103 routed-expert scratch is smaller than its plan")
    if any(t.device != a.device for t in (topk_ids, topk_weights, output, scratch)):
        raise ValueError("SM103 routed-expert tensors must share one device")
    return Sm103Binding(plan, state, a, topk_ids, topk_weights, output, scratch, state.implementation)


def _benchmark_call(state, *, plan, source, experts, top_k, session, name, workspace,
                    allocate_router, router_plan):
    from ignition.preparation.fixtures.fused_moe import ExpertBenchmarkInputs

    operands = benchmark_resource(session, ("moe_inputs", name),
        lambda: ExpertBenchmarkInputs(source, experts=experts.num_experts, top_k=top_k,
            allocate_router=allocate_router, router_plan=router_plan))
    scratch = planned_scratch(plan.memory_requirements().scratch, workspace)
    binding = _bind(plan, state, operands.source, operands.router.ids, operands.router.weights,
                    operands.output, scratch)
    return PreparedCall(run=binding.run, output=operands.output, produce=operands.producers[0],
                        benchmark_producers=operands.producers,
                        owners=(operands, scratch, binding, experts))


class Sm103RoutedExperts:
    """Prepared native expert execution with fixed source and route storage.

    Mirrors ``PreparedRoutedExperts``: one exact-shape plan declared to the
    preparation session, tuned over the native kernel's tactics, then bound
    to caller-owned scratch after tuning.
    """

    def __init__(self, source, experts: Sm103Experts, ids, weights, output, *, session, name: str,
                 workspace: ScratchArena | torch.Tensor | None = None,
                 router_plan=None, a4_prefill=False, router=None, route_output=None):
        from ignition.components.moe import RouterScratch

        if a4_prefill:
            raise ValueError("SM103 routed experts do not provide A4 prefill for A16 experts")
        if source.ndim != 2 or source.shape[1] != experts.hidden_size or ids.ndim != 2:
            raise ValueError("SM103 routed experts require [tokens, hidden] source and [tokens, top_k] routes")
        for field in ("w13", "w13_scale", "w2", "w2_scale", "w13_alpha", "w2_alpha",
                      "input_global_scale", "output1_scale", "output1_gate_scale", "output2_scale",
                      "gate_alpha", "gate_beta", "clamp_limit"):
            tensor = getattr(experts, field, None)
            if tensor is not None:
                record_numeric_tensor(tensor)
        self.router = router
        self.preparation_arena = workspace
        self.source, self.output, self.scratch = source, output, None
        self.output_source = output
        self.plan = plan_routed_experts(experts, tokens=source.shape[0], top_k=ids.shape[1])
        plan = self.plan

        def prepare_call(state):
            scratch = planned_scratch(plan.memory_requirements().scratch, workspace)
            self.scratch = scratch
            binding = _bind(plan, state, source, ids, weights, output, scratch)
            return PreparedCall(run=binding.run, owners=prime_owners(workspace, scratch, binding))

        declare(session, (plan.request(name=name, prepare_call=prepare_call,
            benchmark_call=partial(_benchmark_call, plan=plan, source=source, experts=experts,
                top_k=ids.shape[1], session=session, name=name, workspace=workspace,
                allocate_router=RouterScratch.allocate, router_plan=router_plan)),))

        def finalize():
            self.scratch = planned_scratch(plan.scratch_specs(), workspace)
            state = require_prepared(plan, plan.component_id, source.device)
            self.binding = _bind(plan, state, source, ids, weights, output, self.scratch)

        register_finalizer(session, finalize, workspace=workspace)

    def functional(self, *, session):
        """Allocate native body buffers and expose staged rows in the forward."""
        if self.output is not None:
            self.output_layout = (tuple(self.output.shape), self.output.stride(), self.output.dtype)
            self.output = self.output_source = self.source = None
            if self.router is not None:
                self.router.functional(session=session)
            register_finalizer(session, self._finalize_functional, workspace=self.preparation_arena)
        return self

    def _finalize_functional(self):
        self.buffer_layouts = self.binding.state.buffer_layouts
        self.workspace_nbytes = self.binding.state.scratch_bytes()
        self.binding = self.scratch = None
        self.preparation_arena = None

    def _local_buffers(self, source):
        buffers = [torch.empty(shape, dtype=dtype, device=source.device) for shape, dtype in self.buffer_layouts]
        if self.plan.query.routed_inputs:
            buffers[16].fill_(1)
        return buffers

    @property
    def reduces_into_shared(self) -> bool:
        """Whether ``stage_rows``/``reduce`` can replace the finalized output plus a shared add."""
        return self.plan.contract is TRTLLM_CONTRACT

    def route(self, logits: torch.Tensor, *, image_mask=None, num_valid_tokens=None):
        return self.router(logits, image_mask=image_mask, num_valid_tokens=num_valid_tokens)

    def __call__(self, source=None, ids=None, weights=None, *, router_logits=None, selected_logits=None,
                 image_mask=None, num_valid_tokens=None):
        if self.output is None:
            if source is None or ids is None or weights is None:
                raise ValueError("functional SM103 experts require source, expert IDs, and route weights")
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
            scratch = torch.empty(self.workspace_nbytes, dtype=torch.uint8, device=source.device)
            _experts_out(source, ids, weights, output, scratch, self._local_buffers(source), self.plan.handle, False)
            return output
        return self.binding.run()

    def stage_rows(self, source=None, ids=None, weights=None):
        """Run the routed experts up to unreduced FC2 rows; returns the scratch that orders ``reduce``."""
        if self.output is None:
            if not self.reduces_into_shared:
                raise ValueError("staged routed rows require the trtllm NVFP4 implementation")
            if source is None or ids is None or weights is None:
                raise ValueError("functional staged experts require source, expert IDs, and route weights")
            scratch = torch.empty(self.workspace_nbytes, dtype=torch.uint8, device=source.device)
            buffers = self._local_buffers(source)
            _experts_out(source, ids, weights, buffers[17], scratch, buffers, self.plan.handle, True)
            return StagedRoutedExperts(source, ids, weights, scratch, buffers)
        return self.binding.stage_rows()

    def reduce(self, shared, output=None, *, staged=None):
        if self.output is None:
            if staged is None:
                raise ValueError("functional reduction requires explicit staged routed rows")
            if output is None:
                output = torch.empty_like(shared)
            _reduce_rows(staged.rows, staged.permutation, staged.weights, shared, output, self.plan.handle)
            return output
        return self.binding.reduce(shared, output)
