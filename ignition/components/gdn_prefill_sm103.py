# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen GDN prefill on SM103 through FlashInfer's chunked gated delta rule.

b12x's prefill runs the delta rule one token at a time (``sequential_k1``);
FlashInfer's chunked kernel runs it about 1.7x faster on GB300 for a
4096-token chunk. It reads contiguous query, key and value heads, decays as
natural factors and FP32 update gates, and updates each sequence's state in
place at ``state_indices``. A sequence that starts fresh would otherwise read
its live row, so that row is zeroed first; b12x reads the permanently zero
null row instead, which is the same initial state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.triton import launch_triton, warmup_triton
from ignition.preparation.tuning import declare, register_finalizer

from .native_operator import native_contract, native_plan

_ZERO_CONFIG = triton.Config({}, num_warps=4, num_stages=1)
_ZERO_BLOCK = 2048


@dataclass(frozen=True)
class GdnPrefillQuery:
    tokens: int
    sequences: int
    key_heads: int
    value_heads: int


CONTRACT = native_contract(component_id="flare.sm103_gdn_prefill", query_type=GdnPrefillQuery,
                           tactics=lambda query: ("chunked",))


@triton.autotune(configs=[_ZERO_CONFIG], key=[])
@triton.jit
def _zero_fresh_rows(State, Fresh, Final, NumSeqs, state_stride, BLOCK: tl.constexpr):
    sequence, head = tl.program_id(0), tl.program_id(1)
    if sequence < tl.load(NumSeqs):
        slot = tl.load(Final + sequence).to(tl.int64)
        if tl.load(Fresh + sequence) & (slot != 0):
            base = State + slot * state_stride + head * (128 * 128)
            for start in range(0, 128 * 128, BLOCK):
                tl.store(base + start + tl.arange(0, BLOCK), tl.zeros((BLOCK,), tl.float32))


class _ChunkedState:
    """FlashInfer's chunked delta rule at one prepared token capacity."""

    def __init__(self, query: GdnPrefillQuery):
        from flashinfer.gdn_prefill import chunk_gated_delta_rule

        self.query, self.kernel = query, chunk_gated_delta_rule

    def run(self, q, k, v, g, beta, state, cu_seqlens, indices, output):
        self.kernel(q, k, v, g=g, beta=beta, initial_state=state, output_final_state=True, cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=True, output=output, output_state=state, state_indices=indices,
                    use_cp=False, max_seqlen=self.query.tokens)


@torch.library.custom_op("flare::sm103_gdn_prefill", mutates_args=("state", "output"))
def _sm103_gdn_prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
                       state: torch.Tensor, cu_seqlens: torch.Tensor, indices: torch.Tensor, output: torch.Tensor,
                       plan_handle: int) -> None:
    plan = plan_from_handle(plan_handle)
    require_prepared(plan, plan.component_id, q.device).run(q, k, v, g, beta, state, cu_seqlens, indices, output)


@_sm103_gdn_prefill.register_fake
def _fake_sm103_gdn_prefill(q, k, v, g, beta, state, cu_seqlens, indices, output, plan_handle) -> None:
    pass


def _tracing():
    from torch.fx.experimental.proxy_tensor import get_proxy_mode

    return torch.compiler.is_compiling() or get_proxy_mode() is not None


@dataclass(frozen=True)
class PrefillBuffers:
    """Contiguous query, key and value heads and FP32 decay and update gates for one prefill chunk."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    decay: torch.Tensor
    beta: torch.Tensor

    @classmethod
    def allocate(cls, tokens: int, device, *, key_heads: int, value_heads: int, buffers=None) -> PrefillBuffers:
        def zeros(shape, dtype=torch.bfloat16):
            return (torch.zeros(shape, dtype=dtype, device=device) if buffers is None
                    else buffers.zeros(shape, dtype))
        return cls(zeros((tokens, key_heads, 128)), zeros((tokens, key_heads, 128)),
                   zeros((tokens, value_heads, 128)), zeros((tokens, value_heads), torch.float32).fill_(1),
                   zeros((tokens, value_heads), torch.float32))

    def rows(self, tokens: int) -> PrefillBuffers:
        return PrefillBuffers(*(getattr(self, name)[:tokens] for name in ("query", "key", "value", "decay", "beta")))

    def fits(self, tokens: int, *, key_heads: int, value_heads: int) -> bool:
        return (self.query.shape == (tokens, key_heads, 128) and self.key.shape == self.query.shape
                and self.value.shape == (tokens, value_heads, 128)
                and self.decay.shape == self.beta.shape == (tokens, value_heads))


class PreparedGdnPrefill:
    """One layer's prefill recurrence over packed sequences, writing the recurrence before the gated norm.

    ``mixed_qkv`` is the convolved [tokens, q | k | v] projection; ``a`` and ``b`` are the
    decay and update-gate logits. Final states replace each sequence's row of ``state``.
    """

    def __init__(self, *, mixed_qkv, a, b, A_log, dt_bias, state, metadata, output, key_heads: int,
                 value_heads: int, session, name: str, buffers: PrefillBuffers | None = None):
        tokens = mixed_qkv.shape[0]
        sequences = metadata.query_start_loc.numel() - 1
        if (state.ndim != 4 or state.shape[1:] != (value_heads, 128, 128) or state.dtype != torch.float32
                or state.stride()[1:] != (128 * 128, 128, 1) or output.shape != (tokens, value_heads, 128)
                or output.dtype != torch.bfloat16 or not output.is_contiguous()
                or mixed_qkv.shape[1] != (2 * key_heads + value_heads) * 128):
            raise ValueError("SM103 GDN prefill requires packed FP32 state rows and contiguous BF16 output")
        device = mixed_qkv.device
        self.mixed, self.a, self.b, self.A_log, self.dt_bias = mixed_qkv, a, b, A_log, dt_bias
        self.state, self.metadata, self.output = state, metadata, output
        self.key_heads, self.value_heads = key_heads, value_heads
        # Views of the convolved projection; traced forwards only copy them (no split inside the trace).
        q, k, v = mixed_qkv.split((key_heads * 128, key_heads * 128, value_heads * 128), dim=-1)
        self.sources = (q.view(tokens, key_heads, 128), k.view(tokens, key_heads, 128),
                        v.view(tokens, value_heads, 128))
        # Contiguous heads and gates; layers share one set through the decoder's GDN scratch.
        buffers = (PrefillBuffers.allocate(tokens, device, key_heads=key_heads, value_heads=value_heads)
                   if buffers is None else buffers.rows(tokens))
        if not buffers.fits(tokens, key_heads=key_heads, value_heads=value_heads):
            raise ValueError("SM103 GDN prefill buffers must cover the token capacity and head geometry")
        self.q, self.k, self.v, self.g, self.beta = buffers.query, buffers.key, buffers.value, buffers.decay, buffers.beta
        self.indices = metadata.final_state_indices
        query = GdnPrefillQuery(tokens, sequences, key_heads, value_heads)
        self.plan = native_plan(contract=CONTRACT, query=query, device=device, scratch_bytes=lambda tactic: 16,
                                materialize=lambda tactic: _ChunkedState(query))
        plan = self.plan
        self.chunked = None

        def prepare_call(chunked):
            self.chunked = chunked
            return PreparedCall(run=self._run_chunked)

        declare(session, (plan.request(name=name, prepare_call=prepare_call),))

        def finalize():
            self.chunked = require_prepared(plan, plan.component_id, device)

        register_finalizer(session, finalize)
        self.zero_args = (state, metadata.fresh, metadata.final_state_indices, metadata.num_seqs, state.stride(0))
        self.zero_grid = (sequences, value_heads, 1)
        self.zero_compiled = warmup_triton(_zero_fresh_rows.fn, self.zero_args, self.zero_grid,
                                           num_warps=_ZERO_CONFIG.num_warps,
                                           num_stages=_ZERO_CONFIG.num_stages, BLOCK=_ZERO_BLOCK)

    def _run_chunked(self):
        self.chunked.run(self.q, self.k, self.v, self.g, self.beta, self.state, self.metadata.query_start_loc,
                         self.indices, self.output)

    def functional(self, *, session, dynamic_metadata=False):
        if getattr(self, "_functional_requested", False):
            return self
        self._functional_requested = True
        self._dynamic_metadata = dynamic_metadata
        self._functional_ready = False
        self.output_shape = tuple(self.output.shape)
        register_finalizer(session, self._finalize_functional)
        return self

    def _finalize_functional(self):
        self.mixed = self.a = self.b = self.output = None
        self.sources = ()
        self.q = self.k = self.v = self.g = self.beta = None
        if self._dynamic_metadata:
            self.metadata = self.indices = None
            self.zero_args = (self.state, None, None, None, self.state.stride(0))
        self._functional_ready = True

    def __call__(self, mixed_qkv=None, a=None, b=None, *, metadata=None):
        if getattr(self, "_functional_requested", False):
            if not self._functional_ready:
                raise RuntimeError("functional SM103 GDN requires completed preparation")
            if mixed_qkv is None or a is None or b is None:
                raise ValueError("functional SM103 GDN requires QKV and gate operands")
            tokens = mixed_qkv.shape[0]
            parts = mixed_qkv.split((self.key_heads * 128, self.key_heads * 128, self.value_heads * 128), dim=-1)
            q, k, v = (torch.empty((tokens, heads, 128), dtype=torch.bfloat16, device=mixed_qkv.device)
                       .copy_(part.view(tokens, heads, 128))
                       for part, heads in zip(parts, (self.key_heads, self.key_heads, self.value_heads), strict=True))
            softplus = F.softplus(a.float() + self.dt_bias.float(), threshold=20.0)
            g = torch.empty((tokens, self.value_heads), dtype=torch.float32, device=mixed_qkv.device)
            g.copy_(torch.exp(-torch.exp(self.A_log.float()) * softplus))
            beta = torch.empty_like(g)
            beta.copy_(torch.sigmoid(b.float()).to(torch.bfloat16).float())
            output = torch.zeros(self.output_shape, dtype=torch.bfloat16, device=mixed_qkv.device)
            if metadata is None:
                metadata = self.metadata
                zero_args, indices = self.zero_args, self.indices
            else:
                zero_args = (self.state, metadata.fresh, metadata.final_state_indices,
                             metadata.num_seqs, self.state.stride(0))
                indices = metadata.final_state_indices
            if metadata is None:
                raise ValueError("functional SM103 GDN requires metadata at invocation")
            launch_triton(_zero_fresh_rows, self.zero_compiled, zero_args, self.zero_grid, BLOCK=_ZERO_BLOCK)
            torch.ops.flare.sm103_gdn_prefill(q, k, v, g, beta, self.state,
                metadata.query_start_loc, indices, output, self.plan._handle)
            return output
        for target, source in zip((self.q, self.k, self.v), self.sources):
            target.copy_(source)
        # Gates as b12x's recurrence forms them: decay exp(-exp(A_log) softplus(a + dt_bias)) and the
        # update gate sigmoid(b) rounded through BF16.
        softplus = F.softplus(self.a.float() + self.dt_bias.float(), threshold=20.0)
        self.g.copy_(torch.exp(-torch.exp(self.A_log.float()) * softplus))
        self.beta.copy_(torch.sigmoid(self.b.float()).to(torch.bfloat16).float())
        launch_triton(_zero_fresh_rows, self.zero_compiled, self.zero_args, self.zero_grid, BLOCK=_ZERO_BLOCK)
        if _tracing():
            torch.ops.flare.sm103_gdn_prefill(self.q, self.k, self.v, self.g, self.beta, self.state,
                                              self.metadata.query_start_loc, self.indices, self.output,
                                              self.plan._handle)
        else:
            self._run_chunked()
        return self.output
