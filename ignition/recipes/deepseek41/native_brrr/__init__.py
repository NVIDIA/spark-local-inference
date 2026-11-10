# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""BRRR's TP4 small-verification fusion prepared within Flare's graph boundary.

The target checkpoint remains unchanged. These kernels retain BRRR's activation
arithmetic, which differs from Flare's autotuned decomposed path. Wider rows and
all prefill/draft execution retain the existing operators.
"""
from dataclasses import dataclass
import itertools
import weakref

import torch
import torch.distributed as dist
from b12x.preparation import CollectiveRequirement, MemoryRequirements, PersistentMemory, Plan, PreparedCall

from ignition.components.native_operator import native_contract
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import declare, register_finalizer

from .weights import DecoderAdapter

_BRIDGES = weakref.WeakKeyDictionary()
_STATES = weakref.WeakValueDictionary()
_HANDLES = itertools.count()


def group_eligible(coordinator, device, *, enabled):
    """Choose the same fused collective implementation on every preparation rank."""
    if not enabled or len(coordinator.ranks) != 4 or coordinator.spans_hosts:
        return False
    capabilities = [None] * 4
    dist.all_gather_object(capabilities, torch.cuda.get_device_capability(device),
                          group=coordinator.group)
    return all(capability == (12, 0) for capability in capabilities)


def eligible(weights, source, *, mode, world, boundary, text, spans_hosts=False,
             native_dense_activations=False):
    arithmetic = {
        'rms_norm_eps': 1e-20, 'hc_eps': 1e-6, 'hc_sinkhorn_iters': 20,
        'num_experts_per_tok': 6, 'scoring_func': 'sqrtsoftplus',
        'norm_topk_prob': True, 'routed_scaling_factor': 1.5, 'swiglu_limit': 10.,
    }
    return (mode == 'verify' and world == 4 and not spans_hosts and not native_dense_activations
            and boundary is None and 5 <= source.shape[0] <= 8
            and source.shape[1:] == (4, 5120) and source.dtype == torch.bfloat16
            and source.device.type == 'cuda'
            and torch.cuda.get_device_capability(source.device) == (12, 0)
            and all(text.get(key) == value for key, value in arithmetic.items())
            and weights.experts.experts.num_experts == 384
            and not weights.experts.experts.has_csf_blocks)


def _scales(weight):
    packed = weight.weight
    n, k = packed.values.shape
    return packed.scale_rows.view(torch.uint8).reshape(n // 8, 4, 2, k // 64, 2).permute(0, 3, 1, 4, 2).contiguous()


class _Bridge:
    def __init__(self, runtime, rank):
        self.runtime = runtime
        self.adapter = DecoderAdapter(runtime, rank)
        self.projections = {}
        self.states = []
        self.empty_rope = torch.empty(0, device=torch.device('cuda', torch.cuda.current_device()), dtype=torch.float32)

    def projection(self, index, weights):
        if index not in self.projections:
            attention, residual = weights.attention, weights.residual
            qa, qb = attention['qkv_a'], attention['q_b']
            values = (residual['hc_attn_fn'], residual['attn_norm.weight'],
                      residual['hc_attn_scale'], residual['hc_attn_base'],
                      qa.weight.values.view(torch.uint8), _scales(qa),
                      attention['q_norm'], attention['kv_norm'],
                      qb.weight.values.view(torch.uint8), _scales(qb))
            for tensor in values:
                record_numeric_tensor(tensor)
            self.projections[index] = values
        return self.projections[index]

    def close(self):
        self.adapter.close()
        self.states.clear()
        self.projections.clear()
        _BRIDGES.pop(self.runtime, None)


def close_resource(runtime):
    return _BRIDGES.get(runtime)


class _State:
    def __init__(self, runtime, rank, index, weights, prologue):
        bridge = _BRIDGES.get(runtime)
        if bridge is None:
            bridge = _Bridge(runtime, rank)
            _BRIDGES[runtime] = bridge
        self.bridge = bridge
        self.layer = bridge.adapter.register(index, weights)
        self.project = bridge.projection(index, weights) if prologue else None
        self.scale = weights.residual['hc_attn_scale'] if prologue else None
        self.base = weights.residual['hc_attn_base'] if prologue else None
        bridge.states.append(self)

    def prologue(self, inputs, outputs):
        residual, normalized, qr, kv, qr_q, qr_s, query, post, comb, pre = outputs
        self.bridge.adapter.ext.prologue_out(self.bridge.adapter.handle, *inputs, *self.project,
            residual, normalized, qr, kv, qr_q, qr_s, query, post, comb, pre, True)

    def suffix(self, attended, positions, residual, mixes, outputs):
        self.bridge.adapter.ext.run(self.bridge.adapter.handle, self.layer, attended, positions, None,
            self.bridge.empty_rope, residual, *mixes, self.scale, self.base, *outputs)


@torch.library.custom_op('flare::ds41_brrr_prologue', mutates_args=('outputs',))
def _prologue(inputs: list[torch.Tensor], outputs: list[torch.Tensor], handle: int) -> None:
    _STATES[handle].prologue(inputs, outputs)


@_prologue.register_fake
def _fake_prologue(inputs, outputs, handle):
    pass


@torch.library.custom_op('flare::ds41_brrr_suffix', mutates_args=('mixes', 'outputs'))
def _suffix(attended: torch.Tensor, positions: torch.Tensor, residual: torch.Tensor,
            mixes: list[torch.Tensor], outputs: list[torch.Tensor], handle: int) -> None:
    _STATES[handle].suffix(attended, positions, residual, mixes, outputs)


@_suffix.register_fake
def _fake_suffix(attended, positions, residual, mixes, outputs, handle):
    pass


@dataclass(frozen=True)
class Query:
    rows: int
    prologue: bool


_CONTRACT = native_contract(component_id='flare.deepseek41.brrr_small_verify', query_type=Query,
                            tactics=lambda query: ('persistent',))


class PreparedSmallVerification:
    def __init__(self, index, weights, scratch, metadata, *, carry, runtime, shard,
                 session, coordinator, name, buffers):
        self.has_prologue = bool(carry and weights.engram is None)
        self.handle = next(_HANDLES)
        s = scratch
        self.inputs = [s.mlp_output, s.ffn_hc.residual, s.ffn_hc.post, s.ffn_hc.comb, s.pre_mix]
        self.prologue_outputs = None
        if self.has_prologue:
            rows = metadata.positions.numel()
            qr_q = buffers.zeros((rows, 1280), torch.uint8)
            qr_s = buffers.zeros((5120,), torch.uint8)
            self.prologue_outputs = [s.attention_hc.residual, s.attention_hc.normalized,
                s.attention.query_low, s.attention.window_values, qr_q, qr_s,
                s.attention.query, s.attention_hc.post, s.attention_hc.comb, s.attn_mix]
        self.attended, self.positions, self.residual = s.attention.attended, metadata.positions, s.attention_hc.residual
        self.mixes = [s.attention_hc.post, s.attention_hc.comb, s.attn_mix]
        self.outputs = [s.mlp_output, s.ffn_hc.residual, s.ffn_hc.post, s.ffn_hc.comb, s.pre_mix]
        query = Query(metadata.positions.numel(), self.has_prologue)
        # Keys share the engine allocation and each repacked layer across graph variants.
        plan = Plan(contract=_CONTRACT, query=query, _device=self.attended.device,
            _compile_jobs=lambda config, device: (),
            _memory_requirements=lambda config, device: MemoryRequirements(persistent=(
                PersistentMemory(('brrr-engine', shard.rank), 64 << 20),
                PersistentMemory(('brrr-layer', shard.rank, index), 34 << 20))),
            _materialize=lambda selection, device: _State(runtime, shard.rank, index, weights, self.has_prologue))

        def prepared(state):
            def run():
                if self.has_prologue:
                    state.prologue(self.inputs, self.prologue_outputs)
                state.suffix(self.attended, self.positions, self.residual, self.mixes, self.outputs)
            return PreparedCall(run=run, output=tuple(self.outputs), owners=(state,))

        request = plan.request(name=name, prepare_call=prepared,
            collective=CollectiveRequirement(key=name, ranks=tuple(dist.get_process_group_ranks(runtime.exchange_group))))
        declare(session, (request,), coordinator=coordinator)

        def finalize():
            from b12x.preparation.types import require_prepared
            _STATES[self.handle] = require_prepared(plan, _CONTRACT.component_id, self.attended.device)
        register_finalizer(session, finalize)

    def prologue(self):
        _prologue(self.inputs, self.prologue_outputs, self.handle)

    def suffix(self):
        _suffix(self.attended, self.positions, self.residual, self.mixes, self.outputs, self.handle)
