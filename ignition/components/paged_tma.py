"""Prepared FP8 TMA prefill over NHD cache storage with conversion scratch."""

from dataclasses import dataclass
import math

import torch
from b12x.preparation import PreparedCall
from b12x._lib.scratch import scratch_buffer_spec

from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import declare, register_finalizer
from .native_operator import native_contract, native_plan


@dataclass(frozen=True)
class Query:
    rows: int
    heads: int
    kv_heads: int
    dim: int
    page: int
    gated: bool
    fp8_input: bool


CONTRACT = native_contract(component_id="flare.paged_tma", query_type=Query, tactics=lambda q: (128,), semantic_version=3)


@dataclass(frozen=True)
class State:
    query: Query
    kernel: object

    def run(self, q, key, value, pages, lengths, cu, output, scratch, output_gate=None):
        from cutlass import Float32, Int32

        count = 0 if self.query.fp8_input else q.numel()
        query_fp8 = q if self.query.fp8_input else scratch[:count].view(torch.float8_e4m3fn).view(q.shape)
        lengths_i32 = scratch[count:count + 4 * lengths.numel()].view(torch.int32)
        if not self.query.fp8_input:
            query_fp8.copy_(q)
        lengths_i32.copy_(lengths)
        output.zero_()
        arguments = (query_fp8, key.permute(0, 2, 1, 3), value.permute(0, 2, 1, 3), output, None,
            Float32(math.log2(math.e) / math.sqrt(self.query.dim)), Float32(1), None,
            lengths_i32, cu, pages, None, Int32(self.query.rows), False)
        self.kernel(*arguments, *((output_gate,) if output_gate is not None else ()))


def supports(q, key, value, pages, cu, output):
    return (torch.cuda.get_device_capability(q.device) in ((12, 0), (12, 1))
        and q.dtype in (torch.bfloat16, torch.float8_e4m3fn) and output.dtype == torch.bfloat16
        and q.shape[0] >= 128
        and q.shape[2] == key.shape[3] == value.shape[3] == 256
        and key.dtype == value.dtype == torch.float8_e4m3fn
        and key.shape[1] == value.shape[1] == 128 and pages.shape[0] == 1
        and pages.dtype == cu.dtype == torch.int32
        and all(t.is_contiguous() for t in (q, output, pages, cu))
        and all(t.stride()[1:] == (t.shape[2] * 256, 256, 1) for t in (key, value)))


class PreparedTma:
    def __init__(self, q, key, value, pages, lengths, cu, output, *, session, workspace, name, output_gate=None):
        if output_gate is not None and (output_gate.shape != q.shape or output_gate.dtype != torch.bfloat16
                or output_gate.device != q.device or output_gate.stride(-1) != 1
                or any(stride % 8 for stride in output_gate.stride()[:-1]) or output_gate.data_ptr() % 16):
            raise ValueError("TMA output gate requires an aligned BF16 view matching query geometry")
        self.output_gate = output_gate
        geometry = Query(q.shape[0], q.shape[1], key.shape[2], q.shape[2], key.shape[1],
                         output_gate is not None, q.dtype == torch.float8_e4m3fn)
        self.arguments = q, key, value, pages, lengths, cu, output
        device = q.device
        scratch_bytes = (0 if geometry.fp8_input else q.numel()) + 4 * lengths.numel()

        def materialize(tile):
            from flashinfer.cute_dsl.attention.fmha.sm120 import compile_sm120_fmha_fp8_paged_kernel

            kernel = compile_sm120_fmha_fp8_paged_kernel(
                in_dtype=torch.float8_e4m3fn, out_dtype=torch.bfloat16,
                num_qo_heads=geometry.heads, num_kv_heads=geometry.kv_heads,
                head_dim=geometry.dim, is_causal=True, kv_tile=128, q_tile=tile,
                num_tokens_per_page=geometry.page, device=device,
                balanced_scheduler=True, kv_layout="NHD", with_output_gate=geometry.gated)
            return State(geometry, kernel)

        self.plan = native_plan(contract=CONTRACT, query=geometry, device=device,
            scratch_bytes=lambda tile: scratch_bytes, materialize=materialize)

        def call(state):
            scratch = planned_scratch((scratch_buffer_spec(CONTRACT.component_id,
                nbytes=scratch_bytes, device=device),), workspace)
            return PreparedCall(run=lambda: state.run(*self.arguments, scratch, output_gate),
                owners=prime_owners(workspace, scratch, self.arguments, output_gate))

        declare(session, (self.plan.request(name=name, prepare_call=call),))
        register_finalizer(session, lambda: self.bind(workspace), workspace=workspace)

    def bind(self, workspace):
        self.scratch = planned_scratch(self.plan.scratch_specs(), workspace)
