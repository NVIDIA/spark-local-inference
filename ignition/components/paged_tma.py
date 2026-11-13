"""Prepared FP8 TMA prefill over NHD cache storage with conversion scratch.

The plain program serves head dimension 256 without score modifications. The
extended program adds the operands the SM120 FP8 TMA kernel evaluates in its
score loop and epilogue: a left window, a per-row attention sink logit and a
rank-16 relative-distance bias `round_bf16(factor[row, head] . projection[d])`
for distances `0 <= d < extent`. The kernel implements those operands for head
dimension 128 with 128-row query and K/V tiles and one query head per tile row.
Queries and probabilities are rounded to E4M3 in both programs.
"""

from dataclasses import dataclass
import math

import torch
from b12x.preparation import PreparedCall
from b12x._lib.scratch import scratch_buffer_spec

from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import declare, register_finalizer
from .native_operator import native_contract, native_plan

RELATIVE_RANK = 16


@dataclass(frozen=True)
class Query:
    rows: int
    heads: int
    kv_heads: int
    dim: int
    page: int
    gated: bool
    fp8_input: bool
    window: int = -1
    row_sink: bool = False
    relative_extent: int = 0

    @property
    def extended(self):
        return self.window >= 0 or self.scored

    @property
    def scored(self):
        """Programs with a sink or bias take every optional operand of the extended launch."""
        return self.row_sink or self.relative_extent > 0


CONTRACT = native_contract(component_id="flare.paged_tma", query_type=Query, tactics=lambda q: (128,), semantic_version=4)


@dataclass(frozen=True)
class State:
    query: Query
    kernel: object

    def run(self, q, key, value, pages, lengths, cu, output, scratch, output_gate=None, row_sink=None,
            relative_factor=None, relative_projection=None):
        from cutlass import Float32, Int32

        query = self.query
        if (output_gate is not None) != query.gated or (row_sink is not None) != query.row_sink or (
                relative_factor is not None) != (query.relative_extent > 0):
            raise ValueError("TMA prefill operands differ from the prepared program")
        count = 0 if query.fp8_input else q.numel()
        query_fp8 = q if query.fp8_input else scratch[:count].view(torch.float8_e4m3fn).view(q.shape)
        lengths_i32 = scratch[count:count + 4 * lengths.numel()].view(torch.int32)
        if query.extended:
            from flashinfer.cute_dsl.attention.fmha.sm120 import validate_sm120_fmha_fp8_paged_operands

            # The kernel indexes these operands by query row without bounds checks.
            validate_sm120_fmha_fp8_paged_operands(q, row_sink=row_sink, output_gate=output_gate,
                relative_factor=relative_factor, relative_projection=relative_projection)
            if relative_projection is not None and relative_projection.shape[0] != query.relative_extent:
                raise ValueError("TMA relative projection rows differ from the prepared extent")
        if not query.fp8_input:
            query_fp8.copy_(q)
        lengths_i32.copy_(lengths)
        output.zero_()
        arguments = (query_fp8, key.permute(0, 2, 1, 3), value.permute(0, 2, 1, 3), output, None,
            Float32(math.log2(math.e) / math.sqrt(query.dim)), Float32(1), None,
            lengths_i32, cu, pages, None, Int32(query.rows), False)
        if query.scored:
            # Programs with a sink or bias take output_gate, kv_tile_ranges, relative_bias, row_sink,
            # relative_factor and relative_projection after use_pdl.
            self.kernel(*arguments, output_gate, None, None, row_sink, relative_factor, relative_projection)
        else:
            self.kernel(*arguments, *((output_gate,) if output_gate is not None else ()))


def supports(q, key, value, pages, cu, output, *, extended=False):
    """Whether the TMA program serves this geometry; `extended` requests window, sink or bias operands."""
    dim = 128 if extended else 256
    return (torch.cuda.get_device_capability(q.device) in ((12, 0), (12, 1))
        and q.dtype in (torch.bfloat16, torch.float8_e4m3fn) and output.dtype == torch.bfloat16
        and q.shape[0] >= 128
        and q.shape[2] == key.shape[3] == value.shape[3] == dim
        and key.dtype == value.dtype == torch.float8_e4m3fn
        and key.shape[1] == value.shape[1] == 128 and pages.shape[0] == 1
        and pages.dtype == cu.dtype == torch.int32
        and all(t.is_contiguous() for t in (q, output, pages, cu))
        and all(t.stride()[1:] == (t.shape[2] * dim, dim, 1) for t in (key, value)))


class PreparedTma:
    def __init__(self, q, key, value, pages, lengths, cu, output, *, session, workspace, name, output_gate=None,
                 window=-1, row_sink=None, relative_factor=None, relative_projection=None):
        if output_gate is not None and (output_gate.shape != q.shape or output_gate.dtype != torch.bfloat16
                or output_gate.device != q.device or output_gate.stride(-1) != 1
                or any(stride % 8 for stride in output_gate.stride()[:-1]) or output_gate.data_ptr() % 16):
            raise ValueError("TMA output gate requires an aligned BF16 view matching query geometry")
        if (relative_factor is None) != (relative_projection is None):
            raise ValueError("a factored relative bias requires both its factor and projection")
        if type(window) is not int or window < -1:
            raise ValueError("TMA window must be -1 or a nonnegative number of previous keys")
        if row_sink is not None and (row_sink.dtype != torch.float32 or row_sink.shape != q.shape[:2]):
            raise ValueError("TMA row sinks must be FP32 [rows, heads]")
        if relative_factor is not None and (relative_factor.dtype != torch.bfloat16
                or relative_factor.shape != (*q.shape[:2], RELATIVE_RANK)
                or relative_projection.dtype != torch.bfloat16 or relative_projection.ndim != 2
                or relative_projection.shape[1] != RELATIVE_RANK or not relative_projection.is_contiguous()):
            raise ValueError("TMA relative factors must be BF16 [rows, heads, 16] with a contiguous [extent, 16] projection")
        self.output_gate = output_gate
        self.row_sink, self.relative_factor, self.relative_projection = row_sink, relative_factor, relative_projection
        geometry = Query(q.shape[0], q.shape[1], key.shape[2], q.shape[2], key.shape[1],
                         output_gate is not None, q.dtype == torch.float8_e4m3fn, window, row_sink is not None,
                         0 if relative_projection is None else relative_projection.shape[0])
        if geometry.extended and not supports(q, key, value, pages, cu, output, extended=True):
            raise ValueError("TMA window, sink and relative-bias operands require head dimension 128 and 128-token pages")
        self.query = geometry
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
                balanced_scheduler=True, kv_layout="NHD", with_output_gate=geometry.gated,
                with_row_sink=geometry.row_sink, window_left=geometry.window,
                with_factored_relative_bias=geometry.relative_extent > 0)
            return State(geometry, kernel)

        self.plan = native_plan(contract=CONTRACT, query=geometry, device=device,
            scratch_bytes=lambda tile: scratch_bytes, materialize=materialize)
        operands = (output_gate, row_sink, relative_factor, relative_projection)

        def call(state):
            scratch = planned_scratch((scratch_buffer_spec(CONTRACT.component_id,
                nbytes=scratch_bytes, device=device),), workspace)
            return PreparedCall(run=lambda: state.run(*self.arguments, scratch, *operands),
                owners=prime_owners(workspace, scratch, self.arguments, operands))

        declare(session, (self.plan.request(name=name, prepare_call=call),))
        register_finalizer(session, lambda: self.bind(workspace), workspace=workspace)

    def bind(self, workspace):
        self.scratch = planned_scratch(self.plan.scratch_specs(), workspace)
