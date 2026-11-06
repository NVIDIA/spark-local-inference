# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM103 tensor-FP8 GEMMs with original E4M3 values and FP32 output calibration."""

from dataclasses import dataclass

import torch
from b12x.preparation import plan_from_handle, require_prepared

from .native_operator import native_contract, native_plan

COMPONENT = "flare.sm103.tensor_fp8_projection"
# The pinned CUTLASS getConfigs enumerates six CTA tiles and five cluster shapes.
_TACTICS = tuple(range(6 * 5))


@dataclass(frozen=True)
class TensorFP8Query:
    rows: int
    in_features: int
    padded_in_features: int
    out_features: int


def _runner():
    from flashinfer.gemm.gemm_base import get_gemm_sm100_module_cutlass_fp8

    return get_gemm_sm100_module_cutlass_fp8().cutlass_fp8_gemm_runner()


CONTRACT = native_contract(component_id=COMPONENT, query_type=TensorFP8Query, tactics=lambda query: _TACTICS)


def _layout(query):
    from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE

    output = 256 + DEFAULT_WORKSPACE_SIZE
    padded = (output + query.rows * query.out_features * 2 + 255) // 256 * 256
    size = padded + (query.rows * query.padded_in_features
                     if query.in_features != query.padded_in_features else 0)
    return output, padded, size


class PreparedTensorFP8:
    def __init__(self, query, tactic):
        self.query, self.tactic, self.runner = query, tactic, _runner()
        if tuple(self.runner.get_valid_tactics([], None)) != _TACTICS:
            raise RuntimeError("FlashInfer tensor-FP8 tactics differ from the declared CUTLASS catalog")
        self.required_workspace = _layout(query)[2]

    def run(self, source, weight, scale, output, workspace):
        q = self.query
        output_offset, padded_offset, _ = _layout(q)
        unit = workspace[:4].view(torch.float32)
        unit.fill_(1)
        if q.in_features != q.padded_in_features:
            padded = workspace[padded_offset:padded_offset + q.rows * q.padded_in_features]
            padded = padded.view(torch.float8_e4m3fn).view(q.rows, q.padded_in_features)
            padded.zero_()
            padded[:, :q.in_features].copy_(source)
            source = padded
        result = workspace[output_offset:output_offset + q.rows * q.out_features * 2]
        result = result.view(torch.bfloat16).view(q.rows, q.out_features)
        # Preserve the already-rounded combined scale; multiplying by one is exact.
        self.runner(inputs=[source, weight.T, scale, unit, result, workspace[256:output_offset]],
                    tactic=self.tactic)
        output.copy_(result)
        return output


def plan(rows, weight, device):
    query = TensorFP8Query(rows, weight.in_features, weight.padded_in_features, weight.out_features)
    if (rows <= 0 or query.in_features <= 0 or query.in_features % 32
            or query.padded_in_features % 128 or query.in_features > query.padded_in_features
            or query.out_features <= 0 or query.out_features % 8):
        raise ValueError("SM103 tensor-FP8 GEMMs require positive rows, K32 with stored K128, and N8")
    return native_plan(contract=CONTRACT, query=query, device=device,
        scratch_bytes=lambda tactic: _layout(query)[2],
        materialize=lambda tactic: PreparedTensorFP8(query, tactic))


@torch.library.custom_op("flare::sm103_tensor_fp8_out", mutates_args=("output", "workspace"))
def project(source: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
            output: torch.Tensor, workspace: torch.Tensor, handle: int) -> None:
    require_prepared(plan_from_handle(handle), COMPONENT, source.device).run(
        source, weight, scale, output, workspace)


@project.register_fake
def _fake_project(source, weight, scale, output, workspace, handle):
    pass
