# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""One rank's Inkling 2 router and expert slice.

Every rank holds the router and an intermediate slice of all 384 experts. The
router projects the normalized MLP input to FP32 logits, selects the eight
largest `sigmoid(logit) + bias` and weights them by their renormalized
sigmoids times `route_scale`. The MLP input is quantized once per row into
packed records (NVFP4 four-over-six on layers whose experts are NVFP4, MXFP8
otherwise). Phase one runs each route's `w13` slice and the capped SiTU
activation multiplied by the route's weight, and returns each route's
activation amax over the slice. NVFP4 phase two quantizes the activation with
the maximum of every rank's amax, which the caller supplies, then runs `w2` and
sums the routes with unit weights into this rank's partial output.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ignition.components.linear import UnquantizedProjection
from ignition.components.moe import PreparedRouter, PreparedRowQuantization, PreparedTwoPhaseExperts, RouterScratch

QUANTIZATION = {"nvfp4": "nvfp4_four_over_six", "mxfp8": "mxfp8"}


@dataclass(frozen=True)
class ExpertWeights:
    router: torch.Tensor
    router_bias: torch.Tensor
    experts: object
    format: str
    denominators: tuple[float, float] | None


def prepare_experts(placed, *, format, hidden, intermediate):
    """b12x phase-separated experts from one rank's placed expert tensors.

    `w13` rows interleave gate (even) and up (odd) projections of each
    intermediate channel; b12x's NVFP4 source takes gate rows then up rows and
    its MXFP8 source up rows then gate rows. NVFP4 global scales are the
    per-expert weight amax, divided by the checkpoint's outer-scale denominators
    in the FC gains. No value or scale byte changes.
    """
    from b12x._lib.intrinsics.reference import swizzle_block_scale
    from b12x.moe import fused_moe as moe

    from ignition.preparation.persistent import persistent_empty

    w13, s13 = placed["w13"], placed["w13_scale"]
    experts, rows = w13.shape[:2]
    if rows % 2:
        raise ValueError("interleaved expert projections require gate and up rows in pairs")
    gate, up = (slice(0, None, 2), slice(1, None, 2))
    order = (gate, up) if format == "nvfp4" else (up, gate)

    def deinterleave(tensor, allocate):
        output = allocate(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        half = rows // 2
        for index, rows_of in enumerate(order):
            output[:, index * half:(index + 1) * half].copy_(tensor[:, rows_of])
        return output

    geometry = moe.MoEGeometry(num_experts=experts, hidden_size=hidden, intermediate_size=intermediate)
    if format == "nvfp4":
        plan = moe.plan_weights(
            source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w31"),
            activation=moe.ActivationSpec(mode="a4", nonlinearity="situ", io_dtype=torch.bfloat16,
                                          outer_scale="token", block_scale="four_over_six"),
            geometry=geometry)
        weights = moe.PackedWeights(
            w13=deinterleave(w13, persistent_empty), w2=placed["w2"],
            w13_block_scales=swizzle_block_scale(deinterleave(s13, torch.empty), allocate=persistent_empty),
            w2_block_scales=swizzle_block_scale(placed["w2_scale"], allocate=persistent_empty),
            w13_global_scales=placed["w13_amax"], w2_global_scales=placed["w2_amax"])
        denominators = (float(placed["w13_denominator"].item()), float(placed["w2_denominator"].item()))
    elif format == "mxfp8":
        plan = moe.plan_weights(
            source=moe.PackedSource(format=moe.PackedSourceFormat.MXFP8_E8M0_K32, w13_layout=moe.W13Layout.W13),
            activation=moe.ActivationSpec(mode=moe.ActivationMode.A8, nonlinearity="situ", io_dtype=torch.bfloat16),
            geometry=geometry)
        ones = lambda: torch.ones(experts, dtype=torch.float32, device=w13.device)  # noqa: E731
        weights = moe.PackedWeights(
            w13=deinterleave(w13.view(torch.uint8), persistent_empty), w2=placed["w2"].view(torch.uint8),
            w13_block_scales=deinterleave(s13, persistent_empty), w2_block_scales=placed["w2_scale"],
            w13_global_scales=ones(), w2_global_scales=ones(), input_scale=ones(), intermediate_scale=ones(),
            immutable_input_scales=True)
        denominators = None
    else:
        raise ValueError(f"unsupported Inkling expert format {format!r}")
    return moe.prepare_weights(plan=plan, weights=weights, allocate=persistent_empty), denominators


class PreparedExperts:
    """A rank's router, MLP-input quantization and two expert phases for one layer.

    Routing and quantization serve `route_tokens` rows (by default `tokens`);
    the expert phases serve `tokens` rows.
    """

    def __init__(self, weights: ExpertWeights, text, *, tokens, session, workspace, name, examples=None,
                 route_tokens=None):
        from b12x.moe import fused_moe as moe

        from ignition.preparation.examples import declaration_tensor

        examples = {} if examples is None else examples
        device = weights.router.device
        quantization = QUANTIZATION[weights.format]
        self.format = weights.format
        route_tokens = tokens if route_tokens is None else route_tokens

        def example(role, shape, dtype=torch.bfloat16):
            return declaration_tensor(examples, role, shape, dtype, device)

        source = example("inkling.experts.source", (route_tokens, text.hidden))
        logits = example("inkling.experts.logits", (route_tokens, text.experts), torch.float32)
        self.router_projection = UnquantizedProjection(source, weights.router, logits, session=session,
                                                       name=f"{name}.router").functional(session=session)
        routes = RouterScratch.allocate(route_tokens, device, experts=text.experts, top_k=text.experts_per_token)
        valid = example("inkling.experts.valid", (1,), torch.int32)
        self.router = PreparedRouter(logits, routes, session=session, name=f"{name}.route", score_func="sigmoid",
            renormalize=True, routed_scaling_factor=text.route_scale, correction_bias=weights.router_bias,
            num_valid_tokens=valid).functional(session=session)
        self.record = moe.quantized_input_record_layout(text.hidden, quantization)[-1]
        records = example(f"inkling.experts.records.{quantization}", (route_tokens, self.record), torch.uint8)
        self.quantization = PreparedRowQuantization(source, records, format=quantization, session=session,
                                                    name=f"{name}.quantize").functional(session=session)
        records = example(f"inkling.experts.records.{quantization}", (tokens, self.record), torch.uint8)
        ids = example("inkling.experts.ids", (tokens, text.experts_per_token), torch.int32)
        route_weights = example("inkling.experts.weights", (tokens, text.experts_per_token), torch.float32)
        output = example("inkling.experts.output", (tokens, text.hidden))
        self.experts = PreparedTwoPhaseExperts(weights.experts, records, ids, route_weights, output,
            format=quantization, denominators=weights.denominators, session=session, workspace=workspace,
            name=f"{name}.experts").functional(session=session)

    @property
    def supplies_amax(self):
        """Whether phase two takes the maximum of every rank's per-route activation amax."""
        return self.experts.nvfp4

    def route(self, source, num_valid_tokens):
        """Expert IDs (`-1` past the valid rows) and FP32 routing weights."""
        routes = self.router(self.router_projection(source), num_valid_tokens=num_valid_tokens)
        return routes.ids, routes.weights

    def quantize(self, source, records=None):
        """Expert-input records of `source` rows, written into `records` when given."""
        return self.quantization(source, records)

    def phase1(self, records, ids, weights):
        return self.experts.phase1(records, ids, weights)

    def phase2(self, pending, ids, weights, row_amax=None):
        return self.experts.phase2(pending, ids, weights, row_amax)
