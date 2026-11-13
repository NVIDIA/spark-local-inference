# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared expert routing and packed expert execution with caller-owned storage."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

import torch
from b12x.moe import fused_moe
from b12x.moe.fused_moe.workloads import TUNING_WORKLOAD_VERSION
from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall

from ignition.preparation.fixtures import fused_moe as moe_fixtures
from ignition.preparation.scratch import ScratchArena, planned_scratch, prime_owners
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import benchmark_resource, declare, register_finalizer


def _prepare_scale_expansion(state, *, ids, experts, workspace):
    call = moe_fixtures.scale_expansion_call(state, ids=torch.zeros_like(ids), experts=experts)
    return replace(call, owners=prime_owners(workspace, *call.owners))


@dataclass(frozen=True)
class RouterScratch:
    """Caller-owned selected logits, expert IDs, and FP32 route weights."""

    selected_logits: torch.Tensor
    ids: torch.Tensor
    weights: torch.Tensor

    @classmethod
    def allocate(cls, tokens: int, device, *, experts: int, top_k: int, buffers=None) -> RouterScratch:
        if (any(type(value) is not int or value <= 0 for value in (tokens, experts, top_k))
                or top_k > experts or experts > 1024):
            raise ValueError("router requires positive integer dimensions and top_k <= experts <= 1024")
        def zeros(dtype):
            return (torch.zeros((tokens, top_k), dtype=dtype, device=device) if buffers is None
                    else buffers.zeros((tokens, top_k), dtype))
        return cls(zeros(torch.float32), zeros(torch.int32), zeros(torch.float32).fill_(1 / top_k))

    def rows(self, tokens: int) -> RouterScratch:
        if type(tokens) is not int or not 0 < tokens <= self.ids.shape[0]:
            raise ValueError("router rows exceed scratch capacity")
        return RouterScratch(*(getattr(self, field)[:tokens] for field in self.__dataclass_fields__))


class PreparedRouter:
    """Native top-k routing, FP32 weights, and descending-ID ties.

    Route weights retain the library's FP32 precision without a BF16 roundtrip.
    The fixed route plan has no opaque scratch or mutable binding.
    A prepared image mask can select an image-specific correction bias per row
    within the same routing operation.
    A device-side valid-token count excludes the padded suffix of packed input;
    those rows receive expert ID -1 and zero weights.
    """

    def __init__(self, logits: torch.Tensor, scratch: RouterScratch, *,
                 session: PreparationSession, name: str, score_func="softmax",
                 renormalize=True, routed_scaling_factor=1.0, correction_bias=None,
                 image_correction_bias=None, image_mask=None, num_valid_tokens=None):
        if logits.ndim != 2 or logits.dtype not in (torch.bfloat16, torch.float32) or logits.stride(1) != 1:
            raise ValueError("router logits must be BF16 or FP32 [tokens, experts] with contiguous expert columns")
        tokens, experts = logits.shape
        shape = scratch.ids.shape
        if len(shape) != 2 or shape[0] != tokens or not 0 < shape[1] <= experts <= 1024:
            raise ValueError("router output geometry must match tokens and top_k <= experts <= 1024")
        for tensor, dtype in ((scratch.selected_logits, torch.float32),
                              (scratch.ids, torch.int32), (scratch.weights, torch.float32)):
            if tensor.shape != shape or tensor.dtype != dtype or tensor.device != logits.device or tensor.stride(1) != 1:
                raise ValueError("router outputs require matched shapes, native dtypes/device, and contiguous top-k columns")
        self.logits = logits
        self.scratch, self.device = scratch, logits.device
        self.dtype = logits.dtype
        self.options = {key: value for key, value in (
            ("correction_bias", correction_bias), ("image_correction_bias", image_correction_bias),
            ("image_mask", image_mask), ("num_valid_tokens", num_valid_tokens)) if value is not None}
        self.shape, self.stride = logits.shape, logits.stride()
        self.plan = fused_moe.plan_route_topk(fused_moe.RouteTopKInvocation(
            num_tokens=tokens, num_experts=experts, top_k=shape[1], logits_dtype=str(logits.dtype).removeprefix("torch."),
            score_func=score_func, renormalize=renormalize, routed_scaling_factor=routed_scaling_factor,
            has_correction_bias=correction_bias is not None,
            has_image_correction_bias=image_correction_bias is not None,
            has_image_mask=image_mask is not None,
            has_valid_tokens=num_valid_tokens is not None,
            logits_row_stride=logits.stride(0), topk_row_stride=scratch.selected_logits.stride(0),
            ids_row_stride=scratch.ids.stride(0), weights_row_stride=scratch.weights.stride(0)))
        options = dict(self.options)
        declare(session, (self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(logits, scratch.selected_logits, scratch.ids, scratch.weights, **options),
            owners=(logits, scratch))),))

    def functional(self, *, session):
        """Return route destinations allocated within the traced forward."""
        del session
        if self.scratch is not None:
            self.output_layouts = tuple((tuple(t.shape), t.stride(), t.dtype)
                                       for t in (self.scratch.selected_logits, self.scratch.ids, self.scratch.weights))
            self.dynamic_options = tuple(name for name in ("image_mask", "num_valid_tokens") if name in self.options)
            self.options = {name: value for name, value in self.options.items() if name not in self.dynamic_options}
            self.logits = self.scratch = None
        return self

    def empty_routes(self):
        return RouterScratch(*(torch.empty_strided(shape, stride, dtype=dtype, device=self.device)
                               for shape, stride, dtype in self.output_layouts))

    def runtime_options(self, *, image_mask=None, num_valid_tokens=None):
        options = dict(self.options)
        for name, value in (("image_mask", image_mask), ("num_valid_tokens", num_valid_tokens)):
            if name in self.dynamic_options:
                if value is None:
                    raise ValueError(f"functional router requires {name}")
                options[name] = value
            elif value is not None:
                raise ValueError(f"router was prepared without {name}")
        return options

    def __call__(self, logits: torch.Tensor, *, image_mask=None, num_valid_tokens=None):
        if (logits.dtype != self.dtype or logits.shape != self.shape
                or logits.stride() != self.stride or logits.device != self.device):
            raise ValueError("router logits must preserve the prepared dtype, shape, strides, and device")
        scratch = self.scratch
        options = self.options
        if scratch is None:
            scratch = self.empty_routes()
            options = self.runtime_options(image_mask=image_mask, num_valid_tokens=num_valid_tokens)
        fused_moe.route_topk(self.plan, logits, scratch.selected_logits,
                             scratch.ids, scratch.weights, **options)
        if self.scratch is None:
            return scratch


class PreparedRoutedExperts:
    """Prepared native expert execution with fixed source and route storage.

    Recipes supply expert geometry and enforce their complete model contract.
    Reduction order follows the execution plan selected by b12x.
    """

    def __new__(cls, source=None, experts=None, *args, **kwargs):
        # Experts loaded for SM103 execute through FlashInfer's native kernels. Unpickling, as compiler
        # workers do with a forward's metadata, creates the instance without arguments.
        from ignition.components.moe_sm103 import Sm103Experts, Sm103RoutedExperts

        if isinstance(experts, Sm103Experts):
            return Sm103RoutedExperts(source, experts, *args, **kwargs)
        return super().__new__(cls)

    def __init__(self, source, experts, ids, weights, output, *, session: PreparationSession, name: str,
                 workspace: ScratchArena | torch.Tensor | None = None, router_plan=None,
                 a4_prefill=False, routing=None, router=None, route_output=None):
        from b12x.moe.fused_moe._binding_ops import materialize_scale_templates

        materialize_scale_templates(experts)
        # b12x owns the final weight/scale planes after repacking. Classify those
        # exact tensors, not opaque execution scratch or the discarded source.
        for field in ("w1_fp4", "w2_fp4", "w1_blockscale", "w2_blockscale",
                      "w1_alphas", "w2_alphas", "a1_gscale", "a2_gscale"):
            record_numeric_tensor(getattr(experts._impl, field))
        representation = experts._impl.representation
        if representation is not None:
            # A plan can keep alternate scale layouts beside its canonical planes.
            for field in ("w13_scale", "w2_scale", "w13_sfb", "w2_sfb",
                          "w13_global_scale", "w2_global_scale", "micro_w13_scale", "micro_w2_scale",
                          "micro_w13_global_scale", "micro_w2_global_scale"):
                tensor = getattr(representation.value, field, None)
                if tensor is not None:
                    record_numeric_tensor(tensor)
        decoder = experts._impl.nvfp4_csf
        if decoder is not None:
            # Compressed records contain numeric offsets, not device pointers.
            for plane in (decoder.first, decoder.second):
                for tensor in (plane.fixed, plane.exceptions, plane.task_offsets, plane.value_lut):
                    if tensor is not None:
                        record_numeric_tensor(tensor)
            record_numeric_tensor(decoder.active)
            for plane in decoder.indexed_scales or ():
                record_numeric_tensor(plane.storage)
        # A16 experts prepared with A4 capability run large prefill calls with quantized activations.
        precision = {} if experts.plan.activation.outer_scale == "token" else {"a4_prefill": a4_prefill}
        self.source = source
        self.experts = experts
        self.preparation_arena = workspace
        self.num_experts = experts.num_experts
        self.precision = precision
        self.output = output
        self.output_source = output
        self.grouped_output = False
        if route_output is not None and experts.plan.activation.outer_scale == "token":
            from ignition.components.collectives import WeightedGroupSource

            output = route_output
            self.output = output
            self.output_source = WeightedGroupSource(output, weights, ids, experts.num_experts)
            self.grouped_output = True
        self.scratch = None
        self.router = router
        self.route_inputs = {}
        if router is not None and experts.plan.activation.outer_scale == "token":
            query = router.plan.query
            routing = replace(routing or fused_moe.RoutingSpec(),
                logits_dtype=router.dtype, score_func=query.score_func,
                renormalize=query.renormalize, routed_scaling_factor=query.routed_scaling_factor,
                has_correction_bias=query.has_correction_bias,
                has_image_correction_bias=query.has_image_correction_bias,
                has_image_mask=query.has_image_mask, has_valid_tokens=query.has_valid_tokens)
            self.route_inputs = dict(router_logits=router.logits,
                selected_logits=router.scratch.selected_logits, **router.options)
        integrated_query = router.plan.query if self.route_inputs else None
        invocation = FrozenMapping({**precision, "tuning_route_pattern": TUNING_WORKLOAD_VERSION,
            "tuning_route_source": "b12x",
            **({"tuning_execution": "cuda_graph"} if experts.has_csf_blocks else {})})
        self.scale_plan = fused_moe.plan_scale_expansion(
            experts, routes=ids.numel(), ids_dtype=ids.dtype, invocation=invocation,
        ) if experts.has_csf_blocks else None
        self.plan = fused_moe.plan_execution(
            experts=experts,
            capacity=fused_moe.ExecutionCapacity(
                max_tokens=source.shape[0], top_k=ids.shape[1],
                route_num_experts=experts.num_experts if self.route_inputs else 0,
                output_layout="routes" if output.ndim == 3 else "token"),
            scale_expansion=self.scale_plan, invocation=invocation, routing=routing,
        )
        route_input_templates = dict(self.route_inputs)

        def prepare_call(state):
            scratch = planned_scratch(state.scratch.scratch_specs(), workspace)
            self.scratch = scratch
            route_inputs = dict(route_input_templates)
            if route_inputs:
                route_inputs["router_logits"] = torch.zeros_like(route_inputs["router_logits"])
                route_inputs["selected_logits"] = torch.zeros_like(route_inputs["selected_logits"])
                if "num_valid_tokens" in route_inputs:
                    route_inputs["num_valid_tokens"] = torch.full_like(route_inputs["num_valid_tokens"], source.shape[0])
            binding = state.bind(a=source, topk_ids=torch.zeros_like(ids), topk_weights=torch.full_like(weights, 1 / ids.shape[1]),
                output=output, scratch=scratch, input_scales_static=True, **precision, **route_inputs)
            return PreparedCall(run=lambda: state.run(binding), owners=prime_owners(workspace, scratch, binding))

        scale_requests = () if self.scale_plan is None else (self.scale_plan.request(
            name=f"{name}/scales",
            prepare_call=partial(_prepare_scale_expansion, ids=ids, experts=experts, workspace=workspace),
            benchmark_call=partial(moe_fixtures.scale_expansion_benchmark_call,
                source=source, experts=experts, top_k=ids.shape[1], session=session, name=name,
                allocate_router=RouterScratch.allocate, router_plan=router_plan,
                integrated_query=integrated_query, output_template=output)),)
        declare(session, (*scale_requests, self.plan.request(name=name, prepare_call=prepare_call,
            dependencies=tuple(request.name for request in scale_requests),
            benchmark_call=partial(moe_fixtures.benchmark_call, source=source, experts=experts,
                top_k=ids.shape[1], session=session, name=name, workspace=workspace,
                allocate_router=RouterScratch.allocate, router_plan=router_plan, precision=precision,
                integrated_query=integrated_query, output_template=output)),))

        def finalize():
            self.scratch = planned_scratch(self.plan.scratch_specs(), workspace)
            self.binding = fused_moe.bind(self.plan, a=source, topk_ids=ids, topk_weights=weights,
                output=output, scratch=self.scratch, input_scales_static=True, **precision, **route_input_templates)

        register_finalizer(session, finalize, workspace=workspace)

    def functional(self, *, session):
        """Keep prepared kernels and allocate routed outputs and scratch in the forward."""
        if self.output is not None:
            if self.experts.has_csf_blocks and not isinstance(self.preparation_arena, ScratchArena):
                raise ValueError("functional compressed experts require an arena preparation-release boundary")
            self.output_layout = (tuple(self.output.shape), self.output.stride(), self.output.dtype)
            self.integrated_routing = bool(self.route_inputs)
            self.output = self.output_source = self.source = None
            if self.router is not None:
                self.router.functional(session=session)
            register_finalizer(session, self._finalize_functional, workspace=self.preparation_arena)
        return self

    def _finalize_functional(self):
        from b12x.moe.fused_moe._binding_ops import scale_buffers, release_scale_templates, release_weight_views

        specs = self.plan.scratch_specs()
        if len(specs) != 1 or specs[0].dtype != torch.uint8:
            raise ValueError("functional experts require one prepared byte workspace")
        self.workspace_nbytes = specs[0].nbytes
        self.scale_inputs, storage = scale_buffers(self.binding)
        self.scale_layouts = tuple((tuple(t.shape), t.stride(), t.dtype) for t in storage)
        release_weight_views(self.experts._impl)
        if self.experts.has_csf_blocks and self.preparation_arena is not None:
            self.preparation_arena.register_preparation_release(
                partial(release_scale_templates, self.experts, self.plan))
        self.scratch = self.binding = None
        self.route_inputs = {}
        self.preparation_arena = None

    def route(self, logits: torch.Tensor, *, image_mask=None, num_valid_tokens=None):
        if self.output is None:
            if self.integrated_routing:
                return self.router.empty_routes()
            return self.router(logits, image_mask=image_mask, num_valid_tokens=num_valid_tokens)
        if self.route_inputs:
            if logits is not self.router.logits:
                raise ValueError("integrated expert routing requires the prepared logit storage")
        else:
            self.router(logits)

    def __call__(self, source=None, ids=None, weights=None, *, router_logits=None, selected_logits=None,
                 image_mask=None, num_valid_tokens=None):
        if self.output is None:
            from b12x.moe.fused_moe._binding_ops import prepared_moe_out, prepared_routed_moe_out

            if source is None or ids is None or weights is None:
                raise ValueError("functional experts require source, expert IDs, and route weights")
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
            scratch = torch.empty(self.workspace_nbytes, dtype=torch.uint8, device=source.device)
            storage = [torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
                       for shape, stride, dtype in self.scale_layouts]
            if self.integrated_routing:
                if router_logits is None or selected_logits is None:
                    raise ValueError("integrated experts require router logits and selected-logit destinations")
                options = self.router.runtime_options(image_mask=image_mask, num_valid_tokens=num_valid_tokens)
                prepared_routed_moe_out(source, router_logits, selected_logits, weights, ids, output,
                    scratch, self.scale_inputs, storage, self.plan.handle,
                    options.get("correction_bias"), options.get("image_correction_bias"),
                    options.get("image_mask"), options.get("num_valid_tokens"), False)
            else:
                prepared_moe_out(source, weights, ids, output, scratch, self.scale_inputs, storage,
                    self.plan.handle, True, None, None, None, None, None, False,
                    self.precision.get("a4_prefill"))
            if self.grouped_output:
                from ignition.components.collectives import WeightedGroupSource

                return WeightedGroupSource(output, weights, ids, self.num_experts)
            return output
        fused_moe.run(binding=self.binding)
        return self.output




def record_views(records, width, format):
    """Views `(values, block_scales, row_amax)` into `[rows, record]` uint8 expert-input records.

    The offsets are `fused_moe.quantized_input_record_layout`'s. The record rows
    may lie inside wider rows of one buffer, such as packets that also carry the
    rows' routes; the quantizer and both expert phases take any constant row
    stride and validate 16-byte alignment when they run.
    """
    values, scales, amax, record = fused_moe.quantized_input_record_layout(width, format)
    if records.ndim != 2 or records.shape[1] != record or records.dtype != torch.uint8 or records.stride(1) != 1:
        raise ValueError(f"expert-input records must be uint8 [rows, {record}] with dense rows")
    if format == "mxfp8":
        return records[:, values:values + width], records[:, scales:scales + width // 32], None
    return (records[:, values:values + width // 2], records[:, scales:scales + width // 16],
            records[:, amax:amax + 4].view(torch.float32)[:, 0])


class PreparedRowQuantization:
    """Prepared quantization of BF16 activation rows into packed expert-input records.

    One `[rows, record]` byte buffer holds each row's packed values, block scales
    and, for NVFP4, its FP32 amax (`fused_moe.quantized_input_record_layout`).
    `nvfp4_four_over_six` encodes with outer scale `amax / 1536` and four-over-six
    block scales; `mxfp8` with one UE8M0 scale per 32 values. A rank that owns
    token rows quantizes them once, and the records move between ranks as bytes.
    """

    def __init__(self, source, records, *, format, session: PreparationSession, name: str):
        rows, width = source.shape
        *_, record = fused_moe.quantized_input_record_layout(width, format)
        if source.dtype != torch.bfloat16 or not source.is_contiguous():
            raise ValueError("row quantization reads contiguous BF16 rows")
        if records.shape != (rows, record) or records.dtype != torch.uint8:
            raise ValueError(f"quantized records must be uint8 [rows, {record}]")
        self.width, self.format, self.record = width, format, record
        self.source, self.records = source, records
        self.plan = fused_moe.plan_row_quantization(fused_moe.RowQuantizationInvocation(
            max_rows=rows, width=width, format=format))
        views = fused_moe.quantized_input_record_views(records, width, format)
        declare(session, (self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(source, *views), owners=(source, records))),))

    def functional(self, *, session):
        """Allocate records within the traced forward."""
        del session
        self.source = self.records = None
        return self

    def views(self, records):
        return record_views(records, self.width, self.format)

    def __call__(self, source, records=None):
        if records is None:
            records = torch.empty((source.shape[0], self.record), dtype=torch.uint8, device=source.device)
        fused_moe.quantize_rows(self.plan, source, *self.views(records))
        return records


@dataclass(frozen=True)
class PendingExperts:
    """Phase-one results that phase two consumes: shared scratch and per-route activation amax."""

    scratch: torch.Tensor
    row_amax: torch.Tensor | None


class PreparedTwoPhaseExperts:
    """Experts on quantized input records whose FC1 and FC2 run as separate prepared phases.

    Each route's activation is multiplied by its routing weight before the FC2
    input is quantized, and routes sum with unit weights; inactive routes (IDs
    outside the expert range) contribute zero. Under four-over-six NVFP4
    activations, phase one also returns each route's activation amax and phase
    two encodes the activation with a supplied amax. A rank that holds a slice of
    every expert's intermediate channels supplies the maximum of all ranks' amax,
    so its slice encodes exactly its columns of the unsliced activation. The FC
    gains divide by the checkpoint's weight-scale `denominators`.

    Phase one leaves packed routes and the activation in scratch that phase two
    reads; no operation between the phases may reuse that scratch.
    """

    def __init__(self, experts, records, ids, weights, output, *, format, session: PreparationSession, name: str,
                 denominators=None, workspace: ScratchArena | None = None):
        from b12x.moe.fused_moe.workloads import make_tuning_routes

        tokens, hidden = output.shape
        if ids.shape != weights.shape or ids.shape[0] != tokens or ids.dtype != torch.int32 or weights.dtype != torch.float32:
            raise ValueError("expert routes require int32 IDs and FP32 weights of matching [tokens, top_k] shape")
        self.nvfp4 = format == "nvfp4_four_over_six"
        if self.nvfp4 != (denominators is not None):
            raise ValueError("four-over-six NVFP4 experts require their weight-scale denominators")
        self.format, self.hidden, self.top_k = format, hidden, ids.shape[1]
        self.experts = experts
        self.plan = fused_moe.plan_execution(
            experts=experts, capacity=fused_moe.ExecutionCapacity(max_tokens=tokens, top_k=ids.shape[1]),
            routing=fused_moe.RoutingSpec(apply_router_weight_on_activation=True,
                                          **({} if denominators is None else
                                             {"weight_scale_denominators": tuple(denominators)})))
        self.records, self.ids, self.weights, self.output = records, ids, weights, output

        def bound(state, scratch, records, ids, weights, output):
            values, scales, amax = fused_moe.quantized_input_record_views(records, hidden, format)
            return state.bind(a=values, input_block_scales=scales, input_row_scale=amax, topk_ids=ids,
                              topk_weights=weights, output=output, scratch=scratch)

        def prepare_call(state):
            scratch = planned_scratch(state.scratch_specs(), workspace)
            binding = bound(state, scratch, records, torch.zeros_like(ids), torch.ones_like(weights), output)
            return PreparedCall(run=binding.run, owners=prime_owners(workspace, scratch, binding))

        def benchmark_call(state):
            # Tile races time BF16 rows, which phase one quantizes, over the shared tuning route mix.
            def inputs():
                generator = torch.Generator(device="cpu").manual_seed(7919)
                rows = torch.randn((tokens, hidden), generator=generator).to(device=output.device, dtype=torch.bfloat16)
                routes = make_tuning_routes(tokens, ids.shape[1], experts.num_experts, device=output.device)
                return dict(rows=rows, routes=routes.to(torch.int32), weights=torch.ones_like(weights),
                            output=torch.empty_like(output))

            operands = benchmark_resource(session, ("two_phase_experts", name), inputs)
            scratch = planned_scratch(state.scratch_specs(), workspace)
            bindings = [state.bind(a=operands["rows"], topk_ids=route, topk_weights=operands["weights"],
                                   output=operands["output"], scratch=scratch) for route in operands["routes"]]
            index = [0]

            def run():
                bindings[index[0] % len(bindings)].run()
                index[0] += 1

            # Phase one reads the routes and rows without mutating them, so trials need no fresh inputs.
            return PreparedCall(run=run, produce=lambda: None,
                                owners=(*prime_owners(workspace, scratch, bindings), operands),
                                output=operands["output"])

        declare(session, (self.plan.request(name=name, prepare_call=prepare_call, benchmark_call=benchmark_call),))

        def finalize():
            specs = self.plan.scratch_specs()
            if len(specs) != 1 or specs[0].dtype != torch.uint8:
                raise ValueError("two-phase experts require one prepared byte workspace")
            self.workspace_nbytes = specs[0].nbytes

        register_finalizer(session, finalize, workspace=workspace)

    def functional(self, *, session):
        """Allocate scratch, amax and outputs within the traced forward."""
        del session
        if self.output is not None:
            self.output_layout = (tuple(self.output.shape), self.output.dtype)
            self.records = self.ids = self.weights = self.output = None
        return self

    def views(self, records):
        return record_views(records, self.hidden, self.format)

    def phase1(self, records, ids, weights) -> PendingExperts:
        """Pack routes, project and activate them; return the scratch and each route's activation amax."""
        from b12x.moe.fused_moe._binding_ops import prepared_moe_phase1_out

        scratch = torch.empty(self.workspace_nbytes, dtype=torch.uint8, device=records.device)
        row_amax = (torch.empty(ids.numel(), dtype=torch.float32, device=records.device) if self.nvfp4 else None)
        prepared_moe_phase1_out(*self.views(records), weights, ids, row_amax, scratch, self.plan.handle)
        return PendingExperts(scratch, row_amax)

    def phase2(self, pending: PendingExperts, ids, weights, row_amax=None):
        """Quantize the activation with `row_amax` (every rank's maximum), project it and sum routes."""
        from b12x.moe.fused_moe._binding_ops import prepared_moe_phase2_out

        if self.nvfp4 != (row_amax is not None) or row_amax is not None and row_amax.shape != (ids.numel(),):
            raise ValueError("four-over-six experts require one FP32 amax per route; MXFP8 experts take none")
        shape, dtype = self.output_layout
        output = torch.empty((ids.shape[0], shape[1]), dtype=dtype, device=ids.device)
        prepared_moe_phase2_out(weights, ids, row_amax, output, pending.scratch, self.plan.handle)
        return output
