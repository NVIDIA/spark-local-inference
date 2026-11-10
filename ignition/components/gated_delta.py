# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared Qwen GDN projections, convolution, recurrence, and output gating."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from functools import cached_property

import torch
from b12x.gemm.blockscaled import Weight
from b12x.preparation import PreparationSession
from b12x.sequence import gdn_decode, gdn_prefill, short_conv
from b12x.sequence.gdn_decode._kernels import _gdn_decode_op as gdn_decode_out
from b12x.sequence.gdn_prefill._ops import gdn_prefill_out
from b12x.sequence.short_conv.api import _pipeline as short_conv_out

from ignition.components.gated_norm import PreparedGatedRMSNorm
from ignition.components.linear import QuantizedProjection, UnquantizedProjection
from ignition.components.tensor_fp8 import TensorFP8Projection, TensorFP8Weight
from ignition.components.recurrent import (
    DeltaMetadata,
    DeltaState,
    prepare_sequence_binding,
)
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import register_finalizer

from ignition.components import sm103
from ignition.components.gdn_prefill_sm103 import PreparedGdnPrefill, PrefillBuffers

from .gdn_recovery import GDNRecords, PreparedRecoveredRecurrence


@dataclass(frozen=True)
class GDNState(DeltaState):
    """Persistent chronological convolution history and transposed FP32 recurrence.

    Recurrence slots have shape ``[slot, local_value_heads, 128, 128]``.
    The canonical Transformers cache transposes the final two dimensions.
    Slot zero is reserved as the recurrence null index. Fresh decode slots must
    be zeroed by cache maintenance before replay; prefill selects null initial
    state through metadata. Convolution fresh flags discard prior history.
    """

    @classmethod
    def allocate(
        cls, slots: int, device: torch.device, *, speculative_tokens: int = 0,
        world_size: int = 1,
    ) -> GDNState:
        return super().allocate(slots, device, heads=48 // world_size, channels=10240 // world_size,
                                speculative_tokens=speculative_tokens)


@dataclass(frozen=True)
class GDNScratch:
    """Caller-owned graph operands reusable across serialized decoder layers."""

    projected_qkv: torch.Tensor
    convolved_qkv: torch.Tensor
    z: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    recurrence: torch.Tensor
    gated: torch.Tensor
    output: torch.Tensor
    projected_qkvz: torch.Tensor
    projected_ba: torch.Tensor
    # Contiguous heads and gates for SM103's chunked prefill (see gdn_prefill_sm103.PrefillBuffers).
    prefill_query: torch.Tensor
    prefill_key: torch.Tensor
    prefill_value: torch.Tensor
    prefill_decay: torch.Tensor
    prefill_beta: torch.Tensor

    @classmethod
    def allocate(cls, tokens: int, device: torch.device, *, world_size=1, hidden_size=2560, buffers=None,
                 output=None, pack_ba=False) -> GDNScratch:
        if type(tokens) is not int or tokens <= 0:
            raise ValueError("GDN token capacity must be positive")
        def zeros(width):
            return (torch.zeros((tokens, width), device=device, dtype=torch.bfloat16) if buffers is None
                    else buffers.zeros((tokens, width), torch.bfloat16))
        if pack_ba:
            packed = zeros(16480 // world_size)
            qkvz, ba = packed.split((16384 // world_size, 96 // world_size), dim=-1)
        else:
            qkvz = zeros(16384 // world_size)
            ba = zeros(96 // world_size)
        convolved, recurrence, gated = (
            zeros(width)
            for width in (10240 // world_size, 6144 // world_size, 6144 // world_size))
        output = zeros(hidden_size) if output is None else output
        prefill = PrefillBuffers.allocate(tokens if sm103.is_sm103(device) else 0, device,
                                         key_heads=16 // world_size, value_heads=48 // world_size,
                                         buffers=buffers)
        return cls(qkvz[:, :10240 // world_size], convolved, qkvz[:, 10240 // world_size:], ba[:, 48 // world_size:], ba[:, :48 // world_size],
                   recurrence, gated, output, qkvz, ba, prefill.query, prefill.key, prefill.value, prefill.decay,
                   prefill.beta)

    @cached_property
    def mxfp8_output(self):
        return self.allocate_mxfp8_output()

    def allocate_mxfp8_output(self, *, buffers=None):
        from b12x.quantization import mxfp8

        rows, columns = self.gated.shape
        shapes = mxfp8.dense_rows_storage_shapes(rows, columns)
        bases = tuple(torch.zeros(shape, dtype=dtype, device=self.gated.device) if buffers is None
                      else buffers.zeros(shape, dtype)
                      for shape, dtype in zip(shapes, (torch.float8_e4m3fn, torch.uint8, torch.uint8), strict=True))
        bases[1].fill_(127)
        bases[2].fill_(127)
        return mxfp8.dense_rows_from_bases(*bases)

    def rows(self, tokens: int) -> GDNScratch:
        if type(tokens) is not int or not 0 < tokens <= self.projected_qkv.shape[0]:
            raise ValueError("GDN rows exceed scratch capacity")
        return GDNScratch(
            *(getattr(self, field.name)[:tokens] for field in fields(self))
        )


def _validate_geometry(module: torch.nn.Module, world_size=1) -> None:
    required = {
        "num_k_heads": 16 // world_size,
        "num_v_heads": 48 // world_size,
        "head_k_dim": 128,
        "head_v_dim": 128,
        "conv_dim": 10240 // world_size,
        "conv_kernel_size": 4,
    }
    if any(getattr(module, name, None) != value for name, value in required.items()):
        raise ValueError("GDN geometry requires 16 key heads, 48 value heads, and 128-element head dimensions before sharding")
    if module.activation != "silu" or module.norm.activation not in ("silu", "sigmoid"):
        raise ValueError(
            "Qwen GDN requires SiLU convolution and SiLU/sigmoid output gating"
        )
    if module.conv1d.bias is not None:
        raise ValueError("Qwen GDN convolution must not have bias")


class PreparedGDN:
    """One fixed-capacity decode or prefill pipeline over a canonical GDN module.

    The module supplies geometry and exact unquantized parameters. The projection
    mapping supplies calibrated QKV/Z and output weights with packed or BF16 B/A
    projections. Preparation
    requires zero live counts and must finish before graph capture. Bound calls
    retain caller scratch; ``functional`` keeps selected plans, weights, metadata,
    and persistent state while tracing allocations for sequence intermediates.

    ``workspace`` and ``projection_workspace`` may be shared by serialized
    layers and graph buckets. Their storage must cover every selected plan and
    remain live until all referencing graphs drain. Concurrent graph execution
    over either shared workspace is invalid.

    Prefill preserves the gated norm's BF16 rounding and optional MXFP8 output
    encoding. Captured execution requires no Python callback.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        projections: Mapping[str, Weight],
        source: torch.Tensor,
        scratch: GDNScratch,
        state: GDNState,
        metadata: DeltaMetadata,
        *,
        mode: str,
        session: PreparationSession,
        name: str,
        checkpoint_export: bool = False,
        workspace: torch.Tensor | None = None,
        projection_workspace: torch.Tensor | None = None,
        projection_staging: torch.Tensor | None = None,
        records: GDNRecords | None = None,
        in_place: bool = False,
        partitions=None,
        prepare_projections: bool = True,
        gdn_buffers=None,
    ) -> None:
        self.world_size = 48 // module.num_v_heads
        self.key_heads, self.value_heads = module.num_k_heads, module.num_v_heads
        world_size = self.world_size
        _validate_geometry(module, world_size)
        if mode not in ("decode", "prefill"):
            raise ValueError("GDN graph mode must be decode or prefill")
        self.module, self.scratch, self.state, self.metadata = (
            module,
            scratch,
            state,
            metadata,
        )
        self.mode = mode
        self._binding_workspace = workspace
        self.tokens, self.device = source.shape[0], source.device
        if (
            source.ndim != 2
            or source.shape[1] != module.hidden_size
            or source.dtype != torch.bfloat16
        ):
            raise ValueError("GDN source must be BF16 with the module's hidden width")
        if self.device.type != "cuda":
            raise ValueError("prepared GDN requires CUDA")
        widths = tuple(width if index == 7 else width // world_size for index, width in
                       enumerate((10240, 10240, 6144, 48, 48, 6144, 6144, module.hidden_size, 16384, 96)))
        qkvz_stride, ba_stride = scratch.projected_qkvz.stride(0), scratch.projected_ba.stride(0)
        if qkvz_stride < 16384 // world_size or ba_stride < 96 // world_size:
            raise ValueError("GDN projection row strides must cover their complete outputs")
        strides = (qkvz_stride, 10240 // world_size, qkvz_stride, ba_stride, ba_stride,
                   6144 // world_size, 6144 // world_size, module.hidden_size, qkvz_stride, ba_stride)
        # Projection operands; the chunked-prefill buffers are validated where they are used.
        projected = [field for field in fields(scratch) if not field.name.startswith("prefill_")]
        for field, width, stride in zip(projected, widths, strides, strict=True):
            tensor = getattr(scratch, field.name)
            if (
                tensor.device != self.device
                or tensor.dtype != torch.bfloat16
                or tuple(tensor.shape) != (self.tokens, width)
                or tensor.stride() != (stride, 1)
            ):
                raise ValueError(
                    f"GDN scratch {field.name} has incompatible shape, dtype, or layout"
                )
        for tensor in (
            module.conv1d.weight,
            module.A_log,
            module.dt_bias,
            module.norm.weight,
        ):
            if tensor.device != self.device:
                raise ValueError(
                    "canonical GDN parameters and graph buffers must share a device"
                )
        if state.convolution.ndim != 3 or state.convolution.shape[1] != module.conv_dim:
            raise ValueError(
                f"GDN convolution history must have shape [slots, {module.conv_dim}, 3 + drafts]"
            )
        self.speculative_tokens = state.convolution.shape[2] - 3
        if not 0 <= self.speculative_tokens <= 15:
            raise ValueError(
                "GDN history supports zero through fifteen speculative tokens"
            )
        if state.recurrent.ndim != 4 or tuple(state.recurrent.shape[1:]) != (
            self.value_heads,
            128,
            128,
        ):
            raise ValueError(f"GDN recurrence must have shape [slots, {self.value_heads}, 128, 128]")
        self.sequences = metadata.query_start_loc.numel() - 1
        if (
            self.sequences <= 0
            or metadata.state_indices.ndim != 2
            or metadata.state_indices.shape[0] != self.sequences
        ):
            raise ValueError("GDN metadata capacities disagree")
        columns = metadata.state_indices.shape[1]
        if mode == "decode" and (
            self.tokens > self.sequences * columns
            or columns > self.speculative_tokens + 1
        ):
            raise ValueError(
                "decode token/verification capacity exceeds its convolution history"
            )
        if metadata.num_tokens.item() or metadata.num_seqs.item():
            raise ValueError(
                "GDN preparation requires zero live token and sequence counts"
            )
        self.fused_qkvzba = "in_proj_qkvzba" in projections
        self.fused_qkvz = "in_proj_qkvz" in projections
        expected = ({"in_proj_qkvzba"} if self.fused_qkvzba else
                    {"in_proj_qkvz", "in_proj_ba"} if self.fused_qkvz else
                    {"in_proj_qkv", "in_proj_z", "in_proj_ba"})
        expected.add("out_proj")
        if set(projections) != expected:
            raise ValueError("GDN requires QKV/Z, packed B/A, and output projection weights")

        from b12x.gemm.blockscaled._linear import MXFP8LinearWeight

        self.output_encoding = None
        if (prepare_projections and mode == "prefill" and partitions is None
                and not sm103.is_sm103(self.device) and isinstance(projections["out_proj"], MXFP8LinearWeight)):
            self.output_encoding = (scratch.mxfp8_output if gdn_buffers is None
                                   else scratch.allocate_mxfp8_output(buffers=gdn_buffers))

        def projection(key, inputs, output):
            weight = projections[key]
            options = dict(session=session, name=f"{name}.{key}")
            if isinstance(weight, TensorFP8Weight):
                return TensorFP8Projection(inputs, weight, output, staging=projection_staging, **options)
            if isinstance(weight, torch.Tensor):
                return UnquantizedProjection(inputs, weight, output, **options)
            return QuantizedProjection(inputs, weight, output, activation_mode="auto",
                                       allow_bf16_atomic_reduction=False if key == "in_proj_qkvzba" else None,
                                       workspace=projection_workspace, encoded=self.output_encoding if key == "out_proj" else None, **options)

        if prepare_projections:
            if self.fused_qkvzba:
                width = 16480 // world_size
                if (qkvz_stride != width or ba_stride != width
                        or scratch.projected_ba.data_ptr() != scratch.projected_qkvz.data_ptr() + 2 * 16384 // world_size):
                    raise ValueError("combined GDN projection requires adjacent QKV/Z and B/A column views")
                packed = scratch.projected_qkvz.as_strided((self.tokens, width), (width, 1))
                inputs = (("in_proj_qkvzba", packed),)
            else:
                inputs = (("in_proj_qkvz", scratch.projected_qkvz),) if self.fused_qkvz else (
                    ("in_proj_qkv", scratch.projected_qkv), ("in_proj_z", scratch.z))
                inputs = (*inputs, ("in_proj_ba", scratch.projected_ba))
            self.projections = {key: projection(key, source, output) for key, output in inputs}
            self.projections["out_proj"] = projection("out_proj", scratch.gated, scratch.output)
            self.parallel_projections = None
            if (self.fused_qkvz and mode == "decode" and self.tokens <= 16
                    and isinstance(projection_workspace, ScratchArena) and projection_workspace.overlap
                    and all(isinstance(self.projections[key], QuantizedProjection)
                            for key in ("in_proj_qkvz", "in_proj_ba"))):
                from ignition.components.linear import ParallelProjections

                self.parallel_projections = ParallelProjections(
                    self.projections["in_proj_qkvz"], self.projections["in_proj_ba"],
                    session=session, workspace=projection_workspace)
        self.partitions = partitions
        if partitions is not None:
            self.sequences_by_role = tuple(
                PreparedGDN(module, projections, part.scratch.output, part.scratch, state, part.metadata,
                    mode=part.mode, session=session, name=f"{name}.{part.mode}", workspace=workspace,
                    projection_workspace=projection_workspace, records=records if part.mode == "decode" else None,
                    in_place=in_place if part.mode == "decode" else False, prepare_projections=False)
                for part in partitions.parts)
            return
        self.conv_weight = module.conv1d.weight.squeeze(1)
        conv_args = {
            "x": scratch.projected_qkv,
            "conv_weight": self.conv_weight,
            "query_start_loc": metadata.query_start_loc,
            "state_slot_ids": metadata.convolution_slots,
            "state_is_fresh": metadata.fresh,
            "num_accepted_tokens": metadata.accepted_tokens,
            "num_seqs": metadata.num_seqs,
            "num_tokens": metadata.num_tokens,
            "conv_state": state.convolution,
            "out": scratch.convolved_qkv,
        }
        conv_caps = short_conv.Caps(
            device=self.device,
            mode=mode,
            channels=module.conv_dim,
            kernel_size=4,
            max_tokens=self.tokens,
            max_seqs=self.sequences,
            max_state_slots=state.convolution.shape[0],
            max_speculative_tokens=self.speculative_tokens,
        )
        self.conv_plan = short_conv.plan(
            conv_caps,
            invocation=short_conv.invocation_from_tensors(**conv_args),
        )
        fused_convolution = mode == "decode" and records is not None and metadata.state_indices.shape[1] > 1
        self.conv_binding = None
        if fused_convolution:
            self.conv_plan = None
        else:
            self.conv_binding = prepare_sequence_binding(
                self.conv_plan,
                conv_args,
                session=session,
                name=f"{name}.short_conv",
                public_bind=short_conv.bind,
                bind_plan=True,
                workspace=workspace,
                mode=mode,
                set_binding=lambda binding: setattr(self, "conv_binding", binding),
            )
        common = {
            "A_log": module.A_log,
            "dt_bias": module.dt_bias,
            "recurrent_state": state.recurrent,
            "num_seqs": metadata.num_seqs,
            "num_tokens": metadata.num_tokens,
            "output": (scratch.gated if mode == "decode" else scratch.recurrence).view(
                self.tokens, self.value_heads, 128),
            "a": scratch.a,
            "b": scratch.b,
        }
        self.recovered = None
        self.chunked = None
        if mode == "decode" and (records is not None or in_place):
            # Speculative verification records corrections and the round commits the accepted ones;
            # single-token rounds advance the committed state in place.
            self.recovered = PreparedRecoveredRecurrence(
                mixed_qkv=scratch.convolved_qkv, a=scratch.a, b=scratch.b, A_log=module.A_log,
                dt_bias=module.dt_bias, state=state.recurrent, metadata=metadata, output=scratch.recurrence,
                records=records, key_heads=self.key_heads, value_heads=self.value_heads,
                convolution=conv_args if fused_convolution else None, buffers=gdn_buffers)
        elif mode == "decode":
            caps = gdn_decode.Caps(
                device=self.device,
                max_tokens=self.tokens,
                max_seqs=self.sequences,
                max_state_slots=state.recurrent.shape[0],
                key_heads=self.key_heads,
                value_heads=self.value_heads,
                state_index_columns=columns,
                state_dtype=torch.float32,
                gate_activation=module.norm.activation,
                null_state_index=0,
            )
            args = dict(
                **common,
                mixed_qkv=scratch.convolved_qkv,
                z=scratch.z.view(self.tokens, self.value_heads, 128),
                norm_weight=module.norm.weight,
                query_start_loc=metadata.query_start_loc,
                num_accepted_tokens=metadata.accepted_tokens,
                state_indices=metadata.state_indices,
            )
            self.recurrent_plan = gdn_decode.plan(
                caps,
                invocation=gdn_decode.invocation_from_tensors(caps, **args),
            )
            operation = gdn_decode
        elif sm103.is_sm103(self.device) and not checkpoint_export:
            # FlashInfer's chunked delta rule replaces b12x's token-sequential prefill on SM103.
            self.chunked = PreparedGdnPrefill(
                mixed_qkv=scratch.convolved_qkv, a=scratch.a, b=scratch.b, A_log=module.A_log,
                dt_bias=module.dt_bias, state=state.recurrent, metadata=metadata,
                output=scratch.recurrence.view(self.tokens, self.value_heads, 128),
                key_heads=self.key_heads, value_heads=self.value_heads, session=session, name=f"{name}.prefill",
                buffers=PrefillBuffers(scratch.prefill_query, scratch.prefill_key, scratch.prefill_value,
                                       scratch.prefill_decay, scratch.prefill_beta))
        else:
            caps = gdn_prefill.Caps(
                device=self.device,
                max_tokens=self.tokens,
                max_seqs=self.sequences,
                max_state_slots=state.recurrent.shape[0],
                key_heads=self.key_heads,
                value_heads=self.value_heads,
                checkpoint_export=checkpoint_export,
                null_state_index=0,
                recurrence_policy="sequential_k1",
            )
            q, k, v = scratch.convolved_qkv.split((2048 // world_size, 2048 // world_size, 6144 // world_size), dim=-1)
            args = dict(
                **common,
                q=q.view(self.tokens, self.key_heads, 128),
                k=k.view(self.tokens, self.key_heads, 128),
                v=v.view(self.tokens, self.value_heads, 128),
                cu_seqlens=metadata.query_start_loc,
                initial_state_indices=metadata.initial_state_indices,
                final_state_indices=metadata.final_state_indices,
                checkpoint_state_indices=metadata.checkpoint_state_indices,
                checkpoint_offsets=metadata.checkpoint_offsets,
            )
            self.recurrent_plan = gdn_prefill.plan(
                caps,
                invocation=gdn_prefill.invocation_from_tensors(**args),
            )
            operation = gdn_prefill
        def set_recurrent_binding(binding):
            self.recurrent_binding = binding
            self.workspace = binding.scratch

        if self.recovered is None and self.chunked is None:
            self.recurrent_binding = prepare_sequence_binding(
                self.recurrent_plan,
                args,
                session=session,
                name=f"{name}.{mode}",
                public_bind=operation.bind,
                workspace=workspace,
                mode=mode,
                set_binding=set_recurrent_binding,
                run_options={"eps": module.norm.variance_epsilon} if mode == "decode" else None,
            )
        if mode != "decode" or self.recovered is not None:
            # Qwen's gated norm, fused: it rounds normalized values to BF16 before weighting.
            self.output_norm = PreparedGatedRMSNorm(
                scratch.recurrence.view(self.tokens, self.value_heads, 128),
                scratch.z.view(self.tokens, self.value_heads, 128), module.norm.weight,
                None if self.output_encoding is not None else scratch.gated.view(self.tokens, self.value_heads, 128),
                metadata.num_tokens,
                eps=module.norm.variance_epsilon, round_normalized=True, activation=module.norm.activation,
                encoded=self.output_encoding)

    def functional(self, *, session, dynamic_metadata=False):
        """Keep selected kernels and persistent state while tracing intermediate allocations."""
        if getattr(self, "_functional_requested", False):
            return self
        self._functional_requested = True
        self._dynamic_metadata = dynamic_metadata
        self._functional_ready = False
        self.encoding_layout = ()
        if self.output_encoding is not None:
            from b12x.quantization import mxfp8

            self.encoding_layout = tuple((tuple(base.shape), base.dtype) for base in (
                self.output_encoding.values, self.output_encoding.scale_rows.view(torch.uint8),
                mxfp8.dense_scale_storage(self.output_encoding.scale_mma)))
        if hasattr(self, "projections"):
            for projection in self.projections.values():
                projection.functional(session=session)
            if self.parallel_projections is not None:
                self.parallel_projections.functional(session=session)
        if self.partitions is not None:
            self.partitions = self.partitions.functional()
            for sequence in self.sequences_by_role:
                sequence.functional(session=session)
        elif self.recovered is not None:
            self.recovered.functional(dynamic_metadata=dynamic_metadata)
        elif self.chunked is not None:
            self.chunked.functional(session=session, dynamic_metadata=dynamic_metadata)
        if hasattr(self, "output_norm"):
            self.output_norm.functional(session=session, dynamic_live=dynamic_metadata)
        register_finalizer(session, self._finalize_functional, workspace=self._binding_workspace)
        return self

    def _finalize_functional(self):
        if getattr(self, "conv_binding", None) is not None:
            self.convolution_scratch = tuple((tuple(t.shape), tuple(t.stride()), t.dtype)
                                            for t in (self.conv_binding.gathered_state,
                                                      self.conv_binding.request_ids))
        if (self.partitions is None and self.recovered is None and self.chunked is None
                and self.mode == "prefill" and self.recurrent_binding is not None):
            scratch = self.recurrent_binding.scratch
            self.recurrence_scratch = (tuple(scratch.shape), tuple(scratch.stride()), scratch.dtype)
        self.conv_binding = None
        self.recurrent_binding = None
        self.scratch = None
        self.output_encoding = None
        self.workspace = None
        self._binding_workspace = None
        if getattr(self, "_dynamic_metadata", False):
            self.metadata = None
        self._functional_ready = True

    def _functional_sequence(self, projected, z, a, b, *, records=None, metadata=None):
        runtime_metadata = metadata
        metadata = self.metadata if metadata is None else metadata
        if metadata is None:
            raise ValueError("functional GDN requires metadata at invocation")
        state, module = self.state, self.module
        if records is not None and self.recovered is None:
            raise ValueError("GDN correction records require a prepared verification recurrence")
        convolved = torch.empty((self.tokens, module.conv_dim), dtype=torch.bfloat16, device=projected.device)
        if self.conv_plan is not None:
            gathered, request_ids = (torch.empty_strided(shape, stride, dtype=dtype, device=projected.device)
                                     for shape, stride, dtype in self.convolution_scratch)
            short_conv_out(projected, self.conv_weight, metadata.query_start_loc,
                metadata.convolution_slots, metadata.fresh, metadata.accepted_tokens, metadata.fresh,
                metadata.num_seqs, metadata.num_tokens, state.convolution, convolved,
                gathered, request_ids, None, self.conv_plan.handle)
        if self.recovered is not None:
            recurrence = self.recovered(convolved, a, b, projected=projected, records=records,
                                        metadata=runtime_metadata)
        elif self.chunked is not None:
            recurrence = self.chunked(convolved, a, b, metadata=runtime_metadata)
        elif self.mode == "decode":
            gated = torch.empty((self.tokens, self.value_heads, 128), dtype=torch.bfloat16, device=projected.device)
            gdn_decode_out(convolved, a, b, z.view_as(gated), module.A_log, module.dt_bias,
                module.norm.weight, state.recurrent, metadata.query_start_loc, metadata.accepted_tokens,
                metadata.state_indices, metadata.num_seqs, metadata.num_tokens, gated,
                module.norm.variance_epsilon, 128 ** -.5, 0., self.recurrent_plan.handle)
            return gated.view(self.tokens, -1), None
        else:
            recurrence = torch.zeros((self.tokens, self.value_heads, 128), dtype=torch.bfloat16,
                                     device=projected.device)
            q, k, v = convolved.split((self.key_heads * 128, self.key_heads * 128, self.value_heads * 128), dim=-1)
            shape, stride, dtype = self.recurrence_scratch
            workspace = torch.empty_strided(shape, stride, dtype=dtype, device=projected.device)
            gdn_prefill_out(workspace, q.view(self.tokens, self.key_heads, 128),
                k.view(self.tokens, self.key_heads, 128), v.view(self.tokens, self.value_heads, 128),
                a, b, module.A_log, module.dt_bias, state.recurrent, metadata.query_start_loc,
                metadata.initial_state_indices, metadata.final_state_indices,
                metadata.checkpoint_state_indices, metadata.checkpoint_offsets, metadata.num_seqs,
                metadata.num_tokens, recurrence, self.recurrent_plan.handle, 128 ** -.5, 1e-6, None, None)
        recurrence = recurrence.view(self.tokens, self.value_heads, 128)
        encoded = None
        if self.encoding_layout:
            encoded = tuple(torch.empty(shape, dtype=dtype, device=projected.device) if index == 0
                            else torch.full(shape, 127, dtype=dtype, device=projected.device)
                            for index, (shape, dtype) in enumerate(self.encoding_layout))
        gated = self.output_norm(recurrence, z.view_as(recurrence), encoded_bases=encoded,
                                 live=metadata.num_tokens)
        return (recurrence if gated is None else gated).view(self.tokens, -1), encoded

    def _functional(self, source, *, records=None, metadata=None):
        if not self._functional_ready:
            raise RuntimeError("functional GDN requires completed sequence binding finalization")
        if self.fused_qkvzba:
            packed = self.projections["in_proj_qkvzba"](source)
            qkvz, ba = packed.split((16384 // self.world_size, 96 // self.world_size), dim=-1)
        elif self.parallel_projections is not None:
            qkvz, ba = self.parallel_projections(source)
        else:
            qkvz = self.projections["in_proj_qkvz"](source) if self.fused_qkvz else None
            if qkvz is None:
                projected = self.projections["in_proj_qkv"](source)
                z = self.projections["in_proj_z"](source)
            ba = self.projections["in_proj_ba"](source)
        if qkvz is not None:
            projected, z = qkvz.split((10240 // self.world_size, 6144 // self.world_size), dim=-1)
        b, a = ba.chunk(2, dim=-1)
        if self.partitions is None:
            if metadata is None:
                gated, encoded = self._functional_sequence(projected, z, a, b, records=records)
            else:
                gated, encoded = self._functional_sequence(projected, z, a, b, records=records, metadata=metadata)
        else:
            qkvz = torch.cat((projected, z), dim=-1) if qkvz is None else qkvz
            outputs = []
            for part, sequence in zip(self.partitions.parts, self.sequences_by_role, strict=True):
                packed_qkvz, packed_ba = part.gather(qkvz), part.gather(ba)
                part_projected, part_z = packed_qkvz.split((10240 // self.world_size, 6144 // self.world_size), dim=-1)
                part_b, part_a = packed_ba.chunk(2, dim=-1)
                values, _ = sequence._functional_sequence(part_projected, part_z, part_a, part_b,
                    records=records if part.mode == "decode" else None)
                outputs.append(values)
            gated, encoded = self.partitions.combine(outputs), None
        if encoded is None:
            return self.projections["out_proj"](gated)
        return self.projections["out_proj"](gated, encoded_bases=encoded)

    def __call__(self, source: torch.Tensor, *, records=None, metadata=None) -> torch.Tensor:
        if getattr(self, "_functional_requested", False):
            return self._functional(source, records=records, metadata=metadata)
        if records is not None or metadata is not None:
            raise ValueError("runtime GDN operands require functional execution")
        scratch = self.scratch
        if self.fused_qkvzba:
            self.projections["in_proj_qkvzba"](source)
        elif self.parallel_projections is None:
            if self.fused_qkvz:
                self.projections["in_proj_qkvz"](source)
            else:
                self.projections["in_proj_qkv"](source)
                self.projections["in_proj_z"](source)
            self.projections["in_proj_ba"](source)
        else:
            self.parallel_projections(source)
        if self.partitions is None:
            self.run_sequence()
        else:
            for part, sequence in zip(self.partitions.parts, self.sequences_by_role):
                part.gather(scratch.projected_qkvz, part.scratch.projected_qkvz)
                part.gather(scratch.projected_ba, part.scratch.projected_ba)
                sequence.run_sequence()
            self.partitions.combine(tuple(part.scratch.gated for part in self.partitions.parts), scratch.gated)
        return self.projections["out_proj"](scratch.gated)

    def run_sequence(self):
        scratch = self.scratch
        if self.conv_binding is not None:
            short_conv.run(self.conv_binding)
        if self.recovered is not None:
            self.recovered()
            self.output_norm()
        elif self.mode == "decode":
            gdn_decode.run(
                self.recurrent_binding, eps=self.module.norm.variance_epsilon
            )
        else:
            scratch.recurrence.zero_()
            if self.chunked is not None:
                self.chunked()
            else:
                gdn_prefill.run(self.recurrent_binding)
            self.output_norm()


__all__ = [ "GDNScratch", "GDNState", "PreparedGDN"]
