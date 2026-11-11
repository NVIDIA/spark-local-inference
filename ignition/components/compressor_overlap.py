# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Capture gated compression and cache publication beside query preparation."""

from types import SimpleNamespace

import torch
from b12x.attention import compressed_sparse_mla
from b12x.preparation.types import plan_from_handle

from ignition.components.fp8_indexer import _write_keys
from ignition.components.gated_compression import pool_gated
from ignition.components.rotary import _rotate
from ignition.preparation.tuning import register_finalizer


@torch.library.custom_op("flare::start_compressor_chain", mutates_args=("borrowed", "outputs", "caches"))
def _start(source: torch.Tensor, weights: list[torch.Tensor], history: list[torch.Tensor],
           metadata: list[torch.Tensor], positions: torch.Tensor, frequencies: torch.Tensor, angles: torch.Tensor | None,
           slots: torch.Tensor, borrowed: list[torch.Tensor], outputs: list[torch.Tensor], caches: list[torch.Tensor],
           plans: list[int], ratio: int, eps: float, stream: int) -> None:
    inputs = [source, *metadata, positions, slots]
    if angles is not None:
        inputs.append(angles)
    for destination, value in zip(borrowed, inputs, strict=True):
        destination.copy_(value)
    source, metadata, positions, slots = borrowed[0], borrowed[1:6], borrowed[6], borrowed[7]
    angles = None if angles is None else borrowed[8]
    auxiliary = torch.cuda.ExternalStream(stream, device=source.device)
    auxiliary.wait_stream(torch.cuda.current_stream(source.device))
    batch = SimpleNamespace(**dict(zip(
        ("requests", "positions", "starts", "query_start_loc", "state_rows"), metadata, strict=True)))
    with torch.cuda.stream(auxiliary):
        projected = outputs[0]
        torch.ops.b12x.bf16_gemv_small_n_out(source, weights[0], projected, plans[0], None)
        offset = 0
        for index in range(len(caches)):
            pooled, normalized = outputs[1 + 2 * index:3 + 2 * index]
            width = normalized.shape[1] * (2 if ratio == 4 else 1)
            values, scores = projected[:, offset:offset + width], projected[:, offset + width:offset + 2 * width]
            offset += 2 * width
            ape, norm = weights[1 + 2 * index:3 + 2 * index]
            state_values, state_scores = history[2 * index:2 * index + 2]
            pool_gated(values, scores, ape, state_values, state_scores, batch, pooled, ratio=ratio)
            torch.ops.b12x.hyperconnection_grouped_rmsnorm(
                pooled, norm, normalized, eps, plans[1 + index], False)
            torch.library.wrap_triton(_rotate)[(normalized.shape[0], 1)](
                normalized, positions, frequencies, normalized.stride(0), 0, normalized.shape[1] - 64, 1,
                angles, angles is not None)
            if index == 0:
                compressed_sparse_mla.write_cache(normalized, caches[0], slots,
                    plan=plan_from_handle(plans[-1]))
            else:
                torch.library.wrap_triton(_write_keys)[(normalized.shape[0],)](
                    normalized, caches[1].view(torch.float8_e4m3fn), caches[1].view(torch.float32),
                    slots, caches[1].stride(0), num_warps=4)
        if len(caches) == 2:
            torch.ops.b12x.bf16_gemv_small_n_out(source, weights[-1], outputs[-1], plans[-2], None)


@_start.register_fake
def _start_fake(source, weights, history, metadata, positions, frequencies, angles, slots, borrowed, outputs, caches,
                plans, ratio, eps, stream):
    pass


@torch.library.custom_op("flare::join_compressor_chain", mutates_args=("outputs", "caches"))
def _join(borrowed: list[torch.Tensor], dependencies: list[torch.Tensor], operands: list[torch.Tensor], outputs: list[torch.Tensor],
          caches: list[torch.Tensor], stream: int) -> None:
    source = borrowed[0]
    torch.cuda.current_stream(source.device).wait_stream(torch.cuda.ExternalStream(stream, device=source.device))


@_join.register_fake
def _join_fake(borrowed, dependencies, operands, outputs, caches, stream):
    pass


class ConcurrentGatedCompression:
    """Overlap a read-only-history compressor chain with independent projections.

    The join consumes all main-branch outputs and retains compressor outputs,
    input, history, and cache ownership until both streams have finished.
    Accepted-prefix history commits remain on the main stream after verification.
    """

    def __init__(self, projection, compressors, writer, *, indexer, index_weight, metadata,
                 positions, frequencies, parallel_plans, session, workspace, angles=None):
        if not compressors or any(not c.deferred or not c.projected for c in compressors):
            raise ValueError("compressor overlap requires projected, deferred-commit compressors")
        self.ratio = compressors[0].spec.ratio
        self.eps = compressors[0].normalize.eps
        self.weights = [projection.weight]
        self.outputs = [projection.output]
        self.history, self.side_plans = [], [projection.plan]
        self.plans = [projection.plan.handle]
        for compressor in compressors:
            self.weights.extend((compressor.ape, compressor.normalize.weight))
            self.outputs.extend((compressor.pooled, compressor.output))
            self.history.extend((compressor.state_values, compressor.state_scores))
            self.plans.append(compressor.normalize.plan.handle)
            self.side_plans.append(compressor.normalize.plan)
        self.caches = [writer.cache]
        if indexer is not None:
            self.caches.append(indexer.cache)
            self.weights.append(index_weight.weight)
            self.outputs.append(index_weight.output)
            self.plans.append(index_weight.plan.handle)
            self.side_plans.append(index_weight.plan)
        self.plans.append(writer.plan.handle)
        self.side_plans.append(writer.plan)
        self.metadata = [metadata.requests, metadata.positions, metadata.starts,
                         metadata.query_start_loc, metadata.state_rows]
        self.positions, self.frequencies, self.slots = positions, frequencies, writer.slots
        self.angles = angles
        for side in self.side_plans:
            for main in parallel_plans:
                workspace.register_parallel(main, side)
        self.stream = workspace.auxiliary_stream
        self.stream_handle = self.stream.cuda_stream
        register_finalizer(session, self.validate_scratch, workspace=workspace)

    def validate_scratch(self):
        if any(plan.memory_requirements().scratch_nbytes for plan in self.side_plans):
            raise ValueError("compressor overlap requires scratch-free selected side-branch plans")

    def functional(self, *, session):
        self.output_layouts = tuple((tuple(t.shape), tuple(t.stride()), t.dtype) for t in self.outputs)
        register_finalizer(session, self._release_examples)
        return self

    def _release_examples(self):
        self.outputs = self.metadata = self.positions = self.slots = self.angles = None

    def start(self, source, metadata=None, positions=None, slots=None, *, angles=None):
        outputs = [torch.empty_strided(shape, stride, dtype=dtype, device=source.device).zero_()
                   for shape, stride, dtype in self.output_layouts]
        m = metadata
        operands = [m.requests, m.positions, m.starts, m.query_start_loc, m.state_rows]
        inputs = [source, *operands, positions, slots]
        if angles is not None:
            inputs.append(angles)
        borrowed = [torch.empty_like(value) for value in inputs]
        _start(source, self.weights, self.history, operands, positions, self.frequencies, angles,
               slots, borrowed, outputs, self.caches, self.plans, self.ratio, self.eps, self.stream_handle)
        return borrowed, outputs

    def finish(self, pending, dependencies):
        borrowed, outputs = pending
        _join(borrowed, dependencies, [*self.weights, *self.history, self.frequencies],
              outputs, self.caches, self.stream_handle)
        return outputs
