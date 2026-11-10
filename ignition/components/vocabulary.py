# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Vocabulary-sharded projection followed by native PCIe logit gathering."""

import torch

from ignition.preparation.persistent import persistent_empty

from .collectives import PreparedAllGather
from .linear import QuantizedProjection, UnquantizedProjection


class PreparedVocabularyProjection:
    """Project replicated hidden rows with a contiguous vocabulary weight shard.

    Local and full-vocabulary outputs are caller-owned. Every rank participates
    in the gather; the recipe chooses which rank samples from the full logits.
    A quantized shard, such as an NVFP4 draft head, runs as a b12x A16 projection.
    Unquantized shards honor the output buffer's arithmetic contract; BF16
    checkpoint weights can accumulate and produce logits in FP32.
    Optional row indices expand gathered physical rows into caller-owned sample
    slots after communication, avoiding repeated projection and transfer. With
    ``gather=False``, consumers receive the local projection directly and own
    any distributed normalization or sampling. Indexed local projections require
    a disjoint caller-owned local output with one row per logical sample.
    """

    def __init__(self, source, weight, local_output, output, *, runtime, session,
                 coordinator, name, activation_mode="a16", buffers=None, row_indices=None, workspace=None, gather=True):
        rows = source.shape[0]
        if row_indices is None and output.shape[0] != rows:
            raise ValueError("vocabulary output rows must match projected rows without row indices")
        if row_indices is not None and (row_indices.dtype != torch.int64
                or row_indices.device != source.device or row_indices.shape != (output.shape[0],)
                or output.shape[0] < rows):
            raise ValueError("vocabulary row indices require one device int64 index per expanded output row")
        self.row_indices = row_indices
        if isinstance(weight, torch.Tensor):
            self.projection = UnquantizedProjection(source, weight, local_output,
                session=session, name=f"{name}.projection")
        else:
            self.projection = QuantizedProjection(source, weight, local_output,
                session=session, name=f"{name}.projection", activation_mode=activation_mode, workspace=workspace)
        self.local_only = not gather
        if self.local_only:
            self.output = local_output if row_indices is None else output
            if row_indices is not None:
                if (output.shape != (row_indices.numel(), local_output.shape[1])
                        or output.dtype != local_output.dtype or output.device != local_output.device
                        or not output.is_contiguous() or not local_output.is_contiguous()):
                    raise ValueError("local vocabulary row expansion requires contiguous matching local-column outputs")
                if (output.data_ptr() < local_output.data_ptr() + local_output.nbytes
                        and local_output.data_ptr() < output.data_ptr() + output.nbytes):
                    raise ValueError("local vocabulary row expansion requires disjoint physical and logical storage")
            return
        world = 1 if runtime is None else runtime.world_size
        columns = local_output.shape[1] * world
        if columns < output.shape[1]:
            raise ValueError("vocabulary shards must cover every output token")
        self.gathered = output[:rows] if columns == output.shape[1] else (
            torch.empty((rows, columns), dtype=output.dtype, device=output.device) if buffers is None
            else buffers.zeros((rows, columns), output.dtype))
        self.gather = PreparedAllGather(local_output, self.gathered, runtime=runtime,
            session=session, coordinator=coordinator, name=f"{name}.gather")
        self.output = output
        self.padded = columns != output.shape[1]

    def functional(self, *, session):
        """Return local or gathered logits without retaining projection destinations."""
        if self.output is not None:
            self.output_layout = (tuple(self.output.shape), self.output.stride(), self.output.dtype)
            self.projection.functional(session=session)
            self.output = None
            if not self.local_only:
                self.gather.functional(session=session)
                self.gathered = None
        return self

    def __call__(self, source):
        if self.output is None:
            projected = self.projection(source)
            if self.local_only:
                return projected if self.row_indices is None else projected[self.row_indices]
            gathered = self.gather(projected)
            if self.row_indices is None and not self.padded:
                return gathered
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
            selected = gathered[:, :shape[1]] if self.row_indices is None else gathered[self.row_indices, :shape[1]]
            return output.copy_(selected)
        if self.local_only:
            projected = self.projection(source)
            if self.row_indices is not None:
                self.output.copy_(projected[self.row_indices])
            return self.output
        gathered = self.gather(self.projection(source))
        if self.row_indices is not None:
            # Indexed reads materialize before the overlapping destination write.
            self.output.copy_(gathered[self.row_indices, :self.output.shape[1]])
        elif self.padded:
            self.output.copy_(gathered[:, :self.output.shape[1]])
        return self.output


_E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


@torch.no_grad()
def nvfp4_draft_head(weight: torch.Tensor, *, rows_per_chunk: int = 8192):
    """Quantize a BF16 [vocab, hidden] head or vocabulary shard to NVFP4 for draft proposals.

    E4M3 scales cover 16 values and one amax-derived FP32 gain covers the shard. A
    proposal distribution only has to be the q that generated it, so verification
    keeps the BF16 head and stays exact.
    """
    from b12x._lib.intrinsics.reference import swizzle_block_scale
    from b12x.gemm.blockscaled import pack_weight

    rows, columns = weight.shape
    # Swizzled scales pad rows to 128; a TP4 vocabulary shard holds 38,720.
    if weight.dtype != torch.bfloat16 or columns % 128 or rows % 64 or not weight.is_contiguous():
        raise ValueError("NVFP4 draft head requires a contiguous BF16 head of 64-aligned rows and 128-aligned columns")
    global_scale = (6.0 * 448.0) / weight.abs().amax().float().clamp_min(1e-8)
    values = persistent_empty((rows, columns // 2), dtype=torch.uint8, device=weight.device)
    scales = torch.empty((rows, columns // 16), dtype=torch.float8_e4m3fn, device=weight.device)
    bounds = torch.tensor(_E2M1_BOUNDS, device=weight.device)
    for start in range(0, rows, rows_per_chunk):
        block = weight[start:start + rows_per_chunk].float().view(-1, columns // 16, 16)
        scale = (block.abs().amax(-1) / 6.0 * global_scale).clamp(max=448.0).to(torch.float8_e4m3fn)
        effective = scale.float() / global_scale
        quantized = block / torch.where(effective > 0, effective, torch.ones_like(effective)).unsqueeze(-1)
        codes = (torch.bucketize(quantized.abs(), bounds) | ((quantized < 0).long() << 3)).to(torch.uint8)
        codes = codes.view(-1, columns)
        values[start:start + rows_per_chunk] = codes[:, 0::2] | (codes[:, 1::2] << 4)
        scales[start:start + rows_per_chunk] = scale.view(-1, columns // 16)
    gain = persistent_empty((1,), dtype=torch.float32, device=weight.device)
    gain.copy_((1.0 / global_scale).reshape(1).float())
    return pack_weight(values, swizzle_block_scale(scales, allocate=persistent_empty), recipe="nvfp4",
                       global_scale=gain, global_scale_kind="multiplier", allocate=persistent_empty)


@torch.no_grad()
def mxfp8_draft_head(weight: torch.Tensor, *, rows_per_chunk: int = 1024):
    """Prepare an MXFP8 proposal head while retaining the BF16 target weights.

    Each 32-value input block uses E4M3 values and a power-of-two scale rounded
    upward from its maximum magnitude. Chunking bounds conversion storage.
    """
    from b12x.gemm._shared.wo_mxfp8 import quantize_mxfp8_rows_torch
    from b12x.gemm.blockscaled import pack_weight

    if (weight.ndim != 2 or weight.dtype != torch.bfloat16 or not weight.is_contiguous()
            or weight.shape[0] % 64 or weight.shape[1] % 128):
        raise ValueError("MXFP8 draft head requires contiguous BF16 storage with 64-aligned rows and 128-aligned columns")
    if rows_per_chunk < 1:
        raise ValueError("MXFP8 draft head conversion requires a positive chunk size")
    rows, columns = weight.shape
    values = persistent_empty(weight.shape, dtype=torch.float8_e4m3fn, device=weight.device)
    scales = persistent_empty((rows, columns // 32), dtype=torch.uint8, device=weight.device)
    for start in range(0, rows, rows_per_chunk):
        block = quantize_mxfp8_rows_torch(weight[start:start + rows_per_chunk])
        values[start:start + rows_per_chunk].copy_(block.values)
        scales[start:start + rows_per_chunk].copy_(block.scale_rows.view(torch.uint8).reshape(-1, columns // 32))
    return pack_weight(values, scales, recipe="mxfp8", allocate=persistent_empty)
