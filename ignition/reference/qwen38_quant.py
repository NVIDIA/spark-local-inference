# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-native quantization references for Qwen3.8 Flash Next.

Stored weight decoding is independent of b12x kernels. Projection references use
FP32 accumulation; their output is not a bitwise oracle for tensor-core reduction
order or PTX approximate reciprocals. NVFP4 and MXFP8 projection references
include activation quantization. They never expand a complete model or PLE table.
"""

from __future__ import annotations

import math
from contextlib import ExitStack
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from safetensors import safe_open

from ignition.preparation import CheckpointManifest
from ignition.preparation.qwen38 import PLE_PREFIX


class CheckpointTensors:
    """Read selected CPU tensor payloads with a cumulative byte budget.

    Safetensors mappings are shared across reads and closed with the context.
    Returned tensors own their storage. This reader is for numerical references;
    production weight transfer belongs to b12x.loader.
    """

    def __init__(self, manifest: CheckpointManifest, *, byte_budget: int = 64 << 20):
        if type(byte_budget) is not int or byte_budget <= 0:
            raise ValueError("byte_budget must be a positive integer")
        self.manifest = manifest
        self.byte_budget = byte_budget
        self.bytes_read = 0
        self._stack = ExitStack()
        self._files = {}
        self._closed = False

    def __enter__(self):
        if self._closed:
            raise RuntimeError("checkpoint reader is closed")
        return self

    def __exit__(self, *args):
        self._closed = True
        self._files.clear()
        return self._stack.__exit__(*args)

    def read(self, name: str, *, rows: tuple[int, int] | None = None) -> torch.Tensor:
        """Read a tensor, or a half-open contiguous interval of its first axis."""
        if self._closed:
            raise RuntimeError("checkpoint reader is closed")
        metadata = self.manifest.tensors[name]
        count = metadata.nbytes
        if rows is not None:
            if (
                not metadata.shape or len(rows) != 2
                or any(type(i) is not int for i in rows)
                or not 0 <= rows[0] <= rows[1] <= metadata.shape[0]
            ):
                raise ValueError(f"invalid row interval for {name}: {rows}")
            row_bytes = math.prod(metadata.shape[1:]) * _dtype(metadata.dtype).itemsize
            count = (rows[1] - rows[0]) * row_bytes
        if count > self.byte_budget - self.bytes_read:
            raise ValueError(
                f"{name}: read of {count} bytes exceeds remaining reference budget "
                f"{self.byte_budget - self.bytes_read}; select fewer rows"
            )
        if metadata.shard not in self._files:
            self._files[metadata.shard] = self._stack.enter_context(
                safe_open(self.manifest.snapshot / metadata.shard, framework="pt", device="cpu")
            )
        source = self._files[metadata.shard].get_slice(name)
        if tuple(source.get_shape()) != metadata.shape or source.get_dtype() != metadata.dtype:
            raise ValueError(f"{name}: source tensor metadata changed after inspection")
        value = (
            self._files[metadata.shard].get_tensor(name) if rows is None
            else source[rows[0]:rows[1]]
        ).clone()
        self.bytes_read += count
        return value


def _dtype(name):
    return {
        "BF16": torch.bfloat16, "F32": torch.float32,
        "F8_E4M3": torch.float8_e4m3fn, "U8": torch.uint8,
        "U16": torch.uint16, "U32": torch.uint32, "I64": torch.int64,
    }[name]


def _finite(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value.float()).all():
        raise ValueError(f"{name} must be finite")


def _positive_scalar(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.numel() != 1 or value.dtype != torch.float32:
        raise ValueError(f"{name} must be one FP32 value")
    _finite(value, name)
    if value.item() <= 0:
        raise ValueError(f"{name} must be positive")
    return value.reshape(())


def decode_csf_scales(
    bases: torch.Tensor, codes: torch.Tensor, page_bases: torch.Tensor,
    offsets: torch.Tensor, exceptions: torch.Tensor, *, rows: int, columns: int,
) -> torch.Tensor:
    """Decode csf_32x2_v1 NVFP4 scales independently into logical E4M3 bytes."""
    import numpy as np

    if rows <= 0 or rows % 32 or columns <= 0:
        raise ValueError("CSF requires positive 32-row extents and positive columns")
    groups = ((rows + 127) // 128) * ((columns + 3) // 4)
    pages = (groups + 31) // 32
    blocks = rows // 32 * ((columns + 1) // 2)
    for tensor, dtype, shape in (
        (bases, torch.uint8, (rows,)), (codes, torch.uint8, (rows * columns // 2,)),
        (page_bases, torch.uint32, (pages,)), (offsets, torch.uint16, (blocks + pages,)),
        (exceptions, torch.uint16, (exceptions.numel(),)),
    ):
        if tensor.device.type != "cpu" or tensor.dtype != dtype or tensor.shape != shape:
            raise ValueError("CSF component shape, dtype, or CPU placement mismatch")
    if bool((bases > 240).any()):
        raise ValueError("CSF row palette exceeds a byte")
    base, packed, records = bases.numpy(), codes.numpy(), exceptions.numpy()
    fields = ((packed[:, None] >> np.array([0, 4], dtype=np.uint8)) & 15).ravel()
    decoded = np.empty((rows, columns), dtype=np.uint8)
    prefixes, origins = offsets.tolist(), page_bases.tolist()
    group = index = cursor = record_cursor = 0
    for row in range(0, rows, 128):
        for column in range(0, columns, 4):
            if group % 32 == 0:
                if origins[group // 32] != 2 * record_cursor or prefixes[index] != 0:
                    raise ValueError("CSF page origin or prefix is not consecutive")
                page_origin = record_cursor
            for pair in range(column, min(column + 4, columns), 2):
                width = min(2, columns - pair)
                for block_row in range(row, min(row + 128, rows), 32):
                    tile = decoded[block_row:block_row + 32, pair:pair + width]
                    tile[:] = base[block_row:block_row + 32, None] + fields[cursor:cursor + 32 * width].reshape(32, width)
                    cursor += 32 * width
                    end = page_origin + prefixes[index + 1]
                    if not record_cursor <= end <= min(record_cursor + 32 * width, records.size):
                        raise ValueError("CSF exception prefix exceeds the block extent")
                    words = records[record_cursor:end]
                    positions = (words & 255).astype(np.int64)
                    if (np.any(positions >= 64) or np.any(positions % 2 >= width)
                            or np.any(positions[1:] <= positions[:-1])):
                        raise ValueError("CSF exception positions must be sorted, unique, and in range")
                    tile[positions // 2, positions % 2] = (words >> 8).astype(np.uint8)
                    record_cursor = end
                    index += 1
            group += 1
            if group % 32 == 0 or group == groups:
                index += 1
    if record_cursor != records.size:
        raise ValueError("CSF exceptions are not covered by the index")
    return torch.from_numpy(decoded)


def decode_csf_slab_scales(
    fixed: torch.Tensor, exceptions: torch.Tensor, *, rows: int, columns: int
) -> torch.Tensor:
    """Decode byte-window4-fixed-stream-u24-exceptions/1 into E4M3 bytes.

    columns counts scale elements, not packed weight bytes. Each 16-row slab
    contains 16 base bytes followed by row-major pairs of four-bit offsets.
    Exception words contain a 24-bit row-major position and an 8-bit value.
    """
    if rows <= 0 or rows % 16 or columns <= 0 or columns % 2:
        raise ValueError("CSF requires positive rows divisible by 16 and even columns")
    if rows * columns > 1 << 24:
        raise ValueError("CSF matrix exceeds its 24-bit position field")
    if fixed.dtype != torch.uint8 or fixed.shape != (rows // 16, 16 + 8 * columns):
        raise ValueError("CSF fixed stream has the wrong shape or dtype")
    if exceptions.dtype != torch.uint32 or exceptions.ndim != 1:
        raise ValueError("CSF exceptions must be a one-dimensional uint32 tensor")
    if fixed.device != exceptions.device:
        raise ValueError("CSF streams must be on the same device")
    base = fixed[:, :16].reshape(rows, 1).to(torch.int16)
    pairs = fixed[:, 16:].reshape(rows, columns // 2)
    offsets = torch.stack((pairs & 15, pairs >> 4), -1).reshape(rows, columns)
    decoded = base + offsets.to(torch.int16)
    if (decoded > 255).any():
        raise ValueError("CSF base plus offset exceeds a byte")
    decoded = decoded.to(torch.uint8).flatten()
    words = exceptions.to(torch.int64)
    positions = words & 0xFFFFFF
    if positions.numel() and (
        positions[-1] >= rows * columns or (positions[1:] <= positions[:-1]).any()
    ):
        raise ValueError("CSF exception positions must be sorted, unique, and in range")
    decoded[positions] = (words >> 24).to(torch.uint8)
    return decoded.reshape(rows, columns)


_FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
               -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def decode_nvfp4(
    packed: torch.Tensor, scales: torch.Tensor, weight_scale_2: torch.Tensor
) -> torch.Tensor:
    """Decode ModelOpt row-major NVFP4 weights into FP32 values."""
    if packed.dtype != torch.uint8 or packed.ndim != 2 or packed.shape[1] % 8:
        raise ValueError("NVFP4 weights must be rank-two uint8 with group16 logical width")
    if scales.dtype != torch.float8_e4m3fn or scales.shape != (packed.shape[0], packed.shape[1] // 8):
        raise ValueError("NVFP4 scales must be row-major E4M3 with one scale per 16 values")
    if not packed.device == scales.device == weight_scale_2.device:
        raise ValueError("NVFP4 weights and scales must be on the same device")
    global_scale = _positive_scalar(weight_scale_2, "weight_scale_2")
    _finite(scales, "NVFP4 block scales")
    if (scales.float() < 0).any():
        raise ValueError("NVFP4 block scales must be nonnegative")
    lut = torch.tensor(_FP4_VALUES, dtype=torch.float32, device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    values = lut[codes].reshape(packed.shape[0], -1, 16)
    return (values * scales.float().unsqueeze(-1) * global_scale).flatten(-2)


def decode_mxfp8(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Decode row-major E4M3 weights with per-32-element E8M0 byte scales."""
    if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2 or weight.shape[1] % 32:
        raise ValueError("MXFP8 weights must be rank-two E4M3 with width divisible by 32")
    if scales.dtype != torch.uint8 or scales.shape != (weight.shape[0], weight.shape[1] // 32):
        raise ValueError("MXFP8 scales must be row-major uint8 with one scale per 32 values")
    if weight.device != scales.device:
        raise ValueError("MXFP8 weights and scales must be on the same device")
    if (scales == 255).any():
        raise ValueError("MXFP8 scale byte 255 encodes NaN")
    _finite(weight, "MXFP8 weights")
    multipliers = torch.ldexp(torch.ones_like(scales, dtype=torch.float32), scales.int() - 127)
    return (weight.float().reshape(weight.shape[0], -1, 32) * multipliers[..., None]).flatten(-2)


def round_e2m1(values: torch.Tensor) -> torch.Tensor:
    """Saturating round-to-nearest-even to FP4 E2M1, returned as FP32."""
    _finite(values, "E2M1 input")
    magnitudes = values.float().abs().clamp(max=6)
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=values.device)
    below = torch.searchsorted(boundaries, magnitudes.contiguous(), right=False)
    at_boundary = magnitudes == boundaries[below.clamp(max=6)]
    codes = below + (at_boundary & (below % 2 == 1)).int()
    lut = torch.tensor(_FP4_VALUES[:8], dtype=torch.float32, device=values.device)
    return lut[codes] * values.sign().float()


def quantize_nvfp4_activation(values: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
    """Return NVFP4 activation QDQ in its original physical scale.

    input_scale is the checkpoint's global dequantization scale. Per-group
    scales use E4M3 round-to-nearest with finite saturation; values use E2M1
    round-to-nearest-even. CPU division does not reproduce PTX rcp.approx.ftz
    at every quantization threshold.
    """
    return _nvfp4_activation_unscaled(values, input_scale) * input_scale.reshape(())


def _nvfp4_activation_unscaled(values, input_scale, *, scale_math="micro"):
    if values.ndim == 0 or values.shape[-1] % 16:
        raise ValueError("NVFP4 activations require a last dimension divisible by 16")
    if values.device != input_scale.device:
        raise ValueError("activation and input_scale must be on the same device")
    global_scale = _positive_scalar(input_scale, "input_scale")
    _finite(values, "NVFP4 activations")
    if scale_math == "flashinfer":
        # FlashInfer's quantizer (SM103): FP32 products with a rounded 1/6, then the
        # reciprocal of the dequantization scale, as its fast-math kernels evaluate them.
        groups = values.float().reshape(*values.shape[:-1], -1, 16)
        inverse_global = global_scale.float().reciprocal()
        maximum = groups.abs().amax(-1, keepdim=True)
        sixth = torch.tensor(1 / 6, dtype=torch.float32, device=values.device)
        block_scale = (inverse_global * (maximum * sixth)).clamp(max=448).to(torch.float8_e4m3fn).float()
        multiplier = torch.where(maximum == 0, torch.zeros_like(block_scale),
                                 (block_scale * inverse_global.reciprocal()).reciprocal())
        return (round_e2m1(groups * multiplier) * block_scale).reshape(values.shape)
    if scale_math != "micro":
        from b12x.moe._shared.kernels.reference import _quantize_vec_to_fp4_dequant

        return torch.stack([
            _quantize_vec_to_fp4_dequant(row.float(), global_scale.reciprocal().item(),
                block_size=16, fp8_e4m3_max=448, scale_math=scale_math)
            for row in values.reshape(-1, values.shape[-1])
        ]).reshape(values.shape)
    groups = values.float().reshape(*values.shape[:-1], -1, 16)
    inverse_global = global_scale.reciprocal()
    raw_scale = (
        groups.abs().amax(-1, keepdim=True).double() * inverse_global.double() / 6
    ).float()
    block_scale = raw_scale.clamp(max=448).to(torch.float8_e4m3fn).float()
    # The rounded reciprocal is kernel metadata; inverting it is not bitwise
    # equivalent to multiplying by the stored scale at E2M1 midpoints.
    effective = block_scale / inverse_global
    reciprocal = torch.where(effective == 0, torch.zeros_like(effective), effective.reciprocal())
    return (round_e2m1(groups * reciprocal) * block_scale).reshape(values.shape)


def quantize_mxfp8_activation(values: torch.Tensor) -> torch.Tensor:
    """Return E4M3 activation QDQ using group32 power-of-two ceiling scales."""
    if values.ndim == 0 or values.shape[-1] % 32:
        raise ValueError("MXFP8 activations require a last dimension divisible by 32")
    _finite(values, "MXFP8 activations")
    groups = values.float().reshape(*values.shape[:-1], -1, 32)
    maximum = groups.abs().amax(-1, keepdim=True)
    raw = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 448)
    exponent = raw.log2().ceil().clamp(-127, 127).int()
    scale = torch.ldexp(torch.ones_like(raw), exponent)
    quantized = (groups / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (quantized * scale).reshape(values.shape)


@dataclass(frozen=True)
class Nvfp4Projection:
    packed: torch.Tensor
    scales: torch.Tensor
    weight_scale_2: torch.Tensor
    input_scale: torch.Tensor | None


def read_nvfp4_projection(reader: CheckpointTensors, prefix: str) -> Nvfp4Projection:
    """Read one bounded expert or vision projection, including CSF when present."""
    packed = reader.read(f"{prefix}.weight")
    scale_name = f"{prefix}.weight_scale"
    if f"{scale_name}.csf32_bases" in reader.manifest.tensors:
        components = [reader.read(f"{scale_name}.csf32_{field}")
                      for field in ("bases", "codes", "page_bases", "offsets", "exceptions")]
        scales = decode_csf_scales(
            *components, rows=packed.shape[0], columns=packed.shape[1] // 8,
        ).view(torch.float8_e4m3fn)
    elif f"{scale_name}.nvfp4_csf_fixed" in reader.manifest.tensors:
        scales = decode_csf_slab_scales(
            reader.read(f"{scale_name}.nvfp4_csf_fixed"),
            reader.read(f"{scale_name}.nvfp4_csf_exceptions"),
            rows=packed.shape[0], columns=packed.shape[1] // 8,
        ).view(torch.float8_e4m3fn)
    else:
        scales = reader.read(scale_name)
    input_scale = (
        reader.read(f"{prefix}.input_scale")
        if f"{prefix}.input_scale" in reader.manifest.tensors else None
    )
    return Nvfp4Projection(packed, scales, reader.read(f"{prefix}.weight_scale_2"), input_scale)


def nvfp4_linear_reference(
    values: torch.Tensor, projection: Nvfp4Projection, *, quant_scale_math="micro"
) -> torch.Tensor:
    """FP32 projection with checkpoint-prescribed W4A4 or W4A16 arithmetic."""
    global_scale = _positive_scalar(projection.weight_scale_2, "weight_scale_2")
    weight = decode_nvfp4(projection.packed, projection.scales, torch.ones_like(global_scale))
    if values.shape[-1] != weight.shape[-1]:
        raise ValueError("activation width does not match the NVFP4 projection")
    if projection.input_scale is None:
        inputs = values.to(torch.bfloat16).float()
        weight = weight.to(torch.bfloat16).float()
        alpha = global_scale
    else:
        inputs = _nvfp4_activation_unscaled(values, projection.input_scale, scale_math=quant_scale_math)
        alpha = global_scale / projection.input_scale.reshape(()).reciprocal()
    return (inputs @ weight.T) * alpha


def mxfp8_linear_reference(
    values: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """FP32 projection after native MXFP8 activation QDQ and weight decoding."""
    decoded = decode_mxfp8(weight, scales)
    if values.shape[-1] != decoded.shape[-1]:
        raise ValueError("activation width does not match the MXFP8 projection")
    return quantize_mxfp8_activation(values) @ decoded.T


@dataclass(frozen=True)
class ExpertTrace:
    gate: torch.Tensor
    up: torch.Tensor
    intermediate: torch.Tensor
    down: torch.Tensor


def nvfp4_expert_reference(
    values: torch.Tensor, gate: Nvfp4Projection, up: Nvfp4Projection, down: Nvfp4Projection,
    *, quant_scale_math="micro",
) -> ExpertTrace:
    """Single-expert projections with BF16 SwiGLU, before router/reduction rounding."""
    if any(projection.input_scale is None for projection in (gate, up, down)):
        raise ValueError("NVFP4 expert reference requires all three activation scales")
    gate_output = nvfp4_linear_reference(values, gate, quant_scale_math=quant_scale_math)
    up_output = nvfp4_linear_reference(values, up, quant_scale_math=quant_scale_math)
    intermediate = (F.silu(gate_output) * up_output).to(torch.bfloat16)
    return ExpertTrace(
        gate_output, up_output, intermediate,
        nvfp4_linear_reference(intermediate, down, quant_scale_math=quant_scale_math)
    )


def read_ple_rows(reader: CheckpointTensors, ids: torch.Tensor) -> torch.Tensor:
    """Read and dequantize selected global PLE row ids, returning BF16 width160."""
    if ids.dtype != torch.int64 or ids.device.type != "cpu":
        raise ValueError("PLE row ids must be a CPU int64 tensor")
    if ids.numel() > 4096:
        raise ValueError("PLE reference lookup admits at most 4096 row ids per call")
    rows_per_plane, num_planes = 2500012, 128
    if ((ids < 0) | (ids >= rows_per_plane * num_planes)).any():
        raise ValueError("PLE row id is outside the checkpoint table")
    prefix = f"{PLE_PREFIX}.ngram_embedding"
    scale = reader.read(f"{prefix}.weight_scale_2")
    unique, inverse = ids.flatten().unique(sorted=True, return_inverse=True)
    values = torch.empty((unique.numel(), 160), dtype=torch.bfloat16)
    for index, row in enumerate(unique.tolist()):
        plane, local = divmod(row, rows_per_plane)
        row_prefix = f"{prefix}.shard_{plane}"
        packed = reader.read(f"{row_prefix}.weight", rows=(local, local + 1))
        scales = reader.read(f"{row_prefix}.weight_scale", rows=(local, local + 1))
        values[index] = decode_nvfp4(packed, scales, scale)[0].to(torch.bfloat16)
    return values[inverse].reshape(*ids.shape, 160)


def ple_hash_reference(
    tokens: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
    *,
    history: torch.Tensor | None = None,
    eos_token_id: int = 248044,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hash two- and three-grams using stored constants and a two-token history.

    EOS ends the history available to subsequent tokens. The EOS token itself
    retains its preceding context. Returns global table rows and the next raw
    two-token history, as in the canonical Transformers cache.
    """
    for name, value, shape in (
        ("multipliers", multipliers, (3,)),
        ("vocab_sizes", vocab_sizes, (16,)),
        ("offsets", offsets, (16,)),
    ):
        if value.device.type != "cpu" or value.dtype != torch.int64 or value.shape != shape:
            raise ValueError(f"{name} must be a CPU int64 tensor with shape {shape}")
    if tokens.device.type != "cpu" or tokens.dtype != torch.int64 or tokens.ndim != 2:
        raise ValueError("tokens must be a rank-two CPU int64 tensor")
    if ((tokens < 0) | (tokens >= 248320)).any():
        raise ValueError("token id is outside the checkpoint vocabulary")
    if history is None:
        history = torch.full((tokens.shape[0], 2), eos_token_id, dtype=torch.int64)
    if history.device.type != "cpu" or history.dtype != torch.int64 or history.shape != (tokens.shape[0], 2):
        raise ValueError("history must be a CPU int64 tensor with shape (batch, 2)")
    if ((history < 0) | (history >= 248320)).any():
        raise ValueError("history token id is outside the checkpoint vocabulary")
    if (vocab_sizes <= 0).any() or (offsets < 0).any() or (multipliers <= 0).any():
        raise ValueError("PLE vocabulary sizes and multipliers must be positive; offsets nonnegative")
    factors = multipliers.tolist()
    if max(factors) * 248319 > (1 << 63) - 1:
        raise ValueError("PLE multiplier product exceeds signed int64")
    sizes, starts = vocab_sizes.tolist(), offsets.tolist()
    output = torch.empty((*tokens.shape, 16), dtype=torch.int64)
    next_history = history.clone()
    for batch, token_row in enumerate(tokens.tolist()):
        previous = history[batch].tolist()
        for position, token in enumerate(token_row):
            context = []
            ended = False
            for prior in reversed(previous):
                ended = ended or prior == eos_token_id
                context.append(eos_token_id if ended else prior)
            mixed = token * factors[0] ^ context[0] * factors[1]
            for head in range(8):
                output[batch, position, head] = mixed % sizes[head] + starts[head]
            mixed ^= context[1] * factors[2]
            for head in range(8, 16):
                output[batch, position, head] = mixed % sizes[head] + starts[head]
            previous = [previous[-1], token]
        next_history[batch] = torch.tensor(previous)
    return output, next_history
