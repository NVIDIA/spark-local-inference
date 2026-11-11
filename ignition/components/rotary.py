# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Rotary embeddings with explicit output storage and prepared CUDA kernels."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _angles(Positions, Frequencies, Angles):
    row = tl.program_id(0)
    pair = tl.arange(0, 32)
    position = tl.maximum(tl.load(Positions + row), 0).to(tl.float32)
    angle = position * tl.load(Frequencies + pair)
    tl.store(Angles + row * 64 + 2 * pair, tl.cos(angle))
    tl.store(Angles + row * 64 + 2 * pair + 1, tl.sin(angle))


@triton.jit
def _rotate(Values, Positions, Frequencies, ROW_STRIDE, HEAD_STRIDE, OFFSET, SIGN: tl.constexpr,
            Angles=None, SHARED_ANGLES: tl.constexpr=False):
    """Rotate one row's head in place: its last 64 values hold 32 interleaved pairs."""
    row, head = tl.program_id(0), tl.program_id(1)
    pair = tl.arange(0, 32)
    if SHARED_ANGLES:
        cos = tl.load(Angles + row * 64 + 2 * pair)
        sin = tl.load(Angles + row * 64 + 2 * pair + 1) * SIGN
    else:
        angle = tl.load(Positions + row).to(tl.float32) * tl.load(Frequencies + pair)
        cos, sin = tl.cos(angle), tl.sin(angle) * SIGN
    base = Values + row * ROW_STRIDE + head * HEAD_STRIDE + OFFSET + 2 * pair
    a = tl.load(base).to(tl.float32)
    b = tl.load(base + 1).to(tl.float32)
    tl.store(base, (a * cos - b * sin).to(Values.dtype.element_ty))
    tl.store(base + 1, (a * sin + b * cos).to(Values.dtype.element_ty))


class PreparedRotaryAngles:
    """Compute FP32 sine/cosine pairs once for consumers sharing token positions and frequencies."""

    def __init__(self, positions, frequencies, *, buffers):
        self.positions, self.frequencies = positions, frequencies
        self.values = buffers.zeros((positions.numel(), 32, 2), torch.float32)

    def __call__(self):
        torch.library.wrap_triton(_angles)[(self.positions.numel(),)](
            self.positions, self.frequencies, self.values)

    def apply(self, tensor, *, inverse=False):
        if (tensor.ndim not in (2, 3) or tensor.shape[0] != self.values.shape[0]
                or tensor.stride(-1) != 1 or tensor.shape[-1] < 64):
            raise ValueError("shared rotary angles require matching rows and unit-column-stride heads")
        heads = tensor.shape[1] if tensor.ndim == 3 else 1
        torch.library.wrap_triton(_rotate)[(tensor.shape[0], heads)](
            tensor, self.positions, self.frequencies, tensor.stride(0),
            tensor.stride(1) if tensor.ndim == 3 else 0, tensor.shape[-1] - 64,
            -1 if inverse else 1, self.values, True)


@torch.library.custom_op("flare::cached_rotary", mutates_args=("q_out", "k_out"))
def apply_cached_rotary(q: torch.Tensor, k: torch.Tensor, cache: torch.Tensor,
                       positions: torch.Tensor, q_out: torch.Tensor, k_out: torch.Tensor,
                       interleave: bool) -> None:
    from flashinfer.rope import get_rope_module

    get_rope_module().apply_rope_pos_ids_cos_sin_cache(q, k, q_out, k_out, cache, positions, interleave)


@apply_cached_rotary.register_fake
def _fake_cached_rotary(q, k, cache, positions, q_out, k_out, interleave):
    pass


def prepare_cached_rotary(session):
    from flashinfer.rope import get_rope_module
    from ignition.preparation.tuning import register_finalizer

    register_finalizer(session, get_rope_module)


def interpolated_frequencies(theta, pairs, factor, original_max_position_embeddings, beta_fast, beta_slow,
                             device):
    """Inverse frequencies `1 / theta ** (i / pairs)` interpolated to extend context by `factor`.

    Pairs that complete fewer rotations than `beta_slow` over the original context
    compress by `factor`; pairs beyond `beta_fast` rotations keep their exact
    frequencies; the pairs between ramp linearly, as the YaRN and DeepSeek
    rope-scaling formulations define.
    """
    dims = torch.arange(pairs, device=device, dtype=torch.float32)
    frequencies = 1 / theta ** (dims / pairs)

    def boundary(rotations, original=original_max_position_embeddings, theta=theta):
        return (2 * pairs) * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(theta))

    low = max(math.floor(boundary(beta_fast)), 0)
    high = min(math.ceil(boundary(beta_slow)), pairs - 1)
    smooth = 1 - ((dims - low) / max(high - low, 1e-3)).clamp(0, 1)
    return frequencies / factor * (1 - smooth) + frequencies * smooth


class CompressedRotary:
    def __init__(self, config, device):
        text = config.text
        scaling = text["rope_scaling"]
        self.frequencies = {}
        for compressed in (False, True):
            base = text["compress_rope_theta"] if compressed else text["rope_theta"]
            frequencies = 1 / base ** (torch.arange(32, device=device, dtype=torch.float32) / 32)
            if compressed:
                frequencies = interpolated_frequencies(base, 32, scaling["factor"],
                    scaling["original_max_position_embeddings"], scaling["beta_fast"], scaling["beta_slow"],
                    device)
            self.frequencies[compressed] = frequencies

    def apply(self, tensor, positions, *, compressed, inverse=False):
        """Rotate the last 64 values of each row (and head) in place, in FP32."""
        if tensor.ndim not in (2, 3) or tensor.stride(-1) != 1 or tensor.shape[-1] < 64:
            raise ValueError("rotary expects contiguous (rows, [heads,] width >= 64) values")
        heads = tensor.shape[1] if tensor.ndim == 3 else 1
        torch.library.wrap_triton(_rotate)[(tensor.shape[0], heads)](
            tensor, positions, self.frequencies[compressed], tensor.stride(0),
            tensor.stride(1) if tensor.ndim == 3 else 0, tensor.shape[-1] - 64, -1 if inverse else 1,
            None, False,
        )
