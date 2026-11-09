# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 Engram K4 lookup: Triton kernel, LUT and reference decode.

One Engram row is 256 values coded as one K4 trellis tile (decision D1: natural
order, no rotation; the tile lane order is the packer's 16x16 Tensor-Core
permutation, inverted on the way out).  The release stores, per row:

    engram/layer-LLL.trellis.safetensors   U8      [rows, 128]   32 u32 words
    engram/layer-LLL.scale.safetensors     F8_E8M0 [rows, 8]     one scale per 32 values

The word layout is the owner's packer (``glm53_nvfp4/trellis_nvfp4.py``
``pack_trellis_edges``: adjacent int16 words swapped, so u32 word ``k`` holds
stream bits ``[32k, 32k+32)`` MSB-first).  Decode (DESIGN 1c(3)):

    lane i in 0..31 owns positions 8i..8i+7
    cat        = (w[(i+31) % 32] << 32) | w[i]          # previous word, wrapping
    state(8i+j) = (cat >> (28 - 4j)) & 0xFFFF
    value       = LUT[state] * 2**(e-127)               # e = row scale byte, col//32
    out[col]    = bf16(value)                           # col = permutation[8i+j]

``LUT = state_lut(4, "e4m3", "mcg", 2.0)``: 65,536 E4M3 bytes, one per 16-bit
MCG state (``trellis_mxf.state_lut``).  This module rebuilds it from the
procedural law so the serving image needs no encoder package; the CPU check
proves it equals the owner's ``state_lut`` byte for byte.

The kernel keeps the b12x Engram lookup contract it replaces
(``b12x.sequence.engram._kernels._lookup``): grid ``(tokens, 24)``, one
``[256]`` head row per program, ``out[t, head*256 + col]`` in BF16, and zeros
for every row this rank does not own (so the following TP all-reduce is exact).
Rows stay in host RAM: the table is a mapped-host allocation read zero-copy
over PCIe, exactly as the FP8 table is today.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import triton
import triton.language as tl

# --- Geometry ---------------------------------------------------------------

BITS = 4
HEADS = 24  # 6144 staged columns / 256
ROW_WIDTH = 256  # values per Engram row (8 heads' worth of wkv input per row)
SCALE_BLOCK = 32
ROW_BYTES = ROW_WIDTH * BITS // 8  # 128
ROW_WORDS = ROW_BYTES // 4  # 32 u32 words
SCALE_BYTES = ROW_WIDTH // SCALE_BLOCK  # 8
STATES = 1 << 16  # 16-bit trellis state
COMPANDER_SCALE = 2.0
MCG_MULTIPLIER = 0xCBAC1FED
MCG_MASK = 0x8FFF8FFF
MCG_XOR = 0x3B603B60

# E4M3 byte zero is 2**-127, not floating-point zero; 255 is NaN (the same
# convention the b12x FP8 lookup uses, _kernels.py:96-99).
_UE8M0_MIN = 2.0**-127


# --- Codec tables (rebuilt from the owner's procedural law) -----------------


def procedural_state_values() -> torch.Tensor:
    """The EXL3 procedural MCG state alphabet in FP16 (trellis_nvfp4.py:291)."""
    states = torch.arange(STATES, dtype=torch.int64)
    product = (states * MCG_MULTIPLIER) & 0xFFFFFFFF
    word = (product & MCG_MASK) ^ MCG_XOR
    low = (word & 0xFFFF).to(torch.uint16).view(torch.float16)
    high = ((word >> 16) & 0xFFFF).to(torch.uint16).view(torch.float16)
    return (low + high).to(torch.float16)


def state_lut_bytes(compander_scale: float = COMPANDER_SCALE) -> torch.Tensor:
    """``state_lut(4, alphabet="e4m3", law="mcg", compander_scale)`` as bytes.

    Mirrors ``trellis_mxf.state_lut``: scale the FP16 alphabet by the compander,
    project onto E4M3 with the native ties-to-even conversion, canonicalise zero.
    """
    values = procedural_state_values().float() * compander_scale
    projected = values.to(torch.float8_e4m3fn).float()
    projected[projected == 0] = 0
    encoded = projected.to(torch.float8_e4m3fn)
    if not torch.equal(encoded.float(), projected):
        raise RuntimeError("MCG state alphabet escaped the E4M3 grid")
    return encoded.view(torch.uint8).contiguous()


def tensor_core_permutation(device: torch.device | str = "cpu") -> torch.Tensor:
    """EXL3 16x16 row-major -> Tensor-Core lane order (trellis_nvfp4.py:119)."""
    permutation = [0] * ROW_WIDTH
    for thread in range(32):
        row0 = (thread % 4) * 2
        row1 = row0 + 1
        row2 = row0 + 8
        row3 = row0 + 9
        column0 = thread // 4
        column1 = column0 + 8
        permutation[thread * 8 + 0] = row0 * 16 + column0
        permutation[thread * 8 + 1] = row1 * 16 + column0
        permutation[thread * 8 + 2] = row2 * 16 + column0
        permutation[thread * 8 + 3] = row3 * 16 + column0
        permutation[thread * 8 + 4] = row0 * 16 + column1
        permutation[thread * 8 + 5] = row1 * 16 + column1
        permutation[thread * 8 + 6] = row2 * 16 + column1
        permutation[thread * 8 + 7] = row3 * 16 + column1
    return torch.tensor(permutation, dtype=torch.long, device=device)


@lru_cache(maxsize=None)
def _host_tables() -> tuple[torch.Tensor, torch.Tensor]:
    """(LUT as float32 [65536], lane permutation as int64 [256]) on the host.

    The LUT is E4M3 on disk in the codec; every E4M3 value is exact in FP32, so
    the float32 table is the same number the device would compute after a
    cvt.e4m3 -> f32, and the kernel can index it directly.
    """
    lut = state_lut_bytes().view(torch.float8_e4m3fn).float().contiguous()
    return lut, tensor_core_permutation()


@lru_cache(maxsize=None)
def engram_tables(device: torch.device | str) -> dict[str, torch.Tensor]:
    """Device-resident LUT + column permutation, built once per device."""
    device = torch.device(device)
    lut, permutation = _host_tables()
    return {
        "lut": lut.to(device),
        "perm": permutation.to(torch.int32).to(device),
    }


# --- Kernel -----------------------------------------------------------------


@triton.jit(do_not_specialize=["prepared_tokens"])
def _lookup_k4(
    weight_words,  # int32 [rows, 32]   (the uint8 [rows, 128] table, viewed)
    scale_bytes,  # uint8 [rows, 8]
    hashes,  # int64 [T, 24]
    num_tokens,  # int32 [1]
    lut,  # float32 [65536]
    perm,  # int32 [256]  lane position -> natural column
    out,  # bfloat16 [T, 6144]
    prepared_tokens,
    T: tl.constexpr,
    ROWS: tl.constexpr,
    START: tl.constexpr,
    END: tl.constexpr,
    COMPACT: tl.constexpr,
):
    t, head = tl.program_id(0), tl.program_id(1)
    nt = tl.load(num_tokens)
    live = (t < nt) & (t < prepared_tokens) & (nt >= 0) & (nt <= T)
    row = tl.load(hashes + t.to(tl.int64) * 24 + head, live, -1).to(tl.int64)
    local = live & (row >= START) & (row < END) & (row < ROWS)
    local_row = t.to(tl.int64) * 24 + head if COMPACT else tl.where(local, row - START, 0)

    # One 128-byte row read per program (32 words), not one read per value:
    # the table lives in mapped host memory, so every redundant word load is
    # a PCIe transaction.  cat[lane] = (previous word, wrapping) | word, and
    # positions 8*lane .. 8*lane+7 are its eight 4-bit slices, high bits first.
    word = tl.arange(0, 32)
    cur = tl.load(weight_words + local_row * 32 + word, local, 0)
    prev = tl.load(weight_words + local_row * 32 + (word + 31) % 32, local, 0)
    cat = ((prev.to(tl.int64) & 0xFFFFFFFF) << 32) | (cur.to(tl.int64) & 0xFFFFFFFF)
    state = tl.reshape(
        (cat[:, None] >> (28 - 4 * tl.arange(0, 8))[None, :]) & 0xFFFF, (256,)
    )

    pos = tl.arange(0, 256)
    col = tl.load(perm + pos).to(tl.int64)
    exponent = tl.load(scale_bytes + local_row * 8 + (col // 32), local, 127).to(
        tl.uint32
    )
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    scale = tl.where(exponent == 0, 2.0**-127, scale)
    scale = tl.where(exponent == 255, float("nan"), scale)
    value = tl.load(lut + state, local, 0.0)
    result = (value * scale).to(tl.bfloat16)
    tl.store(
        out + t.to(tl.int64) * 6144 + head * 256 + col, tl.where(local, result, 0)
    )


def lookup_k4(
    weight: torch.Tensor,
    scale: torch.Tensor,
    hashes: torch.Tensor,
    num_tokens: torch.Tensor,
    out: torch.Tensor,
    *,
    shard_start: int,
    shard_end: int,
    rows: int | None = None,
    tables: dict[str, torch.Tensor] | None = None,
    prepared_tokens: int | None = None,
    compact_rows: bool = False,
) -> torch.Tensor:
    """Gather one rank's K4 rows into ``out`` (BF16 [tokens, 6144]).

    ``weight``  uint8 [shard_rows, 128] (or its int32 [shard_rows, 32] view),
    contiguous; ``scale`` uint8/E8M0 [shard_rows, 8]; ``hashes`` int64
    [T, 24] holding GLOBAL row ids with -1 for dead rows; ``out`` bf16
    [T, 6144]; ``rows`` the global table row count (the b12x ``ROWS``
    constant: ids at or past it are not rows at all).  Grid and output
    contract match the b12x lookup this replaces: one program per
    (token, head), zeros for every id outside ``[shard_start, shard_end)``.
    """
    if weight.dtype is torch.uint8:
        if weight.shape[-1] != ROW_BYTES:
            raise ValueError(f"Engram K4 table rows must be {ROW_BYTES} bytes")
        if not weight.is_contiguous():
            raise ValueError("Engram K4 table must be contiguous")
        words = weight.view(torch.int32)
    else:
        words = weight
    if words.shape[-1] != ROW_WORDS:
        raise ValueError(f"Engram K4 table must be [rows, {ROW_WORDS}] int32 words")
    if scale.shape[-1] != SCALE_BYTES:
        raise ValueError(f"Engram scale rows must be {SCALE_BYTES} bytes")
    if hashes.shape[-1] != HEADS:
        raise ValueError(f"Engram hash ids must be [tokens, {HEADS}]")
    if out.shape[-1] != HEADS * ROW_WIDTH:
        raise ValueError(f"Engram staged rows must be [tokens, {HEADS * ROW_WIDTH}]")
    # The grid covers every row of the hash buffer (that is the fixed capture
    # shape), so the staged buffer must be at least that tall.
    if out.shape[0] < hashes.shape[0]:
        raise ValueError(
            f"Engram staged buffer has {out.shape[0]} rows but the hash buffer "
            f"has {hashes.shape[0]}; the kernel writes one staged row per hash row"
        )
    if compact_rows and words.shape[0] < hashes.shape[0] * HEADS:
        raise ValueError("compact K4 cache does not cover hash capacity")
    table_rows = int(rows) if rows is not None else int(words.shape[0])
    if not compact_rows and int(shard_end) - int(shard_start) > int(words.shape[0]):
        raise ValueError(
            f"Engram shard [{shard_start}, {shard_end}) does not fit the "
            f"{int(words.shape[0])}-row allocation"
        )
    if not 0 <= int(shard_start) < table_rows:
        raise ValueError(
            f"Engram shard start {shard_start} outside the {table_rows}-row table"
        )
    tokens = int(hashes.shape[0])
    prepared = tokens if prepared_tokens is None else int(prepared_tokens)
    if tables is None:
        tables = engram_tables(out.device)
    _lookup_k4[(tokens, HEADS)](
        words,
        scale.view(torch.uint8),
        hashes,
        num_tokens,
        tables["lut"],
        tables["perm"],
        out,
        prepared,
        T=tokens,
        ROWS=table_rows,
        START=int(shard_start),
        END=int(shard_end),
        COMPACT=compact_rows,
        num_warps=4,
    )
    return out


# --- Reference decodes (CPU, no Triton) -------------------------------------


def _ue8m0_scale(scale_u8: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
    """[rows, 256] FP32 scales for the given natural column order."""
    exponent = scale_u8.to(torch.int64)[:, col // SCALE_BLOCK]
    bits = (exponent << 23).to(torch.int32)
    scale = bits.view(torch.float32)
    scale = torch.where(exponent == 0, torch.tensor(_UE8M0_MIN), scale)
    return torch.where(exponent == 255, torch.tensor(float("nan")), scale)


def reference_decode_rows(
    trellis_u8: torch.Tensor,
    scale_u8: torch.Tensor,
    *,
    lut: torch.Tensor | None = None,
    perm: torch.Tensor | None = None,
) -> torch.Tensor:
    """DESIGN 1c(3) decode in PyTorch: FP32 [rows, 256], natural column order."""
    if trellis_u8.dtype is not torch.uint8 or trellis_u8.shape[-1] != ROW_BYTES:
        raise ValueError("trellis rows must be uint8 [rows, 128]")
    rows = int(trellis_u8.shape[0])
    if lut is None or perm is None:
        host_lut, host_perm = _host_tables()
        lut = host_lut if lut is None else lut
        perm = host_perm if perm is None else perm
    words = trellis_u8.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    index = torch.arange(ROW_WORDS)
    cat = (words[:, (index + ROW_WORDS - 1) % ROW_WORDS] << 32) | words[:, index]
    shifts = 28 - 4 * torch.arange(8)
    state = (cat[:, :, None] >> shifts[None, None, :]) & 0xFFFF
    values = lut[state.reshape(rows, ROW_WIDTH)]
    natural = torch.empty((rows, ROW_WIDTH), dtype=torch.float32)
    natural[:, perm] = values * _ue8m0_scale(scale_u8, perm)
    return natural


def reference_decode_rows_unpack(
    trellis_u8: torch.Tensor,
    scale_u8: torch.Tensor,
    *,
    lut: torch.Tensor | None = None,
    perm: torch.Tensor | None = None,
) -> torch.Tensor:
    """Independent decode through the packer's own bit order.

    Transcribes ``unpack_trellis_edges`` + ``reconstruct_trellis_states``
    (trellis_nvfp4.py:391-420) instead of DESIGN's u32 window, so the two
    routes check each other: int16 words -> adjacent-pair swap -> 16-bit words
    -> symbol bits -> cyclic 16-bit state.
    """
    if trellis_u8.dtype is not torch.uint8 or trellis_u8.shape[-1] != ROW_BYTES:
        raise ValueError("trellis rows must be uint8 [rows, 128]")
    rows = int(trellis_u8.shape[0])
    if lut is None or perm is None:
        host_lut, host_perm = _host_tables()
        lut = host_lut if lut is None else lut
        perm = host_perm if perm is None else perm
    packed = trellis_u8.contiguous().view(torch.int16).reshape(rows, 1, 16 * BITS)
    words = packed.to(torch.int64) & 0xFFFF
    words = words.reshape(rows, 1, -1, 2).flip(-1).reshape(words.shape)
    words = words.reshape(rows, 1, 16, BITS)
    stream = (
        (words[..., None] >> torch.arange(15, -1, -1)) & 1
    ).reshape(rows, 1, 16, BITS * 16)
    symbols = stream.reshape(rows, 1, 16, 16, BITS)
    edges = (
        (symbols << torch.arange(BITS - 1, -1, -1)).sum(-1).reshape(rows, 1, ROW_WIDTH)
    )
    values = edges.to(torch.int64) & ((1 << BITS) - 1)
    states = torch.zeros_like(values)
    for lag in range(-(-16 // BITS)):
        states |= torch.roll(values, shifts=lag, dims=-1) << (lag * BITS)
    states = (states & 0xFFFF).reshape(rows, ROW_WIDTH)
    quantized = lut[states]
    natural = torch.empty((rows, ROW_WIDTH), dtype=torch.float32)
    natural[:, perm] = quantized * _ue8m0_scale(scale_u8, perm)
    return natural


__all__ = [
    "BITS",
    "HEADS",
    "ROW_BYTES",
    "ROW_WIDTH",
    "ROW_WORDS",
    "SCALE_BLOCK",
    "SCALE_BYTES",
    "STATES",
    "engram_tables",
    "lookup_k4",
    "procedural_state_values",
    "reference_decode_rows",
    "reference_decode_rows_unpack",
    "state_lut_bytes",
    "tensor_core_permutation",
]
