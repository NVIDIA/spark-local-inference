"""Headwise RMS normalization and partial NeoX rotary embeddings into fixed outputs."""

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_rope(source, weight, cosine, sine, output,
               ROWS: tl.constexpr, HEADS: tl.constexpr, WIDTH: tl.constexpr,
               TOKEN_STRIDE: tl.constexpr, HEAD_STRIDE: tl.constexpr,
               ROTARY: tl.constexpr, COS_STRIDE: tl.constexpr, SIN_STRIDE: tl.constexpr,
               EPS: tl.constexpr, BLOCK_ROWS: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    columns = tl.arange(0, WIDTH)
    token, head = rows // HEADS, rows % HEADS
    values = tl.load(source + token[:, None] * TOKEN_STRIDE + head[:, None] * HEAD_STRIDE
                     + columns[None, :], rows[:, None] < ROWS, 0).to(tl.float32)
    gains = tl.load(weight + columns).to(tl.float32) + 1.0
    inverse = tl.rsqrt(tl.sum(values * values, 1) / WIDTH + EPS)
    normalized = (values * inverse[:, None] * gains[None, :]).to(tl.bfloat16).to(tl.float32)
    partner_columns = tl.where(columns < ROTARY // 2, columns + ROTARY // 2,
                               columns - ROTARY // 2)
    partner_columns = tl.where(columns < ROTARY, partner_columns, columns)
    partners = tl.gather(normalized, tl.broadcast_to(partner_columns[None, :],
                                                   (BLOCK_ROWS, WIDTH)), 1)
    partners = tl.where(columns[None, :] < ROTARY // 2, -partners, partners)
    valid = (rows[:, None] < ROWS) & (columns[None, :] < ROTARY)
    cos = tl.load(cosine + token[:, None] * COS_STRIDE + columns[None, :], valid, 0).to(tl.float32)
    sin = tl.load(sine + token[:, None] * SIN_STRIDE + columns[None, :], valid, 0).to(tl.float32)
    left = (normalized * cos).to(tl.bfloat16).to(tl.float32)
    right = (partners * sin).to(tl.bfloat16).to(tl.float32)
    rotated = (left + right).to(tl.bfloat16).to(tl.float32)
    result = tl.where(columns[None, :] < ROTARY, rotated, normalized)
    tl.store(output + rows[:, None] * WIDTH + columns[None, :], result, rows[:, None] < ROWS)


def head_norm_rope(source, weight, cosine, sine, output, *, eps):
    """Write BF16-rounded head normalization and partial rotation without temporary storage."""
    torch.library.wrap_triton(_norm_rope)[(triton.cdiv(source.shape[0] * source.shape[1], 4),)](
        source, weight, cosine, sine, output, source.shape[0] * source.shape[1],
        source.shape[1], source.shape[2], source.stride(0), source.stride(1),
        cosine.shape[-1], cosine.stride(0), sine.stride(0), eps, 4,
        num_warps=4, enable_fp_fusion=False)
