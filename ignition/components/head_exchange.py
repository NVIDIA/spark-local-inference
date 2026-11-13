# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Combine context-parallel attention across hosts by log-sum-exp, over a RoCE all-gather.

Under decode context parallelism every rank attends all heads over its share of the
context and keeps its own head shard of the combined result. Ranks on one host
exchange shards through b12x's PCIe reduce-scatter. Ranks on separate hosts pack, per
row, the partial outputs and natural-log LSEs of every other rank's head shard, in
rank order without their own, and all-gather the packs. Each rank then merges its own
shard from its local partial and the block each peer packed for it.

A pack row holds (W - 1) * H/W * D BF16 partial values, then (W - 1) * H/W FP32 LSEs
stored as BF16 pairs.
"""

import torch
import triton
import triton.language as tl


def pack_width(world, heads, head_dim):
    """BF16 elements in one row's pack of `heads` total heads."""
    return (world - 1) * (heads // world) * (head_dim + 2)


@triton.jit
def _pack(Partial, Lse, Send, SendLse, Rank, send_stride, send_lse_stride, WORLD: tl.constexpr,
          LOCAL: tl.constexpr, DIM: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    head = tl.program_id(2)
    rank = tl.load(Rank).to(tl.int32)
    # Blocks follow rank order and skip this rank's own shard.
    target = block + (block >= rank).to(tl.int32)
    source = row * WORLD * LOCAL + target * LOCAL + head
    columns = tl.arange(0, DIM)
    tl.store(Send + row * send_stride + (block * LOCAL + head) * DIM + columns,
             tl.load(Partial + source * DIM + columns))
    tl.store(SendLse + row * send_lse_stride + block * LOCAL + head, tl.load(Lse + source))


@triton.jit
def _merge(Received, ReceivedLse, Partial, Lse, Output, Rank, received_stride, received_lse_stride, pack_stride,
           pack_lse_stride, WORLD: tl.constexpr, LOCAL: tl.constexpr, DIM: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    rank = tl.load(Rank).to(tl.int32)
    columns = tl.arange(0, DIM)
    own = row * WORLD * LOCAL + rank * LOCAL + head
    own_lse = tl.load(Lse + own)
    # Each contributor's LSE for this head: this rank's own, else the block that peer packed for it.
    largest = own_lse
    for source in tl.static_range(WORLD):
        block = rank - (rank > source).to(tl.int32)
        lse = tl.load(ReceivedLse + row * received_lse_stride + source * pack_lse_stride + block * LOCAL + head,
                      mask=source != rank, other=-float("inf"))
        largest = tl.maximum(largest, lse)
    # A head no rank attended anything for combines to zero.
    base = tl.where(largest > -float("inf"), largest, 0.0)
    weight = tl.exp(own_lse - base)
    total = weight
    combined = tl.where(weight > 0, weight * tl.load(Partial + own * DIM + columns).to(tl.float32), 0.0)
    for source in tl.static_range(WORLD):
        block = rank - (rank > source).to(tl.int32)
        peer = source != rank
        lse = tl.load(ReceivedLse + row * received_lse_stride + source * pack_lse_stride + block * LOCAL + head,
                      mask=peer, other=-float("inf"))
        weight = tl.exp(lse - base)
        values = tl.load(Received + row * received_stride + source * pack_stride + (block * LOCAL + head) * DIM
                         + columns, mask=(columns < DIM) & peer, other=0.0)
        total += weight
        combined += tl.where(weight > 0, weight * values.to(tl.float32), 0.0)
    result = tl.where(total > 0, combined / total, 0.0)
    tl.store(Output + (row * LOCAL + head) * DIM + columns, result.to(Output.dtype.element_ty))


def _launch(kernel, grid, args, **meta):
    from torch.fx.experimental.proxy_tensor import get_proxy_mode

    if torch.compiler.is_compiling() or get_proxy_mode() is not None:
        torch.library.wrap_triton(kernel)[grid](*args, **meta)
    else:
        kernel[grid](*args, **meta)


def _lse_view(packs, world, heads, head_dim):
    """FP32 view of the LSE tails of [rows, packs * width] BF16 packs; row and pack strides in FP32 units."""
    values = (world - 1) * (heads // world) * head_dim
    return packs.view(torch.float32)[:, values // 2:]


def pack(partial, lse, send, rank, world):
    """Write every other rank's head shard of `partial` [rows, H, D] and `lse` [rows, H] into `send` [rows, width].

    `rank` is this rank as a one-element device tensor, so compiled forwards serve every rank.
    """
    rows, heads, dim = partial.shape
    send_lse = _lse_view(send, world, heads, dim)
    _launch(_pack, (rows, world - 1, heads // world), (partial, lse, send, send_lse, rank, send.stride(0),
            send.stride(0) // 2), WORLD=world, LOCAL=heads // world, DIM=dim, num_warps=4)


def merge(received, partial, lse, output, rank, world):
    """Combine this rank's head shard into `output` [rows, H/W, D] from its own partial and the gathered packs."""
    rows, heads, dim = partial.shape
    width = pack_width(world, heads, dim)
    received_lse = _lse_view(received, world, heads, dim)
    _launch(_merge, (rows, heads // world), (received, received_lse, partial, lse, output, rank,
            received.stride(0), received.stride(0) // 2, width, width // 2), WORLD=world, LOCAL=heads // world,
            DIM=dim, num_warps=4)
