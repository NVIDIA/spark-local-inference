# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""One rank's Inkling 2 attention: normalized streams to the partial output projection.

A rank holds whole query heads and the KV heads they read. Its fused BF16
projections produce queries, output gates, raw keys and values, relative
queries, sink logits and, on global layers, the dynamic convolution's lag
weights. Keys and values pass through the layer's causal short convolution,
then queries and relative queries are L2 clipped after their learned scales
(and multiplied by the length scale on global layers), keys are clipped and
values RMS normalized. The cache holds FP8 E4M3 keys and values with unit
scales.

Prefill (one request per graph) evaluates the relative bias as a rank-16
product inside the FP8 TMA kernel, with a per-row sink, the local window and
the sigmoid output gate. Local layers stage the request's preceding window from
its ring ahead of the chunk, then retain the last ring positions. Decode
materializes the BF16 bias `r . proj` and runs b12x paged decode with the
per-row sink; local layers read their ring through a rotated page table, so
key indices follow positions. The gate multiplies the decode output as an
elementwise step. The row-parallel output projection returns this rank's
partial sum, before the output convolution and the cross-rank reduction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ignition.components.linear import UnquantizedProjection
from ignition.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter

from .arithmetic import commit_history, dynamic_short_convolution, l2_clip, log_scaling, rms_norm, short_convolution
from .pool import KV_DTYPE, PAGE_TOKENS

RELATIVE_RANK = 16


@dataclass(frozen=True)
class Columns:
    """Column ranges of one rank's fused attention projections, by stream."""

    streams: tuple[tuple[tuple[str, int], ...], ...]

    @classmethod
    def of(cls, local, heads, kv_heads, dim, width):
        query = (("query", heads * dim), ("gate", heads * dim))
        relative = (("relative", heads * RELATIVE_RANK), ("sink", heads))
        if local:
            return cls(((*query, ("key", kv_heads * dim), ("value", kv_heads * dim), *relative),))
        return cls(((*query, *relative), (("key", kv_heads * dim), ("key_lags", kv_heads * width)),
                    (("value", kv_heads * dim), ("value_lags", kv_heads * width))))

    def widths(self):
        """Fused output widths, padded to 8 columns so that every BF16 row starts 16-byte aligned."""
        return tuple(-(-sum(width for _, width in stream) // 8) * 8 for stream in self.streams)

    def split(self, outputs):
        parts = {}
        for stream, output in zip(self.streams, outputs, strict=True):
            offset = 0
            for name, width in stream:
                parts[name] = output[:, offset:offset + width]
                offset += width
        return parts


@dataclass(frozen=True)
class AttentionWeights:
    """One rank's attention tensors of one layer in serving layouts.

    `projections` hold one fused BF16 weight per input stream with the rows of
    `Columns`. Local layers carry static `[channels, 1, W]` K/V convolution taps;
    global layers carry `[W, channels]` static lag taps. `relative_projection` is
    the checkpoint's `[16, extent]` relative-logit projection and
    `relative_rows` its contiguous `[extent, 16]` transpose.
    """

    projections: tuple[torch.Tensor, ...]
    query_clip_scale: torch.Tensor
    relative_clip_scale: torch.Tensor
    relative_projection: torch.Tensor
    relative_rows: torch.Tensor
    key_convolution: torch.Tensor
    value_convolution: torch.Tensor
    output: torch.Tensor


def fuse_projections(placed, columns: Columns, *, local):
    """Concatenate a rank's projection rows in `Columns` order; padding rows are zero."""
    from ignition.preparation.persistent import persistent_zeros

    names = {"query": "query", "gate": "output_gate", "key": "key", "value": "value", "relative": "relative_query",
             "sink": "sink", "key_lags": "key_convolution_dynamic", "value_lags": "value_convolution_dynamic"}
    fused = []
    for stream, width in zip(columns.streams, columns.widths(), strict=True):
        hidden = placed["query"].shape[1]
        weight = persistent_zeros((width, hidden), dtype=torch.bfloat16, device=placed["query"].device)
        offset = 0
        for name, rows in stream:
            source = placed[names[name]]
            if source.shape != (rows, hidden):
                raise ValueError(f"{names[name]} has {tuple(source.shape)} rows, the fused projection requires {rows}")
            weight[offset:offset + rows].copy_(source)
            offset += rows
        fused.append(weight)
    return tuple(fused)


def attention_weights(placed, *, local, heads, kv_heads, dim, width):
    """Serving attention tensors from one rank's placed checkpoint roles (`weights.checkpoint_layout`)."""
    from ignition.preparation.persistent import persistent_copy

    columns = Columns.of(local, heads, kv_heads, dim, width)
    projection = placed["relative_projection"]
    if local:
        key, value = placed["key_convolution"], placed["value_convolution"]
    else:
        key, value = placed["key_convolution_static"], placed["value_convolution_static"]
    return AttentionWeights(fuse_projections(placed, columns, local=local), placed["query_clip_scale"],
        placed["relative_clip_scale"], projection, persistent_copy(projection.t().contiguous()), key, value,
        placed["output"])


@dataclass(frozen=True)
class LayerCache:
    """A layer's rank-local storage: K/V (paged or ring pages) and K/V convolution histories."""

    key: torch.Tensor
    value: torch.Tensor
    key_history: torch.Tensor
    value_history: torch.Tensor
    ring_row_pages: int = 0

    @classmethod
    def of(cls, pool, layer):
        if pool.text.local(layer):
            return cls(pool.ring_pages(layer, "key"), pool.ring_pages(layer, "value"),
                       pool.history(layer, "key_convolution"), pool.history(layer, "value_convolution"),
                       pool.ring_row_pages)
        return cls(pool.cache(layer, "key"), pool.cache(layer, "value"), pool.history(layer, "key_convolution"),
                   pool.history(layer, "value_convolution"))


def ring_slots(positions, rows, ring, row_pages):
    """Flat cache slots of positions in their requests' rings."""
    slot = positions.clamp_min(0).remainder(ring)
    page = rows * row_pages + slot // PAGE_TOKENS
    return torch.where(positions >= 0, page * PAGE_TOKENS + slot % PAGE_TOKENS, -1)


def rotated_ring(metadata, ring, window, row_pages):
    """Page tables and lengths that list each decoding request's window in position order.

    Returns int32 `[requests, ring pages]` pages, starting with the page that holds
    position `max(p - window + 1, 0)` of the request's current position `p`, and
    each request's key count from that page's first position through `p`.
    """
    m = metadata
    count = ring // PAGE_TOKENS
    current = (m.starts + m.lengths - 1).clamp_min(0)
    first = (current - window + 1).clamp_min(0) // PAGE_TOKENS
    columns = torch.arange(count, device=current.device)
    pages = m.state_rows[:, None] * row_pages + (first[:, None] + columns).remainder(count)
    lengths = torch.where(m.lengths > 0, current + 1 - first * PAGE_TOKENS, 0)
    return pages.int(), lengths.int()


class PreparedAttention(torch.nn.Module):
    """A rank's attention for one layer and graph mode (`prefill` or `decode`)."""

    def __init__(self, layer, weights: AttentionWeights, text, cache: LayerCache, metadata, *, mode, tokens,
                 session, workspace, name, examples=None, tuning_page_count=None):
        super().__init__()
        if mode not in ("prefill", "decode"):
            raise ValueError(f"unsupported Inkling attention mode: {mode}")
        self.layer, self.mode, self.text, self.cache = layer, mode, text, cache
        self.local = text.local(layer)
        dim = text.head_dim
        heads = weights.query_clip_scale.shape[0]
        kv_heads = cache.key.shape[2]
        self.heads, self.kv_heads, self.dim = heads, kv_heads, dim
        self.window = text.window
        self.ring = text.window + PAGE_TOKENS
        self.columns = Columns.of(self.local, heads, kv_heads, dim, text.convolution_width)
        self.weights = weights
        device = weights.output.device
        examples = {} if examples is None else examples

        def example(role, shape, dtype=torch.bfloat16):
            from ignition.preparation.examples import declaration_tensor

            return declaration_tensor(examples, role, shape, dtype, device)

        source = example("inkling.attention.source", (tokens, text.hidden))
        self.projections = tuple(
            UnquantizedProjection(source, weight, example(f"inkling.attention.projection{index}", (tokens, width)),
                                  session=session, name=f"{name}.projection{index}").functional(session=session)
            for index, (weight, width) in enumerate(zip(weights.projections, self.columns.widths(), strict=True)))
        attended = example("inkling.attention.attended", (tokens, heads * dim))
        self.output_projection = UnquantizedProjection(attended, weights.output,
            example("inkling.attention.partial", (tokens, text.hidden)), session=session,
            name=f"{name}.output").functional(session=session)
        query = example("inkling.attention.query", (tokens, heads, dim))
        output = example("inkling.attention.output", (tokens, heads, dim))
        sink = example("inkling.attention.sink", (tokens, heads), torch.float32)
        key = example("inkling.attention.key", (tokens, kv_heads, dim))
        value = example("inkling.attention.value", (tokens, kv_heads, dim))
        slots = example("inkling.attention.slots", (tokens,), torch.int64)
        cu = metadata.query_start_loc
        requests = metadata.lengths.numel()
        self.writer = None
        if mode == "prefill":
            if requests != 1:
                raise ValueError("Inkling prefill graphs hold one request")
            if self.local:
                self.staged_pages = -(-(self.window - 1 + tokens) // PAGE_TOKENS)
                cache_key = example("inkling.attention.staged_key", (self.staged_pages, PAGE_TOKENS, kv_heads, dim),
                                    KV_DTYPE)
                cache_value = example("inkling.attention.staged_value",
                                      (self.staged_pages, PAGE_TOKENS, kv_heads, dim), KV_DTYPE)
                pages = torch.arange(self.staged_pages, dtype=torch.int32, device=device).view(1, -1)
                self.staged_table = pages
            else:
                cache_key, cache_value, pages = cache.key, cache.value, metadata.page_table
                self.writer = PreparedPagedKVWriter(key, value, cache.key, cache.value, slots, session=session,
                                                    name=f"{name}.cache_write").functional()
            lengths = example("inkling.attention.lengths", (1,), torch.int32)
            factor = example("inkling.attention.relative_factor", (tokens, heads, RELATIVE_RANK))
            gate = example("inkling.attention.gate", (tokens, heads, dim))
            self.attention = PreparedPagedAttention(query, cache_key, cache_value, pages, lengths, cu, output,
                sink=sink, mode="prefill", window=self.window - 1 if self.local else -1,
                relative_factor=factor, relative_projection=weights.relative_rows, output_gate=gate,
                tuning_page_count=tuning_page_count or cache_key.shape[0],
                session=session, workspace=workspace, name=f"{name}.attention")
            if self.attention.tma is None:
                raise ValueError("Inkling prefill requires the FP8 TMA attention program")
            self.attention = self.attention.functional(session=session, k_cache=None if self.local else cache.key,
                v_cache=None if self.local else cache.value, cache_seqlens=None, page_table=None, cu_seqlens_q=None)
        else:
            if self.local:
                pages = example("inkling.attention.ring_table", (requests, self.ring // PAGE_TOKENS), torch.int32)
            else:
                pages = metadata.page_table
            self.writer = PreparedPagedKVWriter(key, value, cache.key, cache.value, slots, session=session,
                                                name=f"{name}.cache_write").functional()
            lengths = example("inkling.attention.lengths", (requests,), torch.int32)
            bias = example("inkling.attention.relative_bias", (tokens, heads, weights.relative_projection.shape[1]))
            self.attention = PreparedPagedAttention(query, cache.key, cache.value, pages, lengths, cu, output,
                sink=sink, mode="decode", window=self.window - 1 if self.local else -1, relative_bias=bias,
                tuning_page_count=tuning_page_count or cache.key.shape[0], caller_owned_placeholders=True,
                session=session, workspace=workspace, name=f"{name}.attention")
            if not self.attention.decode:
                raise ValueError("Inkling decode requires b12x paged decode with relative bias and row sinks")
            self.attention = self.attention.functional(session=session, cache_seqlens=None, page_table=None,
                                                        cu_seqlens_q=None)

    def project(self, streams, metadata):
        """Fused projections, convolved and clipped K/V and queries, sinks and gates of `streams`."""
        t, m = self.text, metadata
        inputs = streams if len(streams) == len(self.projections) else streams * len(self.projections)
        parts = self.columns.split([projection(source) for projection, source in zip(self.projections, inputs,
                                                                                    strict=True)])
        tokens = m.positions.shape[0]
        heads, kv_heads, dim = self.heads, self.kv_heads, self.dim
        key_raw, value_raw = parts["key"], parts["value"]
        if self.local:
            key = short_convolution(key_raw, self.weights.key_convolution, self.cache.key_history, m)
            value = short_convolution(value_raw, self.weights.value_convolution, self.cache.value_history, m)
            tau = None
        else:
            width = t.convolution_width
            key = dynamic_short_convolution(key_raw, self.weights.key_convolution,
                                            parts["key_lags"].reshape(tokens, kv_heads, width), self.cache.key_history, m)
            value = dynamic_short_convolution(value_raw, self.weights.value_convolution,
                                              parts["value_lags"].reshape(tokens, kv_heads, width),
                                              self.cache.value_history, m)
            tau = log_scaling(m.positions, t.log_scaling_floor, t.log_scaling_alpha)
        commit_history(self.cache.key_history, key_raw, m)
        commit_history(self.cache.value_history, value_raw, m)
        query = l2_clip(parts["query"].reshape(tokens, heads, dim), t.qk_clip, scale=self.weights.query_clip_scale,
                        tau=tau)
        key = l2_clip(key.reshape(tokens, kv_heads, dim), t.qk_clip)
        value = rms_norm(value.reshape(tokens, kv_heads, dim), None, t.eps)
        relative = l2_clip(parts["relative"].reshape(tokens, heads, RELATIVE_RANK), t.relative_clip,
                           scale=self.weights.relative_clip_scale, tau=tau)
        return dict(query=query, key=key, value=value, relative=relative, sink=parts["sink"].float(),
                    gate=parts["gate"].reshape(tokens, heads, dim))

    def attend(self, streams, metadata):
        """The gated attention `[tokens, heads * dim]` and its partial output projection."""
        m = metadata
        values = self.project(streams, m)
        if self.mode == "prefill":
            attended = self._prefill(values, m)
        else:
            attended = self._decode(values, m)
        attended = attended.flatten(1)
        return attended, self.output_projection(attended)

    def forward(self, streams, metadata):
        return self.attend(streams, metadata)[1]

    def _prefill(self, v, m):
        key, value = v["key"], v["value"]
        if self.local:
            staged_key, staged = _stage(self.cache.key, key, m, window=self.window, ring=self.ring,
                                        row_pages=self.cache.ring_row_pages, pages=self.staged_pages)
            staged_value, _ = _stage(self.cache.value, value, m, window=self.window, ring=self.ring,
                                     row_pages=self.cache.ring_row_pages, pages=self.staged_pages)
            _retain(self.cache.key, key, m, ring=self.ring, row_pages=self.cache.ring_row_pages)
            _retain(self.cache.value, value, m, ring=self.ring, row_pages=self.cache.ring_row_pages)
            return self.attention(v["query"], output_gate=v["gate"], sink=v["sink"], relative_factor=v["relative"],
                                  k_cache=staged_key, v_cache=staged_value, cache_seqlens=staged.reshape(1),
                                  page_table=self.staged_table, cu_seqlens_q=m.query_start_loc)
        self.writer(key, value, _page_slots(m))
        lengths = torch.where(m.lengths > 0, m.starts + m.lengths, 0).int()
        return self.attention(v["query"], output_gate=v["gate"], sink=v["sink"], relative_factor=v["relative"],
                              cache_seqlens=lengths, page_table=m.page_table, cu_seqlens_q=m.query_start_loc)

    def _decode(self, v, m):
        if self.local:
            slots = ring_slots(m.positions, m.state_rows[m.requests], self.ring, self.cache.ring_row_pages)
            pages, lengths = rotated_ring(m, self.ring, self.window, self.cache.ring_row_pages)
        else:
            slots = _page_slots(m)
            pages = m.page_table
            lengths = torch.where(m.lengths > 0, m.starts + m.lengths, 0).int()
        self.writer(v["key"], v["value"], slots)
        tokens = v["relative"].shape[0]
        bias = (v["relative"].reshape(tokens * self.heads, RELATIVE_RANK) @ self.weights.relative_projection)
        attended = self.attention(v["query"], sink=v["sink"], relative_bias=bias.view(tokens, self.heads, -1),
                                  cache_seqlens=lengths, page_table=pages, cu_seqlens_q=m.query_start_loc)
        return (attended.float() * torch.sigmoid(v["gate"].float())).to(attended.dtype)


def _page_slots(m):
    page = m.token_page_table.gather(1, (m.positions.clamp(min=0) // PAGE_TOKENS)[:, None]).squeeze(1)
    return torch.where((m.positions >= 0) & (page >= 0), page.long() * PAGE_TOKENS + m.positions % PAGE_TOKENS, -1)


def _ring_index(positions, row, ring, row_pages):
    slot = positions.clamp_min(0).remainder(ring)
    return row * row_pages + slot // PAGE_TOKENS, slot % PAGE_TOKENS


def _stage(ring_pages, values, m, *, window, ring, row_pages, pages):
    """One prefill request's preceding `window - 1` ring entries followed by its chunk, as FP8 pages.

    Staged index `i` is position `max(start - window + 1, 0) + i`; returns the staged pages and
    the staged key count.
    """
    start, length, row = m.starts[0], m.lengths[0].long(), m.state_rows[0]
    first = (start - window + 1).clamp_min(0)
    positions = first + torch.arange(pages * PAGE_TOKENS, device=values.device)
    page, offset = _ring_index(positions, row, ring, row_pages)
    previous = ring_pages[page, offset].to(values.dtype)
    current = values[(positions - start).clamp(0, values.shape[0] - 1)]
    staged = torch.where((positions < start)[:, None, None], previous, current)
    staged = torch.where((positions < start + length)[:, None, None], staged, 0)
    staged = staged.clamp(-448, 448).to(KV_DTYPE).view(pages, PAGE_TOKENS, *values.shape[1:])
    return staged, (start - first + length).int()


def _retain(ring_pages, values, m, *, ring, row_pages):
    """Write the chunk's positions among the request's last `ring` positions into its ring."""
    start, length, row = m.starts[0], m.lengths[0].long(), m.state_rows[0]
    end = start + length
    positions = end - ring + torch.arange(ring, device=values.device)
    page, offset = _ring_index(positions, row, ring, row_pages)
    current = values[(positions - start).clamp(0, values.shape[0] - 1)].clamp(-448, 448)
    keep = ((positions < start) | (positions < 0) | (length <= 0))[:, None, None]
    # E4M3 values round-trip exactly through BF16.
    retained = torch.where(keep, ring_pages[page, offset].to(values.dtype), current)
    ring_pages[page, offset] = retained.to(KV_DTYPE)
