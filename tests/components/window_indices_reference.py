# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Literal window-address formulas used to check shared metadata buffers.

The formula bodies are copied from the DeepSeek 4 and DeepSeek 4.1 attention
forwards at Flare revision c5f920cc. Cache payload gathers are excluded because
they remain layer-specific; every window-address assignment is retained.
"""

import torch


class WindowReference:
    def __init__(self, metadata, *, mode, tokens, page_size, width, visible=None, replay_start=None):
        self.metadata, self.mode, self.visible = metadata, mode, visible
        self.replay_start, self.window_page_size = replay_start, page_size
        self.scratch = None
        device = metadata.positions.device
        self.offsets = torch.arange(width, device=device)
        self.window_slots = torch.full((tokens,), -1, dtype=torch.int64, device=device)
        self.window_indices = torch.full((tokens, width), -1, dtype=torch.int32, device=device)
        self.window_lengths = torch.zeros(tokens, dtype=torch.int32, device=device)


class LiteralDeepSeek4WindowIndices(WindowReference):
    def __call__(self):
        m = self.metadata
        live, position = m.positions >= 0, m.positions.clamp_min(0)
        rows, starts = m.state_rows[m.requests], m.starts[m.requests]
        left = (position - 127).clamp_min(0)
        right = position
        if self.visible is not None:
            image_left, image_right = self.visible
            left = torch.minimum(left, position - image_left)
            right = position + image_right
        self.window_lengths.copy_(torch.where(live, right - left + 1, 0))
        positions = left[:, None] + self.offsets
        if self.mode == "decode":
            slots = rows * 128 + position % 128
            indices = rows[:, None] * 128 + positions % 128
        else:
            base = m.requests * self.window_page_size
            slots = base + 128 + position - starts
            indices = torch.where(
                positions < starts[:, None], positions % 128, 128 + positions - starts[:, None]
            )
            indices = indices + base[:, None]
            if self.mode == "draft":
                context = starts.clamp(max=128)
                self.window_lengths.copy_(torch.where(live, context + m.lengths[m.requests], 0))
                indices = base[:, None] + torch.where(
                    self.offsets < context[:, None], self.offsets, 128 + self.offsets - context[:, None]
                )
        self.window_slots.copy_(torch.where(live, slots, -1))
        self.window_indices.copy_(torch.where(self.offsets < self.window_lengths[:, None], indices, -1))


class LiteralDeepSeek41WindowIndices(WindowReference):
    def __call__(self):
        m, s = self.metadata, self.scratch  # noqa: F841 - retain the literal forward prefix.
        live = m.positions >= 0
        position = m.positions.clamp(min=0)
        rows = m.state_rows[m.requests]
        starts = m.starts[m.requests]
        available = position + 1
        if self.replay_start is not None:
            available = available - self.replay_start[m.requests]
        self.window_lengths.copy_(torch.where(live, available.clamp(0, 128), 0))
        positions = position[:, None] - self.window_lengths[:, None] + 1 + self.offsets
        if self.mode == "decode":
            slots = rows * 128 + position % 128
            indices = rows[:, None] * 128 + positions % 128
        else:
            base = m.requests * self.window_page_size
            slots = base + 128 + position - starts
            indices = torch.where(positions < starts[:, None], positions % 128, 128 + positions - starts[:, None])
            indices = indices + base[:, None]
            if self.mode == "draft":
                context = starts.clamp(max=128)
                self.window_lengths.copy_(torch.where(live, context + m.lengths[m.requests], 0))
                indices = base[:, None] + torch.where(self.offsets < context[:, None],
                    self.offsets, 128 + self.offsets - context[:, None])
        self.window_slots.copy_(torch.where(live, slots, -1))
        self.window_indices.copy_(torch.where(self.offsets < self.window_lengths[:, None], indices, -1))
