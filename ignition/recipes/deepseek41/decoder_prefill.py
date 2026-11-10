# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Pack a prefill chunk's decoder suffix while preserving absolute cache positions.

Decoder SWA bounded replay is an approximation described in section 3.2.2 of
the DeepSeek V4.1 technical report. Global KV must be produced from every
encoder row before gathering the suffix. Chunks that fit the replay window
retain their complete continuation history; a truncated chunk establishes a
fresh SWA boundary at its first retained row.
"""

from dataclasses import dataclass

import torch

from ignition.components.paged_batch import Metadata


@dataclass(frozen=True)
class DecoderSuffix:
    metadata: Metadata
    indices: torch.Tensor
    live: torch.Tensor
    replay_start: torch.Tensor

    def gather(self, source):
        live = self.live.view((-1,) + (1,) * (source.ndim - 1))
        return torch.where(live, source[self.indices], 0)


class PreparedDecoderSuffix:
    def __init__(self, metadata, *, window=128):
        if metadata.starts.numel() != 1 or metadata.positions.numel() <= window or window <= 0:
            raise ValueError("decoder suffix packing requires one prefill chunk larger than its window")
        self.window = window
        self.metadata = Metadata.allocate(window, 1, metadata.page_table.shape[1], metadata.positions.device,
                                          samples=metadata.sample_indices.numel())
        self.replay_start = torch.zeros_like(metadata.starts)

    def functional(self):
        self.metadata = self.replay_start = None
        return self

    def begin(self, source):
        offsets = torch.arange(self.window, device=source.positions.device)
        count = source.lengths[0].long()
        kept = count.clamp(max=self.window)
        skipped = count - kept
        indices = (source.query_start_loc[0] + skipped + offsets).clamp(0, source.positions.numel() - 1)
        live = (offsets < kept) & (source.positions[indices] >= 0)
        starts = source.starts + skipped
        pages = source.page_table
        metadata = Metadata(
            token_ids=torch.where(live, source.token_ids[indices], 0),
            sample_indices=(source.sample_indices - skipped).clamp(0, self.window - 1),
            query_start_loc=torch.cat((torch.zeros(1, dtype=torch.int32, device=offsets.device), kept.view(1).int())),
            state_rows=source.state_rows,
            starts=starts,
            lengths=kept.view(1).int(),
            requests=torch.zeros(self.window, dtype=torch.int64, device=offsets.device),
            positions=torch.where(live, source.positions[indices], -1),
            page_table=pages,
            token_page_table=pages.expand(self.window, -1),
            live_counts=torch.cat((kept.view(1).int(), source.live_counts[1:])),
            max_query_tokens=self.window,
        )
        return DecoderSuffix(metadata, indices, live, torch.where(skipped > 0, starts, 0))
