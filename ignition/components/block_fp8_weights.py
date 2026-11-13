# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Lossless expansion of square-block UE8M0 scales into native MXFP8 rows."""

import torch

from ignition.components.weights import DirectWeightLoader, _aligned_range
from ignition.preparation.persistent import persistent_cat, persistent_empty


class BlockFP8WeightLoader(DirectWeightLoader):
    @property
    def block_size(self):
        shape = self.config.quantization["weight_block_size"]
        if shape not in ([32, 32], [128, 128]):
            raise ValueError("native FP8 loading requires square block-32 or block-128 scales")
        return shape[0]

    def load_linear_group(self, prefixes, *, row_ranges=None, column_range=None, pad=False, temporary=False):
        """Pack aligned checkpoint slices, optionally zero-padding beyond their source extents."""
        prefixes = tuple(prefixes)
        ranges = (None,) * len(prefixes) if row_ranges is None else row_ranges
        parts = [
            self._dense_blocks(prefix, rows=rows, columns=column_range, pad=pad, temporary=temporary or len(prefixes) > 1)
            for prefix, rows in zip(prefixes, ranges, strict=True)
        ]
        return self._pack_blocks(parts, dim=0, temporary=temporary)

    def load_linear_columns(self, prefix, column_ranges, *, rows=None):
        """Gather aligned input-column ranges without changing checkpoint values or scales."""
        column_ranges = tuple(column_ranges)
        parts = [self._dense_blocks(prefix, rows=rows, columns=columns, temporary=len(column_ranges) > 1)
                 for columns in column_ranges]
        if not parts:
            raise ValueError("a projection requires at least one input-column range")
        return self._pack_blocks(parts, dim=1)

    def _pack_blocks(self, parts, *, dim, temporary=False):
        from b12x.gemm.blockscaled import pack_weight

        self._session.flush()
        columns = sum(part[0].shape[1] for part in parts) if dim == 1 else parts[0][0].shape[1]
        concatenate = torch.cat if temporary or columns % 128 else persistent_cat
        values = concatenate([p[0] for p in parts], dim=dim) if len(parts) > 1 else parts[0][0]
        scales = concatenate(
            [
                p[1].repeat_interleave(self.block_size, dim=0).repeat_interleave(self.block_size // 32, dim=1)
                for p in parts
            ], dim=dim,
        )
        options = {} if temporary else {"allocate": persistent_empty}
        return pack_weight(values, scales, recipe="mxfp8", **options)

    def _dense_blocks(self, prefix, *, rows=None, columns=None, pad=False, temporary=True):
        self._require_open()
        weight, scale = self._sources[f"{prefix}.weight"], self._sources[f"{prefix}.scale"]
        block = self.block_size
        if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
            raise ValueError(f"{prefix}: expected native FP8 dense weights")
        n, k = weight.shape
        if scale.dtype != torch.float8_e8m0fnu or scale.shape != (
            (n + block - 1) // block,
            (k + block - 1) // block,
        ):
            raise ValueError(f"{prefix}: expected UE8M0 scales for {block}-by-{block} blocks")
        r0, r1 = _aligned_range(rows, max(n, rows[1]) if pad and rows else n, block)
        c0, c1 = _aligned_range(columns, max(k, columns[1]) if pad and columns else k, block)
        allocate = torch.empty if temporary or (c1 - c0) % 128 else persistent_empty
        values = allocate((r1 - r0, c1 - c0), dtype=weight.dtype, device=self.device)
        scales = torch.empty(((r1 - r0) // block, (c1 - c0) // block), dtype=torch.uint8, device=self.device)
        nr, nc = max(0, min(r1, n) - r0), max(0, min(c1, k) - c0)
        if nr != r1 - r0 or nc != c1 - c0:
            values.zero_()
            scales.fill_(127)
        if nr and nc:
            sr, sc = -(-nr // block), -(-nc // block)
            self._copy(values[:nr, :nc], weight[r0:r0 + nr, c0:c0 + nc])
            self._copy(scales[:sr, :sc], scale[r0 // block:r0 // block + sr,
                                             c0 // block:c0 // block + sc].view(torch.uint8))
        return values, scales

    def load_attention_output(self, prefix, groups, *, pad=False):
        from b12x.gemm import wo_projection

        a, sa = self._dense_blocks(prefix + ".wo_a", rows=groups, pad=pad, temporary=False)
        b, sb = self._dense_blocks(prefix + ".wo_b", columns=groups, pad=pad, temporary=False)
        self._session.flush()
        c = self.config.text
        return wo_projection.pack_weights(
            a,
            sa.view(torch.float8_e8m0fnu),
            b,
            sb.view(torch.float8_e8m0fnu),
            groups=(groups[1] - groups[0]) // c["o_lora_rank"],
            group_width=c["num_attention_heads"] * c["head_dim"] // c["o_groups"],
            rank=c["o_lora_rank"],
            hidden=c["hidden_size"],
            block_size=(self.block_size, self.block_size),
            allocate=persistent_empty,
        )
