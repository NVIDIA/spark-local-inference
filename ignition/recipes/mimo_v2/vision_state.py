# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Generation-scoped image features injected into MiMo's causal text prefill."""

import torch

from ignition.components.vision_delivery import PreparedFeaturePublication

from .vision import FEATURES


class PreparedVisionGeneration:
    def __init__(self, encoder, state, completion, frame, sequence, *, buffers=None):
        self.encoder, self.state = encoder, state
        self.errors = (torch.zeros(1, dtype=torch.int32, device=completion.records.device) if buffers is None
                       else buffers.zeros((1,), torch.int32))
        self.publish = PreparedFeaturePublication(state, encoder.inputs, completion, frame, sequence)

    def __call__(self):
        e, s = self.encoder, self.state
        i, r = e.inputs, s.requests
        capacity = e.scratch.patches.shape[0]
        slot = i.vision_slot.long().clamp(0, r.capacity - 1)
        count, features, offset = i.vision_num_patches, i.vision_num_features, i.vision_feature_offset
        current = (i.vision_slot >= 0) & (i.vision_slot < r.capacity)
        current &= (i.vision_generation == r.generation[slot]) & (r.status[slot] == 1)
        valid = current & (count > 0) & (count <= capacity) & (count == features * 4)
        valid &= (offset >= 0) & (offset + features <= FEATURES)
        valid &= (offset == 0) | ((s.generation[slot] == i.vision_generation) & (s.count[slot] == offset))
        valid &= (i.vision_cu_seqlens[0] == 0) & (i.vision_cu_seqlens[1] == count)
        patch_live = torch.arange(capacity, device=count.device) < count
        feature_ids = torch.arange(capacity // 4, device=count.device)
        feature_live = feature_ids < features
        columns, rows = i.vision_column_indices[:capacity // 4], i.vision_row_indices[:capacity // 4]
        valid &= (~feature_live | ((columns >= 0) & (columns < features) & (rows >= 0) & (rows < features))).all()
        valid &= (~feature_live | (rows[columns.long().clamp(0, capacity // 4 - 1)] == feature_ids)).all()
        valid &= (~patch_live[:, None] | torch.isfinite(i.vision_pixel_values[:capacity])).all()
        positions = i.vision_position_ids[:capacity]
        valid &= (~patch_live[:, None] | ((positions >= 0) & (positions < capacity))).all()
        valid &= (s.feature_slot >= 0) & (s.feature_slot < s.arena.shape[0])
        self.errors.copy_(~valid)
        e.scratch.patches.copy_(torch.where(valid & patch_live[:, None], i.vision_pixel_values[:capacity], 0))
        e.positions.copy_(torch.where(valid & patch_live[:, None], positions, 0))
        e.columns.copy_(torch.where(valid & feature_live, columns, feature_ids))
        e.rows.copy_(torch.where(valid & feature_live, rows, feature_ids))
        e.cu.copy_(torch.where(valid, i.vision_cu_seqlens, 0))
        e.num_patches.copy_(torch.where(valid, count, 0))
        e.num_features.copy_(torch.where(valid, features, 0))
        output = e()
        self.errors.copy_(self.errors | (~((~feature_live[:, None]) | torch.isfinite(output)).all()).int() * 64)
        self.publish(output, e.num_features, self.errors)
