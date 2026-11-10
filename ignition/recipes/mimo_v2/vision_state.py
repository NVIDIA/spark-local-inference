# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Generation-scoped image features injected into MiMo's causal text prefill."""

import torch

from ignition.components.vision_delivery import PreparedFeaturePublication

from .vision import FEATURES


class PreparedVisionGeneration:
    def __init__(self, encoder, state, completion, frame, sequence):
        self.encoder, self.state = encoder, state
        self.publish = PreparedFeaturePublication(state, encoder.inputs, completion, frame, sequence)

    def __call__(self):
        e, s = self.encoder, self.state
        i, r = e.inputs, s.requests
        capacity = e.capacity
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
        errors = (~valid).int()
        patches = torch.where(valid & patch_live[:, None], i.vision_pixel_values[:capacity], 0).bfloat16()
        positions = torch.where(valid & patch_live[:, None], positions, 0)
        columns = torch.where(valid & feature_live, columns, feature_ids).int()
        rows = torch.where(valid & feature_live, rows, feature_ids).int()
        cu = torch.where(valid, i.vision_cu_seqlens, 0)
        features = torch.where(valid, features, 0)
        output = e(patches, positions, columns, rows, cu)
        errors = errors | (~((~feature_live[:, None]) | torch.isfinite(output)).all()).int() * 64
        self.publish(output, features, errors)
