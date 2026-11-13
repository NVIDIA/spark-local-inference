# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate packed RADIO inputs and publish generation-scoped image features."""

import torch

from ignition.components.vision_delivery import PreparedFeaturePublication
from .vision import FEATURES, REGISTER_TOKENS


class PreparedVisionGeneration:
    def __init__(self, encoder, state, completion, frame, sequence):
        self.encoder, self.state = encoder, state
        self.publish = PreparedFeaturePublication(state, encoder.inputs, completion, frame, sequence)

    def __call__(self):
        e, s = self.encoder, self.state
        i, r = e.inputs, s.requests
        capacity = e.patch_capacity
        slot = i.vision_slot.long().clamp(0, r.capacity - 1)
        count, features, offset = i.vision_num_patches, i.vision_num_features, i.vision_feature_offset
        valid = (i.vision_slot >= 0) & (i.vision_slot < r.capacity)
        valid &= (i.vision_generation == r.generation[slot]) & (r.status[slot] == 1)
        valid &= (count > 0) & (count <= capacity) & (count == features * 4)
        valid &= (offset >= 0) & (offset + features <= FEATURES)
        valid &= (offset == 0) | ((s.generation[slot] == i.vision_generation) & (s.count[slot] == offset))
        valid &= (i.vision_cu_seqlens[0] == 0) & (i.vision_cu_seqlens[1] == count)
        live = torch.arange(capacity, device=count.device) < count
        feature_live = torch.arange(capacity // 4, device=count.device) < features
        indices = i.vision_interpolation_indices[:capacity]
        weights = i.vision_interpolation_weights[:capacity]
        merge = i.vision_merge_indices[:capacity // 4]
        valid &= (~live[:, None] | torch.isfinite(i.vision_pixel_values[:capacity])).all()
        valid &= (~live[:, None] | ((indices >= 0) & (indices < 16384))).all()
        valid &= (~live[:, None] | (torch.isfinite(weights) & (weights >= 0) & (weights <= 1))).all()
        valid &= (~live | ((weights.sum(-1) - 1).abs() < 1e-5)).all()
        valid &= (~feature_live[:, None] | ((merge >= 0) & (merge < count))).all()
        valid &= (s.feature_slot >= 0) & (s.feature_slot < s.arena.shape[0])
        errors = (~valid).int()
        patches = torch.where(valid & live[:, None], i.vision_pixel_values[:capacity], 0)
        indices = torch.where(valid & live[:, None], indices, 0)
        weights = torch.where(valid & live[:, None], weights, 0)
        merge = torch.where(valid & feature_live[:, None], merge, 0)
        cu = torch.cat((torch.zeros_like(count), torch.where(valid, count + REGISTER_TOKENS, 0)))
        num_features = torch.where(valid, features, 0)
        output = e(patches, indices, weights, merge, cu)
        errors = errors | (~((~feature_live[:, None]) | torch.isfinite(output)).all()).int() * 64
        self.publish(output, num_features, errors)
