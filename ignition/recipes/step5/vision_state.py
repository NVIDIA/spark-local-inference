# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate crop and overview geometry and publish request-owned image features."""

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
        valid = (i.vision_slot >= 0) & (i.vision_slot < r.capacity)
        valid &= (i.vision_generation == r.generation[slot]) & (r.status[slot] == 1)
        valid &= (count > 0) & (count <= capacity) & (count == features * 16)
        valid &= (offset >= 0) & (offset + features <= FEATURES)
        valid &= (offset == 0) | ((s.generation[slot] == i.vision_generation) & (s.count[slot] == offset))
        cu = i.vision_cu_seqlens
        lengths = cu[1:] - cu[:-1]
        valid &= (cu[0] == 0) & (cu[-1] == count) & ((lengths == 0) | (lengths == 1296) | (lengths == 2704)).all()
        valid &= ((lengths == 2704).sum() == 1)
        live = torch.arange(capacity, device=count.device) < count
        positions = i.vision_position_ids[:capacity]
        indices, weights = i.vision_interpolation_indices[:capacity], i.vision_interpolation_weights[:capacity]
        valid &= (~live[:, None] | torch.isfinite(i.vision_pixel_values[:capacity])).all()
        valid &= (~live[:, None] | ((positions >= 0) & (positions < 52))).all()
        valid &= (~live[:, None] | ((indices >= 0) & (indices < 2704))).all()
        valid &= (~live[:, None] | (torch.isfinite(weights) & (weights >= 0) & (weights <= 1))).all()
        valid &= (~live | ((weights.sum(-1) - 1).abs() < 1e-5)).all()
        for divisor, source in ((4, i.vision_downsample1_indices), (16, i.vision_downsample2_indices)):
            active = torch.arange(capacity // divisor, device=count.device) < count // divisor
            source = source[:capacity // divisor]
            valid &= (~active[:, None] | ((source >= -1) & (source < count // (divisor // 4)))).all()
        valid &= (s.feature_slot >= 0) & (s.feature_slot < s.arena.shape[0])
        self.errors.copy_(~valid)
        e.scratch.patches.copy_(torch.where(valid & live[:, None], i.vision_pixel_values[:capacity], 0))
        e.positions.copy_(torch.where(valid & live[:, None], positions, 0))
        e.interpolation_indices.copy_(torch.where(valid & live[:, None], indices, 0))
        e.interpolation_weights.copy_(torch.where(valid & live[:, None], weights, 0))
        for divisor, source, destination in ((4, i.vision_downsample1_indices, e.first_indices),
                                              (16, i.vision_downsample2_indices, e.second_indices)):
            active = torch.arange(capacity // divisor, device=count.device) < count // divisor
            destination.copy_(torch.where(valid & active[:, None], source[:capacity // divisor], -1))
        e.cu.copy_(torch.where(valid, cu, 0))
        e.num_patches.copy_(torch.where(valid, count, 0))
        e.num_features.copy_(torch.where(valid, features, 0))
        output = e()
        active = torch.arange(capacity // 16, device=count.device) < features
        self.errors.copy_(self.errors | (~((~active[:, None]) | torch.isfinite(output)).all()).int() * 64)
        self.publish(output, e.num_features, self.errors)
