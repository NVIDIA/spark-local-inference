# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate packed Muse Glimmer image inputs and publish generation-scoped features.

The bootstrap composes ``ignition.components.vision_state.VisionState`` (feature arena,
prefill validation and ``<|patch|>`` embedding injection) with one
``PreparedVisionGeneration`` per patch capacity, as ``nemotron_h/bootstrap.py`` does.
"""

from dataclasses import dataclass

import torch

from ignition.components.vision_delivery import PreparedFeaturePublication

from .vision import PATCH_CAPACITIES, PIXEL_WIDTH, POSITION_TABLE, WIDTH, WINDOW_PATCHES, window_capacity

FEATURE_TOKEN = 200092  # <|patch|>
PATCHES = PATCH_CAPACITIES[-1]
FEATURES = 4 * (PATCHES // 4)  # four images of 4096 merged tokens per request
HIDDEN = WIDTH


@dataclass(frozen=True)
class VisionInputs:
    vision_pixel_values: torch.Tensor
    vision_interpolation_indices: torch.Tensor
    vision_interpolation_weights: torch.Tensor
    vision_position_ids: torch.Tensor
    vision_cu_seqlens: torch.Tensor
    vision_cu_window_seqlens: torch.Tensor
    vision_reverse_index: torch.Tensor
    vision_num_patches: torch.Tensor
    vision_slot: torch.Tensor
    vision_generation: torch.Tensor
    vision_feature_offset: torch.Tensor

    @classmethod
    def allocate(cls, device):
        def zeros(shape, dtype=torch.int32):
            return torch.zeros(shape, dtype=dtype, device=device)
        n = PATCHES
        return cls(zeros((n, PIXEL_WIDTH), torch.float32), zeros((n, 4), torch.int64), zeros((n, 4), torch.float32),
                   zeros((n, 2), torch.int64), zeros(2), zeros(window_capacity(n, 1) + 1), zeros(n, torch.int64),
                   zeros(1), zeros(1) - 1, zeros(1, torch.int64), zeros(1))

    def buffers(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class PreparedVisionGeneration:
    """Sanitize one image's inputs into the encoder's metadata, encode, then publish.

    Invalid input zeroes every count, index and pixel before the encoder runs, so
    no malformed payload reaches a gather; the completion record then carries
    ``128 + error``. Window segments must stay within the window plan's
    ``max_seqlen`` (1024 patches) and end at the patch count; entries past the
    image's cells repeat the count.
    """

    def __init__(self, encoder, inputs, state, completion, frame, sequence, *, buffers=None):
        self.encoder, self.inputs, self.state = encoder, inputs, state
        device = completion.records.device
        self.errors = (torch.zeros(1, dtype=torch.int32, device=device) if buffers is None
                       else buffers.zeros((1,), torch.int32))
        self.features = (torch.zeros(1, dtype=torch.int32, device=device) if buffers is None
                         else buffers.zeros((1,), torch.int32))
        self.publish = PreparedFeaturePublication(state, inputs, completion, frame, sequence)

    def __call__(self):
        e, i, s = self.encoder, self.inputs, self.state
        m, r = e.metadata, s.requests
        capacity = e.scratch.patches.shape[0]
        windows = m.cu_window_seqlens.shape[0]
        slot = i.vision_slot.long().clamp(0, r.capacity - 1)
        count, offset = i.vision_num_patches, i.vision_feature_offset
        features = count // 4
        valid = (i.vision_slot >= 0) & (i.vision_slot < r.capacity)
        valid &= (i.vision_generation == r.generation[slot]) & (r.status[slot] == 1)
        valid &= (count > 0) & (count <= capacity) & (count % 4 == 0)
        valid &= (offset >= 0) & (offset + features <= FEATURES)
        valid &= (offset == 0) | ((s.generation[slot] == i.vision_generation) & (s.count[slot] == offset))
        valid &= (i.vision_cu_seqlens[0] == 0) & (i.vision_cu_seqlens[1] == count)
        cu_window = i.vision_cu_window_seqlens[:windows]
        segments = cu_window[1:] - cu_window[:-1]
        valid &= (cu_window[0] == 0) & (cu_window[-1] == count)
        valid &= ((segments >= 0) & (segments <= WINDOW_PATCHES)).all()
        live = torch.arange(capacity, device=count.device) < count
        feature_live = torch.arange(capacity // 4, device=count.device) < features
        pixels = i.vision_pixel_values[:capacity]
        indices, weights = i.vision_interpolation_indices[:capacity], i.vision_interpolation_weights[:capacity]
        positions, reverse = i.vision_position_ids[:capacity], i.vision_reverse_index[:capacity]
        valid &= (~live[:, None] | torch.isfinite(pixels)).all()
        valid &= (~live[:, None] | ((indices >= 0) & (indices < POSITION_TABLE))).all()
        valid &= (~live[:, None] | (torch.isfinite(weights) & (weights >= 0) & (weights <= 1))).all()
        valid &= (~live[:, None] | ((positions >= 0) & (positions < count))).all()
        valid &= (~live | ((reverse >= 0) & (reverse < count))).all()
        valid &= (s.feature_slot >= 0) & (s.feature_slot < s.arena.shape[0])
        self.errors.copy_(~valid)
        m.interpolation_indices.copy_(torch.where(valid & live[:, None], indices, 0))
        m.interpolation_weights.copy_(torch.where(valid & live[:, None], weights, 0))
        m.position_ids.copy_(torch.where(valid & live[:, None], positions, 0))
        m.reverse_index.copy_(torch.where(valid & live, reverse, 0))
        m.cu_seqlens.copy_(torch.where(valid, i.vision_cu_seqlens, 0))
        m.cu_window_seqlens.copy_(torch.where(valid, cu_window, 0))
        m.num_patches.copy_(torch.where(valid, count, 0))
        self.features.copy_(torch.where(valid, features, 0))
        output = e(torch.where(valid & live[:, None], pixels, 0))
        self.errors.copy_(self.errors | (~((~feature_live[:, None]) | torch.isfinite(output)).all()).int() * 64)
        self.publish(output, self.features, self.errors)


__all__ = ["FEATURE_TOKEN", "FEATURES", "HIDDEN", "PATCHES", "PreparedVisionGeneration", "VisionInputs"]
