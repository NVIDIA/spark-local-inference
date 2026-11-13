# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generation-scoped image features injected into causal text prefill."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrefillVisionInputs:
    vision_token_indices: torch.Tensor
    vision_feature_slot: torch.Tensor

    @classmethod
    def allocate(cls, tokens, device):
        return cls(torch.full((tokens,), -1, dtype=torch.int32, device=device),
                   torch.full((1,), -1, dtype=torch.int32, device=device))

    def buffers(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class VisionState:
    def __init__(self, inputs, prefill, requests, *, image_token, features, hidden, image_slots=None):
        self.inputs, self.prefill, self.requests = inputs, prefill, requests
        self.image_token = image_token
        device = inputs.token_ids.device
        from .capacity import image_capacity

        self.feature_slot = prefill.vision_feature_slot
        self.arena = torch.zeros((image_capacity(requests.capacity, image_slots), features, hidden),
                                 dtype=torch.bfloat16, device=device)
        self.generation = torch.zeros(requests.capacity, dtype=torch.int64, device=device)
        self.count = torch.zeros(requests.capacity, dtype=torch.int32, device=device)

    def validate_prefill(self, tokens, lengths):
        b = self.inputs
        slot = b.row_slot_ids[:1].long().clamp(0, self.requests.capacity - 1)
        index = self.prefill.vision_token_indices[:tokens.numel()]
        live = torch.arange(tokens.numel(), device=tokens.device) < lengths[0]
        image = tokens == self.image_token
        valid = (~image & (index == -1)) | (image & (index >= 0) & (index < self.count[slot]))
        valid &= ~image | (self.generation[slot] == b.row_generations[0])
        valid &= ~image | ((self.feature_slot >= 0) & (self.feature_slot < self.arena.shape[0]))
        return (~live | valid).all()

    def inject(self, metadata, embedding):
        slot = self.feature_slot.long().clamp(0, self.arena.shape[0] - 1)
        indices = self.prefill.vision_token_indices[:embedding.shape[0]].long().clamp(0, self.arena.shape[1] - 1)
        embedding.copy_(torch.where((metadata.token_ids == self.image_token)[:, None],
                                    self.arena[slot, indices], embedding))
