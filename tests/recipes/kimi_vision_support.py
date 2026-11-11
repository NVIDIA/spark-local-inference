# SPDX-License-Identifier: Apache-2.0

"""Mutable inputs for standalone MoonViT encoder replay tests."""

import torch


class VisionCall:
    def __init__(self, encoder, device):
        self.encoder = encoder
        n = encoder.capacity
        self.patches = torch.zeros((n, 588), dtype=torch.bfloat16, device=device)
        self.positions = torch.zeros((n, 2), dtype=torch.int64, device=device)
        self.interpolation_indices = torch.zeros((n, 4), dtype=torch.int64, device=device)
        self.interpolation_weights = torch.zeros((n, 4), dtype=torch.float32, device=device)
        self.merge_indices = torch.zeros((n // 4, 4), dtype=torch.int64, device=device)
        self.cu = torch.zeros((2,), dtype=torch.int32, device=device)

    def __call__(self):
        return self.encoder(self.patches, self.positions, self.interpolation_indices,
                            self.interpolation_weights, self.merge_indices, self.cu)
