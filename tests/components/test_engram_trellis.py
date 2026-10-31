# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from ignition.components.engram_trellis import reference_decode_rows, reference_decode_rows_unpack


def test_reference_decoders_agree_on_synthetic_rows():
    generator = torch.Generator().manual_seed(17)
    trellis = torch.randint(0, 256, (6, 128), generator=generator, dtype=torch.uint8)
    scales = torch.full((6, 8), 127, dtype=torch.uint8)  # UE8M0 exponent 0
    design = reference_decode_rows(trellis, scales)
    packer = reference_decode_rows_unpack(trellis, scales)
    assert design.shape == (6, 256)
    assert torch.equal(design, packer)


def test_reference_decoders_reject_wrong_row_width():
    with pytest.raises(ValueError, match="uint8 .*128"):
        reference_decode_rows(torch.zeros((1, 127), dtype=torch.uint8),
                              torch.zeros((1, 8), dtype=torch.uint8))
