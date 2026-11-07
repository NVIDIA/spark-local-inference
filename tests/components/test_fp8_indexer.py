# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""FP8 index declarations reject invalid head-weight multipliers."""

import pytest

from ignition.components.fp8_indexer import PreparedFP8Indexer


@pytest.mark.parametrize("scale", [0, -1, float("inf"), -float("inf"), float("nan"), True, "1", [1]])
def test_head_weight_scale_requires_a_finite_positive_scalar(scale):
    with pytest.raises(ValueError, match="finite positive scalar"):
        PreparedFP8Indexer(None, None, None, None, None, None, None, mode="decode", session=None,
            name="index", workspace=None, buffers=None, head_weight_scale=scale)


@pytest.mark.parametrize("field", ["hadamard", "key_power_of_two"])
@pytest.mark.parametrize("value", [0, 1, None, "false"])
def test_quantization_declarations_require_booleans(field, value):
    with pytest.raises(ValueError, match="must be booleans"):
        PreparedFP8Indexer(None, None, None, None, None, None, None, mode="decode", session=None,
            name="index", workspace=None, buffers=None, **{field: value})
