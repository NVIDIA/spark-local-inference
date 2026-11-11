"""Rope-interpolation frequencies match the DeepSeek and YaRN closed forms."""

import math
from types import SimpleNamespace

import torch

from ignition.components.rotary import CompressedRotary, interpolated_frequencies


def _deepseek_closed_form(theta, factor, original, beta_fast, beta_slow, pairs=32):
    dims = torch.arange(pairs, dtype=torch.float32)
    frequencies = 1 / theta ** (dims / pairs)

    def boundary(rotations):
        return (2 * pairs) * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(theta))

    low = max(math.floor(boundary(beta_fast)), 0)
    high = min(math.ceil(boundary(beta_slow)), pairs - 1)
    smooth = 1 - ((dims - low) / max(high - low, 1e-3)).clamp(0, 1)
    return frequencies / factor * (1 - smooth) + frequencies * smooth


def test_interpolation_matches_the_deepseek_closed_form():
    scaled = interpolated_frequencies(40000., 32, 40., 4096, 32, 1, torch.device("cpu"))
    expected = _deepseek_closed_form(40000., 40., 4096, 32, 1)
    torch.testing.assert_close(scaled, expected, atol=0, rtol=0)
    # The ramp spans the beta boundaries: low-frequency pairs compress by the factor,
    # high-frequency pairs keep their exact frequencies.
    base = 1 / 40000. ** (torch.arange(32, dtype=torch.float32) / 32)
    assert torch.equal(scaled[0], base[0])
    assert torch.allclose(scaled[-1], base[-1] / 40., rtol=1e-6)


def test_compressed_rotary_keeps_its_interpolated_frequencies_exact():
    config = SimpleNamespace(text={
        "rope_scaling": {"factor": 40., "original_max_position_embeddings": 4096,
                         "beta_fast": 32, "beta_slow": 1},
        "compress_rope_theta": 40000., "rope_theta": 10000.})
    rotary = CompressedRotary(config, torch.device("cpu"))
    torch.testing.assert_close(rotary.frequencies[True],
                               _deepseek_closed_form(40000., 40., 4096, 32, 1), atol=0, rtol=0)
    torch.testing.assert_close(rotary.frequencies[False],
                               1 / 10000. ** (torch.arange(32, dtype=torch.float32) / 32),
                               atol=0, rtol=0)


def test_interpolation_orders_frequencies_and_compresses_by_the_factor():
    scaled = interpolated_frequencies(10000000., 32, 4., 262144, 32., 1., torch.device("cpu"))
    base = 1 / 10000000. ** (torch.arange(32, dtype=torch.float32) / 32)
    assert (scaled[1:] < scaled[:-1]).all() and (scaled > 0).all()
    ratio = scaled / base
    assert math.isclose(ratio[0].item(), 1., rel_tol=1e-6)
    assert math.isclose(ratio[-1].item(), .25, rel_tol=1e-6)
    assert ((ratio >= .25 - 1e-9) & (ratio <= 1. + 1e-9)).all()
    assert (ratio[1:] <= ratio[:-1] + 1e-9).all()
