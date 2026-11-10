# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Host-side admission, byte-layout and teardown contracts for BRRR preparation."""

from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.deepseek41.native_brrr import _scales, eligible, group_eligible
from ignition.recipes.deepseek41.native_brrr.weights import (
    DecoderAdapter, block_scales, interleave, stages,
)


def arithmetic():
    return dict(rms_norm_eps=1e-20, hc_eps=1e-6, hc_sinkhorn_iters=20,
                num_experts_per_tok=6, scoring_func="sqrtsoftplus",
                norm_topk_prob=True, routed_scaling_factor=1.5, swiglu_limit=10.)


def metadata(rows=5, *, device="cuda:0", dtype=torch.bfloat16, shape=None):
    # These tests exercise the host predicate, not CUDA execution.
    source = SimpleNamespace(shape=(rows, 4, 5120) if shape is None else shape,
                             dtype=dtype, device=torch.device(device))
    experts = SimpleNamespace(num_experts=384, has_csf_blocks=False)
    return SimpleNamespace(experts=SimpleNamespace(experts=experts)), source


@pytest.mark.parametrize("rows", [0, 1, 2, 4, 5, 6, 7, 8, 9, 16, 32, 128, 4096])
def test_selection_uses_physical_rows_not_request_count(monkeypatch, rows):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    weights, source = metadata(rows)
    assert eligible(weights, source, mode="verify", world=4, boundary=None,
                    text=arithmetic()) is (5 <= rows <= 8)


@pytest.mark.parametrize("mode", ["decode", "draft", "prefill"])
def test_other_execution_modes_do_not_query_cuda(monkeypatch, mode):
    monkeypatch.setattr(torch.cuda, "get_device_capability",
                        lambda device: pytest.fail("ineligible mode queried CUDA"))
    weights, source = metadata()
    assert not eligible(weights, source, mode=mode, world=4, boundary=None,
                        text=arithmetic())


@pytest.mark.parametrize("change", ["tp2", "tp8", "cross_host", "boundary", "cpu", "meta",
                                    "fp32", "shape", "sm103", "sm121", "experts", "csf", "native"])
def test_selection_preserves_other_supported_paths(monkeypatch, change):
    weights, source = metadata()
    options = dict(mode="verify", world=4, boundary=None, text=arithmetic())
    capability = (12, 0)
    if change in ("tp2", "tp8"):
        options["world"] = int(change[2:])
    elif change == "cross_host":
        options["spans_hosts"] = True
    elif change == "boundary":
        options["boundary"] = object()
    elif change in ("cpu", "meta"):
        source.device = torch.device(change)
    elif change == "fp32":
        source.dtype = torch.float32
    elif change == "shape":
        source.shape = (5, 4, 4096)
    elif change == "sm103":
        capability = (10, 3)
    elif change == "sm121":
        capability = (12, 1)
    elif change == "experts":
        weights.experts.experts.num_experts = 128
    elif change == "csf":
        weights.experts.experts.has_csf_blocks = True
    elif change == "native":
        options["native_dense_activations"] = True
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    assert not eligible(weights, source, **options)


@pytest.mark.parametrize("key,value", [
    ("rms_norm_eps", 1e-6), ("hc_eps", 1e-5), ("hc_sinkhorn_iters", 10),
    ("num_experts_per_tok", 8), ("scoring_func", "sigmoid"),
    ("norm_topk_prob", False), ("routed_scaling_factor", 2.5), ("swiglu_limit", 8.),
])
def test_kernel_constants_cannot_override_checkpoint_arithmetic(monkeypatch, key, value):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    weights, source = metadata()
    text = arithmetic()
    text[key] = value
    assert not eligible(weights, source, mode="verify", world=4, boundary=None, text=text)
    del text[key]
    assert not eligible(weights, source, mode="verify", world=4, boundary=None, text=text)


@pytest.mark.parametrize("rows,columns,dtype", [
    (64, 96, torch.uint8), (128, 256, torch.uint8), (64, 640, torch.uint8),
    (64, 96, torch.bfloat16),
])
def test_stage_packing_preserves_bytes_and_zero_pads(rows, columns, dtype):
    generator = torch.Generator().manual_seed(4107)
    values = torch.randint(0, 200, (rows, columns), generator=generator).to(dtype)
    original = values.clone()
    raw = values.view(torch.uint8)
    width = (raw.shape[1] + 127) // 128 * 128
    expected = torch.empty(rows * width, dtype=torch.uint8)
    r = torch.arange(rows)[:, None].expand(rows, width)
    c = torch.arange(width)[None, :].expand(rows, width)
    # Address decomposition of a 64-row, 128-byte stage, independent of reshape/permute.
    index = r // 64 * (width // 128) + c // 128
    for radix, digit in ((4, c % 128 // 32), (4, r % 64 // 16),
                         (8, r % 8), (4, c % 32 // 8), (2, r % 16 // 8), (8, c % 8)):
        index = index * radix + digit
    padded = torch.zeros((rows, width), dtype=torch.uint8)
    padded[:, :raw.shape[1]] = raw
    expected[index.flatten()] = padded.flatten()
    assert torch.equal(stages(values), expected)
    assert torch.equal(values, original)


@pytest.mark.parametrize("block", [1, 8, 32])
def test_gate_up_interleaving_retains_row_pairs(block):
    source = torch.arange(8 * block * 3).reshape(8 * block, 3)
    expected = torch.cat([source[start:start + block]
                          for group in range(4)
                          for start in (group * block, (4 + group) * block)])
    assert torch.equal(interleave(source, block), expected)


def test_dense_scale_compaction_preserves_calibration_and_rejects_unshared_rows():
    blocks = torch.arange(16, dtype=torch.uint8).reshape(2, 8)
    rows = blocks.repeat_interleave(32, dim=0)
    packed = SimpleNamespace(scale_rows=rows)
    actual = block_scales(packed, 64, 256)
    assert torch.equal(actual, blocks)
    assert torch.equal(rows, actual.repeat_interleave(32, dim=0))
    rows[19, 3] += 1
    with pytest.raises(ValueError, match="not shared across 32 rows"):
        block_scales(packed, 64, 256)


def test_query_scale_packing_preserves_each_scale_byte():
    rows, columns = 16, 128
    scales = torch.arange(rows * columns // 32, dtype=torch.uint8).reshape(rows, columns // 32)
    weight = SimpleNamespace(weight=SimpleNamespace(values=torch.empty(rows, columns), scale_rows=scales))
    packed = _scales(weight)
    for row in range(rows):
        for group in range(columns // 32):
            assert packed[row // 8, group // 2, row % 8 // 2, group % 2, row % 2] == scales[row, group]


def test_teardown_works_before_layer_registration_and_is_idempotent():
    adapter = object.__new__(DecoderAdapter)
    anchor = torch.empty(0, dtype=torch.uint8)
    calls = []
    adapter.anchor, adapter.handle = anchor, 7
    adapter.ext = SimpleNamespace(destroy_engine=lambda handle, tensor: calls.append((handle, tensor)))
    adapter.layers, adapter.owners = {}, []
    adapter.close()
    adapter.close()
    assert len(calls) == 1 and calls[0][0] == 7 and calls[0][1] is anchor
    assert adapter.handle is None and not adapter.layers and not adapter.owners


def test_failed_native_drain_retains_python_owners():
    adapter = object.__new__(DecoderAdapter)
    adapter.anchor, adapter.handle = torch.empty(0), 7
    owner = torch.ones(2)
    adapter.layers, adapter.owners = {0: 0}, [owner]
    def fail(handle, anchor):
        raise RuntimeError("native drain failed")
    adapter.ext = SimpleNamespace(destroy_engine=fail)
    with pytest.raises(RuntimeError, match="native drain failed"):
        adapter.close()
    assert adapter.handle == 7 and adapter.layers == {0: 0} and adapter.owners[0] is owner


@pytest.mark.parametrize("enabled", [False, True])
def test_target_factory_carries_fusion_selection_without_changing_drafter(enabled):
    from ignition.recipes.deepseek41.bootstrap import _forward_factories
    target, draft = _forward_factories("bf16", window_staging=None, suffix_scratch=None,
        vision=None, small_verification_fusion=enabled)
    assert target.keywords["small_verification_fusion"] is enabled
    assert "small_verification_fusion" not in draft.keywords


@pytest.mark.parametrize("capabilities", [
    [(12, 0)] * 4, [(12, 0), (12, 1), (12, 0), (12, 0)], [(12, 1)] * 4,
])
def test_rank_group_uses_one_capability_decision(monkeypatch, capabilities):
    from ignition.recipes.deepseek41 import native_brrr
    group = object()
    coordinator = SimpleNamespace(ranks=(0, 1, 2, 3), spans_hosts=False, group=group)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    calls = []
    def gather(output, value, **kwargs):
        calls.append((value, kwargs["group"]))
        output[:] = capabilities
    monkeypatch.setattr(native_brrr.dist, "all_gather_object", gather)
    assert group_eligible(coordinator, "cuda:0", enabled=True) is all(c == (12, 0) for c in capabilities)
    assert calls == [((12, 0), group)]


@pytest.mark.parametrize("enabled,ranks,hosts", [
    (False, 4, False), (True, 2, False), (True, 8, False), (True, 4, True),
])
def test_ineligible_rank_groups_do_not_allocate_or_exchange(monkeypatch, enabled, ranks, hosts):
    from ignition.recipes.deepseek41 import native_brrr
    def fail(*args, **kwargs):
        pytest.fail("ineligible group accessed CUDA or a collective")
    monkeypatch.setattr(torch.cuda, "get_device_capability", fail)
    monkeypatch.setattr(native_brrr.dist, "all_gather_object", fail)
    coordinator = SimpleNamespace(ranks=tuple(range(ranks)), spans_hosts=hosts)
    assert not group_eligible(coordinator, "cuda:0", enabled=enabled)
