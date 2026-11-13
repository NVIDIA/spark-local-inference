# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The parity reset records shared scratch storage once, including padding."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from qualify import graph_execution
from ignition.preparation.scratch import ScratchArena
from ignition.recipes.qwen38.vision_state import VisionFeatureState


@dataclass
class RequestFixture:
    epoch: torch.Tensor


@pytest.mark.parametrize("registered_view", [False, True])
def test_mutable_reset_retains_complete_arena_once(registered_view):
    arena = ScratchArena("cpu")
    arena.reserve(513)
    arena.tensor.copy_(torch.arange(arena.nbytes, dtype=torch.int64).to(torch.uint8))
    arena.freeze()
    features = VisionFeatureState(torch.empty(4, 2, 3), torch.zeros(4, dtype=torch.int64),
                                  torch.zeros(4, dtype=torch.int32), torch.full((1,), -1, dtype=torch.int32))
    buffers = {"scratch_alias": arena.tensor[512:513]} if registered_view else {}
    model = SimpleNamespace(_resources=[features, arena], buffers=buffers)
    state = SimpleNamespace(backing=torch.empty((2, 16), dtype=torch.uint8))
    request = RequestFixture(torch.zeros(4, dtype=torch.int64))
    reset = graph_execution.MutableReset(model, state, request, [], [], {})
    key = arena.tensor.untyped_storage().data_ptr()
    name, raw, saved = reset.storage[key]
    assert name == ("input.scratch_alias" if registered_view else "workspace.arena")
    assert raw.storage_offset() == 0
    assert raw.shape == saved.shape == (768,)
    assert torch.equal(raw, arena.tensor) and torch.equal(saved, arena.tensor)
    assert sum(storage.untyped_storage().data_ptr() == key
               for _, storage, _ in reset.storage.values()) == 1
    assert state.backing.untyped_storage().data_ptr() not in reset.storage
    assert features.arena.untyped_storage().data_ptr() not in reset.storage


def test_mutable_reset_retains_single_sampler_storage_for_decoder_and_resume():
    features = VisionFeatureState(torch.empty(4, 2, 3), torch.zeros(4, dtype=torch.int64),
                                  torch.zeros(4, dtype=torch.int32), torch.full((1,), -1, dtype=torch.int32))
    model = SimpleNamespace(_resources=[features], buffers={})
    state = SimpleNamespace(backing=torch.empty((2, 16), dtype=torch.uint8))
    request = RequestFixture(torch.zeros(4, dtype=torch.int64))
    logits = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    scratch = SimpleNamespace(scratch_tensors={'probabilities': logits})
    tokens = SimpleNamespace(**{name: torch.zeros(4, dtype=torch.int64)
                               for name in ('pending_tokens', 'positions', 'ple_history', 'active')})
    decoder = SimpleNamespace(state=SimpleNamespace(gdn={}, qsa={}, token_state=tokens),
                              sampler=scratch, embedding_injection=None, layers=[])
    resume = SimpleNamespace(sampler=scratch)
    before = torch.cuda.is_initialized()
    reset = graph_execution.MutableReset(model, state, request, [decoder], [resume], {})
    key = logits.untyped_storage().data_ptr()
    name, raw, pristine = reset.storage[key]
    assert name == 'decoder.0.sampler.probabilities'
    assert torch.equal(raw.view(torch.float32).reshape_as(logits), logits)
    assert pristine.view(torch.float32).reshape(3, 4).tolist() == [[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]]
    assert torch.cuda.is_initialized() == before


def test_idle_native_reset_preserves_tuning_spans_and_restores_shared_siblings(monkeypatch):
    features = VisionFeatureState(torch.empty(4, 2, 3), torch.zeros(4, dtype=torch.int64),
                                  torch.zeros(4, dtype=torch.int32), torch.full((1,), -1, dtype=torch.int32))
    shared = torch.arange(12, dtype=torch.int32)
    original = shared.clone()
    executor = SimpleNamespace(outstanding=0)
    model = SimpleNamespace(_resources=[features], consumed=True, runtime=executor, device=0,
        buffers={'mutable': shared[:4], 'selector': shared[4:5], 'status': shared[8:9]},
        tuning_graphs={'tune': SimpleNamespace(outputs=('selector',), status_buffer='status')})
    state = SimpleNamespace(backing=torch.empty((2, 16), dtype=torch.uint8))
    request = RequestFixture(torch.zeros(4, dtype=torch.int64))
    # CUDA-to-host snapshots own storage; emulate that copy for these CPU operands.
    with monkeypatch.context() as snapshot:
        snapshot.setattr(torch.Tensor, 'cpu', lambda tensor: tensor.clone())
        reset = graph_execution.MutableReset(model, state, request, [], [], {})
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: None)
    shared[4], shared[8] = 12345, 0
    reset.retain_startup_bindings(executor)
    shared.fill_(-7)
    request.epoch.fill_(99)
    reset.reset(executor=executor)
    expected = original.clone()
    expected[4], expected[8] = 12345, 0
    torch.testing.assert_close(shared, expected, rtol=0, atol=0)
    assert not request.epoch.count_nonzero()
    assert state.backing[0].eq(0).all()
    assert state.backing[1].eq(0x3D).all()
    with pytest.raises(RuntimeError, match='forbidden after native ownership'):
        reset.reset()
    with pytest.raises(RuntimeError, match='idle native executor'):
        reset.reset(executor=SimpleNamespace(outstanding=0))
    executor.outstanding = 1
    with pytest.raises(RuntimeError, match='idle native executor'):
        reset.reset(executor=executor)
    with pytest.raises(RuntimeError, match='idle native executor'):
        reset.retain_startup_bindings(executor)
