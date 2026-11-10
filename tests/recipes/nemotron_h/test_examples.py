# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preparation declarations share matching roles and keep writable planes disjoint."""

from types import SimpleNamespace

import torch

from ignition.recipes.nemotron_h.decoder import Scratch
from ignition.recipes.nemotron_h.vision import VisionScratch


def test_target_and_draft_declarations_have_separate_roles():
    config, examples = SimpleNamespace(hidden=64, vocabulary=128), {}
    target = Scratch.allocate(8, 8, config, "cpu", examples=examples)
    repeated = Scratch.allocate(8, 8, config, "cpu", examples=examples)
    draft = Scratch.allocate(8, 8, config, "cpu", examples=examples, role="nemotron.draft")
    for name in target.__dataclass_fields__:
        assert getattr(target, name) is getattr(repeated, name)
        assert getattr(target, name).untyped_storage()._cdata != getattr(draft, name).untyped_storage()._cdata
    assert len({tensor.untyped_storage()._cdata for tensor in vars(target).values()}) == len(vars(target))
    target.activations.fill_(3)
    assert torch.all(repeated.activations == 3)
    assert torch.count_nonzero(target.normalized) == 0
    assert torch.count_nonzero(draft.activations) == 0


def test_vision_declarations_reuse_only_the_same_plane():
    examples = {}
    first = VisionScratch.allocate(4, "cpu", examples=examples)
    second = VisionScratch.allocate(4, "cpu", examples=examples)
    for name in first.__dataclass_fields__:
        assert getattr(first, name) is getattr(second, name)
    assert len({tensor.untyped_storage()._cdata for tensor in vars(first).values()}) == len(vars(first))
    first.query.fill_(7)
    assert torch.all(second.query == 7)
    assert torch.count_nonzero(first.key) == 0
    assert torch.count_nonzero(first.value) == 0
