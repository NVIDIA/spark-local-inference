# SPDX-License-Identifier: Apache-2.0

"""Kimi declaration operands share examples while rank constants retain their own values."""

from types import SimpleNamespace

import torch

from ignition.components.parallel import TensorParallelShard
from ignition.recipes.kimi3.decoder import DecoderStorage, Scratch


def test_decoder_rank_constants_follow_hidden_and_latent_partitions():
    config = SimpleNamespace(hidden=64, text={"routed_expert_hidden_size": 128})
    storage = DecoderStorage.allocate(config, TensorParallelShard(3, 8), "cpu")
    torch.testing.assert_close(storage.hidden_columns, torch.arange(24, 32))
    torch.testing.assert_close(storage.latent_begin, torch.tensor([48], dtype=torch.int32))


def test_declaration_shapes_reuse_examples_without_aliasing_live_roles():
    config = SimpleNamespace(hidden=64, vocabulary=256)
    examples = {}
    first = Scratch.allocate(8, 8, config, 8, "cpu", examples=examples)
    second = Scratch.allocate(8, 8, config, 8, "cpu", examples=examples)
    assert first.activations is second.activations
    assert first.hidden is second.hidden
    first.activations.fill_(3)
    assert torch.count_nonzero(first.hidden) == 0
    assert first.local_logits.shape == (8, 32)
    assert first.logits.shape == (8, 256)
