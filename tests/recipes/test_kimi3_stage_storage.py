# SPDX-License-Identifier: Apache-2.0

"""Serial stage cursors share scratch without overwriting persistent decoder storage."""

from types import SimpleNamespace

import torch

from ignition.components.parallel import TensorParallelShard
from ignition.preparation.scratch import GraphBufferPool
from ignition.recipes.kimi3.decoder import DecoderStorage


def test_stage_cursors_overlay_serial_but_not_simultaneous_values():
    pool = GraphBufferPool("cpu", chunk_bytes=4096)
    cursor = pool.cursor()
    persistent = cursor.full((64,), 7, torch.bfloat16)
    config = SimpleNamespace(hidden=64, text={"routed_expert_hidden_size": 64})
    storage = DecoderStorage.allocate(
        config, TensorParallelShard(0, 8), "cpu", allocator=cursor
    )
    kda = storage.stage_allocator("kda")
    attention = kda.zeros((128,), torch.bfloat16)
    simultaneous = kda.zeros((128,), torch.bfloat16)
    moe = storage.stage_allocator("moe")
    experts = moe.zeros((512,), torch.bfloat16)
    assert attention.data_ptr() == experts.data_ptr()
    assert simultaneous.data_ptr() >= attention.data_ptr() + attention.nbytes
    experts.fill_(3)
    assert torch.equal(persistent, torch.full_like(persistent, 7))
    assert storage.stage_allocator("kda") is kda
    cursor.join(*storage.stage_allocators.values())
    after = cursor.zeros((64,), torch.bfloat16)
    assert after.data_ptr() >= experts.data_ptr() + experts.nbytes


def test_unpooled_decoder_keeps_explicit_allocations():
    config = SimpleNamespace(hidden=64, text={"routed_expert_hidden_size": 64})
    storage = DecoderStorage.allocate(config, TensorParallelShard(0, 8), "cpu")
    assert storage.stage_allocator("kda") is None
