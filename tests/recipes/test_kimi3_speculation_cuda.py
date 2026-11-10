# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kimi KDA verification commits only the accepted recurrent and convolution prefix."""

import os

import pytest
import torch

from .test_kimi3 import CHECKPOINT as TARGET

cuda = pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned CUDA device")


@cuda
@torch.no_grad()
def test_checkpoint_kda_verification_commits_only_accepted_prefix():
    from ignition.components.kda_speculation import KDARecords, PreparedKDAReplay
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.recurrent import DeltaMetadata
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.config import ModelConfig
    from ignition.recipes.kimi3.decoder import _commit_convolution
    from ignition.recipes.kimi3.kda import PreparedKDA
    from ignition.recipes.kimi3.pool import ContinuationPool
    from ignition.recipes.kimi3.weights import KimiWeightLoader

    if not TARGET.is_dir():
        pytest.skip("Kimi K3 checkpoint is unavailable")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    c = ModelConfig.load(TARGET)
    pool = ContinuationPool.for_model(c, 16, 3, 1, device, speculative_tokens=7)
    with KimiWeightLoader(inspect_indexed_checkpoint(TARGET), device, shard=TensorParallelShard(0, 16),
            prefixes=("language_model.model.layers.0.self_attn.",)) as loader:
        w = loader.load_attention(0)
    state = pool.recurrent(0)
    records = KDARecords.allocate({0: w}, 3, 6, device)
    verify_meta, decode_meta = DeltaMetadata.allocate(1, device, verification_columns=8), DeltaMetadata.allocate(1, device)
    source = torch.randn((8, 7168), dtype=torch.bfloat16, device=device) * .2
    counts = torch.zeros(1, dtype=torch.int32, device=device)
    region = pool.layout.region("layer0.convolution")
    offsets = torch.tensor([region.offset // 2], dtype=torch.int64, device=device)
    with RecordedPreparationSession(device=device, checkpoint="kimi-kda-rollback",
            workload_schema="kimi-kda-rollback", numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        decode = PreparedKDA(w, source[:1], state, decode_meta, mode="decode", eps=1e-5,
            session=session, name="decode", workspace=arena)
        verify = PreparedKDA(w, source, state, verify_meta, mode="verify", eps=1e-5,
            session=session, name="verify", workspace=arena, records=records.layer(0))
        replay = PreparedKDAReplay(pool.backing, [state.recurrent], records, verify.plan,
            session=session).functional()
        session.tune(show_progress=False)
        arena.release_preparation()
        for meta, tokens in ((verify_meta, 8), (decode_meta, 1)):
            meta.query_start_loc[1] = tokens
            meta.convolution_slots.fill_(1)
            meta.state_indices[:, 0].fill_(1)
            meta.num_seqs.fill_(1)
            meta.num_tokens.fill_(tokens)
        def verify_commit():
            local_records = KDARecords(records.layers, torch.zeros_like(records.corrections),
                torch.zeros_like(records.keys_and_gates), records.A_log, records.dt_bias)
            verify(source, verify_meta, records=local_records.layer(0))
            replay(counts, local_records, verify_meta)
            torch.library.wrap_triton(_commit_convolution)[(1, 1, 18)](
                pool.backing.view(torch.bfloat16), offsets, verify_meta.state_indices[:, 0], counts,
                pool.layout.row_bytes // 2, 2304, 10, 128)
        compiled = compile_forward(verify_commit)
        counts.fill_(1)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        pool.backing.zero_()
        for i in range(3):
            decode(source[i:i+1], decode_meta)
        initial = pool.backing.clone()
        initial_recurrent = state.recurrent.clone()
        expected, outputs = [], []
        for i in range(8):
            outputs.append(decode(source[i:i+1], decode_meta).clone())
            expected.append((state.convolution[1, :, :3].clone(), state.recurrent[1].clone()))
        for count in range(1, 9):
            pool.backing.copy_(initial)
            actual = verify(source, verify_meta, records=records.layer(0)).clone()
            torch.testing.assert_close(state.recurrent, initial_recurrent, rtol=0, atol=0)
            counts.fill_(count)
            pool.backing.copy_(initial)
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            torch.testing.assert_close(state.convolution[1, :, :3], expected[count-1][0], rtol=.02, atol=.02)
            torch.testing.assert_close(state.recurrent[1], expected[count-1][1], rtol=.02, atol=.002)
        torch.testing.assert_close(actual, torch.cat(outputs), rtol=.04, atol=.02)
