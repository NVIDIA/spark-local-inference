# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint Mamba blocks against independent weight decoding and FP32 recurrence."""

import gc
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("tokens,deferred", [
    (1, False), (129, False), (4096, False),
    pytest.param(1, True, id="decode-deferred"),
])
@torch.no_grad()
def test_checkpoint_mamba_block(tokens, deferred):
    from b12x.preparation import PreparationSession
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.recurrent import DeltaMetadata
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.nemotron_h.mamba import PreparedMamba
    from ignition.recipes.nemotron_h.weights import NemotronWeightLoader
    from ignition.reference.qwen38_quant import CheckpointTensors, decode_nvfp4, read_nvfp4_projection

    path = Path(os.environ.get("FLARE_NEMOTRON_CHECKPOINT",
        "/data/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"))
    if not path.is_dir():
        pytest.skip("set FLARE_NEMOTRON_CHECKPOINT to the Bolt checkpoint")
    manifest = inspect_indexed_checkpoint(path)
    prefix = "language_model.backbone.layers.0.mixer"
    with NemotronWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(0, 1),
                              prefixes=("language_model.backbone.layers.0.",)) as loader:
        weights, config = loader.load_layer(0), loader.config
    references = []
    with CheckpointTensors(manifest, byte_budget=128 << 20) as reader:
        for suffix in ("in_proj", "out_proj"):
            p = read_nvfp4_projection(reader, prefix + "." + suffix)
            decoded = decode_nvfp4(p.packed.cuda(), p.scales.cuda(), torch.ones_like(p.weight_scale_2).cuda())
            references.append((decoded.bfloat16().float(), p.weight_scale_2.cuda().reshape(())))
    torch.manual_seed(15)
    source = torch.randn((tokens, config.hidden), device="cuda", dtype=torch.bfloat16) * .2
    c = config.text
    state = SimpleNamespace(
        convolution=torch.randn((3, config.convolution_width, 3), device="cuda", dtype=torch.bfloat16) * .1,
        recurrent=torch.randn((3, c["mamba_num_heads"], c["mamba_head_dim"], c["ssm_state_size"]),
                              device="cuda", dtype=torch.bfloat16) * .1)
    state.convolution[0].zero_()
    state.recurrent[0].zero_()
    initial_conv, initial_ssm = state.convolution.clone(), state.recurrent.clone()
    m = DeltaMetadata.allocate(1, source.device)
    m.query_start_loc[1] = tokens
    m.convolution_slots.fill_(2)
    m.initial_state_indices.fill_(1)
    m.final_state_indices.fill_(2)
    m.num_seqs.fill_(1)
    m.num_tokens.fill_(tokens)
    if deferred:
        from ignition.preparation.tuning import RecordedPreparationSession

        session = RecordedPreparationSession(device="cuda:0", checkpoint=str(path),
            workload_schema="nemotron-mamba", numerical_policy="default", autotune_effort="off",
            compile_workers=1)
    else:
        session = PreparationSession(device="cuda:0", autotune=False)
    with session:
        arena = ScratchArena("cuda:0")
        if deferred:
            session.register_scratch(arena)
        block = PreparedMamba(weights, source, state, m, config,
            mode="decode" if tokens == 1 else "prefill", session=session, name="block",
            workspace=arena)
        block.functional(session=session)
        if deferred:
            session.tune()
            arena.release_preparation()
        else:
            arena.finalize()
        forward = compile_forward(lambda: block(source, m, state))
        forward()
        session.freeze()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            actual = forward()
        state.convolution.copy_(initial_conv)
        state.recurrent.copy_(initial_ssm)
        gc.collect()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated

        projected = (source.float() @ references[0][0].T * references[0][1]).bfloat16()
        gate, xbc, dt = projected.split((config.mamba_width, config.convolution_width, c["mamba_num_heads"]), -1)
        history = torch.cat((initial_conv[2], xbc.T), dim=1)
        conv = torch.nn.functional.conv1d(history.float()[None], weights["convolution"].float()[:, None],
            weights["convolution_bias"].float(), groups=config.convolution_width)[0].T
        mixed = torch.nn.functional.silu(conv).bfloat16()
        x, B, C = mixed.split((config.mamba_width, c["n_groups"] * c["ssm_state_size"],
                              c["n_groups"] * c["ssm_state_size"]), -1)
        x = x.view(tokens, c["mamba_num_heads"], c["mamba_head_dim"]).float()
        groups = torch.arange(c["mamba_num_heads"], device="cuda") // (c["mamba_num_heads"] // c["n_groups"])
        B, C = (v.view(tokens, c["n_groups"], c["ssm_state_size"]).float()[:, groups] for v in (B, C))
        dt = torch.nn.functional.softplus(dt.float() + weights["dt_bias"].float())
        recurrent = initial_ssm[1].float()
        y = torch.empty_like(x, dtype=torch.bfloat16)
        for t in range(tokens):
            recurrent = (recurrent * (dt[t] * weights["A"]).exp()[:, None, None]
                + x[t, :, :, None] * (dt[t, :, None] * B[t])[:, None, :])
            y[t] = (recurrent * C[t, :, None, :]).sum(-1) + weights["D"].float()[:, None] * x[t]
        y = y.view(tokens, -1).float() * torch.nn.functional.silu(gate.float())
        y = y.view(tokens, c["n_groups"], -1)
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + c["layer_norm_epsilon"])
        y = (y.view(tokens, -1) * weights["output_norm"].float()).bfloat16()
        expected = (y.float() @ references[1][0].T * references[1][1]).bfloat16()
        error = (actual.float() - expected.float()).norm() / expected.float().norm()
        assert error < .015, error.item()
        assert (actual.float() - expected.float()).abs().max() < .05 * expected.float().abs().max()


@pytest.mark.parametrize("autotune", [False, True])
@pytest.mark.parametrize("capacity", [5, 8])
@torch.no_grad()
def test_checkpoint_verification_commits_convolution_and_recurrence(autotune, capacity):
    from dataclasses import replace
    import triton
    from b12x.preparation import PreparationSession
    from ignition.components.mamba_speculation import PreparedMambaReplay
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.recurrent import DeltaMetadata, _commit_convolution
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.nemotron_h.draft_config import DraftConfig
    from ignition.recipes.nemotron_h.mamba import PreparedMamba
    from ignition.recipes.nemotron_h.pool import ContinuationPool, REPLAY_RING
    from ignition.recipes.nemotron_h.weights import NemotronWeightLoader

    path = Path(os.environ.get("FLARE_NEMOTRON_CHECKPOINT", "/data/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"))
    draft_path = Path(os.environ.get("FLARE_LILICORR_CHECKPOINT", str(path) + "-LiLiCorr"))
    if not path.is_dir() or not draft_path.is_dir():
        pytest.skip("requires Nemotron Bolt and LiLiCorr checkpoint metadata")
    with NemotronWeightLoader(inspect_indexed_checkpoint(path), "cuda:0", shard=TensorParallelShard(0, 1),
            prefixes=("language_model.backbone.layers.0.",)) as loader:
        weights, config = loader.load_layer(0), loader.config
    config = replace(config, draft=DraftConfig.load(draft_path, config))
    pool = ContinuationPool.for_model(config, 3, 2, "cuda:0")
    state = pool.recurrent(0)
    state.convolution[1:].normal_(std=.1)
    state.recurrent[1:].normal_(std=.1)
    initial = pool.backing.clone()
    source = torch.randn((capacity, config.hidden), device="cuda", dtype=torch.bfloat16) * .2
    rows = torch.ones(1, device="cuda", dtype=torch.int64)
    lengths = torch.ones(1, device="cuda", dtype=torch.int32)
    counts = torch.ones_like(lengths)
    region = pool.layout.region("layer0.convolution")
    offsets = torch.tensor([region.offset // 2], dtype=torch.int64, device="cuda")
    metadata = [DeltaMetadata.allocate(1, "cuda:0") for _ in range(2)]
    for m, tokens in zip(metadata, (1, capacity), strict=True):
        m.query_start_loc[1] = tokens
        m.initial_state_indices.fill_(1)
        m.final_state_indices.fill_(1)
        m.convolution_slots.fill_(1)
        m.num_seqs.fill_(1)
        m.num_tokens.fill_(tokens)
    with PreparationSession(device="cuda:0", autotune=autotune) as session:
        arena = ScratchArena("cuda:0")
        replay = PreparedMambaReplay(pool.backing, pool.layout,
            [tuple(pool.layout.region("layer0." + field) for field in ("recurrent", "replay_x", "replay_b", "replay_dt"))],
            weights["A"][None],
            ring_region=pool.layout.region("mamba_ring_start"), count_region=pool.layout.region("mamba_pending_count"),
            ring_length=REPLAY_RING)
        decode = PreparedMamba(weights, source[:1], state, metadata[0], config,
            mode="decode", session=session, name="decode", workspace=arena)
        verify = PreparedMamba(weights, source, state, metadata[1], config,
            mode="verify", session=session, name="verify", workspace=arena,
            replay=replay, records=pool.replay_records(0), verification_width=8)
        decode.functional(session=session)
        verify.functional(session=session)
        arena.finalize()
        expected_outputs, expected_states = [], []
        pool.backing.copy_(initial)
        for i in range(capacity):
            expected_outputs.append(decode(source[i:i + 1], metadata[0], state).clone())
            expected_states.append((state.convolution[1, :, :3].clone(), state.recurrent[1].clone()))

        def forward():
            replay_state = replay.prepare(rows, lengths, width=8)
            output = verify(source, metadata[1], state, records=pool.replay_records(0), replay_state=replay_state)
            replay.commit(counts, rows, lengths, *replay_state, width=8)
            torch.library.wrap_triton(_commit_convolution)[(1, 1, triton.cdiv(config.convolution_width, 128))](
                pool.backing.view(torch.bfloat16), offsets, rows, counts, pool.layout.row_bytes // 2,
                config.convolution_width, region.shape[1], 128)
            return output

        forward = compile_forward(forward)
        forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = forward()
        flush = compile_forward(lambda: replay.prepare(rows, lengths, width=1))
        flush()
        session.freeze()
        for length, accepted in ((8, 1), (8, 3), (8, 8), (5, 2), (5, 5)):
            if length > capacity:
                continue
            pool.backing.copy_(initial)
            counts.fill_(accepted)
            lengths.fill_(length)
            metadata[1].query_start_loc[1] = length
            metadata[1].num_tokens.fill_(length)
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            flush()
            expected = torch.cat(expected_outputs[:length])
            assert (actual[:length].float() - expected.float()).norm() / expected.float().norm() < .015
            convolution, recurrent = expected_states[accepted - 1]
            torch.testing.assert_close(state.convolution[1, :, :3], convolution, rtol=0, atol=0)
            assert (state.recurrent[1].float() - recurrent.float()).norm() / recurrent.float().norm() < .015
            assert torch.equal(pool.backing[0], initial[0])
            assert torch.equal(pool.backing[2], initial[2])
