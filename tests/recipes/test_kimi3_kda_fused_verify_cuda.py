# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic one-launch KDA verification equals the three-launch path bitwise."""

import os

import pytest
import torch

cuda = pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned CUDA device")

HEADS, WIDTH, HIDDEN, WINDOW = 12, 12 * 128, 1024, 8


def _weights(device, generator):
    def randn(*shape, scale=1.0, dtype=torch.bfloat16):
        return (torch.randn(shape, generator=generator) * scale).to(dtype).to(device)

    columns = (4 * WIDTH + 128 + HEADS + 7) // 8 * 8
    return {
        "input": randn(columns, HIDDEN, scale=HIDDEN ** -0.5),
        "forget": randn(WIDTH, 128, scale=128 ** -0.5),
        "output": randn(HIDDEN, WIDTH, scale=WIDTH ** -0.5),
        "convolution": randn(3 * WIDTH, 4, scale=0.3),
        "A_log": randn(HEADS, scale=0.5, dtype=torch.float32),
        "dt_bias": randn(WIDTH, scale=0.5, dtype=torch.float32),
        "norm": 1.0 + randn(128, scale=0.1, dtype=torch.float32),
    }


@cuda
@torch.no_grad()
@pytest.mark.parametrize("tokens", [1, 4, 8])
def test_fused_verify_matches_three_launches(tokens):
    from ignition.components.kda_speculation import KDARecords
    from ignition.components.recurrent import DeltaMetadata, DeltaState
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.kda import PreparedKDA

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    generator = torch.Generator().manual_seed(1110 + tokens)
    weights = _weights(device, generator)
    source = (torch.randn((tokens, HIDDEN), generator=generator) * 0.5).to(torch.bfloat16).to(device)
    sources = [(torch.randn((tokens, HIDDEN), generator=generator) * 0.5).to(torch.bfloat16).to(device)
               for _ in range(3)]
    metadata = DeltaMetadata.allocate(1, device, verification_columns=WINDOW)
    layers = {}
    with RecordedPreparationSession(device=device, checkpoint="dynamic-kda-fused-verify",
            workload_schema="kimi-kda-fused-verify", numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        for name in ("off", "on"):
            state = DeltaState.allocate(4, device, heads=HEADS, channels=3 * WIDTH, speculative_tokens=WINDOW - 1)
            records = KDARecords.allocate({0: weights}, 4, HEADS, device, window=WINDOW)
            operation = PreparedKDA(weights, source, state, metadata, mode="verify", eps=1e-5,
                session=session, name="verify_" + name, workspace=arena, buffers={}, records=records.layer(0))
            layers[name] = operation, state, records
        session.tune(show_progress=False)
        assert layers["on"][0].fused is not None
        layers["off"][0].fused = None
        metadata.query_start_loc[1] = tokens
        metadata.convolution_slots.fill_(1)
        metadata.state_indices[:, 0].fill_(2)
        metadata.accepted_tokens.fill_(3)
        metadata.fresh.fill_(False)
        metadata.num_seqs.fill_(1)
        metadata.num_tokens.fill_(tokens)
        conv0 = torch.randn(layers["on"][1].convolution.shape, generator=generator).to(torch.bfloat16).to(device)
        rec0 = (torch.randn(layers["on"][1].recurrent.shape, generator=generator) * 0.1).to(device)

        results = {}
        for name, (operation, state, records) in layers.items():
            compiled = compile_forward(lambda operation=operation, records=records:
                operation(source, metadata, records=records.layer(0)))

            def reset(index):
                source.copy_(sources[index])
                state.convolution.copy_(conv0)
                state.recurrent.copy_(rec0)
                records.corrections.fill_(float("nan"))
                records.keys_and_gates.fill_(float("nan"))
                arena.tensor.fill_(0xFF)

            reset(0)
            compiled()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = compiled()
            outputs = []
            for index in range(3):
                reset(index)
                graph.replay()
                torch.cuda.synchronize()
                outputs.append((output.clone(), state.convolution.clone(), state.recurrent.clone(),
                                records.corrections.clone(), records.keys_and_gates.clone()))
            results[name] = outputs
            graph.reset()
        assert all(torch.isfinite(run[0]).all() for run in results["off"])
        for actual_run, expected_run in zip(results["off"], results["on"], strict=True):
            for actual, expected in zip(actual_run, expected_run, strict=True):
                assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
