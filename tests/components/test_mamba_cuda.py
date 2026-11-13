# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mamba preparation, graph replay, and state mutation against an FP32 recurrence."""

import os
import gc

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an explicitly assigned CUDA device")


def operands(lengths, *, dtype=torch.bfloat16):
    torch.manual_seed(47)
    device = "cuda:0"
    sequences, tokens, heads, groups, dim, n = len(lengths), sum(lengths), 4, 2, 64, 128
    slots = 2 * sequences + 1
    packed = torch.randn((tokens, heads * dim + 2 * groups * n), device=device, dtype=torch.bfloat16)
    x, b, c = packed.split((heads * dim, groups * n, groups * n), dim=-1)
    state = torch.randn((slots, heads, dim, n), device=device, dtype=dtype) * .1
    state[0].zero_()
    args = dict(x=x.view(tokens, heads, dim), B=b.view(tokens, groups, n), C=c.view(tokens, groups, n),
        dt=torch.randn((tokens, heads), device=device, dtype=torch.bfloat16),
        A=-torch.rand(heads, device=device), D=torch.randn(heads, device=device, dtype=torch.bfloat16),
        dt_bias=torch.full((heads,), -3., device=device, dtype=torch.bfloat16), recurrent_state=state,
        cu_seqlens=torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device=device, dtype=torch.int32),
        initial_state_indices=torch.arange(1, sequences + 1, device=device, dtype=torch.int64),
        final_state_indices=torch.arange(sequences + 1, 2 * sequences + 1, device=device, dtype=torch.int64),
        state_is_fresh=torch.zeros(sequences, device=device, dtype=torch.bool),
        num_seqs=torch.tensor([sequences], device=device, dtype=torch.int32),
        num_tokens=torch.tensor([tokens], device=device, dtype=torch.int32),
        out=torch.empty_like(x.view(tokens, heads, dim)))
    return args


def reference(args):
    a = args
    state = a["recurrent_state"].clone()
    output = torch.empty_like(a["out"])
    bounds = a["cu_seqlens"].cpu().tolist()
    for seq, (start, end) in enumerate(zip(bounds, bounds[1:])):
        s = state[a["initial_state_indices"][seq]].float().clone()
        for token in range(start, end):
            dt = torch.nn.functional.softplus(a["dt"][token].float() + a["dt_bias"].float())
            b, c = (a[key][token].float().repeat_interleave(2, dim=0) for key in ("B", "C"))
            x = a["x"][token].float()
            s = s * torch.exp(dt * a["A"])[:, None, None] + x[:, :, None] * (dt[:, None] * b)[:, None, :]
            output[token] = (s * c[:, None, :]).sum(-1) + a["D"].float()[:, None] * x
        if end > start:
            state[a["final_state_indices"][seq]] = s
    return output, state


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mode,lengths", [("decode", (1, 1, 0, 1)), ("prefill", (129, 0, 257))])
def test_prepared_mamba_matches_recurrence_and_replays(mode, lengths, dtype):
    from b12x.preparation import PreparationSession
    from ignition.components.mamba import PreparedMambaScan, PreparedMambaUpdate
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena

    args = operands(lengths, dtype=dtype)
    original = args["recurrent_state"].clone()
    expected, state = reference(args)
    with PreparationSession(device="cuda:0", autotune=False) as session:
        arena = ScratchArena("cuda:0")
        if mode == "prefill":
            operator = PreparedMambaScan(args, session=session, workspace=arena, name="scan")
        else:
            operator = PreparedMambaUpdate(args, session=session, name="update")
        arena.finalize()
        def forward():
            args["out"].copy_(operator(args))
        forward = compile_forward(forward)
        forward()
        args["recurrent_state"].copy_(original)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        gc.collect()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        for _ in range(3):
            args["recurrent_state"].copy_(original)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            torch.testing.assert_close(args["out"], expected, rtol=.025, atol=.06)
            torch.testing.assert_close(args["recurrent_state"], state, rtol=.025, atol=.06)
