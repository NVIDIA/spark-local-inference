# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accepted Mamba replay across verification widths, inactive rows, and snapshots."""

import gc
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("autotune", [False, True])
@pytest.mark.parametrize("prime", [True, False])
@pytest.mark.parametrize("heads,groups", [(16, 4), (128, 8)])
@torch.no_grad()
def test_replay_width_transitions_and_snapshot_branch(autotune, prime, heads, groups):
    from b12x.preparation import PreparationSession
    from ignition.components.mamba import PreparedMambaUpdate
    from ignition.components.mamba_speculation import PreparedMambaReplay, PreparedMambaVerification
    from ignition.components.state import StateLayout, StateStorage
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena

    torch.manual_seed(901)
    dim, n, ring, layers, batch = 64, 128, 16, 2, 3
    specs = []
    for layer in range(layers):
        specs.extend(((f"{layer}.state", (heads, dim, n), torch.bfloat16),
            (f"{layer}.x", (heads, ring, dim), torch.bfloat16),
            (f"{layer}.b", (groups, ring, n), torch.bfloat16),
            (f"{layer}.dt", (heads, ring), torch.float32)))
    specs.extend((("ring", (1,), torch.int32), ("pending", (1,), torch.int32)))
    layout = StateLayout.from_specs(specs)
    pool = StateStorage.allocate(layout, 6, "cuda:0")
    regions = [tuple(layout.region(f"{i}.{field}") for field in ("state", "x", "b", "dt")) for i in range(layers)]
    states = [pool.region(f"{i}.state") for i in range(layers)]
    for state in states:
        state[1:].normal_(std=.1)
    oracle = [s.float().clone() for s in states]
    A = -torch.rand((layers, heads), device="cuda") * 2 - .2
    D = torch.randn((layers, heads), device="cuda", dtype=torch.bfloat16)
    bias = torch.full((layers, heads), -3., device="cuda", dtype=torch.bfloat16)
    rows = torch.tensor([2, 1, 3], dtype=torch.int64, device="cuda")
    lengths, accepted = (torch.zeros(batch, dtype=torch.int32, device="cuda") for _ in range(2))
    workspace = ScratchArena("cuda:0")
    graphs, operands, replayers = {}, {}, {}
    with PreparationSession(device="cuda:0", autotune=autotune, prime=prime) as session:
        for width in range(8, 0, -1):
            replay = PreparedMambaReplay(pool.backing, layout, regions, A,
                ring_region=layout.region("ring"), count_region=layout.region("pending"), ring_length=ring)
            ring_start = torch.zeros(pool.backing.shape[0], dtype=torch.int32, device="cuda")
            pending_count = torch.zeros_like(ring_start)
            records, updates, arguments = [], [], []
            cu = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
            for layer in range(layers):
                tokens = batch * width
                args = dict(x=torch.randn((tokens, heads, dim), device="cuda", dtype=torch.bfloat16) * .2,
                    dt=torch.randn((tokens, heads), device="cuda", dtype=torch.bfloat16) * .1,
                    B=torch.randn((tokens, groups, n), device="cuda", dtype=torch.bfloat16) * .2,
                    C=torch.randn((tokens, groups, n), device="cuda", dtype=torch.bfloat16) * .2,
                    A=A[layer], D=D[layer], dt_bias=bias[layer], recurrent_state=states[layer], cu_seqlens=cu,
                    initial_state_indices=rows, final_state_indices=rows,
                    state_is_fresh=torch.zeros(batch, dtype=torch.bool, device="cuda"),
                    num_seqs=torch.tensor([batch], dtype=torch.int32, device="cuda"),
                    num_tokens=torch.zeros(1, dtype=torch.int32, device="cuda"),
                    out=torch.zeros((tokens, heads, dim), device="cuda", dtype=torch.bfloat16))
                record = tuple(pool.region(f"{layer}.{field}") for field in ("x", "b", "dt"))
                update = (PreparedMambaUpdate(args, session=session, name=f"decode{layer}") if width == 1 else
                    PreparedMambaVerification(args, record, ring_start, pending_count, rows,
                        width=width, session=session, workspace=workspace, name=f"verify{width}.{layer}"))
                updates.append(update)
                arguments.append(args)
                records.append(record)
            workspace.finalize()

            def forward(replay=replay, updates=tuple(updates), arguments=tuple(arguments),
                        records=tuple(records), width=width):
                ring_start, pending_count = replay.prepare(rows, lengths, width=width)
                for update, args, record in zip(updates, arguments, records, strict=True):
                    output = (update(args) if width == 1 else
                              update(args, record, ring_start, pending_count, rows))
                    args["out"].copy_(output)
                if width > 1:
                    replay.commit(accepted, rows, lengths, ring_start, pending_count, width=width)

            forward = compile_forward(forward)
            forward()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                forward()
            graphs[width], operands[width], replayers[width] = (graph, forward, tuple(updates)), arguments, replay
        session.freeze()
        groups_for_head = torch.arange(heads, device="cuda") // (heads // groups)
        rounds = [(2, (2, 1, 0), (2, 1, 0))] * 5
        rounds += [(8, (8, 4, 0), (3, 2, 0)), (1, (1, 0, 1), (1, 0, 1)),
                   (4, (4, 4, 1), (1, 3, 1)), (8, (8, 2, 0), (7, 1, 0)), (1, (1, 1, 0), (1, 1, 0))]
        rounds += [(3, (3, 1, 0), (2, 1, 0)), (7, (7, 4, 0), (6, 1, 0)),
                   (5, (5, 0, 3), (1, 0, 2)), (6, (6, 2, 0), (5, 2, 0)), (1, (1, 1, 1), (1, 1, 1))]
        for iteration, (width, sizes, commits) in enumerate(rounds):
            if iteration == 8:
                pool.backing[4].copy_(pool.backing[2])
                for value in oracle:
                    value[4].copy_(value[2])
                rows[0] = 4
            row_values = rows.cpu().tolist()
            lengths.copy_(torch.tensor(sizes, device="cuda"))
            accepted.copy_(torch.tensor(commits, device="cuda"))
            ends = [0]
            for length in sizes:
                ends.append(ends[-1] + length)
            args_list = operands[width]
            args_list[0]["cu_seqlens"].copy_(torch.tensor(ends, device="cuda"))
            for args in args_list:
                args["x"].normal_(std=.2)
                args["num_tokens"].fill_(sum(sizes))
            before = pool.backing.clone()
            gc.collect()
            torch.cuda.synchronize()
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            graphs[width][0].replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            changed = {slot for slot, length in zip(row_values, sizes, strict=True) if length}
            for slot in set(range(pool.backing.shape[0])) - changed:
                assert torch.equal(pool.backing[slot], before[slot])
            for layer, args in enumerate(args_list):
                expected = []
                for row, (length, count, slot) in enumerate(zip(sizes, commits, row_values, strict=True)):
                    value = oracle[layer][slot].clone()
                    for token in range(length):
                        i = ends[row] + token
                        dt = torch.nn.functional.softplus(args["dt"][i].float() + bias[layer].float())
                        x = args["x"][i].float()
                        B, C = (args[key][i, groups_for_head].float() for key in ("B", "C"))
                        value = value * (dt * A[layer]).exp()[:, None, None] + x[:, :, None] * (dt[:, None] * B)[:, None]
                        expected.append((value * C[:, None]).sum(-1) + D[layer].float()[:, None] * x)
                        if token + 1 == count:
                            oracle[layer][slot].copy_(value)
                    if width == 1 and length:
                        oracle[layer][slot].copy_(oracle[layer][slot].bfloat16().float())
                target = torch.stack(expected).bfloat16().float()
                actual = args["out"][:sum(sizes)].float()
                assert (actual - target).norm() / target.norm() < .012
                torch.testing.assert_close(actual, target, rtol=.04, atol=.04)
        lengths.fill_(1)
        flush = compile_forward(lambda: replayers[1].prepare(rows, lengths, width=1))
        flush()
        for layer, state in enumerate(states):
            for slot in rows.cpu().tolist():
                expected = oracle[layer][slot]
                assert (state[slot].float() - expected).norm() / expected.norm() < .012
        assert not pool.region("pending")[rows].any()
