# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiled TP mixers preserve injection, final mixing, rank equality and graph storage."""

from dataclasses import replace
from datetime import timedelta
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_TP_CUDA_TESTS") != "1",
                               reason="requires explicitly assigned GPUs")


def _rank(rank, rendezvous):
    from b12x.norm.hyperconnection import reference
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime
    from ignition.preparation.aot import aot_forward
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.qwen38.hyperconnection import (
        HyperconnectionScratch, PreparedHyperconnection, hyperconnection_extents,
    )

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(minutes=10))
    runtime = create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=128, hidden_size=10240)
    try:
        generator = torch.Generator().manual_seed(8719)
        originals, modules = [], []
        for injection in (True, False):
            tensors = {"hc_norm.weight": torch.randn(10240, generator=generator) * .05,
                       "input_mix_weight_down.weight": torch.randn(320, 10240, generator=generator).bfloat16() * .01,
                       "input_mix_weight_up.weight": torch.randn(10240, 320, generator=generator).bfloat16() * .1}
            if injection:
                tensors["block_inject_weight.weight"] = torch.randn(4, 10240, generator=generator).bfloat16() * .01
            extents = hyperconnection_extents(TensorParallelShard(rank, 2), True)
            module = torch.nn.Module()
            module.hc_count, module.hidden_size = 4, 2560
            module.block_inject_weight = None
            for name, value in tensors.items():
                child = torch.nn.Module()
                child.weight = torch.nn.Parameter(value[extents.get(name, slice(None))].contiguous().to(device),
                                                   requires_grad=False)
                child.eps = 1e-6
                setattr(module, name.split('.')[0], child)
            originals.append({key: value.to(device) for key, value in tensors.items()})
            modules.append(module)

        def mix(state, weights):
            normalized = reference.grouped_rmsnorm(state, weights["hc_norm.weight"], streams=4, eps=1e-6)
            down = torch.nn.functional.linear(normalized, weights["input_mix_weight_down.weight"])
            bottleneck = reference.scaled_silu(down, streams=4)
            partials = [torch.nn.functional.linear(bottleneck[:, start:start + 160],
                        weights["input_mix_weight_up.weight"][:, start:start + 160]) for start in (0, 160)]
            logits = (partials[0].float() + partials[1].float()).bfloat16()
            return reference.gate_mean(normalized, logits, streams=4), normalized

        coordinator = CollectivePreparation(dist.group.WORLD)
        with torch.inference_mode(), RecordedPreparationSession(device=device, checkpoint="qwen-hc-tp",
                workload_schema="sharded-combine-final", numerical_policy="bf16-transport",
                coordinator=coordinator, autotune_effort="off", compile_workers=2) as session:
            source = torch.zeros((4096, 10240), device=device, dtype=torch.bfloat16)
            scratch = HyperconnectionScratch.allocate(4096, device, lowrank=160)
            hidden = torch.zeros((4096, 2560), device=device, dtype=torch.bfloat16)
            calls = []
            for rows in (1, 6, 128, 4096):
                state = source[:rows]
                local = scratch.rows(rows)
                kwargs = dict(priming_state=state, session=session, runtime=runtime,
                              coordinator=coordinator)
                first = PreparedHyperconnection(modules[0], local, name=f"rows{rows}.inject", **kwargs)
                final = PreparedHyperconnection(modules[1], replace(local, block_input=hidden[:rows]),
                                                name=f"rows{rows}.final", **kwargs)
                first.prepare_combine_norm(final, priming_state=state, session=session, name=f"rows{rows}.combine_norm")

                first.functional(session=session)
                final.functional(session=session)

                def run(first=first, final=final, state=state):
                    block, injection = first.mix_runtime(state)
                    residual, normalized = first.combine_norm_runtime(state, block, injection)
                    final.mix_runtime(residual, normalized)

                calls.append((rows, run))
            session.tune(show_progress=False)
            graphs = []
            stream = torch.cuda.Stream(device=device)
            for rows, run in calls:
                compiled = aot_forward(run)
                compiled(run)
                graph = torch.cuda.CUDAGraph()
                stream.wait_stream(torch.cuda.current_stream())
                with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graph, stream=stream):
                    compiled(run)
                torch.cuda.current_stream().wait_stream(stream)
                graphs.append((rows, graph, compiled))
            session.freeze()
            for iteration in range(3):
                for rows, graph, _ in (graphs if iteration % 2 else reversed(graphs)):
                    value = (torch.randn((rows, 10240), generator=generator) * .25).bfloat16().to(device)
                    block, normalized = mix(value, originals[0])
                    injection = torch.nn.functional.linear(normalized, originals[0]["block_inject_weight.weight"])
                    residual = reference.combine(value, block, injection, streams=4)
                    expected, _ = mix(residual, originals[1])
                    source[:rows].copy_(value)
                    for field in scratch.__dataclass_fields__:
                        tensor = getattr(scratch, field)
                        if tensor is not None:
                            tensor.fill_(float("nan"))
                    hidden.fill_(float("nan"))
                    torch.cuda.synchronize()
                    allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
                    torch.testing.assert_close(source[:rows], residual, rtol=.02, atol=.02)
                    torch.testing.assert_close(hidden[:rows], expected, rtol=.02, atol=.02)
                    actual = hidden[:rows].cpu()
                    peer = actual.clone()
                    dist.broadcast(peer, src=0)
                    assert torch.equal(actual, peer), f"rank outputs differ at {rows} rows"
            for _, graph, _ in graphs:
                graph.reset()
    finally:
        runtime.close()
        dist.destroy_process_group()


def test_compiled_sharded_hyperconnections(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two assigned GPUs")
    mp.spawn(_rank, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
