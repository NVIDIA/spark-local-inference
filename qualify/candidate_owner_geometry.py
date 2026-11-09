"""Measure existing candidate-owner launch geometries with exact queued replay."""

import argparse
from datetime import timedelta
import json
from pathlib import Path
import statistics
import tempfile


def worker(rank, rendezvous, output):
    import torch
    import torch.distributed as dist
    from b12x.comm.pcie.pcie_dcp_topk import PCIeDCPTopKOwnerExchange
    from b12x.comm.pcie import _owner_preparation as preparation
    from ignition.preparation.collectives import CollectivePreparation
    from ignition.preparation.core_storage import CoreStorage
    from ignition.preparation.tuning import RecordedPreparationSession, declare

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=16,
                            timeout=timedelta(minutes=10))
    graphs, runtime = [], None
    try:
        with CoreStorage(rank=rank, world=16, device=rank) as storage:
            runtime = PCIeDCPTopKOwnerExchange.from_process_group(process_group=dist.group.WORLD,
                device=device, max_rows=128, topk=2048, stream_affine=False, ipc=storage.peer_storage)
            runtime.graph_candidate_views(128)
            coordinator = CollectivePreparation(dist.group.WORLD)
            with RecordedPreparationSession(device=device, checkpoint="rank-major-candidate-words",
                    workload_schema="owner-launch-geometries", numerical_policy="exact-int32",
                    coordinator=coordinator, compile_workers=24) as session:
                cases, captured = {}, {}
                for rows in (1, 8, 17, 128):
                    indices = torch.arange(rows * 2048, dtype=torch.int32, device=device).reshape(rows, 2048) + rank * 1000000
                    scores = (indices ^ 0x3f800000).view(torch.float32)
                    for threads in (128, 256, 512):
                        call = dict(local_indices=indices, local_scores=scores, threads=threads, block_limit=128)
                        plan = preparation.plan(preparation.query_from_runtime(runtime, call=call), runtime=runtime)
                        declare(session, (plan.request(name=f"rows{rows}.threads{threads}",
                            prepare_call=lambda state, call=call: preparation.prepared_call(state, **call),
                            collective=preparation.collective(runtime, key=f"rows{rows}.threads{threads}")),), coordinator=coordinator)
                        cases[rows, threads] = indices, scores, plan
                session.tune(show_progress=False)

                def invoke(key):
                    indices, scores, plan = cases[key]
                    with runtime.capture(plan=plan):
                        return runtime.stage_candidates(indices, scores, plan=plan)

                for key in cases:
                    dist.barrier()
                    invoke(key)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with session.capture(), torch.cuda.graph(graph):
                        invoke(key)
                    graphs.append(graph)
                    captured[key] = graph

                def expected(rows):
                    owner_rows = -(-rows // 16)
                    row = rank * owner_rows + torch.arange(owner_rows)[:, None]
                    column = torch.arange(2048).repeat(16)[None]
                    peer = torch.arange(16).repeat_interleave(2048)[None]
                    ids = (row * 2048 + column + peer * 1000000).int()
                    bits = ids ^ 0x3f800000
                    return torch.where(row < rows, ids, -1), torch.where(row < rows, bits, -8388608)

                sequence = ((8, 128), (1, 512), (17, 256), (128, 128), (8, 512), (1, 128), (17, 512), (128, 256))
                snapshots = [(torch.empty_like(value), torch.empty_like(score))
                             for value, score in (runtime.graph_candidate_views(rows) for rows, _ in sequence)]
                dist.barrier()
                for index, key in enumerate(sequence):
                    captured[key].replay()
                    if rank == 0:
                        torch.cuda._sleep(1000000)
                    ids, scores = runtime.graph_candidate_views(key[0])
                    snapshots[index][0].copy_(ids)
                    snapshots[index][1].copy_(scores)
                torch.cuda.synchronize()
                for (rows, _), (ids, scores) in zip(sequence, snapshots, strict=True):
                    expected_ids, expected_bits = expected(rows)
                    assert torch.equal(ids.cpu(), expected_ids)
                    assert torch.equal(scores.view(torch.int32).cpu(), expected_bits)
                if rank == 0:
                    print("queued geometry transitions passed", flush=True)

                timings = []
                for rows in (1, 8, 17, 128):
                    saved, events = {}, {}
                    samples = {threads: [] for threads in (128, 256, 512)}
                    for threads in samples:
                        pairs = [(torch.cuda.Event(enable_timing=True, external=True),
                                  torch.cuda.Event(enable_timing=True, external=True)) for _ in range(16)]
                        graph = torch.cuda.CUDAGraph()
                        dist.barrier()
                        with session.capture(), torch.cuda.graph(graph):
                            invoke((rows, threads))
                            for begin, end in pairs:
                                begin.record()
                                invoke((rows, threads))
                                end.record()
                        graphs.append(graph)
                        saved[threads], events[threads] = graph, pairs
                    for iteration in range(6):
                        for threads in ((128, 256, 512, 512, 256, 128) if iteration % 2 == 0
                                        else (512, 256, 128, 128, 256, 512)):
                            dist.barrier()
                            before = torch.cuda.memory_stats()["allocation.all.allocated"]
                            saved[threads].replay()
                            torch.cuda.synchronize()
                            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                            observed_ids, observed_scores = runtime.graph_candidate_views(rows)
                            expected_ids, expected_bits = expected(rows)
                            assert torch.equal(observed_ids.cpu(), expected_ids)
                            assert torch.equal(observed_scores.view(torch.int32).cpu(), expected_bits)
                            peers = [None] * 16
                            dist.all_gather_object(peers, [begin.elapsed_time(end) for begin, end in events[threads]])
                            if iteration:
                                samples[threads].extend(max(values) for values in zip(*peers, strict=True))
                    timings.append(dict(rows=rows, median_us={threads: statistics.median(values) * 1000
                        for threads, values in samples.items()}, samples_ms=samples))
                    if rank == 0:
                        print(json.dumps({key: value for key, value in timings[-1].items() if key != "samples_ms"}), flush=True)
                    for graph in saved.values():
                        graph.reset()
                        graphs.remove(graph)
                if rank == 0:
                    Path(output).write_text(json.dumps(dict(queued_geometry_passed=True, timings=timings), indent=2) + "\n")
                for graph in graphs:
                    graph.reset()
                graphs.clear()
            runtime.close()
            runtime = None
    except BaseException:
        import os
        import traceback
        traceback.print_exc()
        os._exit(1)
    finally:
        for graph in graphs:
            graph.reset()
        if runtime is not None:
            runtime.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from ignition.cli.config import Configuration
    from ignition.cli.environment import apply
    from ignition.cli.preparation import Host
    from ignition.recipes.kimi31.recipe import RECIPE
    apply(Configuration.load(args.config), Host.discover().gpus[:16], RECIPE.b12x_environment)
    import torch.multiprocessing as mp
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(worker, args=((Path(directory) / "gloo").as_uri(), str(args.output)), nprocs=16, join=True)
