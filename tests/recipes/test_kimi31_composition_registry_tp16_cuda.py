"""Qualify preparation-selected candidate merges after private trials close."""

from datetime import timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
                              reason="requires sixteen explicitly assigned CUDA devices")


def worker(rank, rendezvous, output):
    from ignition import runtime_patches
    runtime_patches.apply()
    import torch.distributed as dist
    from b12x.comm.pcie.pcie_dcp_topk import PCIeDCPTopKOwnerExchange
    from ignition.components.candidate_selection import PreparedCandidateSelection, CandidateSelectionForward
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.collectives import CollectivePreparation, create_head_exchange, create_all_reduce_runtime
    from ignition.preparation.execution import GraphSignature
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.dcp import ContextShards
    from ignition.recipes.kimi31.indexer import SharedBuffers
    from ignition.recipes.kimi31.selection_tuning import CandidateSelectionTuning, _candidate_pattern
    from tests.recipes.test_kimi31_selection_agreement_tp16_cuda import MergePipeline

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=16,
                            timeout=timedelta(minutes=20))
    graphs, channels = {}, []
    rows_catalog = (8, 7, 6, 5, 4, 3, 2, 1)
    try:
        with PackageStorage(rank=rank, world=16, device=rank) as storage:
            coordinator = CollectivePreparation(dist.group.WORLD)
            gather = create_head_exchange(dist.group.WORLD, device, max_rows=128,
                heads=16, head_dim=8192, ipc=storage.peer_storage)
            channels.append(gather)
            owner = PCIeDCPTopKOwnerExchange.from_process_group(process_group=dist.group.WORLD,
                device=device, max_rows=8, topk=2048, stream_affine=False, ipc=storage.peer_storage)
            channels.append(owner)
            reduction = create_all_reduce_runtime(dist.group.WORLD, device,
                max_tokens=1024, hidden_size=7168, ipc=storage.peer_storage)
            channels.append(reduction)
            shards = ContextShards.on(rank, 16, device)
            context = SimpleNamespace(shards=shards, candidates=gather, candidate_owners=owner,
                                      coordinator=coordinator)
            registry = CandidateSelectionTuning(context, reduction)
            with RecordedPreparationSession(device=device, checkpoint="candidate-registry-fixture",
                    workload_schema="tp16-candidate-registry", numerical_policy="ordered-fp32",
                    coordinator=coordinator, compile_workers=24) as session:
                cases, controls = {}, {}
                model_buffers, reference_buffers = {}, {}
                for rows in rows_catalog:
                    buffers = SharedBuffers(model_buffers, device)
                    ids = buffers.buffer("input", (rows, 2048), torch.int32)
                    scores = buffers.buffer("scores", (rows, 2048), torch.float32)
                    positions = buffers.buffer("positions", (rows,), torch.int64)
                    selected = buffers.buffer("output", (rows, 2048), torch.int32)
                    operation = PreparedCandidateSelection(ids, scores, positions, shards.index, selected,
                        world=16, gather_runtime=gather, owner_runtime=owner, reduction_runtime=reduction,
                        session=session, coordinator=coordinator, name=f"model.rows{rows}",
                        buffers=SharedBuffers(model_buffers, device, prefix="merge."))
                    cases[rows] = operation
                    registry.register(operation)
                    controls[rows] = MergePipeline(rows, rank, gather, session, coordinator, reference_buffers, ordered=True)
                session.tune(show_progress=False)
                report = registry.tune(session)
                if rank == 0:
                    print(json.dumps(report), flush=True)
                for rows, operation in cases.items():
                    ids, scores, positions = _candidate_pattern(rows, 16, 2048, rank, 100, 4096)
                    operation.indices.copy_(ids)
                    operation.scores.copy_(scores)
                    operation.positions.copy_(positions)
                    controls[rows].local_ids.copy_(ids)
                    controls[rows].local_scores.copy_(scores)
                    controls[rows].active.copy_(positions >= 0)
                forwards = {f"selected{rows}": CandidateSelectionForward(operation, None)
                            for rows, operation in cases.items()}
                forwards.update({f"reference{rows}": operation for rows, operation in controls.items()})
                signatures = {name: GraphSignature("decode", 1, rows) for rows in cases
                              for name in (f"selected{rows}", f"reference{rows}")}
                graphs, compiled = capture_forwards(forwards, signatures, session=session, device=device,
                    stream=torch.cuda.Stream(device=device), pool=torch.cuda.graph_pool_handle(),
                    coordinator=coordinator, report_catalog=False)
                checks = 0
                for seed, count, partial in ((9001, 0, False), (9002, 256, True),
                                              (9003, 8192, False), (9004, 32768, True)):
                    for rows in reversed(rows_catalog):
                        ids, scores, positions = _candidate_pattern(rows, 16, 2048, rank, seed, count, partial_rows=partial)
                        operation, reference = cases[rows], controls[rows]
                        operation.indices.copy_(ids)
                        operation.scores.copy_(scores)
                        operation.positions.copy_(positions)
                        reference.local_ids.copy_(ids)
                        reference.local_scores.copy_(scores)
                        reference.active.copy_(positions >= 0)
                        operation.output.fill_(-91)
                        torch.cuda.synchronize()
                        dist.barrier()
                        before = torch.cuda.memory_stats()["allocation.all.allocated"]
                        graphs[f"reference{rows}"].replay()
                        if rank == 7:
                            torch.cuda._sleep(1000000)
                        graphs[f"selected{rows}"].replay()
                        torch.cuda.synchronize()
                        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                        assert torch.equal(operation.output, reference.output_ids), (rank, rows, seed)
                        checks += 1
                if rank == 0:
                    Path(output).write_text(json.dumps(dict(choices=report, exact_cases=checks,
                        selected_calls_after_trial_cleanup=True), indent=2)+"\n")
                for graph in graphs.values():
                    graph.reset()
                graphs.clear()
            for channel in reversed(channels):
                channel.close()
            channels.clear()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(1)
    finally:
        for graph in graphs.values():
            graph.reset()
        for channel in reversed(channels):
            channel.close()
        dist.destroy_process_group()


def test_candidate_registry_tp16(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(worker, args=((tmp_path/"gloo").as_uri(), str(tmp_path/"registry.json")), nprocs=16, join=True)


if __name__ == "__main__":
    import argparse
    from importlib import import_module
    import tempfile
    import torch.multiprocessing as mp
    from ignition.cli.config import Configuration
    from ignition.cli.environment import apply
    from ignition.cli.preparation import Host
    from ignition.recipes.kimi31.recipe import RECIPE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    apply(Configuration.load(args.config), Host.discover().gpus[:16], RECIPE.b12x_environment)
    module = import_module("tests.recipes.test_kimi31_composition_registry_tp16_cuda")
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(module.worker, args=((Path(directory)/"gloo").as_uri(), str(args.output)), nprocs=16, join=True)
