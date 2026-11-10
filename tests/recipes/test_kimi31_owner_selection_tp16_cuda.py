"""Compare replicated and row-owned exact DSA selection on sixteen GPUs.

The inputs are candidate score/ID words, not model logits. Timings include
packing, transport, selection and result replication; they are component
measurements and do not establish complete-model throughput.
"""

from datetime import timedelta
import json
import os
from pathlib import Path
import statistics

import pytest
import torch

from tests.recipes.test_kimi31_selection_agreement_tp16_cuda import MergePipeline, source_rows

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
                              reason="requires sixteen explicitly assigned CUDA devices")


class OwnerPipeline:
    def __init__(self, rows, rank, owner, reduction, session, coordinator, buffers):
        from ignition.components.owner_topk import PreparedOwnerTopK
        from ignition.recipes.kimi3.dcp import ContextShards
        from ignition.recipes.kimi31.indexer import SharedBuffers

        self.shards = ContextShards.on(rank, 16, owner.device)
        storage = SharedBuffers(buffers, owner.device)
        self.local_ids = storage.buffer("local_ids", (rows, 2048), torch.int32)
        self.local_scores = storage.buffer("local_scores", (rows, 2048), torch.float32)
        self.active = storage.buffer("active", (rows,), torch.bool)
        self.output_ids = storage.buffer("output", (rows, 2048), torch.int32)
        self.planar = storage.buffer("planar", (2 * rows, 2048), torch.float32)
        self.scores = self.planar[:rows]
        self.ids = self.planar[rows:2 * rows].view(torch.int32)
        self.selection = PreparedOwnerTopK(self.ids, self.scores, self.active,
            self.shards.index, self.output_ids, owner_runtime=owner,
            reduction_runtime=reduction, session=session, coordinator=coordinator,
            name=f"owner.{rows}", buffers=SharedBuffers(buffers, owner.device, prefix="selection."))

    def __call__(self):
        live = self.local_ids >= 0
        self.ids.copy_(torch.where(live, self.local_ids * 16 + self.shards.owner, -1).to(torch.int32))
        self.scores.copy_(torch.where(live, self.local_scores, -torch.inf))
        return self.selection()


def worker(rank, rendezvous, destination, compile_components, rows_catalog=(128, 32, 8, 6, 4, 3, 1), profile=False, tune=False):
    from ignition import runtime_patches
    runtime_patches.apply()
    import torch.distributed as dist
    from b12x.comm.pcie.pcie_dcp_topk import PCIeDCPTopKOwnerExchange
    from ignition.components.collectives import PreparedAllReduce
    from ignition.components import row_ownership
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime, create_head_exchange
    from ignition.preparation.execution import GraphSignature
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=16,
                            timeout=timedelta(minutes=20))
    graphs, runtimes = [], []
    # Handles must be rebound on each rank rather than frozen from rank zero.
    for _ in range(rank + 1):
        next(row_ownership._HANDLES)
    try:
        with PackageStorage(rank=rank, world=16, device=rank) as storage:
            gather = create_head_exchange(dist.group.WORLD, device,
                max_rows=128, heads=16, head_dim=8192, ipc=storage.peer_storage)
            runtimes.append(gather)
            owner = PCIeDCPTopKOwnerExchange.from_process_group(process_group=dist.group.WORLD,
                device=device, max_rows=128, topk=2048, stream_affine=False, ipc=storage.peer_storage)
            runtimes.append(owner)
            reduction = create_all_reduce_runtime(dist.group.WORLD, device,
                max_tokens=128, hidden_size=8192, ipc=storage.peer_storage)
            runtimes.append(reduction)
            coordinator = CollectivePreparation(dist.group.WORLD)
            with RecordedPreparationSession(device=device, checkpoint="candidate-word-fixture",
                    workload_schema="owner-row-selection-tp16", numerical_policy="exact-score-bits-and-ordered-ids",
                    coordinator=coordinator, compile_workers=24) as session:
                pools, cases = {"replicated": {}, "owner": {}}, {}
                for rows in rows_catalog:
                    cases[rows, "replicated"] = MergePipeline(rows, rank, gather, session, coordinator,
                        pools["replicated"], ordered=True)
                    cases[rows, "owner"] = OwnerPipeline(rows, rank, owner, reduction, session, coordinator,
                        pools["owner"])
                sync_input = torch.zeros((1, 64), dtype=torch.bfloat16, device=device)
                sync_output = torch.empty_like(sync_input)
                synchronize = PreparedAllReduce(sync_input, sync_output, runtime=reduction,
                    session=session, coordinator=coordinator, name="timing.synchronize")
                session.tune(show_progress=False)
                dist.barrier()
                synchronize(sync_input)
                torch.cuda.synchronize()
                if rank == 0:
                    print("tuning completed", flush=True)

                def inputs(operation, rows, seed, count=None):
                    scores, ids, active = source_rows(rows, seed, count=count)
                    operation.local_scores.copy_(scores[:, rank * 2048:(rank + 1) * 2048])
                    operation.local_ids.copy_(ids[:, rank * 2048:(rank + 1) * 2048])
                    operation.active.copy_(active)

                # All shapes share row-capacity storage, exactly as serial recipe calls do.
                for (rows, _), operation in cases.items():
                    inputs(operation, rows, 0)
                functions = {f"{kind}{rows}": operation for (rows, kind), operation in cases.items()}
                signatures = {name: GraphSignature("decode", 1, cases[key].active.numel(), ())
                              for key, name in zip(cases, functions, strict=True)}
                stream = torch.cuda.Stream(device=device)
                captured, compiled = capture_forwards(functions, signatures, session=session, device=device,
                    stream=stream, pool=torch.cuda.graph_pool_handle(), coordinator=coordinator,
                    compile_decoder=compile_components,
                    before_run=lambda name: inputs(functions[name], functions[name].active.numel(), 0),
                    cache_root=str(Path(destination).parent / "compiled"), report_catalog=False, retain_forwards=True)
                graphs.extend(captured.values())
                calls = {key: compiled.get(f"{key[1]}{key[0]}", operation) for key, operation in cases.items()}
                if rank == 0:
                    print("capture completed", flush=True)
                checks = 0
                for seed in (0, 1, 2):
                    for rows in reversed(rows_catalog):
                        observed = {}
                        for kind in ("replicated", "owner"):
                            operation = cases[rows, kind]
                            inputs(operation, rows, seed)
                            operation.output_ids.fill_(-91)
                            torch.cuda.synchronize()
                            dist.barrier()
                            before = torch.cuda.memory_stats()["allocation.all.allocated"]
                            captured[f"{kind}{rows}"].replay()
                            torch.cuda.synchronize()
                            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                            observed[kind] = operation.output_ids.cpu()
                        assert torch.equal(observed["replicated"], observed["owner"]), (rows, seed, rank)
                        control = cases[rows, "replicated"]
                        selected = cases[rows, "owner"].selection
                        first = rank * selected.owner_rows
                        last = min(rows, first + selected.owner_rows)
                        owned = max(0, last - first)
                        if owned:
                            expected = control.gathered[first:last]
                            assert torch.equal(selected.transport.scores[:owned].view(torch.int32),
                                               expected[:, :, 0].contiguous().view(torch.int32))
                            assert torch.equal(selected.transport.indices[:owned], expected.view(torch.int32)[:, :, 1])
                            expected = expected.clone()
                            inactive = ~control.active[first:last]
                            expected[inactive, :, 0] = -torch.inf
                            expected.view(torch.int32)[inactive, :, 1] = -1
                            assert torch.equal(selected.packed[:owned].view(torch.int32), expected.view(torch.int32))
                            assert torch.equal(selected.selected_scores[:owned].view(torch.int32),
                                               control.output_values[first:last].view(torch.int32))
                        if owned < selected.owner_rows:
                            assert torch.all(selected.selected_ids[owned:] == -1)
                            assert torch.all(selected.selected_scores[owned:].view(torch.int32) == -8388608)
                        peers = [None] * 16
                        dist.all_gather_object(peers, observed["owner"])
                        assert all(torch.equal(peers[0], value) for value in peers), (rows, seed)
                        checks += 1
                if rank == 0:
                    print(f"exact selection: {checks} cases on every rank", flush=True)

                # Queue graph variants without host synchronization. Slow one owner before
                # the sequence and poison reused outputs to expose channel lifetime defects.
                ordered_rows = sorted(rows_catalog)
                sequence = tuple(ordered_rows[::2] + ordered_rows[1::2][::-1] + ordered_rows[:1])
                patterns, snapshots, controls = [], [], []
                for index, rows in enumerate(sequence):
                    scores, ids, active = source_rows(rows, index + 70, count=32768)
                    patterns.append((scores[:, rank * 2048:(rank + 1) * 2048].to(device),
                        ids[:, rank * 2048:(rank + 1) * 2048].to(device), active.to(device)))
                    operation = cases[rows, "replicated"]
                    for target, source in zip((operation.local_scores, operation.local_ids, operation.active), patterns[-1], strict=True):
                        target.copy_(source)
                    dist.barrier()
                    captured[f"replicated{rows}"].replay()
                    torch.cuda.synchronize()
                    controls.append(operation.output_ids.cpu())
                    snapshots.append(torch.empty_like(operation.output_ids))
                torch.cuda.synchronize()
                dist.barrier()
                if rank == 0:
                    torch.cuda._sleep(2000000)
                for index, rows in enumerate(sequence):
                    operation = cases[rows, "owner"]
                    for target, source in zip((operation.local_scores, operation.local_ids, operation.active), patterns[index], strict=True):
                        target.copy_(source)
                    operation.output_ids.fill_(-91)
                    captured[f"owner{rows}"].replay()
                    snapshots[index].copy_(operation.output_ids)
                torch.cuda.synchronize()
                assert all(torch.equal(snapshot.cpu(), control) for snapshot, control in zip(snapshots, controls, strict=True))

                choices = []
                if tune:
                    from b12x.preparation import PreparedCall
                    from ignition.preparation.composition_tuning import PreparedComposition, race_compositions
                    for rows in sorted(rows_catalog):
                        patterns, references = [], []
                        control = cases[rows, "replicated"]
                        for seed, count in ((101, 256), (102, 8192), (103, 32768), (207, 4096)):
                            scores, ids, active = source_rows(rows, seed, count=count)
                            pattern = (scores[:, rank*2048:(rank+1)*2048].to(device),
                                       ids[:, rank*2048:(rank+1)*2048].to(device), active.to(device))
                            patterns.append(pattern)
                            for target, source in zip((control.local_scores, control.local_ids, control.active), pattern, strict=True):
                                target.copy_(source)
                            torch.cuda.synchronize()
                            dist.barrier()
                            calls[rows, "replicated"]()
                            torch.cuda.synchronize()
                            references.append(control.output_ids.clone())
                        candidates, producers = {}, {}
                        for kind in ("replicated", "owner"):
                            operation, forward = cases[rows, kind], calls[rows, kind]
                            state = [0]
                            def produce(index, operation=operation, state=state):
                                state[0] = index
                                for target, source in zip((operation.local_scores, operation.local_ids, operation.active), patterns[index], strict=True):
                                    target.copy_(source)
                            from functools import partial
                            producers[kind] = tuple(partial(produce, index) for index in range(4))
                            def validate(operation=operation, state=state):
                                assert torch.equal(operation.output_ids, references[state[0]])
                            if kind == "replicated":
                                plans = (("transport", operation.gather.plan), ("selection", operation.fold.plan))
                            else:
                                plans = (("transport", operation.selection.transport.plan),
                                         ("selection", operation.selection.fold.plan),
                                         ("replication", operation.selection.broadcast.reduction.plan))
                            candidates[kind] = PreparedComposition(forward, plans, PreparedCall(
                                run=forward, output=operation.output_ids, produce=producers[kind][0],
                                benchmark_producers=producers[kind][:3], validate=validate,
                                reset=partial(operation.output_ids.fill_, -91), capture_safe=True,
                                benchmark_repetitions=8, owners=(operation, patterns, references)))
                        for attempt in range(2):
                            chosen = race_compositions(candidates, operation="candidate-selection",
                                geometry=dict(rows=rows, world_size=16, local_topk=2048, output_topk=2048),
                                numerics="exact-fp32-words-ordered-ties",
                                workload=dict(version=1, seeds=(101,102,103), counts=(256,8192,32768)),
                                default="replicated", session=session, coordinator=coordinator, cache_policy="warm")
                            if attempt:
                                assert chosen["source"] == "cached", chosen
                            kind = chosen["implementation"]
                            producers[kind][3]()
                            torch.cuda.synchronize()
                            dist.barrier()
                            candidates[kind].forward()
                            torch.cuda.synchronize()
                            candidates[kind].call.validate()
                            choices.append(dict(rows=rows, attempt=attempt, **chosen))
                            if rank == 0:
                                print(json.dumps(choices[-1]), flush=True)

                timings = []
                if profile:
                    torch.cuda.synchronize()
                    dist.barrier()
                    torch.cuda.cudart().cudaProfilerStart()
                for rows in sorted(rows_catalog):
                    for count in (256, 8192, 32768):
                        saved, events, expected_outputs = {}, {}, {}
                        samples = {kind: [] for kind in pools}
                        for kind in pools:
                            operation = cases[rows, kind]
                            inputs(operation, rows, 100, count=count)
                            torch.cuda.synchronize()
                            dist.barrier()
                            captured[f"{kind}{rows}"].replay()
                            torch.cuda.synchronize()
                            expected_outputs[kind] = operation.output_ids.cpu()
                            pairs = [(torch.cuda.Event(enable_timing=True, external=True),
                                      torch.cuda.Event(enable_timing=True, external=True)) for _ in range(8)]
                            graph = torch.cuda.CUDAGraph()
                            torch.cuda.synchronize()
                            dist.barrier()
                            with session.capture(), torch.cuda.graph(graph):
                                synchronize(sync_input)
                                for begin, end in pairs:
                                    begin.record()
                                    calls[rows, kind]()
                                    end.record()
                            graphs.append(graph)
                            saved[kind], events[kind] = graph, pairs
                        assert torch.equal(expected_outputs["replicated"], expected_outputs["owner"]), (rows, count, rank)
                        for iteration in range(6):
                            for kind in (("replicated", "owner", "owner", "replicated") if iteration % 2 == 0
                                         else ("owner", "replicated", "replicated", "owner")):
                                dist.barrier()
                                torch.cuda.nvtx.range_push(f"selection/{kind}/rows{rows}/count{count}/iteration{iteration}")
                                saved[kind].replay()
                                torch.cuda.synchronize()
                                torch.cuda.nvtx.range_pop()
                                peers = [None] * 16
                                dist.all_gather_object(peers, [begin.elapsed_time(end) for begin, end in events[kind]])
                                if iteration:
                                    samples[kind].extend(max(values) for values in zip(*peers, strict=True))
                        timings.append(dict(rows=rows, valid_candidates=count, samples_ms=samples,
                            median_ms={kind: statistics.median(values) for kind, values in samples.items()}))
                        if rank == 0:
                            print(json.dumps({key: value for key, value in timings[-1].items() if key != "samples_ms"}), flush=True)
                        for graph in saved.values():
                            graph.reset()
                            graphs.remove(graph)
                if profile:
                    torch.cuda.synchronize()
                    dist.barrier()
                    torch.cuda.cudart().cudaProfilerStop()
                if rank == 0:
                    Path(destination).write_text(json.dumps(dict(compiled=compile_components,
                        exact_selection_cases=checks, queued_variants_passed=True,
                        composition_choices=choices, timings=timings), indent=2) + "\n")
                for graph in graphs:
                    graph.reset()
                graphs.clear()
            for runtime in reversed(runtimes):
                runtime.close()
            runtimes.clear()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(1)
    finally:
        for graph in graphs:
            graph.reset()
        for runtime in reversed(runtimes):
            runtime.close()
        dist.destroy_process_group()


def test_owner_selection_tp16(tmp_path):
    import torch.multiprocessing as mp
    assert torch.cuda.device_count() == 16
    mp.spawn(worker, args=((tmp_path / "gloo").as_uri(), str(tmp_path / "owner-selection.json"), True),
             nprocs=16, join=True)


if __name__ == "__main__":
    import argparse
    from importlib import import_module
    import tempfile
    import torch.multiprocessing as mp
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--rows", type=int, nargs="+", default=[128, 32, 8, 6, 4, 3, 1])
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    if args.config is not None:
        from ignition.cli.config import Configuration
        from ignition.cli.environment import apply
        from ignition.cli.preparation import Host
        from ignition.recipes.kimi31.recipe import RECIPE
        apply(Configuration.load(args.config), Host.discover().gpus[:16], RECIPE.b12x_environment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Compiler workers deserialize forwards by importable class identity.
    # A module executed with -m otherwise leaves these classes in __mp_main__.
    imported = import_module("tests.recipes.test_kimi31_owner_selection_tp16_cuda")
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(imported.worker, args=((Path(directory) / "gloo").as_uri(), str(args.output), args.compiled,
                              tuple(sorted(set(args.rows), reverse=True)), args.profile, args.tune),
                 nprocs=16, join=True)
