# SPDX-License-Identifier: Apache-2.0

"""Compare separate and aliased Kimi sparse-prefill histories on TP16.

Both paths load one checkpoint attention layer and execute its complete
4096-token prefill. The control copies latent history before index selection;
the production path consumes index history before reusing its backing for
latent records. Cache bytes and selected sets must match exactly, except for
equivalent members tied at the scorer's exact top-k cutoff. Each path's
attention must satisfy the existing probability-encoding oracle in its emitted
index order; a replay of the control's compiled attention/output nodes checks
exact output parity when supplied that same order. These checks qualify the
storage transformation, not full-model response quality.
"""

import argparse
from datetime import timedelta
import json
from pathlib import Path
import tempfile


ROWS, WORLD, RECORD_BYTES = 4096, 16, 576


class SeparateHistoryForward:
    """Prefill schedule with independent index and latent history buffers."""

    def __init__(self, attention, source):
        self.attention, self.source = attention, source

    def __call__(self):
        from ignition.recipes.kimi31.dcp import selection_slots, write_records

        op = self.attention
        op.project(self.source)
        op.slots.copy_(op.shards.token_slots(op.metadata.positions, op.metadata.token_page_table))
        write_records(op.latent, op.slots, op.cache.view(-1, RECORD_BYTES))
        op.indexer.project(self.source)
        op.history_end.copy_((op.metadata.starts + op.metadata.lengths) * (op.metadata.lengths > 0))
        op.history()
        op.indexer.sharded()
        for start, end in op.indexer.bounds:
            part = op.parts[end - start]
            selection_slots(op.indexer.sharded.selected[start:end], op.metadata.positions[start:end],
                op.metadata.token_page_table[start:end], part["owners"], part["slots"], part["counts"],
                op.shards, op.cache.shape[0], local=False, global_records=True)
            part["gather"]()
            part["attention"](op.absorbed[start:end], op.attended[start:end])
        return op.finish()


class AliasedHistoryForward:
    def __init__(self, attention, source):
        self.attention, self.source = attention, source

    def __call__(self):
        return self.attention(self.source)


def _graph_nodes(graph):
    from cuda.bindings import driver

    raw = driver.CUgraph(graph.raw_cuda_graph())
    status, _, count = driver.cuGraphGetNodes(raw)
    assert status == driver.CUresult.CUDA_SUCCESS
    status, nodes, _ = driver.cuGraphGetNodes(raw, count)
    assert status == driver.CUresult.CUDA_SUCCESS
    kernels = {}
    for node in nodes:
        status, kind = driver.cuGraphNodeGetType(node)
        assert status == driver.CUresult.CUDA_SUCCESS
        if kind == driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            status, params = driver.cuGraphKernelNodeGetParams(node)
            assert status == driver.CUresult.CUDA_SUCCESS
            status, symbol = driver.cuFuncGetName(params.func)
            assert status == driver.CUresult.CUDA_SUCCESS
            kernels[int(node)] = dict(function=int(params.func),
                symbol=symbol.decode() if isinstance(symbol, bytes) else str(symbol))
    status, _, _, _, count = driver.cuGraphGetEdges(raw)
    assert status == driver.CUresult.CUDA_SUCCESS
    status, parents, children, _, _ = driver.cuGraphGetEdges(raw, count)
    assert status == driver.CUresult.CUDA_SUCCESS
    return kernels, [(int(a), int(b)) for a, b in zip(parents, children, strict=True)]


class RecordedAttentionTail:
    """Replay the original compiled attention/output nodes with recorded selection inputs."""

    def __init__(self, graph, candidate_functions, *, expected_roots=1):
        from cuda.bindings import driver

        kernels, edges = _graph_nodes(graph)
        roots = [node for node, value in kernels.items() if value['symbol'] == '_selection_slots']
        assert len(roots) == expected_roots, 'recorded selection boundaries do not match the prepared slices'
        successors = {}
        for parent, child in edges:
            successors.setdefault(parent, []).append(child)
        retained, pending = set(), roots.copy()
        while pending:
            node = pending.pop()
            if node not in retained:
                retained.add(node)
                pending.extend(successors.get(node, ()))
        retained_kernels = [value for node, value in kernels.items() if node in retained]
        assert not any(value['function'] in candidate_functions for value in retained_kernels)
        assert not any('DSAFusedIndexerKernel' in value['symbol'] for value in retained_kernels)
        self.kernel_symbols = [value['symbol'] for value in retained_kernels]
        raw = driver.CUgraph(graph.raw_cuda_graph())
        status, self.graph = driver.cuGraphClone(raw)
        assert status == driver.CUresult.CUDA_SUCCESS
        status, _, count = driver.cuGraphGetNodes(raw)
        assert status == driver.CUresult.CUDA_SUCCESS
        status, nodes, _ = driver.cuGraphGetNodes(raw, count)
        assert status == driver.CUresult.CUDA_SUCCESS
        for node in nodes:
            if int(node) not in retained:
                status, clone = driver.cuGraphNodeFindInClone(node, self.graph)
                assert status == driver.CUresult.CUDA_SUCCESS
                status, = driver.cuGraphDestroyNode(clone)
                assert status == driver.CUresult.CUDA_SUCCESS
        status, self.executable = driver.cuGraphInstantiate(self.graph, 0)
        assert status == driver.CUresult.CUDA_SUCCESS

    def replay(self):
        import torch
        from cuda.bindings import driver

        status, = driver.cuGraphLaunch(self.executable, driver.CUstream(torch.cuda.current_stream().cuda_stream))
        assert status == driver.CUresult.CUDA_SUCCESS

    def close(self):
        from cuda.bindings import driver

        assert driver.cuGraphExecDestroy(self.executable)[0] == driver.CUresult.CUDA_SUCCESS
        assert driver.cuGraphDestroy(self.graph)[0] == driver.CUresult.CUDA_SUCCESS



class WindowScores:
    def __init__(self, cache, *, session, arena, maximum):
        import torch
        from ignition.components.fp8_indexer import PreparedFP8Indexer
        from ignition.recipes.kimi31.indexer import SharedBuffers

        self.width = min(maximum, 131072)
        device = cache.device
        self.query = torch.empty((128, 32, 128), dtype=torch.bfloat16, device=device)
        self.weights = torch.empty((128, 32), dtype=torch.bfloat16, device=device)
        self.pages = torch.zeros((1, self.width // 64), dtype=torch.int32, device=device)
        self.lengths = torch.zeros(128, dtype=torch.int32, device=device)
        output = torch.empty((128, 2048), dtype=torch.int32, device=device)
        self.indexer = PreparedFP8Indexer(self.query, self.weights, cache, self.pages.expand(128, -1),
            self.lengths, None, output, mode="prefill", session=session, name="diagnostic_score_window",
            workspace=arena, buffers=SharedBuffers({}, device), prefill_records_per_call=131072,
            topk=2048, head_weight_scale=1., hadamard=False, key_power_of_two=False)

    def __call__(self, query, weights, lengths, pages, row):
        import torch

        self.query.copy_(query)
        self.weights.copy_(weights)
        pieces = []
        for begin in range(0, pages.shape[1], self.pages.shape[1]):
            end = min(begin + self.pages.shape[1], pages.shape[1])
            self.pages.fill_(-1)
            self.pages[:, :end - begin].copy_(pages[:1, begin:end])
            self.lengths.copy_((lengths - begin * 64).clamp(0, self.width))
            self.indexer(self.query, self.weights)
            scratch = self.indexer.selection.binding.runtime.scratch
            width, block = scratch.paged_tile_logits_k_rows, scratch.prefill_block_k
            assert width == self.width
            values = scratch.get_indexer_contiguous_tile_logits()[:4 * width * 32].view(
                -1, width // block, 32, block)
            pieces.append(values[row // 32, :, row % 32].flatten()[:(end - begin)*64].clone())
        return torch.cat(pieces)



def _attention_oracle(reference, operation, snapshot, *, start, active, rank):
    """Check each rank's heads at query-slice boundaries and the active tail."""
    import torch

    begin, end = rank * ROWS // WORLD, (rank + 1) * ROWS // WORLD
    rows = {begin, end - 1}
    if active:
        rows.update((0, active - 1))
    checks = []
    for row in sorted(rows):
        ids = snapshot["selected"][row]
        ids = ids[ids >= 0].long()
        count = min(2048, start + row + 1) if row < active else 0
        assert ids.numel() == count
        records = snapshot["history"].view(torch.uint8)[ids].view(torch.float8_e4m3fn).float()
        query = snapshot["absorbed"][row].float().clamp(-448, 448).to(torch.float8_e4m3fn).float()
        splits = operation.parts[128]["attention"].plan.prepared.state.layout.num_splits
        expected = reference(query, records, splits=splits)
        actual = snapshot["attended"][row].float()
        mismatches = int((~torch.isfinite(actual) | ~torch.isfinite(expected)
            | ((actual - expected).abs() > .002 + .03 * expected.abs())).sum())
        checks.append(dict(row=row, selected=count, splits=splits, mismatches=mismatches,
                           max_absolute=float((actual - expected).abs().max())))
    return checks


def worker(rank, rendezvous, checkpoint, config_path, maximum):
    import torch
    import torch.distributed as dist
    from ignition import runtime_patches
    from ignition.components.fp8_indexer import write_fp8_index_keys
    from ignition.components.paged_batch import Metadata
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.shared_pages import SharedPages
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime, create_head_exchange
    from ignition.preparation.execution import GraphSignature
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.bootstrap import REDUCE_ROWS, allocate_prefill_histories
    from ignition.recipes.kimi3.weights import KimiWeightLoader
    from ignition.recipes.kimi31.attention import PreparedDSA
    from ignition.recipes.kimi31.dcp import AttentionConstants, ContextParallel, ContextShards
    from ignition.cli.config import Configuration


    runtime_patches.apply()
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=WORLD,
                            timeout=timedelta(minutes=20))
    graphs, owners, tails = {}, [], {}
    from tests.components.test_candidate_rows_cuda import probability_encoding_reference

    reference = probability_encoding_reference
    tail_type = RecordedAttentionTail
    try:
        with PackageStorage(rank=rank, world=WORLD, device=rank) as storage:
            with KimiWeightLoader(inspect_indexed_checkpoint(Path(checkpoint)), device,
                    shard=TensorParallelShard(rank, WORLD),
                    prefixes=("language_model.model.layers.3.self_attn.",),
                    online_mxfp8=False) as loader:
                weights = loader.load_attention(3)
                eps = loader.config.mla_norm_eps
            coordinator = CollectivePreparation(dist.group.WORLD)
            barrier = create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=REDUCE_ROWS,
                hidden_size=7168, ipc=storage.peer_storage)
            latents = create_head_exchange(dist.group.WORLD, device, max_rows=REDUCE_ROWS,
                heads=WORLD, head_dim=136, ipc=storage.peer_storage)
            selections = create_head_exchange(dist.group.WORLD, device, max_rows=ROWS//WORLD,
                heads=WORLD, head_dim=4096, ipc=storage.peer_storage)
            owners.extend((barrier, latents, selections))
            logical_pages = maximum // (WORLD*128)
            physical_pages = logical_pages + 3
            table = torch.randperm(physical_pages, generator=torch.Generator().manual_seed(8451))[
                :logical_pages].to(device=device, dtype=torch.int32)
            metadata = Metadata.allocate(ROWS, 1, logical_pages, device)
            metadata.page_table.copy_(table[None])
            shards = ContextShards.on(rank, WORLD, device)
            generator = torch.Generator(device=device).manual_seed(971)
            source = (torch.randn((ROWS, 7168), device=device, generator=generator)*.25).bfloat16()
            caches, index_caches, contexts, histories = {}, {}, {}, {}
            for kind in ("separate", "aliased"):
                cache_storage = SharedPages(dist.group.WORLD, device,
                    physical_pages*128*RECORD_BYTES, ipc=storage.peer_storage)
                stage = SharedPages(dist.group.WORLD, device, maximum//WORLD*RECORD_BYTES,
                    ipc=storage.peer_storage)
                index_stage = SharedPages(dist.group.WORLD, device, maximum//WORLD*132,
                    ipc=storage.peer_storage)
                owners.extend((cache_storage, stage, index_stage))
                cache = cache_storage.tensor.view(torch.float8_e4m3fn).view(physical_pages, 128, RECORD_BYTES)
                index = torch.empty((physical_pages*2, 64*132), dtype=torch.uint8, device=device)
                if kind == "aliased":
                    records, index_records = allocate_prefill_histories(maximum, device)
                else:
                    records = torch.empty((maximum, RECORD_BYTES), dtype=torch.float8_e4m3fn, device=device)
                    index_records = torch.empty((maximum//64, 64*132), dtype=torch.uint8, device=device)
                histories[kind] = records, index_records
                caches[kind], index_caches[kind] = cache, index
                contexts[kind] = ContextParallel(shards, None, coordinator,
                    {"latent": cache_storage.addresses}, AttentionConstants.allocate(device),
                    staging=stage, staged_records=records,
                    staged_address=torch.tensor([records.data_ptr()], dtype=torch.int64, device=device),
                    stage_runtime=barrier, stage_signals=tuple(torch.zeros((1,8), dtype=torch.bfloat16,
                        device=device) for _ in range(4)),
                    index_staging=index_stage, index_records=index_records,
                    index_columns=torch.arange(maximum//64, dtype=torch.int32, device=device),
                    index_signals=tuple(torch.zeros((1,8), dtype=torch.bfloat16, device=device) for _ in range(4)),
                    selections=selections)
            assert histories["aliased"][0].untyped_storage()._cdata == histories["aliased"][1].untyped_storage()._cdata
            assert histories["separate"][0].untyped_storage()._cdata != histories["separate"][1].untyped_storage()._cdata

            with RecordedPreparationSession(device=device, checkpoint=str(checkpoint),
                    workload_schema="kimi31-history-alias-layer", numerical_policy="exact-storage-and-prefill",
                    coordinator=coordinator, compile_workers=24) as session:
                arena = ScratchArena(device)
                session.register_scratch(arena)
                operations = {kind: PreparedDSA(weights, source, metadata, caches[kind],
                    index_cache=index_caches[kind], eps=eps, mode="prefill", session=session,
                    name=kind, workspace=arena, buffers={}, latents=latents,
                    coordinator=coordinator, context=context, peers=context.peers["latent"])
                    for kind, context in contexts.items()}
                score_windows = WindowScores(histories["separate"][1], session=session, arena=arena,
                    maximum=maximum)
                forwards = dict(separate=SeparateHistoryForward(operations["separate"], source),
                                aliased=AliasedHistoryForward(operations["aliased"], source))
                # Empty rows are valid during preparation and leave model histories untouched.
                for cache in (*caches.values(), *index_caches.values()):
                    cache.view(torch.uint8).zero_()
                session.tune(show_progress=False)
                functions = {kind: forward for kind, forward in forwards.items()}
                signatures = {kind: GraphSignature("prefill", 1, ROWS) for kind in functions}
                graphs, compiled = capture_forwards(functions, signatures, session=session, device=device,
                    stream=torch.cuda.Stream(device=device), pool=storage.graph_pool.id,
                    coordinator=coordinator, cache_root=Configuration.load(Path(config_path)).directory("kernel_cache"),
                    report_catalog=False, retain_graphs=True)
                tails["separate"] = tail_type(graphs["separate"], set(), expected_roots=ROWS // 128)

                generator.manual_seed(1307+rank)
                latent_seed = (torch.randn(caches["separate"].shape, device=device, generator=generator)*.125).to(torch.float8_e4m3fn)
                key_seed = (torch.randn((physical_pages*128, 128), device=device, generator=generator)*.2).bfloat16()
                slots = torch.arange(physical_pages*128, dtype=torch.int64, device=device)
                index_seed = torch.empty_like(index_caches["separate"])
                write_fp8_index_keys(key_seed, slots, index_seed, hadamard=False, power_of_two=False)
                row_ids = torch.arange(ROWS, device=device, dtype=torch.int64)
                cases = ((0,0), (0,17), (0,2047), (0,ROWS), (ROWS,ROWS-3),
                         (max(ROWS, maximum//2),ROWS), (maximum-ROWS,ROWS-7), (maximum-ROWS,ROWS),
                         (0,17), (maximum-ROWS,0), (0,0))
                for case, (start, active) in enumerate(cases):
                    metadata.starts.fill_(start)
                    metadata.lengths.fill_(active)
                    metadata.positions.copy_(torch.where(row_ids < active, start+row_ids, -1))
                    metadata.query_start_loc[0].zero_()
                    metadata.query_start_loc[1].fill_(active)
                    source.add_(.0009765625)
                    results = []
                    for kind in ("separate", "separate", "aliased"):
                        caches[kind].copy_(latent_seed)
                        index_caches[kind].copy_(index_seed)
                        for history in histories[kind]:
                            history.view(torch.uint8).fill_(255)
                        arena.tensor.fill_(255)
                        torch.cuda.synchronize()
                        coordinator.barrier(f"history case {case} {len(results)}", coordinator.ranks)
                        if rank == case % WORLD:
                            torch.cuda._sleep(1000000)
                        before = torch.cuda.memory_stats()
                        for _ in range(3):
                            graphs[kind].replay()
                        torch.cuda.synchronize()
                        after = torch.cuda.memory_stats()
                        for field in ("allocation.all.allocated", "allocation.all.freed",
                                      "segment.all.allocated", "segment.all.freed",
                                      "allocated_bytes.all.current", "active_bytes.all.current"):
                            assert after[field] == before[field], (kind, case, field)
                        operation = operations[kind]
                        value = operation.projections["output"].output.clone()
                        assert torch.isfinite(value).all()
                        assert not value[active:].count_nonzero()
                        visible = start + active if active else 0
                        results.append(dict(output=value, selected=operation.indexer.sharded.selected.clone(),
                            cache=caches[kind].view(torch.uint8).clone(), index_cache=index_caches[kind].clone(),
                            index_query=operation.indexer.query.clone(),
                            index_weight=operation.indexer.head_weights.clone(), gate=operation.gate.clone(),
                            absorbed=operation.absorbed.clone(), attended=operation.attended.clone(),
                            history=histories[kind][0][:visible].view(torch.uint8).clone().view(torch.float8_e4m3fn)))
                    control, twin, aliased = results
                    flags = {}
                    diagnostics = {}
                    ordered_control = control["selected"].sort(dim=-1).values
                    for label, value in (("twin", twin), ("aliased", aliased)):
                        for name in ("cache", "index_cache", "index_query", "index_weight", "absorbed", "gate", "history"):
                            flags[f"{label}_{name}"] = torch.equal(control[name].view(torch.uint8), value[name].view(torch.uint8))
                        delta = value["output"].float()-control["output"].float()
                        diagnostics[label] = dict(
                            identical_sets=torch.equal(ordered_control, value["selected"].sort(dim=-1).values),
                            identical_order=torch.equal(control["selected"], value["selected"]),
                            identical_output=torch.equal(control["output"], value["output"]),
                            relative_l2=float(delta.norm()/control["output"].float().norm().clamp_min(1e-12)),
                            maximum_absolute=float(delta.abs().max()))
                    attention_checks = {}
                    # Reuse the control's actual compiled nodes. A self-replay first
                    # checks that the extracted tail retains every required input.
                    # Supplying the other paths' emitted IDs then isolates the
                    # storage transformation from nondeterministic top-k ordering.
                    operation = operations["separate"]
                    for label, snapshot in (("control", control), ("twin", twin), ("aliased", aliased)):
                        local_ids = snapshot["selected"].cpu()
                        designated_ids = local_ids.clone()
                        dist.broadcast(designated_ids, src=0)
                        flags[f"{label}_selected_replicas"] = torch.equal(local_ids, designated_ids)
                        ordered = snapshot["selected"].sort(dim=-1).values
                        valid = ordered >= 0
                        counts = valid.sum(-1)
                        expected_counts = torch.where(row_ids < active, (start + row_ids + 1).clamp_max(2048), 0)
                        flags[f"{label}_selected_counts"] = torch.equal(counts, expected_counts)
                        flags[f"{label}_selected_padding"] = not bool((ordered < -1).any())
                        flags[f"{label}_selected_unique"] = not bool(((ordered[:, 1:] == ordered[:, :-1]) & valid[:, 1:]).any())
                        flags[f"{label}_selected_causal"] = not bool((valid & (ordered > start + row_ids[:, None])).any())
                        attention_checks[label] = _attention_oracle(reference, operations["aliased" if label == "aliased" else "separate"],
                            snapshot, start=start, active=active, rank=rank)
                        flags[f"{label}_attention_oracle"] = not any(check["mismatches"] for check in attention_checks[label])
                        operation.indexer.sharded.selected.copy_(snapshot["selected"])
                        operation.absorbed.copy_(snapshot["absorbed"])
                        operation.gate.copy_(snapshot["gate"])
                        tails["separate"].replay()
                        torch.cuda.synchronize()
                        flags[f"{label}_recorded_attended"] = torch.equal(operation.attended, snapshot["attended"])
                        flags[f"{label}_recorded_output"] = torch.equal(operation.projections["output"].output, snapshot["output"])
                    # Parallel top-k may choose different IDs only among values
                    # tied at the exact cutoff. Inspect every changed query on
                    # its query-owning rank using the original scorer geometry.
                    # These diagnostic calls follow all latent-history consumers.
                    changed = ((ordered_control != twin["selected"].sort(dim=-1).values).any(-1)
                        | (ordered_control != aliased["selected"].sort(dim=-1).values).any(-1))
                    tie_checks = []
                    for row in torch.where(changed & (row_ids % WORLD == rank))[0].tolist():
                        sharded = operation.indexer.sharded
                        begin = ((row // WORLD) // 128) * 128
                        scores = score_windows(sharded.local_query[begin:begin+128],
                            sharded.local_weights[begin:begin+128], sharded.lengths[begin:begin+128],
                            sharded.pages, (row // WORLD) % 128)[:start + row + 1]
                        cutoff = scores.topk(min(2048, scores.numel())).values[-1]
                        strict = torch.where(scores > cutoff)[0]
                        record = dict(row=row, cutoff=float(cutoff), strictly_better_ids=strict.numel(), paths={})
                        baseline_ids = control["selected"][row].long()
                        for label, snapshot in (("control", control), ("twin", twin), ("aliased", aliased)):
                            ids = snapshot["selected"][row].long()
                            differences = torch.cat((ids[~torch.isin(ids, baseline_ids)],
                                baseline_ids[~torch.isin(baseline_ids, ids)]))
                            valid = bool(torch.isfinite(scores[ids]).all() and (scores[ids] >= cutoff).all()
                                and torch.isin(strict, ids).all() and (scores[differences] == cutoff).all())
                            record["paths"][label] = dict(valid=valid, differing_ids=differences.numel(),
                                minimum_selected_margin=float((scores[ids] - cutoff).min()))
                        tie_checks.append(record)
                    flags["changed_sets_only_cutoff_ties"] = all(
                        path["valid"] for record in tie_checks for path in record["paths"].values())
                    all_ranks = [None]*WORLD
                    dist.all_gather_object(all_ranks, dict(flags=flags, diagnostics=diagnostics,
                        attention=attention_checks, cutoff_ties=tie_checks))
                    passed = all(all(row["flags"].values()) for row in all_ranks)
                    if rank == 0:
                        print(json.dumps(dict(start=start, active=active, passed=passed)),flush=True)
                    if not passed:
                        failures = [(rank, key) for rank, result in enumerate(all_ranks)
                                    for key, value in result["flags"].items() if not value]
                        raise AssertionError(f"sparse-prefill history check failed: {failures}")
                if rank == 0:
                    print(f"passed {len(cases)} complete attention-layer cases on {WORLD} ranks", flush=True)
                for tail in tails.values():
                    tail.close()
                tails.clear()
                for graph in graphs.values():
                    graph.reset()
                graphs.clear()
            for owner in reversed(owners):
                owner.close()
            owners.clear()
    except BaseException:
        import traceback
        import os
        traceback.print_exc()
        os._exit(1)
    finally:
        for tail in tails.values():
            tail.close()
        for graph in graphs.values():
            graph.reset()
        for owner in reversed(owners):
            owner.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    from importlib import import_module
    from ignition.cli.config import Configuration
    from ignition.cli.environment import apply
    from ignition.cli.preparation import Host
    from ignition.recipes.kimi31.recipe import RECIPE
    import torch.multiprocessing as mp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--context-capacity", type=int, default=1048576)
    args = parser.parse_args()
    if args.context_capacity < 4*ROWS or args.context_capacity % (WORLD*128):
        parser.error("context capacity must cover four chunks and whole TP16 cache pages")
    apply(Configuration.load(args.config), Host.discover().gpus[:16], RECIPE.b12x_environment)
    module = import_module("qualify.kimi_history_alias")
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(module.worker, args=((Path(directory)/"gloo").as_uri(), str(args.checkpoint),
            str(args.config), args.context_capacity), nprocs=WORLD, join=True)
