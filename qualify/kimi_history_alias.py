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
import hashlib
import importlib.util
import inspect
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
        for part in op.parts:
            selector = part["selector"]
            selection_slots(selector.selected, selector.positions, selector.tables, part["owners"],
                part["slots"], part["counts"], op.shards, op.cache.shape[0], local=False, global_records=True)
            part["gather"]()
            part["attention"]()
        return op.finish()


class AliasedHistoryForward:
    def __init__(self, attention, source):
        self.attention, self.source = attention, source

    def __call__(self):
        return self.attention(self.source)


def _test_module(relative, name):
    path = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        splits = operation.parts[row // 128]["attention"].plan.prepared.state.layout.num_splits
        expected = reference(query, records, splits=splits)
        actual = snapshot["attended"][row].float()
        mismatches = int((~torch.isfinite(actual) | ~torch.isfinite(expected)
            | ((actual - expected).abs() > .002 + .03 * expected.abs())).sum())
        checks.append(dict(row=row, selected=count, splits=splits, mismatches=mismatches,
                           max_absolute=float((actual - expected).abs().max())))
    return checks


def worker(rank, rendezvous, checkpoint, config_path, output, maximum):
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
    from ignition.preparation.core_storage import CoreStorage
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.bootstrap import REDUCE_ROWS, allocate_prefill_histories
    from ignition.recipes.kimi3.weights import KimiWeightLoader
    from ignition.recipes.kimi31.attention import PreparedDSA
    from ignition.recipes.kimi31.dcp import AttentionConstants, ContextParallel, ContextShards
    from ignition.cli.config import Configuration
    from qualify.kimi_query_prefill import WindowScores

    runtime_patches.apply()
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=WORLD,
                            timeout=timedelta(minutes=20))
    graphs, owners, tails = {}, [], {}
    report = dict(rows=ROWS, prepared_context=maximum, queued_replays=3, cases=[], status="running")
    reference = _test_module("tests/components/test_candidate_rows_cuda.py",
        "kimi_history_probability_oracle").probability_encoding_reference
    tail_type = _test_module("tests/recipes/test_kimi31_query_overlap_tp16_cuda.py",
        "kimi_history_compiled_tail").RecordedAttentionTail
    try:
        with CoreStorage(rank=rank, world=WORLD, device=rank) as storage:
            with KimiWeightLoader(inspect_indexed_checkpoint(Path(checkpoint)), device,
                    shard=TensorParallelShard(rank, WORLD),
                    prefixes=("language_model.model.layers.3.self_attn.",),
                    online_mxfp8=True, online_mxfp8_scope="kda-input") as loader:
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
            report["logical_history_bytes_removed_per_rank"] = histories["separate"][1].numel()

            with RecordedPreparationSession(device=device, checkpoint=str(checkpoint),
                    workload_schema="kimi31-history-alias-layer", numerical_policy="exact-storage-and-prefill",
                    namespace={"forward_source": hashlib.sha256((inspect.getsource(SeparateHistoryForward)
                        + inspect.getsource(AliasedHistoryForward)).encode()).hexdigest()},
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
                report["projection_plans"] = {
                    f"{kind}.{role}": plan.contract.config_payload(plan.selection.config).to_dict()
                    for kind, op in operations.items()
                    for role, plan in (
                        *((name, projection.plan) for name, projection in op.projections.items()),
                        *((f"index{index}", projection.plan) for index, projection in enumerate(op.indexer.projections)))}
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
                        indexer = operation.indexer.sharded.indexers[(row // WORLD) // 128]
                        if indexer.pages.shape[1] * 64 <= 131072:
                            indexer()
                        scores = score_windows(indexer, (row // WORLD) % 128)[:start + row + 1]
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
                    report["cases"].append(dict(start=start, active=active, ranks=all_ranks))
                    if not passed:
                        report["status"] = "failed"
                    if rank == 0:
                        Path(output).write_text(json.dumps(report,indent=2)+"\n")
                        print(json.dumps(dict(start=start, active=active, passed=passed)),flush=True)
                    if not passed:
                        if rank == 0:
                            rows = torch.tensor(sorted({0,16,127,1023,2047,2048,ROWS-1}), device=device)
                            torch.save({"rows": rows.cpu(), "start": start, "active": active,
                                "snapshots": {label: {name: value[name][rows].cpu()
                                    for name in ("output", "selected", "index_query", "index_weight", "absorbed", "attended")}
                                    for label, value in (("control",control),("twin",twin),("aliased",aliased))}},
                                Path(output).with_suffix(".samples.pt"))
                        raise AssertionError("Prefill storage, selected-set, attention-oracle or recorded-order output check failed")
                report["status"] = "passed"
                if rank == 0:
                    Path(output).write_text(json.dumps(report,indent=2)+"\n")
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-capacity", type=int, default=1048576)
    args = parser.parse_args()
    if args.context_capacity < 4*ROWS or args.context_capacity % (WORLD*128):
        parser.error("context capacity must cover four chunks and whole TP16 cache pages")
    apply(Configuration.load(args.config), Host.discover().gpus[:16], RECIPE.b12x_environment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    module = import_module("qualify.kimi_history_alias")
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(module.worker, args=((Path(directory)/"gloo").as_uri(), str(args.checkpoint),
            str(args.config), str(args.output), args.context_capacity), nprocs=WORLD, join=True)
