# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP16 query overlap preserves selected records and outputs with one collective in flight."""

from copy import copy
from datetime import timedelta
import json
import statistics
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import torch

from tests.recipes.kimi31_attention_observations import (
    AttentionForward, AttentionTail, allocate_observations, poison_observations,
)

pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_TP_CUDA_TESTS') != '1',
                               reason='requires sixteen explicitly assigned GPUs')


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


def _check_collective_edges(graph, query_functions, candidate_functions, *, overlap):
    kernels, edges = _graph_nodes(graph)
    assert query_functions.isdisjoint(candidate_functions), 'collective function identities are ambiguous'
    query = {node for node, item in kernels.items() if item['function'] in query_functions}
    candidate = {node for node, item in kernels.items() if item['function'] in candidate_functions}
    assert query and candidate, 'captured graph lacks its prepared collective kernels'
    successors = {}
    for parent, child in edges:
        successors.setdefault(parent, []).append(child)
    def reaches(source, target):
        pending, visited = [source], set()
        while pending:
            node = pending.pop()
            if node == target:
                return True
            if node not in visited:
                visited.add(node)
                pending.extend(successors.get(node, ()))
        return False
    if overlap:
        assert all(reaches(q, c) for q in query for c in candidate), 'candidate exchange can precede query completion'
    else:
        assert all(reaches(c, q) for q in query for c in candidate), 'serial collective order changed'
    scoring = {node for node, item in kernels.items() if 'DSAFusedIndexerKernel' in item['symbol']}
    if overlap:
        assert scoring, 'local index-scoring kernel was not identified'
        assert any(not reaches(q, local) and not reaches(local, q) for q in query for local in scoring), (
            'query gather and local index scoring have no independent branches')
    return dict(query_nodes=sorted(query), candidate_nodes=sorted(candidate), local_scoring_nodes=sorted(scoring),
                kernels=kernels, edges=edges, query_before_candidates=overlap)


def _rank(rank, directory, history_capacity, implementation, measure_latency):
    import torch.distributed as dist
    from cuda.bindings import driver

    from ignition.preparation.capture import capture_forwards
    from b12x.comm.pcie.pcie_dcp_topk import PCIeDCPTopKOwnerExchange
    from ignition.preparation.collectives import (CollectivePreparation, create_head_exchange,
                                                 create_all_reduce_runtime)
    from ignition.preparation.execution import GraphSignature
    from ignition.preparation.core_storage import CoreStorage
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi31.attention import PreparedDSA
    from ignition.recipes.kimi31.dcp import AttentionConstants, ContextParallel, ContextShards, RECORD_BYTES

    from ignition import runtime_patches
    runtime_patches.apply()

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    directory = Path(directory)
    from ignition.preparation import compilation
    compilation.DEFAULT_CACHE_ROOT = directory / 'kernels'
    dist.init_process_group('gloo', init_method=(directory / 'group').as_uri(), rank=rank, world_size=16,
                            timeout=timedelta(minutes=20))
    graphs = {}
    compiled = {}
    tails = {}
    tail_compiled = {}
    runtimes = []
    session = None
    try:
        with CoreStorage(rank=rank, world=16, device=rank) as storage:
            generator = torch.Generator(device=device).manual_seed(7391)
            def weight(shape, scale=.015625):
                return (torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator) * scale)
            weights = {
                'query_norm': torch.ones(1536, device=device, dtype=torch.bfloat16),
                'kv_norm': torch.ones(512, device=device, dtype=torch.bfloat16),
                'latent_gate': weight((136 + 6 * 128, 7168)),
                'q_b': weight((6 * 192, 1536)), 'key': weight((6, 128, 512)),
                'value': weight((6, 512, 128)), 'output': weight((7168, 6 * 128)),
                'indexer.wq_b': weight((4096, 1536)), 'indexer.wk': weight((128, 7168)),
                'indexer.weights_proj': weight((32, 7168)),
                'indexer.k_norm.weight': torch.ones(128, device=device, dtype=torch.bfloat16),
                'indexer.k_norm.bias': torch.zeros(128, device=device, dtype=torch.bfloat16),
            }
            source = weight((1, 7168), .125)
            seed = source.clone()
            assert history_capacity % (16 * 128) == 0
            pages = history_capacity // (16 * 128)
            physical_pages = pages + 3
            page_map = (torch.arange(pages, device=device, dtype=torch.int32) + 2) % physical_pages
            metadata = SimpleNamespace(starts=torch.zeros(1, dtype=torch.int64, device=device),
                                       positions=torch.tensor([16], dtype=torch.int64, device=device),
                                       token_page_table=page_map[None])
            cache = torch.zeros((physical_pages, 128, RECORD_BYTES), device=device, dtype=torch.float8_e4m3fn)
            index_cache = torch.zeros((physical_pages * 2, 64 * 132), device=device, dtype=torch.uint8)
            coordinator = CollectivePreparation(dist.group.WORLD)
            heads = create_head_exchange(dist.group.WORLD, device, max_rows=1, heads=96,
                head_dim=512, query_head_dim=RECORD_BYTES, ipc=storage.peer_storage)
            candidates = create_head_exchange(dist.group.WORLD, device, max_rows=1, heads=16,
                head_dim=8192, ipc=storage.peer_storage)
            latents = create_head_exchange(dist.group.WORLD, device, max_rows=1, heads=16,
                head_dim=136, ipc=storage.peer_storage)
            runtimes.extend((heads, candidates, latents))
            owner = PCIeDCPTopKOwnerExchange.from_process_group(process_group=dist.group.WORLD,
                device=device, max_rows=1, topk=2048, stream_affine=False, ipc=storage.peer_storage)
            reduction = create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=1, hidden_size=8192,
                                                  ipc=storage.peer_storage)
            runtimes.extend((owner, reduction))
            tuning = SimpleNamespace(reduction=reduction, register=lambda selection: selection.bind(implementation))
            context = ContextParallel(ContextShards.on(rank, 16, device), heads, coordinator, {},
                                      AttentionConstants.allocate(device, query_capacity=1), candidates=candidates,
                                      candidate_owners=owner, selection_tuning=tuning)
            session = RecordedPreparationSession(device=device, checkpoint='kimi31-query-overlap',
                workload_schema='one-row-local-index-and-serialized-collectives', numerical_policy='exact',
                coordinator=coordinator, autotune_effort='default', compile_workers=16)
            session.__enter__()
            arena = ScratchArena(device, overlap=True)
            session.register_scratch(arena)
            operation = PreparedDSA(weights, source, metadata, cache, index_cache=index_cache, eps=1e-5,
                mode='decode', session=session, name='attention', workspace=arena, buffers={}, context=context, latents=latents)
            assert operation.query_overlap is not None
            serial = copy(operation)
            serial.query_overlap = None
            session.tune(show_progress=False)
            assert operation.parts[1]['gather'].plan.memory_requirements().scratch_nbytes == 0
            local_positions = torch.arange(pages * 128, device=device, dtype=torch.int64)
            physical = page_map[local_positions // 128].long() * 128 + local_positions % 128
            cache.view(-1, RECORD_BYTES)[physical] = weight((pages * 128, RECORD_BYTES), .125).to(cache.dtype)
            operation.indexer.selections[1].indexer.write_keys(weight((pages * 128, 128), .125), physical)
            torch.cuda.synchronize(device)
            saved_cache, saved_index = cache.clone(), index_cache.clone()
            part = operation.parts[1]
            selector = part['selector']
            merge = selector.merge
            observations = allocate_observations(operation._run(source, metadata))
            forwards = {name: AttentionForward(prepared, source, metadata, observations=observations)
                        for name, prepared in (('serial', serial), ('overlap', operation))}
            production = {name: AttentionForward(prepared, source, metadata)
                          for name, prepared in (('serial', serial), ('overlap', operation))}
            observed, = observations.stages
            packed, local_ids, local_scores, index_query, index_scales, index_weights = observed.index

            def restore():
                cache.copy_(saved_cache)
                index_cache.copy_(saved_index)
                arena.tensor.fill_(0xA5)
                poison_observations(observations)
                for forward in (*forwards.values(), *production.values()):
                    forward.output.fill_(float('nan'))

            def snapshot(output):
                values = dict(output=output, gathered_query=observed.gathered, selected_ids=observed.selected,
                    physical_slots=observed.slots, counts=observed.counts, latent_cache=cache, index_cache=index_cache,
                    local_ids=local_ids, local_scores=local_scores,
                    index_query=index_query, index_query_scales=index_scales,
                    index_query_weights=index_weights)
                return {name: value.view(torch.uint8).clone() for name, value in values.items()}

            eager_tail = AttentionTail(operation, observations, metadata)

            def check(actual, expected, stage, visible, step, tail=None):
                differences = []
                def same(name, left, right):
                    mask = left != right
                    changed = int(mask.count_nonzero().item())
                    if changed:
                        positions = mask.flatten().nonzero().flatten()[:16]
                        differences.append(dict(field=name, changed_bytes=changed, shape=list(left.shape),
                            offsets=positions.cpu().tolist(), actual=left.flatten()[positions].cpu().tolist(),
                            expected=right.flatten()[positions].cpu().tolist()))
                        torch.save(dict(actual=left.cpu(), expected=right.cpu()),
                                   directory / f'rank-{rank}-{stage}-{step}-{name}.pt')
                for name in actual.keys() - {'output', 'selected_ids', 'physical_slots', 'counts', 'local_ids', 'local_scores'}:
                    same(name, actual[name], expected[name])
                # The local top-k also emits unordered candidate pairs. Align by ID
                # only in the oracle, retaining exact FP32 score bits and membership.
                def candidate_pairs(record):
                    ids = record['local_ids'].cpu().view(torch.int32).flatten()
                    bits = record['local_scores'].cpu().view(torch.int32).flatten()
                    order = ids.argsort(stable=True)
                    return torch.stack((ids[order], bits[order]), -1).view(torch.uint8)
                same('local_candidate_pairs', candidate_pairs(actual), candidate_pairs(expected))
                same_order = torch.equal(actual['selected_ids'], expected['selected_ids'])
                if same_order:
                    for name in ('output', 'physical_slots', 'counts'):
                        same(name, actual[name], expected[name])
                # Selection permits arbitrary order and arbitrary cutoff-tie membership.
                # Each returned ID must attain the exact FP32 kth score of this invocation.
                candidates = [None] * 16
                dist.all_gather_object(candidates, (local_ids[0].cpu(), local_scores[0].cpu()))
                ids = torch.cat([torch.where(local >= 0, local * 16 + owner, -1)
                                 for owner, (local, _) in enumerate(candidates)])
                scores = torch.cat([values for _, values in candidates])
                chosen = observed.selected[0].cpu()
                valid = ids >= 0
                live_scores, live_ids = scores[valid], ids[valid]
                assert torch.isfinite(live_scores).all()
                live = chosen[chosen >= 0]
                count = min(2048, len(live_ids))
                assert len(live) == count and len(live.unique()) == count
                assert ((live >= 0) & (live < visible)).all()
                if count:
                    cutoff = live_scores.topk(count).values[-1]
                    score_by_id = dict(zip(live_ids.tolist(), live_scores.tolist(), strict=True))
                    assert all(value in score_by_id and score_by_id[value] >= cutoff.item() for value in live.tolist())
                    must_select = set(live_ids[live_scores > cutoff].tolist())
                    assert must_select <= set(live.tolist())
                agreed = [None] * 16
                dist.all_gather_object(agreed, chosen)
                assert all(torch.equal(value, agreed[0]) for value in agreed)
                # Query storage may be reused after its join; the recorded gather must
                # still contain this invocation's absorbed query.
                assert not observed.quantized.view(torch.uint8).count_nonzero()
                recorded_query = observations.projected.absorbed.float().clamp(-448., 448.).to(observed.quantized.dtype)
                same('recorded_query_source', recorded_query.view(torch.uint8),
                     observed.gathered.view(1, 96, RECORD_BYTES)[:, rank * 6:(rank + 1) * 6].contiguous().view(torch.uint8))
                if tail is None:
                    replayed_output = eager_tail()
                else:
                    tails[tail].replay()
                    replayed_output = tail_forwards[tail].output
                torch.cuda.synchronize(device)
                for name, value in snapshot(replayed_output).items():
                    same('recorded_tail_' + name, value, actual[name])
                record = dict(rank=rank, stage=stage, visible=visible, step=step, same_selection_order=same_order,
                              exact_recorded_selection_tail=True, differences=differences)
                with (directory / f'rank-{rank}-checks.jsonl').open('a') as log:
                    log.write(json.dumps(record) + '\n')
                peers = [None] * 16
                dist.all_gather_object(peers, record)
                assert not any(value['differences'] for value in peers), peers

            # Capture each exact prepared call to identify its CUfunction handles in
            # the complete graph. No kernel-name substring substitutes for this mapping.
            collective_functions = {}
            transport = ((lambda: merge.gather(packed.view(torch.bfloat16).view(1, -1)))
                         if implementation == 'replicated' else
                         (lambda: merge.owner.transport(packed[1:].view(torch.int32), packed[:1])))
            for name, launch in (
                    ('query', lambda: part['gather'](observed.quantized.view(1, -1))),
                    ('candidate', transport)):
                launch()
                torch.cuda.synchronize(device)
                probe = torch.cuda.CUDAGraph(keep_graph=True)
                with session.capture(), torch.cuda.graph(probe):
                    launch()
                kernels, _ = _graph_nodes(probe)
                assert kernels
                collective_functions[name] = {entry['function'] for entry in kernels.values()}
                probe.reset()

            functions = {**forwards, **{name + '-production': forward for name, forward in production.items()}}
            signatures = {name: GraphSignature('decode', 1, 1) for name in functions}
            graphs, compiled = capture_forwards(functions, signatures, session=session, device=device,
                stream=torch.cuda.Stream(device=device), pool=torch.cuda.graph_pool_handle(),
                coordinator=coordinator, cache_root=directory / 'compiled', retain_graphs=True,
                retain_forwards=True, before_run=lambda name: restore())
            tail_forwards = {name: AttentionTail(operation, observations, metadata) for name in forwards}
            tail_graphs, tail_compiled = capture_forwards(tail_forwards, signatures={
                name: GraphSignature('decode', 1, 1) for name in tail_forwards}, session=session,
                device=device, stream=torch.cuda.Stream(device=device), pool=torch.cuda.graph_pool_handle(),
                coordinator=coordinator, cache_root=directory / 'tails', retain_forwards=True)
            tails.update(tail_graphs)
            session.freeze()
            for name in production:
                graph = graphs[name + '-production']
                status, = driver.cuGraphDebugDotPrint(driver.CUgraph(graph.raw_cuda_graph()),
                    str(directory / f'rank-{rank}-{name}.dot').encode(), 1)
                assert status == driver.CUresult.CUDA_SUCCESS
                dag = _check_collective_edges(graph, collective_functions['query'],
                                              collective_functions['candidate'], overlap=name == 'overlap')
                (directory / f'rank-{rank}-{name}-dag.json').write_text(json.dumps(dag) + '\n')

            rebound = copy(forwards['overlap'])
            rebound.source = source.clone()
            rebound.output = torch.empty_like(source)
            production_rebound = copy(production['overlap'])
            production_rebound.source = rebound.source
            production_rebound.output = torch.empty_like(source)
            contexts = tuple(value for value in (0, 17, 2051, 16384, 65536) if value <= history_capacity)
            contexts += (history_capacity, 17)
            for step, visible in enumerate(contexts):
                source.copy_(seed * (step + 1) / 4)
                rebound.source.copy_(source)
                metadata.positions.fill_(visible - 1)
                restore()
                reference = forwards['serial']()
                torch.cuda.synchronize(device)
                expected = snapshot(reference)
                check(expected, expected, 'serial-reference', visible, step)
                for name in ('serial', 'overlap'):
                    restore()
                    actual = forwards[name]()
                    torch.cuda.synchronize(device)
                    check(snapshot(actual), expected, name + '-eager', visible, step)
                    restore()
                    dist.barrier()
                    if rank == step % 16:
                        time.sleep(.025)
                    before = torch.cuda.memory_stats(device)
                    for _ in range(3):
                        graphs[name].replay()
                    torch.cuda.synchronize(device)
                    after = torch.cuda.memory_stats(device)
                    for key in ('allocation.all.allocated', 'allocation.all.freed', 'segment.all.allocated'):
                        assert before[key] == after[key], (name, key)
                    check(snapshot(forwards[name].output), expected, name + '-compiled-queued', visible, step, tail=name)
                    for captured in (False, True):
                        restore()
                        before = torch.cuda.memory_stats(device)
                        if captured:
                            for _ in range(3):
                                graphs[name + '-production'].replay()
                        else:
                            production[name]()
                        torch.cuda.synchronize(device)
                        after = torch.cuda.memory_stats(device)
                        if captured:
                            for key in ('allocation.all.allocated', 'allocation.all.freed', 'segment.all.allocated'):
                                assert before[key] == after[key], (name, key)
                        torch.testing.assert_close(cache.view(torch.uint8), expected['latent_cache'], rtol=0, atol=0)
                        torch.testing.assert_close(index_cache, expected['index_cache'], rtol=0, atol=0)
                        torch.testing.assert_close(production[name].output.view(torch.uint8), expected['output'],
                                                   rtol=0, atol=0)
                    dist.barrier()
                restore()
                compiled['overlap'].artifact(rebound)
                torch.cuda.synchronize(device)
                check(snapshot(rebound.output), expected, 'overlap-rebound', visible, step, tail='overlap')
                cache.copy_(saved_cache)
                index_cache.copy_(saved_index)
                compiled['overlap-production'].artifact(production_rebound)
                torch.cuda.synchronize(device)
                torch.testing.assert_close(production_rebound.output.view(torch.uint8), expected['output'], rtol=0, atol=0)
                torch.testing.assert_close(cache.view(torch.uint8), expected['latent_cache'], rtol=0, atol=0)
                torch.testing.assert_close(index_cache, expected['index_cache'], rtol=0, atol=0)
                agreed = [None] * 16
                dist.all_gather_object(agreed, observed.selected.cpu())
                assert all(torch.equal(value, agreed[0]) for value in agreed)
            print(f'rank {rank}: correctness and captured dependency checks passed', flush=True)
            if not measure_latency:
                return
            samples = []
            source.copy_(seed)
            repetitions = 100
            for visible in (1024, 8192, 65536):
                metadata.positions.fill_(visible - 1)
                for block in range(5):
                    for name in ('serial', 'overlap', 'overlap', 'serial'):
                        restore()
                        for _ in range(5):
                            graphs[name + '-production'].replay()
                        torch.cuda.synchronize(device)
                        dist.barrier()
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record()
                        for _ in range(repetitions):
                            graphs[name + '-production'].replay()
                        end.record()
                        end.synchronize()
                        elapsed = begin.elapsed_time(end) * 1000 / repetitions
                        timings = [None] * 16
                        dist.all_gather_object(timings, elapsed)
                        samples.append(dict(context=visible, block=block, schedule=name,
                                            rank_microseconds=timings, maximum_rank_microseconds=max(timings)))
                if rank == 0:
                    summary = {name: statistics.median(row['maximum_rank_microseconds'] for row in samples
                        if row['context'] == visible and row['schedule'] == name) for name in ('serial', 'overlap')}
                    print(f'context {visible}: complete component maximum-rank medians us {summary}', flush=True)
            if rank == 0:
                (directory / 'component-timings.json').write_text(json.dumps(dict(
                    boundary='CUDA events around repeated uninstrumented complete compiled DSA graphs; each sample uses the maximum of 16 rank-local intervals',
                    prepared_history_capacity=history_capacity, correctness_contexts=[0, 17, 2051, 16384, 65536, history_capacity, 17],
                    repetitions=repetitions, sampling_order='ABBA, five blocks per context', samples=samples), indent=2) + '\n')
    finally:
        for tail in tails.values():
            tail.reset()
        for graph in graphs.values():
            graph.reset()
        for forward in (*compiled.values(), *tail_compiled.values()):
            forward.retire_after_capture()
        if session is not None:
            session.close()
        for runtime in reversed(runtimes):
            runtime.close()
        dist.destroy_process_group()


@pytest.mark.parametrize('history_capacity,measure_latency', [(8192, False), (1048576, True)])
@pytest.mark.parametrize('implementation', ['replicated', 'owner'])
def test_tp16_query_overlap_eager_compiled_rebinding_and_queued_replay(
        tmp_path, history_capacity, implementation, measure_latency):
    import torch.multiprocessing as mp

    assert torch.cuda.device_count() == 16
    mp.spawn(_rank, args=(str(tmp_path), history_capacity, implementation, measure_latency), nprocs=16, join=True)
