"""Compare latent projection and routing against a full-K numerical reference.

The reference multiplies the checkpoint's BF16 weights and BF16 inputs in FP32
with TF32 disabled, then rounds to BF16. It specifies projection arithmetic;
it is not a complete Moonshot model implementation or a model-quality oracle.
After every numerical gate passes, paired timings include projections, packing,
transport and expert selection. They omit expert execution and model throughput.
"""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import traceback

import torch


def projection_errors(actual, reference):
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise AssertionError('latent projection contains nonfinite values')
    difference = actual.float() - reference.float()
    row_error = difference.norm(dim=1)
    row_norm = reference.float().norm(dim=1)
    if torch.any((row_norm == 0) & (row_error != 0)):
        raise AssertionError('zero-reference rows changed')
    relative = torch.where(row_norm > 0, row_error / row_norm.clamp_min(torch.finfo(torch.float32).tiny), 0)
    norm = reference.float().norm()
    return dict(relative_l2=float(difference.norm() / norm) if norm > 0 else 0.,
                worst_row_relative_l2=float(relative.max()), max_absolute=float(difference.abs().max()))


def check_projection_gate(control, candidate, reference):
    errors = {'column': projection_errors(control, reference), 'row': projection_errors(candidate, reference)}
    for name, values in errors.items():
        if values['relative_l2'] >= .01:
            raise AssertionError(f'{name} projection exceeds the established relative-L2 bound: {values}')
    for metric in ('relative_l2', 'worst_row_relative_l2'):
        if errors['row'][metric] > errors['column'][metric]:
            raise AssertionError(f'row-sharded projection regresses {metric}: {errors}')
    return errors


class ProjectionRouting:
    def __init__(self, source, row_weight, column_weight, router_weight, correction, columns, *,
                 row_sharded, reduction, router_runtime, pair_runtime, session, coordinator,
                 buffers, name):
        from ignition.components.column_pair_gather import PreparedColumnPairGather
        from ignition.components.collectives import PreparedChunkedAllGather, PreparedChunkedAllReduce
        from ignition.components.linear import UnquantizedProjection
        from ignition.components.moe import PreparedRouter, RouterScratch

        rows, device = source.shape[0], source.device
        self.source, self.columns, self.row_sharded = source, columns, row_sharded
        self.local_logits = torch.empty((rows, 56), dtype=torch.float32, device=device)
        self.logits = torch.empty((rows, 896), dtype=torch.float32, device=device)
        self.latent = torch.empty((rows, 3584), dtype=torch.bfloat16, device=device)
        self.router_projection = UnquantizedProjection(source, router_weight, self.local_logits,
            session=session, name=name + '.router')
        self.routes = RouterScratch.allocate(rows, device, experts=896, top_k=16)
        self.router = PreparedRouter(self.logits, self.routes, score_func='sigmoid', correction_bias=correction,
            renormalize=True, routed_scaling_factor=1., session=session, name=name + '.route')
        if row_sharded:
            self.local_down = torch.empty((rows, 224), dtype=torch.bfloat16, device=device)
            self.down = UnquantizedProjection(source, row_weight, self.local_down, session=session, name=name + '.down')
            self.gather = PreparedColumnPairGather(self.local_down, self.local_logits, self.latent, self.logits,
                runtime=pair_runtime, session=session, coordinator=coordinator, name=name + '.gather', buffers=buffers)
        else:
            self.local_source = torch.empty((rows, 448), dtype=torch.bfloat16, device=device)
            self.partial = torch.empty((rows, 3584), dtype=torch.bfloat16, device=device)
            self.down = UnquantizedProjection(self.local_source, column_weight, self.partial,
                session=session, name=name + '.down')
            self.reduce = PreparedChunkedAllReduce(self.partial, self.latent, runtime=reduction,
                coordinator=coordinator, session=session, name=name + '.reduce')
            self.send = torch.zeros((rows, 64), dtype=torch.float32, device=device)
            self.received = torch.empty((rows, 1024), dtype=torch.float32, device=device)
            self.gather = PreparedChunkedAllGather(self.send.view(torch.bfloat16), self.received.view(torch.bfloat16),
                runtime=router_runtime, coordinator=coordinator, session=session, name=name + '.gather')

    def __call__(self):
        logits = self.router_projection(self.source)
        if self.row_sharded:
            self.down(self.source)
            self.gather()
            self.router(self.logits)
        else:
            self.send[:, :56].copy_(logits)
            self.gather(self.send.view(torch.bfloat16))
            self.logits.view(-1, 16, 56).copy_(self.received.view(-1, 16, 64)[:, :, :56])
            self.router(self.logits)
            self.local_source.copy_(self.source.index_select(1, self.columns))
            self.reduce(self.down(self.local_source))
        return self.latent


def worker(rank, rendezvous, checkpoint, destination):
    from ignition import runtime_patches
    runtime_patches.apply()
    import torch.distributed as dist
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.collectives import CollectivePreparation, create_all_gather_runtime, create_all_reduce_runtime, create_head_exchange
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.weights import KimiWeightLoader

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group('gloo', rank=rank, world_size=16, init_method=rendezvous,
                            timeout=timedelta(minutes=30))
    graphs, runtimes, results, timings = [], [], [], []

    def save():
        if rank == 0:
            Path(destination).write_text(json.dumps(dict(checkpoint=str(checkpoint), layer=1,
                reference='Full-K FP32 projection of BF16 inputs and weights, rounded to BF16; TF32 disabled.',
                gate='Each case: relative L2 below 0.01; candidate overall and worst-row relative L2 no worse than control; exact router transport and routing.',
                cases=results, component_timings=timings), indent=2) + '\n')

    try:
        shard = TensorParallelShard(rank, 16)
        prefix = 'language_model.model.layers.1.block_sparse_moe'
        manifest = inspect_indexed_checkpoint(Path(checkpoint))
        with KimiWeightLoader(manifest, device, shard=shard,
                prefixes=(prefix + '.routed_expert_down_proj', prefix + '.gate')) as loader:
            full_down = loader.linear(prefix + '.routed_expert_down_proj')
            full_router = loader.linear(prefix + '.gate')
            correction = loader.load_tensor(prefix + '.gate.e_score_correction_bias')
        assert full_down.shape == (3584, 7168) and full_down.dtype == torch.bfloat16
        assert full_router.shape == (896, 7168) and full_router.dtype == torch.bfloat16
        h0, h1 = shard.partition(7168)
        d0, d1 = shard.partition(3584)
        r0, r1 = shard.partition(896, alignment=8)
        column_weight, row_weight = full_down[:, h0:h1].contiguous(), full_down[d0:d1].contiguous()
        router_weight = full_router[r0:r1].contiguous()
        columns = torch.arange(h0, h1, device=device)
        coordinator = CollectivePreparation(dist.group.WORLD)
        reduction = create_all_reduce_runtime(dist.group.WORLD, device, collective_tuning='static', max_tokens=1024, hidden_size=7168)
        router_runtime = create_all_gather_runtime(dist.group.WORLD, device, max_rows=1024, columns=2048)
        pair_runtime = create_head_exchange(dist.group.WORLD, device, max_rows=1024, heads=16, head_dim=336)
        runtimes = [pair_runtime, router_runtime, reduction]
        with RecordedPreparationSession(device=device, checkpoint=str(Path(checkpoint).resolve()),
                workload_schema='kimi-full-k-latent-routing', numerical_policy='fp32-reference-bf16-output',
                coordinator=coordinator, compile_workers=16) as session:
            cases, buffers, compiled, timing_cases = [], {}, [], []
            for rows, strided in ((4096, False), (128, False), (16, False), (128, True)):
                backing = torch.zeros((rows, 7200 if strided else 7168), dtype=torch.bfloat16, device=device)
                source = backing[:, :7168]
                variants = {name: ProjectionRouting(source, row_weight, column_weight, router_weight, correction, columns,
                    row_sharded=name == 'row', reduction=reduction, router_runtime=router_runtime, pair_runtime=pair_runtime,
                    session=session, coordinator=coordinator, buffers=buffers, name=f'{name}-{rows}-{strided}')
                    for name in ('column', 'row')}
                cases.append((rows, strided, source, variants))
            if rank == 0:
                print('Tuning latent projection and router fixtures with ordinary b12x selection', flush=True)
            session.tune(show_progress=False)
            for rows, strided, source, variants in cases:
                for mode in ('eager', 'compiled'):
                    if rank == 0:
                        print(f'Preparing projection/routing gates: rows={rows}, strided={strided}, mode={mode}', flush=True)
                    calls = {name: compile_forward(operation) if mode == 'compiled' else operation for name, operation in variants.items()}
                    compiled.extend(calls.values())
                    captures = {}
                    for name, call in calls.items():
                        call()
                        torch.cuda.synchronize(device)
                        graph = torch.cuda.CUDAGraph()
                        dist.barrier()
                        with session.capture(), torch.cuda.graph(graph):
                            call()
                        captures[name] = graph
                        graphs.append(graph)
                    for seed in (8101, 8102, 8103):
                        generator = torch.Generator(device=device).manual_seed(seed + rows)
                        source.copy_(torch.randn(source.shape, dtype=source.dtype, device=device, generator=generator))
                        active = rows if seed == 8101 else rows - 7
                        source[active:].zero_()
                        reference = torch.nn.functional.linear(source.float(), full_down.float()).bfloat16()
                        for name in ('column', 'row'):
                            operation = variants[name]
                            operation.latent.fill_(torch.nan)
                            operation.logits.fill_(torch.nan)
                            if name == 'row':
                                operation.gather.send.fill_(torch.nan)
                                operation.gather.received.fill_(torch.nan)
                            torch.cuda.synchronize(device)
                            dist.barrier()
                            before = torch.cuda.memory_stats(device)['allocation.all.allocated']
                            captures[name].replay()
                            torch.cuda.synchronize(device)
                            assert torch.cuda.memory_stats(device)['allocation.all.allocated'] == before
                        errors = check_projection_gate(variants['column'].latent, variants['row'].latent, reference)
                        assert torch.equal(variants['column'].logits.view(torch.int32), variants['row'].logits.view(torch.int32))
                        assert torch.equal(variants['column'].routes.ids, variants['row'].routes.ids)
                        assert torch.equal(variants['column'].routes.weights.view(torch.int32), variants['row'].routes.weights.view(torch.int32))
                        assert not variants['row'].latent[active:].count_nonzero()
                        per_rank = [None] * 16
                        dist.all_gather_object(per_rank, errors)
                        results.append(dict(rows=rows, source_stride=list(source.stride()), mode=mode,
                                            seed=seed, active_rows=active, errors_by_rank=per_rank, exact_routing=True))
                        save()
                    if mode == 'compiled' and not strided:
                        timing_cases.append((rows, source, variants, calls))
                if rank == 0:
                    print(f'Projection/routing gates passed: rows={rows}, strided={strided}', flush=True)
            from b12x.preparation._measurement import _l2_flush_fn

            flush = _l2_flush_fn(device, enabled=True)
            flush()
            for rows, source, variants, calls in timing_cases:
                generator = torch.Generator(device=device).manual_seed(8103 + rows)
                source.copy_(torch.randn(source.shape, dtype=source.dtype, device=device, generator=generator))
                source[rows - 7:].zero_()
                reference = torch.nn.functional.linear(source.float(), full_down.float()).bfloat16()
                captures, events, samples = {}, {}, {name: [] for name in calls}
                for name, call in calls.items():
                    pairs = [(torch.cuda.Event(enable_timing=True, external=True),
                              torch.cuda.Event(enable_timing=True, external=True)) for _ in range(2)]
                    graph = torch.cuda.CUDAGraph()
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    with session.capture(), torch.cuda.graph(graph):
                        for start, stop in pairs:
                            flush()
                            start.record()
                            call()
                            stop.record()
                    graphs.append(graph)
                    captures[name], events[name] = graph, pairs
                for pair in range(6):
                    for name in (('column', 'row') if pair % 2 == 0 else ('row', 'column')):
                        dist.barrier()
                        before = torch.cuda.memory_stats(device)['allocation.all.allocated']
                        captures[name].replay()
                        torch.cuda.synchronize(device)
                        assert torch.cuda.memory_stats(device)['allocation.all.allocated'] == before
                        durations = [start.elapsed_time(stop) for start, stop in events[name]]
                        per_rank = [None] * 16
                        dist.all_gather_object(per_rank, durations)
                        samples[name].extend(max(values) for values in zip(*per_rank, strict=True))
                    check_projection_gate(variants['column'].latent, variants['row'].latent, reference)
                    assert torch.equal(variants['column'].logits.view(torch.int32), variants['row'].logits.view(torch.int32))
                    assert torch.equal(variants['column'].routes.ids, variants['row'].routes.ids)
                    assert torch.equal(variants['column'].routes.weights.view(torch.int32), variants['row'].routes.weights.view(torch.int32))
                timing = dict(rows=rows, source_stride=list(source.stride()), mode='compiled',
                    metric='maximum rank CUDA-event duration; L2 flush before each measured call',
                    active_rows=rows - 7, samples_ms=samples,
                    median_ms={name: statistics.median(values) for name, values in samples.items()})
                timings.append(timing)
                save()
                if rank == 0:
                    print(f'Projection/routing component timing: {timing}', flush=True)
            for graph in graphs:
                graph.reset()
            graphs.clear()
    except BaseException:
        # A rank-local failure can leave peers in a collective. Exit before any
        # collective cleanup so multiprocessing can terminate the whole group.
        print(f'Projection qualifier failed on rank {rank}', file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    finally:
        for graph in graphs:
            graph.reset()
        for runtime in runtimes:
            runtime.close()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must not already exist')
    if torch.cuda.device_count() != 16:
        parser.error('assign exactly sixteen idle CUDA devices')
    import torch.multiprocessing as mp
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='kimi-latent-prefill-') as directory:
        mp.spawn(worker, args=((Path(directory) / 'gloo').as_uri(), str(args.checkpoint), str(args.output)), nprocs=16, join=True)


if __name__ == '__main__':
    main()
