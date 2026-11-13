# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare complete replicated and head-sharded Engram graphs across TP ranks."""

import argparse
from datetime import timedelta
import json
from types import SimpleNamespace

import torch
import torch.distributed as dist


class Forward:
    def __init__(self, operation, source):
        self.operation, self.source = operation, source

    def __call__(self):
        return self.operation(self.source)


@torch.no_grad()
def run(args):
    from b12x.gemm.blockscaled import pack_weight
    from b12x.sequence import engram

    from ignition.components.engram import EngramInputs
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.collectives import (
        CollectivePreparation, create_all_reduce_runtime, create_roce_runtime,
    )
    from ignition.preparation.execution import GraphSignature
    from ignition.preparation.host_ranks import _use_interface
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.deepseek41.engram import EngramScratch, PreparedEngram

    torch.cuda.set_device(args.device)
    device = torch.device('cuda', args.device)
    _use_interface(args.address)
    dist.init_process_group('gloo', init_method=args.rendezvous, rank=args.rank,
                            world_size=args.world, timeout=timedelta(minutes=30))
    coordinator = CollectivePreparation(dist.group.WORLD, timeout=timedelta(minutes=30))
    shard = TensorParallelShard(args.rank, args.world)
    first, last = shard.partition(6144, alignment=256)
    geometry = engram.build_geometry(layer_ids=(1,), base_table_size=101, compressed_vocab_size=32)
    start = geometry.offsets[0][first // 256]
    end = geometry.offsets[0][last // 256] if last < 6144 else geometry.num_embeddings[0]
    capacity = max(args.rows)
    # Use the recipe's hidden-width channel so prefill also exercises chunked reduction.
    runtime = (create_roce_runtime(dist.group.WORLD, device, max_tokens=128, pipe_tokens=capacity,
                                   hidden_size=5120, gather_bytes=4096)
               if coordinator.spans_hosts else
               create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=128, hidden_size=5120))
    generator = torch.Generator().manual_seed(1017)
    values = torch.randint(-7, 8, (25600, 6144), dtype=torch.int8, generator=generator).to(torch.float8_e4m3fn)
    scales = torch.full((25600, 6144 // 32), 118, dtype=torch.uint8)
    full_weight = pack_weight(values.to(device), scales.to(device), recipe='mxfp8')
    local_weight = pack_weight(values[:, first:last].contiguous().to(device),
                              scales[:, first // 32:last // 32].contiguous().to(device), recipe='mxfp8')
    del values, scales
    arena = ScratchArena(device)
    graphs = {}
    try:
        with RecordedPreparationSession(device=device, checkpoint='engram-head-sharding-fp8',
                workload_schema='engram-tp-v1', numerical_policy='default', coordinator=coordinator,
                compile_workers=2) as session:
            session.register_scratch(arena)
            functions, signatures, cases = {}, {}, {}
            for rows in args.rows:
                source = torch.randn((rows, 4, 5120), generator=generator, dtype=torch.bfloat16).to(device)
                metadata = SimpleNamespace(token_ids=torch.zeros(rows, dtype=torch.int64, device=device),
                    positions=torch.arange(rows, device=device),
                    live_counts=torch.tensor([rows], dtype=torch.int32, device=device))
                outputs, inputs, operations = [], [], []
                for label, partition, weight in (
                        ('replicated', TensorParallelShard(0, 1), full_weight),
                        ('sharded', shard, local_weight)):
                    compact = EngramInputs.allocate(rows, device)
                    columns = torch.arange(6144, device=device) if label == 'replicated' else torch.arange(first, last, device=device)
                    tables = SimpleNamespace(inputs={1: compact}, token_map=tuple(range(32)), geometry=geometry,
                                             columns=columns, shard=partition)
                    output = torch.empty_like(source)
                    operation = PreparedEngram(1, {'projection': weight, 'weight': torch.ones((4, 5120), device=device)},
                        tables, source, output, metadata, SimpleNamespace(config=SimpleNamespace(text={'rms_norm_eps': 1e-6})),
                        scratch=EngramScratch.allocate(rows, device, world=partition.world_size), session=session,
                        name=f'{label}.{rows}', workspace=arena, runtime=runtime if label == 'sharded' else None,
                        coordinator=coordinator)
                    name = f'{label}.{rows}'
                    functions[name] = Forward(operation, source)
                    signatures[name] = (GraphSignature('prefill', 1, rows) if rows >= 128 else
                                        GraphSignature('decode', rows, 1))
                    outputs.append(output)
                    inputs.append(compact)
                    operations.append(operation)
                cases[rows] = source, metadata, inputs, outputs, operations
            session.tune(show_progress=False)
            stream = torch.cuda.Stream(device=device)
            graphs, compiled = capture_forwards(functions, signatures, session=session, device=device,
                stream=stream, pool=torch.cuda.graph_pool_handle(), collective=runtime, coordinator=coordinator)
            session.freeze()
            for rows, (source, metadata, inputs, outputs, operations) in cases.items():
                dense, local = inputs
                # Each head samples its own table segment; changing generations changes every lookup.
                offsets = torch.tensor(geometry.offsets[0], dtype=torch.int64)
                primes = torch.tensor(geometry.primes[0], dtype=torch.int64)
                for generation, live in enumerate((rows, max(1, rows // 2), 0, rows), 1):
                    ids = (torch.arange(rows)[:, None] * 19 + generation) % primes + offsets
                    quant = ((ids[:, :, None] + torch.arange(256)) % 15 - 7).to(torch.float8_e4m3fn)
                    scale = torch.full((rows, 24, 8), 124, dtype=torch.uint8)
                    dense.hashes.copy_(ids)
                    local.hashes.copy_(torch.where((ids >= start) & (ids < end), ids - start, -1))
                    for compact in inputs:
                        compact.weights.copy_(quant.flatten(0, 1).view(torch.uint8))
                        compact.scales.copy_(scale.flatten(0, 1))
                        compact.ready.record()
                    metadata.live_counts.fill_(live)
                    metadata.positions.copy_(torch.where(torch.arange(rows, device=device) < live,
                                                         torch.arange(rows, device=device), -1))
                    for output in outputs:
                        output.fill_(float('nan'))
                    torch.cuda.synchronize()
                    allocated = torch.cuda.memory_allocated()
                    for label in ('replicated', 'sharded'):
                        graphs[f'{label}.{rows}'].replay()
                    torch.cuda.synchronize()
                    assert torch.cuda.memory_allocated() == allocated
                    reference, actual = (output.float() for output in outputs)
                    error = (actual - reference).square().mean().sqrt()
                    relative = error / reference.square().mean().sqrt().clamp_min(1e-6)
                    assert torch.isfinite(actual).all()
                    assert relative < 0.01, (rows, live, relative.item())
                    torch.testing.assert_close(outputs[1][live:], source[live:], rtol=0, atol=0)
                    print(json.dumps({'rank': args.rank, 'rows': rows, 'live': live,
                        'relative_rms_error': relative.item(), 'max_abs_error': (actual - reference).abs().max().item()}), flush=True)
                times = {}
                for label in ('replicated', 'sharded'):
                    dist.barrier()
                    begin, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(10):
                        graphs[f'{label}.{rows}'].replay()
                    finish.record()
                    finish.synchronize()
                    times[label] = begin.elapsed_time(finish) / 10
                print(json.dumps({'rank': args.rank, 'rows': rows, 'gpu_ms': times}), flush=True)
            dist.barrier()
    finally:
        for graph in graphs.values():
            graph.reset()
        runtime.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    # Compiler workers deserialize forwards by their importable module name.
    from scripts.qualify_engram_tp import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world', type=int, default=4)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--rendezvous', required=True)
    parser.add_argument('--address', required=True, help='local IPv4 address used by the rank control group')
    parser.add_argument('--rows', type=int, nargs='+', default=[1, 16, 96, 4096])
    run(parser.parse_args())
