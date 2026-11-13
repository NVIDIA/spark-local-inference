# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare per-position and complete-block canonical proposal broadcasts.

The fixture uses input-dependent replicated transition rows and deliberately
different peer logits. It covers seven proposal filters, retained distributions,
canonical preceding-token IDs, and real PCIe transport. Model, vocabulary-head,
Markov GEMM and confidence projection execution are outside this diagnostic.
"""

import argparse
from datetime import timedelta
import json
from pathlib import Path
import statistics

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class Pipeline:
    def __init__(self, proposals, base, transitions, bias, drafts, retained, valid,
                 offset, sampled, pending, preceding, block=None):
        self.proposals, self.base, self.transitions = proposals, base, transitions
        self.bias, self.drafts, self.retained, self.valid = bias, drafts, retained, valid
        self.offset, self.sampled, self.pending = offset, sampled, pending
        self.preceding, self.block = preceding, block
        self.active = torch.ones_like(pending, dtype=torch.bool)
        self.limits = torch.full_like(pending, len(proposals) + 1)
        self.functional = False
        self.scratch_layout = {}

    def __call__(self):
        block_state = None if self.block is None else self.block.begin(self.pending.device)
        previous = torch.where(self.active, self.pending, 0)
        for index, operation in enumerate(self.proposals):
            self.bias.copy_(self.transitions[previous.remainder(self.transitions.shape[0])])
            self.offset.fill_(4 * index)
            self.preceding[:, index].copy_(previous)
            if self.functional:
                scratch = {name: torch.empty(shape, dtype=dtype, device=self.pending.device)
                           for name, (shape, dtype) in self.scratch_layout.items()}
                scratch['support_ids'] = block_state['support_ids'][index]
                scratch['support_probabilities'] = block_state['support_probabilities'][index]
                logits = torch.empty((self.drafts.shape[0], 1, self.base.shape[-1]),
                                     dtype=torch.float32, device=self.pending.device)
                operation.run(scratch, logits, source_logits=self.base[:, index], candidates=self.drafts,
                              source_bias=self.bias, seed=self.seed, offset=self.offset, sampled=self.sampled,
                              slots=self.slots, generations=self.generations, live_rows=self.live_rows)
            else:
                operation()
            self.drafts[:, index].copy_(self.sampled)
            if self.block is None:
                self.retained[:, index].copy_(operation.probabilities[:, 0])
                self.valid[:, index].copy_(operation.valid[:, 0] & self.active & (index < self.limits - 1))
            else:
                if self.functional:
                    self.block.retain(block_state, index, scratch, self.sampled)
                else:
                    support_ids, support_probabilities = operation.proposal._support
                    self.block.retain(block_state, index, dict(support_ids=support_ids,
                        support_probabilities=support_probabilities, valid=operation.valid), self.sampled)
            previous = self.drafts[:, index]
        if self.block is not None:
            self.block.finish(block_state, self.drafts, self.retained, self.valid, self.sampled, self.active, self.limits)
            self.preceding[:, 0].copy_(torch.where(self.active, self.pending, 0))
            self.preceding[:, 1:].copy_(self.drafts[:, :-1])


def rank_main(rank, rendezvous, output):
    from ignition.components.draft_sampling import PreparedDraftDistribution, PreparedDraftBlockBroadcast
    from ignition.components.parallel_draft import prepare_sampling_broadcasts
    from ignition.components.sampling import DeviceRequestState, RUNNING
    from ignition.components.speculation import DRAFT_RNG_DOMAIN
    from ignition.preparation.collectives import CollectivePreparation, create_result_broadcast_runtime
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.graphs import PreparedGraphBundle
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(minutes=15))
    coordinator = CollectivePreparation(dist.group.WORLD)
    depth, vocab = 7, 129280
    runtime = create_result_broadcast_runtime(dist.group.WORLD, device,
        layouts=(PreparedDraftDistribution.broadcast_layout(32 * depth),))
    owners, graphs, records = {}, {}, []
    worker = None
    try:
        with RecordedPreparationSession(device=device, checkpoint='conditional-proposal-block',
                workload_schema='conditional-proposal-block-v1', numerical_policy='default',
                coordinator=coordinator) as session:
            for batch in (1, 8, 32):
                generator = torch.Generator().manual_seed(4187 + batch + rank)
                base = torch.randn((batch, depth, vocab), generator=generator).to(device, torch.bfloat16)
                transitions = torch.randn((32, vocab), generator=generator).to(device)
                for variant in ('serial', 'block'):
                    state = DeviceRequestState.allocate(device, vocab_size=vocab, capacity=max(4, batch))
                    state.generation.fill_(7)
                    state.status.fill_(RUNNING)
                    state.parameters.copy_(torch.tensor([1., 1., 0., 1., 0., 0.], device=device))
                    state.top_k.zero_()
                    slots = torch.arange(batch, dtype=torch.int32, device=device)
                    epochs = torch.full((batch,), 7, dtype=torch.int64, device=device)
                    live = torch.tensor([batch], dtype=torch.int32, device=device)
                    scratch = {"proposal": {
                        name: torch.zeros(shape, dtype=dtype, device=device)
                        for name, (shape, dtype) in PreparedDraftDistribution.scratch_layout(batch, 1, vocab).items()}}
                    drafts = torch.zeros((batch, depth), dtype=torch.int64, device=device)
                    retained = torch.zeros((batch, depth, vocab), device=device)
                    valid = torch.zeros((batch, depth), dtype=torch.bool, device=device)
                    logits = torch.zeros((batch, 1, vocab), device=device)
                    bias = torch.zeros((batch, vocab), device=device)
                    seed = torch.arange(batch, dtype=torch.int64, device=device) ^ DRAFT_RNG_DOMAIN
                    offset, sampled = torch.zeros_like(seed), torch.zeros_like(seed)
                    pending = torch.arange(batch, dtype=torch.int64, device=device)
                    preceding = torch.zeros_like(drafts)
                    proposals = tuple(PreparedDraftDistribution(logits, state, slots, epochs, live, drafts,
                        prefix_offset=index, seed=seed, offset=offset, sampled=sampled,
                        workspace=scratch['proposal']) for index in range(depth))
                    block = None
                    for index, operation in enumerate(proposals):
                        operation.bind_logits(base[:, index], bias)
                        operation.prepare()
                    if variant == 'serial':
                        prepare_sampling_broadcasts(proposals, None,
                            collectives=dict(vocabulary_runtime=runtime, coordinator=coordinator),
                            session=session, name=f'serial{batch}')
                    else:
                        block = PreparedDraftBlockBroadcast(proposals, drafts, retained, valid, sampled,
                            runtime=runtime, session=session, coordinator=coordinator,
                            name=f'block{batch}')
                    pipeline = Pipeline(proposals, base, transitions, bias, drafts, retained, valid,
                                        offset, sampled, pending, preceding, block)
                    pipeline.seed = seed
                    pipeline.slots, pipeline.generations, pipeline.live_rows = slots, epochs, live
                    pipeline.scratch_layout = {name: (tuple(value.shape), value.dtype)
                                              for name, value in scratch['proposal'].items()}
                    owners[batch, variant] = (pipeline, state, live, base, transitions)
            session.tune(show_progress=False)
            for key, owner in owners.items():
                if key[1] == 'block':
                    for operation in owner[0].proposals:
                        operation.functional(session=None, dynamic_inputs=True)
                    owner[0].functional = True
                call = compile_forward(owner[0])
                call()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph):
                    call()
                graphs[key] = graph
                owners[key] = (*owner, call)
            session.freeze()
            graph_ids = {key: index for index, key in enumerate(graphs)}
            bundle = PreparedGraphBundle(device=rank,
                graphs={str(graph_ids[key]): graph for key, graph in graphs.items()}, buffers={},
                resources=(session, runtime, tuple(owners.values())))
            worker = bundle.start_worker()
            for batch in (1, 8, 32):
                # Different peer prefixes must not change any final canonical field.
                for fixture in ('default', 'penalties', 'inactive', 'prefix_limits', 'ties', 'token_zero', 'invalid'):
                    for variant in ('serial', 'block'):
                        pipeline, state, live, *_ = owners[batch, variant]
                        state.parameters.copy_(torch.tensor(
                            [0.8, 0.95, 0.03, 1.1, 0.2, 0.1] if fixture == 'penalties'
                            else [1., 1., 0., 1., 0., 0.], device=device))
                        state.top_k.fill_(17 if fixture == 'penalties' else 0)
                        live.fill_(max(0, batch - 1) if fixture == 'inactive' else batch)
                        state.status.fill_(0 if fixture == 'invalid' else RUNNING)
                        pipeline.pending.copy_((torch.arange(batch, device=device) + 13) % vocab)
                        pipeline.active.copy_((torch.arange(batch, device=device) % 3 != 1)
                                              if fixture == 'prefix_limits' else True)
                        pipeline.limits.copy_((torch.arange(batch, device=device) % (depth + 1) + 1)
                                              if fixture == 'prefix_limits' else depth + 1)
                        if fixture == 'ties':
                            pipeline.base.zero_()
                            pipeline.transitions.zero_()
                        if fixture == 'token_zero':
                            pipeline.base.fill_(-10.)
                            pipeline.base[..., 0] = 2.
                            pipeline.base[..., 4] = 1.
                            pipeline.base[..., 7] = .5
                            pipeline.transitions.zero_()
                            state.top_k.fill_(3)
                        for _ in range(2):
                            pipeline.retained.fill_(float('nan'))
                            torch.cuda.synchronize()
                            ticket = worker.submit(graph_ids[batch, variant])
                            completion = worker.poll(10000)
                            assert completion is not None and completion.ticket == ticket and completion.error is None
                        torch.cuda.synchronize()
                    serial, block = owners[batch, 'serial'][0], owners[batch, 'block'][0]
                    for name in ('drafts', 'valid', 'preceding', 'retained'):
                        left, right = getattr(serial, name), getattr(block, name)
                        if left.dtype == torch.float32:
                            left, right = left.view(torch.int32), right.view(torch.int32)
                        assert torch.equal(left, right), (batch, fixture, name, rank)
                # Restore representative non-tied inputs before timing.
                generator = torch.Generator().manual_seed(8381 + batch + rank)
                base = torch.randn((batch, depth, vocab), generator=generator).to(device, torch.bfloat16)
                transitions = torch.randn((32, vocab), generator=generator).to(device)
                for variant in ('serial', 'block'):
                    pipeline, state, live, *_ = owners[batch, variant]
                    pipeline.base.copy_(base)
                    pipeline.transitions.copy_(transitions)
                    state.parameters.copy_(torch.tensor([1., 1., 0., 1., 0., 0.], device=device))
                    state.top_k.zero_()
                    state.status.fill_(RUNNING)
                    live.fill_(batch)
                    pipeline.active.fill_(True)
                    pipeline.limits.fill_(depth + 1)
                    for _ in range(20):
                        graphs[batch, variant].replay()
                    torch.cuda.synchronize()
                for sample, variant in enumerate(('serial', 'block', 'block', 'serial') * 3):
                    dist.barrier()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    allocations = torch.cuda.memory_stats()['allocation.all.allocated']
                    start.record()
                    for _ in range(25):
                        graphs[batch, variant].replay()
                    end.record()
                    end.synchronize()
                    assert allocations == torch.cuda.memory_stats()['allocation.all.allocated']
                    own = start.elapsed_time(end) / 25
                    paired = [None, None]
                    dist.all_gather_object(paired, own)
                    records.append(dict(batch=batch, sample=sample, variant=variant,
                                        rank_ms=paired, slowest_rank_ms=max(paired)))
            if rank == 0:
                summary = {str(batch): {variant: statistics.median(
                    record['slowest_rank_ms'] for record in records
                    if record['batch'] == batch and record['variant'] == variant)
                    for variant in ('serial', 'block')} for batch in (1, 8, 32)}
                output.write_text(json.dumps(dict(scope=__doc__, summary=summary, records=records,
                    correctness='Bitwise canonical drafts/q/valid/preceding IDs on both native workers; conditional peer divergence, penalties, inactive rows, prefix limits, ties, token-zero padding and invalid requests; stable replay allocation counters.'), indent=2) + '\n')
                print(json.dumps(summary), flush=True)
    finally:
        if worker is not None:
            worker.close()
        for graph in graphs.values():
            graph.reset()
        runtime.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    assert not args.output.exists()
    mp.spawn(rank_main, args=((args.output.parent / (args.output.stem + '.rendezvous')).as_uri(), args.output),
             nprocs=2, join=True)
