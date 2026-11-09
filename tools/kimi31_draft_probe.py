# SPDX-License-Identifier: Apache-2.0

"""Validate complete Kimi K3.1 DFlash2 proposals on an assigned TP group.

Loads the actual target embedding/head and draft checkpoint without target
decoder weights. Uses ordinary preparation and compiled graph capture, checks
accepted-history ownership and poisoned scratch, and records GPU timings.
This fixture does not measure speculative acceptance or whole-model throughput.
"""

import argparse
from dataclasses import replace
from datetime import timedelta
from functools import partial
import ctypes
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace


def rank_main(rank, world, rendezvous, options):
    import torch
    import torch.distributed as dist
    from ignition.components.dflash2 import PreparedDraft
    from ignition.components.paged_batch import Metadata
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.weights import DirectWeightLoader
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.collectives import (CollectivePreparation,
        create_all_reduce_runtime, create_all_gather_runtime, create_head_exchange)
    from ignition.preparation.disposable import storage_ranges
    from ignition.preparation.execution import GraphSignature
    from ignition.preparation.core_storage import CoreStorage
    from ignition.preparation.runtime_core import _Cuda
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.config import ModelConfig, DSparkConfig
    from ignition.recipes.kimi31.dflash import DraftScratch, PreparedContext, DraftAttention
    from ignition.recipes.kimi31.dflash_weights import DraftWeights

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    device, shard = torch.device("cuda", rank), TensorParallelShard(rank, world)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=15))
    target, checkpoint = Path(options.target), Path(options.draft)
    config = ModelConfig.load(target)
    config = replace(config, draft=DSparkConfig.load(checkpoint, config))
    block, batch, history = 8, 2, 1056
    h, vocab_size = config.hidden, config.vocabulary
    outcomes = []
    with CoreStorage(rank=rank, world=world, device=rank) as storage:
        with DirectWeightLoader(inspect_indexed_checkpoint(target), device,
                prefixes=("language_model.model.embed_tokens", "language_model.lm_head")) as loader:
            embedding = loader.load_tensor("language_model.model.embed_tokens.weight",
                extent=(slice(None), slice(*shard.partition(h))))
            head = loader.load_tensor("language_model.lm_head.weight",
                extent=slice(*shard.partition(vocab_size)))
        draft = DraftWeights.load(checkpoint, config, device, shard, target_head=head,
            online_mxfp8=options.mxfp8_body, draft_head_quantization=options.head_quantization)
        weights = SimpleNamespace(config=config, shard=shard, draft=draft, embedding=embedding, head=head)
        backing = torch.zeros((batch+1, config.draft.layers*1024*576),
                              dtype=torch.float8_e4m3fn, device=device)
        rings = {f"draft{i}.latent": backing[:, i*1024*576:(i+1)*1024*576].view(batch+1, 1024, 576)
                 for i in range(config.draft.layers)}
        pool = SimpleNamespace(reserved_region=rings.__getitem__)
        c = Metadata.allocate(history, 1, 1, device)
        c.state_rows.fill_(1)
        c.lengths.fill_(history)
        c.query_start_loc[-1] = history
        m = Metadata.allocate(batch*block, batch, 1, device, samples=batch*block)
        m.lengths.fill_(block)
        m.query_start_loc.copy_(torch.arange(batch+1, device=device, dtype=torch.int32)*block)
        m.sample_indices.copy_(torch.arange(batch*block, device=device))
        m.state_rows.copy_(torch.arange(1, batch+1, device=device))
        m.live_counts.copy_(torch.tensor([batch*block, batch], device=device))
        m.token_ids.fill_(config.draft.noise_token)
        m.token_ids[0] = 163584
        m.token_ids[block] = 42
        torch.manual_seed(6731)
        auxiliary = (torch.randn((history, len(config.draft.auxiliary_layers)*h), device=device)*.2).bfloat16()
        group, ipc = dist.group.WORLD, storage.peer_storage
        coordinator = CollectivePreparation(group) if world > 1 else None
        def channel(factory, **kwargs):
            return factory(group, device, ipc=ipc, **kwargs) if world > 1 else None
        runtime = channel(create_all_reduce_runtime, max_tokens=batch*block, hidden_size=h)
        gather = channel(create_all_gather_runtime, max_rows=128, columns=h)
        vocab = channel(create_all_gather_runtime, max_rows=batch*block, columns=vocab_size)
        latents = channel(create_head_exchange, max_rows=batch*block, heads=world,
                                      head_dim=((2112//world+7)//8)*8)
        context_latents = channel(create_head_exchange, max_rows=128, heads=world,
            head_dim=((config.draft.layers*576//world+7)//8)*8)
        conv = channel(create_head_exchange, max_rows=batch*block, heads=world,
                                   head_dim=4*h//16//world)
        with RecordedPreparationSession(device=device, checkpoint=str(checkpoint),
                workload_schema=f"dflash-mla-tp{world}", numerical_policy="a16-fp8-kv",
                coordinator=coordinator, compile_workers=16,
                benchmark_storage=storage, collective_tuning="static") as session:
            arena = ScratchArena(device)
            session.register_scratch(arena)
            scratch = DraftScratch.allocate(config, shard, batch*block, block, device, GraphBufferPool(device))
            context_buffers, attention_buffers = GraphBufferPool(device), GraphBufferPool(device)
            context = PreparedContext(weights, auxiliary, c, pool, session=session,
                name="context", workspace=arena, runtime=gather, coordinator=coordinator,
                latent_runtime=context_latents, buffers=context_buffers)
            op = PreparedDraft(weights, scratch, m, pool, runtime=runtime, vocabulary_runtime=vocab,
                embedding_runtime=gather, coordinator=coordinator, session=session, name="draft",
                workspace=arena, buffers=attention_buffers.cursor(), embedding=embedding,
                head=head if draft.head is None else draft.head, output_multiplier=1., final_logit_softcapping=None,
                attention_factory=partial(DraftAttention, latent_runtime=latents), convolution_runtime=conv)
            session.tune(show_progress=False)

            def position(start, active=batch):
                c.starts.fill_(start)
                c.positions.copy_(torch.arange(start, start+history, device=device))
                m.starts.fill_(start+history)
                m.lengths.zero_()
                m.lengths[:active].fill_(block)
                m.query_start_loc.zero_()
                m.query_start_loc[1:].copy_(m.lengths.cumsum(0))
                m.live_counts.copy_(torch.tensor([active*block, active], device=device))
                m.positions.fill_(-1)
                m.positions[:active*block].copy_(torch.arange(start+history, start+history+block, device=device).repeat(active))

            position(0)
            context(auxiliary, c.lengths)
            for ring in rings.values():
                ring[2].copy_(ring[1])
            functions = dict(context=partial(context, auxiliary, c.lengths), draft=op)
            signatures = dict(context=GraphSignature("prefill", 1, history, ()),
                draft=GraphSignature("speculative", batch, block, (), packed_token_capacity=batch*block))
            graphs, retained = capture_forwards(functions, signatures, session=session, device=device,
                stream=torch.cuda.Stream(device=device), pool=storage.graph_pool.id, coordinator=coordinator,
                compile_decoder=True, warmup_storage=storage,
                cache_root=None if options.kernel_cache is None else Path(options.kernel_cache),
                progress=(lambda text: print(text, flush=True)) if rank == 0 else None)
            session.freeze()
            cuda = _Cuda()
            for start, active in ((0, 2), (1023, 2), (500001, 2), (500001, 1), (0, 0)):
                position(start, active)
                auxiliary.add_(.00390625)
                graphs["context"].replay()
                for ring in rings.values():
                    ring[2].copy_(ring[1])
                saved = backing.clone()
                op()
                expected = scratch.logits.clone()
                torch.cuda.synchronize()
                for address, size in storage_ranges((scratch.scratch_tensors(), attention_buffers.chunks)):
                    cuda.call("cuMemsetD8_v2", ctypes.c_uint64(address), ctypes.c_ubyte(255), ctypes.c_size_t(size))
                dist.barrier()
                allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                for _ in range(3):
                    graphs["draft"].replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
                actual = scratch.logits.clone()
                assert torch.isfinite(actual).all()
                error = float((actual-expected).float().norm()/expected.float().norm().clamp_min(1e-10))
                assert error < .003, (rank, start, error)
                assert torch.equal(backing.view(torch.uint8), saved.view(torch.uint8)), "Proposal mutated accepted history"
                summaries = [None]*world
                dist.all_gather_object(summaries, dict(rank=rank, error=error, argmax=actual.argmax(-1).tolist()))
                assert all(item["argmax"] == summaries[0]["argmax"] for item in summaries)
                start_event, stop_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                dist.barrier()
                start_event.record()
                for _ in range(20):
                    graphs["draft"].replay()
                stop_event.record()
                stop_event.synchronize()
                timings = [None]*world
                dist.all_gather_object(timings, start_event.elapsed_time(stop_event)/20)
                result = dict(start=start, active_requests=active, ranks=summaries, gpu_ms_per_rank=timings,
                              slowest_rank_gpu_ms=max(timings))
                outcomes.append(result)
                if rank == 0:
                    print(json.dumps(result), flush=True)
            for graph in graphs.values():
                graph.reset()
        for channel in (conv, context_latents, latents, vocab, gather, runtime):
            if channel is not None:
                channel.close()
    dist.destroy_process_group()
    if rank == 0:
        Path(options.output).write_text(json.dumps(dict(configuration=vars(options), measurements=outcomes), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--world", type=int, default=16)
    parser.add_argument("--mxfp8-body", action="store_true")
    parser.add_argument("--head-quantization", choices=("mxfp8", "nvfp4"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--kernel-cache", help="Optional prepared-forward cache directory")
    options = parser.parse_args()
    import torch
    import torch.multiprocessing as mp
    assert torch.cuda.device_count() == options.world, "Expose exactly the assigned TP group"
    with tempfile.TemporaryDirectory(prefix="k31-dflash-tp-") as directory:
        mp.spawn(rank_main, args=(options.world, Path(directory, "gloo").as_uri(), options),
                 nprocs=options.world, join=True)
