# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distributed probability normalization and rejection over real vocabulary shards."""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
                               reason="requires two explicitly assigned CUDA devices")


def _rank(rank, rendezvous, fp8):
    from ignition.components.sampling import DeviceRequestState, RUNNING
    from ignition.components.speculation import DRAFT_RNG_DOMAIN, SpeculativeRngState
    from ignition.components.vocabulary_sampling import PreparedVocabularySoftmax, PreparedVocabularyBlockVerification
    from ignition.preparation.collectives import CollectivePreparation, create_all_gather_runtime
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.graphs import PreparedGraphBundle
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(minutes=3))
    coordinator = CollectivePreparation(dist.group.WORLD)
    b, d, v, local, features = 2, 2, 1025, 576, 64
    runtime = create_all_gather_runtime(dist.group.WORLD, device, max_rows=b * (d + 1), columns=128)
    generator = torch.Generator().manual_seed(125)
    full_base = torch.randn(b, d, v, generator=generator).to(device, torch.bfloat16)
    full_target = torch.randn(b, d + 1, v, generator=generator).to(device)
    embedding = (torch.randn(v, features, generator=generator) / 8).to(device, torch.bfloat16)
    head = (torch.randn(v, features, generator=generator) / 8).to(device, torch.bfloat16)
    covered, start = min(local, v - rank * local), rank * local
    base = torch.full((b, d, local), 1000., device=device, dtype=torch.bfloat16)
    base[:, :, :covered].copy_(full_base[:, :, start:start + covered])
    logits = torch.full((b, d + 1, local), 1000., device=device)
    logits[:, :, :covered].copy_(full_target[:, :, start:start + covered])
    local_head = torch.zeros(local, features, device=device, dtype=torch.bfloat16)
    local_head[:covered].copy_(head[start:start + covered])
    if fp8:
        from ignition.components.markov_sampling import fp8_markov_head

        local_head = fp8_markov_head(local_head)
        full_fp8 = fp8_markov_head(head)
        head = (full_fp8.values.float() * full_fp8.scale[:, None]).contiguous()
    state = DeviceRequestState.allocate(device, vocab_size=v, capacity=b)
    state.status.fill_(RUNNING)
    state.generation.fill_(7)
    state.seed.copy_(torch.arange(b, device=device, dtype=torch.int64) + 123)
    state.parameters.copy_(torch.tensor([1., 1., 0., 1., 0., 0.], device=device))
    state.top_k.fill_(-1)
    slots = torch.arange(b, device=device, dtype=torch.int32)
    epochs = torch.full((b,), 7, device=device, dtype=torch.int64)
    live = torch.tensor([b], device=device, dtype=torch.int32)
    tokens = torch.zeros(b, d, device=device, dtype=torch.int64)
    q = torch.empty(b, d, local, device=device)
    p = torch.empty(b, d + 1, local, device=device)
    qvalid = torch.empty(b, d, device=device, dtype=torch.bool)
    pvalid = torch.empty(b, d + 1, device=device, dtype=torch.bool)
    seeds = state.seed.view(torch.int64) ^ DRAFT_RNG_DOMAIN
    offsets = torch.full((b,), 2**32, device=device, dtype=torch.int64)
    pending = torch.zeros(b, device=device, dtype=torch.int64)
    rng = SpeculativeRngState.allocate(device, capacity=b)
    rng.generation.fill_(7)
    graph = worker = None
    try:
        with RecordedPreparationSession(device=device, checkpoint="vocabulary-block-test",
                workload_schema="summary-shards-v1", numerical_policy="fp32", coordinator=coordinator) as session:
            options = dict(start=start, runtime=runtime, session=session, coordinator=coordinator)
            proposal = PreparedVocabularySoftmax(base, q, qvalid, state, slots, epochs, live, tokens,
                **options, name="proposal", embedding=embedding, head=local_head, pending=pending,
                seeds=seeds, offsets=offsets, tokens=tokens)
            target = PreparedVocabularySoftmax(logits, p, pvalid, state, slots, epochs, live, tokens,
                **options, name="target")
            verify = PreparedVocabularyBlockVerification(q, tokens, p, state, slots, epochs, live,
                **options, name="verify", rng=rng, draft_valid=qvalid, target_valid=pvalid)
            proposal.prepare()
            target.prepare()
            verify.prepare()
            hashes = [None, None]
            dist.all_gather_object(hashes, [step[0].compiled.hash for step in proposal.steps]
                                   + [target.steps[0][0].compiled.hash, verify.kernels[2].compiled.hash])
            assert hashes[0] == hashes[1]
            session.tune(show_progress=False)

            class Pipeline:
                def __call__(self):
                    proposal()
                    target()
                    verify()

            call = compile_forward(Pipeline())
            call()
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph, stream=stream):
                call()
            bundle = PreparedGraphBundle(device=rank, graphs={"sampling": graph}, buffers={},
                                         resources=(proposal, target, verify, call, session, runtime))
            worker = bundle.start_worker()
            for iteration in range(4):
                state.status[1] = 0 if iteration == 2 else RUNNING
                offsets.add_(8)
                rng.target_offset.add_(verify.draw_reservation)
                q.fill_(float("nan"))
                p.fill_(float("nan"))
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
                dist.barrier()
                ticket = worker.submit(0)
                result = worker.poll(10000)
                assert result is not None and result.error is None and result.ticket == ticket
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
                assert verify.output.valid.tolist() == [True, iteration != 2]
                expected_p = full_target.double().softmax(-1)
                torch.testing.assert_close(p[0, :, :covered].double(), expected_p[0, :, start:start + covered],
                                           rtol=3e-6, atol=1e-9)
                previous = pending
                for step in range(d):
                    score = full_base[:, step].double() + embedding[previous].double() @ head.double().T
                    expected_q = score.softmax(-1)
                    torch.testing.assert_close(q[0, step, :covered].double(), expected_q[0, start:start + covered],
                                               rtol=3e-5, atol=1e-8)
                    previous = tokens[:, step]
                peers = [None, None]
                dist.all_gather_object(peers, verify.output.token_ids.cpu().tolist())
                assert peers[0] == peers[1]
            dist.barrier()
    finally:
        if worker is not None:
            worker.close()
        if graph is not None:
            graph.reset()
        runtime.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("fp8", [False, True])
def test_two_rank_vocabulary_sampling_native_replay(tmp_path, fp8):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two explicitly assigned CUDA devices")
    mp.spawn(_rank, args=((tmp_path / "rendezvous").as_uri(), fp8), nprocs=2, join=True)
