"""Known proposal support preserves dense probabilities and native sampling."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (os.environ.get("FLARE_RUN_CUDA_TESTS") and torch.cuda.is_available()),
    reason="set FLARE_RUN_CUDA_TESTS=1 on a CUDA host",
)


@pytest.mark.parametrize("batch,vocab,prefix,compiled", [
    (1, 248320, 14, True), (4, 10000, 3, True),
    (8, 257, 7, False), (1, 17, 0, False),
])
@pytest.mark.parametrize("top_k", [0, 1, 8, 20])
@pytest.mark.parametrize("support_cap", [16, 64])
@torch.no_grad()
def test_shortlist_matches_dense_native_replay(batch, vocab, prefix, compiled, top_k, support_cap):
    from ignition.components.draft_sampling import PreparedDraftDistribution
    from ignition.components.sampling import DeviceRequestState, RUNNING
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.graphs import PreparedGraphBundle

    generator = torch.Generator().manual_seed(731 + prefix)
    state = DeviceRequestState.allocate("cuda:0", vocab_size=vocab, capacity=max(4, batch))
    state.generation.fill_(7)
    state.status.fill_(RUNNING)
    state.histogram.copy_(torch.randint(0, 4, state.histogram.shape, generator=generator, dtype=torch.int32))
    state.parameters.copy_(torch.tensor([1., .9, .02, 1., 0., 0.], device="cuda"))
    state.top_k.fill_(top_k)
    state.grammar_mask.fill_(-1)
    state.grammar_enabled.zero_()
    slots = torch.arange(batch, dtype=torch.int32, device="cuda")
    epochs = torch.full((batch,), 7, dtype=torch.int64, device="cuda")
    live = torch.tensor([batch], dtype=torch.int32, device="cuda")
    prefix_ids = torch.full((batch, max(1, prefix)), 7, dtype=torch.int64, device="cuda")
    support = min(support_cap, vocab)
    storage = torch.empty((batch, 2, support), dtype=torch.int64, device="cuda")
    ids = storage[:, 1]
    for row in range(batch):
        ids[row].copy_(torch.cat((torch.tensor([0, 1]), torch.randperm(vocab - 2, generator=generator)[:support - 2] + 2)))
    values = torch.randn((batch, support), generator=generator, dtype=torch.float32).cuda()
    initial_ids, initial_values = ids.clone(), values.clone()
    logits = torch.empty((batch, 1, vocab), device="cuda")
    seed = torch.arange(batch, dtype=torch.int64, device="cuda") + 931
    offset = torch.zeros_like(seed)
    operations, calls, graphs = [], [], {}
    for sparse in (False, True):
        logits.fill_(-torch.inf)
        logits[:, 0].scatter_(1, ids, values)
        operation = PreparedDraftDistribution(logits, state, slots, epochs, live, prefix_ids,
            seed=seed, offset=offset, sampled=torch.zeros_like(seed), prefix_offset=prefix)
        if sparse:
            operation.bind_shortlist(ids)
        operation.prepare()

        def run(operation=operation):
            logits.fill_(-torch.inf)
            logits[:, 0].scatter_(1, ids, values)
            operation()

        call = compile_forward(run) if compiled else run
        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        operations.append(operation)
        calls.append(call)
        graphs[str(sparse)] = graph
    bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={},
        resources=(operations, calls, state, ids, storage, values, logits))
    worker = bundle.start_worker()
    outputs = lambda op: (op.probabilities, op.sampled, op.valid, op.maximum, op.packed_top_k, op.packed_top_p)
    expected = [torch.empty_like(tensor) for tensor in outputs(operations[0])]
    modes = ("neutral", "penalties", "grammar", "ties", "signed_zero", "tiny_temperature",
             "nonfinite", "all_masked", "penalty_overflow", "invalid_prefix", "inactive",
             "duplicate_or_stale", "rng_wrap", "mutable_ids", "neutral")
    try:
        for phase, mode in enumerate(modes):
            values.copy_(initial_values)
            ids.copy_(initial_ids)
            state.status.fill_(RUNNING)
            state.parameters.copy_(torch.tensor([1., .9, .02, 1., 0., 0.], device="cuda"))
            state.grammar_enabled.zero_()
            slots.copy_(torch.arange(batch, dtype=torch.int32, device="cuda"))
            epochs.fill_(7)
            live.fill_(batch)
            prefix_ids.fill_(7)
            expected_offset = phase * 4 + (2**32 if mode == "rng_wrap" else 0)
            offset.fill_(expected_offset)
            if mode == "penalties":
                state.parameters.copy_(torch.tensor([.7, .8, .1, 1.3, -.2, -.1], device="cuda"))
            if mode in ("grammar", "all_masked"):
                state.grammar_enabled.fill_(1)
                state.grammar_mask.copy_(torch.randint(-2**31, 2**31 - 1, state.grammar_mask.shape,
                    generator=generator, dtype=torch.int64).to(torch.int32))
                if mode == "all_masked":
                    state.grammar_mask.zero_()
            if mode in ("ties", "signed_zero"):
                values.zero_()
                values[:, 1::2] = -0. if mode == "signed_zero" else 1.
            if mode == "signed_zero":
                state.parameters[:, 4:] = -0.
            if mode == "tiny_temperature":
                state.parameters[:, 0] = torch.finfo(torch.float32).tiny
            if mode == "nonfinite":
                values[:, :3] = torch.tensor([float("nan"), float("inf"), -float("inf")], device="cuda")
            if mode == "penalty_overflow":
                state.parameters[:, 5] = -torch.finfo(torch.float32).max
            if mode == "invalid_prefix":
                prefix_ids[:, 0] = -1
            if mode == "inactive":
                live.fill_(batch // 2)
            if mode == "duplicate_or_stale":
                epochs[-1] = 99
                if batch > 1:
                    slots[-1] = slots[0]
            if mode == "mutable_ids":
                ids.copy_(initial_ids.roll(5, dims=1))
            for index, operation in enumerate(operations):
                operation.probabilities.fill_(float("nan"))
                torch.cuda.synchronize()
                allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                ticket = worker.submit(index)
                completed = worker.poll(timeout_ms=30000)
                assert completed is not None and completed.ticket == ticket and completed.error is None
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
                if index == 0:
                    for destination, tensor in zip(expected, outputs(operation), strict=True):
                        destination.copy_(tensor)
                else:
                    for actual, reference in zip(outputs(operation), expected, strict=True):
                        if actual is operation.maximum:
                            torch.testing.assert_close(actual, reference, rtol=0, atol=0, equal_nan=True)
                        elif actual.dtype.is_floating_point:
                            assert torch.equal(actual.view(torch.int32), reference.view(torch.int32)), mode
                        else:
                            assert torch.equal(actual, reference), mode
            assert torch.equal(offset, torch.full_like(offset, expected_offset))
    finally:
        worker.close()
        for graph in graphs.values():
            graph.reset()
