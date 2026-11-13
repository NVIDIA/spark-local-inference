# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Replicated Markov heads retain FP32 logits through compiled graph replay."""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.components.dspark import PreparedMarkov
from ignition.components.embedding import PreparedEmbedding
from ignition.components.markov_sampling import PreparedRowScaledFP8MarkovProjection, fp8_markov_head
from ignition.preparation.tuning import RecordedPreparationSession


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("rows", [1, 8, 32, 160])
@pytest.mark.parametrize("quantization", [None, "nvfp4"])
@torch.no_grad()
def test_replicated_markov_head_on_a_tensor_parallel_rank(rows, quantization):
    device = torch.device("cuda:0")
    torch.manual_seed(4101)
    head = torch.randn((129280, 256), device=device, dtype=torch.bfloat16) * .02
    if quantization:
        from ignition.components.vocabulary import nvfp4_draft_head

        # Every block has the same maximum; all entries are exactly NVFP4-representable.
        values = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=device) / 32
        head.copy_(values[torch.randint(0, 8, head.shape, device=device)])
        head[:, ::16] = 6 / 32
        packed = nvfp4_draft_head(head)
    embedding = torch.randn_like(head) * .1
    weights = SimpleNamespace(
        config=SimpleNamespace(draft=SimpleNamespace(markov_rank=256)),
        shard=SimpleNamespace(world_size=4),
        draft=SimpleNamespace(markov_head=packed if quantization else head, markov_embedding=embedding))
    tokens = torch.arange(rows, device=device, dtype=torch.int64)
    with RecordedPreparationSession(device=device, checkpoint="markov-projection",
            workload_schema="replicated-fp32", numerical_policy=quantization or "checkpoint",
            compile_workers=8) as session:
        operation = PreparedMarkov(weights, tokens, runtime=None, coordinator=None,
                                   session=session, name="markov")
        session.tune(show_progress=False)
        assert operation.gather is None
        output = torch.empty((rows, 129280), dtype=torch.float32, device=device)
        compiled = torch.compile(lambda: output.copy_(operation(tokens)), fullgraph=True)
        compiled()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        for _ in range(3):
            tokens.add_(17)
            expected = torch.nn.functional.linear(embedding[tokens].float(), head.float())
            if quantization:
                expected = expected.bfloat16().float()
            output.fill_(float("nan"))
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            assert output.dtype == torch.float32
            torch.testing.assert_close(output, expected, atol=2e-5,
                                       rtol=.008 if quantization else 2e-5)
        graph.reset()


@pytest.mark.parametrize("rows", [1, 8, 32])
@torch.no_grad()
def test_row_scaled_fp8_markov_projection_mutable_native_replay(rows):
    device = torch.device("cuda:0")
    generator = torch.Generator(device="cpu").manual_seed(4121)
    checkpoint = (torch.randn((1024, 256), generator=generator) / 16).to(device, torch.bfloat16)
    weight = fp8_markov_head(checkpoint)
    embedding_weight = (torch.randn((64, 256), generator=generator) / 8).to(device, torch.bfloat16)
    ids = torch.arange(rows, device=device, dtype=torch.int64) % 64
    source = torch.zeros((rows, 256), device=device, dtype=torch.bfloat16)
    output = torch.empty((rows, 1024), device=device, dtype=torch.float32)
    with RecordedPreparationSession(device=device, checkpoint="markov-fp8-projection",
            workload_schema="row-scaled-fp32", numerical_policy="quantized-reconstruction",
            compile_workers=2) as session:
        embedding = PreparedEmbedding(embedding_weight, ids, source, session=session, name="markov.embed").functional(session=session, dynamic_inputs=True)
        projection = PreparedRowScaledFP8MarkovProjection(source, weight, output,
            session=session, name="markov").functional(session=session)
        session.tune(show_progress=False)

        class Forward:
            def __call__(self):
                values = embedding(token_ids=ids)
                source.copy_(values)
                output.copy_(projection(values))
                return output

        compiled = torch.compile(Forward(), fullgraph=True)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        u = 2.0**-24
        gamma = 255 * u / (1 - 255 * u)
        values = weight.values.double()
        scale = weight.scale.double()
        for turn in range(3):
            ids.add_(17).remainder_(64)
            output.fill_(float("nan"))
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocation.all.freed",
                        "segment.all.allocated", "segment.all.freed"):
                assert before[key] == after[key]
            assert torch.equal(source, embedding_weight[ids])
            exact_dot = source.double() @ values.T
            exact = exact_dot * scale
            absolute = (source.double().abs() @ values.abs().T) * scale.abs()
            bound = gamma * absolute + u * exact.abs()
            assert ((output.double() - exact).abs() <= bound + 2.0**-149).all()
        graph.reset()


@torch.no_grad()
def test_replicated_row_scaled_fp8_markov_wiring_native_replay():
    device = torch.device("cuda:0")
    generator = torch.Generator(device="cpu").manual_seed(4122)
    checkpoint = (torch.randn((129280, 256), generator=generator) / 16).to(device, torch.bfloat16)
    packed = fp8_markov_head(checkpoint)
    embedding = (torch.randn((64, 256), generator=generator) / 8).to(device, torch.bfloat16)
    tokens = torch.arange(8, device=device, dtype=torch.int64)
    weights = SimpleNamespace(
        config=SimpleNamespace(draft=SimpleNamespace(markov_rank=256)),
        shard=SimpleNamespace(world_size=4),
        draft=SimpleNamespace(markov_head=packed, markov_embedding=embedding),
    )
    with RecordedPreparationSession(device=device, checkpoint="markov-fp8-replicated",
            workload_schema="markov-row-scaled", numerical_policy="quantized-reconstruction",
            compile_workers=2) as session:
        operation = PreparedMarkov(weights, tokens, runtime=None, coordinator=None,
                                   session=session, name="markov")
        session.tune(show_progress=False)
        output = torch.empty((8, 129280), dtype=torch.float32, device=device)
        compiled = torch.compile(lambda: output.copy_(operation(tokens)), fullgraph=True)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        selected = torch.tensor([0, 1, 7, 127, 1024, 32768, 65535, 129279], device=device)
        values = packed.values[selected].double()
        scales = packed.scale[selected].double()
        u = 2.0**-24
        gamma = 255 * u / (1 - 255 * u)
        for _ in range(3):
            tokens.add_(13).remainder_(64)
            output.fill_(float("nan"))
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocation.all.freed",
                        "segment.all.allocated", "segment.all.freed"):
                assert before[key] == after[key]
            source = embedding[tokens].double()
            dot = source @ values.T
            exact = dot * scales
            absolute = (source.abs() @ values.abs().T) * scales.abs()
            bound = gamma * absolute + u * exact.abs()
            assert ((output[:, selected].double() - exact).abs() <= bound + 2.0**-149).all()
            assert torch.isfinite(output).all()
        graph.reset()
