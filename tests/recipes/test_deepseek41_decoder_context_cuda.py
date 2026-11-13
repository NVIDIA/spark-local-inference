# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compacted target feedback preserves draft windows across accepted-prefix transitions."""

from copy import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ignition.components.dspark import PreparedContext
from ignition.components.paged_batch import Metadata
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.deepseek41.decoder_prefill import PreparedDecoderSuffix
from ignition.recipes.deepseek41.pool import ContinuationPool
from ignition.recipes.deepseek41.recipe import RECIPE
from ignition.recipes.deepseek41.weights import DeepSeekWeightLoader

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@torch.no_grad()
def test_compacted_prefill_feedback_transitions_to_accepted_verification_under_native_replay():
    checkpoint = RECIPE.checkpoints[0]
    snapshot = Path.home() / ".cache/huggingface/hub" / (
        "models--" + checkpoint.repository.replace("/", "--")) / "snapshots" / checkpoint.revision
    snapshot = Path(os.environ.get("FLARE_DS41_SNAPSHOT", snapshot))
    if not (snapshot / "model.safetensors.index.json").is_file():
        pytest.skip("DeepSeek V4.1 Flash checkpoint is unavailable")
    device = torch.device("cuda:0")
    torch.manual_seed(410638)
    prefixes = ("mtp.0.main_proj", "mtp.0.main_norm", *(f"mtp.{i}.attn.{part}"
                for i in range(3) for part in ("wkv", "kv_norm")))
    with DeepSeekWeightLoader(inspect_indexed_checkpoint(snapshot), device,
            shard=TensorParallelShard(0, 4), prefixes=prefixes, dspark=True) as loader:
        config = loader.config
        draft = SimpleNamespace(main_projection=loader.load_linear("mtp.0.main_proj"),
            context_kv=loader.load_linear_group(tuple(f"mtp.{i}.attn.wkv" for i in range(3))),
            main_norm=loader.load_tensor("mtp.0.main_norm.weight"), layers=tuple(
                SimpleNamespace(attention={"kv_norm": loader.load_tensor(f"mtp.{i}.attn.kv_norm.weight")}) for i in range(3)))
        separate = [loader.load_linear(f"mtp.{i}.attn.wkv").weight for i in range(3)]
        packed = draft.context_kv.weight
        assert torch.equal(packed.values.view(torch.uint8),
                           torch.cat([weight.values.view(torch.uint8) for weight in separate]))
        assert torch.equal(packed.scale_rows[0].view(torch.uint8),
                           torch.cat([weight.scale_rows[0].view(torch.uint8) for weight in separate]))
    weights = SimpleNamespace(config=config, draft=draft)
    pool = ContinuationPool.allocate(config, 4, 2, device, dspark=True)
    metadata = Metadata.allocate(4096, 1, 1, device)
    packing = PreparedDecoderSuffix(metadata).functional()
    hidden = config.text["hidden_size"]
    source = torch.randn((3, 4096, 4, hidden), dtype=torch.bfloat16, device=device)
    suffix_positions = torch.empty(128, dtype=torch.int64, device=device)
    feedback = torch.empty((128, 3 * hidden), dtype=torch.bfloat16, device=device)
    verification_feedback = torch.randn((8, 3 * hidden), dtype=torch.bfloat16, device=device)
    verification = Metadata.allocate(8, 1, 1, device)
    reference_prefill = Metadata.allocate(128, 1, 1, device)
    reference_verify = Metadata.allocate(8, 1, 1, device)
    prefill_counts = torch.zeros(1, dtype=torch.int32, device=device)
    reference_prefill_counts = torch.zeros_like(prefill_counts)
    accepted = torch.zeros_like(prefill_counts)
    arena = ScratchArena(device)
    graphs, owners = {}, []
    with RecordedPreparationSession(device=device, checkpoint=checkpoint.revision,
            workload_schema="deepseek41-decoder-feedback", numerical_policy="checkpoint") as session:
        session.register_scratch(arena)
        context = PreparedContext(weights, feedback, metadata, pool, session=session,
            name="prefill_context", workspace=arena, compact_prefill=True)
        verify_context = PreparedContext(weights, verification_feedback, verification, pool,
            session=session, name="verify_context", workspace=arena)
        # Reuse the exact numerical plans and scratch. Only row coordinates differ
        # in the independently constructed reference, so state bytes are comparable.
        prefill_reference = copy(context)
        prefill_reference.compact = False
        verify_reference = copy(verify_context)
        session.tune()

        def prefill():
            suffix = packing.begin(metadata)
            suffix_positions.copy_(suffix.metadata.positions)
            for column in range(3):
                gathered = suffix.gather(source[column])
                feedback[:, column * hidden:(column + 1) * hidden].copy_(gathered.float().mean(1))
            context(feedback, prefill_counts, metadata)

        forwards = {
            "prefill": prefill,
            "prefill_reference": lambda: prefill_reference(feedback, reference_prefill_counts, reference_prefill),
            "verify": lambda: verify_context(verification_feedback, accepted, verification),
            "verify_reference": lambda: verify_reference(verification_feedback, accepted, reference_verify),
        }
        for name, forward in forwards.items():
            compiled = compile_forward(forward)
            compiled()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            graphs[name] = graph
            owners.append(compiled)
        session.freeze()
        bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={"source": source, "feedback": feedback},
            resources=(weights, pool, arena, session, owners, forwards))
        worker = bundle.start_worker()

        def replay(name):
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            ticket = worker.submit(bundle.graph_indices[name])
            result = worker.poll(timeout_ms=120000)
            assert result is not None and result.ticket == ticket and result.error is None
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocation.all.freed",
                        "segment.all.allocated", "segment.all.freed"):
                assert before[key] == after[key]

        def coordinates(m, start, count):
            m.starts.fill_(start)
            m.lengths.fill_(count)
            m.state_rows.fill_(2)
            m.query_start_loc.copy_(torch.tensor([0, count], device=device))
            m.positions.fill_(-1)
            m.positions[:count].copy_(torch.arange(start, start + count, device=device))
            m.sample_indices.fill_(max(0, count - 1))

        try:
            pool.backing.fill_(37)
            start = 127
            for count in (4096, 33):
                coordinates(metadata, start, count)
                prefill_counts.fill_(count)
                kept = min(count, 128)
                coordinates(reference_prefill, start + count - kept, kept)
                reference_prefill_counts.fill_(kept)
                expected_feedback = torch.zeros_like(feedback)
                for column in range(3):
                    expected_feedback[:kept, column * hidden:(column + 1) * hidden] = (
                        source[column, count - kept:count].float().mean(1))
                snapshot = pool.backing.clone()
                feedback.copy_(expected_feedback)
                replay("prefill_reference")
                expected = pool.backing.clone()
                pool.backing.copy_(snapshot)
                feedback.fill_(float("nan"))
                arena.tensor.fill_(0x5A)
                replay("prefill")
                assert torch.equal(feedback, expected_feedback)
                assert torch.equal(pool.backing, expected)
                assert torch.equal(suffix_positions, reference_prefill.positions)
                start += count
            # Counts include the pending target input: eight accepted feedback
            # rows represent that input plus at most seven accepted drafts.
            fixture = verification_feedback.clone()
            for count in (0, 1, 3, 8, 1):
                coordinates(verification, start, 8)
                coordinates(reference_verify, start, count)
                accepted.fill_(count)
                verification_feedback.copy_(fixture)
                snapshot = pool.backing.clone()
                replay("verify_reference")
                expected = pool.backing.clone()
                pool.backing.copy_(snapshot)
                arena.tensor.fill_(0xA5)
                replay("verify")
                assert torch.equal(pool.backing, expected)
                if count == 0:
                    assert torch.equal(pool.backing, snapshot)
                # Retired feedback is reusable after the commit; poisoning it
                # cannot affect the stored draft history or a later continuation.
                verification_feedback[count:].fill_(float("nan"))
                feedback.fill_(float("nan"))
                source.fill_(float("nan"))
                torch.cuda.synchronize()
                assert torch.equal(pool.backing, expected)
                start += count
        finally:
            worker.close()
            for graph in graphs.values():
                graph.reset()
