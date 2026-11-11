# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Check real-model scoring against an independent head and FP64 softmax."""

import json
import os
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="CUDA opt-in")


def test_real_qwen_prompt_head_and_reduction():
    from tokenizers import Tokenizer
    from ignition.preparation.core_storage import CoreStorage
    from ignition.recipes.qwen38.bootstrap import PreparationConfig, prepare_model
    from ignition.recipes.qwen38.decoder import DecoderScratch, ModelWeights
    from ignition.recipes.qwen38.decoder_state import DecoderInputs
    from ignition.recipes.qwen38.sampling import DeviceRequestState, RequestAdmission
    from ignition.recipes.qwen38.state import HybridStatePool
    from ignition.recipes.qwen38.ple import PLETable, ROWS_PER_PLANE
    from ignition.components.prompt_logprobs import PromptScoring

    snapshot = Path(os.environ.get("FLARE_QWEN38_SNAPSHOT", "/data/checkpoint-releases/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD"))
    if not snapshot.is_dir():
        pytest.skip("exact Qwen checkpoint is unavailable")
    tokens = Tokenizer.from_file(str(snapshot / "tokenizer.json")).encode(
        "The capital of France is Paris. The city is known for its art, architecture, museums and history.",
        add_special_tokens=False).ids
    config = PreparationConfig(prompt_logprobs=True, max_active=1, context=128, prefill_chunk_size=128,
        pages=4, snapshot_budget_bytes=256 << 20, compile_workers=12,
        kernel_cache=Path.home() / ".cache/flare/kernels-qwen")
    with CoreStorage(world=1) as storage:
        with prepare_model(snapshot, config, compile_decoder=False, core_storage=storage,
                           progress=lambda message: print(message, flush=True)) as model:
            def owner(kind):
                return next(value for value in model.resources if isinstance(value, kind))
            weights, scratch = owner(ModelWeights), owner(DecoderScratch)
            state, requests, inputs = owner(HybridStatePool), owner(DeviceRequestState), owner(DecoderInputs)
            scoring = owner(PromptScoring)
            state.reset_for_preparation()
            requests.admit(0, RequestAdmission(1, 1, max_new_tokens=1, top_k=1), tokens)
            inputs.request_state_rows.zero_()
            inputs.request_state_rows[0, 0] = 1
            inputs.row_slot_ids.fill_(-1)
            inputs.row_slot_ids[0] = 0
            inputs.row_generations[0] = 1
            inputs.live_rows.fill_(1)
            inputs.query_lengths[0] = len(tokens)
            inputs.sample_enabled.zero_()
            inputs.token_ids[:len(tokens)] = torch.tensor(tokens, device="cuda")
            inputs.prefill_positions[:len(tokens)] = torch.arange(len(tokens), device="cuda")[:, None].expand(-1, 3)
            inputs.vision_token_indices.fill_(-1)
            inputs.page_tables[0, :2] = torch.tensor([1, 2], device="cuda")
            scoring.outputs.targets.fill_(-1)
            scoring.outputs.targets[:len(tokens) - 1] = torch.tensor(tokens[1:], device="cuda", dtype=torch.int32)
            model.graphs["scored_prefill128"].replay()
            torch.cuda.synchronize()
            # The diagnostic replay stages the same compact PLE rows native serving delivers.
            table = owner(PLETable)
            ids = table.hashes.cpu().flatten().tolist()
            with ExitStack() as files:
                opened = {}
                for destination, sources, width in ((table.weights, table.sources[0], 80),
                                                    (table.scales, table.sources[1], 10)):
                    packed = bytearray(len(ids) * width)
                    for index, row in enumerate(ids):
                        if row < 0:
                            continue
                        plane, local = divmod(row, ROWS_PER_PLANE)
                        source = sources[plane]
                        if source.path not in opened:
                            opened[source.path] = files.enter_context(source.path.open("rb", buffering=0))
                        file = opened[source.path]
                        file.seek(source.offset + local * width)
                        value = file.read(width)
                        assert len(value) == width
                        packed[index * width:(index + 1) * width] = value
                    destination[:len(ids)].copy_(torch.frombuffer(packed, dtype=torch.uint8).view(len(ids), width))
            table.ready.record()
            state.reset_for_preparation()
            requests.admit(0, RequestAdmission(2, 1, max_new_tokens=1, top_k=1), tokens)
            inputs.row_generations[0] = 2
            model.graphs["scored_prefill128"].replay()
            torch.cuda.synchronize()
            actual = scoring.outputs.values[int(scoring.frame.item()), :len(tokens) - 1].cpu()
            precision = torch.get_float32_matmul_precision()
            try:
                torch.set_float32_matmul_precision("highest")
                logits = torch.nn.functional.linear(scratch.hidden[:len(tokens) - 1].float(),
                    weights.tree.lm_head.weight.float()).bfloat16().cpu()
            finally:
                torch.set_float32_matmul_precision(precision)
            reference = logits.double().log_softmax(-1)
            errors = []
            for row, token in enumerate(tokens[1:]):
                assert actual[row, 0] == token
                error = abs(actual[row, 1].view(torch.float32).item() - reference[row, token].item())
                errors.append(error)
                assert error < 3e-4
                reported = logits[row].float() - torch.logsumexp(logits[row].float(), 0)
                assert actual[row, 2].item() == 1 + int((reported > reported[token]).sum())
                expected = sorted(range(logits.shape[1]), key=lambda i: (-float(reported[i]), i))[:20]
                assert actual[row, 3::2].tolist() == expected
            print(json.dumps({"positions":len(errors), "max_absolute_error":max(errors),
                "mean_absolute_error":sum(errors) / len(errors),
                "scoring_pool_bytes":scoring.outputs.scratch.nbytes}), flush=True)
