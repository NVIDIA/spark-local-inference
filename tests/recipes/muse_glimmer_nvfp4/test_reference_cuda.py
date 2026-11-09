# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared Muse Glimmer prefill and decode against the layer-streamed NVFP4 text reference.

The traced prompt is a chat-templated conversation tokenized by the checkpoint tokenizer and
long enough to cross the 2048-token sliding window, so sliding (rope) and full (NoPE) layers
are compared on both sides of the boundary. Prefill rows come from a forward hook on the last
prepared layer through the decoder's norm and head arithmetic; the sampled row and the decode
row come from the decoder's own head. The reference rounds keys and values to E4M3 at the
cache seam (``Fp8History``) as the prepared rings and pages do, so what remains is NVFP4 kernel
versus FP32-decode arithmetic and BF16 reduction order over 52 layers.

Tolerance (relative L2, reference norm as denominator, docs/core-design.md): at most 1e-1 on the
softcapped logits of every compared row group, top-1 agreement of at least 0.98 over the prefill
rows, the reference's decode argmax within the prepared top-3, and hidden states calibrated
against the reference itself: the prepared decoder may sit no farther from the FP8-history
reference than ``CALIBRATION`` times the distance a single rounding source (BF16 histories
instead of E4M3) moves that reference, per layer on the residual stream and on the final
normalized hidden states. This model's residual stream drifts in direction by 3-7% per layer
under any one rounding (measured on the reference alone; norm ratio stays at 1.000), growing
roughly linearly, while the head reads almost none of it; a wiring defect flips far more than
2% of teacher-forced rows. Diagnostics printed for localization: per-layer residual relative L2
of the prepared decoder and of the BF16-history reference against the FP8-history reference,
per-row median/p95/max (with the row of the max, row 0 alone and the median norm ratio) for
layers 0, 1, 2, 13, 26, 51 and the final hidden, and raw pre-softcap logits for the sampled
and decode rows.
"""

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

CUDA = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                          reason="requires explicit GPU assignment and FLARE_RUN_CUDA_TESTS=1")
SNAPSHOT = os.environ.get("FLARE_MUSE_GLIMMER_SNAPSHOT")
CHECKPOINT = pytest.mark.skipif(not SNAPSHOT, reason="FLARE_MUSE_GLIMMER_SNAPSHOT names no checkpoint")
WINDOW = 2048
PROMPT_MINIMUM = WINDOW + 200
RELATIVE_L2 = .1
TOP1_AGREEMENT = .98
CALIBRATION = 2.
DIAGNOSTIC_LAYERS = (0, 1, 2, 13, 26, 51)


def device():
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("assign a CUDA device explicitly")
    selected = torch.device("cuda", 0)
    torch.cuda.set_device(selected)
    torch.backends.cuda.matmul.allow_tf32 = False
    return selected


def traced_prompt(snapshot, minimum=PROMPT_MINIMUM):
    """Token ids of a chat-templated prompt with at least ``minimum`` tokens, from the checkpoint tokenizer."""
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(snapshot)
    template = (Path(snapshot) / "chat_template.jinja").read_text()
    lines = 64
    while True:
        passage = "\n".join(f"Entry {i}: the reading of gauge {i} was {i * 7 % 101} units." for i in range(lines))
        messages = [{"role": "system", "content": "You are a careful assistant."},
                    {"role": "user", "content": passage + "\n\nWhich entry has the largest reading?"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, chat_template=template,
                                             add_generation_prompt=True, current_date="2026-11-06")
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) >= minimum:
            return ids
        lines += 16


def relative_l2(actual, expected):
    return float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12))


def row_statistics(actual, expected):
    """Per-row relative L2 spread and the median per-row norm ratio (1 with large error = direction drift)."""
    norms = expected.float().norm(dim=-1).clamp_min(1e-12)
    errors = (actual.float() - expected.float()).norm(dim=-1) / norms
    ratio = actual.float().norm(dim=-1) / norms
    return dict(median=round(float(errors.median()), 4), p95=round(float(errors.quantile(.95)), 4),
                max=round(float(errors.max()), 4), argmax=int(errors.argmax()), row0=round(float(errors[0]), 4),
                norm_ratio_median=round(float(ratio.median()), 4))


def softcapped(hidden, head, config):
    logits = F.linear(hidden, head).float() * config["output_multiplier"]
    return torch.tanh(logits / config["final_logit_softcapping"]) * config["final_logit_softcapping"]


def prefill(weights, pool, ids, selected, *, blocks, session, workspace, name):
    """Construct the prepared prefill of ``ids`` on state row 1 over pages ``0..blocks-1``."""
    from ignition.components.paged_batch import Metadata
    from ignition.recipes.muse_glimmer_nvfp4.decoder import PreparedDecoder, Scratch

    tokens = len(ids)
    metadata = Metadata.allocate(tokens, 1, blocks, selected)
    metadata.set(torch.tensor([0], device=selected), torch.tensor([tokens], dtype=torch.int32, device=selected),
                 torch.tensor([1], device=selected), torch.arange(blocks, dtype=torch.int32, device=selected)[None],
                 torch.tensor(ids, device=selected).view(1, -1), torch.tensor([1], dtype=torch.int32, device=selected))
    # ``Metadata.set`` samples each request's first row; serving (``PreparedBatch.begin``) samples its last.
    metadata.sample_indices.fill_(tokens - 1)
    scratch = Scratch.allocate(weights.config, weights.shard, tokens, 1, selected)
    decoder = PreparedDecoder(weights, scratch, metadata, pool, mode="prefill", runtime=None, embedding_runtime=None,
                              vocabulary_runtime=None, coordinator=None, session=session, name=name, workspace=workspace)
    return decoder, metadata


@CUDA
@CHECKPOINT
@torch.no_grad()
def test_prepared_prefill_and_decode_match_the_reference_across_the_window():
    from b12x.preparation import PreparationSession
    from ignition.components.paged_batch import Metadata
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.muse_glimmer_nvfp4.decoder import PreparedDecoder, Scratch
    from ignition.recipes.muse_glimmer_nvfp4.weights import ModelWeights
    from ignition.recipes.muse_spark.pool import PAGE_TOKENS, ContinuationPool
    from ignition.reference.muse_glimmer import Fp8History, text_logits

    snapshot, selected = Path(SNAPSHOT), device()
    ids = traced_prompt(snapshot)
    tokens = len(ids)
    assert tokens > WINDOW
    shard = TensorParallelShard(0, 1)
    weights = ModelWeights.load(inspect_indexed_checkpoint(snapshot), selected, shard=shard)
    c = weights.config.text
    blocks = (tokens + 1 + PAGE_TOKENS - 1) // PAGE_TOKENS
    pool = ContinuationPool.allocate(weights.config, shard, 3, blocks, selected, dflash=False)
    with PreparationSession(device=selected, autotune=False) as session:
        arena = ScratchArena(selected)
        decoder, _ = prefill(weights, pool, ids, selected, blocks=blocks, session=session, workspace=arena,
                             name="glimmer.reference.prefill")
        decode_metadata = Metadata.allocate(1, 1, blocks, selected)
        decode = PreparedDecoder(weights, Scratch.allocate(weights.config, shard, 1, 1, selected), decode_metadata, pool,
                                 mode="decode", runtime=None, embedding_runtime=None, vocabulary_runtime=None,
                                 coordinator=None, session=session, name="glimmer.reference.decode", workspace=arena)
        arena.finalize()
        pool.clear_for_preparation()
        residuals = {}
        handles = [layer.register_forward_hook(
            lambda module, args, output, index=index: residuals.__setitem__(index, output.clone()))
            for index, layer in enumerate(decoder.layers)]
        decoder()
        torch.cuda.synchronize()
        for handle in handles:
            handle.remove()
        prefill_hidden = decoder.norm(residuals[len(decoder.layers) - 1])
        prefill_logits = softcapped(prefill_hidden, weights.head, c)
        sampled_hidden, sampled_logits = decoder.scratch.hidden[0].clone(), decoder.scratch.logits[0].clone()
        next_token = int(sampled_logits.argmax())
        decode_metadata.set(torch.tensor([tokens], device=selected), torch.tensor([1], dtype=torch.int32, device=selected),
                            torch.tensor([1], device=selected), torch.arange(blocks, dtype=torch.int32, device=selected)[None],
                            torch.tensor([[next_token]], device=selected), torch.tensor([1], dtype=torch.int32, device=selected))
        decode()
        torch.cuda.synchronize()
        decode_hidden, decode_logits = decode.scratch.hidden[0].clone(), decode.scratch.logits[0].clone()
    assert torch.isfinite(prefill_hidden).all() and torch.isfinite(prefill_logits).all()
    assert prefill_logits.abs().max() <= c["final_logit_softcapping"]

    # Residual stream after each layer, prompt rows: prepared decoder and BF16-history reference, both
    # against the FP8-history reference. The second distance is the yardstick one rounding source sets.
    references, layer_errors, layer_rounding, layer_rows = {}, [], [], {}

    def localize(index, reference_hidden):
        references[index] = reference_hidden[0, :tokens].clone()
        prepared = residuals.pop(index)
        layer_errors.append(round(relative_l2(prepared, references[index]), 4))
        if index in DIAGNOSTIC_LAYERS:
            layer_rows[index] = dict(prepared=row_statistics(prepared, references[index]))

    def calibrate(index, reference_hidden):
        layer_rounding.append(round(relative_l2(reference_hidden[0, :tokens], references[index]), 4))
        if index in DIAGNOSTIC_LAYERS:
            layer_rows[index]["bf16_history"] = row_statistics(reference_hidden[0, :tokens], references[index])

    hidden, logits = text_logits(snapshot, ids + [next_token], device=str(selected), history=Fp8History(),
                                 progress=localize)
    hidden, logits = hidden.to(selected), logits.to(selected)
    assert hidden.shape == (tokens + 1, c["hidden_size"]) and logits.shape == (tokens + 1, c["vocab_size"])
    assert len(layer_errors) == len(decoder.layers) and not residuals
    rounding_hidden, rounding_logits = text_logits(snapshot, ids + [next_token], device=str(selected), progress=calibrate)
    rounding_hidden, rounding_logits = rounding_hidden.to(selected), rounding_logits.to(selected)
    del references
    report = {}
    for name, span in (("within_window", slice(0, WINDOW)), ("past_window", slice(WINDOW, tokens))):
        report[name] = dict(rows=span.stop - span.start,
            hidden=relative_l2(prefill_hidden[span], hidden[span]),
            rounding_hidden=relative_l2(rounding_hidden[span], hidden[span]),
            logits=relative_l2(prefill_logits[span], logits[span]),
            rounding_logits=relative_l2(rounding_logits[span], logits[span]),
            top1=float((prefill_logits[span].argmax(-1) == logits[span].argmax(-1)).float().mean()),
            rows_prepared=row_statistics(prefill_hidden[span], hidden[span]),
            rows_bf16_history=row_statistics(rounding_hidden[span], hidden[span]))
    for name, actual_hidden, actual_logits, row in (("sampled", sampled_hidden, sampled_logits, tokens - 1),
                                                     ("decode", decode_hidden, decode_logits, tokens)):
        report[name] = dict(hidden=relative_l2(actual_hidden, hidden[row]),
            rounding_hidden=relative_l2(rounding_hidden[row], hidden[row]),
            logits=relative_l2(actual_logits, logits[row]),
            raw_logits=relative_l2(F.linear(actual_hidden, weights.head), F.linear(hidden[row], weights.head)),
            norm_ratio=float(actual_hidden.float().norm() / hidden[row].float().norm()))
    decode_scores = decode_logits.float()
    report["decode"]["reference_argmax_rank"] = int((decode_scores > decode_scores[logits[tokens].argmax()]).sum())
    print(dict(tokens=tokens, next_token=next_token, relative_l2_bound=RELATIVE_L2, top1_bound=TOP1_AGREEMENT,
               calibration=CALIBRATION, **report, layer_relative_l2=layer_errors, layer_rounding_relative_l2=layer_rounding,
               layer_rows=layer_rows), flush=True)
    for index, (prepared, rounding) in enumerate(zip(layer_errors, layer_rounding, strict=True)):
        assert prepared <= CALIBRATION * rounding, (index, prepared, rounding)
    for name, errors in report.items():
        assert errors["hidden"] <= CALIBRATION * errors["rounding_hidden"], (name, errors)
        assert errors["logits"] <= RELATIVE_L2, (name, errors)
    assert report["within_window"]["top1"] >= TOP1_AGREEMENT and report["past_window"]["top1"] >= TOP1_AGREEMENT, report
    assert report["decode"]["reference_argmax_rank"] < 3, report["decode"]
