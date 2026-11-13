# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded checkpoint adapters and complete canonical text-forward qualification."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ignition.preparation import inspect_qwen38_checkpoint
from ignition.preparation.qwen38 import (
    MODEL_ID,
    REVISION,
    TEXT_PREFIX,
    TRANSFORMERS_REVISION,
)
from ignition.reference.qwen38_decoder import (
    CanonicalTextReference,
    _SelectedEmbedding,
    _SelectedExperts,
    _StreamLinear,
)
from ignition.reference.qwen38_quant import (
    CheckpointTensors,
    mxfp8_linear_reference,
    nvfp4_expert_reference,
    read_nvfp4_projection,
)


@pytest.fixture(scope="module")
def manifest():
    default = Path("/data/checkpoint-releases/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD")
    snapshot = Path(os.environ.get("FLARE_QWEN38_SNAPSHOT", default))
    if not snapshot.is_dir():
        pytest.skip("exact Qwen checkpoint is unavailable")
    return inspect_qwen38_checkpoint(snapshot)


@pytest.fixture(scope="module")
def cpu_reference(manifest):
    return CanonicalTextReference(manifest)


def test_exact_model_bounded_residency_and_canonical_gdn_functions(cpu_reference):
    from transformers.models.qwen4_exp import modeling_qwen4_exp as canonical

    model = cpu_reference.model
    assert len(model.layers) == 48
    assert (
        sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )
        == 132990720
    )
    assert not any(parameter.is_meta for parameter in model.parameters())
    assert model.config.per_layer_config[1].number_of_conv_states == 3
    assert isinstance(model.embed_tokens, _SelectedEmbedding)
    assert isinstance(
        model.layers[1].ple.ple_embedding.ngram_embedding, _SelectedEmbedding
    )
    assert all(
        isinstance(layer.mlp.experts, _SelectedExperts) for layer in model.layers
    )
    namespace = model.layers[0].linear_attn.forward.__func__.__globals__
    for name in (
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    ):
        assert namespace[name] is inspect.unwrap(getattr(canonical, name))
    assert model.layers[0].linear_attn.norm.weight.dtype == torch.bfloat16
    assert model.layers[1].ple.norm_conv.weight.dtype == torch.float32
    with pytest.raises(RuntimeError, match="active forward"):
        _ = cpu_reference.reader


def test_actual_streamed_mxfp8_and_bf16_row_chunks(manifest):
    generator = torch.Generator().manual_seed(94)
    for prefix, width, row_block in (
        (f"{TEXT_PREFIX}.layers.0.linear_attn.in_proj_a", 2560, None),
        (
            f"{TEXT_PREFIX}.layers.0.attn_hyper_connection.input_mix_weight_down",
            10240,
            37,
        ),
    ):
        source = (torch.randn((3, width), generator=generator) * 0.1).bfloat16()
        with CheckpointTensors(manifest, byte_budget=32 << 20) as reader:
            owner = SimpleNamespace(manifest=manifest, reader=reader)
            operation = _StreamLinear(owner, prefix, row_block=row_block)
            actual = operation(source)
            weight = reader.read(prefix + ".weight")
            if row_block is None:
                expected = mxfp8_linear_reference(
                    source, weight, reader.read(prefix + ".weight_scale")
                ).bfloat16()
            else:
                expected = F.linear(source, weight)
            assert torch.equal(actual, expected)


def test_actual_selected_embedding_preserves_duplicates_and_order(manifest):
    ids = torch.tensor([[248044, 14, 14, 100]])
    with CheckpointTensors(manifest, byte_budget=1 << 20) as reader:
        owner = SimpleNamespace(manifest=manifest, reader=reader)
        embedding = _SelectedEmbedding(owner, f"{TEXT_PREFIX}.embed_tokens")
        actual = embedding(ids)
        assert reader.bytes_read == 3 * 2560 * 2
        expected = torch.stack(
            [
                reader.read(f"{TEXT_PREFIX}.embed_tokens.weight", rows=(i, i + 1))[0]
                for i in ids[0].tolist()
            ]
        )[None]
        assert torch.equal(actual, expected)


def test_actual_selected_experts_use_canonical_bf16_accumulation(manifest):
    prefix = f"{TEXT_PREFIX}.layers.0.mlp.experts"
    source = (
        torch.randn((3, 2560), generator=torch.Generator().manual_seed(912)) * 0.1
    ).bfloat16()
    ids = torch.tensor([[511, 0], [0, 511], [511, 0]])
    probabilities = torch.tensor(
        [[0.3, 0.7], [0.25, 0.75], [0.625, 0.375]], dtype=torch.bfloat16
    )
    with CheckpointTensors(manifest, byte_budget=32 << 20) as reader:
        owner = SimpleNamespace(manifest=manifest, reader=reader)
        actual = _SelectedExperts(owner, prefix)(source, ids, probabilities)
        expected = torch.zeros_like(source)
        for expert in (0, 511):
            projections = tuple(
                read_nvfp4_projection(reader, f"{prefix}.{expert}.{name}")
                for name in ("gate_proj", "up_proj", "down_proj")
            )
            down = nvfp4_expert_reference(source, *projections).down.bfloat16()
            weight = probabilities.gather(
                1, (ids == expert).long().argmax(-1, keepdim=True)
            )
            expected += down * weight
        assert torch.equal(actual, expected)
        assert reader.bytes_read < 16 << 20


@pytest.mark.parametrize(
    "ids",
    [
        torch.tensor([1]),
        torch.tensor([[248320]]),
        torch.tensor([[-1]]),
        torch.empty((1, 0), dtype=torch.int64),
    ],
)
def test_reference_rejects_invalid_inputs_before_checkpoint_reads(cpu_reference, ids):
    with pytest.raises(ValueError):
        cpu_reference.forward(ids)
    assert cpu_reference._reader is None


@pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires explicitly assigned GPU",
)
@pytest.mark.parametrize("case", ["short", "64_tokens"])
def test_complete_canonical_text_prefill_then_decode(manifest, case, record_property):
    from tokenizers import Tokenizer

    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    tokenizer = Tokenizer.from_file(str(manifest.snapshot / "tokenizer.json"))
    text = (
        "The capital of France is"
        if case == "short"
        else (
            "A local assistant keeps useful context between conversations. It reads code, checks assumptions, "
            "and answers questions with concrete evidence. Reliable memory lets the assistant resume work "
            "without repeating the same analysis. " * 4
        )
    )
    tokens = tokenizer.encode(text, add_special_tokens=False).ids
    if case == "64_tokens":
        tokens = tokens[:64]
        assert len(tokens) == 64
    inputs = torch.tensor([tokens], dtype=torch.int64)
    layer_values = []
    reference = CanonicalTextReference(
        manifest,
        device="cuda:0",
        progress=lambda message: print(f"{case}: {message}", flush=True),
        layer_output=lambda layer, hidden: layer_values.append((layer, hidden)),
    )
    torch.cuda.reset_peak_memory_stats()
    prefill = reference.forward(inputs, logits_to_keep=1)
    teacher_tokens = (
        [11751, 13, 561] if case == "short" else [prefill.logits[0, -1].argmax().item()]
    )
    outputs = [prefill]
    for token in teacher_tokens:
        outputs.append(
            reference.forward(
                torch.tensor([[token]]), past_key_values=outputs[-1].past_key_values
            )
        )
    assert all(output.logits.shape == (1, 1, 248320) for output in outputs)
    assert prefill.hidden_states.shape == (1, len(tokens), 2560)
    assert all(torch.isfinite(output.logits).all() for output in outputs)
    assert outputs[-1].past_key_values.get_seq_length() == len(tokens) + len(
        teacher_tokens
    )
    assert [layer for layer, _ in layer_values] == list(range(48)) * len(outputs)
    artifact = Path(f"/tmp/sparkgap-canonical-text-{case}.pt")
    torch.save(
        {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "transformers_revision": TRANSFORMERS_REVISION,
            "text": text,
            "input_ids": inputs,
            "teacher_tokens": teacher_tokens,
            "logits": [output.logits[0, -1] for output in outputs],
            "hidden": [output.hidden_states[0, -1] for output in outputs],
            "prefill_hidden": prefill.hidden_states,
            "layer_outputs": layer_values,
            "checkpoint_bytes_read": [
                output.checkpoint_bytes_read for output in outputs
            ],
            "peak_device_bytes": torch.cuda.max_memory_allocated(),
        },
        artifact,
    )
    record_property("reference_artifact", str(artifact))
    record_property("peak_device_bytes", torch.cuda.max_memory_allocated())
