# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inkling 2 configuration validation, checkpoint tensor coverage and four-rank placement.

Checkpoint tests read `FLARE_INKLING2_SNAPSHOT` (default: the pinned revision in the
Hugging Face hub cache) and skip without it. They read safetensors headers and
metadata files only, never tensor payloads.
"""

import copy
import json
import math
import os
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import _read_header, inspect_indexed_checkpoint
from ignition.recipes.inkling2.checkpoint import CHECKPOINT
from ignition.recipes.inkling2.config import DRAFT_FILE, ModelConfig
from ignition.recipes.inkling2.weights import checkpoint_layout, stored_tensors
from tests.reference.inkling2_support import random_tensor, shard, synthetic_config

SNAPSHOT = Path(os.environ.get("FLARE_INKLING2_SNAPSHOT", Path.home() / ".cache/huggingface/hub" /
    "models--thinkingmachines--Inkling-2-Small/snapshots/b0e7afb8892a07897c78ce1c5b801a0d9cbd5e7b"))


def assemble(shards, placement):
    if placement.axis is None:
        assert all(torch.equal(part, shards[0]) for part in shards)
        return shards[0]
    return torch.cat(shards, placement.axis).narrow(placement.axis, 0, placement.shape[placement.axis])


def assert_reassembles(placement, tensor):
    parts = [shard(tensor, placement, rank) for rank in range(4)]
    for rank, part in enumerate(parts):
        assert part.shape == placement.local_shape
        if placement.axis is not None:
            first, last = placement.rows(rank)
            tail = part.narrow(placement.axis, last - first, placement.width - (last - first))
            assert not bool(tail.view(torch.uint8).any())
    rebuilt = assemble(parts, placement)
    assert rebuilt.dtype == tensor.dtype and torch.equal(rebuilt.view(torch.uint8), tensor.view(torch.uint8))


@pytest.fixture(scope="module")
def checkpoint():
    if not (SNAPSHOT / "model.safetensors.index.json").is_file() or not (SNAPSHOT / DRAFT_FILE).is_file():
        pytest.skip("Inkling 2 Small checkpoint headers are unavailable; set FLARE_INKLING2_SNAPSHOT")
    manifest = inspect_indexed_checkpoint(SNAPSHOT)
    _, draft = _read_header(SNAPSHOT / DRAFT_FILE, DRAFT_FILE)
    config = ModelConfig.load(SNAPSHOT)
    return config, manifest, draft, checkpoint_layout(config, 4)


def test_layout_maps_every_checkpoint_tensor_once(checkpoint):
    config, manifest, draft, layout = checkpoint
    layout.validate(stored_tensors(manifest.tensors.values()), stored_tensors(draft.values()))
    assert (len(layout.model), len(layout.draft)) == (1770, 84)
    roles = Counter(placement.role for placement in layout.model.values())
    assert roles["query"] == roles["output"] == roles["router_bias"] == 48
    assert roles["attention_norm"] == roles["key_convolution"] == 40
    assert roles["key_convolution_dynamic"] == roles["query_norm"] == 8
    assert roles["w13_amax"] == 40 and roles["w13_scale"] == 48 and roles["attention_query"] == 47
    precision = Counter(placement.precision for placement in layout.model.values())
    assert precision == {"bfloat16": 1273, "float32": 144, "nvfp4": 320, "mxfp8": 32, "bfloat16_float32_output": 1}
    for layer in range(48):
        assert config.expert_format(layer) == ("nvfp4" if layer < 40 else "mxfp8")
        if layer:
            query = layout.model[f"model.llm.attn_res.attn_queries.{layer - 1}"]
            assert query.layer == layer and query.shape == ((3, 4608) if layer % 6 == 5 else (4608,))


def test_checkpoint_declaration_matches_the_layout(checkpoint):
    _, manifest, draft, layout = checkpoint
    stored = sum(tensor.nbytes for tensor in manifest.tensors.values()) + sum(tensor.nbytes for tensor in draft.values())
    assert CHECKPOINT.stored_bytes == stored == 393_479_869_088
    assert CHECKPOINT.device_bytes == 4 * layout.rank_bytes()
    assert CHECKPOINT.draft_bytes == 4 * sum(placement.rank_bytes for placement in layout.draft.values())
    assert layout.rank_bytes() == 99_086_416_544


def test_real_geometry_places_heads_experts_and_vocabulary(checkpoint):
    config, _, _, layout = checkpoint
    config.validate_shard(TensorParallelShard(0, 4))
    assert config.text.global_layers == (5, 11, 17, 23, 29, 35, 41, 47)
    assert (config.text.extent(0), config.text.extent(5)) == (1024, 1024)
    p = "model.llm.layers.5"
    expected = {"attn.wq_du.weight": (0, 1536), "attn.gate_attn.weight": (0, 1536), "attn.wr_du.weight": (0, 192),
                "attn.attn_sink_proj.weight": (0, 12), "attn.q_l2_clip.scale": (0, 12),
                "attn.rel_l2_clip.scale": (0, 12), "attn.wk_dv.weight": (0, 256), "attn.wv_dv.weight": (0, 256),
                "attn.k_sconv.weight_proj": (0, 8), "attn.k_sconv.static_w": (1, 256), "attn.wo_ud.weight": (1, 1536),
                "mlp.experts.w13_weight": (1, 1152), "mlp.experts.w13_weight_scale": (1, 1152),
                "mlp.experts.w2_weight": (2, 288), "mlp.experts.w2_weight_scale": (2, 36)}
    for name, (axis, width) in expected.items():
        assert (layout.model[f"{p}.{name}"].axis, layout.model[f"{p}.{name}"].width) == (axis, width), name
    for name in ("attn.rel_logits_proj.proj", "mlp.gate.weight", "mlp.gate.bias", "attn_sconv.weight",
                 "mlp_sconv.weight", "q_input_norm.weight", "mlp.experts.w13_weight_amax",
                 "mlp.experts.w2_weight_outer_scale_denominator"):
        assert layout.model[f"{p}.{name}"].axis is None, name
    mxfp8 = "model.llm.layers.40.mlp.experts"
    assert (layout.model[f"{mxfp8}.w2_weight"].width, layout.model[f"{mxfp8}.w2_weight_scale"].width) == (576, 18)
    embedding = layout.model["model.llm.embed.weight"]
    assert embedding.width == 50304 and embedding.rows(3) == (150912, 201024)
    assert layout.model["model.llm.unembed.weight"].precision == "bfloat16_float32_output"
    assert layout.draft["markov_head.markov_w1.weight"].axis is None
    assert (layout.draft["layers.2.mlp.down_proj.weight"].axis, layout.draft["layers.2.mlp.down_proj.weight"].width) == (
        1, 4608)


def test_real_shapes_shard_and_reassemble(checkpoint):
    _, _, _, layout = checkpoint
    generator = torch.Generator().manual_seed(0)
    classes = {}
    for placement in (*layout.model.values(), *layout.draft.values()):
        classes.setdefault((placement.role, placement.shape, placement.dtype, placement.axis), placement)
    for placement in classes.values():
        dtype = DTYPES[placement.dtype]
        shape = list(placement.shape)
        # Above 64 MiB, axes that no rank partitions are shortened; the partitioned axis keeps its stored size.
        for axis in range(len(shape)):
            while axis != placement.axis and shape[axis] > 2 and math.prod(shape) * dtype.itemsize > 64 << 20:
                shape[axis] = max(2, shape[axis] // 8)
        if dtype.itemsize == 1:
            tensor = torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator).view(dtype)
        else:
            tensor = torch.randn(shape, generator=generator).to(dtype)
        assert_reassembles(replace(placement, shape=tuple(shape)), tensor)


DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn, "U8": torch.uint8}


@pytest.fixture
def synthetic():
    return synthetic_config()


def test_synthetic_layout_shards_and_reassembles_every_tensor(synthetic):
    config = ModelConfig.from_dict(*synthetic)
    layout = checkpoint_layout(config, 4)
    generator = torch.Generator().manual_seed(1)
    for placement in (*layout.model.values(), *layout.draft.values()):
        assert_reassembles(placement, random_tensor(placement, generator))
    embedding = layout.model["model.llm.embed.weight"]
    assert embedding.width == 256 and embedding.rows(3) == (768, 1000)
    whole = checkpoint_layout(config, 1)
    assert all(placement.axis is None for placement in whole.model.values())
    assert whole.rank_bytes() == sum(placement.stored_bytes for placement in
                                     (*layout.model.values(), *layout.draft.values()))


def test_interleaved_gate_up_rows_and_dynamic_lags_stay_with_their_rank(synthetic):
    config = ModelConfig.from_dict(*synthetic)
    layout = checkpoint_layout(config, 4)
    w13 = layout.model["model.llm.layers.2.mlp.experts.w13_weight"]
    rows = torch.arange(w13.shape[1])
    gate, up = rows[0::2], rows[1::2]
    width = config.text.intermediate // 4
    for rank in range(4):
        local = rows[w13.extent(rank)[1]]
        assert torch.equal(local[0::2], gate[rank * width:(rank + 1) * width])
        assert torch.equal(local[1::2], up[rank * width:(rank + 1) * width])
    lags = layout.model["model.llm.layers.5.attn.k_sconv.weight_proj"]
    head_rows = torch.arange(lags.shape[0]).reshape(config.text.kv_heads, config.text.convolution_width)
    keys = layout.model["model.llm.layers.5.attn.wk_dv.weight"]
    for rank in range(4):
        heads = torch.arange(*keys.rows(rank))[::config.text.head_dim] // config.text.head_dim
        assert torch.equal(torch.arange(*lags.rows(rank)), head_rows[heads].flatten())


def test_layout_rejects_stored_tensors_it_does_not_map(synthetic):
    config = ModelConfig.from_dict(*synthetic)
    layout = checkpoint_layout(config, 4)
    stored = {name: (placement.shape, placement.dtype) for name, placement in layout.model.items()}
    layout.validate(stored)
    for change in ({"model.llm.extra.weight": ((1,), "BF16")}, {"model.llm.norm.weight": ((256,), "F32")},
                   {"model.llm.layers.0.mlp.experts.w2_weight": ((16, 256, 256), "U8")}):
        with pytest.raises(ValueError):
            layout.validate({**stored, **change})
    with pytest.raises(ValueError, match="missing"):
        layout.validate({name: value for name, value in stored.items() if name != "model.llm.norm.weight"})


def _edit(path, value):
    def apply(config, quantization):
        target = {"config": config, "quantization": quantization}
        *keys, last = path
        for key in keys:
            target = target[key]
        if value is _DELETE:
            del target[last]
        else:
            target[last] = value
    return apply


_DELETE = object()


@pytest.mark.parametrize("edit,message", [
    (_edit(("config", "text_config", "q_bias"), True), "q_bias"),
    (_edit(("config", "text_config", "pre_norm_pre_affine"), True), "pre_norm_pre_affine"),
    (_edit(("config", "text_config", "use_value_norm_no_gamma"), False), "use_value_norm_no_gamma"),
    (_edit(("config", "text_config", "attn_res_normalize_values"), False), "attn_res_normalize_values"),
    (_edit(("config", "text_config", "n_shared_experts"), 1), "shared experts"),
    (_edit(("config", "text_config", "num_experts_per_tok"), 6), "top-8"),
    (_edit(("config", "text_config", "swa_num_key_value_heads"), 2), "sliding-window"),
    (_edit(("config", "text_config", "sconv_kernel_size"), 1), "sconv_kernel_size"),
    (_edit(("config", "text_config", "local_layer_ids"), [0, 0, 1]), "local_layer_ids"),
    (_edit(("config", "text_config", "rope_theta"), 1e6), "unsupported fields"),
    (_edit(("config", "text_config", "route_scale"), _DELETE), "missing route_scale"),
    (_edit(("config", "architectures"), ["Inkling1ForCausalLM"]), "Inkling2ForConditionalGeneration"),
    (_edit(("config", "vision_config", "projector_bias"), True), "projector_bias"),
    (_edit(("config", "mtp_config", "markov_head_type"), "vanilla"), "rnn_hidden"),
    (_edit(("config", "mtp_config", "target_layer_ids"), [1, 5]), "target_layer_ids"),
    (_edit(("config", "mtp_config", "torch_dtype"), "float16"), "bfloat16"),
    (_edit(("quantization", "model", "parameter_groups", 2, "required_activation_quantization",
            "scale_selection_rule"), "max"), "four-over-six"),
    (_edit(("quantization", "model", "parameter_groups", 3, "weight_transform"), "mxfp8_floor"), "MXFP8"),
    (_edit(("quantization", "quant_method"), "modelopt"), "inkling2"),
])
def test_configuration_rejects_values_the_recipe_does_not_implement(synthetic, edit, message):
    config, quantization = copy.deepcopy(synthetic)
    edit(config, quantization)
    with pytest.raises(ValueError, match=message):
        ModelConfig.from_dict(config, quantization)


def test_precision_groups_must_cover_each_tensor_once(synthetic):
    config, quantization = copy.deepcopy(synthetic)
    groups = quantization["model"]["parameter_groups"]
    nvfp4, mxfp8 = groups[2]["tensors"], groups[3]["tensors"]
    moved = "model.llm.layers.0.mlp.experts.w2_weight"
    nvfp4.remove(moved)
    mxfp8.append(moved)
    with pytest.raises(ValueError, match="layer 0 experts"):
        ModelConfig.from_dict(config, quantization)
    config, quantization = copy.deepcopy(synthetic)
    quantization["model"]["parameter_groups"][0]["tensors"].append("model.llm.norm.weight")
    with pytest.raises(ValueError, match="more than one"):
        ModelConfig.from_dict(config, quantization)
    config, quantization = copy.deepcopy(synthetic)
    quantization["model"]["parameter_groups"][1]["tensors"].remove("model.llm.norm.weight")
    quantization["model"]["parameter_groups"][0]["tensors"].append("model.llm.norm.weight")
    with pytest.raises(ValueError, match="model.llm.norm.weight"):
        checkpoint_layout(ModelConfig.from_dict(config, quantization), 4)


def test_four_ranks_are_required(synthetic):
    config = ModelConfig.from_dict(*synthetic)
    config.validate_shard(TensorParallelShard(3, 4))
    with pytest.raises(ValueError, match="tensor parallelism of 4"):
        config.validate_shard(TensorParallelShard(0, 2))


def test_checkpoint_declaration_imports_no_model_code():
    result = subprocess.run([sys.executable, "-c", "import sys; import ignition.recipes.inkling2.checkpoint; "
                             "print(sorted(m for m in ('torch', 'ignition.recipes.inkling2.weights', "
                             "'ignition.recipes.inkling2.config') if m in sys.modules))"],
                            capture_output=True, text=True, check=True, cwd=Path(__file__).parents[2])
    assert json.loads(result.stdout.replace("'", '"')) == []
