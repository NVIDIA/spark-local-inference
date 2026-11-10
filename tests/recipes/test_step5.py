# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step-5 checkpoint declarations, quantization exclusions, and cache geometry."""

import copy
import json
import os
from pathlib import Path

import pytest

from ignition.components.parallel import TensorParallelShard
from ignition.recipes import resolve
from ignition.recipes.step5.config import ModelConfig
from ignition.recipes.step5.frontend import prepare_frontend
from ignition.recipes.step5.pool import continuation_layout, page_bytes


@pytest.fixture(scope="module")
def checkpoint_config():
    model = resolve("step5-preview-nvfp4")
    snapshot = Path(os.environ.get("FLARE_STEP5_SNAPSHOT", model.snapshot(Path.home() / ".cache/huggingface/hub")))
    if not (snapshot / "config.json").exists():
        pytest.skip("Step-5 checkpoint configuration is unavailable")
    return json.loads((snapshot / "config.json").read_text())


@pytest.mark.parametrize("world", [4, 8, 16])
@pytest.mark.parametrize("mtp", [False, True])
def test_declared_cache_geometry(checkpoint_config, world, mtp):
    config = ModelConfig.from_dict(checkpoint_config)
    shard = TensorParallelShard(0, world)
    config.validate_shard(shard)
    recipe = resolve("stepfun-ai/Step-5-Preview-NVFP4").recipe
    costs = recipe.costs(world, recipe.speculation if mtp else None)
    assert costs.state_row_bytes == continuation_layout(config, shard, mtp=mtp).row_bytes
    assert costs.page_bytes == page_bytes(config, shard)
    covered = []
    for rank in range(world):
        first, last, width = config.vocabulary_partition(TensorParallelShard(rank, world))
        assert width % 64 == 0
        covered.extend(range(first, last))
    assert covered == list(range(config.vocabulary))


def test_expert_precision_comes_from_metadata(checkpoint_config):
    config = ModelConfig.from_dict(checkpoint_config)
    assert config.expert_format("model.layers.3.moe") == "nvfp4"
    assert config.expert_format("model.layers.88.moe") == "bf16"
    changed = copy.deepcopy(checkpoint_config)
    changed["quantization_config"]["ignore"].append("*layers.3.moe.experts*")
    assert ModelConfig.from_dict(changed).expert_format("model.layers.3.moe") == "bf16"
    changed["quantization_config"]["ignore"].pop()
    changed["quantization_config"]["ignore"].append("*layers.3.moe.experts.0.gate_proj")
    with pytest.raises(ValueError, match="one declared weight format"):
        ModelConfig.from_dict(changed).expert_format("model.layers.3.moe")


def test_sampling_defaults():
    assert resolve("step5-preview-nvfp4").recipe.sampling_defaults.as_dict() == {
        "seed": 0, "temperature": 1., "top_p": .95, "top_k": -1, "min_p": 0., "repetition_penalty": 1.,
        "presence_penalty": 0., "frequency_penalty": 0.}


@pytest.mark.parametrize("world", [4, 8, 16])
def test_serial_expert_layers_share_scale_storage(checkpoint_config, world):
    import torch

    from ignition.recipes.step5.weights import Step5WeightLoader

    loader = Step5WeightLoader.__new__(Step5WeightLoader)
    loader.config = ModelConfig.from_dict(checkpoint_config)
    loader.shard = TensorParallelShard(0, world)
    loader.device = torch.device("meta")
    loader.expert_scale_compression = "csf"
    local = loader.config.text["moe_intermediate_size"] // world
    a4 = loader.expert_scale_scratch(local)
    a16 = loader.expert_scale_scratch((local + 127) // 128 * 128)
    for source, target, owner in zip(a4, a16, loader.scale_scratch, strict=True):
        assert source.is_contiguous() and target.is_contiguous()
        assert source.untyped_storage()._cdata == target.untyped_storage()._cdata == owner.untyped_storage()._cdata
        assert source.numel() <= target.numel() == owner.numel()


def test_generation_stops_at_chat_and_base_eos(tmp_path):
    from tokenizers import Tokenizer, models

    Tokenizer(models.WordLevel({"<base_eos>": 0, "<pad>": 1, "<|im_end|>": 2})).save(
        str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|im_end|>"}))
    assert prepare_frontend(tmp_path, eos_token_ids=[0, 1])["eos_token_ids"] == [0, 1, 2]
    assert prepare_frontend(tmp_path, eos_token_ids=[0, 2])["eos_token_ids"] == [0, 2]
    with pytest.raises(ValueError, match="EOS token IDs"):
        prepare_frontend(tmp_path, eos_token_ids=[0, 0])
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<missing>"}))
    with pytest.raises(ValueError, match="EOS marker is missing"):
        prepare_frontend(tmp_path, eos_token_ids=[0, 1])


def test_execution_declarations_use_independent_tensor_storage(checkpoint_config):
    from dataclasses import fields, is_dataclass
    import torch
    from ignition.recipes.step5.decoder import Scratch
    from ignition.recipes.step5.vision import VisionScratch

    config = ModelConfig.from_dict(checkpoint_config)
    shard = TensorParallelShard(0, 4)
    vision = VisionScratch.allocate(65536, shard, 'meta')
    text = Scratch.allocate(config, shard, 4096, 1, 1048576, 'meta')
    decode = Scratch.allocate(config, shard, 16, 16, 1048576, 'meta')
    draft = Scratch.allocate(config, shard, 64, 64, 1048576, 'meta')

    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif is_dataclass(value):
            for field in fields(value):
                yield from tensors(getattr(value, field.name))

    values = [tensor for execution in (vision, text, decode, draft) for tensor in tensors(execution)]
    assert len({tensor.untyped_storage()._cdata for tensor in values}) == len(values)
    assert vision.query.shape == (65536, 4, 96)
    assert text.embedding.shape == (4096, config.hidden)
    assert decode.logits.shape == (16, config.vocabulary)
    assert draft.attention.key.dtype == draft.attention.value.dtype == torch.bfloat16
    selected = text.rows(128, 1)
    assert selected.embedding.shape == (128, config.hidden)
    assert selected.embedding.untyped_storage()._cdata == text.embedding.untyped_storage()._cdata
    assert selected.attention.query.untyped_storage()._cdata == text.attention.query.untyped_storage()._cdata
