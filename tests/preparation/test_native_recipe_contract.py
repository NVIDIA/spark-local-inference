# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe metadata builders are checked by the serving contract validator.

CPU tensors provide real buffer geometry. Graph handles are identifiers only;
these tests validate metadata and do not execute or capture model programs.
"""

import importlib
import inspect
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ignition.components.completion import CompletionOutputs
from ignition.components.paged_batch import BatchInputs
from ignition.components.sampling import AdmissionBuffers
from ignition.components.state import StateLayout, StateStorage
from ignition.components.state_transfer import StateCommands
from ignition.preparation import contract as c
from ignition.preparation.execution import GraphSignature, PreparedExecution


RECIPES = (
    "deepseek41", "deepseek4", "qwen38", "qwen38_dense", "glm53", "mimo_v2",
    "kimi3", "minimax_m3", "nemotron_h", "step5", "gemma4",
)


def _vision_buffers(recipe):
    if recipe in ("gemma4", "qwen38_dense"):
        return {}
    if recipe in ("deepseek4", "deepseek41"):
        from ignition.components.deepseek_vision import VisionInputs

        capacities = importlib.import_module(f"ignition.recipes.{recipe}.bootstrap").PATCH_CAPACITIES
        inputs = VisionInputs.allocate("cpu", patches=max(capacities))
    else:
        module = "vision_state" if recipe in ("qwen38", "glm53") else "vision"
        inputs = importlib.import_module(f"ignition.recipes.{recipe}.{module}").VisionInputs.allocate("cpu")
    from ignition.components.vision_state import PrefillVisionInputs

    values = inputs.buffers()
    values["vision_feature_slot"] = PrefillVisionInputs.allocate(1, "cpu").vision_feature_slot
    return values


def _recipe_metadata(recipe, *, all_variants=True, config_options=None, completion_capacity=None):
    module = importlib.import_module(f"ignition.recipes.{recipe}.contract")
    bootstrap = importlib.import_module(f"ignition.recipes.{recipe}.bootstrap")
    options = {}
    if recipe == "deepseek41":
        options = dict(max_active=32, prefill_chunk_size=4096,
            speculative_widths=tuple(range(1, 9)), speculative_token_capacity=128)
    fields = bootstrap.PreparationConfig.__dataclass_fields__
    if recipe == "qwen38_dense":
        options.update(draft_snapshot=Path("metadata-draft"), max_active=32)
    if recipe in ("qwen38", "glm53"):
        options["mtp"] = True
    if recipe == "qwen38":
        options["mixed_prefill"] = True
    if "draft_checkpoint" in fields:
        options["draft_checkpoint"] = Path("metadata-draft")
    options.update(config_options or {})
    config = bootstrap.PreparationConfig(**options)
    capacity = config.max_active
    vocabulary = getattr(module, "VOCABULARY", 262144)
    page_tokens = 256 if recipe == "glm53" else 64 if recipe == "qwen38" else 128
    inputs = BatchInputs.allocate(config.prefill_chunk_size, config.context // page_tokens,
        "cpu", capacity=capacity)
    buffers = inputs.buffers()
    admission = AdmissionBuffers.allocate("cpu", vocab_size=vocabulary, capacity=capacity)
    buffers.update({name: getattr(admission, name) for name in admission.__dataclass_fields__})
    buffers.update(StateCommands.allocate("cpu", capacity=capacity).buffers())
    layout = StateLayout.from_specs((("consumed_tokens", (1,), torch.int64),))
    storage = StateStorage.allocate(layout, capacity * 8 + 1, "cpu")
    pool = SimpleNamespace(layout=layout, pages={}, backing=storage.backing,
        config=SimpleNamespace(text={"compress_ratios": []}))
    buffers.update(state_pool=storage.backing, memory_capacity=storage.memory_capacity,
        request_pending_token=torch.zeros(capacity, dtype=torch.int64),
        sampling_grammar_mask=torch.zeros((capacity, (vocabulary + 31) // 32), dtype=torch.int32),
        frame_index=torch.zeros(1, dtype=torch.int32),
        submission_sequence=torch.zeros(1, dtype=torch.int64),
        draft_page_tables=torch.zeros_like(inputs.page_tables))
    completion_capacity = config.completion_token_capacity if completion_capacity is None else completion_capacity
    buffers.update(CompletionOutputs.allocate((2, capacity, 16 + completion_capacity), device="cpu").buffers())
    vision = getattr(config, "vision", not getattr(config, "language_model_only", False))
    vision_buffers = _vision_buffers(recipe) if vision else {}
    buffers.update(vision_buffers)
    if vision_buffers:
        from ignition.components.vision_state import PrefillVisionInputs
        buffers.update(PrefillVisionInputs.allocate(config.prefill_chunk_size, "cpu").buffers())
    from ignition.components.prompt_logprobs import PromptOutputs
    buffers.update(PromptOutputs(config.prefill_chunk_size, "cpu").buffers())
    signatures = {}

    def add(kind, batch, width, *, packed=0, drafts=None, confidence=0, prefill=0, enabled=None):
        fields = list(c.text_inputs(kind)) if kind != "vision" else list(vision_buffers)
        if recipe == "deepseek41" and kind != "vision":
            row_fields = ["request_state_rows", "row_slot_ids", "row_generations", "live_rows"]
            if kind in ("init", "restore", "snapshot"):
                fields = (row_fields if kind == "snapshot" else
                    ["request_state_rows", *admission.__dataclass_fields__])
                fields = [*fields, "page_tables", *StateCommands.allocate("cpu", capacity=capacity).buffers()]
            elif kind == "resume":
                fields = [*row_fields, "sampling_grammar_mask"]
            else:
                fields = [*inputs.buffers(), "sampling_grammar_mask"]
                if kind in ("prefill", "scored_prefill"):
                    fields.extend(("vision_token_indices", "vision_feature_slot"))
                if kind == "scored_prefill":
                    fields.append("prompt_target_tokens")
        if confidence:
            fields.remove("query_lengths")
            fields.append("speculative_prefix_limits")
        for field in fields:
            if field not in buffers:
                shape = (1,) if field in ("vision_slot", "vision_generation", "vision_feature_slot", "prefill_row") else (capacity,)
                dtype = torch.int32 if field in ("speculative_prefix_limits", "prefill_row") else torch.int64
                buffers[field] = torch.zeros(shape, dtype=dtype)
        name = f"{kind}-{batch}-{width}-{packed}-{enabled}"
        signatures[name] = GraphSignature(kind, batch, width, tuple(fields),
            speculation_enabled=(True if kind == "speculative" else None) if enabled is None else enabled,
            packed_token_capacity=packed, draft_tokens=drafts,
            confidence_width=confidence, prefill_capacity=prefill)

    from ignition.components.capacity import batch_capacities, prefill_capacities
    batches = getattr(config, "batch_capacities", batch_capacities(capacity))
    for batch in batches:
        for kind in ("init", "restore", "snapshot", "resume", "decode"):
            for enabled in ((True, False) if recipe == "deepseek41" and kind != "decode" else
                            (False,) if recipe == "deepseek41" else (None,)):
                add(kind, batch, 1, enabled=enabled)
    for tokens in getattr(config, "prefill_capacities", prefill_capacities(config.prefill_chunk_size)):
        for enabled in ((True, False) if recipe == "deepseek41" else (None,)):
            add("prefill", 1, tokens, enabled=enabled)
    from ignition.components.prompt_logprobs import scored_prefill_capacities
    for tokens in scored_prefill_capacities(config.prefill_chunk_size):
        for enabled in ((True, False) if recipe == "deepseek41" else (None,)):
            add("scored_prefill", 1, tokens, enabled=enabled)
    if getattr(config, "mtp", True):
        for batch, width in getattr(config, "speculative_shapes", ()):
            add("speculative", batch, width)
    for batch, width, rows in getattr(config, "speculative_round_shapes", ()):
        add("speculative", batch, width, packed=rows)
    for batch, rows in getattr(config, "packed_speculative_shapes", ()):
        width = max(config.speculative_widths)
        add("speculative", batch, width, packed=rows, drafts=width - 1, confidence=width - 1)
        buffers["proposal_confidence"] = torch.zeros((2, capacity, width - 1), dtype=torch.float32)
    for enabled, batch, width, prefill in getattr(config, "mixed_shapes", ()):
        add("mixed", batch, width, prefill=prefill, enabled=enabled)
    arguments = inspect.signature(module.prepare).parameters
    kwargs = {}
    for key, value in dict(page_tokens=256 if recipe == "glm53" else 64 if recipe == "qwen38" else 128,
            draft_lag=0 if recipe in ("deepseek41", "deepseek4", "mimo_v2") else 1,
            rows=None if recipe in ("deepseek41", "deepseek4") else 1,
            vocabulary=vocabulary, mtp=getattr(config, "mtp", True), dspark=True,
            image_slots=getattr(config, "image_concurrency", 4), vision=vision).items():
        if key in arguments:
            kwargs[key] = value
    if recipe == "glm53" and not config.mtp:
        kwargs["draft_lag"] = None
    if "frontend" in arguments:
        kwargs["frontend"] = importlib.import_module(f"ignition.recipes.{recipe}.frontend").FRONTEND
    if vision_buffers:
        add("vision", 1, 1)
    contract = module.prepare(signatures, buffers, pool, **kwargs)
    if contract["vision"]["max_images"]:
        signatures.pop("vision-1-1-0-None", None)
        for tokens in contract["vision"]["patch_capacities"]:
            add("vision", 1, tokens)
        contract = module.prepare(signatures, buffers, pool, **kwargs)
    else:
        signatures.pop("vision-1-1-0-None", None)
        contract = module.prepare(signatures, buffers, pool, **kwargs)
    bundle = PreparedExecution(buffers=buffers, signatures=signatures,
        graphs=dict.fromkeys(signatures), max_active=capacity, device=0,
        estimated_durations_ns=dict.fromkeys(signatures, 1))
    bundle._validate_resources = lambda: (1, list(range(1, len(signatures) + 1)),
        [(tensor.data_ptr(), tensor.nbytes) for tensor in buffers.values()])
    manifest = PreparedExecution.manifest(bundle, all_variants=all_variants)
    return config, contract, manifest


@pytest.mark.parametrize("recipe", RECIPES)
def test_generated_recipe_catalog_passes_native_contract_validation(recipe):
    import flare_heliopause

    config, contract, manifest = _recipe_metadata(recipe)
    flare_heliopause.validate_prepared_contract(manifest, json.dumps(contract))
    kinds = {graph["kind"] for graph in manifest["graphs"]}
    assert {"init", "restore", "snapshot", "resume", "prefill", "scored_prefill", "decode", "speculative"} <= kinds
    if recipe == "deepseek41":
        assert config.max_active == 32
        assert (32, 7, 8, 128) in {
            (graph["batch_capacity"], graph["draft_tokens"], graph["token_capacity"], graph.get("packed_token_capacity", 0))
            for graph in manifest["graphs"] if graph["kind"] == "speculative"
        }


@pytest.mark.parametrize("image_concurrency", [None, 0])
def test_kimi_text_only_zero_image_capacity_passes_native_contract_validation(image_concurrency):
    import flare_heliopause

    config, contract, manifest = _recipe_metadata("kimi3", config_options=dict(
        vision=False, image_concurrency=image_concurrency))
    assert config.image_concurrency == 0
    assert contract["vision"] == dict(feature_slots=0, features_per_request=0,
        max_images=0, patches_per_feature=0, patch_capacities=[])
    assert not contract["frontend"].get("image_preparation")
    assert all(graph["kind"] != "vision" for graph in manifest["graphs"])
    flare_heliopause.validate_prepared_contract(manifest, json.dumps(contract))


def test_deepseek41_export_geometry_passes_native_bootstrap():
    import flare_heliopause
    from huggingface_hub import try_to_load_from_cache
    from ignition.preparation.frontend import prepare_frontend, PromptEncoder
    from ignition.recipes.deepseek41.recipe import RECIPE

    checkpoint = RECIPE.checkpoints[0]
    tokenizer = try_to_load_from_cache(
        "DeepSeek-AI/DeepSeek-V4.1-Flash-1017", "tokenizer.json", revision=checkpoint.revision
    )
    if not isinstance(tokenizer, str):
        pytest.skip("DeepSeek V4.1 pinned checkpoint tokenizer is not cached")
    snapshot = Path(tokenizer).parent
    config, contract, manifest = _recipe_metadata("deepseek41", all_variants=False)
    frontend = prepare_frontend(snapshot, eos_token_ids=[1],
        prompt_encoder=PromptEncoder.DEEPSEEK_V41, image_profile="deepseek_row_major")
    value = dict(identity="00" * 32, model_id=RECIPE.served_model_name,
        max_sequence_tokens=config.context, prefill_chunk_size=config.prefill_chunk_size,
        pages=32768, state_rows=257, snapshots=32, page_tokens=128,
        frontend=frontend, sampling_defaults=RECIPE.sampling_defaults.as_dict(),
        assets={"tokenizer": {"path": str(snapshot / "tokenizer.json"),
            "sha256": hashlib.sha256((snapshot / "tokenizer.json").read_bytes()).hexdigest()}},
        model_contract=contract, speculative_widths=list(config.speculative_widths),
        speculative_token_capacity=config.speculative_token_capacity)
    flare_heliopause.validate_prepared_catalog(manifest, json.dumps(value))


def test_generated_admission_pending_token_rejects_an_undeclared_read():
    import flare_heliopause
    from ignition.preparation import runtime_package_pb2 as p

    _, contract, manifest = _recipe_metadata("deepseek41")
    consumer = next(graph for graph in contract["graphs"] if graph["kind"] == p.SPECULATIVE)
    producer = next(graph for graph in contract["graphs"] if graph["kind"] == p.INIT)
    producer["pending_token"] = consumer["pending_token"]
    with pytest.raises(ValueError, match="pending token requires a declared device-state read"):
        flare_heliopause.validate_prepared_contract(manifest, json.dumps(contract))


def test_qwen_dense_packed_catalog_and_zero_lag_cursor_pass_native_contract_validation():
    import flare_heliopause
    from ignition.preparation import runtime_package_pb2 as p

    config, contract, manifest = _recipe_metadata("qwen38_dense")
    flare_heliopause.validate_prepared_contract(manifest, json.dumps(contract))
    speculative = [graph for graph in manifest["graphs"] if graph["kind"] == "speculative"]
    assert (32, 7, 8, 128) in {
        (graph["batch_capacity"], graph["draft_tokens"], graph["token_capacity"], graph.get("packed_token_capacity", 0))
        for graph in speculative
    }
    assert config.speculative_token_capacity == 128
    assert all(graph["draft_cursor"]["lag_tokens"] == 0
        for graph in contract["graphs"] if graph["kind"] == p.SPECULATIVE)


@pytest.mark.parametrize("tokens", [8, 13, 16, 32])
def test_native_contract_uses_explicit_completion_capacity_for_every_output_axis(tokens):
    import flare_heliopause

    _, contract, manifest = _recipe_metadata("qwen38_dense", completion_capacity=tokens)
    assert manifest["completion_token_capacity"] == tokens
    fields = {buffer["name"]: buffer["shape"] for buffer in manifest["buffers"]}
    assert fields["completion_records"][-1] == 16 + tokens
    assert fields["completion_logprobs"][2] == tokens
    assert fields["completion_speculative_survival"][2] == tokens - 1
    flare_heliopause.validate_prepared_contract(manifest, json.dumps(contract))
    manifest["completion_token_capacity"] = tokens + 1
    with pytest.raises(ValueError):
        flare_heliopause.validate_prepared_contract(manifest, json.dumps(contract))


def test_native_contract_accepts_omitted_or_zero_capacity_only_with_eight_token_buffers():
    import flare_heliopause

    _, contract, manifest = _recipe_metadata("qwen38_dense", completion_capacity=8)
    manifest.pop("completion_token_capacity")
    flare_heliopause.validate_prepared_contract(manifest, json.dumps(contract))
    manifest["completion_token_capacity"] = 0
    flare_heliopause.validate_prepared_contract(manifest, json.dumps(contract))


def test_native_capacity_one_omits_survival_without_narrowing_state_row_storage():
    import flare_heliopause

    config, contract, manifest = _recipe_metadata("glm53", config_options=dict(mtp=False))
    assert config.completion_token_capacity == manifest["completion_token_capacity"] == 1
    fields = {buffer["name"]: buffer["shape"] for buffer in manifest["buffers"]}
    assert fields["completion_records"] == [2, 4, 17]
    assert fields["request_state_rows"] == [4, 8]
    assert "completion_speculative_survival" not in fields
    assert all(graph["kind"] != "speculative" for graph in manifest["graphs"])
    flare_heliopause.validate_prepared_contract(manifest, json.dumps(contract))
