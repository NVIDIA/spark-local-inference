# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 Flash TrellisMX release: detection gates, sidecar integrity, and the expert bridge."""

import hashlib
import json
import math
import os
import sys
import types
from pathlib import Path

import pytest

from ignition.recipes.glm53.trellismx import (
    BITS, CHECKPOINT_ID, GLOBAL_INTERMEDIATE, HIDDEN_SIZE, MANIFEST_SCHEMA, MTP_LAYER, NUM_EXPERTS,
    SIDECAR_CONTRACT, SIDECAR_IDENTITY, SWIGLU_LIMIT, TENSOR_DTYPES, TOP_K,
    WORLD_SIZE, expected_tensor_shapes, is_trellismx_overlay, load_sidecar_tensors, load_trellis_experts,
    prepare_trellismx_experts, sidecar_header, validate_sidecar_metadata, validate_sidecar_tensors,
)
from ignition.preparation.trellismx_release import trellismx_assets_root
from tests.recipes.glm53_trellismx_release import RELEASE, requires_cuda, requires_release

_LOCAL = GLOBAL_INTERMEDIATE // WORLD_SIZE
_ELEMENT_BYTES = {"I16": 2, "U8": 1, "F16": 2}
_LAYERS = tuple(range(3, MTP_LAYER + 1))
# The synthetic releases' encoder transform, which every synthetic sidecar pins.
_TRANSFORM = b"{}"
_TRANSFORM_SHA256 = hashlib.sha256(_TRANSFORM).hexdigest()


def _sidecar_name(layer, rank):
    return f"p8-layer-{layer:03d}-tp4-rank-{rank}.safetensors"


def _write_sidecar(path, *, layer, rank, values=()):
    """Write a sidecar with the release's header, metadata contract and geometry.

    The payload is sparse zeros except ``values``: (tensor name, bytes) written at the
    start of that tensor.
    """
    metadata = dict(SIDECAR_IDENTITY, **SIDECAR_CONTRACT, layer=str(layer), rank=str(rank),
                    local_atom_begin=str(rank * _LOCAL // 32), source_design_sha256="1" * 64,
                    encoder_transform_sha256=_TRANSFORM_SHA256)
    header, offsets, end = {"__metadata__": metadata}, {}, 0
    for name, shape in expected_tensor_shapes(_LOCAL).items():
        nbytes = math.prod(shape) * _ELEMENT_BYTES[TENSOR_DTYPES[name]]
        header[name] = {"dtype": TENSOR_DTYPES[name], "shape": list(shape), "data_offsets": [end, end + nbytes]}
        offsets[name], end = end, end + nbytes
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as file:
        file.write(len(raw).to_bytes(8, "little") + raw)
        file.truncate(8 + len(raw) + end)
        for name, value in values:
            file.seek(8 + len(raw) + offsets[name])
            file.write(value)


def _payload_offset(path, name):
    """Absolute byte offset of one tensor's payload in a sidecar."""
    with open(path, "rb") as file:
        length = int.from_bytes(file.read(8), "little")
        return 8 + length + json.loads(file.read(length))[name]["data_offsets"][0]


def _sha256(path):
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


@pytest.fixture(scope="module")
def sidecar_files(tmp_path_factory):
    """The 172 sidecars a release names, sparse at the release's real sizes.

    Detection reads every header but no payload, so only layer 3 rank 0, which the
    loading tests read, carries its real digest in the default manifest.
    """
    base = tmp_path_factory.mktemp("trellismx-sidecars")
    for layer in _LAYERS:
        for rank in range(WORLD_SIZE):
            _write_sidecar(base / _sidecar_name(layer, rank), layer=layer, rank=rank)
    sizes = {(layer, rank): (base / _sidecar_name(layer, rank)).stat().st_size
             for layer in _LAYERS for rank in range(WORLD_SIZE)}
    return base, sizes, {(3, 0): _sha256(base / _sidecar_name(3, 0))}


def _manifest(sizes, digests):
    files = [{"path": f"sidecars/{_sidecar_name(layer, rank)}", "bytes": sizes[layer, rank],
              "sha256": digests.get((layer, rank), "0" * 64), "layer": layer, "rank": rank, "bits": BITS,
              "source_design_sha256": "1" * 64}
             for layer in _LAYERS for rank in range(WORLD_SIZE)]
    return {
        "schema": MANIFEST_SCHEMA,
        "checkpoint": CHECKPOINT_ID,
        "carrier": {"repo_id": "local-inference-lab/GLM-5.3-Flash-NVFP4",
                    "revision": "520de24eabf507659eaef7c70f14fd584527facc"},
        "allocation": {str(layer): WORLD_SIZE for layer in _LAYERS},
        "mtp": {"layer": MTP_LAYER, "bits": BITS},
        "files": files,
    }


def _config():
    return {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "text_config": {"hidden_size": HIDDEN_SIZE, "moe_intermediate_size": GLOBAL_INTERMEDIATE,
                        "n_routed_experts": NUM_EXPERTS, "swiglu_limit": SWIGLU_LIMIT},
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers": {
            f"model.language_model.layers.{layer}.mlp.experts":
                {"quant_algo": "MXFP8", "group_size": 32} if layer == MTP_LAYER else
                {"quant_algo": "NVFP4", "group_size": 16}
            for layer in _LAYERS}},
    }


@pytest.fixture
def manifest(sidecar_files):
    """A manifest naming the shared sidecars, edited by tests before ``build`` writes it."""
    _, sizes, digests = sidecar_files
    return lambda: _manifest(sizes, digests)


@pytest.fixture
def build(tmp_path, sidecar_files):
    """Write a release root over the shared sidecars.

    ``own`` maps (layer, rank) to the values of a private sidecar written for that
    test, whose size and digest the default manifest declares; ``without_sidecar``
    leaves one sidecar out.
    """
    base, sizes, digests = sidecar_files

    def build(*, manifest=None, config=None, carrier_config=None, drop=(), without_sidecar=None, own=None):
        root = tmp_path / "release"
        (root / "carrier").mkdir(parents=True)
        if without_sidecar is None and not own:
            (root / "sidecars").symlink_to(base, target_is_directory=True)
        else:
            (root / "sidecars").mkdir()
            sizes_, digests_ = dict(sizes), dict(digests)
            for layer in _LAYERS:
                for rank in range(WORLD_SIZE):
                    path = root / "sidecars" / _sidecar_name(layer, rank)
                    if (layer, rank) == without_sidecar:
                        continue
                    if own and (layer, rank) in own:
                        _write_sidecar(path, layer=layer, rank=rank, values=own[layer, rank])
                        sizes_[layer, rank], digests_[layer, rank] = path.stat().st_size, _sha256(path)
                    else:
                        path.symlink_to(base / _sidecar_name(layer, rank))
            if manifest is None:
                manifest = _manifest(sizes_, digests_)
        (root / "trellismx-manifest.json").write_text(json.dumps(
            _manifest(sizes, digests) if manifest is None else manifest))
        (root / "config.json").write_text(json.dumps(_config() if config is None else config))
        (root / "carrier" / "config.json").write_text(
            json.dumps(_config() if carrier_config is None else carrier_config))
        (root / "carrier" / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"lm_head.weight": "model.safetensors"}, "metadata": {"total_size": 1}}))
        (root / "carrier" / "model.safetensors").write_bytes(b"\0")
        (root / "design").mkdir()
        (root / "design" / "transform.json").write_bytes(_TRANSFORM)
        for name in drop:
            (root / name).unlink()
        return root
    return build


def _load(release, **overrides):
    """The routed-expert loader exactly as ``GlmWeightLoader.load_experts`` calls it, on the CPU."""
    import torch

    arguments = dict(layer=3, rank=0, world_size=WORLD_SIZE, hidden_size=HIDDEN_SIZE,
                     global_intermediate_size=GLOBAL_INTERMEDIATE, num_experts=NUM_EXPERTS, top_k=TOP_K,
                     swiglu_limit=SWIGLU_LIMIT, device="cpu", params_dtype=torch.bfloat16)
    arguments.update(overrides)
    return load_trellis_experts(release, **arguments)


@pytest.fixture
def unreachable_preparation(monkeypatch):
    """Fail the test if a sidecar that should be rejected reaches b12x's preparation."""
    module = types.ModuleType("b12x.moe.fused_moe.trellismx")

    def prepare(*args, **kwargs):
        raise AssertionError("an unverified sidecar reached b12x's preparation")

    module.prepare_trellismx_experts = prepare
    monkeypatch.setitem(sys.modules, "b12x.moe.fused_moe.trellismx", module)


def test_detection_ignores_a_directory_without_a_manifest(tmp_path):
    assert is_trellismx_overlay(tmp_path) is None


def test_detection_ignores_another_recipes_manifest(build, manifest):
    document = manifest()
    document["schema"] = "ds41-trellismx-k4-tp4.v1"
    assert is_trellismx_overlay(build(manifest=document)) is None


def test_a_plain_checkpoint_root_is_unchanged(tmp_path):
    """The assets-root hook must not move a checkpoint that is not a release."""
    from ignition.recipes import _snapshot_cached

    root = tmp_path / "plain"
    root.mkdir()
    (root / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}, "metadata": {"total_size": 8}}))
    assert _snapshot_cached(root) is False
    (root / "model-00001-of-00001.safetensors").write_bytes(b"\0" * 8)
    assert _snapshot_cached(root) is True


def test_the_release_is_declared_by_its_published_identity():
    """The recipe ships a repository and revision, never a machine path."""
    from ignition.recipes import resolve
    from ignition.recipes.glm53.recipe import RECIPE

    trellismx = next(c for c in RECIPE.checkpoints if c.alias == "glm-5.3-flash-trellismx")
    assert trellismx.repository == "brandonmusic/glm-5.3-flash-tr3-fp8"
    assert trellismx.revision == "80cbe535eabbeddfbd567e376a09ac559bcbac75"
    assert not Path(trellismx.repository).is_absolute()
    for name in ("glm-5.3-flash-trellismx", "brandonmusic/glm-5.3-flash-tr3-fp8",
                 "brandonmusic/glm-5.3-flash-tr3-fp8@80cbe535eabbeddfbd567e376a09ac559bcbac75"):
        model = resolve(name)
        assert model.checkpoint is trellismx, name
        assert model.checkpoint.stored_bytes == 185_061_247_064
    # The other GLM checkpoint is still named exactly, by alias and by repository.
    for name in ("glm-5.3-flash", "nvidia/GLM-5.3-Flash-FP4-Local"):
        assert resolve(name).checkpoint.alias == "glm-5.3-flash"


def test_a_bare_directory_keeps_the_first_checkpoint_placement(tmp_path):
    """A directory no checkpoint declares is served by the recipe's first checkpoint.

    That is the pre-existing behaviour for every GLM directory on disk, so it stays with
    ``nvidia/GLM-5.3-Flash-FP4-Local``; the TrellisMX release is addressed by its alias or
    repository instead. ``ignition/recipes/__init__.py:resolve`` is the branch under test.
    """
    from ignition.recipes import resolve

    carrier = tmp_path / "carrier-copy"
    carrier.mkdir()
    (carrier / "config.json").write_text(json.dumps(
        {"architectures": ["Glm5NextForConditionalGeneration"], "model_type": "glm5_next"}))
    model = resolve(str(carrier))
    assert model.directory == carrier.resolve()
    assert model.checkpoint.alias == "glm-5.3-flash"
    assert model.checkpoint.stored_bytes == 177_421_592_008


@requires_release
def test_a_local_release_is_a_hub_cache_snapshot_of_the_published_checkpoint(tmp_path):
    """The local copy is reached the way a downloaded snapshot is: hub_cache/models--<repo>/snapshots/<rev>.

    A link at that path is all ``resolve(alias)`` needs; ``cached`` then reads the release's
    carrier index through ``trellismx_assets_root`` and nothing is downloaded.
    """
    from ignition.recipes import resolve

    model = resolve("glm-5.3-flash-trellismx")
    snapshot = model.snapshot(tmp_path)
    assert snapshot.name == "80cbe535eabbeddfbd567e376a09ac559bcbac75"
    assert snapshot.parent.name == "snapshots"
    assert snapshot.parent.parent.name == "models--brandonmusic--glm-5.3-flash-tr3-fp8"
    assert model.cached(tmp_path) is False
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(RELEASE)
    assert model.cached(tmp_path) is True
    assert model.snapshot(tmp_path) == snapshot


def test_the_assets_root_moves_a_release_to_its_carrier(build, tmp_path):
    release = build()
    assert trellismx_assets_root(release) == release / "carrier"
    assert trellismx_assets_root(tmp_path) == tmp_path


def test_the_assets_root_ignores_another_recipes_manifest(tmp_path):
    (tmp_path / "trellismx-manifest.json").write_text(json.dumps({"schema": "ds41-trellismx-k4-tp4.v1"}))
    assert trellismx_assets_root(tmp_path) == tmp_path


def test_the_assets_root_keeps_a_root_without_a_carrier_index(tmp_path):
    (tmp_path / "trellismx-manifest.json").write_text(json.dumps({"schema": MANIFEST_SCHEMA}))
    assert trellismx_assets_root(tmp_path) == tmp_path


def test_detection_accepts_a_complete_release(build):
    release = is_trellismx_overlay(build())
    assert release is not None
    assert release.layers == _LAYERS
    assert (release.world_size, release.bits, release.mtp_layer) == (WORLD_SIZE, BITS, MTP_LAYER)
    assert len(release.files) == len(release.layers) * WORLD_SIZE
    assert len(release.routed_layers) == 42
    assert release.sidecar(7, 3).path.name == "p8-layer-007-tp4-rank-3.safetensors"
    assert release.local_intermediate_size() == _LOCAL


@pytest.mark.parametrize("field,value", [
    ("checkpoint", "codec-v2-uniform-k3-mtp45-k3"),
    ("allocation", {str(layer): WORLD_SIZE for layer in range(3, MTP_LAYER)}),
    ("allocation", {str(layer): WORLD_SIZE for layer in _LAYERS} | {"46": WORLD_SIZE}),
    ("mtp", {"layer": 44, "bits": BITS}),
    ("mtp", {"layer": MTP_LAYER, "bits": 3}),
])
def test_gate_two_rejects_a_different_checkpoint_or_allocation(build, manifest, field, value):
    document = manifest()
    document[field] = value
    with pytest.raises(ValueError, match="gate 2"):
        is_trellismx_overlay(build(manifest=document))


def test_gate_two_rejects_a_second_spelling_of_a_layer(build, manifest):
    document = manifest()
    document["allocation"]["03"] = WORLD_SIZE
    with pytest.raises(ValueError, match="gate 2: allocation key '03'"):
        is_trellismx_overlay(build(manifest=document))


def test_a_manifest_with_a_duplicated_key_is_unreadable(build, manifest):
    root = build()
    text = json.dumps(manifest())
    (root / "trellismx-manifest.json").write_text(text[:-1] + ', "checkpoint": "' + CHECKPOINT_ID + '"}')
    with pytest.raises(ValueError, match="duplicate JSON key 'checkpoint'"):
        is_trellismx_overlay(root)


def test_an_unreadable_manifest_never_passes_for_a_plain_checkpoint(build):
    from ignition.errors import UsageError
    from ignition.recipes import _snapshot_cached, resolve

    root = build()
    (root / "trellismx-manifest.json").write_text("{not json")
    with pytest.raises(ValueError, match="unreadable"):
        is_trellismx_overlay(root)
    with pytest.raises(ValueError, match="unreadable"):
        trellismx_assets_root(root)
    assert _snapshot_cached(root) is False
    with pytest.raises(UsageError, match="unreadable release manifest"):
        resolve(str(root))


def test_gate_two_rejects_a_two_rank_allocation(build, manifest):
    document = manifest()
    document["allocation"] = {layer: 2 for layer in document["allocation"]}
    with pytest.raises(ValueError, match="gate 2"):
        is_trellismx_overlay(build(manifest=document))


@pytest.mark.parametrize("carrier", [
    {"repo_id": "", "revision": "520de24eabf507659eaef7c70f14fd584527facc"},
    {"repo_id": "local-inference-lab/GLM-5.3-Flash-NVFP4", "revision": "520de24e"},
    {"repo_id": "local-inference-lab/GLM-5.3-Flash-NVFP4", "revision": "Z" * 40},
    {"repo_id": "local-inference-lab/GLM-5.3-Flash-NVFP4"},
    "local-inference-lab/GLM-5.3-Flash-NVFP4",
])
def test_gate_three_requires_a_pinned_carrier(build, manifest, carrier):
    document = manifest()
    document["carrier"] = carrier
    with pytest.raises(ValueError, match="gate 3"):
        is_trellismx_overlay(build(manifest=document))


def test_gate_three_requires_the_carrier_index(build):
    with pytest.raises(ValueError, match="gate 3"):
        is_trellismx_overlay(build(drop=("carrier/model.safetensors.index.json",)))


def test_gate_three_rejects_a_root_config_that_differs_from_the_carrier(build):
    config = _config()
    config["text_config"]["n_routed_experts"] = 256
    with pytest.raises(ValueError, match="gate 3"):
        is_trellismx_overlay(build(config=config))


def test_gate_four_requires_one_sidecar_per_layer_and_rank(build, manifest):
    document = manifest()
    document["files"] = document["files"][:-1]
    with pytest.raises(ValueError, match="gate 4"):
        is_trellismx_overlay(build(manifest=document))


def test_gate_four_rejects_a_duplicated_sidecar(build, manifest):
    document = manifest()
    document["files"][-1] = dict(document["files"][0])
    with pytest.raises(ValueError, match="gate 4"):
        is_trellismx_overlay(build(manifest=document))


@pytest.mark.parametrize("mutation", ["path", "layer", "rank", "bits", "bytes", "digest", "hex", "design"])
def test_gate_four_rejects_a_misdeclared_sidecar(build, manifest, mutation):
    document = manifest()
    entry = document["files"][17]
    if mutation == "path":
        entry["path"] = f"sidecars/{_sidecar_name(entry['layer'] + 1, entry['rank'])}"
    elif mutation == "layer":
        entry["layer"] = 5
    elif mutation == "rank":
        entry["rank"] = 2
    elif mutation == "bits":
        entry["bits"] = 3
    elif mutation == "bytes":
        entry["bytes"] += 1
    elif mutation == "digest":
        entry["sha256"] = "0" * 63
    elif mutation == "hex":
        entry["sha256"] = "Z" * 64
    else:
        entry["source_design_sha256"] = "1" * 63 + "G"
    with pytest.raises(ValueError, match="gate 4"):
        is_trellismx_overlay(build(manifest=document))


def test_gate_four_rejects_a_missing_sidecar_file(build):
    with pytest.raises(ValueError, match="gate 4: missing sidecar"):
        is_trellismx_overlay(build(without_sidecar=(7, 2)))


def test_gate_four_requires_the_encoder_transform(build):
    with pytest.raises(ValueError, match="gate 4: missing .*design/transform.json"):
        is_trellismx_overlay(build(drop=("design/transform.json",)))


def test_a_release_is_cached_only_with_every_file_its_manifest_names(build):
    """A carrier without its sidecars is an incomplete download that export must complete."""
    from ignition.recipes import _snapshot_cached

    assert _snapshot_cached(build()) is True


@pytest.mark.parametrize("damage", ["missing sidecar", "short sidecar", "missing transform"])
def test_an_incomplete_release_is_not_cached(build, damage):
    from ignition.recipes import _snapshot_cached

    if damage == "missing sidecar":
        root = build(without_sidecar=(44, 3))
    elif damage == "short sidecar":
        root = build(own={(44, 3): ()})
        with open(root / "sidecars" / _sidecar_name(44, 3), "r+b") as file:
            file.truncate(4096)
    else:
        root = build(drop=("design/transform.json",))
    assert (root / "carrier" / "model.safetensors").is_file()
    assert _snapshot_cached(root) is False


@pytest.mark.parametrize("edit", ["dropped", "repeated", "respelled", "added"])
def test_a_file_list_that_is_not_the_canonical_inventory_is_never_complete(build, manifest, edit):
    """Every listed file exists at its declared size, but the list is not one sidecar per allocated layer and rank."""
    from ignition.errors import UsageError
    from ignition.preparation.trellismx_release import trellismx_files_present
    from ignition.recipes import _snapshot_cached, resolve

    document = manifest()
    files = document["files"]
    if edit == "dropped":
        files.pop()
    elif edit == "repeated":
        files[-1] = dict(files[0])
    elif edit == "respelled":
        files[0]["path"] = "sidecars/./" + files[0]["path"].split("/")[-1]
    else:
        files.append({"path": "design/transform.json", "bytes": len(_TRANSFORM)})
    root = build(manifest=document)
    assert all((root / entry["path"]).stat().st_size == entry["bytes"] for entry in files)
    with pytest.raises(ValueError, match="one sidecar per allocated layer and rank"):
        trellismx_files_present(root)
    assert _snapshot_cached(root) is False
    with pytest.raises(UsageError, match="one sidecar per allocated layer and rank"):
        resolve(str(root))


def test_post_download_verification_requires_every_release_file(build, tmp_path):
    from ignition.errors import UsageError
    from ignition.recipes import resolve

    # A private sidecar makes every other sidecar its own link, which this test may remove.
    root = build(own={(3, 0): ()})
    resolve(str(root)).verify(tmp_path)
    link = root / "sidecars" / _sidecar_name(9, 1)
    assert link.is_symlink()
    link.unlink()
    with pytest.raises(UsageError, match="lacks files its TrellisMX release manifest names"):
        resolve(str(root)).verify(tmp_path)


def test_a_release_directory_is_placed_like_the_published_release(build, sidecar_files):
    """Resolving the release by path keeps its overlay placement, storage and TP4 constraint."""
    from ignition.recipes import resolve

    _, sizes, _ = sidecar_files
    published = resolve("glm-5.3-flash-trellismx")
    model = resolve(str(build()))
    assert model.directory is not None and model.recipe is published.recipe
    assert model.checkpoint.alias == "glm-5.3-flash-trellismx"
    assert model.checkpoint.stored_bytes == 1 + sum(sizes.values())
    assert model.checkpoint.draft_bytes == published.checkpoint.draft_bytes
    assert model.recipe.checkpoint_tensor_parallel(model.checkpoint) == (4,)


def _gpus(count):
    from ignition import planning

    return tuple(planning.Gpu(index, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{index:04d}", 512 << 30, 512 << 30,
                              (12, 0)) for index in range(count))


def test_planning_offers_only_tensor_parallel_four_for_the_release():
    from ignition import planning
    from ignition.errors import CapacityError
    from ignition.recipes import resolve

    model = resolve("glm-5.3-flash-trellismx")
    arguments = dict(concurrency=1, reserve_bytes=0, speculation=None, language_model_only=True)
    assert planning.plan(model.recipe, model.checkpoint, _gpus(4), selected=None, **arguments).world_size == 4
    for selected in ((0,), (0, 1)):
        with pytest.raises(CapacityError, match="needs 4 GPUs"):
            planning.plan(model.recipe, model.checkpoint, _gpus(4), selected=selected, **arguments)
    with pytest.raises(CapacityError, match="needs 4"):
        planning.plan(model.recipe, model.checkpoint, _gpus(2), selected=None, **arguments)
    # The NVFP4 checkpoint keeps every size its recipe declares.
    nvfp4 = resolve("glm-5.3-flash")
    assert planning.plan(nvfp4.recipe, nvfp4.checkpoint, _gpus(4), selected=None, **arguments).world_size == 1


@pytest.mark.parametrize("sizes", [(8,), (0,), (-4,), (4, 4), (4, 2), [4], (4.0,), (True,)])
def test_a_checkpoint_declares_increasing_sizes_its_recipe_declares(sizes):
    from dataclasses import replace

    from ignition.recipes.glm53.recipe import RECIPE

    checkpoint = replace(RECIPE.checkpoints[1], tensor_parallel=sizes)
    with pytest.raises(ValueError, match="increasing tuple of sizes its recipe declares"):
        replace(RECIPE, checkpoints=(RECIPE.checkpoints[0], checkpoint))
    accepted = replace(RECIPE, checkpoints=(RECIPE.checkpoints[0], replace(checkpoint, tensor_parallel=(2, 4))))
    assert accepted.checkpoint_tensor_parallel(accepted.checkpoints[1]) == (2, 4)


def test_help_offers_only_the_release_sizes(capsys):
    from ignition.cli import main

    with pytest.raises(SystemExit):
        main(["export", "glm-5.3-flash-trellismx", "--help"])
    assert "--gpus: select 4 GPUs" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--help"])
    listing = {line.split()[0]: line.split()[1] for line in capsys.readouterr().out.splitlines()
               if line.strip().startswith("glm-5.3-flash")}
    assert listing == {"glm-5.3-flash": "1,2,4", "glm-5.3-flash-trellismx": "4"}


def test_gate_five_requires_every_sidecar_to_pin_the_release_transform(build):
    root = build()
    (root / "design" / "transform.json").write_bytes(b'{"transform": "another"}')
    with pytest.raises(ValueError, match="gate 5: .*encoder_transform_sha256"):
        is_trellismx_overlay(root)


def test_gate_five_reads_every_sidecar_header(build):
    """A same-size sidecar whose own metadata names another layer is rejected at detection."""
    root = build(own={(7, 2): ()})
    _write_sidecar(root / "sidecars" / _sidecar_name(7, 2), layer=8, rank=2)
    with pytest.raises(ValueError, match="gate 5"):
        is_trellismx_overlay(root)


@pytest.mark.parametrize("mutation", [
    lambda config: config.update(architectures=["Glm5NextForCausalLM"]),
    lambda config: config["text_config"].update(moe_intermediate_size=1024),
    lambda config: config["text_config"].update(n_routed_experts=256),
    lambda config: config["text_config"].update(hidden_size=5120),
    lambda config: config["text_config"].update(swiglu_limit=20.0),
    lambda config: config["quantization_config"]["quantized_layers"].update(
        {"model.language_model.layers.45.mlp.experts": {"quant_algo": "NVFP4"}}),
    lambda config: config["quantization_config"]["quantized_layers"].update(
        {"model.language_model.layers.12.self_attn.q_a_proj": {"quant_algo": "NVFP4"}}),
    lambda config: config["quantization_config"]["quantized_layers"].pop(
        "model.language_model.layers.20.mlp.experts"),
    lambda config: config["quantization_config"]["quantized_layers"].update(
        {"model.language_model.layers.0.mlp.experts": "malformed"}),
    lambda config: config["quantization_config"]["quantized_layers"].update(
        {"model.language_model.layers.7.mlp.experts": "NVFP4"}),
    lambda config: config["quantization_config"]["quantized_layers"]["model.language_model.layers.7.mlp.experts"].pop(
        "group_size"),
    lambda config: config["quantization_config"]["quantized_layers"]["model.language_model.layers.7.mlp.experts"].update(
        weight_scale_encoding="csf"),
    lambda config: config["quantization_config"].update(quant_method="compressed-tensors"),
    lambda config: config["quantization_config"].update(kv_cache_quant_algo="FP8"),
])
def test_gate_six_rejects_a_config_that_does_not_describe_the_release(build, mutation):
    config = _config()
    mutation(config)
    with pytest.raises(ValueError, match="gate 6"):
        is_trellismx_overlay(build(config=config, carrier_config=config))


def test_expected_shapes_follow_the_release_geometry():
    shapes = expected_tensor_shapes(_LOCAL)
    assert shapes["w13_trellis"] == (2, NUM_EXPERTS, HIDDEN_SIZE // 16, 32, 16 * BITS)
    assert shapes["w2_trellis"] == (NUM_EXPERTS, 32, HIDDEN_SIZE // 16, 16 * BITS)
    assert shapes["w13_scale_ue8m0"] == (NUM_EXPERTS, 1024, HIDDEN_SIZE // 32)
    assert shapes["w2_scale_ue8m0"] == (NUM_EXPERTS, HIDDEN_SIZE, 16)
    assert shapes["intermediate_scales_fp16"] == (NUM_EXPERTS, 1536)
    assert shapes["coupled_sign_draw_u8"] == (NUM_EXPERTS,)
    assert shapes["gate_up_suh_fp16"] == shapes["down_svh_fp16"] == (HIDDEN_SIZE,)
    with pytest.raises(ValueError, match="32-channel slots"):
        expected_tensor_shapes(1000)


def test_validate_sidecar_tensors_names_the_offending_tensor():
    tensors = {name: (TENSOR_DTYPES[name], shape) for name, shape in expected_tensor_shapes(_LOCAL).items()}
    validate_sidecar_tensors(tensors, layer=3, rank=0, local_intermediate_size=_LOCAL)
    broken = dict(tensors, w2_trellis=("I16", (NUM_EXPERTS, 32, HIDDEN_SIZE // 16, 32)))
    with pytest.raises(ValueError, match="w2_trellis"):
        validate_sidecar_tensors(broken, layer=3, rank=0, local_intermediate_size=_LOCAL)
    missing = dict(tensors)
    del missing["down_svh_fp16"]
    with pytest.raises(ValueError, match="down_svh_fp16"):
        validate_sidecar_tensors(missing, layer=3, rank=0, local_intermediate_size=_LOCAL)


def test_validate_sidecar_metadata_accepts_the_release_contract():
    metadata = dict(SIDECAR_IDENTITY, **SIDECAR_CONTRACT)
    metadata.update(layer="3", rank="2", local_atom_begin=str(2 * 512 // 32), source_design_sha256="a" * 64)
    validate_sidecar_metadata(metadata, layer=3, rank=2, world_size=WORLD_SIZE,
                              expected_design_sha256="a" * 64)


@pytest.mark.parametrize("name", sorted(SIDECAR_IDENTITY) + sorted(SIDECAR_CONTRACT))
def test_validate_sidecar_metadata_rejects_each_contract_field(name):
    metadata = dict(SIDECAR_IDENTITY, **SIDECAR_CONTRACT)
    metadata.update(layer="3", rank="0", local_atom_begin="0", source_design_sha256="a" * 64)
    metadata[name] = "not-the-release-contract"
    with pytest.raises(ValueError, match=name):
        validate_sidecar_metadata(metadata, layer=3, rank=0, world_size=WORLD_SIZE)


@pytest.mark.parametrize("field,value", [("layer", "4"), ("rank", "1"), ("local_atom_begin", "16"),
                                         ("source_design_sha256", "b" * 64),
                                         ("encoder_transform_sha256", "d" * 64)])
def test_validate_sidecar_metadata_rejects_a_mismatched_identity(field, value):
    metadata = dict(SIDECAR_IDENTITY, **SIDECAR_CONTRACT)
    metadata.update(layer="3", rank="0", local_atom_begin="0", source_design_sha256="a" * 64,
                    encoder_transform_sha256="c" * 64)
    validate_sidecar_metadata(dict(metadata), layer=3, rank=0, world_size=WORLD_SIZE,
                              expected_design_sha256="a" * 64, expected_transform_sha256="c" * 64)
    metadata[field] = value
    with pytest.raises(ValueError):
        validate_sidecar_metadata(metadata, layer=3, rank=0, world_size=WORLD_SIZE,
                                  expected_design_sha256="a" * 64, expected_transform_sha256="c" * 64)


def test_the_sidecar_header_reader_rejects_bytes_outside_the_declared_tensors(tmp_path):
    path = tmp_path / "extended.safetensors"
    _write_sidecar(path, layer=3, rank=0)
    assert set(sidecar_header(path)[1]) == set(TENSOR_DTYPES)
    with open(path, "ab") as file:
        file.write(b"\0")
    with pytest.raises(ValueError, match="cover"):
        sidecar_header(path)


@pytest.mark.parametrize("overrides", [{"world_size": 2}, {"hidden_size": 5120}, {"num_experts": 256},
                                       {"top_k": 6}, {"global_intermediate_size": 1024}])
def test_expert_loading_rejects_geometry_the_release_does_not_ship(build, unreachable_preparation, overrides):
    with pytest.raises(ValueError):
        _load(is_trellismx_overlay(build()), **overrides)


@pytest.mark.parametrize("limit", [7.0, 20.0, float("inf")])
def test_expert_loading_requires_the_glm_swiglu_clamp(build, unreachable_preparation, limit):
    """The sidecars encode SiLU capped at 10; preparation accepts no other clamp."""
    with pytest.raises(ValueError, match="swiglu_limit 10.0"):
        _load(is_trellismx_overlay(build()), swiglu_limit=limit)


def test_expert_loading_requires_bf16_activations(build, unreachable_preparation):
    import torch

    with pytest.raises(ValueError, match="BF16"):
        _load(is_trellismx_overlay(build()), params_dtype=torch.float16)


def test_expert_loading_rejects_a_layer_the_release_does_not_ship(build, unreachable_preparation):
    with pytest.raises(ValueError, match="no sidecar for layer 46"):
        _load(is_trellismx_overlay(build()), layer=46)


def test_the_default_loader_prepares_only_verified_bytes(build):
    """The loader reads, verifies, parses and prepares one sidecar from the same bytes."""
    import torch

    root = build(own={(3, 0): (("w13_trellis", (0x1234).to_bytes(2, "little")),
                               ("w2_trellis", (0x0BCD).to_bytes(2, "little")))})
    prepared = _load(is_trellismx_overlay(root))
    assert (prepared.num_experts, prepared.hidden_size, prepared.intermediate_size) == (
        NUM_EXPERTS, HIDDEN_SIZE, _LOCAL)
    assert prepared._impl.w1_fp4.dtype is torch.int16
    assert int(prepared._impl.w1_fp4.flatten()[0]) == 0x1234
    assert int(prepared._impl.w2_fp4.flatten()[0]) == 0x0BCD


@pytest.mark.parametrize("tensor", ["w13_trellis", "w2_trellis", "w13_scale_ue8m0", "intermediate_scales_fp16"])
def test_the_default_loader_rejects_same_size_payload_corruption(build, unreachable_preparation, tensor):
    """One byte changed after the manifest was written, the file size and header intact."""
    root = build(own={(3, 0): ()})
    release = is_trellismx_overlay(root)
    path = root / "sidecars" / _sidecar_name(3, 0)
    size = path.stat().st_size
    with open(path, "r+b") as file:
        file.seek(_payload_offset(path, tensor) + 4097)
        file.write(b"\x01")
    assert path.stat().st_size == size
    # The header and its gates still pass: only the manifest digest can catch this.
    metadata, tensors = sidecar_header(path)
    validate_sidecar_metadata(metadata, layer=3, rank=0, world_size=WORLD_SIZE)
    validate_sidecar_tensors(tensors, layer=3, rank=0, local_intermediate_size=_LOCAL)
    with pytest.raises(ValueError, match="SHA-256 is [0-9a-f]{64}, but the release manifest declares"):
        _load(release)
    with pytest.raises(ValueError, match="SHA-256"):
        load_sidecar_tensors(release, layer=3, rank=0, device="cpu", names=("coupled_sign_draw_u8",))


def test_the_default_loader_rejects_a_mismatched_manifest_digest(build, manifest, unreachable_preparation):
    document = manifest()
    entry = next(entry for entry in document["files"] if (entry["layer"], entry["rank"]) == (3, 0))
    entry["sha256"] = entry["sha256"][::-1]
    release = is_trellismx_overlay(build(manifest=document))
    with pytest.raises(ValueError, match="SHA-256 is [0-9a-f]{64}, but the release manifest declares"):
        _load(release)


def test_the_default_loader_rejects_a_sidecar_resized_after_detection(build, unreachable_preparation):
    root = build(own={(3, 0): ()})
    release = is_trellismx_overlay(root)
    with open(root / "sidecars" / _sidecar_name(3, 0), "ab") as file:
        file.write(b"\0" * 8)
    with pytest.raises(ValueError, match="but the release manifest declares"):
        _load(release)


def test_the_expert_bridge_names_the_missing_source_format(monkeypatch):
    monkeypatch.setitem(sys.modules, "b12x.moe.fused_moe.trellismx", None)
    with pytest.raises(RuntimeError, match="trellismx source format"):
        prepare_trellismx_experts({}, {}, num_experts=NUM_EXPERTS, hidden_size=HIDDEN_SIZE,
                                  intermediate_size=_LOCAL, top_k=TOP_K, device="cpu")


def _prepared_with(swiglu_limit, nonlinearity="silu", clamp_after_silu=False):
    activation = types.SimpleNamespace(nonlinearity=nonlinearity, swiglu_limit=swiglu_limit,
                                       swiglu_clamp_after_silu=clamp_after_silu)
    return types.SimpleNamespace(plan=types.SimpleNamespace(activation=activation))


def test_the_expert_bridge_forwards_the_sidecar_to_b12x(monkeypatch):
    calls, prepared = {}, _prepared_with(SWIGLU_LIMIT)

    def prepare(tensors, metadata, **kwargs):
        calls.update(tensors=tensors, metadata=metadata, **kwargs)
        return prepared

    module = types.ModuleType("b12x.moe.fused_moe.trellismx")
    module.prepare_trellismx_experts = prepare
    monkeypatch.setitem(sys.modules, "b12x.moe.fused_moe.trellismx", module)
    tensors, metadata = {"w13_trellis": object()}, {"bits": "4"}
    assert prepare_trellismx_experts(
        tensors, metadata, num_experts=NUM_EXPERTS, hidden_size=HIDDEN_SIZE,
        intermediate_size=_LOCAL, top_k=TOP_K, device="cpu") is prepared
    assert calls == {"tensors": tensors, "metadata": metadata, "num_experts": NUM_EXPERTS,
                     "hidden_size": HIDDEN_SIZE, "intermediate_size": _LOCAL, "top_k": TOP_K, "device": "cpu"}


@pytest.mark.parametrize("prepared", [_prepared_with(20.0), _prepared_with(float("inf")),
                                      _prepared_with(SWIGLU_LIMIT, clamp_after_silu=True),
                                      _prepared_with(SWIGLU_LIMIT, nonlinearity="relu2")])
def test_the_expert_bridge_rejects_a_bundle_without_the_glm_clamp(monkeypatch, prepared):
    module = types.ModuleType("b12x.moe.fused_moe.trellismx")
    module.prepare_trellismx_experts = lambda tensors, metadata, **kwargs: prepared
    monkeypatch.setitem(sys.modules, "b12x.moe.fused_moe.trellismx", module)
    with pytest.raises(ValueError, match="SwiGLU clamp 10.0"):
        prepare_trellismx_experts({}, {}, num_experts=NUM_EXPERTS, hidden_size=HIDDEN_SIZE,
                                  intermediate_size=_LOCAL, top_k=TOP_K, device="cpu")


# A stand-in for the P8 runtime, exposing exactly what b12x's attachment reads from P8NativeTPMoE
# with the values the release is served with. Its module lives in a temporary package that its own
# backend declaration pins.
_P8_STAND_IN = """
import torch
from b12x.moe._shared.p8.p8_native_kernel import P8ProgramSpec


def compile_stand_in_programs(spec, tokens, ordinal):
    raise AssertionError("the stand-in was compiled")


class StandInP8Runtime:
    def __init__(self, tensors, record, *, top_k):
        local = int(record.intermediate_size)
        self.device = record.w13.device
        self.experts, self.hidden, self.intermediate = int(record.num_experts), int(record.hidden_size), local
        self.topk, self.trellis_bits, self.swiglu_limit = int(top_k), int(record.trellis_bits), 10.0
        self.full_coupled, self.deterministic_output = True, True
        self.debug_capture = self.compact_scale_storage = False
        self.p8_down_remainder, self._dx2_phases = False, "both"
        self.p8_dx2_rowpack = self.p8_input_rowpack = self.rp2_skip_input_clear = self.p8_epi_par = True
        self.rp2_direct_max_tokens, self.rp2_direct_tile_m = 16, 0
        self.small_m_scheduler, self.fc1_tile_n, self.fc1_pipeline_stages, self.fc1_warps = True, 128, 2, 4
        self.fc1_warp_quant, self.fc1_broadcast_a, self.fc1_a_swizzle_rotate = False, True, False
        self.fc1_exact_staging, self.prefill_chunk_tokens = False, 0
        self.grid_policy = True
        self.tile_major_tasks = self.compact_input_storage = False
        self.force_materialized = self.mac_override = None
        self.w13_stream = tensors["w13_trellis"].view(torch.int32).reshape(-1)
        self.w2_stream = tensors["w2_trellis"].view(torch.int32).reshape(-1)
        self.w13_scale_mx = tensors["w13_scale_ue8m0"].reshape(-1)
        self.w2_scale_mx = tensors["w2_scale_ue8m0"].reshape(-1)
        self.w13_sfb = record.w13_sfb.clone().reshape(-1)
        self.w2_sfb = record.w2_sfb.clone().reshape(-1)
        self.scale_component_packed = torch.cat((
            tensors["gate_up_suh_fp16"].reshape(-1), tensors["intermediate_scales_fp16"].reshape(-1),
            tensors["down_svh_fp16"].reshape(-1), record.intermediate_rotations[0, 3 * local:]))
        # What a real runtime declares so a plan can size its arena and its programs.
        self.program_spec = P8ProgramSpec(
            trellis_bits=self.trellis_bits, experts=self.experts, hidden=self.hidden,
            intermediate=self.intermediate, topk=self.topk, swiglu_limit=self.swiglu_limit,
            full_coupled=self.full_coupled, scale_sandwich=False, mac_override=self.mac_override,
            deterministic_output=self.deterministic_output, small_m_scheduler=self.small_m_scheduler,
            force_materialized=self.force_materialized, rp2_direct_max_tokens=self.rp2_direct_max_tokens,
            rp2_skip_input_clear=self.rp2_skip_input_clear, prefill_chunk_tokens=self.prefill_chunk_tokens,
            compact_input_storage=self.compact_input_storage, p8_down_remainder=self.p8_down_remainder,
            dx2_phases=self._dx2_phases, p8_dx2_rowpack=self.p8_dx2_rowpack,
            p8_input_rowpack=self.p8_input_rowpack, p8_epi_par=self.p8_epi_par,
            tile_major_tasks=self.tile_major_tasks, fc1_tile_n=self.fc1_tile_n,
            rp2_direct_tile_m=self.rp2_direct_tile_m, grouped_tile_m=0, rp2_stride=192,
            fc1_warps=self.fc1_warps, fc1_pipeline_stages=self.fc1_pipeline_stages,
            fc1_a_swizzle_rotate=self.fc1_a_swizzle_rotate, fc1_warp_quant=self.fc1_warp_quant,
            fc1_exact_staging=self.fc1_exact_staging, fc1_broadcast_a=self.fc1_broadcast_a)
        self.program_declaration = (__name__ + ":compile_stand_in_programs", self.program_spec)

    def __call__(self, x, topk_weights, topk_ids, out=None):
        raise AssertionError("the P8 runtime was launched")
"""

# What a compiler worker does with a lent forward: load it in a fresh interpreter, which cannot
# import the stand-in's package, and inspect the experts it would trace.
_COMPILE_WORKER = """
import sys
from pathlib import Path

from b12x.moe.fused_moe.trellismx_p8 import TrellisMxP8Unavailable, trellismx_p8_attachment

from ignition.preparation.forward_metadata import loads

forward = loads(Path(sys.argv[1]).read_bytes())
entry = trellismx_p8_attachment(forward.experts)
assert entry is not None and entry.live is False, entry
imported = sorted(name for name in sys.modules if name.split(".")[0] == sys.argv[2])
assert not imported, imported
try:
    entry.runtime
except TrellisMxP8Unavailable:
    print("inert", entry.identity)
"""


def test_compile_workers_receive_the_p8_attachment_without_its_runtime(tmp_path, monkeypatch):
    """A forward holding an attached bundle crosses Flare's compile pool; the runtime stays in its rank."""
    import importlib
    import subprocess

    import torch
    from b12x.moe import fused_moe
    from b12x.moe.fused_moe import trellismx as source_format
    from b12x.moe.fused_moe.trellismx_p8 import (
        TrellisMxP8Backend, attach_trellismx_p8_runtime, trellismx_p8_attachment,
    )

    from ignition.preparation.forward_metadata import dumps

    package = f"flare_p8_stand_in_{os.getpid()}"
    (tmp_path / package).mkdir()
    (tmp_path / package / "__init__.py").write_text("")
    (tmp_path / package / "runtime.py").write_text(_P8_STAND_IN)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    runtime_class = importlib.import_module(f"{package}.runtime").StandInP8Runtime
    experts, hidden, local, top_k = 4, 512, 128, 2
    generator = torch.Generator().manual_seed(7)
    tensors = {
        "w13_trellis": torch.randint(0, 1 << 15, (2, experts, hidden // 16, local // 16, 16 * BITS),
                                     generator=generator, dtype=torch.int16),
        "w2_trellis": torch.randint(0, 1 << 15, (experts, local // 16, hidden // 16, 16 * BITS),
                                    generator=generator, dtype=torch.int16),
        "w13_scale_ue8m0": torch.full((experts, 2 * local, hidden // 32), 127, dtype=torch.uint8),
        "w2_scale_ue8m0": torch.full((experts, hidden, local // 32), 127, dtype=torch.uint8),
        "intermediate_scales_fp16": torch.ones(experts, 3 * local, dtype=torch.float16),
        "coupled_sign_draw_u8": torch.zeros(experts, dtype=torch.uint8),
        "gate_up_suh_fp16": torch.ones(hidden, dtype=torch.float16),
        "down_svh_fp16": torch.ones(hidden, dtype=torch.float16),
    }
    metadata = {"law": "procedural-npk", "compander_scale": "2.0", "alphabet": "e4m3", "scale": "ue8m0-k32",
                "boundary": "coupled-h512-h128-suh-svh-v1", "activation": "silu-cap10", "bits": str(BITS),
                "hidden": str(hidden), "experts_start": "0", "experts_end": str(experts),
                "intermediate_per_rank": str(local), "topk": str(top_k), "world_size": "1",
                "slice_begin": "0", "slice_end": str(local)}
    prepared = source_format.prepare_trellismx_experts(
        tensors, metadata, num_experts=experts, hidden_size=hidden, intermediate_size=local, top_k=top_k,
        device=torch.device("cpu"))
    runtime = runtime_class(tensors, prepared._impl.representation.value, top_k=top_k)
    digest = hashlib.sha256((tmp_path / package / "runtime.py").read_bytes()).hexdigest()
    backend = TrellisMxP8Backend(package=package, files={"runtime.py": digest}, upstream="stand-in")
    # The stand-in carries the serving configuration's values and declares its programs.
    attached = attach_trellismx_p8_runtime(prepared, runtime, backend=backend)
    plan = fused_moe.plan_execution(experts=attached, capacity=fused_moe.ExecutionCapacity(max_tokens=8, top_k=top_k))
    forward = types.SimpleNamespace(experts=attached, plan=plan)

    payload = dumps(forward)
    assert b"StandInP8Runtime" not in payload, "the runtime must never reach a compiler worker"
    # Serializing leaves the rank's own forward running its runtime.
    live = trellismx_p8_attachment(forward.experts)
    assert live.live is True and live.runtime is runtime
    (tmp_path / "forward.bin").write_bytes(payload)
    environment = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
    inherited = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    environment.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=os.pathsep.join(inherited))
    worker = subprocess.run([sys.executable, "-c", _COMPILE_WORKER, str(tmp_path / "forward.bin"), package],
                            cwd=Path(__file__).resolve().parents[2], env=environment, capture_output=True,
                            text=True, timeout=300)
    assert worker.returncode == 0, worker.stderr
    assert worker.stdout.split() == ["inert", live.identity]
    for name in [name for name in sys.modules if name.split(".")[0] == package]:
        del sys.modules[name]


def test_the_recipe_attaches_only_b12xs_declared_serving_configuration(monkeypatch):
    """The recipe owns no P8 option: preparation records and builds with b12x's declared
    GLM-5.3 Flash serving configuration, and attach rejects a runtime that resolved another."""
    from b12x.moe.fused_moe import trellismx_p8
    from b12x.moe.fused_moe.trellismx_p8_runtime import GLM53_SERVING_OPTIONS, P8RuntimeOptions

    from ignition.recipes.glm53.trellismx import attach_p8, p8_settings

    assert p8_settings()["options"] == GLM53_SERVING_OPTIONS.as_dict()
    attached = {}
    monkeypatch.setattr(trellismx_p8, "attach_trellismx_p8_runtime",
                        lambda prepared, runtime, **kwargs: attached.update(
                            prepared=prepared, runtime=runtime, **kwargs) or "attached")
    runtime = types.SimpleNamespace(options=GLM53_SERVING_OPTIONS)
    assert attach_p8("bundle", runtime) == "attached"
    assert attached["prepared"] == "bundle" and attached["runtime"] is runtime
    assert attached["backend"].package == "b12x" and set(attached) == {"prepared", "runtime", "backend"}
    with pytest.raises(ValueError, match="serving"):
        attach_p8("bundle", types.SimpleNamespace(options=P8RuntimeOptions(fc1_tile_n=64)))


def test_the_report_requires_a_live_runtime_on_every_trellismx_layer(monkeypatch):
    from b12x.moe.fused_moe import trellismx_p8

    from ignition.recipes.glm53.trellismx import p8_attachments
    from ignition.recipes.glm53.weights import MoEWeights

    def moe(experts, release=object()):
        return types.SimpleNamespace(mlp=MoEWeights(None, None, experts, None, trellismx_release=release))

    def entry(live):
        options = types.SimpleNamespace(as_dict=lambda: {"down_remainder": "rp2", "mac_override": None})
        return types.SimpleNamespace(live=live, identity="a" * 64, options=options)

    attached = {"layer": entry(True), "mtp": entry(True), "plain": None}
    monkeypatch.setattr(trellismx_p8, "trellismx_p8_attachment", lambda experts: attached[experts])
    # A dense layer, a TrellisMX layer and the carrier's own experts, then the MTP layer.
    layers = [types.SimpleNamespace(mlp=types.SimpleNamespace()), moe("layer"), moe("plain", release=None)]
    weights = types.SimpleNamespace(layers=layers, mtp=moe("mtp"))
    assert p8_attachments(weights) == {"identities": ["a" * 64],
                                       "options": [{"down_remainder": "rp2", "mac_override": None}]}
    attached["mtp"] = entry(False)
    with pytest.raises(RuntimeError, match="without a live P8 runtime"):
        p8_attachments(weights)
    attached["mtp"] = None
    with pytest.raises(RuntimeError, match="without a live P8 runtime"):
        p8_attachments(weights)


def test_detection_hashes_the_manifest_bytes_it_parsed(build, manifest, monkeypatch):
    """A manifest replaced while the sidecar headers are read cannot lend the release its digest."""
    from ignition.recipes.glm53 import trellismx

    root = build()
    original = (root / "trellismx-manifest.json").read_bytes()
    replacement = manifest()
    replacement["carrier"]["revision"] = "f" * 40
    reader = trellismx.sidecar_header

    def header(path):
        (root / "trellismx-manifest.json").write_text(json.dumps(replacement))
        return reader(path)

    monkeypatch.setattr(trellismx, "sidecar_header", header)
    release = is_trellismx_overlay(root)
    assert release.manifest_sha256 == hashlib.sha256(original).hexdigest()
    assert release.carrier_revision == json.loads(original)["carrier"]["revision"]


# What rank 1 detects while rank 0 detects manifest "a..." with transform "b...".
_AGREEMENT_CASES = {
    "same": ("a" * 64, "b" * 64),
    "other manifest": ("c" * 64, "b" * 64),
    "other transform": ("a" * 64, "c" * 64),
    "no release": None,
}


def _agreement_rank(rank, rendezvous, results):
    import torch.distributed as dist

    from ignition.recipes.glm53.trellismx import agree_on_release

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        outcomes = {}
        for case, detected in _AGREEMENT_CASES.items():
            identity = ("a" * 64, "b" * 64) if rank == 0 else detected
            release = None if identity is None else types.SimpleNamespace(
                manifest_sha256=identity[0], transform_sha256=identity[1])
            try:
                agree_on_release(release, dist.group.WORLD)
                outcomes[case] = "agreed"
            except ValueError as error:
                outcomes[case] = str(error)
        (Path(results) / f"{rank}.json").write_text(json.dumps(outcomes))
    finally:
        dist.destroy_process_group()


def test_ranks_must_detect_the_same_release_before_loading(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_agreement_rank, args=((tmp_path / "rendezvous").as_uri(), str(tmp_path)), nprocs=2, join=True)
    for rank in range(2):
        outcomes = json.loads((tmp_path / f"{rank}.json").read_text())
        assert outcomes.pop("same") == "agreed"
        assert all("detected different TrellisMX release" in outcome for outcome in outcomes.values()), outcomes


def _qualify(tmp_path, body, environment):
    """Run the qualification entry point over a one-module suite in a fresh interpreter."""
    import subprocess

    module = tmp_path / "test_suite.py"
    module.write_text(body)
    script = Path(__file__).resolve().parents[2] / "scripts" / "qualify_glm53_trellismx.py"
    driver = ("import runpy, sys; main = runpy.run_path(sys.argv[1])['main'];"
              " sys.exit(main(['--release', sys.argv[2]], tests=(sys.argv[3],)))")
    return subprocess.run([sys.executable, "-c", driver, str(script), str(tmp_path), str(module)],
                          env={**os.environ, **environment}, capture_output=True, text=True, timeout=300)


def test_qualification_fails_on_any_skip(tmp_path):
    passing = "def test_runs():\n    pass\n"
    skipping = "import pytest\n\n\ndef test_runs():\n    pass\n\n\n@pytest.mark.skip(reason='absent')\ndef test_skips():\n    pass\n"
    assigned = {"CUDA_VISIBLE_DEVICES": "qualification-test"}
    result = _qualify(tmp_path, passing, assigned)
    assert result.returncode == 0 and "qualification PASSED" in result.stdout, result.stdout + result.stderr
    result = _qualify(tmp_path, skipping, assigned)
    assert result.returncode == 1 and "skipped in setup" in result.stdout, result.stdout + result.stderr
    result = _qualify(tmp_path, passing, {"CUDA_VISIBLE_DEVICES": ""})
    assert result.returncode == 2 and "CUDA_VISIBLE_DEVICES" in result.stderr


@requires_release
def test_detection_accepts_the_real_release():
    release = is_trellismx_overlay(RELEASE)
    assert release is not None
    assert release.manifest_sha256 == "99d189ded46416baed158b195c7f087a2ee7b37108f7596463a43db0c7808fb2"
    assert release.layers == _LAYERS
    assert (release.world_size, release.bits, release.mtp_layer) == (WORLD_SIZE, BITS, MTP_LAYER)
    assert release.carrier_repository == "local-inference-lab/GLM-5.3-Flash-NVFP4"
    assert release.carrier_revision == "520de24eabf507659eaef7c70f14fd584527facc"
    assert len(release.files) == 172
    assert sum(entry.bytes for entry in release.files if entry.layer == MTP_LAYER) == 3_853_990_880
    assert release.carrier_root.is_dir()


@requires_release
def test_the_real_sidecar_metadata_carries_the_coupled_contract():
    metadata, tensors = sidecar_header(RELEASE / "sidecars/p8-layer-003-tp4-rank-0.safetensors")
    validate_sidecar_metadata(metadata, layer=3, rank=0, world_size=WORLD_SIZE,
                              expected_transform_sha256=_sha256(RELEASE / "design/transform.json"))
    validate_sidecar_tensors(tensors, layer=3, rank=0, local_intermediate_size=_LOCAL)
    assert metadata["sha256_coupled_signs_fp16"] == \
        "95693b3d933cef1caca1d4aed176789faff9746303ec5f8bcac4395c68e25ff0"
    assert metadata["runtime_regenerated_sign_bytes"] == "3072"


@requires_release
@pytest.mark.parametrize("layer,rank", [(3, 0), (44, 3), (MTP_LAYER, 1)])
def test_the_real_sidecars_load_after_their_manifest_digests_verify(layer, rank):
    """Every tensor the loader returns, the trellis weight payloads included, matches its recorded hash."""
    import numpy

    release = is_trellismx_overlay(RELEASE)
    tensors = load_sidecar_tensors(release, layer=layer, rank=rank, device="cpu")
    metadata, _ = sidecar_header(release.sidecar(layer, rank).path)
    assert set(tensors) == set(TENSOR_DTYPES)
    for name, tensor in tensors.items():
        digest = hashlib.sha256(numpy.ascontiguousarray(tensor.numpy()).tobytes()).hexdigest()
        assert digest == metadata[f"sha256_{name}"], name


@requires_release
def test_the_default_loader_rejects_a_real_sidecar_its_manifest_does_not_describe(tmp_path, unreachable_preparation):
    """The real release with one manifest digest changed: detection passes and loading refuses."""
    root = tmp_path / "release"
    root.mkdir()
    for name in ("carrier", "sidecars", "design", "config.json"):
        (root / name).symlink_to(RELEASE / name)
    document = json.loads((RELEASE / "trellismx-manifest.json").read_text())
    entry = next(entry for entry in document["files"] if (entry["layer"], entry["rank"]) == (3, 0))
    entry["sha256"] = hashlib.sha256(entry["sha256"].encode()).hexdigest()
    (root / "trellismx-manifest.json").write_text(json.dumps(document))
    release = is_trellismx_overlay(root)
    with pytest.raises(ValueError, match="but the release manifest declares"):
        _load(release)


@requires_release
def test_the_declared_expansion_matches_the_prepared_bundle():
    """The recipe's resident-weight estimate counts what b12x's preparation keeps per sidecar.

    Each sidecar prepares into its payload plus one repacked copy of both UE8M0 planes and
    the rotations. On CUDA the attached P8 runtime's byte-identical planes replace the
    bundle's, and the recipe adds the runtime's packed boundary vectors and small tensors.
    """
    import torch

    from ignition.recipes.glm53.recipe import RECIPE

    prepared = _load(is_trellismx_overlay(RELEASE))
    storages, pending, visited = {}, [prepared], set()
    while pending:
        value = pending.pop()
        if id(value) in visited:
            continue
        visited.add(id(value))
        if isinstance(value, torch.Tensor):
            storages[value.untyped_storage().data_ptr()] = value.untyped_storage().nbytes()
        elif isinstance(value, (tuple, list)):
            pending.extend(value)
        elif hasattr(value, "__dataclass_fields__"):
            pending.extend(getattr(value, name) for name in value.__dataclass_fields__)
    # Packed FP16 boundary vectors, FP32 ones per expert, and three one- or two-byte tensors.
    p8 = 2 * (2 * HIDDEN_SIZE + NUM_EXPERTS * 3 * _LOCAL + 3 * _LOCAL) + 4 * NUM_EXPERTS + 4
    checkpoint = next(c for c in RECIPE.checkpoints if c.alias == "glm-5.3-flash-trellismx")
    release = is_trellismx_overlay(RELEASE)
    assert checkpoint.expansion_bytes == len(release.files) * (sum(storages.values()) + p8) - sum(
        entry.bytes for entry in release.files)


@requires_release
def test_the_release_runtime_is_built_and_attached_as_declared():
    """Layer 3 rank 0 through the recipe's own runtime construction and attach, on the CPU.

    The runtime is b12x's in-tree P8 runtime, built from the verified sidecar with b12x's
    declared serving configuration. It must resolve that configuration and the serving
    launch geometry, and b12x must accept it under the in-tree backend declaration and
    verify it against the bundle. Running it, which needs CUDA and the programs b12x
    preparation compiles, is left to the GPU tests.
    """
    import torch
    from b12x.moe._shared.p8.launch import P8LaunchConfig
    from b12x.moe.fused_moe.trellismx_p8 import (
        TrellisMxP8Options, trellismx_p8_attachment, verify_trellismx_p8_attachment,
    )
    from b12x.moe.fused_moe.trellismx_p8_runtime import GLM53_SERVING_OPTIONS, in_tree_p8_backend

    from ignition.recipes.glm53.trellismx import (
        attach_p8, p8_attachments, p8_runtime, p8_settings, prepared_metadata,
    )

    release = is_trellismx_overlay(RELEASE)
    entry = release.sidecar(3, 0)
    metadata, _ = sidecar_header(entry.path)
    payload = load_sidecar_tensors(release, layer=3, rank=0, device="cpu")
    prepared = prepare_trellismx_experts(
        payload, prepared_metadata(metadata, global_intermediate_size=GLOBAL_INTERMEDIATE),
        num_experts=NUM_EXPERTS, hidden_size=HIDDEN_SIZE, intermediate_size=_LOCAL, top_k=TOP_K,
        device=torch.device("cpu"))
    runtime = p8_runtime(release, entry, metadata, payload, layer=3, rank=0, world_size=WORLD_SIZE,
                         local_intermediate_size=_LOCAL, device="cpu")
    assert runtime.options == GLM53_SERVING_OPTIONS
    serving = P8LaunchConfig.serving()
    resolved = TrellisMxP8Options.from_runtime(runtime).as_dict()
    assert (resolved["rp2_direct_tile_m"], resolved["fc1_pipeline_stages"], resolved["fc1_warps"]) == (
        serving.direct_tile_m, serving.fc1_pipeline_stages, serving.fc1_warps)
    attached = attach_p8(prepared, runtime)
    verify_trellismx_p8_attachment(attached)
    attachment = trellismx_p8_attachment(attached)
    assert attachment.live and attachment.runtime is runtime
    assert attachment.backend.identity == in_tree_p8_backend().identity == p8_settings()["backend"]["identity"]
    from ignition.recipes.glm53.weights import MoEWeights

    layer = types.SimpleNamespace(mlp=MoEWeights(None, None, attached, None, trellismx_release=release))
    assert p8_attachments(types.SimpleNamespace(layers=[layer], mtp=None)) == {
        "identities": [attachment.identity], "options": [attachment.options.as_dict()]}


@requires_release
def test_the_real_sidecar_prepares_through_the_source_format():
    """The release's own tensors through b12x's trellismx preparation, on CPU.

    The carriers must come back byte-identical: preparation maps them into the
    W4A8 trellis prepared layout without touching a codeword. A b12x without the
    source format, or one that rejects the release's metadata, fails here.
    """
    import torch

    release = is_trellismx_overlay(RELEASE)
    prepared = _load(release)
    assert (prepared.num_experts, prepared.hidden_size, prepared.intermediate_size) == (
        NUM_EXPERTS, HIDDEN_SIZE, _LOCAL)
    sidecar = load_sidecar_tensors(release, layer=3, rank=0, device="cpu", names=("w13_trellis", "w2_trellis"))
    assert prepared._impl.w1_fp4.dtype is torch.int16
    assert torch.equal(prepared._impl.w1_fp4, sidecar["w13_trellis"])
    assert torch.equal(prepared._impl.w2_fp4, sidecar["w2_trellis"])


@requires_release
@pytest.mark.parametrize("rank", [0, 3])
def test_each_rank_prepares_its_own_sign_slice(rank):
    """Rank 0 cannot distinguish an atom index from a channel index; rank 3 can.

    The prepared rotations must be the sidecar's values with b12x's sign vector
    appended for the rank's own global extent, bit for bit.
    """
    import torch
    from b12x.moe.fused_moe.source import TrellisExtent
    from b12x.moe.fused_moe.trellis_layout import append_intermediate_signs

    release = is_trellismx_overlay(RELEASE)
    prepared = _load(release, rank=rank)
    sidecar = load_sidecar_tensors(release, layer=3, rank=rank, device="cpu",
                                   names=("intermediate_scales_fp16", "coupled_sign_draw_u8"))
    expected = append_intermediate_signs(
        sidecar["intermediate_scales_fp16"], sidecar["coupled_sign_draw_u8"],
        extent=TrellisExtent(global_intermediate_size=GLOBAL_INTERMEDIATE,
                             first_slot=rank * _LOCAL // 32, slot_count=_LOCAL // 32))
    rotations = prepared._impl.representation.value.intermediate_rotations
    assert rotations.dtype is torch.float16 and rotations.shape == expected.shape
    assert torch.equal(rotations, expected)


@pytest.fixture(scope="module")
def p8_layers():
    """Attached P8 experts per (layer, rank), each loaded once through the recipe path."""
    import torch

    release, loaded = is_trellismx_overlay(RELEASE), {}

    def load(layer, rank):
        if (layer, rank) not in loaded:
            from b12x.moe.fused_moe.trellismx_p8 import verify_trellismx_p8_attachment

            device = torch.device("cuda", torch.cuda.current_device())
            prepared = _load(release, layer=layer, rank=rank, device=device)
            # Every attach-time comparison again, against device memory.
            verify_trellismx_p8_attachment(prepared)
            loaded[layer, rank] = prepared, load_sidecar_tensors(release, layer=layer, rank=rank, device="cpu")
        return loaded[layer, rank]

    yield load
    loaded.clear()
    torch.cuda.empty_cache()


@requires_cuda
@requires_release
def test_an_attached_layer_occupies_its_declared_bytes_and_releases_them():
    """The recipe's resident estimate on the device, and no owner left behind when the bundle goes."""
    import gc

    import torch

    from ignition.recipes.glm53.recipe import RECIPE

    release = is_trellismx_overlay(RELEASE)
    checkpoint = next(c for c in RECIPE.checkpoints if c.alias == "glm-5.3-flash-trellismx")
    declared = (checkpoint.expansion_bytes + sum(entry.bytes for entry in release.files)) // len(release.files)
    device = torch.device("cuda", torch.cuda.current_device())
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated(device)
    prepared = _load(release, device=device)
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated(device) - before
    # The caching allocator rounds each of the bundle's and runtime's few tensors up to 512 bytes.
    assert abs(resident - declared) < 1 << 20, f"{resident} resident bytes, {declared} declared"
    del prepared
    gc.collect()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated(device) == before


@requires_cuda
@requires_release
def test_the_real_sidecar_copies_to_the_device_unchanged():
    """The whole verified sidecar on the assigned GPU, byte-for-byte what the host read."""
    import torch

    release = is_trellismx_overlay(RELEASE)
    device = torch.device("cuda", torch.cuda.current_device())
    host = load_sidecar_tensors(release, layer=3, rank=0, device="cpu")
    moved = load_sidecar_tensors(release, layer=3, rank=0, device=device)
    assert set(moved) == set(host)
    for name, tensor in moved.items():
        assert tensor.device.type == "cuda"
        assert tensor.dtype == host[name].dtype and tensor.shape == host[name].shape
        assert torch.equal(tensor.cpu(), host[name])
    del moved
    torch.cuda.empty_cache()


@requires_cuda
@requires_release
@pytest.mark.parametrize("layer,rank", [(3, 0), (3, 3), (MTP_LAYER, 1)])
@pytest.mark.parametrize("rows", [1, 2, 16, 17, 128, 129, 256])
def test_p8_experts_match_the_reference_on_gpu(p8_layers, layer, rank, rows):
    """One routed layer and rank through the recipe path, one MoE forward against the b12x reference.

    The rows cross every boundary of the P8 tile policy under b12x's serving launch
    configuration, which an untuned plan materializes: direct RP2 at 1 and 2..16 rows,
    grouped at 17..128 and above 128. The bound is the b12x-level comparison's: a BF16
    output of a four-term FP32 accumulation chain, so eight BF16 ulps of the largest
    output magnitude. The P8 runtime reduces the top-k contributions in a fixed order,
    so graph replay must reproduce the eager result bit for bit.
    """
    import torch
    from b12x.moe.fused_moe import ExecutionCapacity, bind, plan_execution, run
    from b12x.moe.fused_moe.trellismx_p8 import trellismx_p8_attachment
    from b12x.moe.fused_moe.trellismx_p8_runtime import GLM53_SERVING_OPTIONS
    from b12x.testing.reference.trellismx_moe import trellismx_moe_reference

    prepared, sidecar = p8_layers(layer, rank)
    assert trellismx_p8_attachment(prepared) is not None, "the recipe path must attach the P8 runtime"
    device = torch.device("cuda", torch.cuda.current_device())
    meta = {"hidden_size": HIDDEN_SIZE, "intermediate_size": _LOCAL, "num_experts": NUM_EXPERTS, "top_k": TOP_K,
            "bits": BITS, "rank": rank, "world_size": WORLD_SIZE}

    generator = torch.Generator(device="cpu").manual_seed(20261022 + 1000 * layer + 100 * rank + rows)
    x = (torch.randn(rows, HIDDEN_SIZE, generator=generator) * 0.02).to(device).to(torch.bfloat16)
    # Mixed routing: every row takes a different, unordered set of 8 of 288 experts.
    ids = torch.stack([torch.randperm(NUM_EXPERTS, generator=generator)[:TOP_K] for _ in range(rows)])
    weights = torch.rand(rows, TOP_K, generator=generator, dtype=torch.float32)
    weights = weights / weights.sum(-1, keepdim=True)
    ids, weights = ids.to(device=device, dtype=torch.int32), weights.to(device)

    # The direct RP2 arm keeps activation precision and follows the unquantized reference;
    # the grouped arm quantizes both GEMM inputs.
    reference_meta = dict(meta)
    if rows <= GLM53_SERVING_OPTIONS.rp2_direct_max_tokens:
        reference_meta["quantize_a"] = False
        reference_meta["quantize_down"] = False
    reference = trellismx_moe_reference(x.cpu(), ids.cpu(), weights.cpu(), sidecar, reference_meta).to(device)

    output = torch.zeros((rows, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
    execution = plan_execution(experts=prepared,
                               capacity=ExecutionCapacity(max_tokens=rows, top_k=TOP_K, route_num_experts=0))
    scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=device) for spec in execution.scratch_specs())
    binding = bind(execution, a=x, topk_ids=ids, topk_weights=weights, output=output, scratch=scratch,
                   input_scales_static=True)
    run(binding=binding)
    torch.cuda.synchronize()
    eager = output.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(binding=binding)
    # Replay must write every output element itself, not inherit the eager result.
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    captured = output.clone()

    assert bool(torch.isfinite(eager.float()).all())
    bound = 8 * float(reference.float().abs().max()) * 2**-8
    assert float((eager.float() - reference.float()).abs().max()) <= bound
    assert torch.equal(captured, eager), "graph replay differs from the eager P8 result"
