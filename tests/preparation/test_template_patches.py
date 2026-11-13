# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify exact-source patching, HF selection, and exported template bytes."""

import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from ignition.preparation.frontend import annotated_template_asset

ROOT = Path(__file__).resolve().parents[2]
PATCHES = sorted((ROOT / "ignition/recipes").glob("*/assets/*.patch.json"))


def fixture(recipe, filename):
    directory = os.environ.get("FLARE_TEST_TEMPLATE_DIR")
    assert directory, "run tests/frontend/prepare_templates.py and set FLARE_TEST_TEMPLATE_DIR"
    return Path(directory) / recipe / filename


def test_annotations_preserve_utf8_and_require_exact_source(tmp_path):
    source = "{{ messages[0].content }} café 世界".encode()
    expected = source + b"{%- mark -%}"
    patch = dict(source_sha256=hashlib.sha256(source).hexdigest(),
                 result_sha256=hashlib.sha256(expected).hexdigest(), insertions=[len(source)])
    path = tmp_path / "chat_template.patch.json"
    path.write_text(json.dumps(patch))
    assert annotated_template_asset(source, path) == {
        "text": expected.decode(), "sha256": hashlib.sha256(expected).hexdigest()}
    with pytest.raises(ValueError, match="checkpoint chat template does not match"):
        annotated_template_asset(source + b" ", path)
    patch["insertions"] = [0]
    path.write_text(json.dumps(patch))
    with pytest.raises(ValueError, match="patched chat template digest"):
        annotated_template_asset(source, path)
    for offsets in ([len(source) + 1], [-1], [True], [2, 1]):
        patch["insertions"] = offsets
        path.write_text(json.dumps(patch))
        with pytest.raises(ValueError, match="insertion offset"):
            annotated_template_asset(source, path)


@pytest.mark.parametrize("patch_path", PATCHES, ids=lambda p: f"{p.parent.parent.name}/{p.name}")
def test_recipe_patches_preserve_previously_exported_bytes(patch_path, tmp_path):
    recipe = patch_path.parent.parent.name
    filename = patch_path.name.replace(".patch.json", ".jinja")
    source = fixture(recipe, filename.replace(".jinja", ".source.jinja")).read_bytes()
    if recipe == "step5":
        from ignition.recipes.step5.frontend import template_asset

        (tmp_path / "chat_template.jinja").write_bytes(source)
        asset = template_asset(tmp_path)
        source = source.replace(b"{%- generation %}", b"{%- if true %}").replace(
            b"{%- endgeneration %}", b"{%- endif %}")
    else:
        asset = annotated_template_asset(source, patch_path)
    assert asset["text"].encode() == fixture(recipe, filename).read_bytes()
    assert asset["text"].encode().replace(b"{%- mark -%}", b"") == source


@pytest.mark.parametrize("hidden,filename,repository,revision", [
    (hidden, "chat_template.google-e2b-e4b", "google/gemma-4-E4B-it",
     "ee0ef6023621cff504d758262d4e04895a5af4a2") for hidden in (1536, 2560)
] + [
    (hidden, "chat_template.google", "google/gemma-4-26B-A4B-it",
     "4d7ae4984b7db7de8f8457170b3f1a419ee76d52") for hidden in (2816, 3840, 5376)
])
def test_gemma_fetches_pinned_canonical_template(hidden, filename, repository, revision, monkeypatch, tmp_path):
    from ignition.recipes.gemma4.frontend import chat_template_asset

    calls = []
    def download(repo, name, **kwargs):
        calls.append((repo, name, kwargs))
        return fixture("gemma4", filename + ".source.jinja")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    asset = chat_template_asset(hidden, cache_dir=tmp_path)
    assert calls == [(repository, "chat_template.jinja", dict(revision=revision, cache_dir=tmp_path))]
    assert asset["text"].encode() == fixture("gemma4", filename + ".jinja").read_bytes()
    with pytest.raises(ValueError, match="unsupported Gemma 4"):
        chat_template_asset(4096, cache_dir=tmp_path)


def test_step_assets_preserve_checkpoint_and_validate_generation_span(tmp_path):
    from ignition.recipes.step5.frontend import runtime_assets, inference_template

    source = fixture("step5", "chat_template.source.jinja").read_bytes()
    checkpoint = tmp_path / "chat_template.jinja"
    checkpoint.write_bytes(source)
    assets = runtime_assets(tmp_path)
    expected = fixture("step5", "chat_template.jinja").read_bytes()
    assert assets["chat_template"] == {"text": expected.decode(), "sha256": hashlib.sha256(expected).hexdigest()}
    assert inference_template(tmp_path).encode() == expected
    assert assets["tokenizer"] == {"path": str(tmp_path / "tokenizer.json")}
    assert checkpoint.read_bytes() == source
    assert list(tmp_path.iterdir()) == [checkpoint]
    checkpoint.write_bytes(source + b" ")
    with pytest.raises(ValueError, match="checkpoint chat template does not match"):
        runtime_assets(tmp_path)
    checkpoint.write_bytes(source.replace(b"{%- generation %}", b""))
    with pytest.raises(ValueError, match="one assistant generation span"):
        runtime_assets(tmp_path)
