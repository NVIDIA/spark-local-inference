# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inkling 2 frontend metadata: TMLv0 syntax, image kinds, sampling defaults and checkpoint settings."""

import json
import os
from pathlib import Path

import pytest

from ignition.preparation.frontend import PromptEncoder
from ignition.preparation.runtime_core_pb2 import ModelIndex
from ignition.recipes.inkling2 import frontend as inkling2

# The native tests validate this frontend contract and prepare Inkling 2 requests with it.
FIXTURE = Path(__file__).parents[2] / "helios/tests/support/contracts/inkling2_frontend.json"
FILES = ("tokenizer.json", "tokenizer_config.json", "config.json", "processor_config.json")


def test_sampling_defaults_disable_filters_at_the_evaluated_settings():
    assert inkling2.SAMPLING_DEFAULTS.as_dict() == dict(seed=0, temperature=1., top_p=1., top_k=-1, min_p=0.,
        repetition_penalty=1., presence_penalty=0., frequency_penalty=0.)


def test_frontend_contract_matches_the_native_fixture():
    fixture = json.loads(FIXTURE.read_text())
    assert inkling2.frontend() == fixture["vision"]
    assert inkling2.frontend(vision=False) == fixture["text"]


def snapshot():
    path = os.environ.get("FLARE_INKLING2_TEST_SNAPSHOT")
    if not path or not all((Path(path) / name).is_file() for name in FILES):
        pytest.skip("requires FLARE_INKLING2_TEST_SNAPSHOT with the Inkling 2 tokenizer and configuration files")
    return Path(path)


@pytest.mark.parametrize("vision", [True, False])
def test_checkpoint_settings_export_the_native_encoder(vision):
    from ignition.preparation.core_proto import serving

    settings = inkling2.prepare(snapshot(), vision=vision)
    assert settings["prompt_encoder"] == PromptEncoder.TML_V0
    assert settings["tokenizer"] == {}
    assert settings["eos_token_ids"] == [200006]
    assert settings.get("image_placeholder_token_id") == (200054 if vision else None)
    assert settings.get("image") == ({"padded_grid": {}} if vision else None)
    assert inkling2.runtime_assets(snapshot()).keys() == {"tokenizer"}
    source = dict(assets={"tokenizer": {}}, frontend=settings, sampling_defaults=inkling2.SAMPLING_DEFAULTS.as_dict())
    index = ModelIndex()
    serving(index, source)
    restored = ModelIndex.FromString(index.SerializeToString())
    assert restored.frontend.prompt_encoder == PromptEncoder.TML_V0
    assert restored.frontend.HasField("image_placeholder_token_id") == vision
    assert restored.frontend.image.HasField("padded_grid") == vision
    assert (restored.sampling_defaults.temperature, restored.sampling_defaults.top_p,
            restored.sampling_defaults.top_k) == (1., 1., -1)
    source["assets"]["chat_template"] = {}
    with pytest.raises(ValueError, match="runtime assets"):
        serving(ModelIndex(), source)


def test_checkpoint_settings_reject_a_different_end_of_turn(tmp_path):
    source = snapshot()
    for name in FILES:
        (tmp_path / name).symlink_to(source / name)
    (tmp_path / "config.json").unlink()
    (tmp_path / "config.json").write_text(json.dumps({**json.loads((source / "config.json").read_text()),
                                                      "eos_token_id": 200010}))
    with pytest.raises(ValueError, match="EOS token 200006"):
        inkling2.prepare(tmp_path)
