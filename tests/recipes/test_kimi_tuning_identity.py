"""Tuning namespaces remain stable when draft operators join a target model."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from ignition.recipes.kimi3.bootstrap import target_tuning_identity


def fixture():
    return (SimpleNamespace(text={"hidden_size": 7168, "num_hidden_layers": 93},
                            vision={"hidden_size": 1024}, draft=None),
            {"projection": {"shape": [224, 7168], "dtype": "BF16"}})


def test_target_namespace_retains_existing_selection_records():
    config, headers = fixture()
    assert target_tuning_identity(config, headers, 16) == (
        "a79c785a1af74005a487917102ebf1487fa5ea65de6607539be14a62ababf6d8")


@pytest.mark.parametrize("draft", [
    {"architectures": ["K3DSparkModel"], "num_hidden_layers": 5},
    {"architectures": ["DFlash2DraftModel"], "dflash_config": {"block_size": 8}},
])
def test_attached_draft_retains_target_namespace_and_provenance(draft):
    config, headers = fixture()
    expected = target_tuning_identity(config, headers, 16)
    config.draft = SimpleNamespace(raw=deepcopy(draft))
    assert target_tuning_identity(config, headers, 16) == expected
    assert config.draft.raw == draft


@pytest.mark.parametrize("field", ["config", "headers", "vision", "world"])
def test_target_geometry_changes_selection_namespace(field):
    config, headers = fixture()
    before = target_tuning_identity(config, headers, 16)
    world = 16
    if field == "config":
        config.text["hidden_size"] = 8192
    elif field == "headers":
        headers["projection"]["dtype"] = "F32"
    elif field == "vision":
        config.vision["hidden_size"] = 1280
    else:
        world = 8
    assert target_tuning_identity(config, headers, world) != before
