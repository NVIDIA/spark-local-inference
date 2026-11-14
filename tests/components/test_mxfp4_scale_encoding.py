"""Indexed MXFP4 scale fields select the corresponding lossless loader."""

from types import SimpleNamespace

import pytest

from ignition.components.weights import DirectWeightLoader


PREFIXES = ("experts.0", "experts.1")
PROJECTIONS = ("up_proj", "gate_proj", "down_proj")
SUFFIXES = {
    "csf32": ("csf32_bases", "csf32_codes", "csf32_page_bases", "csf32_offsets", "csf32_exceptions"),
    "csf": ("mxfp4_csf_fixed", "mxfp4_csf_exceptions"),
}


def loader(encoding):
    owner = DirectWeightLoader.__new__(DirectWeightLoader)
    owner._session = object()
    names = [f"{prefix}.{projection}.weight_scale" for prefix in PREFIXES for projection in PROJECTIONS]
    owner.manifest = SimpleNamespace(compressed_scales=frozenset(names) if encoding == "manifest" else frozenset())
    owner._sources = {f"{name}.{suffix}": None for name in names for suffix in SUFFIXES.get(encoding, ())}
    owner.load_fp4_experts = lambda *args, **kwargs: ("native", args, kwargs)
    owner.load_mxfp4_csf_experts = lambda *args, **kwargs: ("csf", args, kwargs)
    return owner


def load(owner):
    return owner.load_mxfp4_experts(PREFIXES, hidden_size=128, intermediate_size=128,
                                  scale_scratch={(2, 128, 128): (object(), object())})


@pytest.mark.parametrize("encoding", [None, "manifest", "csf32", "csf"])
def test_encoding_fields_preserve_native_and_compressed_loader_selection(encoding):
    result, prefixes, options = load(loader(encoding))
    assert result == ("native" if encoding is None else "csf")
    assert prefixes == (PREFIXES,)
    if encoding is not None:
        assert options["projection_names"] == ("gate_proj", "up_proj", "down_proj")


@pytest.mark.parametrize("encoding", ["csf32", "csf"])
def test_partial_encoded_plane_sets_are_rejected(encoding):
    owner = loader(encoding)
    owner._sources.pop(next(iter(owner._sources)))
    with pytest.raises(ValueError, match="requires complete planes"):
        load(owner)


def test_mixed_native_and_encoded_experts_are_rejected():
    owner = loader("csf32")
    owner._sources = {name: value for name, value in owner._sources.items()
                      if not name.startswith("experts.1.down_proj.")}
    with pytest.raises(ValueError, match="one complete MXFP4 scale encoding"):
        load(owner)
