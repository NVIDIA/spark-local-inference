# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inkling 2 Small checkpoint access and reference activations for GPU tests.

`FLARE_INKLING2_SNAPSHOT` names the checkpoint snapshot (revision b0e7afb8);
its default is the Hugging Face cache path. Reference activations are the
canonical single-rank decoder's residual state at the inputs of selected
layers for a fixed prompt: the first `PROMPT_TOKENS` tokens of the
checkpoint's model card after its metadata header. Computing them runs every
earlier layer through `ignition.reference.inkling2`, so the results are cached
in the pytest cache directory under a key of the revision, prompt, layers and
the reference's source.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.recipes.inkling2.config import ModelConfig
from ignition.recipes.inkling2.weights import PlacedWeightLoader, checkpoint_layout
from ignition.reference import inkling2 as ref

SNAPSHOT = Path(os.environ.get("FLARE_INKLING2_SNAPSHOT", Path.home() / ".cache/huggingface/hub" /
    "models--thinkingmachines--Inkling-2-Small/snapshots/b0e7afb8892a07897c78ce1c5b801a0d9cbd5e7b"))
REVISION = "b0e7afb8892a07897c78ce1c5b801a0d9cbd5e7b"
PROMPT_TOKENS = 1408
WORLD = 4


def require_snapshot():
    if not (SNAPSHOT / "model.safetensors.index.json").is_file():
        pytest.skip("Inkling 2 Small checkpoint is unavailable; set FLARE_INKLING2_SNAPSHOT")


class Checkpoint:
    """The checkpoint's configuration, four-rank layout and direct tensor reads."""

    def __init__(self, device):
        require_snapshot()
        self.device = torch.device(device)
        self.config = ModelConfig.load(SNAPSHOT)
        self.layout = checkpoint_layout(self.config, WORLD)
        self.manifest = inspect_indexed_checkpoint(SNAPSHOT)

    def loader(self, names):
        return PlacedWeightLoader(self.manifest, self.device, layout=self.layout, names=names)

    def layer_names(self, layer):
        return {placement.name for placement in self.layout.layer(layer).values()}

    def reference_layer(self, layer, ranks=()):
        """Whole reference `LayerWeights` and those of the requested ranks."""
        placements = self.layout.layer(layer)
        with self.loader(self.layer_names(layer)) as loader:
            whole = ref.LayerWeights(**{role: loader.load(p.name) for role, p in placements.items()})
            parts = [ref.LayerWeights(**{role: loader.load(p.name, rank=rank) for role, p in placements.items()})
                     for rank in ranks]
        return whole, parts

    def global_weights(self, rank=None):
        p = self.layout.model
        names = {name: f"model.llm.{name}" for name in ("embed.weight", "embed_norm.weight", "norm.weight",
                                                        "attn_res.output_query", "unembed.weight")}
        with self.loader(set(names.values())) as loader:
            values = {key: loader.load(name, rank=rank if p[name].axis is not None else None)
                      for key, name in names.items()}
        return ref.GlobalWeights(values["embed.weight"], values["embed_norm.weight"], values["norm.weight"],
                                 values["attn_res.output_query"], values["unembed.weight"])


def prompt_ids():
    """Token IDs of the model card's prose after its metadata header."""
    import tokenizers

    tokenizer = tokenizers.Tokenizer.from_file(str(SNAPSHOT / "tokenizer.json"))
    text = (SNAPSHOT / "README.md").read_text()
    if text.startswith("---"):
        text = text[text.index("---", 3) + 3:]
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if len(ids) < PROMPT_TOKENS:
        raise ValueError("the model card is shorter than the test prompt")
    return torch.tensor(ids[:PROMPT_TOKENS], dtype=torch.long)


def reference_activations(request, checkpoint: Checkpoint, layers):
    """Per layer in `layers`: the residual state at its input and its MLP input, from the canonical decoder.

    Returns `{layer: {"blocks": [...], "partial": tensor, "mlp_input": tensor}}` on the
    checkpoint's device, with `"ids"` holding the prompt.
    """
    layers = tuple(sorted(set(layers)))
    ids = prompt_ids()
    source = Path(ref.__file__).read_bytes()
    key = hashlib.sha256(json.dumps([REVISION, ids.tolist(), layers]).encode() + source).hexdigest()[:16]
    path = Path(request.config.cache.mkdir("inkling2-activations")) / f"{key}.pt"
    if path.is_file():
        stored = torch.load(path, map_location=checkpoint.device)
        return stored
    device = checkpoint.device
    text = checkpoint.config.text
    positions = torch.arange(ids.numel(), device=device)
    globals_ = checkpoint.global_weights()
    state = ref.ResidualState((), ref.embed(ids.to(device), globals_, text))
    del globals_
    result = {"ids": ids.to(device)}
    for layer in range(max(layers) + 1):
        weights, _ = checkpoint.reference_layer(layer)
        entry = None
        if layer in layers:
            entry = {"blocks": list(state.blocks), "partial": state.partial}
        state, _, record = ref.decoder_layer(weights, text, layer, state, positions,
                                             ref.LayerCache.empty(weights, text.head_dim))
        if entry is not None:
            entry["mlp_input"] = record.mlp_input
            entry["attention_output"] = record.attention_output
            result[layer] = entry
        del weights, record
        torch.cuda.empty_cache()
    torch.save(result, path)
    return result
