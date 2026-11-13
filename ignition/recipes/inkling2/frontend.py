# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inkling 2 tokenizer, TMLv0 prompt and output syntax, image kinds and sampling defaults.

Prompts render with Helios's native TMLv0 encoder, whose reference is Thinking
Machines' tml-renderers 0.4.1, so the core carries the checkpoint tokenizer and
no chat template. Requests without an effort use the encoder's default of 0.9,
the default of the checkpoint chat template and of tml-renderers' Tinker cookbook.
"""

import json
from pathlib import Path

from ignition.preparation import contract as c
from ignition.preparation import runtime_core_pb2 as p
from ignition.preparation.frontend import PromptEncoder, prepare_frontend
from ignition.recipes.declaration import SamplingDefaults

# The model card at the pinned revision recommends no sampling settings. Vals.ai
# evaluated Inkling 2 Small with "Thinking Machines' recommended settings:
# temperature 1, top_p 1 and reasoning effort 0.99"
# (https://www.vals.ai/models/thinkingmachines_inkling-2-small). Other filters are
# disabled and penalties neutral.
SAMPLING_DEFAULTS = SamplingDefaults(temperature=1., top_p=1., top_k=-1)

END_OF_TURN = "<|content_model_end_sampling|>"
IMAGE_TOKEN = "<|unused_200054|>"
EOS_TOKEN_ID = 200006
IMAGE_TOKEN_ID = 200054
# Framing tokens the TMLv0 encoder and output parser require as single control tokens.
CONTROL_TOKENS = (
    "<|message_user|>", "<|message_model|>", "<|message_system|>", "<|message_tool|>", "<|content_text|>",
    "<|content_thinking|>", "<|content_xml|>", "<|content_image|>", "<|content_invoke_tool_json|>",
    "<|content_invoke_tool_text|>", "<|content_tool_error|>", "<|end_message|>", END_OF_TURN,
    "<|begin_of_text|>",
)


def frontend(vision=True):
    """TMLv0 output frames; with ``vision``, padded-grid image preparation with a class-token layout."""
    result = c.frontend(tools=p.TML_JSON, assistant_marker="<|message_model|>",
        message_end_markers=["<|end_message|>", END_OF_TURN], prefix_boundary=c.prefix_boundary(["system"]),
        rejected_tokens=() if vision else (IMAGE_TOKEN_ID,),
        **(dict(image_preparation=p.PADDED_GRID_IMAGE_PREPARATION, image_layout=p.CLASS_TOKEN_IMAGE_LAYOUT,
                image_tokens=dict(feature_token=IMAGE_TOKEN_ID, span=None)) if vision else {}))
    result["reasoning"] = dict(kind=p.CONTENT_HEADERS, open="<|content_thinking|>", close="<|end_message|>",
                               wrapping_newlines=0, terminal_markers=[])
    return result


FRONTEND = frontend()


def prepare(snapshot, vision=True):
    """Typed frontend settings validated against the checkpoint's tokenizer and configuration."""
    from tokenizers import Tokenizer

    snapshot = Path(snapshot)
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    for token in CONTROL_TOKENS:
        token_id = tokenizer.token_to_id(token)
        if token_id is None or tokenizer.encode(token, add_special_tokens=False).ids != [token_id]:
            raise ValueError(f"Inkling 2 tokenizer must encode {token} as one control token")
    if tokenizer.token_to_id(END_OF_TURN) != EOS_TOKEN_ID:
        raise ValueError("Inkling 2 ends each turn with token 200006")
    model = json.loads((snapshot / "config.json").read_text())
    if model.get("eos_token_id") != EOS_TOKEN_ID:
        raise ValueError("Inkling 2 configuration must declare EOS token 200006")
    settings = prepare_frontend(snapshot, eos_token_ids=[EOS_TOKEN_ID], prompt_encoder=PromptEncoder.TML_V0,
        image_profile="padded_grid" if vision else None, preprocessor=snapshot / "processor_config.json")
    if vision:
        processor = json.loads((snapshot / "processor_config.json").read_text())
        if (processor.get("image_token") != IMAGE_TOKEN or tokenizer.token_to_id(IMAGE_TOKEN) != IMAGE_TOKEN_ID
                or tokenizer.encode(IMAGE_TOKEN, add_special_tokens=False).ids != [IMAGE_TOKEN_ID]):
            raise ValueError("Inkling 2 images require the atomic placeholder token 200054")
        settings["image_placeholder_token_id"] = IMAGE_TOKEN_ID
    return settings


def runtime_assets(snapshot):
    return {"tokenizer": {"path": str(Path(snapshot) / "tokenizer.json")}}
