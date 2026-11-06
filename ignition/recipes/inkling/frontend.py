# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content-header messages and JSON tool invocations for Inkling text graphs."""

from pathlib import Path

from tokenizers import Tokenizer

from ignition.preparation import contract as c
from ignition.preparation import runtime_package_pb2 as p
from ignition.preparation.frontend import annotated_template_asset, prepare_frontend as _prepare_frontend


FRONTEND = c.frontend(tools=p.TML_JSON, assistant_marker="<|message_model|>",
    message_end_markers=["<|end_message|>", "<|content_model_end_sampling|>"], prefix_boundary=None,
    rejected_tokens=[200005, 200020, 200023, 200053, 200054])
FRONTEND["reasoning"] = dict(kind=p.CONTENT_HEADERS, open="<|content_thinking|>",
    close="<|end_message|>", wrapping_newlines=0, terminal_markers=[])


def prepare_frontend(snapshot, *, eos_token_ids):
    snapshot = Path(snapshot)
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    for marker, expected in {"<|message_model|>": 200001, "<|content_text|>": 200004,
            "<|content_model_end_sampling|>": 200006, "<|content_thinking|>": 200008,
            "<|end_message|>": 200010, "<|content_invoke_tool_json|>": 200049}.items():
        if tokenizer.token_to_id(marker) != expected:
            raise ValueError(f"Inkling requires {marker} token ID {expected}")
    eos = [eos_token_ids] if type(eos_token_ids) is int else list(eos_token_ids)
    if eos != [200006]:
        raise ValueError("Inkling requires the model-end-sampling token as its sole EOS")
    return _prepare_frontend(snapshot, eos_token_ids=eos)


def runtime_assets(snapshot):
    snapshot = Path(snapshot)
    return {"tokenizer": {"path": str(snapshot / "tokenizer.json")},
            "chat_template": annotated_template_asset((snapshot / "chat_template.jinja").read_bytes(),
                Path(__file__).with_name("assets") / "chat_template.patch.json")}
