# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipient-header prompt syntax and ATEM tools for text-only Muse Spark."""

import hashlib
from pathlib import Path

from tokenizers import Tokenizer

from ignition.preparation import contract as c
from ignition.preparation import runtime_package_pb2 as p
from ignition.preparation.frontend import annotated_template_asset, prepare_frontend as _prepare_frontend


FRONTEND = c.frontend(
    tools=p.ATEM,
    assistant_marker="<|start|>assistant",
    message_end_markers=["<|eom|>", "<|eot|>"],
    prefix_boundary=None,
)
FRONTEND["reasoning"] = dict(
    kind=p.RECIPIENT_HEADERS,
    open="<|start|>assistant to=self<|message|>",
    close="<|eom|>",
    wrapping_newlines=0,
    terminal_markers=[],
)


def prepare_frontend(snapshot, *, eos_token_ids):
    snapshot = Path(snapshot)
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    for marker, expected in {"<|end_of_text|>": 200001, "<|start|>": 200022,
                             "<|message|>": 200023, "<|eom|>": 200007,
                             "<|eot|>": 200008}.items():
        if tokenizer.token_to_id(marker) != expected:
            raise ValueError(f"Muse Spark requires {marker} token ID {expected}")
    eos_token_ids = [eos_token_ids] if isinstance(eos_token_ids, int) else list(eos_token_ids)
    if 200007 in eos_token_ids:
        raise ValueError("Muse Spark <|eom|> is an internal message boundary")
    return _prepare_frontend(snapshot, eos_token_ids=list(dict.fromkeys([*eos_token_ids, 200001, 200008])))


def runtime_assets(snapshot, patch=Path(__file__).with_name("assets") / "chat_template.patch.json"):
    snapshot = Path(snapshot)
    tokenizer = snapshot / "tokenizer.json"
    with tokenizer.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"tokenizer": {"path": str(tokenizer), "sha256": digest},
            "chat_template": annotated_template_asset((snapshot / "chat_template.jinja").read_bytes(), patch)}
