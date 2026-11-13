# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared graph inputs, continuation storage, and frontend syntax."""

from ignition.preparation import contract as c
from ignition.preparation import runtime_core_pb2 as p


VOCABULARY = 248320
INPUT_VOCABULARY = 248320
VISION = {'features_per_request': 16384, 'max_images': 4, 'patches_per_feature': 4, 'patch_capacities': [256, 1024, 4096, 16384]}
FRONTEND = c.frontend(tools=p.QWEN_CODER, wrapping_newlines=1, terminal_markers=["<tool_call>"],
    assistant_marker="<|im_start|>assistant\n", message_end_markers=["<|im_end|>"],
    prefix_boundary=c.prefix_boundary(["system", "developer"], append_role="user", minimum_messages=0,
        required_start="<|im_start|>system\n", required_marker="<|im_end|>"),
    image_tokens=dict(feature_token=248056, span=None),
    image_preparation=p.QWEN_IMAGE_PREPARATION, image_layout=p.QWEN_IMAGE_LAYOUT, position_dimensions=3)


def prepare(signatures, buffers, pool, *, image_slots=4, page_tokens, draft_lag, rows=1, embedding_tables=(), regions=()):
    inputs = {kind: c.text_inputs(kind) for kind in {s.kind for s in signatures.values()} if kind != "vision"}
    inputs["vision"] = c.vision_inputs({field for s in signatures.values() if s.kind == "vision" for field in s.inputs})
    continuation = c.state_view("state_pool", pool.layout.row_bytes)
    consumed = pool.layout.region("consumed_tokens")
    cursor = None if draft_lag is None else dict(state=c.state_view("state_pool", pool.layout.row_bytes,
        offset=consumed.offset, nbytes=consumed.nbytes), lag_tokens=draft_lag)
    pending = c.state_view("request_pending_token", 8)
    for kind, values in inputs.items():
        if kind in ("decode", "speculative"):
            values.pop("token_ids", None)
        if kind == "mixed":
            values["token_ids"] = c.binding("token_ids", p.PROMPT_TOKEN, p.PACKED_TOKEN)
        if kind == "resume":
            values["sample_enabled"] = c.binding("sample_enabled", p.SAMPLE_ENABLED)
            values["page_tables"] = c.binding("page_tables", p.PAGE_TABLE, p.REQUEST_ROW)
            values["draft_page_tables"] = c.binding("draft_page_tables", p.PAGE_TABLE, p.REQUEST_ROW)
        if kind in ("init", "restore", "snapshot"):
            values["state_source_slots"] = c.binding("state_source_slots",
                p.PREFIX_STATE_ROW if kind == "restore" else p.LIVE_STATE_ROW)
            values["state_destination_slots"]["padding"] = 0
            values["state_qsa_slots"] = c.binding("state_qsa_slots", p.CONSTANT if kind == "snapshot" else p.REQUEST_SLOT,
                offset=-1 if kind == "snapshot" else 1, padding=-1)
            values["state_accepted_tokens"] = c.binding("state_accepted_tokens", p.CONSTANT, offset=int(kind != "snapshot"))
    graphs = c.graph_contracts(signatures, inputs, continuation=continuation, pending=pending,
        cursor=cursor, rows=rows, embedding_tables=embedding_tables)
    return c.model_contract(vocabulary=VOCABULARY, input_vocabulary=INPUT_VOCABULARY,
        page_tokens=page_tokens, page_tables=[dict(buffer="page_tables", page_offset=0),
            *([dict(buffer="draft_page_tables", page_offset=1)] if "draft_page_tables" in buffers else [])],
        regions=[dict(kind=p.CONTINUATION_STATE, storage=continuation, tokens_per_record=1, window_tokens=0), *regions],
        graphs=graphs, vision={**VISION, "feature_slots": image_slots}, frontend=FRONTEND)
