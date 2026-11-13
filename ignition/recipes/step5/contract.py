# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ChatML frontend, image crops, paged GQA and compressed region summaries."""

from ignition.preparation import contract as c
from ignition.preparation import runtime_core_pb2 as p

from .config import PAGE_TOKENS
from .vision import FEATURES, PATCH_CAPACITIES

FRONTEND = c.frontend(tools=p.QWEN_CODER, terminal_markers=["<tool_call>"],
    assistant_marker="<|im_start|>assistant\n", message_end_markers=["<|im_end|>"],
    prefix_boundary=c.prefix_boundary(["system", "developer"], append_role="user", minimum_messages=0,
        required_start="<|im_start|>system\n", required_marker="<|im_end|>"),
    image_tokens={"feature_token": 128001, "span": None},
    image_preparation=p.STEP_IMAGE_PREPARATION, image_layout=p.STEP_IMAGE_LAYOUT)
FRONTEND["tools"]["wrapping_newlines"] = 1


def prepare(signatures, buffers, pool, *, image_slots=4, vocabulary, mtp):
    inputs = {kind: c.text_inputs(kind) for kind in {s.kind for s in signatures.values()} if kind != "vision"}
    inputs["vision"] = c.vision_inputs({field for s in signatures.values() if s.kind == "vision" for field in s.inputs})
    continuation = c.state_view("state_pool", pool.layout.row_bytes)
    consumed = pool.layout.region("consumed_tokens")
    cursor = {"state": c.state_view("state_pool", pool.layout.row_bytes, offset=consumed.offset,
                                    nbytes=consumed.nbytes), "lag_tokens": 0} if mtp else None
    regions = [c.history_region(buffers, f"cache.{name}", tensor,
        p.PAGED_LATENT if name.endswith(".index") else p.PAGED_KEY_VALUE,
        tokens_per_record=8 if name.endswith(".index") else 1) for name, tensor in pool.pages.items()]
    return c.model_contract(vocabulary=vocabulary, page_tokens=PAGE_TOKENS,
        page_tables=[{"buffer": "page_tables", "page_offset": 0}],
        regions=[{"kind": p.CONTINUATION_STATE, "storage": continuation, "tokens_per_record": 1, "window_tokens": 0}, *regions],
        graphs=c.graph_contracts(signatures, inputs, continuation=continuation,
                                 pending=c.state_view("request_pending_token", 8), cursor=cursor),
        vision={"feature_slots": image_slots, "features_per_request": FEATURES, "max_images": 4, "patches_per_feature": 16,
                    "patch_capacities": list(PATCH_CAPACITIES)}, frontend=FRONTEND)
