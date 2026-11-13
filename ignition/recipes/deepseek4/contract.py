# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared graph inputs, continuation storage, and frontend syntax."""

from ignition.preparation import contract as c
from ignition.preparation import runtime_core_pb2 as p


VOCABULARY = 129280
INPUT_VOCABULARY = 129285
VISION = {'features_per_request': 1536, 'max_images': 4, 'patches_per_feature': 9, 'patch_capacities': [576, 1152, 2304, 3456]}
FRONTEND = c.frontend(tools=p.DSML, assistant_marker="<｜Assistant｜>",
    message_end_markers=["<｜User｜>", "<｜end▁of▁sentence｜>"],
    prefix_boundary=c.prefix_boundary(["system"]),
    image_tokens=dict(feature_token=129282, span=dict(start_token=129280, end_token=129284, max_tokens=384, interior_tokens=[129283], unscoped_tokens=[129281])),
    image_preparation=p.DEEPSEEK_IMAGE_PREPARATION, image_layout=p.DEEPSEEK_IMAGE_LAYOUT)


def prepare(signatures, buffers, pool, *, image_slots=4, page_tokens, draft_lag, rows=1, embedding_tables=(), regions=()):
    regions = list(regions)
    for name, tensor in pool.pages.items():
        layer = int(name.split(".")[0].removeprefix("layer"))
        regions.append(c.history_region(buffers, "cache." + name, tensor, p.PAGED_LATENT,
            tokens_per_record=pool.config.text["compress_ratios"][layer]))
    inputs = {kind: c.text_inputs(kind) for kind in {s.kind for s in signatures.values()} if kind != "vision"}
    inputs["vision"] = c.vision_inputs({field for s in signatures.values() if s.kind == "vision" for field in s.inputs})
    continuation = c.state_view("state_pool", pool.layout.row_bytes)
    consumed = pool.layout.region("consumed_tokens")
    cursor = None if draft_lag is None else dict(state=c.state_view("state_pool", pool.layout.row_bytes,
        offset=consumed.offset, nbytes=consumed.nbytes), lag_tokens=draft_lag)
    pending = c.state_view("request_pending_token", 8)
    for kind in ("prefill", "mixed"):
        if kind in inputs:
            inputs[kind]["vision_left"] = c.binding("vision_left", p.IMAGE_LEFT_CONTEXT, p.PACKED_TOKEN)
            inputs[kind]["vision_right"] = c.binding("vision_right", p.IMAGE_RIGHT_CONTEXT, p.PACKED_TOKEN)
    graphs = c.graph_contracts(signatures, inputs, continuation=continuation, pending=pending,
        cursor=cursor, rows=rows, embedding_tables=embedding_tables)
    return c.model_contract(vocabulary=VOCABULARY, input_vocabulary=INPUT_VOCABULARY,
        page_tokens=page_tokens, page_tables=[dict(buffer="page_tables", page_offset=0),
            *([dict(buffer="draft_page_tables", page_offset=1)] if "draft_page_tables" in buffers else [])],
        regions=[dict(kind=p.CONTINUATION_STATE, storage=continuation, tokens_per_record=1, window_tokens=0), *regions],
        graphs=graphs, vision={**VISION, "feature_slots": image_slots}, frontend=FRONTEND)
