# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""XTML frontend, KDA continuations, and paged latent attention bindings."""

from ignition.preparation import contract as c
from ignition.preparation import runtime_core_pb2 as p

from .config import PAGE_TOKENS
from .vision import PATCH_CAPACITIES, FEATURES

VISION = dict(features_per_request=FEATURES, max_images=4, patches_per_feature=4, patch_capacities=list(PATCH_CAPACITIES))


IMAGES = dict(image_tokens=dict(feature_token=163605, span=None),
    image_preparation=p.MOONVIT_IMAGE_PREPARATION, image_layout=p.MOONVIT_IMAGE_LAYOUT)


def frontend(vision=True):
    """XTML chat and reasoning markers; with ``vision``, MoonViT image preparation and placement."""
    result = c.frontend(tools=p.XTML, **(IMAGES if vision else {}),
        assistant_marker='<|open|>message role="assistant"<|sep|>',
        message_end_markers=["<|close|>message<|sep|>", "<|end_of_msg|>"],
        prefix_boundary=c.prefix_boundary(["system", "developer"], append_role="user", minimum_messages=0,
            required_start='<|open|>message role="system"', required_marker="<|end_of_msg|>"))
    result["reasoning"] = dict(kind=p.XTML_THINK, open="<|open|>think<|sep|>", close="<|close|>think<|sep|>",
                               wrapping_newlines=0, terminal_markers=[])
    return result


FRONTEND = frontend()


def prepare(signatures, buffers, pool, *, image_slots=4, vocabulary, page_tokens=PAGE_TOKENS, draft_lag=None,
            vision=True):
    inputs = {kind: c.text_inputs(kind) for kind in {s.kind for s in signatures.values()} if kind != "vision"}
    inputs["vision"] = c.vision_inputs({field for s in signatures.values() if s.kind == "vision" for field in s.inputs})
    continuation = c.state_view("state_pool", pool.layout.row_bytes)
    consumed = pool.layout.region("consumed_tokens")
    # Speculative continuations draft from every accepted position: the draft context trails consumption by draft_lag.
    cursor = None if draft_lag is None else dict(state=c.state_view("state_pool", pool.layout.row_bytes,
        offset=consumed.offset, nbytes=consumed.nbytes), lag_tokens=draft_lag)
    pending = c.state_view("request_pending_token", 8)
    consumed = pool.layout.region("consumed_tokens")
    cursor = None if draft_lag is None else dict(state=c.state_view("state_pool", pool.layout.row_bytes,
        offset=consumed.offset, nbytes=consumed.nbytes), lag_tokens=draft_lag)
    regions = [c.history_region(buffers, f"cache.{name}", tensor, p.PAGED_LATENT)
               for name, tensor in pool.pages.items()]
    return c.model_contract(vocabulary=vocabulary, page_tokens=page_tokens,
        page_tables=[dict(buffer="page_tables", page_offset=0)],
        regions=[dict(kind=p.CONTINUATION_STATE, storage=continuation, tokens_per_record=1, window_tokens=0), *regions],
        graphs=c.graph_contracts(signatures, inputs, continuation=continuation, pending=pending, cursor=cursor),
        vision={**VISION, "feature_slots": image_slots} if vision else None,
        frontend=FRONTEND if vision else frontend(vision=False))
