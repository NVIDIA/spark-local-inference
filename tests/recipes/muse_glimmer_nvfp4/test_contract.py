# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Glimmer contracts: Spark's recipient-header frontend, FP8 page bytes, a vision block, and draft lag 0."""

import copy

import pytest
from google.protobuf.json_format import ParseDict

from ignition.components.paged_batch import BatchInputs
from ignition.components.parallel import TensorParallelShard
from ignition.preparation import contract as c, runtime_package_pb2 as p
from ignition.preparation.execution import GraphSignature
from ignition.recipes.muse_glimmer_nvfp4.config import ModelConfig
from ignition.recipes.muse_glimmer_nvfp4.contract import prepare
from ignition.recipes.muse_glimmer_nvfp4.frontend import FRONTEND
from ignition.recipes.muse_glimmer_nvfp4.vision import PATCH_CAPACITIES
from ignition.recipes.muse_glimmer_nvfp4.vision_state import FEATURES, VisionInputs
from ignition.recipes.muse_spark.pool import ContinuationPool, page_bytes

from .test_config import CONFIG

TEXT_KINDS = ("init", "restore", "snapshot", "prefill", "scored_prefill", "decode", "resume")
VISION_FIELDS = (*VisionInputs.__dataclass_fields__, "vision_feature_slot")


@pytest.fixture
def pool():
    config = ModelConfig.from_dict(copy.deepcopy(CONFIG))
    # One state row plus two snapshots and two pages suffice for the layout the contract reads.
    return ContinuationPool.allocate(config, TensorParallelShard(0, 1), 3, 2, "cpu")


def text_signatures():
    return {kind: GraphSignature(kind, 1 if "prefill" in kind else 4, 128 if "prefill" in kind else 1,
                                 tuple(c.text_inputs(kind))) for kind in TEXT_KINDS}


def vision_signatures():
    return {f"vision{capacity}": GraphSignature("vision", 1, capacity, VISION_FIELDS) for capacity in PATCH_CAPACITIES}


def test_text_contract_carries_spark_grammar_and_fp8_full_layer_pages(pool):
    buffers = BatchInputs.allocate(128, 2, "cpu", capacity=4).buffers()
    encoded = ParseDict(prepare(text_signatures(), buffers, pool, vocabulary=202048), p.ModelContract())
    assert encoded.version == 3 and encoded.vocabulary == 202048
    assert encoded.cache.page_tokens == 128
    assert encoded.frontend.tools.kind == p.ATEM
    assert encoded.frontend.reasoning.kind == p.RECIPIENT_HEADERS
    assert list(encoded.frontend.rejected_tokens) == [200091]
    # The text-only package names no image preparation: Helios rejects image parts outright.
    assert encoded.vision.max_images == 0 and not encoded.vision.patch_capacities
    assert encoded.frontend.image_preparation == 0 and not encoded.frontend.HasField("image_tokens")
    histories = [r for r in encoded.cache.regions if r.kind == p.PAGED_KEY_VALUE]
    # 13 full-attention layers, key and value planes each: 128 tokens x 2 KV heads x 128 head_dim x 1 byte.
    assert len(histories) == 26
    assert all(r.storage.stride == 128 * 2 * 128 for r in histories)
    assert sum(r.storage.stride for r in histories) == page_bytes(pool.config, pool.shard) == 13 * 128 * 2 * 128 * 2
    assert {r.storage.buffer for r in histories} == {f"cache.layer{i}.{kind}" for i in range(3, 52, 4) for kind in ("key", "value")}
    for graph in encoded.graphs:
        assert not graph.HasField("draft_cursor")
        assert any(s.view.buffer == "state_pool" and s.view.nbytes == pool.layout.row_bytes
                   and s.access == p.READ_WRITE for s in graph.state)
    decode = next(g for g in encoded.graphs if g.kind == p.DECODE)
    assert decode.pending_token.buffer == "request_pending_token"
    assert all(b.source not in (p.PROMPT_TOKEN, p.PENDING_TOKEN) for b in decode.inputs)


def test_vision_block_and_vision_graph_inputs_join_the_text_contract(pool):
    buffers = BatchInputs.allocate(128, 2, "cpu", capacity=4).buffers()
    signatures = {**text_signatures(), **vision_signatures()}
    encoded = ParseDict(prepare(signatures, buffers, pool, vocabulary=202048, image_slots=2), p.ModelContract())
    assert encoded.vision.feature_slots == 2 and encoded.vision.max_images == 4
    assert encoded.vision.features_per_request == FEATURES == 16384
    assert encoded.vision.patches_per_feature == 4
    assert list(encoded.vision.patch_capacities) == [1024, 4096, 16384]
    assert encoded.frontend.image_preparation == FRONTEND["image_preparation"] == p.MUSE_GLIMMER_IMAGE_PREPARATION
    assert encoded.frontend.image_layout == p.BRACKETED_IMAGE_LAYOUT
    assert encoded.frontend.image_tokens.feature_token == 200092
    vision = [g for g in encoded.graphs if g.kind == p.VISION]
    assert [g.graph for g in vision] == [f"vision{capacity}" for capacity in PATCH_CAPACITIES]
    for graph in vision:
        assert not graph.state and not graph.HasField("pending_token")
        bound = {b.buffer: b for b in graph.inputs}
        assert set(bound) == set(VISION_FIELDS)
        # The two window tensors Helios emits beside cu_seqlens reach the encoder as image tensors.
        assert bound["vision_cu_window_seqlens"].source == bound["vision_reverse_index"].source == p.IMAGE_TENSOR
        assert bound["vision_slot"].source == p.REQUEST_SLOT
        assert bound["vision_generation"].source == p.REQUEST_GENERATION
        assert bound["vision_feature_slot"].source == p.IMAGE_FEATURE_SLOT
    assert len(encoded.graphs) == len(signatures)


def test_vision_graphs_and_image_slots_are_declared_together(pool):
    buffers = BatchInputs.allocate(128, 2, "cpu", capacity=4).buffers()
    with pytest.raises(ValueError, match="together"):
        prepare(text_signatures(), buffers, pool, vocabulary=202048, image_slots=1)
    with pytest.raises(ValueError, match="together"):
        prepare({**text_signatures(), **vision_signatures()}, buffers, pool, vocabulary=202048)


def test_speculative_contract_tracks_the_committed_cursor_with_lag_zero(pool):
    buffers = BatchInputs.allocate(128, 2, "cpu", capacity=4).buffers()
    signatures = {
        "speculative4x16-tokens64": GraphSignature("speculative", 4, 16, tuple(c.text_inputs("speculative")),
                                                   speculation_enabled=True, packed_token_capacity=64),
        "decode4-no-mtp": GraphSignature("decode", 4, 1, tuple(c.text_inputs("decode")), speculation_enabled=False),
    }
    encoded = ParseDict(prepare(signatures, buffers, pool, vocabulary=202048, draft_lag=0), p.ModelContract())
    speculative, ordinary = encoded.graphs
    consumed = pool.layout.region("consumed_tokens")
    assert speculative.draft_cursor.state.buffer == "state_pool"
    assert speculative.draft_cursor.state.offset == consumed.offset
    assert speculative.draft_cursor.state.stride == pool.layout.row_bytes
    assert speculative.draft_cursor.lag_tokens == 0
    assert speculative.pending_token.buffer == "request_pending_token"
    assert not ordinary.HasField("draft_cursor")
