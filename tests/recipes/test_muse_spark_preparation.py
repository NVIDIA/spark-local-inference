# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Spark text contracts retain device tokens and complete cache row storage."""

from types import SimpleNamespace

from google.protobuf.json_format import ParseDict

from ignition.components.paged_batch import BatchInputs
from ignition.components.parallel import TensorParallelShard
from ignition.preparation import contract as c, runtime_core_pb2 as p
from ignition.preparation.execution import GraphSignature
from ignition.recipes.muse_spark.contract import prepare
from ignition.recipes.muse_spark.pool import ContinuationPool

from .test_muse_spark_attention import Geometry


def test_text_contract_binds_complete_state_and_physical_fp8_page_bytes():
    config = SimpleNamespace(layers=2, attention=lambda i: Geometry(2048 if i == 0 else -1),
                             text={"hidden_size": 128})
    pool = ContinuationPool.allocate(config, TensorParallelShard(0, 1), 3, 2, 'cpu')
    buffers = BatchInputs.allocate(128, 2, 'cpu', capacity=4).buffers()
    signatures = {kind: GraphSignature(kind, 1 if 'prefill' in kind else 4,
        128 if 'prefill' in kind else 1, tuple(c.text_inputs(kind)))
        for kind in ('init', 'restore', 'snapshot', 'prefill', 'scored_prefill', 'decode', 'resume')}
    encoded = ParseDict(prepare(signatures, buffers, pool, vocabulary=201088), p.ModelContract())
    assert encoded.version == 3
    assert encoded.cache.page_tokens == 128
    assert encoded.vision.max_images == 0
    assert encoded.frontend.tools.kind == p.ATEM
    assert encoded.frontend.reasoning.kind == p.RECIPIENT_HEADERS
    histories = [r for r in encoded.cache.regions if r.kind == p.PAGED_KEY_VALUE]
    assert {r.storage.buffer for r in histories} == {'cache.layer1.key', 'cache.layer1.value'}
    assert all(r.storage.stride == 128 * 2 * 64 for r in histories)
    for graph in encoded.graphs:
        assert not graph.HasField('draft_cursor')
        assert any(s.view.buffer == 'state_pool' and s.view.nbytes == pool.layout.row_bytes
                   and s.access == p.READ_WRITE for s in graph.state)
    decode = next(g for g in encoded.graphs if g.kind == p.DECODE)
    assert decode.pending_token.buffer == 'request_pending_token'
    assert all(b.source not in (p.PROMPT_TOKEN, p.PENDING_TOKEN) for b in decode.inputs)
    scored = next(g for g in encoded.graphs if g.kind == p.SCORED_PREFILL)
    assert any(b.source == p.PROMPT_TARGET_TOKEN for b in scored.inputs)


def test_preparation_accepts_shared_64_token_chunk_geometry():
    from ignition.recipes.muse_spark.bootstrap import PreparationConfig
    assert PreparationConfig(prefill_chunk_size=64).prefill_chunk_size == 64
    assert PreparationConfig(prefill_chunk_size=192).prefill_chunk_size == 192


def test_speculative_contract_tracks_committed_cursor_and_device_pending_token():
    config = SimpleNamespace(layers=2, attention=lambda i: Geometry(2048 if i == 0 else -1),
                             text={"hidden_size": 128})
    pool = ContinuationPool.allocate(config, TensorParallelShard(0, 1), 3, 2, 'cpu')
    buffers = BatchInputs.allocate(128, 2, 'cpu', capacity=4).buffers()
    signatures = {
        'speculative4x8-tokens17': GraphSignature('speculative', 4, 8,
            tuple(c.text_inputs('speculative')), speculation_enabled=True, packed_token_capacity=17),
        'decode4': GraphSignature('decode', 4, 1, tuple(c.text_inputs('decode')),
            speculation_enabled=False),
    }
    encoded = ParseDict(prepare(signatures, buffers, pool, vocabulary=201088, draft_lag=0), p.ModelContract())
    speculative, ordinary = encoded.graphs
    consumed = pool.layout.region('consumed_tokens')
    assert speculative.draft_cursor.state.buffer == 'state_pool'
    assert speculative.draft_cursor.state.offset == consumed.offset
    assert speculative.draft_cursor.state.stride == pool.layout.row_bytes
    assert speculative.draft_cursor.lag_tokens == 0
    assert speculative.pending_token.buffer == 'request_pending_token'
    assert not ordinary.HasField('draft_cursor')
    assert all(b.source not in (p.PROMPT_TOKEN, p.PENDING_TOKEN) for b in speculative.inputs)


def test_dflash_catalog_covers_width_one_and_packed_full_depth(tmp_path):
    from ignition.recipes.muse_spark.bootstrap import PreparationConfig

    config = PreparationConfig(dflash=True, draft_snapshot=tmp_path, speculative_token_capacity=32)
    shapes = config.speculative_round_shapes
    for batch in (1, 2, 4):
        assert (batch, 1, batch) in shapes
        assert (batch, 8, batch * 8) in shapes
    assert (4, 8, 17) in shapes
    assert config.decode_tokens == config.decode_samples == 32
    ordinary = PreparationConfig()
    assert not ordinary.speculative_round_shapes
    assert ordinary.decode_tokens == ordinary.decode_samples == 4
