# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-geometry hybrid snapshot storage and native graph state maintenance."""

from __future__ import annotations

import os

import pytest
import torch

from ignition.recipes.qwen38.state import (
    COPY,
    EOS_TOKEN_ID,
    GDN_LAYERS,
    RESET,
    HybridStateLayout,
    HybridStatePool,
    PreparedStateTransfers,
    StateTransferMetadata,
)

CUDA = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires explicitly assigned CUDA_VISIBLE_DEVICES",
)


def test_exact_hybrid_state_layout_and_snapshot_budget():
    layout = HybridStateLayout.build()
    assert layout.row_bytes == 116505344
    assert layout.row_bytes % 256 == 0
    assert layout.snapshot_capacity(4 << 30) == 36
    assert len(layout.regions) == 161
    assert GDN_LAYERS == tuple(i for i in range(48) if i % 4 != 3)
    for index, region in enumerate(layout.regions):
        assert region.offset % 256 == 0
        end = region.offset + region.nbytes
        assert end <= (
            layout.regions[index + 1].offset
            if index + 1 < len(layout.regions)
            else layout.row_bytes
        )
    assert layout.region("gdn.46.recurrent").shape == (48, 128, 128)
    with pytest.raises(KeyError):
        layout.region("gdn.3.recurrent")
    with pytest.raises(ValueError):
        layout.snapshot_capacity(-1)


def test_typed_views_share_exact_real_geometry_backing():
    pool = HybridStatePool.allocate(2, "cpu")
    assert pool.backing.shape == (2, 116505344)
    assert pool.consumed_tokens.shape == (2, 1)
    assert pool.ple_history.shape == (2, 2)
    assert torch.count_nonzero(pool.backing[0]) == 0
    assert torch.equal(pool.ple_history[1], torch.tensor([EOS_TOKEN_ID, EOS_TOKEN_ID]))
    for layer, state in pool.gdn.items():
        assert state.recurrent.shape == (2, 48, 128, 128)
        assert state.convolution.shape == (2, 10240, 3)
        assert state.recurrent.stride() == (pool.layout.row_bytes // 4, 16384, 128, 1)
        assert state.convolution.stride() == (pool.layout.row_bytes // 2, 3, 1)
        assert state.recurrent.untyped_storage().data_ptr() == pool.backing.data_ptr()
        assert (
            state.recurrent.data_ptr() - pool.backing.data_ptr()
            == pool.layout.region(f"gdn.{layer}.recurrent").offset
        )
    pool.gdn[0].recurrent[1, 0, 0, 0] = 1.25
    region = pool.layout.region("gdn.0.recurrent")
    assert (
        pool.backing[1, region.offset : region.offset + 4].view(torch.float32).item()
        == 1.25
    )
    assert pool.ple_convolution.stride() == (pool.layout.row_bytes // 2, 9, 1)
    assert pool.ple_history.stride() == (pool.layout.row_bytes // 8, 1)
    assert torch.count_nonzero(pool.gdn[1].recurrent) == 0
    pool.backing.fill_(0x3D)
    pool.reset_for_preparation()
    assert not pool.backing[0].count_nonzero()
    assert torch.equal(pool.ple_history[1], torch.tensor([EOS_TOKEN_ID, EOS_TOKEN_ID]))
    history = pool.layout.region("ple_history")
    assert not pool.backing[1, :history.offset].count_nonzero()
    assert not pool.backing[1, history.offset + history.nbytes:].count_nonzero()
    with pytest.raises(ValueError, match="CUDA"):
        PreparedStateTransfers(pool, StateTransferMetadata.allocate(1, "cpu"))


def test_speculative_state_retains_checkpoint_horizon_and_draft_boundary():
    from b12x.attention import qsa

    layout = HybridStateLayout.build(speculative_tokens=7)
    geometry = qsa.cache_requirements(main_page_size=64, position_axes=3, max_speculative_tokens=7)
    assert layout.speculative_tokens == 7
    assert layout.draft_ring_capacity == geometry.raw_ring_capacity == 16
    assert layout.region("gdn.0.convolution").shape == (10240, 10)
    assert layout.region("ple_convolution").shape == (10240, 16)
    assert layout.region("target_residual").shape == (10240,)
    assert layout.region("draft_raw_keys").shape == geometry.raw_k_ring_shape
    assert layout.region("draft_boundary_key").shape == geometry.main_k_page_shape
    assert layout.region("draft_boundary_value").shape == geometry.main_v_page_shape
    assert layout.region("draft_boundary_selector").shape == geometry.compressed_page_shape
    pool = HybridStatePool.allocate(2, "meta", speculative_tokens=7)
    for region in layout.regions:
        tensor = pool.region(region.name)
        assert tensor.shape == (2, *region.shape)
        assert tensor.dtype == region.dtype
        assert tensor.stride(0) * tensor.element_size() == layout.row_bytes
    for value in (-1, 8, True):
        with pytest.raises(ValueError):
            HybridStateLayout.build(speculative_tokens=value)


def test_partial_state_backing_initializes_and_resets_only_mapped_rows(monkeypatch):
    layout = HybridStateLayout.build()
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (8 * layout.row_bytes,) * 2)

    class Storage:
        def device_allocate(self, name, shape, dtype, *, group, maximum, units):
            assert name == "continuations" and group == "STATE_ROWS"
            assert shape == (4, layout.row_bytes) and maximum == 8 and units == 2
            backing = torch.zeros(shape, dtype=dtype)
            backing[2:].fill_(0xA5)
            return backing

    pool = HybridStatePool.allocate(4, "cpu", storage=Storage(), backed_slots=2)
    assert pool.backing.shape == (4, layout.row_bytes)
    assert pool.memory_capacity.tolist() == [2, 0]
    assert pool.gdn[0].recurrent.shape[0] == 4
    assert torch.all(pool.backing[2:] == 0xA5)
    pool.backing[:2].fill_(0x3D)
    pool.reset_for_preparation()
    assert not pool.backing[0].count_nonzero()
    assert torch.equal(pool.ple_history[1], torch.tensor([EOS_TOKEN_ID, EOS_TOKEN_ID]))
    assert torch.all(pool.backing[2:] == 0xA5)
    with pytest.raises(ValueError, match="core storage"):
        HybridStatePool.allocate(4, "meta", backed_slots=2)
    for count in (0, 1, 5, True):
        with pytest.raises(ValueError, match="backed state slots"):
            HybridStatePool.allocate(4, "meta", backed_slots=count)


def test_state_transfer_metadata_starts_inactive():
    metadata = StateTransferMetadata.allocate(4, "cpu")
    assert metadata.num_ops.item() == 0
    assert torch.all(metadata.source_slots == -1)
    assert torch.all(metadata.destination_slots == -1)
    assert torch.all(metadata.qsa_slots == -1)
    assert torch.all(metadata.accepted_tokens == 1)
    with pytest.raises(ValueError):
        StateTransferMetadata.allocate(0, "cpu")


def _device():
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    assert torch.cuda.get_device_capability(0) in ((12, 0), (12, 1))
    return torch.device("cuda", 0)


def _commands(metadata, commands):
    metadata.source_slots.fill_(-1)
    metadata.destination_slots.fill_(-1)
    metadata.qsa_slots.fill_(-1)
    metadata.operation.zero_()
    metadata.accepted_tokens.fill_(1)
    for i, (source, destination, operation, raw_slot, accepted) in enumerate(commands):
        metadata.source_slots[i] = source
        metadata.destination_slots[i] = destination
        metadata.operation[i] = operation
        metadata.qsa_slots[i] = raw_slot
        metadata.accepted_tokens[i] = accepted
    metadata.num_ops.fill_(len(commands))


def _replay(worker, index):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
    ticket = worker.submit(index)
    result = worker.poll(timeout_ms=30000)
    assert result is not None and result.ticket == ticket and result.error is None
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated


@CUDA
@torch.no_grad()
def test_native_full_state_copy_reset_restore_and_metadata_replay():
    from ignition.preparation.graphs import PreparedGraphBundle
    from ignition.recipes.qwen38.attention import QSAState

    device = _device()
    pool = HybridStatePool.allocate(6, device)
    states = tuple(
        QSAState.allocate(pages=1, slots=4, device=device) for _ in range(12)
    )
    metadata = StateTransferMetadata.allocate(4, device)
    operation = PreparedStateTransfers(pool, metadata, qsa_states=states)
    operation.prepare()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    bundle = PreparedGraphBundle(
        device=0,
        graphs={"state": graph},
        buffers={"state": pool.backing},
        resources=(operation, metadata, states),
    )
    worker = bundle.start_worker()
    try:
        pool.backing[1].random_(0, 256)
        pool.consumed_tokens[1] = 64
        pool.ple_history[1].copy_(torch.tensor([11, 12], device=device))
        pool.backing[2:5].fill_(83)
        for state in states:
            state.raw_k_ring.fill_(9)
            state.raw_logical_positions.fill_(777)
            state.raw_rope_positions.fill_(888)
            state.raw_interval_start_positions.fill_(999)
        _commands(
            metadata,
            [
                (1, 2, COPY, 0, 1),
                (-1, 3, RESET, 2, 1),
                (0, 4, COPY, 1, 1),
                (1, 0, COPY, 1, 1),
            ],
        )
        _replay(worker, 0)
        assert torch.count_nonzero(metadata.errors) == 0
        assert torch.equal(pool.backing[1], pool.backing[2])
        assert torch.count_nonzero(pool.backing[0]) == 0
        assert torch.all(pool.backing[4] == 83)
        expected_reset = torch.zeros(
            pool.layout.row_bytes, dtype=torch.uint8, device=device
        )
        region = pool.layout.region("ple_history")
        expected_reset[region.offset : region.offset + region.nbytes].view(
            torch.int64
        ).fill_(EOS_TOKEN_ID)
        assert torch.equal(pool.backing[3], expected_reset)
        for state in states:
            assert torch.all(state.raw_k_ring == 9)
            assert torch.all(state.raw_logical_positions[[0, 2]] == -1)
            assert torch.all(state.raw_rope_positions[[0, 2]] == -1)
            assert torch.all(state.raw_logical_positions[[1, 3]] == 777)
            assert torch.all(state.raw_rope_positions[[1, 3]] == 888)
            assert torch.equal(
                state.raw_interval_start_positions.cpu(),
                torch.tensor([63, 999, -1, 999]),
            )
        _commands(metadata, [(2, 4, COPY, 1, 2)])
        _replay(worker, 0)
        assert torch.equal(pool.backing[2], pool.backing[4])
        assert all(
            state.raw_interval_start_positions[1].item() == 62 for state in states
        )
        previous = pool.backing[1:3].clone()
        _commands(metadata, [(1, 2, COPY, -1, 1), (2, 1, COPY, -1, 1)])
        _replay(worker, 0)
        assert metadata.errors[:2].tolist() == [6, 6]
        assert torch.equal(pool.backing[1:3], previous)
        _commands(metadata, [(1, 2, COPY, -1, 1), (4, 2, COPY, -1, 1)])
        _replay(worker, 0)
        assert metadata.errors[:2].tolist() == [6, 6]
        assert torch.equal(pool.backing[1:3], previous)
        pool.consumed_tokens[3] = 65
        _commands(metadata, [(3, 2, COPY, 0, 1)])
        _replay(worker, 0)
        assert metadata.errors[0].item() == 4
        assert torch.equal(pool.backing[1:3], previous)
        _commands(metadata, [(999, 2, COPY, 0, 1)])
        _replay(worker, 0)
        assert metadata.errors[0].item() == 2
        assert torch.equal(pool.backing[1:3], previous)
        for command, error in (
            ((1, 2, 9, -1, 1), 1),
            ((1, 2, COPY, 4, 1), 3),
            ((1, 2, COPY, 0, 0), 5),
        ):
            _commands(metadata, [command])
            _replay(worker, 0)
            assert metadata.errors[0].item() == error
            assert torch.equal(pool.backing[1:3], previous)
        _commands(metadata, [(1, 2, COPY, 0, 1)])
        for count in (-1, 5):
            metadata.num_ops.fill_(count)
            _replay(worker, 0)
            assert metadata.errors.tolist() == [7] * 4
            assert torch.equal(pool.backing[1:3], previous)
        _commands(metadata, [])
        _replay(worker, 0)
        assert torch.count_nonzero(metadata.errors) == 0
        assert torch.equal(pool.backing[1:3], previous)
    finally:
        worker.close()
        graph.reset()


@CUDA
@torch.no_grad()
def test_actual_checkpoint_gdn_and_ple_convolution_use_shared_strided_slots():
    from b12x.preparation import PreparationSession
    from b12x.sequence import short_conv

    from ignition.components.recurrent import DeltaMetadata, prepare_sequence_binding
    from ignition.preparation.graphs import PreparedGraphBundle
    from ignition.components.gated_delta import GDNScratch, PreparedGDN
    from ignition.recipes.qwen38.ple import PLE_LAYER_PREFIX
    from ignition.recipes.qwen38.weights import QwenWeightLoader
    from tests.recipes import test_qwen38_gdn as canonical

    device = _device()
    manifest = canonical.manifest.__wrapped__()
    reference = canonical.reference_module.__wrapped__(manifest)
    module, projections = canonical._load_cuda(manifest)
    with QwenWeightLoader(
        manifest, device, prefixes=(PLE_LAYER_PREFIX + ".conv1d.",)
    ) as loader:
        conv_weight = loader.load_tensor(PLE_LAYER_PREFIX + ".conv1d.weight").squeeze(1)
    pool = HybridStatePool.allocate(4, device)
    transfer_metadata = StateTransferMetadata.allocate(1, device)
    transfers = PreparedStateTransfers(pool, transfer_metadata)
    transfers.prepare()
    inputs = {
        mode: torch.zeros((tokens, 2560), dtype=torch.bfloat16, device=device)
        for mode, tokens in (("prefill", 8), ("decode", 1))
    }
    conv_inputs = {
        mode: torch.zeros((value.shape[0], 10240), dtype=torch.bfloat16, device=device)
        for mode, value in inputs.items()
    }
    conv_outputs = {
        mode: torch.zeros_like(value) for mode, value in conv_inputs.items()
    }
    scratch = {
        mode: GDNScratch.allocate(value.shape[0], device)
        for mode, value in inputs.items()
    }
    metadata = {mode: DeltaMetadata.allocate(1, device) for mode in inputs}
    workspace = torch.empty(80 << 20, dtype=torch.uint8, device=device)
    projection_workspace = torch.empty(64 << 20, dtype=torch.uint8, device=device)
    with PreparationSession(
        device=device, autotune=False, compile_workers=1
    ) as session:
        adapters, convolution_bindings = {}, {}
        for mode, value in inputs.items():
            adapters[mode] = PreparedGDN(
                module,
                projections,
                value,
                scratch[mode],
                pool.gdn[0],
                metadata[mode],
                mode=mode,
                session=session,
                name=f"strided.gdn.{mode}",
                workspace=workspace,
                projection_workspace=projection_workspace,
            )
            assert adapters[mode].workspace.data_ptr() == workspace.data_ptr()
            assert all(
                projection.workspace is projection_workspace
                for projection in adapters[mode].projections.values()
            )
            m = metadata[mode]
            args = {
                "x": conv_inputs[mode],
                "conv_weight": conv_weight,
                "query_start_loc": m.query_start_loc,
                "state_slot_ids": m.convolution_slots,
                "state_is_fresh": m.fresh,
                "num_accepted_tokens": m.accepted_tokens,
                "num_seqs": m.num_seqs,
                "num_tokens": m.num_tokens,
                "conv_state": pool.ple_convolution,
                "out": conv_outputs[mode],
            }
            plan = short_conv.plan(
                short_conv.Caps(
                    device=device,
                    mode=mode,
                    channels=10240,
                    kernel_size=4,
                    max_tokens=value.shape[0],
                    max_seqs=1,
                    max_state_slots=4,
                    dilation=3,
                ),
                invocation=short_conv.invocation_from_tensors(**args),
            )
            def set_convolution(binding, mode=mode):
                convolution_bindings[mode] = binding

            prepare_sequence_binding(
                plan,
                args,
                session=session,
                name=f"strided.ple.conv.{mode}",
                public_bind=short_conv.bind,
                bind_plan=True,
                mode=mode,
                set_binding=set_convolution,
            )
            adapters[mode](value)
            short_conv.run(convolution_bindings[mode])
        session.freeze()
        graph_pool = torch.cuda.graph_pool_handle()
        graphs = {mode: torch.cuda.CUDAGraph() for mode in (*inputs, "copy")}
        for mode, value in inputs.items():
            with torch.cuda.graph(graphs[mode], pool=graph_pool):
                adapters[mode](value)
                short_conv.run(convolution_bindings[mode])
        with torch.cuda.graph(graphs["copy"], pool=graph_pool):
            transfers()
        bundle = PreparedGraphBundle(
            device=0,
            graphs=graphs,
            buffers={"state": pool.backing, **inputs},
            resources=(
                session,
                adapters,
                transfers,
                metadata,
                scratch,
                convolution_bindings,
                conv_inputs,
                conv_outputs,
            ),
        )
        worker = bundle.start_worker()
        try:
            generator = torch.Generator().manual_seed(517)
            source = (torch.randn((8, 2560), generator=generator) * 0.1).bfloat16()
            convolution_source = (
                torch.randn((8, 10240), generator=generator) * 0.1
            ).bfloat16()
            cache = canonical._cache()
            expected = reference(source[:7][None], cache_params=cache)[0]
            inputs["prefill"][:7].copy_(source[:7])
            conv_inputs["prefill"][:7].copy_(convolution_source[:7])
            canonical._metadata(metadata["prefill"], [7], [1], fresh=True, initial=[0])
            _replay(worker, bundle.graph_indices["prefill"])
            canonical._numerical(scratch["prefill"].output[:7], expected)
            expected_conv = torch.nn.functional.silu(
                torch.nn.functional.conv1d(
                    torch.nn.functional.pad(convolution_source[:7].T[None], (9, 0)),
                    conv_weight.cpu()[:, None],
                    groups=10240,
                    dilation=3,
                )
            )[0].T
            torch.testing.assert_close(
                conv_outputs["prefill"][:7].cpu(), expected_conv, rtol=0, atol=0
            )
            pool.consumed_tokens[1] = 7
            _commands(transfer_metadata, [(1, 2, COPY, -1, 1)])
            _replay(worker, bundle.graph_indices["copy"])
            assert torch.equal(pool.backing[1], pool.backing[2])
            original = pool.backing[1].clone()
            expected = reference(source[7:8][None], cache_params=cache)[0]
            inputs["decode"].copy_(source[7:8])
            conv_inputs["decode"].copy_(convolution_source[7:8])
            canonical._metadata(metadata["decode"], [1], [2], fresh=False, initial=[2])
            _replay(worker, bundle.graph_indices["decode"])
            canonical._numerical(scratch["decode"].output, expected)
            canonical._numerical(
                pool.gdn[0].recurrent[2].transpose(-1, -2),
                cache.layers[0].recurrent_states[0][0],
                limit=0.012,
            )
            expected_conv = torch.nn.functional.silu(
                torch.nn.functional.conv1d(
                    torch.nn.functional.pad(convolution_source.T[None], (9, 0)),
                    conv_weight.cpu()[:, None],
                    groups=10240,
                    dilation=3,
                )
            )[0].T[-1:]
            torch.testing.assert_close(
                conv_outputs["decode"].cpu(), expected_conv, rtol=0, atol=0
            )
            assert torch.equal(pool.backing[1], original)
            assert torch.count_nonzero(pool.backing[0]) == 0
        finally:
            worker.close()
            for graph in graphs.values():
                graph.reset()
