# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Speculative metadata, cross-width checkpoints, and MTP prefix alignment."""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.qwen38.draft_state import DraftMetadata
from ignition.recipes.qwen38.state import HybridStatePool


@pytest.mark.parametrize("shared", [False, True])
def test_draft_rows_pack_unequal_lengths_and_inactive_lanes(shared):
    backing = torch.full((16, 10240), 123, dtype=torch.bfloat16) if shared else None
    metadata = DraftMetadata(tokens=8, rows=4, context=256, device="cpu",
                             hidden=None if backing is None else backing[:8])
    slots = torch.tensor([2, 0, 3, -1], dtype=torch.int32)
    lengths = torch.tensor([2, 0, 1, 0])
    starts = torch.tensor([62, 71, 128, 0])
    tokens = torch.tensor([[901, 902], [100, 101], [401, 402], [0, 0]])
    hidden = torch.arange(8).to(torch.bfloat16)[:, None].expand(8, 10240).contiguous()
    rope = (torch.arange(8)[:, None] + torch.tensor([10, 20, 30])[None, :]).reshape(4, 2, 3)
    tables = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34], [41, 42, 43, 44]], dtype=torch.int32)
    metadata.set(slots, lengths, starts, tokens, hidden, rope, tables, capacity=2)
    assert metadata.qsa.query_start_loc.tolist() == [0, 2, 2, 3, 3]
    assert metadata.qsa.request_ids.tolist() == [0, 0, 2, -1, -1, -1, -1, -1]
    assert metadata.qsa.query_positions[:3].tolist() == [62, 63, 128]
    assert metadata.qsa.slot_mapping[:3].tolist() == [31 * 64 + 62, 31 * 64 + 63, 43 * 64]
    assert metadata.token_ids[:3].tolist() == [901, 902, 401]
    assert metadata.hidden[:3, 0].tolist() == [0, 1, 4]
    assert metadata.qsa.rope_positions[:3].tolist() == [[10, 20, 30], [11, 21, 31], [14, 24, 34]]
    assert metadata.qsa.raw_state_slot_ids.tolist() == [3, -1, 4, -1]
    assert metadata.qsa.sequence_lengths.tolist() == [64, 0, 129, 0]
    assert not metadata.hidden[3:].count_nonzero()
    if shared:
        assert metadata.hidden.data_ptr() == backing.data_ptr()
        assert backing[8:].eq(123).all()


def test_empty_draft_absorption_preserves_inactive_metadata():
    metadata = DraftMetadata(tokens=1, rows=1, context=64, device="cpu")
    metadata.set(torch.tensor([0], dtype=torch.int32), torch.tensor([0]), torch.tensor([0]),
        torch.tensor([7]), torch.zeros((1, 10240), dtype=torch.bfloat16), torch.zeros((1, 3), dtype=torch.int64),
        torch.full((4, 1), -1, dtype=torch.int32), capacity=1)
    assert metadata.qsa.query_start_loc.tolist() == [0, 0]
    assert metadata.qsa.request_ids.tolist() == [-1]
    assert metadata.qsa.raw_state_slot_ids.tolist() == [-1]
    assert metadata.qsa.slot_mapping.tolist() == [-1]


CUDA = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                         reason="requires an explicitly assigned CUDA device")


@CUDA
@torch.no_grad()
def test_live_row_state_survives_width_one_and_retains_accepted_endpoint():
    from ignition.recipes.qwen38.decoder_state import DecoderInputs
    from ignition.recipes.qwen38.sampling import DeviceRequestState, RequestAdmission
    from ignition.recipes.qwen38.speculative_state import PreparedVerificationState

    device = torch.device("cuda", 0)
    pool = HybridStatePool.allocate(34, device, speculative_tokens=7)
    requests = DeviceRequestState.allocate(device, consumed_tokens=pool.consumed_tokens[1:5, 0],
        ple_history=pool.ple_history[1:5])
    inputs = DecoderInputs.allocate(device, context=256, tokens=64)
    inputs.request_state_rows[:, 0] = torch.arange(1, 5, device=inputs.token_ids.device)
    inputs.request_state_rows[:, 1:] = torch.arange(6, 34, device=inputs.token_ids.device).view(4, 7)
    candidates = torch.arange(32, device=device, dtype=torch.int64).reshape(4, 8) + 901
    accepted = torch.ones(4, dtype=torch.int32, device=device)
    states = {width: PreparedVerificationState(inputs, requests, pool, candidates, accepted,
        width=width, rows=1) for width in range(1, 9)}
    counts = torch.tensor([5], dtype=torch.int32, device=device)
    before = torch.tensor([61], dtype=torch.int64, device=device)
    residual = torch.arange(8, device=device, dtype=torch.bfloat16)[:, None].expand(8, 10240).contiguous()
    hidden = residual[:, :2560].contiguous()
    states[8].prepare_commit(counts, before, residual, hidden)
    for state in states.values():
        state.functional()
    requests.admit(2, RequestAdmission(11, 42), list(range(128)), consumed_tokens=61)
    requests.pending_token[2] = 901
    inputs.row_slot_ids[0] = 2
    inputs.row_generations[0] = 11
    inputs.live_rows.fill_(1)
    inputs.page_tables[2].copy_(torch.tensor([5, 3, 1, 7], device=device, dtype=torch.int32))
    accepted[2] = 8
    runtime = states[1].begin_runtime(candidates)
    assert runtime.gdn.accepted_tokens.tolist() == [8]
    # Verification resumes from and commits into the live row; later columns stay null.
    assert runtime.gdn.state_indices.tolist() == [[3, 0, 0, 0, 0, 0, 0, 0]]
    assert runtime.ple.token_ids.tolist() == [901]
    for width in range(2, 8):
        runtime = states[width].begin_runtime(candidates)
        assert runtime.ple.token_ids.tolist() == list(range(901, 901 + width))
        assert runtime.qsa.query_positions.tolist() == list(range(61, 61 + width))
        assert runtime.gdn.accepted_tokens.tolist() == [8]
    runtime = states[8].begin_runtime(candidates)
    assert runtime.qsa.slot_mapping.tolist() == [381, 382, 383, 192, 193, 194, 195, 196]
    for layer in pool.gdn.values():
        layer.convolution[3].copy_(torch.arange(100, 110, device=device, dtype=torch.bfloat16))
    pool.ple_convolution[3].copy_(torch.arange(100, 116, device=device, dtype=torch.bfloat16))
    states[8].commit_runtime(runtime, hidden, residual, counts)
    assert accepted[2].item() == 5
    assert pool.last_hidden[3, 0].item() == 4
    assert pool.region("target_residual")[3, 0].item() == 4
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    captured_accepted = torch.empty((1,), dtype=torch.int32, device=device)
    with torch.cuda.graph(graph, stream=stream):
        captured_runtime = states[1].begin_runtime(candidates)
        captured_accepted.copy_(captured_runtime.gdn.accepted_tokens)
    before_allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
    graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before_allocations
    assert captured_accepted.tolist() == [5]
    assert captured_runtime.gdn.state_indices.shape == (1, 8)
    graph.reset()


@pytest.mark.parametrize("temperature", [.7, 1.])
@pytest.mark.parametrize("copy_epoch", [12, 13])
def test_recursive_draft_reorders_residuals_and_physical_copy_hints(temperature, copy_epoch):
    from ignition.recipes.qwen38.sampling import DeviceRequestState, RequestAdmission
    from ignition.recipes.qwen38.speculative_model import PreparedSpeculativeRound

    batch, width = 2, 4
    requests = DeviceRequestState.allocate("cpu", vocab_size=128)
    requests.admit(3, RequestAdmission(13, 42, temperature=temperature, eos_token_ids=()), [3] * 64, consumed_tokens=64)
    requests.pending_token[3] = 17
    # Packed lane zero is inactive; lane one maps to physical slot three.
    slots = torch.tensor([1, 3], dtype=torch.int32)
    epochs = torch.tensor([0, 13], dtype=torch.int64)
    regions = {"target_residual": torch.zeros((5, 10240), dtype=torch.bfloat16),
               "target_residual_position": torch.zeros((5, 3), dtype=torch.int64)}
    pool = SimpleNamespace(region=lambda name: regions[name], rope_delta=torch.zeros((5, 1), dtype=torch.int64))
    from ignition.recipes.qwen38.draft_state import DRAFT_RAW_FIELDS

    cache = SimpleNamespace(**{name: torch.zeros((1,), dtype=torch.int64) for name in DRAFT_RAW_FIELDS})
    owner = SimpleNamespace(requests=requests, pool=pool,
        copy_generation=torch.tensor([0, 0, 0, copy_epoch]),
        inputs=SimpleNamespace(request_state_rows=torch.arange(1, 5)[:, None].expand(4, 8),
            row_slot_ids=slots, row_generations=epochs, live_rows=torch.tensor([2], dtype=torch.int32),
            query_lengths=torch.full((batch,), width, dtype=torch.int32)),
        cache={48: cache},
        draft_rng_generation=torch.zeros(4, dtype=torch.int64), draft_rng_offset=torch.zeros(4, dtype=torch.int64),
        copy_expected=torch.tensor([-1, -1, -1, 64]), copy_count=torch.tensor([0, 0, 0, 1]),
        copy_ids=torch.full((4, 7), -1, dtype=torch.int64),
        draft_tables=torch.ones((4, 4), dtype=torch.int32))
    owner.copy_ids[3, 0] = 27
    operation = PreparedSpeculativeRound.__new__(PreparedSpeculativeRound)
    operation.owner, operation.batch, operation.width = owner, batch, width
    operation.gdn_commit = None
    operation.draft_metadata = DraftMetadata(tokens=batch, rows=batch, context=256, device="cpu").functional()
    operation.absorb_metadata = DraftMetadata(tokens=batch * width, rows=batch, context=256, device="cpu").functional()

    class Draft:
        inputs = []
        tokens = []

        def __call__(self, tokens, hidden, metadata, indices):
            self.inputs.append(hidden.clone())
            self.tokens.append(tokens.clone())
            residual = torch.zeros((batch, 10240), dtype=torch.bfloat16)
            residual[0].fill_(99 + len(self.inputs))
            return SimpleNamespace(residual=residual, logits=torch.zeros((batch, 128)))

    class Proposal:
        layout = {"valid": ((batch, 1), torch.bool)}

        def run(self, scratch, logits, *, sampled, **kwargs):
            sampled.fill_(21)
            scratch["valid"].copy_(torch.tensor([[False], [True]]))
            return torch.nn.functional.one_hot(torch.full((batch, 1), 21), 128).float()

    class Target:
        layout, vocab_ids = {}, torch.arange(128)

        def run(self, scratch, logits, **kwargs):
            scratch["valid"] = torch.ones((batch, width), dtype=torch.bool)
            return torch.full((batch, width, 128), 1 / 128)

    class Verify:
        layout = {}
        probabilities = None
        drafts = None

        def run(self, scratch, draft_probs, target_probs, target_valid, *, drafts, **kwargs):
            self.probabilities, self.drafts = draft_probs.clone(), drafts.clone()
            return object()

    class Commit:
        mask = None

        def __call__(self, logits, *, copy_offered_mask, **kwargs):
            self.mask = copy_offered_mask.clone()
            return SimpleNamespace(emitted_count=torch.tensor([0, 1], dtype=torch.int32))

    verification, commit = Verify(), Commit()
    operation.draft = Draft()
    operation.absorb = lambda *args: None
    operation.operations = (Target(), (Proposal(),) * 3, verification, commit)
    operation.state = SimpleNamespace(
        begin_runtime=lambda candidates, inputs: SimpleNamespace(ple=SimpleNamespace(token_ids=candidates[:, :width].reshape(-1)),
            qsa=SimpleNamespace(rope_positions=torch.zeros((batch * width, 3), dtype=torch.int64))),
        commit_runtime=lambda *args: None)
    operation.decoder = lambda runtime: (torch.zeros((batch * width, 2560), dtype=torch.bfloat16),
        torch.zeros((batch * width, 128)), SimpleNamespace(residual=torch.zeros((batch * width, 10240), dtype=torch.bfloat16)), None)
    operation()
    chosen = 27 if copy_epoch == 13 else 21
    assert verification.drafts[1].tolist() == [chosen, 21, 21]
    assert commit.mask.tolist() == [0, int(copy_epoch == 13)]
    assert verification.probabilities[1, 0, chosen] == 1
    assert verification.probabilities[1, 0].count_nonzero() == 1
    assert operation.draft.tokens[1][0] == chosen
    assert operation.draft.inputs[1][0].eq(100).all()
    assert operation.draft.inputs[2][0].eq(101).all()
    assert owner.draft_rng_offset.tolist() == [0, 0, 0, 12]
    assert owner.draft_rng_generation.tolist() == [0, 0, 0, 13]


@pytest.mark.parametrize("copy_epoch, expected_mask", [(13, 7), (12, 0)])
def test_functional_round_passes_physical_copy_mask_to_commit(copy_epoch, expected_mask):
    from ignition.recipes.qwen38.sampling import DeviceRequestState, RequestAdmission
    from ignition.recipes.qwen38.speculative_model import PreparedSpeculativeRound
    from ignition.recipes.qwen38.draft_state import DRAFT_RAW_FIELDS

    batch, width = 1, 4
    requests = DeviceRequestState.allocate("cpu", vocab_size=128)
    requests.admit(3, RequestAdmission(13, 42, eos_token_ids=()), [3] * 64, consumed_tokens=64)
    requests.pending_token[3] = 17
    slots = torch.tensor([3], dtype=torch.int32)
    regions = {"target_residual": torch.zeros((5, 10240), dtype=torch.bfloat16),
               "target_residual_position": torch.zeros((5, 3), dtype=torch.int64)}
    pool = SimpleNamespace(region=lambda name: regions[name], rope_delta=torch.zeros((5, 1), dtype=torch.int64))
    cache = SimpleNamespace(**{name: torch.zeros((1,), dtype=torch.int64) for name in DRAFT_RAW_FIELDS})
    owner = SimpleNamespace(requests=requests, pool=pool, cache={48: cache},
        inputs=SimpleNamespace(request_state_rows=torch.arange(1, 5)[:, None].expand(4, 8),
            row_slot_ids=slots, row_generations=torch.tensor([13]), live_rows=torch.tensor([1], dtype=torch.int32), query_lengths=torch.tensor([width], dtype=torch.int32)),
        copy_generation=torch.tensor([0, 0, 0, copy_epoch]), copy_expected=torch.tensor([-1, -1, -1, 64]),
        copy_count=torch.tensor([0, 0, 0, 3]), copy_ids=torch.full((4, 7), -1, dtype=torch.int64),
        draft_rng_generation=torch.zeros(4, dtype=torch.int64), draft_rng_offset=torch.zeros(4, dtype=torch.int64),
        draft_tables=torch.ones((4, 4), dtype=torch.int32))
    owner.copy_ids[3, :3] = torch.tensor([27, 28, 29])
    operation = PreparedSpeculativeRound.__new__(PreparedSpeculativeRound)
    operation.owner, operation.batch, operation.width, operation.token_limits = owner, batch, width, None
    operation.gdn_commit = None
    operation.draft_metadata = DraftMetadata(tokens=batch, rows=batch, context=256, device="cpu").functional()
    operation.absorb_metadata = DraftMetadata(tokens=batch * width, rows=batch, context=256, device="cpu").functional()

    class Proposal:
        layout = {"valid": ((batch, 1), torch.bool)}

        def run(self, scratch, logits, *, sampled, **kwargs):
            sampled.fill_(21)
            scratch["valid"].fill_(True)
            return torch.nn.functional.one_hot(torch.full((batch, 1), 21), 128).float()

    class Target:
        layout, vocab_ids = {}, torch.arange(128)

        def run(self, scratch, logits, *, candidates, **kwargs):
            scratch["valid"] = torch.ones((batch, width), dtype=torch.bool)
            return torch.full((batch, width, 128), 1 / 128)

    class Verify:
        layout = {}

        def run(self, scratch, draft_probs, target_probs, target_valid, **kwargs):
            return object()

    class Commit:
        seen = None

        def __call__(self, logits, *, acceptance, inputs, copy_offered_mask, **kwargs):
            self.seen = copy_offered_mask.clone()
            return SimpleNamespace(emitted_count=torch.ones(batch, dtype=torch.int32))

    commit = Commit()
    operation.operations = (Target(), (Proposal(),) * 3, Verify(), commit)
    operation.draft = lambda *args: SimpleNamespace(residual=torch.zeros((batch, 10240), dtype=torch.bfloat16),
                                                     logits=torch.zeros((batch, 128)))
    operation.absorb = lambda *args: None
    operation.state = SimpleNamespace(_validate=lambda: None,
        begin_runtime=lambda candidates, inputs: SimpleNamespace(ple=SimpleNamespace(token_ids=candidates[:, :width].reshape(-1)),
            qsa=SimpleNamespace(rope_positions=torch.zeros((batch * width, 3), dtype=torch.int64))),
        commit_runtime=lambda *args: None)
    operation.decoder = lambda runtime: (torch.zeros((batch * width, 2560), dtype=torch.bfloat16),
        torch.zeros((batch * width, 128)), SimpleNamespace(residual=torch.zeros((batch * width, 10240), dtype=torch.bfloat16)), None)
    operation()
    assert commit.seen.tolist() == [expected_mask]


@CUDA
@torch.no_grad()
def test_fixed_verification_masks_context_tail_without_changing_shape():
    from ignition.recipes.qwen38.decoder_state import DecoderInputs
    from ignition.recipes.qwen38.sampling import DeviceRequestState, RequestAdmission
    from ignition.recipes.qwen38.speculative_state import PreparedVerificationState

    pool = HybridStatePool.allocate(34, "cuda", speculative_tokens=7)
    requests = DeviceRequestState.allocate("cuda", consumed_tokens=pool.consumed_tokens[1:5, 0],
        ple_history=pool.ple_history[1:5])
    inputs = DecoderInputs.allocate("cuda", context=64, tokens=64)
    inputs.request_state_rows[:, 0] = torch.arange(1, 5, device=inputs.token_ids.device)
    inputs.request_state_rows[:, 1:] = torch.arange(6, 34, device=inputs.token_ids.device).view(4, 7)
    candidates = torch.zeros((4, 8), dtype=torch.int64, device="cuda")
    accepted = torch.ones(4, dtype=torch.int32, device="cuda")
    inputs.query_lengths.copy_(torch.tensor([2, 4, 1, 3], dtype=torch.int32, device="cuda"))
    for slot, position in enumerate([62, 60, 63, 61]):
        requests.admit(slot, RequestAdmission(1, 42), [7] * position, consumed_tokens=position)
        requests.pending_token[slot] = 7
    inputs.row_slot_ids.copy_(torch.arange(4, dtype=torch.int32, device="cuda"))
    inputs.row_generations.fill_(1)
    inputs.live_rows.fill_(4)
    inputs.page_tables.fill_(0)
    state = PreparedVerificationState(inputs, requests, pool, candidates, accepted,
        width=4, rows=4, token_limits=inputs.query_lengths)
    state.functional()
    runtime = state.begin_runtime(candidates)
    assert requests.status.tolist() == [1] * 4
    assert runtime.qsa.query_positions.reshape(4, 4).tolist() == [
        [62, 63, -1, -1], [60, 61, 62, 63], [63, -1, -1, -1], [61, 62, 63, -1]]
    assert runtime.qsa.sequence_lengths.tolist() == [64] * 4
    assert runtime.gdn.query_start_loc.tolist() == [0, 4, 8, 12, 16]
    assert runtime.qsa.slot_mapping.reshape(4, 4).tolist() == [
        [62, 63, -1, -1], [60, 61, 62, 63], [63, -1, -1, -1], [61, 62, 63, -1]]
