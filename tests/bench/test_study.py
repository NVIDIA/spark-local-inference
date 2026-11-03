"""Private speculative width geometry and shared request commit semantics."""

import pytest
import torch

from ignition.components.sampling import DeviceRequestState, RequestAdmission
from ignition.components.speculation import PreparedSpeculativeStateCommit, SpeculativeAcceptance


@pytest.mark.parametrize("width", [8, 13, 16])
@pytest.mark.parametrize("clipping", ["none", "budget", "eos", "inactive"])
def test_private_commit_every_accepted_prefix(width, clipping):
    for accepted in range(width):
        state = DeviceRequestState.allocate("cpu", vocab_size=64, capacity=1)
        state.admit(0, RequestAdmission(generation=1, seed=0, max_new_tokens=64,
                    temperature=1, top_k=0, top_p=1, eos_token_ids=()), [1, 2] * 4096, consumed_tokens=8192)
        state.pending_token[0] = 3
        state.generated_tokens[0] = 1
        ids = torch.arange(3, 3 + width)[None, :]
        acceptance = SpeculativeAcceptance.allocate(1, width, "cpu")
        acceptance.token_ids.copy_(ids + 1)
        acceptance.proposed_count.fill_(accepted + 1)
        acceptance.accepted_drafts.fill_(accepted)
        acceptance.offered_drafts.fill_(width - 1)
        acceptance.valid.fill_(True)
        live = torch.tensor([0 if clipping == "inactive" else 1], dtype=torch.int32)
        if clipping == "budget":
            state.max_new_tokens[0] = 3
        if clipping == "eos":
            state.eos_count[0] = 1
            state.eos_token_ids[0, 0] = 5
        commit = PreparedSpeculativeStateCommit(state, acceptance, ids,
            torch.tensor([0], dtype=torch.int32), torch.tensor([1]), live, None, None, None)
        commit.prepare()
        commit()
        count = 0 if clipping == "inactive" else min(accepted + 1, 2 if clipping in ("budget", "eos") else width)
        assert commit.emitted_count.item() == count
        assert commit.accepted_drafts.item() == min(accepted, count)
        assert state.consumed_tokens.item() == 8192 + count
        assert state.pending_token.item() == 3 + count
        assert commit.emitted_ids[0, :count].tolist() == list(range(4, 4 + count))
        assert commit.emitted_ids[0, count:].eq(-1).all()


def test_native_width_requires_matching_completion_capacity():
    from ignition.components.completion import CompletionOutputs
    from ignition.components.speculation import PreparedSpeculativeCommit
    from ignition.preparation.execution import GraphSignature
    from tests.preparation.test_execution import _execution

    completion = CompletionOutputs.allocate((2, 4, 24), device="cpu")
    for width in (13, 16, 32):
        signature = GraphSignature("speculative", 1, width)
        with pytest.raises(ValueError, match="completion token capacity"):
            _execution(graphs={"round": object()}, signatures={"round": signature}, buffers={
                **completion.buffers(), "frame_index": torch.zeros(1, dtype=torch.int32),
                "submission_sequence": torch.zeros(1, dtype=torch.int64)})
        with pytest.raises(ValueError, match="completion token capacity"):
            PreparedSpeculativeCommit(None, None, torch.zeros((1, width)), None, None, None,
                                      completion, None, None)


def test_study_wrapper_rejects_manifest_and_closes_graphs_before_model():
    from types import SimpleNamespace
    from ignition.bench.study import StudyModel

    closed = []
    model = StudyModel(SimpleNamespace(close=lambda: closed.append("model")),
                       {8: SimpleNamespace(close=lambda: closed.append("study8")),
                        16: SimpleNamespace(close=lambda: closed.append("study16"))})
    with pytest.raises(ValueError, match="cannot produce native package metadata"):
        model.manifest(all_variants=True)
    model.close()
    assert closed == ["study8", "study16", "model"]


@pytest.mark.parametrize("width", [8, 13, 16])
def test_private_sampling_capture_all_prefixes_and_clipping(width):
    import os
    from dataclasses import fields

    if os.environ.get("FLARE_RUN_CUDA_TESTS") != "1":
        pytest.skip("requires an explicitly assigned CUDA device")
    from ignition.components.speculation import PreparedStochasticVerification, SpeculativeRngState
    from ignition.preparation.compilation import compile_forward

    state = DeviceRequestState.allocate("cuda", vocab_size=248320, capacity=1)
    state.admit(0, RequestAdmission(generation=7, seed=19, max_new_tokens=128,
                temperature=1, top_k=0, top_p=1, eos_token_ids=()), [1] * 8191, consumed_tokens=8191)
    state.pending_token[0] = 3
    state.generated_tokens[0] = 1
    slots = torch.zeros(1, dtype=torch.int32, device="cuda")
    epochs = torch.full((1,), 7, dtype=torch.int64, device="cuda")
    live = torch.ones(1, dtype=torch.int32, device="cuda")
    inputs = torch.arange(3, 3 + width, device="cuda")[None, :]
    drafts = torch.zeros((1, width - 1, state.vocab_size), device="cuda")
    targets = torch.zeros((1, width, state.vocab_size), device="cuda")
    for column in range(width - 1):
        drafts[:, column, 4 + column] = 1
    rng = SpeculativeRngState.allocate("cuda", capacity=1)
    verify = PreparedStochasticVerification(drafts, inputs[:, 1:].contiguous(), targets,
        state, slots, epochs, live, rng=rng)
    verify.prepare()
    commit = PreparedSpeculativeStateCommit(state, verify.output, inputs, slots, epochs, live,
                                            None, None, None, rng=rng)
    saved = {field.name: getattr(state, field.name).clone() for field in fields(state)}

    def restore():
        for name, tensor in saved.items():
            getattr(state, name).copy_(tensor)
        rng.generation.zero_()
        rng.target_offset.zero_()
        live.fill_(1)
        targets.zero_()
        for column in range(width):
            targets[:, column, 4 + column] = 1

    def run():
        verify()
        commit()

    restore()
    compiled = compile_forward(run)
    compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    for accepted in range(width):
        restore()
        targets[:, accepted].zero_()
        targets[:, accepted, 100] = 1
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        assert commit.emitted_count.item() == accepted + 1
        assert commit.accepted_drafts.item() == accepted
        assert state.consumed_tokens.item() == 8192 + accepted
        assert state.pending_token.item() == 100
        assert rng.target_offset.item() == verify.draw_reservation
    for clipping in ("eos", "budget", "inactive"):
        restore()
        if clipping == "eos":
            state.eos_count[0] = 1
            state.eos_token_ids[0, 0] = 4
        elif clipping == "budget":
            state.max_new_tokens[0] = 2
        else:
            live.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert commit.emitted_count.item() == (0 if clipping == "inactive" else 1)
    graph.reset()
