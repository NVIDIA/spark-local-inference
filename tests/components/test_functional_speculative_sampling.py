# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Runtime sampler operands and commit outputs have independent invocation storage."""

import pytest
import torch

from ignition.components.sampling import DeviceRequestState, RequestAdmission, RUNNING
from ignition.components.speculation import (
    PreparedSpeculativeDistributions, PreparedSpeculativeStateCommit,
    PreparedStochasticVerification, SpeculativeAcceptance, SpeculativeRngState,
)
from ignition.preparation.triton import PreparedTriton


def request_state(batch=2, vocabulary=64):
    state = DeviceRequestState.allocate("cpu", vocab_size=vocabulary, capacity=batch)
    for slot in range(batch):
        state.admit(slot, RequestAdmission(generation=slot + 1, seed=9 + slot, max_new_tokens=24,
                    temperature=1., top_k=0, top_p=1., eos_token_ids=()), [1, 2], consumed_tokens=2)
        state.pending_token[slot] = 3
        state.generated_tokens[slot] = 1
    return state


def metadata(batch=2):
    return torch.arange(batch, dtype=torch.int32), torch.arange(1, batch + 1), torch.tensor([batch], dtype=torch.int32)


def scratch_for(operation):
    return {name: torch.empty(shape, dtype=dtype) for name, (shape, dtype) in operation.layout.items()}


@pytest.mark.parametrize("release_first", [False, True])
def test_conditional_scores_consume_runtime_candidates_after_template_release(release_first):
    state = request_state()
    state.parameters[:] = torch.tensor([.7, .9, .1, 1.2, .3, .2])
    state.histogram[:, 3] = 2
    logits = torch.linspace(-2, 2, 2 * 3 * 64).reshape(2, 3, 64)
    candidates = torch.tensor([[3, 7], [4, 5]])
    bound = PreparedSpeculativeDistributions(logits, state, *metadata(), candidates)
    bound.prepare_scores()
    functional = PreparedSpeculativeDistributions(torch.zeros_like(logits), state, *metadata(), torch.full_like(candidates, -1))
    if release_first:
        functional.release_scratch()
    functional.functional()
    scratch = scratch_for(functional)
    functional._prepare_scores(scratch, logits, candidates=candidates)
    for name in ("scores", "maximum", "valid", "packed_top_k", "packed_top_p"):
        torch.testing.assert_close(scratch[name], getattr(bound, name), rtol=0, atol=0)
    assert functional.logits is functional.candidates is functional.rows is functional.columns is None
    assert all(getattr(functional, name) is None for name in functional.layout)
    assert all(getattr(functional.normalization, name) is None
               for name in ("output", "valid", "maxima", "masses", "totals"))


class NoLogprobs:
    def __init__(self):
        self.inputs = []

    def prepare(self):
        pass

    def functional(self):
        return self

    def __call__(self):
        pass

    def run(self, logits):
        self.inputs.append(logits)


@pytest.mark.parametrize("scenario", ["accepted", "budget", "eos", "inactive", "duplicate", "invalid"])
@pytest.mark.parametrize("native", [False, True])
def test_functional_commit_matches_bound_request_and_completion_state(scenario, native):
    from tests.components.completion_support import completion_outputs

    states = [request_state(), request_state()]
    slots, generations, live = metadata()
    if scenario == "inactive":
        live.fill_(1)
    if scenario == "duplicate":
        slots.fill_(0)
        generations.fill_(1)
    inputs = torch.tensor([[3, 4, 5, 6], [3, 4, 5, 6]])
    acceptance = SpeculativeAcceptance.allocate(2, 4, "cpu")
    acceptance.token_ids.copy_(inputs + 1)
    acceptance.proposed_count.fill_(3)
    acceptance.accepted_drafts.fill_(2)
    acceptance.offered_drafts.fill_(3)
    acceptance.valid.fill_(True)
    acceptance.rng_increment.fill_(8)
    acceptance.survival.fill_(.75)
    if scenario == "invalid":
        acceptance.token_ids[0, 0] = -1
    commits, completions, rngs = [], [], []
    for state in states:
        state.ple_history = torch.tensor([[1, 2], [1, 2]])
        if scenario == "budget":
            state.max_new_tokens.fill_(2)
        if scenario == "eos":
            state.eos_count.fill_(1)
            state.eos_token_ids[:, 0] = 5
        rng = SpeculativeRngState.allocate("cpu", capacity=2)
        rng.generation.copy_(state.generation)
        rng.target_offset.copy_(torch.tensor([11, 23]))
        completion = completion_outputs(torch.zeros((2, 2, 20), dtype=torch.int64)) if native else None
        commit = PreparedSpeculativeStateCommit(state, acceptance, inputs, slots, generations, live,
            completion, torch.tensor([0], dtype=torch.int32) if native else None,
            torch.tensor([13]) if native else None, logprobs=NoLogprobs() if native else None, rng=rng)
        commit.prepare()
        commits.append(commit)
        completions.append(completion)
        rngs.append(rng)
    expected = commits[0]()
    commits[1].functional()
    logits = torch.zeros((2, 4, 64)) if native else None
    actual = commits[1](logits, acceptance=acceptance, inputs=inputs)
    for name in commits[1]._output_names:
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name
        assert getattr(commits[1], name) is None
    for name in ("histogram", "generated_tokens", "pending_token", "status", "error_code", "consumed_tokens", "ple_history", "rng_counter"):
        assert torch.equal(getattr(states[0], name), getattr(states[1], name)), name
    assert torch.equal(rngs[0].target_offset, rngs[1].target_offset)
    assert torch.equal(rngs[0].generation, rngs[1].generation)
    if native:
        assert torch.equal(completions[0].records, completions[1].records)
        assert torch.equal(completions[0].speculative_survival, completions[1].speculative_survival)
        assert commits[1].logprobs.inputs == [logits]
    assert commits[1].acceptance is commits[1].inputs is None
    assert all(getattr(commits[1], name) is None for name in commits[1]._temporary_names)
    saved = actual.emitted_ids.clone()
    repeated = commits[1](logits, acceptance=acceptance, inputs=inputs)
    assert repeated.emitted_ids.data_ptr() != actual.emitted_ids.data_ptr()
    assert torch.equal(actual.emitted_ids, saved)


@pytest.mark.parametrize("release_first", [False, True])
@pytest.mark.parametrize("source_kind", ["materialized", "fused", "shortlist"])
def test_proposal_launches_use_runtime_prefix_rng_and_output_operands(monkeypatch, release_first, source_kind):
    from ignition.components import draft_sampling as draft
    import ignition.preparation.triton as launches

    fused_source = source_kind == "fused"
    state = request_state()
    logits = torch.zeros((2, 1, 64))
    prefix = torch.zeros((2, 3), dtype=torch.int64)
    seed, offset, sampled = (torch.zeros(2, dtype=torch.int64) for _ in range(3))
    operation = draft.PreparedDraftDistribution(logits, state, *metadata(), prefix,
        seed=seed, offset=offset, sampled=sampled, prefix_offset=1)
    tensors = dict(Logits=logits, Histogram=state.histogram, Candidates=prefix, Slots=operation.slots,
        Params=state.parameters, GrammarMask=state.grammar_mask, GrammarEnabled=state.grammar_enabled,
        Scores=operation.scores, BlockMax=operation.block_max, Generations=operation.generations,
        LiveRows=operation.live_rows, Generation=state.generation, Status=state.status, TopK=state.top_k,
        Maximum=operation.maximum, Valid=operation.valid, PackedTopK=operation.packed_top_k,
        PackedTopP=operation.packed_top_p)

    def prepared(kernel, args):
        return PreparedTriton(kernel, None, tuple(args), (2, 1, 1), {})

    def score(kernel):
        return prepared(kernel, [tensors.get(name, 1) for name in kernel.arg_names])

    operation._score_kernels = (score(draft._draft_scores), score(draft._draft_scores_finish))
    if fused_source:
        source = torch.zeros((2, 64))
        operation.bind_logits(source)
        tensors.update(Logits=source, Bias=source, PrefixIds=prefix, Probabilities=operation.probabilities,
                       LocalCandidates=operation.proposal_candidates)
        operation._candidate_kernels = (score(draft._draft_score_candidates), operation._score_kernels[1])
    if source_kind == "shortlist":
        operation.bind_shortlist(torch.zeros((2, 5), dtype=torch.int64))
    proposal = operation.proposal
    operation.proposal._kernels = (
        prepared(draft._block_candidates, proposal._block_args),
        prepared(draft._select_and_sample, (proposal._candidate_source, *proposal._select_args,
            *proposal._candidate_geometry, proposal.selected, proposal._block_args[1])),
    )
    if source_kind == "shortlist":
        operation.proposal._kernels = (prepared(draft._shortlist_candidates, proposal._shortlist_args),
                                       operation.proposal._kernels[-1])
    if fused_source:
        operation.proposal._kernels = operation.proposal._kernels[-1:]
    operation._prepared = True
    if release_first:
        operation.release_scratch()
    operation.functional()
    work = scratch_for(operation)
    runtime_prefix = torch.tensor([[7, 8, 9, 10], [11, 12, 13, 14]])[:, :3]
    runtime_seed = torch.tensor([17, 19])
    runtime_offset = torch.tensor([23, 29])
    runtime_sampled = torch.full((2,), -1, dtype=torch.int64)
    calls = []
    monkeypatch.setattr(launches, "launch_triton", lambda kernel, compiled, args, grid, **options: calls.append((kernel, args)))
    runtime_source = torch.ones((2, 80))[:, :64] if fused_source else None
    runtime_shortlist = torch.arange(16).reshape(2, 8)[:, :5] if source_kind == "shortlist" else None
    operation.run(work, torch.ones_like(logits), source_logits=runtime_source, shortlist_ids=runtime_shortlist, candidates=runtime_prefix,
                  seed=runtime_seed, offset=runtime_offset, sampled=runtime_sampled)
    score_args = dict(zip(calls[0][0].arg_names, calls[0][1]))
    finish_args = dict(zip(calls[1][0].arg_names, calls[1][1]))
    select_args = dict(zip(calls[-1][0].arg_names, calls[-1][1]))
    assert score_args["PrefixIds" if fused_source else "Candidates"] is runtime_prefix
    assert score_args["candidate_stride"] == runtime_prefix.stride(0)
    if fused_source:
        assert score_args["Logits"] is runtime_source and score_args["Bias"] is runtime_source
        assert score_args["logit_stride"] == runtime_source.stride(0)
        assert score_args["bias_stride"] == runtime_source.stride(0)
    assert finish_args["Candidates"] is runtime_prefix and finish_args["Valid"] is work["valid"]
    if source_kind == "shortlist":
        shortlist_args = dict(zip(calls[-2][0].arg_names, calls[-2][1]))
        assert shortlist_args["Ids"] is runtime_shortlist
        assert shortlist_args["id_stride"] == runtime_shortlist.stride(0)
    assert select_args["Seed"] is runtime_seed and select_args["Offset"] is runtime_offset
    assert select_args["Sampled"] is runtime_sampled and select_args["Valid"] is work["valid"]
    assert operation.sampled is operation.valid is operation.candidates is None
    for kernel in (*operation._score_kernels, *operation.proposal._kernels):
        assert all(value is not old for value in kernel.args for old in (prefix, seed, offset, sampled))


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("invalid_target", [False, True])
def test_verification_routes_runtime_validity_rng_and_independent_results(monkeypatch, width, invalid_target):
    import ignition.components.speculation as verification

    state = request_state()
    rng = SpeculativeRngState.allocate("cpu", capacity=2)
    rng.generation.copy_(torch.tensor([1, 0]))
    rng.target_offset.copy_(torch.tensor([17, 29]))
    draft = torch.full((2, width - 1, 64), 1 / 64)
    target = torch.full((2, width, 64), 1 / 64)
    drafts = torch.arange(3, 3 + width - 1).expand(2, -1).contiguous()
    limits = torch.tensor([width, min(width, 2)], dtype=torch.int32)
    draft_valid = torch.ones((2, width - 1), dtype=torch.bool)
    target_valid = torch.ones((2, width), dtype=torch.bool)
    if invalid_target:
        target_valid[0, -1] = False
    operation = PreparedStochasticVerification(draft, torch.full_like(drafts, -1), target, state,
        *metadata(), rng=rng, draft_valid=torch.zeros_like(draft_valid), token_limits=limits)
    operation._prepared = True
    for mask in operation.probability_masks:
        mask.prepared = True

        def apply(source, destination, valid, token_limits, dummy, *, offset=mask.limit_offset):
            columns = torch.arange(source.shape[1])[None, :]
            allowed = valid[:, None] & (columns < token_limits[:, None] - offset)
            destination.copy_(torch.where(allowed[:, :, None], source, dummy))

        monkeypatch.setattr(mask, "run_disjoint", apply)
    calls = []

    def chain(draft_probs, ids, target_probs, independent, emitted, *, seed, offset):
        calls.append((seed.clone(), offset.clone(), ids.clone()))
        emitted.fill_(width - 1)
        return torch.cat((ids.long(), target_probs[:, -1].argmax(-1, keepdim=True)), -1), independent, emitted

    monkeypatch.setattr(verification, "verify_chain", chain)
    operation.release_scratch()
    operation.functional()
    scratch = scratch_for(operation)
    actual = operation.run(scratch, draft, target, target_valid, drafts=drafts, draft_valid=draft_valid)
    assert actual.valid.tolist() == [not invalid_target, True]
    assert actual.proposed_count.tolist() == [0 if invalid_target else width, min(width, 2)]
    assert calls[0][0].tolist() == (state.seed.view(torch.int64) ^ verification.TARGET_RNG_DOMAIN).tolist()
    assert calls[0][1].tolist() == [17, 0]
    assert rng.target_offset.tolist() == [17, 29]
    assert operation.output is operation.drafts is operation.draft_valid is operation.dummy is None
    assert all(mask.source is mask.destination is mask.valid is mask.dummy is mask.snapshot is None
               for mask in operation.probability_masks)
    repeated = operation.run(scratch, draft, target, target_valid, drafts=drafts, draft_valid=draft_valid)
    assert actual.token_ids.data_ptr() != repeated.token_ids.data_ptr()
    assert actual.valid.tolist() == [not invalid_target, True]


def test_functional_commit_traces_with_explicit_inputs_and_local_outputs():
    state = request_state(batch=1)
    acceptance = SpeculativeAcceptance.allocate(1, 3, "cpu")
    acceptance.token_ids.copy_(torch.tensor([[4, 5, 6]]))
    acceptance.proposed_count.fill_(3)
    acceptance.accepted_drafts.fill_(2)
    acceptance.offered_drafts.fill_(2)
    acceptance.valid.fill_(True)
    inputs = torch.tensor([[3, 4, 5]])
    commit = PreparedSpeculativeStateCommit(state, acceptance, inputs, *metadata(1), None, None, None).functional()

    def forward(ids, token_ids, proposed, accepted, offered, valid, increment, survival):
        result = commit(acceptance=SpeculativeAcceptance(token_ids, proposed, accepted, offered, valid, increment, survival), inputs=ids)
        return result.emitted_ids, result.emitted_count, result.commit_column

    compiled = torch.compile(forward, backend="eager", fullgraph=True)
    emitted_ids, emitted_count, column = compiled(inputs, *vars(acceptance).values())
    assert emitted_ids.tolist() == [[4, 5, 6]] and emitted_count.tolist() == [3] and column.tolist() == [2]
    assert state.pending_token.tolist() == [6] and state.consumed_tokens.tolist() == [5]


@pytest.mark.parametrize("tokens", [1, 16])
@pytest.mark.parametrize("context_lag", [None, 0, 1])
def test_idle_completion_record_traces_without_host_tensor_constants(tokens, context_lag):
    from torch.fx.experimental.proxy_tensor import make_fx
    from ignition.components.completion import VECTOR_TOKENS_FIELD
    from ignition.components.speculation import _idle_completion_record

    fields = VECTOR_TOKENS_FIELD + tokens
    def initialize():
        return _idle_completion_record(fields, device="cpu", draft_context_lag=context_lag)

    traced = make_fx(initialize)()
    expected = torch.zeros(fields, dtype=torch.int64)
    expected[4] = expected[12] = -1
    expected[VECTOR_TOKENS_FIELD:] = -1
    if context_lag is None:
        expected[13] = -1
    actual = traced()
    assert actual.dtype == torch.int64 and actual.shape == (fields,)
    assert torch.equal(actual, expected)
    assert all(node.op != "get_attr" for node in traced.graph.nodes)
    assert all(node.target != torch.ops.aten.index_put_.default for node in traced.graph.nodes)
    assert any(node.target == torch.ops.aten.fill_.Scalar for node in traced.graph.nodes)
