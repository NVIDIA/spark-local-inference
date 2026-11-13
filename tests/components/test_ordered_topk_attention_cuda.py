"""Selected-order MLA checks for deterministic row top-k, including cutoff ties."""

import json
import os

import pytest
import torch

from .test_candidate_rows_cuda import CandidateAttention, candidate_input, probability_encoding_reference
from .test_ordered_row_topk import ordered_reference


pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1',
                               reason='requires an explicitly assigned CUDA device')


class OrderedAttentionSelection:
    """Compare ordinary and ordered top-k after identical candidate preprocessing."""

    def __init__(self, rows, width, topk, device, session):
        from ignition.components.candidate_rows import PreparedCandidateRows
        from ignition.components.ordered_row_topk import PreparedOrderedRowTopK
        from ignition.components.row_topk import PreparedRowTopK
        from ignition.recipes.kimi31.indexer import SharedBuffers

        self.packed = torch.empty((rows, width, 2), dtype=torch.float32, device=device)
        self.active = torch.empty(rows, dtype=torch.bool, device=device)
        self.values = torch.empty((rows, width), dtype=torch.float32, device=device)
        self.indices = torch.empty((rows, width), dtype=torch.int32, device=device)
        self.lengths = torch.empty(rows, dtype=torch.int32, device=device)
        self.preprocess = PreparedCandidateRows(self.packed, self.active, self.values, self.indices,
            self.lengths, topk=topk, buffers=SharedBuffers({}, device))
        self.selected_values = [torch.empty((rows, topk), dtype=torch.float32, device=device) for _ in range(2)]
        self.selected_indices = [torch.empty((rows, topk), dtype=torch.int32, device=device) for _ in range(2)]
        self.control = PreparedRowTopK(self.values, self.lengths, self.indices,
            self.selected_values[0], self.selected_indices[0], session=session, name='control-fold')
        self.candidate = PreparedOrderedRowTopK(self.values, self.lengths, self.indices,
            self.selected_values[1], self.selected_indices[1], session=session, name='ordered-fold',
            buffers=SharedBuffers({}, device))

    def __call__(self):
        self.preprocess()
        self.control()
        self.candidate()
        return self.selected_indices[1]


def attention_candidates(seed):
    """Finite candidates with changing gathered order and overfull cutoff ties."""
    counts = (17, 1021, 2048, 2049, 4096, 3072, 0, 17)
    packed, active = candidate_input(32768, seed, counts)
    ids = packed.view(torch.int32)[:, :, 1]
    tied = ids[4] >= 0
    packed[4, tied, 0] = (ids[4, tied] // 300).float()
    packed[5, ids[5] >= 0, 0] = .5
    return packed, active, counts


@pytest.mark.parametrize('query_dtype', [torch.bfloat16, torch.float8_e4m3fn],
                         ids=('bf16-input-prefill', 'fp8-input-decode'))
@torch.no_grad()
def test_ordered_topk_attention_preserves_existing_numerical_contract(query_dtype, tmp_path):
    """Use each selected order and membership in the existing probability oracle.

    Equal-cutoff candidates can produce different valid memberships in the
    ordinary selector. Each path/replay therefore also has its own exact-softmax
    diagnostic reference. Oracle bounds and the error/repeatability comparison
    are unchanged from the candidate-compaction attention test.
    """
    from ignition import runtime_patches
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    runtime_patches.apply()
    torch.set_num_threads(2)
    device = torch.device('cuda', 0)
    packed, active, counts = attention_candidates(122)
    with RecordedPreparationSession(device=device, checkpoint='ordered-topk-attention',
            workload_schema='canonical-order-selected-mla', numerical_policy='existing-selected-mla-bounds',
            compile_workers=16) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        selection = OrderedAttentionSelection(len(counts), 32768, 2048, device, session)
        operation = CandidateAttention(selection, query_dtype, device, session, arena)
        selection.packed.copy_(packed)
        selection.active.copy_(active)
        session.tune(show_progress=False)
        forward = compile_forward(operation)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        session.freeze()
        failures = []
        try:
            for seed, input_scale in ((seed, scale) for seed in (122, 123, 124) for scale in (.2, 1., 3.)):
                packed, active, counts = attention_candidates(seed)
                selection.packed.copy_(packed)
                selection.active.copy_(active)
                generator = torch.Generator().manual_seed(seed)
                operation.query.copy_(torch.randn(operation.query.shape, generator=generator) * input_scale)
                operation.cache.copy_(torch.randn(operation.cache.shape, generator=generator) * input_scale)
                observed = [[], []]
                orderings = []
                for _ in range(4):
                    for _, _, _, _, output, records in operation.parts:
                        output.fill_(torch.nan)
                        records.fill_(torch.nan)
                    arena.tensor.fill_(255)
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()
                    graph.replay()
                    torch.cuda.synchronize()
                    after = torch.cuda.memory_stats()
                    for field in ('allocation.all.allocated', 'allocation.all.freed',
                                  'segment.all.allocated', 'segment.all.freed'):
                        assert before[field] == after[field], field
                    for path in range(2):
                        observed[path].append(operation.parts[path][4].float().cpu())
                    orderings.append([selected.cpu() for selected in selection.selected_indices])

                expected_values, expected_ids, _ = ordered_reference(selection.values.cpu(),
                    selection.lengths.cpu(), selection.indices.cpu(), 2048)
                assert torch.equal(selection.selected_values[1].view(torch.int32).cpu(), expected_values)
                for selected in orderings:
                    assert torch.equal(selected[1], expected_ids)
                    for row in (0, 1, 2, 3, 6, 7):
                        assert torch.equal(selected[0][row].sort().values, selected[1][row].sort().values)
                    for row in (4, 5):
                        for path in range(2):
                            valid = selected[path][row]
                            assert valid.unique().numel() == 2048
                            assert (valid >= 0).all() and (valid < counts[row]).all()
                for row in (4, 5):
                    assert torch.equal(selection.selected_values[0][row].sort().values,
                                       selection.selected_values[1][row].sort().values)

                query, cache = operation.query.double(), operation.cache.double()
                native_query = operation.query.float().to(torch.float8_e4m3fn).double()
                splits = operation.parts[0][3].plan.prepared.state.layout.num_splits
                assert splits == operation.parts[1][3].plan.prepared.state.layout.num_splits
                encoded, exact, ideal = [[], []], [[], []], [[], []]
                reference_cache = {}
                for path in range(2):
                    for selected in orderings:
                        encoded_rows, exact_rows, ideal_rows = [], [], []
                        for row, ids in enumerate(selected[path]):
                            ids = ids[ids >= 0].long()
                            key = (row, tuple(ids.tolist()))
                            if key not in reference_cache:
                                ordered_records = cache[ids.to(device)]
                                probability = probability_encoding_reference(native_query[row], ordered_records, splits=splits)
                                exact_row = torch.zeros((query.shape[1], 512), dtype=torch.float64, device=device)
                                ideal_row = torch.zeros_like(exact_row)
                                if ids.numel():
                                    records = cache[ids.sort().values.to(device)]
                                    logits = native_query[row] @ records.T * 192 ** -.5
                                    exact_row = logits.softmax(-1) @ records[:, :512]
                                    ideal_logits = query[row] @ records.T * 192 ** -.5
                                    ideal_row = ideal_logits.softmax(-1) @ records[:, :512]
                                reference_cache[key] = (probability.cpu(), exact_row.float().cpu(), ideal_row.float().cpu())
                            probability, exact_row, ideal_row = reference_cache[key]
                            encoded_rows.append(probability)
                            exact_rows.append(exact_row)
                            ideal_rows.append(ideal_row)
                        encoded[path].append(torch.stack(encoded_rows))
                        exact[path].append(torch.stack(exact_rows))
                        ideal[path].append(torch.stack(ideal_rows))
                output = [torch.stack(values) for values in observed]
                encoded = [torch.stack(values) for values in encoded]
                exact = [torch.stack(values) for values in exact]
                ideal = [torch.stack(values) for values in ideal]
                mismatches = {}
                condition_failures = []
                for path, label in enumerate(('baseline', 'candidate')):
                    mismatches[label] = int(((output[path] - encoded[path]).abs() > .002 + .03 * encoded[path].abs()).sum())
                    for value, contract in zip(observed[path], encoded[path], strict=True):
                        assert not value[-2:].count_nonzero()
                        try:
                            torch.testing.assert_close(value, contract, rtol=.03, atol=.002)
                        except AssertionError:
                            condition_failures.append(dict(path=label, seed=seed, input_scale=input_scale,
                                                           mismatches=mismatches[label]))
                # Membership changes alter the exact answer. Only variation in
                # numerical error contributes to the repeatability allowance.
                baseline_residual = output[0] - exact[0]
                baseline_repeat = float((baseline_residual - baseline_residual[0]).abs().max())
                baseline_error = float((output[0] - exact[0]).abs().max())
                candidate_error = float((output[1] - exact[1]).abs().max())
                repeatability_passed = candidate_error <= baseline_error + baseline_repeat
                report = dict(query_dtype=str(query_dtype), seed=seed, input_scale=input_scale,
                    oracle_mismatches=mismatches, baseline_repeat_max_abs=baseline_repeat,
                    baseline_oracle_max_abs=baseline_error, candidate_oracle_max_abs=candidate_error,
                    baseline_probability_oracle_max_abs=float((output[0] - encoded[0]).abs().max()),
                    candidate_probability_oracle_max_abs=float((output[1] - encoded[1]).abs().max()),
                    candidate_vs_baseline_max_abs=float((output[1] - output[0][0]).abs().max()),
                    baseline_ideal_query_rms=float((output[0] - ideal[0]).square().mean().sqrt()),
                    candidate_ideal_query_rms=float((output[1] - ideal[1]).square().mean().sqrt()),
                    tied_membership_differences=sum(not torch.equal(selected[0][row].sort().values,
                        selected[1][row].sort().values) for selected in orderings for row in (4, 5)),
                    active_splits=splits, selected_plan=repr(operation.parts[0][3].plan.selection.config),
                    repeatability_passed=repeatability_passed)
                print(json.dumps(report), flush=True)
                if not repeatability_passed:
                    condition_failures.append(dict(path='candidate', seed=seed, input_scale=input_scale,
                                                   bound='control error plus control repeatability'))
                if condition_failures:
                    fixture = tmp_path / f'ordered-attention-seed{seed}-scale{input_scale}.pt'
                    torch.save(dict(packed=packed, active=active, query=operation.query.cpu(), cache=operation.cache.cpu(),
                        baseline=output[0], candidate=output[1], exact_reference=exact, ideal_reference=ideal,
                        probability_reference=encoded, selected_orders=orderings, report=report), fixture)
                    print(f'Failing numerical fixture: {fixture}', flush=True)
                    failures.extend(condition_failures)
            assert not failures, failures
        finally:
            graph.reset()
