"""Captured candidate preprocessing preserves sets and unchanged exceptional inputs."""

import json
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


class CandidateSelection:
    def __init__(self, rows, width, topk, device, session):
        from ignition.components.candidate_rows import PreparedCandidateRows
        from ignition.components.row_topk import PreparedRowTopK
        from ignition.recipes.kimi31.indexer import SharedBuffers

        self.packed = torch.empty((rows, width, 2), dtype=torch.float32, device=device)
        self.active = torch.empty(rows, dtype=torch.bool, device=device)
        self.values = torch.empty((rows, width), dtype=torch.float32, device=device)
        self.indices = torch.empty((rows, width), dtype=torch.int32, device=device)
        self.lengths = torch.empty(rows, dtype=torch.int32, device=device)
        self.preprocess = PreparedCandidateRows(self.packed, self.active, self.values, self.indices,
            self.lengths, topk=topk, buffers=SharedBuffers({}, device))
        self.control_values = torch.empty_like(self.values)
        self.control_indices = torch.empty_like(self.indices)
        self.control_lengths = torch.empty_like(self.lengths)
        self.selected_values = [torch.empty((rows, topk), dtype=torch.float32, device=device) for _ in range(2)]
        self.selected_indices = [torch.empty((rows, topk), dtype=torch.int32, device=device) for _ in range(2)]
        self.folds = [PreparedRowTopK(v, n, i, self.selected_values[index], self.selected_indices[index],
            session=session, name=f"fold{index}") for index, (v, n, i) in enumerate((
                (self.control_values, self.control_lengths, self.control_indices),
                (self.values, self.lengths, self.indices)))]
        self.width = width

    def __call__(self):
        self.control_values.copy_(self.packed[:, :, 0])
        self.control_indices.copy_(self.packed.view(torch.int32)[:, :, 1])
        self.control_lengths.copy_(torch.where(self.active, self.width, 0))
        self.folds[0]()
        self.preprocess()
        self.folds[1]()
        return self.selected_indices[1]


def candidate_input(width, seed, counts=None):
    generator = torch.Generator().manual_seed(seed)
    exceptional = counts is None
    counts = counts or (0, 1, 17, 2047, 2048, 2049, 32768, 17, 17, 17, 17, 17, 17, 17)
    packed = torch.full((len(counts), width, 2), -torch.inf, dtype=torch.float32)
    ids = packed.view(torch.int32)[:, :, 1]
    ids.fill_(-1)
    active = torch.ones(len(counts), dtype=torch.bool)
    active[-1] = False
    for row, count in enumerate(counts):
        if seed % 2:
            positions = torch.randperm(width, generator=generator)[:count]
        else:
            positions = torch.tensor([(token % 16) * 2048 + token // 16 for token in range(count)], dtype=torch.long)
        values = torch.randperm(max(count, 1), generator=generator)[:count].float() / max(count, 1)
        packed[row, positions, 0] = values
        ids[row, positions] = torch.arange(count, dtype=torch.int32)
    if not exceptional:
        return packed, active
    live = [torch.where(ids[row] >= 0)[0] for row in range(len(counts))]
    if seed % 2:
        packed[5, live[5], 0] = .5
    packed[7, live[7][0], 0] = float("nan")
    packed.view(torch.int32)[7, live[7][0], 0] = 0x7FC12345 if seed % 2 == 0 else 0xFFC12345 - 2**32
    packed[8, live[8][0], 0] = -float("inf")
    packed[8, live[8][1], 0] = -0.0
    packed[8, live[8][2], 0] = 0.0
    packed[9, live[9][0], 0] = float("inf")
    packed[10, live[10][0], 0] = -0.0
    packed[10, live[10][1], 0] = 0.0
    padding = torch.where(ids[11] < 0)[0][0]
    packed[11, padding, 0] = 123.
    ids[12, torch.where(ids[12] < 0)[0][0]] = -2
    return packed, active


def check_preprocessing(operation, packed, active, topk):
    actual_values = operation.values.cpu()
    actual_ids = operation.indices.cpu()
    lengths = operation.lengths.cpu()
    source_ids = packed.view(torch.int32)[:, :, 1]
    for row in range(len(active)):
        values, ids = packed[row, :, 0], source_ids[row]
        live = ids >= 0
        ordinary_padding = ((ids == -1) & torch.isneginf(values)) | live
        compact = int(live.sum()) <= topk and bool(torch.isfinite(values[live]).all()) and bool(ordinary_padding.all())
        if not active[row]:
            assert lengths[row] == 0 and (actual_ids[row] == -1).all()
            assert torch.isneginf(actual_values[row]).all()
        elif compact:
            count = int(live.sum())
            assert lengths[row] == count
            assert torch.equal(actual_values[row, :count].view(torch.int32), values[live].contiguous().view(torch.int32))
            assert torch.equal(actual_ids[row, :count], ids[live])
            assert torch.isneginf(actual_values[row, count:]).all()
            assert (actual_ids[row, count:] == -1).all()
            for selected in operation.selected_indices:
                observed = selected[row].cpu()
                assert torch.equal(observed[observed >= 0].sort().values, ids[live].sort().values)
        else:
            assert lengths[row] == operation.width
            assert torch.equal(actual_values[row].view(torch.int32), values.contiguous().view(torch.int32))
            assert torch.equal(actual_ids[row], ids)
        if row in (5, 6):
            if values[live].unique().numel() == int(live.sum()):
                control = operation.selected_indices[0][row].cpu().sort().values
                candidate = operation.selected_indices[1][row].cpu().sort().values
                assert torch.equal(control, candidate)
            else:
                for selected, selected_values in zip(operation.selected_indices, operation.selected_values, strict=True):
                    observed = selected[row].cpu()
                    assert observed.unique().numel() == topk
                    assert (observed >= 0).all() and (observed < int(live.sum())).all()
                    assert (selected_values[row] == .5).all()


@pytest.mark.parametrize("compiled", [False, True])
def test_candidate_compaction_tracks_live_rows_and_preserves_exceptional_layout(compiled):
    from ignition import runtime_patches
    runtime_patches.apply()
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda", 0)
    with RecordedPreparationSession(device=device, checkpoint="candidate-row-preprocessing",
            workload_schema="underfull-compaction", numerical_policy="unchanged-fp32-pairs",
            autotune_effort="off") as session:
        packed, active = candidate_input(32768, 20)
        operation = CandidateSelection(len(active), 32768, 2048, device, session)
        operation.packed.copy_(packed)
        operation.active.copy_(active)
        session.tune(show_progress=False)
        forward = compile_forward(operation) if compiled else operation
        forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            forward()
        session.freeze()
        try:
            for seed in (20, 21, 22, 23):
                packed, active = candidate_input(32768, seed)
                operation.packed.copy_(packed)
                operation.active.copy_(active)
                operation.values.fill_(123.)
                operation.indices.fill_(-7)
                operation.lengths.fill_(-8)
                operation.preprocess.counts.fill_(-9)
                operation.preprocess.exceptional.fill_(-9)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for field in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                    assert before[field] == after[field], field
                check_preprocessing(operation, packed, active, 2048)
        finally:
            graph.reset()


class CandidateAttention:
    def __init__(self, selection, query_dtype, device, session, arena):
        from b12x.attention import dense_mla
        from ignition.components.dense_mla import PreparedDenseMLA
        from ignition.components.peer_record_gather import PreparedPeerRecordGather

        self.selection = selection
        rows, topk = selection.selected_indices[0].shape
        heads = 96 if query_dtype == torch.float8_e4m3fn else 6
        self.query = torch.zeros((rows, heads, 576), dtype=query_dtype, device=device)
        self.cache = torch.zeros((4096, 576), dtype=torch.float8_e4m3fn, device=device)
        peers = torch.tensor([self.cache.data_ptr()], dtype=torch.int64, device=device)
        starts = torch.arange(rows + 1, dtype=torch.int32, device=device)
        pages = torch.arange(rows * topk, dtype=torch.int32, device=device).view(rows, topk)
        self.parts = []
        for index, selected in enumerate(selection.selected_indices):
            owners = torch.empty_like(selected)
            counts = torch.empty(rows, dtype=torch.int32, device=device)
            records = torch.empty((rows, topk, 576), dtype=self.cache.dtype, device=device)
            output = torch.empty((rows, heads, 512), dtype=torch.bfloat16, device=device)
            gather = PreparedPeerRecordGather(peers, owners, selected, counts, records,
                peer_records=4096, session=session, name=f"attention-records{index}")
            caps = dense_mla.Caps(device=device, mode="decode", q_dtype=query_dtype,
                kv_dtype=self.cache.dtype, page_size=1, num_q_heads=heads, max_total_q=rows,
                max_batch=rows, max_cache_tokens=topk, max_page_table_width=topk,
                num_cache_pages=rows * topk, partial_dtype=torch.float32, use_cuda_graph=True)
            attention = PreparedDenseMLA(self.query, records.view(-1, 1, 576), output, pages,
                counts, starts, caps=caps, scale=192 ** -.5, session=session,
                name=f"attention{index}", workspace=arena)
            self.parts.append((owners, counts, gather, attention, output, records))

    def __call__(self):
        self.selection()
        for selected, (owners, counts, gather, attention, _, _) in zip(
                self.selection.selected_indices, self.parts, strict=True):
            owners.copy_(torch.where(selected >= 0, 0, -1))
            counts.copy_((selected >= 0).sum(-1).to(torch.int32))
            gather()
            attention()
        return self.parts[1][4]


def probability_encoding_reference(query, records, *, splits):
    """Reference the documented E4M3 probability encoding and split composition.

    Each 64-record window uses two E4M3 parts with a per-head maximum scale.
    The denominator retains the unencoded probabilities. Online rescaling and
    split LSE merging follow dense_mla's traversal, while dot products use FP64
    instead of reproducing a device's FP32 MMA accumulation order.
    """
    count, heads = records.shape[0], query.shape[0]
    if not count:
        return torch.zeros((heads, 512), dtype=torch.float32, device=query.device)
    chunks = (count + 63) // 64
    chunks_per_split = max(1, (chunks + splits - 1) // splits)
    partials, lses = [], []
    for split in range(splits):
        begin = split * chunks_per_split * 64
        end = min(count, begin + chunks_per_split * 64)
        if begin >= end:
            continue
        maximum = torch.full((heads,), -torch.inf, dtype=torch.float32, device=query.device)
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros((heads, 512), dtype=torch.float32, device=query.device)
        for offset in range(begin, end, 64):
            values = records[offset:min(offset + 64, end)]
            dot = (query.double() @ values.double().T).float()
            logits = dot * (192 ** -.5 * 1.4426950408889634)
            following = torch.maximum(maximum, logits.amax(-1))
            alpha = (maximum - following).exp2()
            probability = (logits - following[:, None]).exp2()
            scale = probability.amax(-1).clamp_min(1e-10) * (1. / 448)
            normalized = probability * scale.reciprocal()[:, None]
            high = normalized.clamp(-448, 448).to(torch.float8_e4m3fn).float()
            low = (normalized - high).clamp(-448, 448).to(torch.float8_e4m3fn).float()
            high_dot = (high.double() @ values[:, :512].double()).float()
            low_dot = (low.double() @ values[:, :512].double()).float()
            numerator = numerator * alpha[:, None] + (high_dot + low_dot) * scale[:, None]
            denominator = denominator * alpha + probability.sum(-1)
            maximum = following
        partials.append(numerator / denominator[:, None])
        lses.append(maximum + denominator.log2())
    lse = torch.stack(lses)
    weights = (lse - lse.amax(0)).exp2()
    return (torch.stack(partials) * weights[:, :, None]).sum(0) / weights.sum(0)[:, None]


@pytest.mark.parametrize("query_dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=("bf16-input-prefill", "fp8-input-decode"))
def test_compacted_order_attention_matches_the_existing_numerical_contract(query_dtype, tmp_path):
    """Check the existing query/probability encoding, reporting ideal softmax separately.

    Reordering can change E4M3 probability groups and reductions. The test does
    not require bitwise attention equality or substitute a wider tolerance for
    the backend's probability arithmetic.
    """
    from ignition import runtime_patches
    runtime_patches.apply()
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda", 0)
    with RecordedPreparationSession(device=device, checkpoint="candidate-row-attention",
            workload_schema="underfull-ordering", numerical_policy="existing-selected-mla-bounds",
            autotune_effort="off") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        counts = (17, 1021, 2048, 2049, 0)
        selection = CandidateSelection(len(counts), 32768, 2048, device, session)
        operation = CandidateAttention(selection, query_dtype, device, session, arena)
        packed, active = candidate_input(32768, 20, counts)
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
                packed, active = candidate_input(32768, seed, counts)
                selection.packed.copy_(packed)
                selection.active.copy_(active)
                generator = torch.Generator().manual_seed(seed)
                operation.query.copy_(torch.randn(operation.query.shape, generator=generator) * input_scale)
                operation.cache.copy_(torch.randn(operation.cache.shape, generator=generator) * input_scale)
                baseline, candidate, orderings = [], [], []
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
                    for field in ("allocation.all.allocated", "allocation.all.freed",
                                  "segment.all.allocated", "segment.all.freed"):
                        assert before[field] == after[field], field
                    baseline.append(operation.parts[0][4].float().cpu())
                    candidate.append(operation.parts[1][4].float().cpu())
                    orderings.append([selected.cpu() for selected in selection.selected_indices])
                q, cache = operation.query.double(), operation.cache.double()
                # dense_mla documents E4M3 query conversion for BF16 input with
                # an E4M3 cache. Both plans here bind unit query/cache scales.
                native_q = operation.query.float().to(torch.float8_e4m3fn).double()
                reference = torch.zeros_like(operation.parts[0][4], dtype=torch.float64)
                ideal = torch.zeros_like(reference)
                for row in range(len(counts) - 1):
                    ids = selection.selected_indices[0][row]
                    other = selection.selected_indices[1][row]
                    ids, other = ids[ids >= 0].long(), other[other >= 0].long()
                    assert torch.equal(ids.sort().values, other.sort().values)
                    ordered = ids.sort().values
                    logits = native_q[row] @ cache[ordered].T * 192 ** -.5
                    reference[row] = logits.softmax(-1) @ cache[ordered, :512]
                    ideal_logits = q[row] @ cache[ordered].T * 192 ** -.5
                    ideal[row] = ideal_logits.softmax(-1) @ cache[ordered, :512]
                expected = reference.float().cpu()
                base = torch.stack(baseline)
                changed = torch.stack(candidate)
                errors, ideal_errors = {}, {}
                encoded_references = {}
                reference_cache = {}
                split_count = operation.parts[0][3].plan.prepared.state.layout.num_splits
                assert split_count == operation.parts[1][3].plan.prepared.state.layout.num_splits
                for label, outputs in (("baseline", baseline), ("candidate", candidate)):
                    path = 0 if label == "baseline" else 1
                    references = []
                    for repeat in range(len(outputs)):
                        rows = []
                        for row, ids in enumerate(orderings[repeat][path]):
                            ids = ids[ids >= 0].long()
                            key = (row, tuple(ids.tolist()))
                            if key not in reference_cache:
                                reference_cache[key] = probability_encoding_reference(native_q[row],
                                    cache[ids.to(device)], splits=split_count).cpu()
                            rows.append(reference_cache[key])
                        references.append(torch.stack(rows))
                    encoded_references[label] = torch.stack(references)
                    errors[label] = max(int(((output - contract).abs() > .002 + .03 * contract.abs()).sum())
                                        for output, contract in zip(outputs, references, strict=True))
                    ideal_errors[label] = max(int(((output - expected).abs() > .002 + .03 * expected.abs()).sum())
                                             for output in outputs)
                    for output, contract in zip(outputs, references, strict=True):
                        assert not output[-1].count_nonzero()
                        try:
                            # These are the existing Kimi selected-MLA oracle bounds.
                            torch.testing.assert_close(output, contract, rtol=.03, atol=.002)
                        except AssertionError:
                            failures.append(dict(path=label, seed=seed, input_scale=input_scale,
                                                 mismatches=errors[label]))
                report = dict(query_dtype=str(query_dtype), seed=seed, input_scale=input_scale,
                    oracle_mismatches=errors,
                    ideal_softmax_bound_mismatches=ideal_errors,
                    baseline_repeat_max_abs=float((base - base[0]).abs().max()),
                    candidate_vs_baseline_max_abs=float((changed - base[0]).abs().max()),
                    baseline_oracle_max_abs=float((base - expected).abs().max()),
                    candidate_oracle_max_abs=float((changed - expected).abs().max()),
                    baseline_oracle_rms=float((base - expected).square().mean().sqrt()),
                    candidate_oracle_rms=float((changed - expected).square().mean().sqrt()),
                    baseline_ideal_query_rms=float((base - ideal.float().cpu()).square().mean().sqrt()),
                    candidate_ideal_query_rms=float((changed - ideal.float().cpu()).square().mean().sqrt()),
                    baseline_probability_oracle_max_abs=float((base - encoded_references['baseline']).abs().max()),
                    candidate_probability_oracle_max_abs=float((changed - encoded_references['candidate']).abs().max()),
                    active_splits=split_count,
                    selected_plan=repr(operation.parts[0][3].plan.selection.config))
                print(json.dumps(report), flush=True)
                assert report['candidate_oracle_max_abs'] <= (report['baseline_oracle_max_abs']
                                                              + report['baseline_repeat_max_abs'])
                if any(errors.values()):
                    fixture = tmp_path / f"attention-seed{seed}-scale{input_scale}.pt"
                    torch.save(dict(packed=packed, active=active, query=operation.query.cpu(),
                                    cache=operation.cache.cpu(), baseline=base, candidate=changed,
                                    reference=expected, ideal=ideal.cpu(), report=report,
                                    probability_reference=encoded_references, selected_orders=orderings,
                                    active_splits=split_count), fixture)
                    print(f"Failing numerical fixture: {fixture}", flush=True)
            assert not failures, failures
        finally:
            graph.reset()
