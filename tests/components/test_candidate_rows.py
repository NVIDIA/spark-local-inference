"""Metadata and ownership checks for padded candidate preprocessing."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components import candidate_selection, owner_topk
from ignition.components.candidate_rows import PreparedCandidateRows, validate_candidate_rows
from ignition.recipes.kimi31.indexer import SharedBuffers


def inputs(device="cpu"):
    rows, width = 3, 32768
    return [torch.empty((rows, width, 2), dtype=torch.float32, device=device),
            torch.empty(rows, dtype=torch.bool, device=device),
            torch.empty((rows, width), dtype=torch.float32, device=device),
            torch.empty((rows, width), dtype=torch.int32, device=device),
            torch.empty(rows, dtype=torch.int32, device=device),
            torch.empty((rows, 64), dtype=torch.int32, device=device),
            torch.empty((rows, 64), dtype=torch.int32, device=device)]


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_metadata_preserves_independent_storage_and_accepts_adjacent_views(device):
    values = inputs(device)
    arena = torch.empty(3 * 32768 * 2, dtype=torch.int32, device=device)
    values[2] = arena[:3 * 32768].view(torch.float32).view(3, 32768)
    values[3] = arena[3 * 32768:].view(3, 32768)
    validate_candidate_rows(*values, topk=2048)
    values[3] = arena[:3 * 32768].view(3, 32768)
    with pytest.raises(ValueError, match="overlaps"):
        validate_candidate_rows(*values, topk=2048)


@pytest.mark.parametrize("index", range(7))
def test_operand_dtype_and_shape_are_checked(index):
    values = inputs()
    values[index] = values[index].to(torch.float64)
    with pytest.raises(TypeError, match="dtype"):
        validate_candidate_rows(*values, topk=2048)
    values = inputs()
    values[index] = values[index][:2]
    with pytest.raises(ValueError, match="shape"):
        validate_candidate_rows(*values, topk=2048)


@pytest.mark.parametrize("topk", [0, -1, 32769, 1.5, True])
def test_selection_budget_is_a_positive_integer_within_the_row(topk):
    with pytest.raises(ValueError, match="selection budget"):
        validate_candidate_rows(*inputs(), topk=topk)


def test_output_and_metadata_cannot_alias_the_read_inputs():
    values = inputs()
    values[2] = values[0].view(-1)[:3 * 32768].view(3, 32768)
    with pytest.raises(ValueError, match="overlaps"):
        validate_candidate_rows(*values, topk=2048)
    values = inputs()
    values[6] = values[5]
    with pytest.raises(ValueError, match="overlaps"):
        validate_candidate_rows(*values, topk=2048)


def test_serial_shapes_reuse_caller_owned_metadata():
    values = inputs()
    buffers = {}
    first = PreparedCandidateRows(*values[:5], topk=2048,
        buffers=SharedBuffers(buffers, "cpu"))
    second = PreparedCandidateRows(*(value[:1] for value in values[:5]), topk=2048,
        buffers=SharedBuffers(buffers, "cpu"))
    assert second.counts.untyped_storage()._cdata == first.counts.untyped_storage()._cdata
    assert second.exceptional.untyped_storage()._cdata == first.exceptional.untyped_storage()._cdata
    assert first.counts.numel() * first.counts.element_size() * 2 == 1536


def _owner_topk(indices, scores, output, counts):
    rows = indices.shape[0]
    runtime = SimpleNamespace(world_size=2, rank=0)
    return owner_topk.PreparedOwnerTopK(indices, scores, torch.ones(rows, dtype=torch.bool),
        torch.tensor([0]), output, owner_runtime=runtime, reduction_runtime=runtime, session=None,
        coordinator=None, name="candidate-scratch", buffers=SharedBuffers({}, "cpu"), local_counts=counts)


@pytest.mark.parametrize("operand", ["indices", "scores", "output"])
def test_candidate_count_scratch_cannot_overlap_owner_selection_operands(operand):
    indices = torch.zeros((3, 4), dtype=torch.int32)
    scores = torch.zeros_like(indices, dtype=torch.float32)
    output = torch.empty_like(indices)
    counts = {"indices": indices, "scores": scores, "output": output}[operand].view(torch.int32)[0, :3]
    with pytest.raises(ValueError, match="count scratch must not overlap"):
        _owner_topk(indices, scores, output, counts)


def test_candidate_count_scratch_can_share_storage_with_disjoint_operands(monkeypatch):
    def transport(*args, local_counts, **kwargs):
        return SimpleNamespace(local_counts=local_counts)

    monkeypatch.setattr(owner_topk, "PreparedCandidateOwnerStage", transport)
    monkeypatch.setattr(owner_topk, "PreparedOrderedRowTopK", lambda *args, **kwargs: object())
    monkeypatch.setattr(owner_topk, "PreparedOwnedRowBroadcast", lambda *args, **kwargs: object())
    storage = torch.empty(27, dtype=torch.int32)
    indices = storage[:12].view(3, 4)
    scores = storage[12:24].view(torch.float32).view(3, 4)
    counts = storage[24:]
    operation = _owner_topk(indices, scores, torch.empty_like(indices), counts)
    assert operation.transport.local_counts is counts


@pytest.fixture
def prepare_candidate_selection(monkeypatch):
    def owner(*args, local_counts, **kwargs):
        return SimpleNamespace(transport=SimpleNamespace(local_counts=local_counts))

    monkeypatch.setattr(candidate_selection, "PreparedOwnerTopK", owner)

    def prepare(indices, scores, positions, rank, output, counts):
        runtime = SimpleNamespace(world_size=2, rank=0)
        return candidate_selection.PreparedCandidateSelection(indices, scores, positions, rank, output,
            world=2, gather_runtime=None, owner_runtime=runtime, reduction_runtime=runtime, session=None,
            coordinator=None, name="candidate-selection", buffers=SharedBuffers({}, "cpu"),
            alternatives=("owner",), local_counts=counts)

    return prepare


@pytest.mark.parametrize("operand", ["indices", "scores", "positions", "rank", "output"])
def test_candidate_count_scratch_cannot_overlap_original_selection_operands(prepare_candidate_selection, operand):
    operands = dict(indices=torch.zeros((2, 4), dtype=torch.int32), scores=torch.zeros((2, 4)),
                    positions=torch.zeros(2, dtype=torch.int64), rank=torch.zeros(1, dtype=torch.int64),
                    output=torch.empty((2, 4), dtype=torch.int32))
    counts = operands[operand].view(torch.int32).reshape(-1)[:2]
    with pytest.raises(ValueError, match="count scratch must not overlap"):
        prepare_candidate_selection(**operands, counts=counts)


def test_candidate_count_scratch_can_share_storage_with_original_selection_operands(prepare_candidate_selection):
    storage = torch.arange(18, dtype=torch.int32)
    indices = storage[:8].view(2, 4)
    scores = storage[8:16].view(torch.float32).view(2, 4).fill_(0.5)
    counts = storage[16:]
    operation = prepare_candidate_selection(indices, scores, torch.zeros(2, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int64), torch.empty_like(indices), counts)

    operation.pack("owner")

    assert operation.owner.transport.local_counts is counts
    assert torch.equal(operation.owner_ids, indices * 2)
    assert torch.equal(operation.owner_scores, scores)
    assert counts.tolist() == [16, 17]


@pytest.mark.parametrize("splits", [1, 2, 8, 32])
def test_probability_reference_preserves_uniform_attention_across_split_boundaries(splits):
    from tests.components.test_candidate_rows_cuda import probability_encoding_reference

    query = torch.zeros((2, 576))
    records = ((torch.arange(130 * 576).reshape(130, 576) % 7 - 3).float() * .125).to(torch.float8_e4m3fn).float()
    actual = probability_encoding_reference(query, records, splits=splits)
    expected = records[:, :512].double().mean(0).float().expand(2, -1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    empty = probability_encoding_reference(query, records[:0], splits=splits)
    assert not empty.count_nonzero()
