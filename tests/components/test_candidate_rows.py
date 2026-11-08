"""Metadata and ownership checks for padded candidate preprocessing."""

import pytest
import torch

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
