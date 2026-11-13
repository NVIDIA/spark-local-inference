"""Integer-bit selection oracles and exceptional inputs for ordered row top-k."""

import random

import pytest
import torch


NEGATIVE_INFINITY_BITS = 0xFF800000
SPECIAL_SCORE_BITS = (
    0xFFFFFFFF, 0xFFC12345, 0xFF800001, 0xFF800000,
    0xFF7FFFFF, 0xBF800000, 0x80000001, 0x80000000,
    0x00000000, 0x00000001, 0x3F800000, 0x7F7FFFFF,
    0x7F800000, 0x7F800001, 0x7FC12345, 0x7FFFFFFF,
)


def signed32(bits):
    """Represent an unsigned word in an int32 tensor without a float conversion."""
    bits &= 0xFFFFFFFF
    return bits if bits < 0x80000000 else bits - 0x100000000


def ordered_key(bits):
    """The selector's total key on an FP32 word, including sign and NaN payload."""
    bits &= 0xFFFFFFFF
    return (~bits & 0xFFFFFFFF) if bits & 0x80000000 else bits | 0x80000000


def ordered_row_reference(score_bits, ids, length, topk):
    """Rank input positions by integer key, resolve ties by position, then emit input order.

    IDs are mapped only after selection. A negative ID still occupies a selected
    position and becomes -1; its original score word remains paired with it.
    """
    if type(length) is not int or not 0 <= length <= len(score_bits) or len(ids) != len(score_bits):
        raise ValueError('reference length must describe a prefix of matching score/ID rows')
    if type(topk) is not int or topk <= 0:
        raise ValueError('reference selection budget must be a positive integer')
    ranked = sorted(range(length), key=lambda index: (-ordered_key(score_bits[index]), index))
    positions = sorted(ranked[:topk])
    padding = topk - len(positions)
    values = [signed32(score_bits[index]) for index in positions] + [signed32(NEGATIVE_INFINITY_BITS)] * padding
    mapped = [max(int(ids[index]), -1) for index in positions] + [-1] * padding
    return values, mapped, positions + [-1] * padding


def ordered_reference(scores, lengths, ids, topk):
    """Return CPU int32 score words, IDs and input positions for a tensor fixture."""
    if any(value.device.type != 'cpu' for value in (scores, lengths, ids)):
        raise ValueError('copy diagnostic inputs to CPU before invoking the integer oracle')
    if scores.dtype != torch.float32 or lengths.dtype != torch.int32 or ids.dtype != torch.int32:
        raise TypeError('oracle inputs require FP32 scores and int32 lengths/IDs')
    if scores.ndim != 2 or ids.shape != scores.shape or lengths.shape != (scores.shape[0],):
        raise ValueError('oracle inputs require matching rows and one length per row')
    rows = [ordered_row_reference(bits, mapped, int(length), topk) for bits, mapped, length in zip(
        scores.contiguous().view(torch.int32).tolist(), ids.tolist(), lengths.tolist(), strict=True)]
    return tuple(torch.tensor([row[field] for row in rows], dtype=torch.int32) for field in range(3))


def adversarial_rows(width=32768, topk=2048, seed=0):
    """Build mixed empty, ordinary, tied and exceptional rows without normalizing score bits."""
    if type(width) is not int or type(topk) is not int or not 0 < topk < width:
        raise ValueError('adversarial fixtures require positive topk below width')
    rng = random.Random(seed)
    permutation = list(range(width))
    rng.shuffle(permutation)
    rows, mapped, lengths, labels = [], [], [], []

    def add(label, bits, length=width, row_ids=None):
        rows.append([signed32(value) for value in bits])
        mapped.append(list(range(width)) if row_ids is None else list(row_ids))
        lengths.append(length)
        labels.append(label)

    add('empty_poisoned', [0x7FC12345] * width, 0)
    add('one_signed_zero', [0x80000000] + [0x7FFFFFFF] * (width - 1), 1)
    exceptional = [SPECIAL_SCORE_BITS[(index + seed) % len(SPECIAL_SCORE_BITS)] for index in range(width)]
    negative_ids = [(-2 if index % 7 == 0 else -1 if index % 11 == 0 else index) for index in range(width)]
    add('underfull_exceptional', exceptional, min(topk - 1, 17), negative_ids)
    add('exactly_k_exceptional', exceptional, topk, negative_ids)
    add('overfull_unique', [0x3F000000 + value for value in permutation])
    add('all_equal', [0x3F000000] * width)

    greater = max(1, topk // 4)
    ties = min(width - greater, topk + 17)
    bits = [0xBF800000] * width
    for position in permutation[:greater]:
        bits[position] = 0x40000000
    for position in permutation[greater:greater + ties]:
        bits[position] = 0x3F800000
    add('cutoff_ties', bits)
    add('signed_zeros', [0x80000000 if (index + seed) % 2 else 0x00000000 for index in range(width)])
    add('infinities', [(0xFF800000, 0x3F800000, 0x7F800000)[(index + seed) % 3] for index in range(width)])
    add('positive_nan_payloads', [(0x7F800001, 0x7F812345, 0x7FC12345, 0x7FFFFFFF,
                                  0x7F800000)[(index + seed) % 5] for index in range(width)])
    add('negative_nan_payloads', [(0xFF800001, 0xFF812345, 0xFFC12345, 0xFFFFFFFF,
                                  0xFF800000)[(index + seed) % 5] for index in range(width)])
    add('mixed_score_words', exceptional)
    add('negative_ids_are_ranked', [0x40000000 - value for value in permutation], row_ids=negative_ids)
    add('duplicate_ids', [0x3F000000 + value for value in permutation], row_ids=[19] * width)
    prefix = min(33, topk - 1)
    add('ignored_poisoned_suffix', [0x3F800000] * prefix + [0x7FFFFFFF] * (width - prefix), prefix)
    return (torch.tensor(rows, dtype=torch.int32).view(torch.float32),
            torch.tensor(lengths, dtype=torch.int32), torch.tensor(mapped, dtype=torch.int32), tuple(labels))


def negative_nan_without_coarse_overflow():
    """A negative NaN below 4095 positive zeros, within the selector's coarse-bin capacity."""
    words = torch.zeros((1, 4096), dtype=torch.int32)
    words[0, 0] = signed32(0xFFC12345)
    return (words.view(torch.float32), torch.tensor([4096], dtype=torch.int32),
            torch.arange(4096, dtype=torch.int32).view(1, -1), ('negative_nan_below_zeros',))


def test_negative_nan_does_not_displace_zero_at_the_topk_boundary():
    scores, lengths, ids, _ = negative_nan_without_coarse_overflow()
    values, selected, positions = ordered_reference(scores, lengths, ids, 2048)
    assert not values.count_nonzero()
    expected = torch.arange(1, 2049, dtype=torch.int32).view(1, -1)
    assert torch.equal(selected, expected) and torch.equal(positions, expected)
    bad_cutoff = ordered_key(0xFFC12345)
    keys = [ordered_key(value) for value in scores.view(torch.int32)[0].tolist()]
    assert sum(key > bad_cutoff for key in keys) == 4095 > 2048


def test_uint32_keys_separate_signed_zero_infinities_and_nan_payloads():
    assert sorted(SPECIAL_SCORE_BITS, key=ordered_key) == list(SPECIAL_SCORE_BITS)
    assert ordered_key(0x80000000) + 1 == ordered_key(0x00000000)
    assert ordered_key(0xFF800001) < ordered_key(0xFF800000)
    assert ordered_key(0x7F800000) < ordered_key(0x7F800001) < ordered_key(0x7FC12345)


def test_cutoff_ties_use_input_position_and_keep_score_id_pairs():
    bits = [0x3F800000, 0x40000000, 0x40000000, 0, 0x40400000, 0x40000000, 0x40000000]
    values, ids, positions = ordered_row_reference(bits, [3, 19, -2, 4, 19, 8, 7], 7, 4)
    assert positions == [1, 2, 4, 5]
    assert values == [signed32(bits[index]) for index in positions]
    assert ids == [19, -1, 19, 8]


def test_negative_ids_consume_selected_slots_and_do_not_drop_scores():
    bits = [0x7F800000, 0xFF800000, 0x3F800000, 0x7FC12345]
    values, ids, positions = ordered_row_reference(bits, [-7, 10, 44, -2147483648], 4, 3)
    assert positions == [0, 2, 3]
    assert ids == [-1, 44, -1]
    assert values == [signed32(bits[index]) for index in positions]


def test_underfull_rows_copy_exceptional_bits_in_input_order_and_ignore_suffix():
    bits = [0xFFC12345, 0, 0x80000000, 0x7FFFFFFF]
    values, ids, positions = ordered_row_reference(bits, [-2, 5, 5, 0], 3, 5)
    assert values == [signed32(value) for value in bits[:3]] + [signed32(NEGATIVE_INFINITY_BITS)] * 2
    assert ids == [-1, 5, 5, -1, -1]
    assert positions == [0, 1, 2, -1, -1]
    assert ordered_row_reference(bits, [-2, 5, 5, 0], 0, 3) == (
        [signed32(NEGATIVE_INFINITY_BITS)] * 3, [-1] * 3, [-1] * 3)


@pytest.mark.parametrize('width,topk,seed', [(4096, 512, 0), (32768, 2048, 1)])
def test_adversarial_fixture_has_exact_budget_and_cutoff_membership(width, topk, seed):
    scores, lengths, ids, labels = adversarial_rows(width, topk, seed)
    values, selected_ids, selected_positions = ordered_reference(scores, lengths, ids, topk)
    raw = scores.view(torch.int32)
    for row, length in enumerate(lengths.tolist()):
        count = min(length, topk)
        positions = selected_positions[row, :count].tolist()
        assert len(positions) == len(set(positions)) == count, labels[row]
        assert positions == sorted(positions), labels[row]
        assert all(0 <= position < length for position in positions), labels[row]
        assert torch.equal(values[row, :count], raw[row, positions]), labels[row]
        assert selected_ids[row, :count].tolist() == [max(int(ids[row, position]), -1) for position in positions]
        assert (selected_positions[row, count:] == -1).all()
        assert (selected_ids[row, count:] == -1).all()
        assert (values[row, count:] == signed32(NEGATIVE_INFINITY_BITS)).all()
        if length > topk:
            keys = [ordered_key(value) for value in raw[row, :length].tolist()]
            cutoff = min(keys[position] for position in positions)
            greater = [index for index, key in enumerate(keys) if key > cutoff]
            equal = [index for index, key in enumerate(keys) if key == cutoff]
            assert len(greater) <= topk <= len(greater) + len(equal), labels[row]
            assert positions == sorted(greater + equal[:topk - len(greater)]), labels[row]
    assert selected_positions[labels.index('all_equal')].tolist() == list(range(topk))
    assert (selected_ids[labels.index('duplicate_ids')] == 19).all()


@pytest.mark.parametrize('length,topk', [(-1, 2), (4, 2), (True, 2), (2, 0), (2, 1.5)])
def test_oracle_rejects_unbounded_extents(length, topk):
    with pytest.raises(ValueError):
        ordered_row_reference([0, 0, 0], [0, 1, 2], length, topk)


def metadata_inputs(device='cpu', *, rows=3, width=32768, topk=2048):
    tiles = (width + 511) // 512
    return [torch.empty((rows, width), dtype=torch.float32, device=device),
            torch.empty(rows, dtype=torch.int32, device=device),
            torch.empty((rows, width), dtype=torch.int32, device=device),
            torch.empty((rows, topk), dtype=torch.float32, device=device),
            torch.empty((rows, topk), dtype=torch.int32, device=device),
            torch.empty(rows, dtype=torch.int32, device=device),
            torch.empty((rows, tiles), dtype=torch.int32, device=device),
            torch.empty((rows, tiles), dtype=torch.int32, device=device)]


@pytest.mark.parametrize('device', ['cpu', 'meta'])
def test_validation_accepts_independent_storage_and_adjacent_output_views(device):
    from ignition.components.ordered_row_topk import validate_ordered_topk

    operands = metadata_inputs(device)
    validate_ordered_topk(*operands)
    shared = torch.empty(2 * 3 * 2048, dtype=torch.int32, device=device)
    operands[3] = shared[:3 * 2048].view(torch.float32).view(3, 2048)
    operands[4] = shared[3 * 2048:].view(3, 2048)
    validate_ordered_topk(*operands)
    operands[4] = shared[:3 * 2048].view(3, 2048)
    with pytest.raises(ValueError, match='disjoint'):
        validate_ordered_topk(*operands)


@pytest.mark.parametrize('index', range(8))
def test_validation_rejects_changed_dtype_shape_or_stride(index):
    from ignition.components.ordered_row_topk import validate_ordered_topk

    operands = metadata_inputs()
    operands[index] = operands[index].to(torch.float64)
    with pytest.raises(ValueError):
        validate_ordered_topk(*operands)
    operands = metadata_inputs()
    operands[index] = operands[index][:-1]
    with pytest.raises(ValueError):
        validate_ordered_topk(*operands)
    operands = metadata_inputs()
    original = operands[index]
    shape = (*original.shape[:-1], original.shape[-1] * 2)
    operands[index] = torch.empty(shape, dtype=original.dtype)[..., ::2]
    with pytest.raises(ValueError):
        validate_ordered_topk(*operands)


@pytest.mark.parametrize('index', range(3, 8))
def test_mutable_outputs_and_metadata_cannot_overlap_input_storage(index):
    from ignition.components.ordered_row_topk import validate_ordered_topk

    operands = metadata_inputs()
    destination = operands[index]
    operands[index] = operands[0].view(torch.int32).flatten()[:destination.numel()].view(
        destination.dtype).view(destination.shape)
    with pytest.raises(ValueError, match='disjoint'):
        validate_ordered_topk(*operands)


@pytest.mark.parametrize('topk', [512, 1024, 2048])
def test_underfull_widths_do_not_require_a_full_selection_budget(topk):
    from ignition.components.ordered_row_topk import validate_ordered_topk

    validate_ordered_topk(*metadata_inputs(width=17, topk=topk))


@pytest.mark.parametrize('width', [1, 64, 511, 512, 513])
def test_radix_launch_has_complete_count_tiles_for_underfull_widths(monkeypatch, width):
    from ignition.components import ordered_row_topk as component

    operands = metadata_inputs(width=width, topk=512)
    operation = object.__new__(component.PreparedOrderedRowTopK)
    operation.rows, operation.width, operation.topk = 3, width, 512
    operation.radix, operation.tiles = True, operands[6].shape[1]
    (operation.scores, operation.lengths, operation.indices, operation.output_scores,
     operation.output, operation.cutoffs, operation.greater, operation.ties) = operands
    launches = []

    class Launch:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, grid):
            return lambda *args, **metadata: launches.append((self.kernel, grid, args, metadata))

    monkeypatch.setattr(torch.library, 'wrap_triton', Launch)
    operation()
    radix, emit = launches
    assert radix[0] is component._radix_cutoff_counts
    assert emit[0] is component._emit
    metadata = radix[3]
    assert metadata['ROW_BLOCK'] >= max(512, width)
    assert metadata['BLOCK'] == 512
    assert metadata['ROW_BLOCK'] % metadata['BLOCK'] == 0
    assert metadata['ROW_BLOCK'] // metadata['BLOCK'] >= metadata['TILES']
    assert operands[6].shape == operands[7].shape == (3, metadata['TILES'])


def test_fake_tensor_alias_validation_uses_storage_identity():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from ignition.components.ordered_row_topk import validate_ordered_topk

    with FakeTensorMode():
        operands = metadata_inputs()
        validate_ordered_topk(*operands)
        operands[7] = operands[6]
        with pytest.raises(ValueError, match='disjoint'):
            validate_ordered_topk(*operands)
