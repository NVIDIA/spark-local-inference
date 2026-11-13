# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch

from qualify.mtp import MODEL_ID, REVISION, fixture_from_trace


def trace():
    layers = []
    for rows in (3, 1, 1):
        layers.append(torch.arange(48 * rows * 10240, dtype=torch.float32).reshape(48, rows, 10240).bfloat16().float())
    return {'model_id': MODEL_ID, 'revision': REVISION, 'prompt_tokens': [12, 34, 56],
            'teacher_tokens': [78, 90], 'layer_outputs': layers}


def test_pairs_shift_tokens_but_not_target_hidden_or_position(monkeypatch):
    def unexpected_cuda_access():
        pytest.fail("CPU draft fixtures must not access CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", unexpected_cuda_access)
    source = trace()
    fixture = fixture_from_trace(source)
    first, second = fixture['cases']
    assert first['token_ids'].tolist() == [34, 56, 78]
    assert first['positions'].tolist() == [0, 1, 2]
    assert torch.equal(first['hidden'], source['layer_outputs'][0][-1].bfloat16())
    assert second['token_ids'].tolist() == [90]
    assert second['positions'].tolist() == [3]
    assert torch.equal(second['hidden'], source['layer_outputs'][1][-1].bfloat16())
    assert fixture['unconsumed_target_contexts'] == 1


@pytest.mark.parametrize('change,match', [
    (lambda x: x.update(revision='other'), 'exact checkpoint'),
    (lambda x: x['layer_outputs'].pop(), 'one target context'),
    (lambda x: x['layer_outputs'].__setitem__(0, torch.zeros(48, 3, 2560)), 'four-stream'),
    (lambda x: x['layer_outputs'][0][-1].fill_(float('nan')), 'finite'),
    (lambda x: x['layer_outputs'][0][-1].fill_(0.1234567), 'exact BF16'),
])
def test_rejects_wrong_identity_alignment_or_lossy_feedback(change, match):
    source = trace()
    change(source)
    with pytest.raises(ValueError, match=match):
        fixture_from_trace(source)


def test_reference_tail_selection_is_explicit_and_metrics_use_reference_norm():
    from qualify.mtp import compare_output
    actual = {'residual': torch.tensor([[99., 99.], [3., 4.]]),
              'hidden': torch.tensor([[99., 99.], [3., 4.]]),
              'logits': torch.arange(12, dtype=torch.float32)[None]}
    reference = {'selected_rows': [1], 'residual': torch.tensor([[0., 4.]]),
                 'hidden': torch.tensor([[3., 4.]]), 'logits': actual['logits'].clone()}
    result = compare_output(actual, reference)
    assert result['residual']['relative_l2'] == .75
    assert result['hidden']['fp32_value_bits_equal']
    assert result['logits']['fp32_value_bits_equal']
    reference['selected_rows'] = [0]
    with pytest.raises(ValueError, match='exact request-tail'):
        compare_output(actual, reference)


def test_reference_tail_rejects_nonfinite_outputs():
    from qualify.mtp import compare_output
    actual = {'residual': torch.zeros(1, 2), 'hidden': torch.zeros(1, 2), 'logits': torch.zeros(1, 12)}
    reference = {'selected_rows': [0], **copy.deepcopy(actual)}
    reference['hidden'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        compare_output(actual, reference)


def test_stage_copies_preserve_return_identity_and_do_not_retain_mutable_view():
    from types import SimpleNamespace

    from qualify.mtp import StageTrace
    tensor = torch.tensor([[1., 2.], [3., 4.]])
    owner = SimpleNamespace(operation=lambda value: value)
    original = owner.operation
    trace = StageTrace()
    trace.wrap(owner, 'operation', after=lambda args, kwargs, result: trace.record('result', result))
    assert owner.operation(tensor) is tensor
    tensor.zero_()
    assert torch.equal(trace.buffers['result'], torch.tensor([[1., 2.], [3., 4.]]))
    trace.close()
    assert owner.operation is original


def test_hyperconnection_stage_trace_excludes_volatile_gate_workspace():
    from types import SimpleNamespace

    from ignition.recipes.qwen38.hyperconnection import (
        HyperconnectionScratch,
        PreparedHyperconnection,
    )
    from qualify.mtp import StageTrace

    scratch = HyperconnectionScratch.allocate(2, torch.device('cpu'))
    scratch.gate_workspace.fill_(float('nan'))
    mixers = []
    for index in range(3):
        mixer = PreparedHyperconnection.__new__(PreparedHyperconnection)
        mixer.scratch = scratch
        mixer.injection = scratch.injection_logits if index < 2 else None
        mixer.mix = lambda source: scratch.block_input
        mixers.append(mixer)
    op = SimpleNamespace(attn_hc=mixers[0], mlp_hc=mixers[1], final_mixer=mixers[2],
                         feedback=lambda: None, attention=lambda: None, moe=lambda: None)
    trace = StageTrace()
    try:
        trace.install_flare(op)
        for mixer in mixers:
            assert mixer.mix(scratch.normalized) is scratch.block_input
        expected = {f'{prefix}_{field}' for prefix in ('attn', 'mlp', 'final')
                    for field in ('normalized', 'projected_down', 'bottleneck', 'mix')}
        expected.update(('attn_injection_logits', 'mlp_injection_logits'))
        assert set(trace.buffers) == expected
        assert all(torch.isfinite(value).all() for value in trace.buffers.values())
        scratch.normalized.fill_(17)
        assert not trace.buffers['attn_normalized'].count_nonzero()
        assert scratch.gate_workspace.isnan().all()
    finally:
        trace.close()


def test_stage_comparison_only_selects_declared_request_tail_and_records_missing_keys():
    from qualify.mtp import compare_output
    actual = {'residual': torch.zeros(2, 2), 'hidden': torch.zeros(2, 2), 'logits': torch.zeros(1, 12),
              'stages': {'feedback': torch.ones(2, 2), 'mlp_mix': torch.tensor([[1., 2.], [3., 4.]]),
                         'only_actual': torch.zeros(2, 2)}}
    reference = {'selected_rows': [1], 'residual': torch.zeros(1, 2), 'hidden': torch.zeros(1, 2),
                 'logits': torch.zeros(1, 12), 'stages': {'feedback': torch.ones(2, 2),
                 'mlp_mix': torch.tensor([[3., 4.]]), 'only_reference': torch.zeros(1, 2)}}
    measured = compare_output(actual, reference)
    assert measured['stages']['feedback']['selected_actual_rows'] is None
    assert measured['stages']['mlp_mix']['selected_actual_rows'] == [1]
    assert measured['stages']['mlp_mix']['fp32_value_bits_equal']
    assert measured['stage_coverage'] == {'missing_in_flare': ['only_reference'], 'missing_in_reference': ['only_actual']}


def test_functional_linear_observer_preserves_values_and_restores_function():
    import torch.nn.functional as functional

    from qualify.mtp import observe_functional_linear
    module = torch.nn.Linear(3, 2, bias=False)
    value = torch.tensor([[1., 2., 3.]])
    original, expected = functional.linear, module(value)
    with observe_functional_linear(module) as calls:
        actual = module(value)
        assert torch.equal(actual, expected)
    assert functional.linear is original
    assert calls == [{'modules': [''], 'source_shape': [1, 3], 'weight_shape': [2, 3],
                      'source_dtype': 'torch.float32', 'weight_dtype': 'torch.float32', 'output_dtype': 'torch.float32'}]
    with pytest.raises(RuntimeError, match='observer test'):
        with observe_functional_linear(module):
            raise RuntimeError('observer test')
    assert functional.linear is original
