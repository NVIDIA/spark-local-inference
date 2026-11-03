# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib

import pytest
import torch

from qualify.replay_floor import _byte_view, _hash, decode_groups


@pytest.mark.parametrize('field,value', [('temperature', 0), ('top_k', 1), ('top_p', 1)])
def test_decode_groups_reject_different_sampling(field, value):
    request = dict(temperature=1, top_k=20, top_p=.95)
    request[field] = value
    corpus = dict(cases=[dict(id='case', request=request, output_tokens=256,
                             expected_cache='cold', prompt_tokens=[17])],
                  groups=[dict(kind='decode-b1', measured=True, batch=1, cases=['case'])])
    with pytest.raises(ValueError, match='temperature=1, top_k=20, top_p=0.95'):
        decode_groups(corpus)


@pytest.mark.parametrize('dtype', [torch.int64, torch.uint64, torch.float32, torch.bfloat16])
def test_single_request_strided_state_has_packed_byte_readback(dtype):
    value = torch.ones(1, dtype=dtype).as_strided((1,), (14456032,))
    packed = torch.ones(1, dtype=dtype).view(torch.uint8)
    assert torch.equal(_byte_view(value), packed)
    assert _hash(value) == hashlib.sha256(packed.numpy().tobytes()).hexdigest()


def test_state_comparison_distinguishes_signed_zero():
    assert torch.equal(torch.tensor([0.]), torch.tensor([-0.]))
    assert not torch.equal(_byte_view(torch.tensor([0.])), _byte_view(torch.tensor([-0.])))


def test_state_readback_respects_noncontiguous_layout():
    value = torch.arange(12).reshape(3, 4).T
    assert torch.equal(_byte_view(value), torch.tensor([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]).view(torch.uint8))


@pytest.mark.parametrize('batch', [1, 2, 4])
def test_init_stages_exact_empty_endpoint_before_first_replay(batch):
    from dataclasses import fields
    from types import SimpleNamespace

    from ignition.components.sampling import AdmissionBuffers
    from ignition.recipes.qwen38.sampling import DeviceRequestState
    from ignition.recipes.qwen38.state import StateTransferMetadata
    from qualify.replay_floor import ReplayCase

    runner = ReplayCase.__new__(ReplayCase)
    admission = AdmissionBuffers.allocate('cpu', vocab_size=248320)
    metadata = StateTransferMetadata.allocate(4, 'cpu')
    runner.host = {field.name: getattr(admission, field.name) for field in fields(admission)}
    runner.host.update({'state_' + field.name: getattr(metadata, field.name) for field in fields(metadata)})
    runner.host['page_tables'] = torch.full((4, 8), 777, dtype=torch.int32)
    runner.host['state_expected_consumed'].fill_(123)
    runner.admission = admission
    runner.requests = DeviceRequestState.allocate('cpu')
    runner.state = SimpleNamespace(reset_for_preparation=lambda: None)
    runner.model = SimpleNamespace(buffers={'completion_records': torch.zeros((2, 4, 32))},
                                   memory_report={'pages': 32})
    runner.touched_pages, runner.cache = 0, {}

    class AtInit(Exception):
        pass

    def execute(name):
        assert name == f'init{batch}'
        torch.testing.assert_close(runner.host['admission_parameters'][:batch, :2],
                                   torch.tensor([[1., .95]]).expand(batch, -1), rtol=0, atol=0)
        assert runner.host['admission_top_k'][:batch].tolist() == [20] * batch
        assert runner.host['state_expected_consumed'].tolist() == [0] * batch + [-1] * (4 - batch)
        assert runner.host['state_source_slots'].tolist() == list(range(1, batch + 1)) + [0] * (4 - batch)
        assert runner.host['state_destination_slots'].tolist() == runner.host['state_source_slots'].tolist()
        assert runner.host['state_operation'].tolist() == [1] * batch + [0] * (4 - batch)
        for slot in range(batch):
            assert runner.host['page_tables'][slot, :5].tolist() == list(range(slot * 5, slot * 5 + 5))
        raise AtInit

    runner.execute = execute
    cases = [{'request': {}, 'prompt_tokens': [17] * 63, 'output_tokens': 256} for _ in range(batch)]
    with pytest.raises(AtInit):
        runner.prepare(cases)
