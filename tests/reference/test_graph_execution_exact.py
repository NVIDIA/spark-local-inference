# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU command and reset contracts for ordinary exact-checkpoint replay."""
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from ignition.components.sampling import AdmissionBuffers
from ignition.preparation.execution import GraphSignature
from ignition.recipes.qwen38.decoder_state import DecoderInputs
from qualify.graph_execution import Scenario


def make_scenario(tmp_path, cap=512):
    inputs = DecoderInputs.allocate('cpu', context=8192, tokens=cap)
    admission = AdmissionBuffers.allocate('cpu', vocab_size=248320)
    buffers = {**inputs.buffers(), **{f.name: getattr(admission, f.name) for f in fields(admission)}}
    for name in ('source_slots', 'destination_slots', 'qsa_slots', 'expected_consumed'):
        buffers['state_' + name] = torch.zeros(4, dtype=torch.int64)
    for name in ('operation', 'accepted_tokens'):
        buffers['state_' + name] = torch.zeros(4, dtype=torch.int32)
    buffers['state_num_ops'] = torch.zeros(1, dtype=torch.int32)
    for name in ('slot', 'generation', 'feature_offset', 'num_patches'):
        buffers['vision_' + name] = torch.zeros(1, dtype=torch.int32)
    signatures = {}
    for kind in ('init', 'restore', 'snapshot', 'resume', 'decode'):
        for batch in (1, 2, 4):
            signatures[f'{kind}{batch}'] = GraphSignature(kind, batch, 1, tuple(buffers))
    for size in (128, 256, 512, *([1024] if cap > 512 else [])):
        signatures[f'prefill{size}'] = GraphSignature('prefill', 1, size, tuple(buffers))
    for size in (256, 1024, 4096, 16384):
        signatures[f'vision{size}'] = GraphSignature('vision', 1, size, tuple(buffers))
    serving = dict.fromkeys(signatures)
    model = SimpleNamespace(buffers=buffers, signatures=signatures, serving_graphs=serving,
                            graphs={**serving, 'collective-tuning-0': object()},
                            memory_report={'pages': 4096})
    # These values exercise host packing only; they are not model numerical fixtures.
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(range(len(text))))
    image = {'prompt': list(range(80)), 'mrope': torch.arange(80)[:, None].expand(-1, 3),
             'token_indices': torch.full((80,), -1, dtype=torch.int32),
             'vision_inputs': {'vision_num_patches': torch.tensor([256], dtype=torch.int32)}}
    (tmp_path / 'uploads').mkdir()
    return Scenario(model, tokenizer, image, tmp_path)


def uploaded(scenario, command, name, dtype):
    digest = dict(command['uploads'])[name]
    return torch.frombuffer(bytearray(scenario.payloads[digest]), dtype=dtype)


@pytest.mark.parametrize('cap', [512, 1024])
def test_complete_scenario_uses_live_exact_snapshots_and_keeps_original_branch(tmp_path, cap):
    before = torch.cuda.is_initialized()
    scenario = make_scenario(tmp_path, cap).build()
    commands = [c for g in scenario.groups for c in g]
    assert set(c['graph'] for c in commands) == set(scenario.model.serving_graphs)
    assert 'collective-tuning-0' not in {c['graph'] for c in commands}
    assert all(t > 0 for c in commands for t in c['temperatures'])
    assert any(len(set(c['temperatures'])) > 1 for c in commands if c['graph'].startswith('decode'))
    for command in commands:
        if command['graph'].startswith(('init', 'restore')):
            rows = len(command['slots'])
            parameters = uploaded(scenario, command, 'admission_parameters', torch.float32).reshape(-1, 6)
            torch.testing.assert_close(parameters[:rows, :2],
                torch.tensor([[t, .95] for t in command['temperatures']]), rtol=0, atol=0)
            assert uploaded(scenario, command, 'admission_top_k', torch.int32)[:rows].tolist() == [20] * rows
        for transfer in command.get('transfers', []):
            assert transfer['consumed'] == command['expected_records'][str(transfer['slot'])]['consumed']
            if command['graph'].startswith('snapshot'):
                assert transfer['source'] == transfer['slot'] + 1
                assert transfer['destination'] >= 5
                assert transfer['consumed'] > 0
    branch = next(c for c in commands if c['graph'] == 'restore2' and c['expected_records']['0']['generation'] == 13)
    prompt = scenario.tokens(67, 'cold-tail')[:64] + scenario.tokens(13, 'branch')
    assert branch['admissions']['0']['prompt'] == prompt
    assert next(t for t in branch['transfers'] if t['slot'] == 0)['consumed'] == 64
    # The original unaligned67-token run and its exact snapshot still exist.
    assert any(c['graph'] == 'prefill128' and c['expected_records'].get('2', {}).get('consumed') == 67 for c in commands)
    assert any(t['consumed'] == 67 for c in commands if c['graph'] == 'snapshot4' for t in c['transfers'])
    assert len(scenario.extended_cases) == (2 if cap == 1024 else 0)
    assert torch.cuda.is_initialized() == before


def test_scenario_still_requires_every_serving_variant_with_startup_tuning(tmp_path):
    scenario = make_scenario(tmp_path)
    scenario.model.serving_graphs['uncovered'] = object()
    with pytest.raises(ValueError, match='every prepared graph variant'):
        scenario.build()


def test_partial_restore_has_private_tail_and_physical_tables_before_maintenance(tmp_path):
    scenario = make_scenario(tmp_path)
    prompt = scenario.tokens(75, 'partial')
    scenario.admit('init', 1, [(2, prompt, 1, 4, None, 32)])
    scenario.prefill(2, 67, 128, sample=False)
    scenario.snapshot(1, [2], [0])
    original_pages = scenario.requests[2].pages
    assert scenario.snapshots[0] == (prompt[:67], original_pages[:1])
    scenario.admit('restore', 1, [(3, prompt, 2, 4, 0, 32)])
    command = scenario.groups[-1][0]
    restored_pages = scenario.requests[3].pages
    assert restored_pages[:1] == original_pages[:1]
    assert restored_pages[1] != original_pages[1]
    assert uploaded(scenario, command, 'state_expected_consumed', torch.int64).tolist() == [67, -1, -1, -1]
    assert uploaded(scenario, command, 'state_source_slots', torch.int64).tolist() == [5, 0, 0, 0]
    assert uploaded(scenario, command, 'state_destination_slots', torch.int64).tolist() == [4, 0, 0, 0]
    pages = uploaded(scenario, command, 'page_tables', torch.int32).reshape(4, -1)
    assert pages[3, :len(restored_pages)].tolist() == restored_pages
    scenario.prefill(3, 75, 128)
    assert scenario.groups[-1][0]['expected_records']['3']['consumed'] == 75
    bad = prompt.copy(); bad[66] += 1
    with pytest.raises(ValueError, match='exact checkpoint token prefix'):
        scenario.admit('restore', 1, [(0, bad, 3, 4, 0, 32)])
