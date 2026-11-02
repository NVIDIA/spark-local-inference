# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The sampler measurement oracle follows live state across page boundaries."""

import torch

from scripts.qualify_sampling_paths import SLOTS, cpu_case


def test_cpu_sampler_oracle_crosses_page_boundary_without_rolling_state():
    initialized = torch.cuda.is_initialized()
    logits = torch.full((248320,), -torch.inf, dtype=torch.bfloat16)
    logits[42] = 1
    case = cpu_case({'prompt': [17] * 63, 'tokens': [42], 'logits': [logits]},
                    rows=1, block_steps=3, policy='checkpoint_defaults')
    expected = case['expected']
    slot = SLOTS[0]
    assert expected['state.consumed_tokens'][slot].item() == 66
    assert expected['state.ple_history'][slot].tolist() == [42, 42]
    assert expected['sampled_ids'][:, slot].tolist() == [42, 42, 42]
    assert expected['emitted_counts'][:, slot].tolist() == [1, 1, 1]
    assert expected['vector_tokens'][:, slot].tolist() == [42, 42, 42]
    records = expected['completion_records']
    assert records.shape == (2, 4, 32)
    assert records[0, slot].tolist() == [31 + slot, 1, 65, 3, 42, 1, 0, 0,
        1, 0, 0, 0, -1, 64, 0, 1, 42] + [-1] * 15
    assert records[1].eq(-99).all()
    assert not any('boundary' in name for name in expected)
    assert torch.cuda.is_initialized() == initialized


def test_cpu_sampler_oracle_clears_vector_emission_after_eos():
    logits = torch.full((248320,), -torch.inf, dtype=torch.bfloat16)
    logits[248044] = 1
    case = cpu_case({'prompt': [17] * 63, 'tokens': [248044], 'logits': [logits]},
                    rows=1, block_steps=3, policy='checkpoint_defaults')
    expected, slot = case['expected'], SLOTS[0]
    assert expected['sampled_ids'][:, slot].tolist() == [248044, -1, -1]
    assert expected['emitted_counts'][:, slot].tolist() == [1, 0, 0]
    assert expected['vector_tokens'][:, slot].tolist() == [248044, -1, -1]
    assert expected['state.pending_token'][slot] == 248044
    assert expected['state.consumed_tokens'][slot] == 63
    assert expected['state.rng_counter'][slot] == 1
    assert expected['completion_records'][0, slot].tolist() == [31 + slot, 1, 63, 1, -1, 2, 0, 0,
        0, 0, 0, 0, -1, 62, 0, 1] + [-1] * 16
