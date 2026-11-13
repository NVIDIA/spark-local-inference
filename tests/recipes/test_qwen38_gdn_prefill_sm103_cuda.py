# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM103 GDN prefill through FlashInfer's chunked delta rule against b12x's token-sequential prefill."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an explicitly assigned CUDA device")

KEY_HEADS, VALUE_HEADS, SLOTS = 16, 48, 6


def _case(lengths, slots, fresh, tokens):
    from ignition.components.recurrent import DeltaMetadata

    generator = torch.Generator(device="cuda").manual_seed(sum(lengths) + len(lengths))
    mixed = (torch.randn((tokens, (2 * KEY_HEADS + VALUE_HEADS) * 128), generator=generator, device="cuda")
             * 0.5).bfloat16()
    ba = torch.randn((tokens, 2 * VALUE_HEADS), generator=generator, device="cuda").bfloat16()
    a, b = ba[:, VALUE_HEADS:], ba[:, :VALUE_HEADS]
    a_log = (torch.rand(VALUE_HEADS, generator=generator, device="cuda") * 0.5).bfloat16()
    dt_bias = (torch.randn(VALUE_HEADS, generator=generator, device="cuda") * 0.1).bfloat16()
    state = torch.randn((SLOTS, VALUE_HEADS, 128, 128), generator=generator, device="cuda") * 0.05
    state[0].zero_()
    metadata = DeltaMetadata.allocate(len(lengths), torch.device("cuda"))
    starts = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    metadata.query_start_loc.copy_(starts)
    live = torch.tensor([length > 0 for length in lengths], device="cuda")
    final = torch.tensor(slots, dtype=torch.int64, device="cuda")
    metadata.fresh.copy_(torch.tensor(fresh, device="cuda") & live)
    metadata.final_state_indices.copy_(final)
    metadata.initial_state_indices.copy_(torch.where(metadata.fresh, 0, final))
    metadata.num_seqs.fill_(len(lengths))
    metadata.num_tokens.fill_(sum(lengths))
    return dict(mixed=mixed, a=a, b=b, a_log=a_log, dt_bias=dt_bias, state=state, metadata=metadata)


def _b12x(case, tokens):
    from b12x.preparation import PreparationSession
    from b12x.sequence import gdn_prefill

    state = case["state"].clone()
    output = torch.zeros((tokens, VALUE_HEADS, 128), device="cuda").bfloat16()
    q, k, v = case["mixed"].split((2048, 2048, 6144), dim=-1)
    m = case["metadata"]
    args = dict(A_log=case["a_log"], dt_bias=case["dt_bias"], recurrent_state=state, num_seqs=m.num_seqs,
                num_tokens=m.num_tokens, output=output, a=case["a"], b=case["b"],
                q=q.view(tokens, KEY_HEADS, 128), k=k.view(tokens, KEY_HEADS, 128),
                v=v.view(tokens, VALUE_HEADS, 128), cu_seqlens=m.query_start_loc,
                initial_state_indices=m.initial_state_indices, final_state_indices=m.final_state_indices,
                checkpoint_state_indices=m.checkpoint_state_indices, checkpoint_offsets=m.checkpoint_offsets)
    caps = gdn_prefill.Caps(device=torch.device("cuda"), max_tokens=tokens, max_seqs=len(m.final_state_indices),
                            max_state_slots=SLOTS, key_heads=KEY_HEADS, value_heads=VALUE_HEADS,
                            checkpoint_export=False, null_state_index=0, recurrence_policy="sequential_k1")
    plan = gdn_prefill.plan(caps, invocation=gdn_prefill.invocation_from_tensors(**args))
    from ignition.components.recurrent import prepare_sequence_binding

    original = state.clone()
    box = {}
    with PreparationSession(device=torch.device("cuda"), autotune=False, compile_workers=0) as session:
        prepare_sequence_binding(plan, args, session=session, name="gdn_prefill", public_bind=gdn_prefill.bind,
                                 mode="prefill", set_binding=lambda binding: box.setdefault("binding", binding))
        session.freeze()
    # Preparation primes the call; the comparison starts from the original state.
    state.copy_(original)
    output.zero_()
    gdn_prefill.run(box["binding"])
    return output, state


def _fp64(case, lengths, slots, fresh):
    """Token-sequential delta rule in FP64 from the BF16 inputs: final states per slot."""
    mixed = case["mixed"].double()
    a_scale = torch.exp(case["a_log"].double())
    states = {}
    start = 0
    for length, slot, is_fresh in zip(lengths, slots, fresh):
        if not length:
            continue
        state = torch.zeros_like(case["state"][0], dtype=torch.float64) if is_fresh else case["state"][slot].double()
        for token in range(start, start + length):
            k = mixed[token, 2048:4096].view(KEY_HEADS, 128).repeat_interleave(VALUE_HEADS // KEY_HEADS, 0)
            v = mixed[token, 4096:].view(VALUE_HEADS, 128)
            k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            decay = torch.exp(-a_scale * torch.nn.functional.softplus(case["a"][token].double() + case["dt_bias"].double()))
            beta = torch.sigmoid(case["b"][token].float()).bfloat16().double()
            state = state * decay[:, None, None]
            delta = (v - torch.einsum("hvk,hk->hv", state, k)) * beta[:, None]
            state = state + delta[:, :, None] * k[:, None, :]
        states[slot] = state
        start += length
    return states


@pytest.mark.parametrize(("lengths", "slots", "fresh"), [
    ((4096,), (3,), (True,)),
    ((1000, 0, 3000), (2, 0, 5), (False, False, True)),
    ((17, 2000, 64), (1, 4, 2), (True, False, False)),
])
def test_chunked_prefill_matches_sequential_b12x(lengths, slots, fresh):
    from b12x.preparation import PreparationSession

    from ignition.components.gdn_prefill_sm103 import PreparedGdnPrefill
    from ignition.components.sm103 import is_sm103

    if not is_sm103("cuda"):
        pytest.skip("SM103 path")
    tokens = 4096
    case = _case(lengths, slots, fresh, tokens)
    expected_output, expected_state = _b12x(case, tokens)
    state = case["state"].clone()
    output = torch.zeros((tokens, VALUE_HEADS, 128), device="cuda").bfloat16()
    with PreparationSession(device=torch.device("cuda"), autotune=False, compile_workers=0) as session:
        prefill = PreparedGdnPrefill(mixed_qkv=case["mixed"], a=case["a"], b=case["b"], A_log=case["a_log"],
                                     dt_bias=case["dt_bias"], state=state, metadata=case["metadata"], output=output,
                                     key_heads=KEY_HEADS, value_heads=VALUE_HEADS, session=session, name="prefill")
        session.freeze()
    state.copy_(case["state"])
    output.zero_()
    prefill()
    torch.cuda.synchronize()

    live = int(sum(lengths))
    torch.testing.assert_close(output[:live].float(), expected_output[:live].float(), rtol=2**-5, atol=3e-2)
    assert not output[live:].any()
    exact = _fp64(case, lengths, slots, fresh)
    for slot in range(SLOTS):
        if slot in exact:
            # The chunked kernel's final state is at least as close to FP64 as b12x's sequential one.
            chunked_error = (state[slot].double() - exact[slot]).abs().max().item()
            sequential_error = (expected_state[slot].double() - exact[slot]).abs().max().item()
            assert chunked_error <= 2 * sequential_error + 1e-5, (slot, chunked_error, sequential_error)
        else:
            assert torch.equal(state[slot], case["state"][slot])
