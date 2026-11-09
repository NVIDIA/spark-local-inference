# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One width-16 speculative round of the prepared Muse Glimmer drafter against a DFlash2 reference.

The prepared prefill of the traced prompt supplies the target taps (layers 1, 13, 25, 37, 49)
and fills the draft rings through ``PreparedContext``; the round runs through
``PreparedParallelDraftRound`` exactly as the catalog's ``speculative1x16`` graph does. The
oracle is the fork's own DFlash2 arithmetic in the shape of
``tests/recipes/muse_spark/test_dflash_checkpoint_cuda.py`` over the same taps: context
projected through ``fc`` and ``hidden_norm`` into per-layer keys (key norm, rope 500000) and
values rounded to E4M3 at the cache seam as the rings store them, the block
``[pending token, 201818 x 15]`` embedded raw (no embedding norm), five layers each with the
two-tap grouped convolution (kernel 2, group 16) before attention and before the MLP, q/k norms,
bidirectional attention under the inclusive 2048-token sliding mask, the final norm, ``lm_head``
times the target's ``output_multiplier`` under its softcap, and the rank-256 top-16 candidate
selector walked greedily over the block.

Tolerance (BF16 accumulation order through five layers): relative L2 at most 4e-2 on the
block's normalized hidden states and at most 2e-2 on its softcapped FP32 logits, over the
fifteen proposal rows and over the anchor row. The hidden bound is set by the oracle's own
noise floor: recomputed under a different attention kernel it moves its block rows by up to
3.6e-2 (rounding compounds along the two-tap chain), while its logits move by under 2e-2.

Acceptance diagnostics: the prepared selector walk, the oracle's walk, the sampled drafts and
the target's own greedy continuation (fifteen prepared decode steps after the round) are printed
decoded, with the positions and tap layers the composition used. Oracle == greedy but prepared
!= greedy means the draft readout or row mapping; both != greedy means the shared inputs (taps,
positions, mask embedding, context K/V); all three equal means verification. The first
proposal must equal the first greedy token: a drafter with no acceptance is a defect.
"""

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from tests.recipes.muse_glimmer_nvfp4.test_reference_cuda import (
    CHECKPOINT, CUDA, SNAPSHOT, WINDOW, device, prefill, relative_l2, traced_prompt,
)

DFLASH2_SNAPSHOT = os.environ.get("FLARE_MUSE_GLIMMER_DFLASH2")
DFLASH2 = pytest.mark.skipif(not DFLASH2_SNAPSHOT, reason="FLARE_MUSE_GLIMMER_DFLASH2 names no DFlash2 checkpoint")
RELATIVE_L2 = .02
HIDDEN_RELATIVE_L2 = .04  # the oracle's measured self-noise on the block rows is 3.6e-2
BLOCK = 16
GROUP = 16


def rms(x, weight, eps):
    y = x.float()
    return weight * (y * (y.square().mean(-1, keepdim=True) + eps).rsqrt()).to(x.dtype)


def rotary(x, positions, theta):
    dim = x.shape[-1]
    inverse = theta ** (-torch.arange(0, dim, 2, device=x.device).float() / dim)
    angle = positions.float()[:, None] * inverse
    angle = torch.cat((angle, angle), -1)
    left, right = x.chunk(2, -1)
    return x * angle.cos().to(x.dtype)[:, None] + torch.cat((-right, left), -1) * angle.sin().to(x.dtype)[:, None]


def fp8(x):
    return x.clamp(-448, 448).to(torch.float8_e4m3fn)


def convolve(x, delta, base):
    """The serving convolution: BF16 coefficient addition before products, taps within the block."""
    groups = x.shape[-1] // GROUP
    grouped = x.reshape(-1, groups, GROUP)
    coefficients = base.reshape(1, 2, groups, GROUP) + delta.unsqueeze(-1)
    result = coefficients[:, 0] * grouped
    shifted = F.pad(grouped[:-1], (0, 0, 0, 0, 1, 0))
    valid = (torch.arange(x.shape[0], device=x.device) % BLOCK >= 1).reshape(-1, 1, 1)
    result += coefficients[:, 1] * shifted * valid
    return result.flatten(1)


def conv_prepare(x, weights):
    delta = F.linear(x, weights["projection"]).reshape(x.shape[0], 2, 2, -1)
    return convolve(x, delta[:, 0], weights["base"][0]), delta[:, 1]


def reference(weights, taps, block_ids, positions, context_positions):
    """Hidden rows, softcapped logits and the greedy selector walk of the DFlash2 draft over ``taps``."""
    d, c, a = weights.draft, weights.config.draft, weights.config.draft.attention
    eps, theta = c.text["rms_norm_eps"], a.theta
    heads, kv_heads, dim = a.heads, a.kv_heads, a.key_dim
    projected = rms(F.linear(taps, d.projection), d.hidden_norm, eps)
    kv = F.linear(projected, d.context_kv).reshape(taps.shape[0], c.layers, 2 * kv_heads * dim)
    history = []
    for index, layer in enumerate(d.layers):
        key = rotary(rms(kv[:, index, :kv_heads * dim].reshape(-1, kv_heads, dim), layer.key_norm, eps),
                     context_positions, theta)
        history.append((fp8(key).bfloat16(), fp8(kv[:, index, kv_heads * dim:].reshape(-1, kv_heads, dim)).bfloat16()))
    # transformers.masking_utils.sliding_window_bidirectional_overlay: |q - k| <= sliding_window.
    all_positions = torch.cat((context_positions, positions))
    visible = (positions[:, None] - all_positions[None, :]).abs() <= a.window

    def attention(x, layer, cached):
        q, k, v = F.linear(x, layer.qkv).split((heads * dim, kv_heads * dim, kv_heads * dim), -1)
        q = rotary(rms(q.reshape(-1, heads, dim), layer.query_norm, eps), positions, theta)
        k = fp8(rotary(rms(k.reshape(-1, kv_heads, dim), layer.key_norm, eps), positions, theta)).bfloat16()
        v = fp8(v.reshape(-1, kv_heads, dim)).bfloat16()
        k = torch.cat((cached[0], k)).repeat_interleave(heads // kv_heads, 1)
        v = torch.cat((cached[1], v)).repeat_interleave(heads // kv_heads, 1)
        attended = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                                  v.transpose(0, 1)[None], attn_mask=visible[None, None])
        return F.linear(attended[0].transpose(0, 1).reshape(-1, heads * dim), layer.output)

    x = F.embedding(block_ids, weights.embedding)
    for layer, cached in zip(d.layers, history, strict=True):
        before, delta = conv_prepare(rms(x, layer.input_norm, eps), layer.attention_conv)
        x = x + convolve(attention(before, layer, cached), delta, layer.attention_conv["base"][1])
        before, delta = conv_prepare(rms(x, layer.post_attention_norm, eps), layer.mlp_conv)
        gate, up = F.linear(before, layer.mlp.gate_up).chunk(2, -1)
        x = x + convolve(F.linear(F.silu(gate) * up, layer.mlp.down), delta, layer.mlp_conv["base"][1])
    hidden = rms(x, d.norm, eps)
    options = c.text["dflash_config"]
    logits = F.linear(hidden, weights.head).float() * options["output_multiplier"]
    logits = (logits / options["final_logit_softcapping"]).tanh() * options["final_logit_softcapping"]
    return hidden, logits, selector_walk(d.selector, F.linear(hidden, d.selector["projection"]), logits,
                                         block_ids[0], options["selector_top_k"])


def selector_walk(selector, projected, logits, pending, top_k):
    """Greedy candidates through the block: each row's shortlist corrected by its predecessor."""
    walk, predecessor = [], pending
    for position in range(1, logits.shape[0]):
        top, indices = logits[position].topk(top_k, -1, sorted=False)
        coded = selector["predecessor"][predecessor] * projected[position]
        correction = (selector["successor"][indices].float() * coded.float()[None]).sum(-1)
        corrected = torch.full_like(logits[position], -torch.inf).scatter_(0, indices, top + correction)
        predecessor = corrected.argmax()
        walk.append(int(predecessor))
    return walk


@CUDA
@CHECKPOINT
@DFLASH2
@torch.no_grad()
def test_width_16_round_proposes_the_dflash2_block():
    from b12x.preparation import PreparationSession
    from ignition.components.completion import CompletionOutputs, VECTOR_TOKENS_FIELD
    from ignition.components.dflash2 import PreparedContext
    from ignition.components.paged_batch import BatchInputs, Metadata
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.parallel_draft import PreparedParallelDraftRound
    from ignition.components.sampling import DeviceRequestState, RequestAdmission
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    from ignition.recipes.muse_glimmer_nvfp4.decoder import PreparedDecoder, Scratch
    from ignition.recipes.muse_glimmer_nvfp4.dflash import SpeculativePreparation
    from ignition.recipes.muse_glimmer_nvfp4.weights import ModelWeights
    from ignition.recipes.muse_spark.pool import PAGE_TOKENS, ContinuationPool

    snapshot, draft_snapshot, selected = Path(SNAPSHOT), Path(DFLASH2_SNAPSHOT), device()
    ids = traced_prompt(snapshot)
    tokens = len(ids)
    assert tokens > WINDOW
    shard = TensorParallelShard(0, 1)
    weights = ModelWeights.load(inspect_indexed_checkpoint(snapshot), selected, shard=shard,
                                draft_snapshot=draft_snapshot)
    c, draft = weights.config.text, weights.config.draft
    assert draft.block_size == BLOCK and draft.target_layers == (1, 13, 25, 37, 49)
    assert weights.draft.selector is not None and draft.text["dflash_config"]["conv_group_size"] == GROUP
    blocks = (tokens + BLOCK + PAGE_TOKENS - 1) // PAGE_TOKENS
    pool = ContinuationPool.allocate(weights.config, shard, 3, blocks, selected, dflash=True)
    requests = DeviceRequestState.allocate(selected, vocab_size=c["vocab_size"], capacity=1)
    inputs = BatchInputs.allocate(BLOCK, blocks, selected, capacity=1)
    completion = CompletionOutputs.allocate((2, 1, VECTOR_TOKENS_FIELD + BLOCK), device=selected)
    frame = torch.zeros(1, dtype=torch.int32, device=selected)
    sequence = torch.zeros(1, dtype=torch.int64, device=selected)
    cursor = lambda: GraphBufferPool(selected, chunk_bytes=2 << 20).cursor()
    with PreparationSession(device=selected, autotune=False) as session:
        arena = ScratchArena(selected)
        decoder, metadata = prefill(weights, pool, ids, selected, blocks=blocks, session=session, workspace=arena,
                                    name="glimmer.dflash.prefill")
        context = PreparedContext(weights, decoder.auxiliary, metadata, pool, session=session,
                                  name="glimmer.dflash.prefill.context", workspace=arena, runtime=None,
                                  coordinator=None, buffers=cursor())
        owner = SpeculativePreparation(weights, pool, requests, inputs, Scratch.allocate(weights.config, shard, BLOCK, BLOCK, selected),
            completion=completion, frame=frame, sequence=sequence, session=session, workspace=arena,
            buffers=cursor(), decoder_buffers=cursor(), context_buffers=cursor())
        operation = PreparedParallelDraftRound(owner, 1, BLOCK, "glimmer.dflash.speculative1x16", buffers=cursor(),
                                               target_token_capacity=BLOCK)
        operation.prepare()
        assert operation.draft.selector is not None
        # The target's own continuation, one decode step at a time, after the round (decode writes the ring
        # slots of positions N.. that verification still reads for a prompt past the window).
        decode_metadata = Metadata.allocate(1, 1, blocks, selected)
        decode = PreparedDecoder(weights, Scratch.allocate(weights.config, shard, 1, 1, selected), decode_metadata, pool,
                                 mode="decode", runtime=None, embedding_runtime=None, vocabulary_runtime=None,
                                 coordinator=None, session=session, name="glimmer.dflash.decode", workspace=arena)
        arena.finalize()
        pool.clear_for_preparation()
        decoder()
        context(decoder.auxiliary, metadata.lengths)
        torch.cuda.synchronize()
        pending = int(decoder.scratch.logits[0].argmax())
        taps = decoder.auxiliary[:tokens].clone()
        # The target's own greedy continuation before the round, with the continuation row and the
        # full-layer pages restored afterwards: a round's commit would otherwise leave committed block
        # positions in ring slots a decode behind the commit still attends.
        row_before, pages_before = pool.backing[1].clone(), {name: value.clone() for name, value in pool.pages.items()}
        greedy, token = [], pending
        for step in range(BLOCK - 1):
            decode_metadata.set(torch.tensor([tokens + step], device=selected), torch.tensor([1], dtype=torch.int32, device=selected),
                                torch.tensor([1], device=selected), torch.arange(blocks, dtype=torch.int32, device=selected)[None],
                                torch.tensor([[token]], device=selected), torch.tensor([1], dtype=torch.int32, device=selected))
            decode()
            torch.cuda.synchronize()
            if step == 0:
                decode_anchor_logits = decode.scratch.logits[0].clone()
            token = int(decode.scratch.logits[0].argmax())
            greedy.append(token)
        pool.backing[1].copy_(row_before)
        for name, value in pages_before.items():
            pool.pages[name].copy_(value)
        requests.admit(0, RequestAdmission(generation=1, seed=2718, max_new_tokens=64, top_k=20, top_p=.95), ids,
                       consumed_tokens=tokens)
        requests.pending_token[0] = pending
        inputs.live_rows.fill_(1)
        inputs.row_slot_ids.zero_()
        inputs.row_generations.fill_(1)
        inputs.request_state_rows[0].fill_(1)
        inputs.page_tables[0].copy_(torch.arange(blocks, dtype=torch.int32, device=selected))
        inputs.query_lengths.fill_(BLOCK)
        operation()
        torch.cuda.synchronize()
        target, _, verify, commit = operation.operations
        candidates = operation.candidates[0].tolist()
        verify_logits = operation.decoder.scratch.logits.view(BLOCK, -1).clone()
        target_argmax = target.probabilities[0].argmax(-1).tolist()
        draft_ids = operation.drafts[0]
        draft_at = operation.probabilities[0].gather(-1, draft_ids[:, None])[:, 0]
        target_at = target.probabilities[0, :-1].gather(-1, draft_ids[:, None])[:, 0]
        acceptance = dict(
            candidates=candidates, verify_token_ids=operation.metadata.token_ids.tolist(),
            verify_positions=operation.metadata.positions.tolist(), verify_lengths=operation.metadata.lengths.tolist(),
            pending_token=int(requests.pending_token[0]), query_lengths=inputs.query_lengths.tolist(),
            parameters=requests.parameters[0].tolist(), top_k=int(requests.top_k[0]),
            verify_logits_argmax=verify_logits.argmax(-1).tolist(),
            target_argmax_equals_next_candidate=[a == b for a, b in zip(target_argmax[:-1], candidates[1:])],
            target_valid=target.valid[0].tolist(), target_mass=target.probabilities[0].sum(-1).tolist(),
            draft_valid=operation.valid[0].tolist(), draft_mass=operation.probabilities[0].sum(-1).tolist(),
            draft_prob_at_draft=[round(p, 4) for p in draft_at.tolist()],
            target_prob_at_draft=[round(p, 4) for p in target_at.tolist()],
            verify_valid=verify.output.valid.tolist(), independent_accepted=verify.independent_accepted.tolist(),
            emitted_drafts=verify.emitted_drafts.tolist(), accepted_drafts=verify.output.accepted_drafts.tolist(),
            offered_drafts=verify.output.offered_drafts.tolist(), proposed_count=verify.output.proposed_count.tolist(),
            output_token_ids=verify.output.token_ids[0].tolist(), survival=[round(s, 4) for s in verify.output.survival[0].tolist()],
            commit_emitted=commit.emitted_count.tolist(), commit_accepted=commit.accepted_count.tolist(),
            status=requests.status.tolist(), error_code=requests.error_code.tolist())
        print(acceptance, flush=True)
        assert not requests.error_code.count_nonzero(), requests.error_code.tolist()
        emitted = int(operation.operations[-1].emitted_count[0])
        assert 1 <= emitted <= BLOCK
        assert int(requests.consumed_tokens[0]) == tokens + emitted
        assert torch.equal(operation.draft_metadata.token_ids[1:], torch.full((BLOCK - 1,), draft.noise_token, device=selected))
        assert int(operation.draft_metadata.token_ids[0]) == pending
        proposed_hidden = operation.draft.scratch.hidden.clone()
        proposed_logits = operation.draft.scratch.logits.clone()
        ring = pool.reserved_region("draft0.key")[1]
        for position in range(tokens, tokens + emitted):
            assert ring[position % draft.attention.window].view(torch.uint8).count_nonzero(), position
        block_positions = operation.draft_metadata.positions.tolist()
        context_positions = (int(metadata.positions[0]), int(metadata.positions[tokens - 1]))
        print(dict(verify_vs_decode_anchor_logits=relative_l2(verify_logits[0], decode_anchor_logits),
                   verify_anchor_argmax=int(verify_logits[0].argmax()), decode_anchor_argmax=int(decode_anchor_logits.argmax())),
              flush=True)
        # The prepared selector's greedy walk over the prepared rows, through the component the round used.
        proposed, predecessor = [], torch.tensor([pending], device=selected)
        for position in range(1, BLOCK):
            unary = proposed_logits[position:position + 1].clone()
            operation.draft.selector(unary, predecessor, position)
            predecessor = unary.argmax(-1)
            proposed.append(int(predecessor))

    block_ids = torch.full((BLOCK,), draft.noise_token, dtype=torch.int64, device=selected)
    block_ids[0] = pending
    window = draft.attention.window
    expected_hidden, expected_logits, oracle = reference(weights, taps[tokens - window:], block_ids,
        torch.arange(tokens, tokens + BLOCK, device=selected), torch.arange(tokens - window, tokens, device=selected))
    report = {name: dict(hidden=relative_l2(proposed_hidden[span], expected_hidden[span]),
                         logits=relative_l2(proposed_logits[span], expected_logits[span]),
                         top1=float((proposed_logits[span].argmax(-1) == expected_logits[span].argmax(-1)).float().mean()))
              for name, span in (("anchor", slice(0, 1)), ("proposals", slice(1, BLOCK)))}
    print(dict(tokens=tokens, pending=pending, emitted=emitted, width=BLOCK, relative_l2_bound=RELATIVE_L2, **report), flush=True)
    for name, errors in report.items():
        assert errors["hidden"] <= HIDDEN_RELATIVE_L2 and errors["logits"] <= RELATIVE_L2, (name, errors)

    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(snapshot)
    sampled = draft_ids.tolist()

    def prefix(candidates):
        return next((i for i, (a, b) in enumerate(zip(candidates, greedy, strict=True)) if a != b), len(greedy))

    print(dict(pending=tokenizer.decode([pending]), prepared_walk=proposed, prepared_text=tokenizer.decode(proposed),
               oracle_walk=oracle, oracle_text=tokenizer.decode(oracle), sampled_drafts=sampled,
               sampled_text=tokenizer.decode(sampled), target_greedy=greedy, target_text=tokenizer.decode(greedy),
               prepared_matches_greedy=prefix(proposed), oracle_matches_greedy=prefix(oracle),
               sampled_matches_greedy=prefix(sampled), prepared_equals_oracle=sum(a == b for a, b in zip(proposed, oracle)),
               block_positions=block_positions, context_positions=context_positions, tap_layers=draft.target_layers,
               tap_convention="output of decoder layer index i: PreparedDecoder taps after layer(hidden) (decoder.py); "
                              "the oracle reads the same rows through fc and hidden_norm; noise embedded without "
                              "embed_norm as PreparedShardedEmbedding reads the raw table row"), flush=True)
    assert block_positions == list(range(tokens, tokens + BLOCK)) and context_positions == (0, tokens - 1)
    assert prefix(proposed) >= 1, "the first proposal must be the target's own next token"
    # Proposals that equal the target's greedy token carry near-unit draft and target probability, so
    # stochastic verification accepts them; a round that emits fewer is a readout or verification defect.
    assert emitted >= 1 + min(prefix(proposed), 4), (emitted, prefix(proposed))
