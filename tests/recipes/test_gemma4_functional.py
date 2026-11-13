"""CPU dataflow checks for Gemma's runtime metadata and local outputs."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.recipes.gemma4.attention import PreparedAttention, cache_bindings, declaration_tensor, rms_norm
from ignition.recipes.gemma4.decoder import PreparedDecoder, Scratch, SoftcappedResume
from ignition.recipes.gemma4.speculative import PreparedAssistant


def test_declaration_examples_share_shapes_without_aliasing_operator_roles():
    examples = {}
    config = SimpleNamespace(hidden=16, vocabulary=32, vocabulary_partition=lambda shard: (0, 32, 32))
    first = Scratch.allocate(128, 1, config, device="cpu", examples=examples)
    second = Scratch.allocate(128, 1, config, device="cpu", examples=examples)
    assert first.activations is second.activations
    assert first.local_logits is second.local_logits
    assert first.logits is not first.local_logits
    left = declaration_tensor(examples, "layer.gate", (128, 64), torch.bfloat16, "cpu")
    right = declaration_tensor(examples, "layer.up", (128, 64), torch.bfloat16, "cpu")
    assert left is declaration_tensor(examples, "layer.gate", (128, 64), torch.bfloat16, "cpu")
    assert left is not right


@pytest.mark.parametrize("sliding", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_attention_passes_local_queries_and_runtime_cache_operands(sliding, shared):
    attention = PreparedAttention.__new__(PreparedAttention)
    attention.heads, attention.kv_heads, attention.dim, attention.q_cap = 2, 1, 4, 4
    attention.shared_kv, attention.is_sliding = shared, sliding
    attention.eps = 1e-6
    attention.q_norm = attention.k_norm = torch.ones(4)
    attention.q_proj = lambda source: source.clone()
    attention.k_proj = lambda source: source[:, :4].clone()
    attention.v_proj = None
    attention.o_proj = lambda source: source
    attention.reduce = lambda source: source
    written, queries = [], []
    def writer(key, value, slots):
        written.append((key.clone(), value.clone(), slots.clone()))
    attention.writer = attention.writer_v = attention.writer_k = writer
    metadata = SimpleNamespace(page_table=torch.tensor([[3, 4]]), query_start_loc=torch.tensor([0, 2]))
    bindings = SimpleNamespace(slots=torch.tensor([384, 385]), lengths=torch.tensor([2]))

    def attend(query, *, page_table, cache_seqlens, cu_seqlens_q):
        assert page_table is metadata.page_table
        assert cache_seqlens is bindings.lengths
        assert cu_seqlens_q is metadata.query_start_loc
        queries.append(query.clone())
        return query

    attention.attention = attend
    attention.attention_lo = lambda query, **kwargs: attend(query, **kwargs)[..., :2]
    attention.attention_hi = lambda query, **kwargs: attend(query, **kwargs)[..., 2:]
    source = torch.arange(16).reshape(2, 8).to(torch.bfloat16)
    output = attention(source, torch.ones(2, 4), torch.zeros(2, 4),
                       metadata=metadata, cache_metadata=bindings)
    expected = rms_norm(source.view(2, 2, 4), attention.q_norm, attention.eps)
    assert torch.equal(output, expected.flatten(1))
    assert len(written) == (0 if shared else 1 if sliding else 2)
    assert all(torch.equal(row[2], bindings.slots) for row in written)
    if not sliding:
        assert all(query[2:].count_nonzero() == 0 for query in queries)


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_cache_bindings_follow_runtime_lengths_and_pages(batch):
    starts = torch.arange(batch, dtype=torch.int64) * 128 + 127
    lengths = torch.ones(batch, dtype=torch.int32)
    states = torch.arange(1, batch + 1, dtype=torch.int64)
    pages = torch.arange(batch * 4, dtype=torch.int32).view(batch, 4)
    tokens = torch.ones(batch, 1, dtype=torch.int64)
    live = torch.tensor([batch], dtype=torch.int32)
    first = Metadata.pack(starts, lengths, states, pages, tokens, live,
                          token_capacity=batch, sample_count=batch)
    pack_cache = torch.compile(cache_bindings, backend="eager", fullgraph=True)
    bindings = pack_cache(first)
    assert torch.equal(bindings.lengths, (starts + 1).int())
    expected_pages = pages.gather(1, (starts // 128)[:, None]).flatten()
    assert torch.equal(bindings.slots, expected_pages.long() * 128 + 127)
    lengths[-1] = 0
    second = Metadata.pack(starts + 1, lengths, states, pages.flip(0), tokens, live,
                           token_capacity=batch, sample_count=batch)
    updated = pack_cache(second)
    assert updated.slots[-1] == -1
    assert updated.lengths[-1] == 0
    assert bindings.lengths[-1] > 0


def test_decoder_uses_runtime_ids_and_softcaps_prompt_and_completion_logits():
    decoder = PreparedDecoder.__new__(PreparedDecoder)
    decoder.sliding_inv_freq = decoder.full_inv_freq = torch.ones(1)
    decoder.embedding = lambda *, token_ids, num_rows: token_ids[:, None].expand(-1, 4).to(torch.bfloat16)
    decoder.embed_scale = torch.tensor(1, dtype=torch.bfloat16)
    decoder.ple_embedding = None
    decoder.layers = []
    decoder.weights = SimpleNamespace(norm=torch.ones(4))
    decoder.eps, decoder.softcap = 1e-6, 0.5
    decoder.head = lambda source: source.clone()
    prompt_logits = []
    decoder.scoring = SimpleNamespace(parts=((0, 2, lambda source: source.clone(), prompt_logits.append),))
    for ids in (torch.tensor([1, 2]), torch.tensor([3, 5])):
        m = Metadata.pack(torch.tensor([0]), torch.tensor([2], dtype=torch.int32), torch.tensor([1]),
                          torch.zeros((1, 1), dtype=torch.int32), ids[None], torch.tensor([1], dtype=torch.int32),
                          token_capacity=2, sample_count=1)
        hidden, logits, auxiliary, records = decoder(m)
        expected = rms_norm(ids[:, None].expand(-1, 4).to(torch.bfloat16), torch.ones(4), decoder.eps)
        capped = (torch.tanh(expected.float() / .5) * .5).to(expected.dtype)
        assert torch.equal(hidden, expected[-1:])
        assert torch.equal(logits, capped[-1:])
        assert torch.equal(prompt_logits[-1], capped)
        assert auxiliary is None and records is None


@pytest.mark.parametrize("ordered", [False, True])
def test_assistant_returns_recurrent_projection_and_runtime_logits(ordered):
    assistant = PreparedAssistant.__new__(PreparedAssistant)
    assistant.batch = 2
    assistant.target_config = SimpleNamespace(vocabulary=8)
    assistant.draft_config = SimpleNamespace(hidden=4)
    assistant.weights = SimpleNamespace(draft=SimpleNamespace(norm=torch.ones(4)))
    assistant.sliding_inv_freq = assistant.full_inv_freq = torch.ones(1)
    assistant.embedding = lambda *, token_ids, num_rows: token_ids[:, None].expand(-1, 4).to(torch.bfloat16)
    assistant.embed_scale = torch.tensor(1, dtype=torch.bfloat16)
    assistant.pre_projection = lambda source: source[:, :4] + source[:, 4:]
    assistant.post_projection = lambda source: source * 2
    assistant.layers = []
    assistant.eps, assistant.softcap = 1e-6, None
    if ordered:
        assistant.head = None
        assistant.centroid_proj = lambda source: source[:, :2]
        assistant.top_k, assistant.vocab_per_centroid = 1, 4
        assistant.canonical_table = torch.arange(8).view(2, 4)
        assistant.head_weight = torch.arange(32).view(8, 4).to(torch.bfloat16)
    else:
        assistant.head = lambda source: source.repeat(1, 2)
    previous = torch.arange(8).view(2, 4).to(torch.bfloat16)
    args = (torch.tensor([3, 4]), previous, torch.tensor([129, 130]), torch.tensor([1, 2]),
            torch.zeros((2, 4), dtype=torch.int32), torch.tensor([2], dtype=torch.int32), torch.ones(2, dtype=torch.bool))
    logits, projected = assistant.step(0, *args)
    normalized = rms_norm(args[0][:, None].to(torch.bfloat16) + previous, torch.ones(4), assistant.eps)
    assert torch.equal(projected, normalized * 2)
    assert logits.shape == (2, 8)
    assert torch.isfinite(logits).all()
    _, changed = assistant.step(1, args[0] + 1, projected, *args[2:])
    assert not torch.equal(changed, projected)


def test_resume_softcaps_runtime_projection():
    resume = SoftcappedResume.__new__(SoftcappedResume)
    resume.state = SimpleNamespace(validate=lambda: (torch.ones(1), torch.ones(1, dtype=torch.bool)),
                                   gather=lambda totals, *, present: totals.reshape(1, 1))
    resume.validation = lambda source: source
    resume.projection = lambda hidden: hidden * 100
    observed = []
    resume.completion = SimpleNamespace(run=observed.append)
    resume.softcap = 2
    resume()
    assert torch.equal(observed[0], torch.tensor([[2.]]))
