"""CPU checks of decoder operand rows and declaration-example ownership."""

import gc
from types import SimpleNamespace
import weakref

import pytest
import torch

from ignition.recipes.deepseek41 import attention, decoder
from tests.recipes.test_deepseek41_row_shard_attention import _attention_layer, _Recorder


@pytest.mark.parametrize("policy", ["prefill-only", "gather-refused", "attention-sharded"])
def test_router_input_matches_attention_sharding(monkeypatch, policy):
    scratch = decoder.Scratch.allocate(1024, 8, 2, "cpu", row_shard=SimpleNamespace(rank=1),
                                       attention_shard=True)
    shard = scratch.row_shard
    layer, _ = _attention_layer(scratch)
    monkeypatch.setattr(decoder.PreparedGatherShards, "fits",
                        staticmethod(lambda *args: policy != "gather-refused"))
    layer.attention_shard = (policy != "prefill-only"
                             and decoder.attention_shard_ready(scratch, shard, None))
    if not layer.attention_shard:
        layer.attention_gather = _Recorder(scratch.attention_hc.normalized)
        layer.attention = _Recorder((scratch.attention.output, {}, None, {}))
        layer.attention_reduce = _Recorder(shard.attention_output)
        layer.router_projection = _Recorder(scratch.router_logits)
        layer.router_gather = None
        layer.route_gather = None
        layer.router = _Recorder(scratch.router)
    layer._run((shard.residual,), shard.pre_mix,
               SimpleNamespace(token_ids=torch.zeros(1024, dtype=torch.int64)), None,
               (scratch.attention.selected, None, None))
    source, = layer.router_projection.calls[0]
    expected = shard.ffn_hc.normalized if layer.attention_shard else scratch.ffn_hc.normalized
    assert source is expected
    assert source.shape == (512 if layer.attention_shard else 1024, 5120)
    assert layer.experts.calls[0][0] is scratch.ffn_hc.normalized


def _component(*args, **kwargs):
    # GPU plan declarations are replaced, not the owning constructors or finalizers.
    return SimpleNamespace(plan=object(), functional=lambda **options: None)


def _prepare_layer(monkeypatch, scratch, metadata, sharded):
    for name in ("PreparedMHC", "PreparedAttention", "PreparedAllReduce", "PreparedReduceScatter",
                 "PreparedChunkedGatherShards", "PreparedGatherShards", "UnquantizedProjection",
                 "PreparedRoutedExperts", "PreparedMLP", "PreparedMHCPost", "PreparedRouter"):
        monkeypatch.setattr(decoder, name, _component)
    monkeypatch.setattr(decoder, "_mhc_weights", lambda *args: None)
    text = dict(rms_norm_eps=1e-6, hc_eps=1e-6, hc_sinkhorn_iters=20, scoring_func="sigmoid",
                norm_topk_prob=True, routed_scaling_factor=1., swiglu_limit=10.)
    weights = SimpleNamespace(engram=None, attention=None, experts=SimpleNamespace(
        vision_correction=None, router=None, correction=None, experts=None, shared=None))
    return decoder.PreparedLayer(0, weights, None, scratch, metadata,
        SimpleNamespace(config=SimpleNamespace(text=text)), None,
        shard=SimpleNamespace(world_size=2), mode="prefill", runtime=None, coordinator=None,
        session=object(), name="layer", workspace=SimpleNamespace(overlap=False), carry=False,
        defer_post=False, row_shard=scratch.row_shard if sharded else None, attention_shard=sharded)


@pytest.mark.parametrize("sharded", [False, True])
def test_layer_and_decoder_release_attention_examples(monkeypatch, sharded):
    scratch = decoder.Scratch.allocate(1024, 8, 2, "cpu",
        row_shard=SimpleNamespace(rank=1) if sharded else None, attention_shard=sharded)
    examples = [weakref.ref(scratch.ffn_hc.normalized)]
    if sharded:
        examples += [weakref.ref(scratch.row_shard.window_values),
                     weakref.ref(scratch.row_shard.attention.window_values),
                     weakref.ref(scratch.row_shard.router_logits)]
    metadata = SimpleNamespace(token_ids=torch.zeros(1024, dtype=torch.int64))
    layer = _prepare_layer(monkeypatch, scratch, metadata, sharded)
    prepared = object.__new__(decoder.PreparedDecoder)
    prepared.row_shard = scratch.row_shard
    prepared.scratch = prepared.output_scratch = scratch
    prepared.layers = [layer]
    layer.release_examples()
    prepared.release_outputs()
    del scratch
    gc.collect()
    assert all(example() is None for example in examples)
    if sharded:
        assert torch.equal(layer.row_shard.rows_index, torch.arange(512, 1024))
        assert layer.row_shard.rows_table is prepared.row_shard.rows_table


@pytest.mark.parametrize("sharded", [False, True])
def test_attention_releases_window_examples(monkeypatch, sharded):
    for name in ("QuantizedProjection", "PreparedRMSNorm", "PreparedGroupedOutputProjection",
                 "PreparedCompressedCacheWriter", "PreparedCompressedMLA", "PreparedGatherShards"):
        monkeypatch.setattr(attention, name, _component)
    monkeypatch.setattr(attention.sm103_projection, "is_sm103", lambda device: False)
    tokens, rows = 1024, 512 if sharded else 1024
    scratch = attention.AttentionScratch.allocate(tokens, 2, "cpu", candidate_tokens=1,
                                                  shard_rows=rows if sharded else None)
    target = torch.zeros(tokens, 512) if sharded else None
    examples = [weakref.ref(scratch.window_values)]
    if target is not None:
        examples.append(weakref.ref(target))
    metadata = SimpleNamespace(positions=torch.arange(tokens), starts=torch.zeros(1, dtype=torch.int64),
        token_page_table=torch.zeros(tokens, 1, dtype=torch.int32), max_query_tokens=tokens)
    monkeypatch.setattr(attention, "AttentionIndices", lambda *args, **kwargs: SimpleNamespace(
        window_slots=torch.zeros(tokens, dtype=torch.int64),
        window_indices=torch.zeros(tokens, 128, dtype=torch.int32),
        window_lengths=torch.zeros(tokens, dtype=torch.int32), functional=lambda: None))
    config = SimpleNamespace(text={"compress_ratios": [0], "rms_norm_eps": 1e-6,
                                   "candidate_source_layer_id": 0}, source=lambda layer: None)
    pool = SimpleNamespace(config=config, window=lambda *args, **kwargs: torch.zeros(1, 128 * 528,
                                                                                 dtype=torch.uint8))
    weights = dict.fromkeys(("qkv_a", "q_b", "q_b_full", "output", "output_full", "sink", "sink_full",
                             "q_norm", "kv_norm"))
    shard = SimpleNamespace(rows_index=torch.arange(512, 1024), attention=scratch,
                            window_values=target) if sharded else None
    prepared = attention.PreparedAttention(0, weights, torch.zeros(rows, 5120), scratch, metadata,
        pool, None, mode="prefill", world=2, session=object(), name="attention", workspace=None,
        row_shard=shard, window_target=target)
    prepared.release_examples()
    del scratch, target, shard
    gc.collect()
    assert all(example() is None for example in examples)
    if sharded:
        assert torch.equal(prepared.rows_index, torch.arange(512, 1024))
