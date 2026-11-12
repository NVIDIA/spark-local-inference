# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Row-sharded prefill attention replicates every head and keeps each cache complete."""

from types import SimpleNamespace

import pytest
import torch


def test_attention_weight_extents_partition_per_rank_and_replicate_every_group():
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.deepseek41.weights import attention_weight_extents

    text = {"head_dim": 512, "num_attention_heads": 64, "o_groups": 8, "o_lora_rank": 1024}
    extents = [attention_weight_extents(text, TensorParallelShard(rank, 4)) for rank in range(4)]
    # The tensor-parallel partition keeps 16 heads and 2 output groups per rank: disjoint and whole.
    assert [extent.heads for extent in extents] == [
        (0, 8192), (8192, 16384), (16384, 24576), (24576, 32768)]
    assert [extent.groups for extent in extents] == [
        (0, 2048), (2048, 4096), (4096, 6144), (6144, 8192)]
    assert [extent.sink for extent in extents] == [(0, 16), (16, 32), (32, 48), (48, 64)]
    # A row-sharded rank replicates all 64 heads, all 8 output groups, and all 64 sinks.
    assert all(extent.all_heads == (0, 32768) for extent in extents)
    assert all(extent.all_groups == (0, 8192) for extent in extents)
    assert all(extent.all_sink == (0, 64) for extent in extents)


def test_replicated_attention_weights_cost_the_documented_bytes_per_rank():
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.deepseek41.weights import attention_weight_extents

    text = {"head_dim": 512, "num_attention_heads": 64, "o_groups": 8, "o_lora_rank": 1024}
    hidden, q_lora, layers = 5120, 1280, 40
    extent = attention_weight_extents(text, TensorParallelShard(0, 4))
    group_width = text["num_attention_heads"] * text["head_dim"] // text["o_groups"]
    extra_heads = extent.all_heads[1] - extent.all_heads[0] - extent.heads[1] + extent.heads[0]
    extra_groups = extent.all_groups[1] - extent.all_groups[0] - extent.groups[1] + extent.groups[0]
    # wq_b gains rows, wo_a gains group rows at the fixed 4096-wide group input, and wo_b gains
    # its group columns. MXFP8 holds one value byte per weight and one scale byte per 32 values.
    per_layer = extra_heads * q_lora + extra_groups * group_width + extra_groups * hidden
    delta = per_layer * layers * (1 + 1 / 32)
    assert 3.5e9 < delta < 3.8e9, delta


def _surfaces(monkeypatch, *, abi=11, pipe_bytes=1 << 26):
    import b12x.comm.roce._proxy as proxy
    from ignition.components import collectives

    monkeypatch.setattr(proxy, "ABI_VERSION", abi)
    monkeypatch.setattr(collectives, "_roce", lambda runtime: True)
    return SimpleNamespace(world_size=2, pipe_bytes=pipe_bytes, _group=object(),
        should_reduce_scatter=lambda inp: True, should_gather_shards=lambda inp: True)


def test_attention_scratch_keeps_the_full_row_layout_unless_a_shard_is_asked_for():
    from ignition.recipes.deepseek41.attention import AttentionScratch

    plain = AttentionScratch.allocate(1024, 4, "cpu", candidate_tokens=16)
    # Without a shard: one rank's 16 heads and a full-row query side.
    assert plain.query.shape == plain.attended.shape == (1024, 16, 512)
    assert plain.projected.shape == (1024, 1792) and plain.window_values.shape == (1024, 512)
    assert plain.index_keys.shape == plain.normalized_keys.shape == (1024, 128)


def test_attention_scratch_shard_holds_every_head_and_full_row_index_keys():
    from ignition.recipes.deepseek41.attention import AttentionScratch

    shard = AttentionScratch.allocate(1024, 4, "cpu", candidate_tokens=16, shard_rows=256)
    assert shard.query.shape == shard.attended.shape == (256, 64, 512)
    assert shard.projected.shape == (256, 1792) and shard.window_values.shape == (256, 512)
    assert shard.query_low.shape == (256, 1280) and shard.selected.shape == (256, 512)
    assert shard.candidates.shape == (16, 16384) and shard.candidates.dtype is torch.int32
    # The compressor still runs over every row, so its key planes stay full-row.
    assert shard.index_keys.shape == shard.normalized_keys.shape == (1024, 128)
    narrowed = shard.rows(128, 1024)
    assert narrowed.query.shape == (128, 64, 512) and narrowed.selected.shape == (128, 512)
    # A narrower graph keeps the key planes at that graph's row count, which indexes its slots.
    assert narrowed.index_keys.shape == narrowed.normalized_keys.shape == (1024, 128)
    assert shard.rows(128, 512).index_keys.shape == (512, 128)
    with pytest.raises(ValueError, match="attention scratch rows"):
        shard.rows(257, 1024)
    with pytest.raises(ValueError, match="attention scratch tokens"):
        shard.rows(128, 1025)


def test_row_shard_allocates_attention_and_full_row_gather_targets_only_when_asked():
    from ignition.recipes.deepseek41.decoder import RowShard

    plain = RowShard.allocate(2048, 2, 1, "cpu", candidate_tokens=16)
    assert (plain.rows, plain.start) == (1024, 1024) and plain.attention is None
    assert plain.window_values is None and plain.router_logits is None

    asked = RowShard.allocate(2048, 2, 1, "cpu", candidate_tokens=16, attention=True)
    assert (asked.rows, asked.start, asked.rows_index[0].item()) == (1024, 1024, 1024)
    assert asked.attention.query.shape == (1024, 64, 512)
    # The KV latent target spans every row: it is what the gathered window write stores.
    assert asked.window_values.shape == (2048, 512)
    assert asked.router_logits.shape == (1024, 384) and asked.router_logits.dtype is torch.float32

    narrowed = asked.graph(1024)
    # A narrower graph gives every rank a shorter block, so rank 1 starts at 512, not at 1024.
    assert (narrowed.rows, narrowed.start) == (512, 512)
    assert narrowed.rows_index.tolist() == list(range(512, 1024))
    assert narrowed.attention.query.shape == (512, 64, 512)
    # The narrowed graph's index keys cover its own full rows, and its KV target all of them.
    assert narrowed.attention.index_keys.shape == (1024, 128)
    assert narrowed.window_values.shape == (1024, 512) and narrowed.router_logits.shape == (512, 384)
    assert asked.graph(1023) is None


@pytest.mark.parametrize("tokens", [8192, 6144, 4096, 2048])
@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_a_narrower_graph_keeps_each_rank_inside_its_own_rows(tokens, rank):
    from ignition.recipes.deepseek41.decoder import RowShard

    captured = RowShard.allocate(8192, 4, rank, "cpu", candidate_tokens=16, attention=True)
    shard = captured.graph(tokens)
    rows = tokens // 4
    # One prefill chunk size captures the layout; every smaller graph re-derives this rank's block
    # from the graph's own row count, so an attention slice never runs past the graph's rows.
    assert (shard.rows, shard.start) == (rows, rank * rows)
    assert shard.rows_index.tolist() == list(range(rank * rows, (rank + 1) * rows))
    assert captured.graph(tokens).start == shard.start


@pytest.mark.parametrize("abi,ask,ready", [
    (11, True, True), (10, True, False), (11, False, False)])
def test_attention_shard_ready_requires_the_abi_and_the_shard_it_gathers(monkeypatch, abi, ask, ready):
    from ignition.recipes.deepseek41.decoder import RowShard, attention_shard_ready

    runtime = _surfaces(monkeypatch, abi=abi)
    tokens = 1024
    shard = RowShard.allocate(tokens, 2, 0, "cpu", candidate_tokens=16, attention=ask)
    scratch = SimpleNamespace(router_logits=torch.zeros(tokens, 384, dtype=torch.float32),
                              attention=SimpleNamespace(selected=torch.zeros(tokens, 512, dtype=torch.int32)))
    assert attention_shard_ready(scratch, shard, runtime) is ready


def test_attention_shard_ready_rejects_a_gather_that_does_not_fit_the_pipe(monkeypatch):
    from ignition.recipes.deepseek41.decoder import RowShard, attention_shard_ready

    shard = RowShard.allocate(1024, 2, 0, "cpu", candidate_tokens=16, attention=True)
    scratch = SimpleNamespace(router_logits=torch.zeros(1024, 384, dtype=torch.float32),
                              attention=SimpleNamespace(selected=torch.zeros(1024, 512, dtype=torch.int32)))
    # Every sharded row is gathered whole; the widest one decides whether the runtime serves it.
    runtime = _surfaces(monkeypatch)
    assert attention_shard_ready(scratch, shard, runtime) is True
    assert attention_shard_ready(scratch, shard, _surfaces(monkeypatch, pipe_bytes=1 << 12)) is False
    assert attention_shard_ready(scratch, shard, _surfaces(monkeypatch, abi=10)) is False
    monkeypatch.setattr("ignition.components.collectives._roce", lambda runtime: False)
    assert attention_shard_ready(scratch, shard, runtime) is False


def _indexing_config(sources, candidate):
    return SimpleNamespace(decoder_start=candidate, text={"candidate_source_layer_id": candidate},
        index_source=lambda layer: next((i for i in reversed(sources) if i <= layer), None))


def _scheduled_layers(engram=(1, 14)):
    return [SimpleNamespace(engram=object() if index in engram else None) for index in range(40)]


def test_selection_moves_follow_the_row_transition_around_the_engram_layer():
    """Layers 15-18 read layer 14's latent pages, so they must carry its selection state.

    A selection is not recomputed between index sources: the value one layer returns is the one the
    next layer reads, and its rows are those of the layer that wrote it. Layer 14 runs on full rows
    and layer 15 on a shard, so layer 15 takes this rank's rows; layer 18 crosses back and gathers
    them again for the packed suffix.
    """
    from ignition.recipes.deepseek41.decoder import SelectionMove, selection_moves, shard_schedule

    config = _indexing_config([2, 8, 14, 20, 24, 28, 32, 36], 20)
    packed = selection_moves(config, shard_schedule(_scheduled_layers(), 18, object()), attention_shard=True)
    # A source reads no carried selection, so layer 14 only writes its own, in full rows.
    assert all(move is None for move in packed[:15])
    assert packed[15] == SelectionMove(14, "full_to_shard")
    assert all(move is None for move in packed[16:18])
    assert packed[18] == SelectionMove(14, "shard_to_full")
    assert all(move is None for move in packed[19:])
    # Attention sharding is the only thing that creates the two representations.
    assert selection_moves(config, shard_schedule(_scheduled_layers(), 18, object()),
                           attention_shard=False) == (None,) * 40


def test_a_full_row_consumer_gathers_a_sharded_sources_selection():
    """Scored prefill has no packed suffix, so its last full-row layer consumes a shard's state."""
    from ignition.recipes.deepseek41.decoder import SelectionMove, selection_moves, shard_schedule

    config = _indexing_config([2, 8, 14, 20, 24, 28, 32, 36], 20)
    scored = selection_moves(config, shard_schedule(_scheduled_layers(), 39, object()), attention_shard=True)
    assert scored[15] == SelectionMove(14, "full_to_shard")
    assert all(move is None for move in scored[16:39])
    assert scored[39] == SelectionMove(36, "shard_to_full")


def test_a_shard_takes_its_own_rows_of_the_carried_selection():
    """Rows, candidate rows and counts all move by the shard's row numbers, not from row zero."""
    from ignition.recipes.deepseek41 import attention, decoder

    tokens, world, rows = 128, 2, 64
    full = attention.AttentionScratch.allocate(tokens, world, "cpu", candidate_tokens=tokens)
    shard = attention.AttentionScratch.allocate(tokens, world, "cpu", candidate_tokens=tokens, shard_rows=rows)
    full.selected.copy_(torch.arange(tokens, dtype=torch.int32)[:, None].expand(tokens, 512))
    full.candidates.copy_(torch.arange(tokens, dtype=torch.int32)[:, None].expand(tokens, 16384))
    full.candidate_counts.copy_(torch.arange(tokens, dtype=torch.int32))
    prepared = decoder.PreparedSelectionRefill(
        decoder.SelectionMove(14, "full_to_shard", "full_to_shard"), full=full, shard=shard,
        rows_index=torch.arange(rows) + rows, runtime=None, session=None, coordinator=None, name="test")
    selected, candidates, counts = prepared((full.selected, full.candidates, full.candidate_counts))
    assert torch.equal(selected, full.selected[rows:])
    assert torch.equal(candidates, full.candidates[rows:])
    assert torch.equal(counts, full.candidate_counts[rows:])
    assert not torch.equal(selected, full.selected[:rows])


def test_a_sharded_candidate_shortlist_is_refused_rather_than_read_stale():
    from ignition.recipes.deepseek41 import attention, decoder

    full = attention.AttentionScratch.allocate(64, 2, "cpu", candidate_tokens=64)
    shard = attention.AttentionScratch.allocate(64, 2, "cpu", candidate_tokens=64, shard_rows=32)
    with pytest.raises(ValueError, match="candidate shortlist"):
        decoder.PreparedSelectionRefill(decoder.SelectionMove(20, None, "shard_to_full"),
            full=full, shard=shard, rows_index=torch.arange(32) + 32, runtime=None, session=None,
            coordinator=None, name="test")


class _GatheredRows:
    """Stand-in for the shard gather: this rank's rows land in its block, the others keep theirs."""

    def __init__(self, shard, source, output, **options):
        self.shard, self.source, self.output, self.plan = shard, source, output, object()

    def functional(self, **options):
        return self

    def __call__(self, source):
        gathered = self.output.clone()
        gathered[self.shard.start:self.shard.start + self.shard.rows] = source
        return gathered


def _stub(*args, **kwargs):
    return SimpleNamespace(plan=object(), functional=lambda **options: None)


def _full_row_layer(monkeypatch, scratch, metadata, *, index, selection_move, shard):
    """`PreparedLayer` as the decoder builds the full-row layer that crosses back off the shard."""
    from ignition.recipes.deepseek41 import decoder

    for name in ("PreparedMHC", "PreparedAttention", "PreparedAllReduce", "PreparedReduceScatter",
                 "PreparedChunkedGatherShards", "UnquantizedProjection", "PreparedRoutedExperts",
                 "PreparedMLP", "PreparedMHCPost", "PreparedRouter"):
        monkeypatch.setattr(decoder, name, _stub)
    monkeypatch.setattr(decoder, "PreparedGatherShards",
                        lambda source, output, **options: _GatheredRows(shard, source, output, **options))
    monkeypatch.setattr(decoder, "_mhc_weights", lambda *args: None)
    text = dict(rms_norm_eps=1e-6, hc_eps=1e-6, hc_sinkhorn_iters=20, scoring_func="sigmoid",
                norm_topk_prob=True, routed_scaling_factor=1., swiglu_limit=10.)
    weights = SimpleNamespace(engram=None, attention=None, experts=SimpleNamespace(
        vision_correction=None, router=None, correction=None, experts=None, shared=None))
    return decoder.PreparedLayer(index, weights, None, scratch, metadata,
        SimpleNamespace(config=SimpleNamespace(text=text)), None,
        shard=SimpleNamespace(world_size=2), mode="prefill", runtime=None, coordinator=None,
        session=object(), name="layer", workspace=SimpleNamespace(overlap=False), carry=False,
        defer_post=False, row_shard=None, attention_shard=False, selection_move=selection_move)


@pytest.mark.parametrize("scored", [False, True])
def test_the_layer_that_crosses_back_to_full_rows_gathers_the_carried_selection(monkeypatch, scored):
    """A full-row consumer gathers the selection through the decoder's carried shard.

    Packed and scored schedules cross back to full rows after different index sources.
    The consuming layer has no shard of its own, but its attention needs full-row selection.
    """
    from ignition.recipes.deepseek41 import decoder

    config = _indexing_config([2, 8, 14, 20, 24, 28, 32, 36], 20)
    crossing = 39 if scored else 18
    schedule = decoder.shard_schedule(_scheduled_layers(), crossing, object())
    moves = decoder.selection_moves(config, schedule, attention_shard=True)
    assert schedule[crossing][0] is False
    assert moves[crossing] == decoder.SelectionMove(14 if not scored else 36, "shard_to_full")

    tokens, world = 1024, 2
    scratch = decoder.Scratch.allocate(tokens, 8, world, "cpu", row_shard=SimpleNamespace(rank=1),
                                       attention_shard=True)
    shard = scratch.row_shard
    metadata = SimpleNamespace(token_ids=torch.empty(tokens),
                               live_counts=torch.tensor([tokens, 1], dtype=torch.int32))
    layer = _full_row_layer(monkeypatch, scratch, metadata, index=crossing,
                            selection_move=moves[crossing], shard=shard)
    assert layer.selection_move is not None
    gather = layer.selection_move.gather
    # The move reads the decoder's shard and writes the full-row selection this layer's attention reads.
    assert gather.source.data_ptr() == shard.attention.selected.data_ptr()
    assert gather.output.data_ptr() == scratch.attention.selected.data_ptr()
    # Shape check only: the stub stands in for the collective, so this says nothing about the rows.
    selected, _, _ = layer.selection_move((shard.attention.selected, None, None))
    assert selected.shape == (tokens, 512) and selected.dtype is torch.int32


class _Recorder:
    """A stand-in for one prepared component that returns the surface the forward reads."""

    def __init__(self, value=None):
        self.value, self.calls, self.kwargs = value, [], {}

    def __call__(self, *args, **kwargs):
        self.calls.append(args)
        self.kwargs = kwargs
        return self.value

    def _run(self, *args, **kwargs):
        self.calls.append(args)
        self.kwargs = kwargs
        return self.value


def _attention_layer(scratch):
    """`PreparedLayer` over recorders, for a layer that runs attention on its own rows."""
    from ignition.recipes.deepseek41 import decoder

    shard = scratch.row_shard
    layer = object.__new__(decoder.PreparedLayer)
    layer.scratch, layer.row_shard, layer.attention_shard = scratch, shard, True
    layer.reseed, layer.selection_move, layer.engram, layer.boundary = False, None, None, None
    layer.concurrent_shared = None
    layer.post = _Recorder(shard.residual)
    layer.has_image_mask, layer.image_mask = False, None
    branch = shard.attention_hc
    layer.attention_hc = _Recorder((branch.residual, branch.post, branch.comb, branch.normalized))
    layer.attention_gather = None
    # The shard's attention returns this rank's completed rows; no reduction follows.
    attended = torch.zeros(scratch.row_shard.rows, 5120, dtype=torch.bfloat16)
    layer.attention = _Recorder((attended, {}, None, {}))
    layer.attention_reduce = None
    ffn = shard.ffn_hc
    layer.ffn_hc = _Recorder((ffn.residual, ffn.post, ffn.comb, ffn.normalized))
    layer.ffn_gather = _Recorder(scratch.ffn_hc.normalized)
    layer.router_projection = _Recorder(shard.router_logits)
    layer.router_gather = _Recorder(scratch.router_logits)
    layer.router = _Recorder(SimpleNamespace(ids=torch.zeros(1, dtype=torch.int32),
                                             weights=torch.zeros(1), selected_logits=None))
    layer.experts = _Recorder(scratch.routed)
    layer.shared = _Recorder(scratch.shared.output)
    layer.mlp_reduce = _Recorder(scratch.mlp_output)
    layer.residual_gather, layer.mix_gather = _Recorder(scratch.residual), _Recorder(scratch.pre_mix)
    return layer, attended


def test_a_sharded_layer_needs_no_attention_gather_or_reduction():
    """Attention reads this rank's branch input and its own rows are complete once it returns."""
    from ignition.recipes.deepseek41 import decoder

    scratch = decoder.Scratch.allocate(1024, 8, 2, "cpu", row_shard=SimpleNamespace(rank=1),
                                       attention_shard=True)
    shard = scratch.row_shard
    layer, attended = _attention_layer(scratch)
    layer._run((shard.residual,), shard.pre_mix, SimpleNamespace(token_ids=torch.zeros(1024, dtype=torch.int64)),
               None, (torch.full((shard.rows, 512), -1, dtype=torch.int32), None, None))
    (attention_source,) = layer.attention.calls[0][:1]
    assert attention_source.data_ptr() == shard.attention_hc.normalized.data_ptr()
    attention_output = layer.ffn_hc.calls[0][0]
    assert attention_output.data_ptr() == attended.data_ptr()
    assert attended.shape == (shard.rows, 5120)


def test_a_sharded_layer_projects_its_router_logits_and_gathers_them_to_every_row():
    """Top-k reads one row at a time, so the shard projects its rows and one gather feeds the router."""
    from ignition.recipes.deepseek41 import decoder

    scratch = decoder.Scratch.allocate(1024, 8, 2, "cpu", row_shard=SimpleNamespace(rank=1),
                                       attention_shard=True)
    shard = scratch.row_shard
    layer, _ = _attention_layer(scratch)
    layer._run((shard.residual,), shard.pre_mix, SimpleNamespace(token_ids=torch.zeros(1024, dtype=torch.int64)),
               None, (torch.full((shard.rows, 512), -1, dtype=torch.int32), None, None))
    (router_source,) = layer.router_projection.calls[0]
    assert router_source.data_ptr() == shard.ffn_hc.normalized.data_ptr()
    (logits,) = layer.router_gather.calls[0]
    # The gather moves BF16 packs of the FP32 logits it was prepared over.
    assert logits.dtype is torch.bfloat16 and logits.shape == (shard.rows, 768)
    assert layer.router.calls[0][0].data_ptr() == scratch.router_logits.data_ptr()
    # The routed experts keep every row.
    assert layer.experts.calls[0][0].data_ptr() == scratch.ffn_hc.normalized.data_ptr()
    (residual,) = layer.residual_gather.calls[0]
    assert residual.data_ptr() == shard.residual.data_ptr()


def test_compressor_gathers_values_and_gates_through_the_view_they_were_prepared_from(monkeypatch):
    from ignition.recipes.deepseek41 import attention

    values, gates = torch.zeros(16, 512), torch.zeros(16, 512)
    compressor = object.__new__(attention.PreparedCompressor)
    compressor.ratio, compressor.layer, compressor.deferred, compressor.total = 2, 0, True, 64
    compressor.weight, compressor.plan = None, SimpleNamespace(handle=None)
    compressor.metadata = SimpleNamespace(state_rows=4, starts=torch.zeros(4, dtype=torch.int64),
        query_start_loc=torch.zeros(5, dtype=torch.int32), live_counts=torch.zeros(2, dtype=torch.int32))
    compressor.pool = SimpleNamespace(region=lambda name: None)
    compressor.local_values, compressor.local_gates = values, gates
    compressor.project_values, compressor.project_gates = _Recorder(values), _Recorder(gates)
    compressor.values_gather = _Recorder(torch.zeros(64, 512, dtype=torch.float32))
    compressor.gates_gather = _Recorder(torch.zeros(64, 512, dtype=torch.float32))
    monkeypatch.setattr(attention, "gather_rows_three", lambda *args: None)
    calls = []
    import b12x.attention.mla_compress._impl as implementation
    monkeypatch.setattr(implementation, "_run_ratio2", lambda *args: calls.append(args))
    compressor(torch.zeros(16, 512), compressor.metadata, None)
    # A ratio-two compressor holds FP32 values and gates; both gathers move BF16 packs of them.
    (gathered_gates,) = compressor.gates_gather.calls[0]
    assert gathered_gates.dtype is torch.bfloat16 and gathered_gates.shape == (16, 1024)
    assert gathered_gates.data_ptr() == gates.data_ptr()
    (gathered_values,) = compressor.values_gather.calls[0]
    assert gathered_values.dtype is torch.bfloat16 and gathered_values.shape == (16, 1024)
    assert gathered_values.data_ptr() == values.data_ptr()
    # The compression kernel still spans every row, so it reads the gathered products.
    kernel_values, kernel_gates = calls[0][1], calls[0][2]
    assert kernel_values.shape == kernel_gates.shape == (64, 512)
    assert kernel_values.dtype is torch.float32


def test_compiled_row_shard_forwards_read_no_rank_offset_as_a_python_integer():
    """One TP forward compiles per shape and every rank binds that artifact.

    Dynamo turns an attribute integer read inside a compiled forward into a guard, so a rank's row
    offset would be compiled into the artifact and every other rank would be rejected with a binding
    mismatch. Each forward therefore reads per-rank data only through the row numbers the graph has.
    """
    import inspect
    import re

    from ignition.recipes.deepseek41 import attention, decoder

    for forward in (attention.PreparedAttention._run, decoder.PreparedLayer._run):
        source = inspect.getsource(forward)
        assert re.search(r"\b(?:self|row_shard|shard)\.start\b", source) is None
    prepared = inspect.getsource(attention.PreparedAttention._run)
    assert "rows_index = self.rows_index" in prepared
    assert "full.index_select(0, rows_index)" in prepared
    assert "indices.window_indices[self" not in prepared
    assert "m.token_page_table[self" not in prepared


class _Operand:
    """A stand-in prepared operand that returns its buffer and is its own functional result."""

    def __init__(self, value):
        self.value, self.plan = value, object()

    def functional(self, **options):
        return self

    def __call__(self, *args, **kwargs):
        return self.value


@pytest.mark.parametrize("ratio,row_shard", [(2, False), (2, True), (1, False), (1, True)])
def test_a_prepared_compressor_passes_the_norm_its_kernel_reads(monkeypatch, ratio, row_shard):
    """The compression kernel takes the norm from the compressor, not from the prepared declaration.

    `self.arguments` is the declaration the preparation binds and it is released with it, so the
    forward's operands are attributes of the compressor.
    """
    from ignition.recipes.deepseek41 import attention

    import b12x.attention.mla_compress._impl as implementation

    calls = []
    monkeypatch.setattr(implementation, "_run_ratio1", lambda *args: calls.append(("ratio1", args)))
    monkeypatch.setattr(implementation, "_run_ratio2", lambda *args: calls.append(("ratio2", args)))
    monkeypatch.setattr(attention, "declare", lambda *args, **kwargs: None)
    monkeypatch.setattr(attention, "register_finalizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(attention, "gather_rows_three", lambda *args: None)
    monkeypatch.setattr(attention, "UnquantizedProjection",
                        lambda source, weight, out, **options: _Operand(out))
    monkeypatch.setattr(attention, "PreparedGatherShards",
                        lambda source, output, **options: _Operand(output))
    # The native planner requires CUDA, and this case runs the compressor's operand flow on CPU.
    monkeypatch.setattr(attention.mla_compress, "Caps", lambda **options: SimpleNamespace(**options))
    monkeypatch.setattr(attention.mla_compress, "plan",
                        lambda caps: SimpleNamespace(handle=object(), request=lambda **options: _Operand(None)))
    tokens, rows = 64, 4
    text = {"compress_ratios": [0, 0] + [2] * 18 + [1] * 20}
    pool = SimpleNamespace(config=SimpleNamespace(text=text), region=lambda name: None)
    metadata = SimpleNamespace(starts=torch.zeros(rows, dtype=torch.int64),
        query_start_loc=torch.zeros(rows + 1, dtype=torch.int32),
        state_rows=torch.zeros(rows, dtype=torch.int32),
        lengths=torch.zeros(rows, dtype=torch.int32),
        live_counts=torch.zeros(2, dtype=torch.int32),
        positions=torch.zeros(tokens, dtype=torch.int64))
    norm = torch.zeros(512)
    weights = {"norm": norm, "values": torch.zeros(8), "gates": torch.zeros(8)}
    compressor = attention.PreparedCompressor(2 if ratio == 2 else 20, weights,
        torch.zeros((tokens, 512), dtype=torch.bfloat16), metadata, pool,
        torch.zeros(tokens, dtype=torch.int64), session=object(), name="test", deferred=True,
        row_shard=SimpleNamespace() if row_shard else None, runtime=None, coordinator=None)
    compressor(torch.zeros((tokens, 512), dtype=torch.bfloat16), metadata, torch.zeros(tokens, dtype=torch.int64))
    kind, args = calls[0]
    # `_run_ratio2(handle, values, gates, weight, ...)` and `_run_ratio1(handle, values, weight, ...)`.
    assert args[3 if kind == "ratio2" else 2].data_ptr() == norm.data_ptr()
