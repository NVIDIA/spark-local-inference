# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Run the whole row-sharded prefill decoder on simulated ranks and compare it with full rows.

`PreparedDecoder`, `PreparedLayer`, `PreparedSelectionRefill`, `RouteRecord` and
`attention.PreparedCompressor` are the prepared implementations this test runs. Everything that
exists only as a b12x kernel is replaced by a deterministic PyTorch reference, because no GPU is
present: the mHC branches, MLA attention, the router, the routed and shared experts, the Engram mix,
the embedding and the vocabulary head. The reference arithmetic is row-local wherever the prepared
forward itself does not cross rows, so a full-row run and a row-sharded run of the same chunk compute
the same value for every row.

The collectives are a simulated fabric. Each reduce-scatter, gather-shards and all-reduce meets every
rank at one barrier in lock-step, exchanges its source between the simulated ranks, and records the
operation name, that rank's sequence number and the operand shape. The sharded run therefore fails if
the ranks issue a different number, order or shape of collectives, and each sharded layer's experts
must receive the route IDs and weights the full-row run produced for the same rows.
"""

import threading
from types import SimpleNamespace

import pytest
import torch


# ---------------------------------------------------------------------------- simulated collectives


class BrokenCollective(RuntimeError):
    """A simulated rank did not meet its peers at a collective in lock-step."""


class Fabric:
    """One barrier per collective call, shared by every simulated rank of a world."""

    def __init__(self, world, timeout=300.0):
        self.world = world
        self.barrier = threading.Barrier(world, timeout=timeout)
        self.lock = threading.Lock()
        self.turns, self.meta, self.seen = {}, {}, {}
        self.logs = [[] for _ in range(world)]

    def abort(self):
        self.barrier.abort()

    def collect(self, rank, kind, name, payload):
        """Publish `payload`, wait for every rank, and return every rank's payload in rank order."""
        with self.lock:
            sequence = len(self.logs[rank])
            self.logs[rank].append((name, sequence, kind, tuple(payload.shape), str(payload.dtype)))
            self.turns.setdefault(sequence, [None] * self.world)[rank] = payload
            self.meta.setdefault(sequence, [None] * self.world)[rank] = (kind, name)
        try:
            self.barrier.wait()
        except threading.BrokenBarrierError:
            raise BrokenCollective(
                f"rank {rank} waited at {kind} '{name}', sequence {sequence}, while the ranks had"
                f" issued {self.summary()}") from None
        with self.lock:
            arrived = list(self.meta[sequence])
            payloads = list(self.turns[sequence])
            self.seen[sequence] = self.seen.get(sequence, 0) + 1
            if self.seen[sequence] == self.world:
                del self.turns[sequence], self.meta[sequence], self.seen[sequence]
        if any(entry != (kind, name) for entry in arrived):
            raise BrokenCollective(
                f"rank {rank} issued {kind} '{name}' at sequence {sequence} where the ranks issued"
                f" {arrived}")
        shapes = {tuple(payload.shape) for payload in payloads}
        if len(shapes) != 1:
            raise BrokenCollective(f"{kind} '{name}' exchanged the shapes {sorted(shapes)}")
        return payloads

    def summary(self):
        return [log[-1] if log else None for log in self.logs]

    def sequence(self, rank):
        """One rank's recorded (op name, sequence number, shape, dtype) collective calls."""
        return self.logs[rank]


class SimulatedRuntime:
    """One rank's view of the fabric: its rank, the world, and the row-shard surfaces it serves."""

    def __init__(self, fabric, rank, world):
        self.fabric, self.rank, self.world_size = fabric, rank, world
        self.pipe_bytes = 1 << 30
        self._group = object()
        self.should_reduce_scatter = lambda source: True
        self.should_gather_shards = lambda source: True


class Collective:
    """One prepared collective over the simulated fabric, or a local copy when it has no runtime."""

    kind = None

    def __init__(self, source, output, *, runtime=None, session=None, coordinator=None, name=""):
        self.runtime, self.name, self.output = runtime, name, output
        self.single = runtime is None
        self.rank = 0 if runtime is None else runtime.rank
        if runtime is None:
            if source.shape != output.shape or source.dtype is not output.dtype:
                raise ValueError("a single-rank collective copies its source into a matching output")
            return
        if not self.fits(runtime, source, output):
            raise ValueError(f"{self.kind} does not fit the simulated runtime")

    def functional(self, *, session=None):
        if self.output is not None:
            self.output_layout = (tuple(self.output.shape), self.output.dtype)
            self.output = None
        return self

    def __call__(self, source, output=None):
        if output is None:
            output = self.output
        if output is None:
            shape, dtype = self.output_layout
            output = torch.empty(shape, dtype=dtype, device=source.device)
        if self.single:
            output.copy_(source)
        else:
            output.copy_(self.combine(self.runtime.fabric.collect(self.rank, self.kind, self.name, source)))
        return output

    def combine(self, payloads):
        raise NotImplementedError


class ReduceScatter(Collective):
    """Sum the ranks' partials in rank order, then keep this rank's contiguous row block."""

    kind = "reduce_scatter"

    @staticmethod
    def fits(runtime, source, output):
        world = runtime.world_size
        return (isinstance(runtime, SimulatedRuntime) and source.ndim > 0
                and source.shape[0] % world == 0
                and tuple(output.shape) == (source.shape[0] // world, *source.shape[1:])
                and source.dtype is output.dtype is torch.bfloat16)

    def combine(self, payloads):
        world = self.runtime.world_size
        rows = payloads[0].shape[0] // world
        total = payloads[0].float()
        for payload in payloads[1:]:
            total = total + payload.float()
        return total.to(payloads[0].dtype)[self.rank * rows:(self.rank + 1) * rows]


class AllReduce(Collective):
    kind = "all_reduce"

    @staticmethod
    def fits(runtime, source, output):
        return (isinstance(runtime, SimulatedRuntime) and source.shape == output.shape
                and source.dtype is output.dtype)

    def combine(self, payloads):
        total = payloads[0].float()
        for payload in payloads[1:]:
            total = total + payload.float()
        return total.to(payloads[0].dtype)


class GatherShards(Collective):
    """Publish rank-major rows: rank zero's shard first, then rank one's, and so on."""

    kind = "gather_shards"

    @staticmethod
    def fits(runtime, source, output):
        world = runtime.world_size
        return (isinstance(runtime, SimulatedRuntime) and source.ndim > 0
                and tuple(output.shape) == (source.shape[0] * world, *source.shape[1:])
                and source.dtype is output.dtype)

    def combine(self, payloads):
        return torch.cat(payloads, dim=0)


class ChunkedGatherShards(GatherShards):
    """The simulated pipe holds every chunk, so a chunked gather is one gather of whole shard rows."""

    kind = "chunked_gather_shards"


# ---------------------------------------------------------------------------- reference components


_CONTEXT = threading.local()


class Run:
    """The simulated rank, shard partition and route log of the forward running on this thread."""

    def __init__(self, rank=0, sharded=False, world=1):
        self.rank, self.sharded, self.world = rank, sharded, world
        self.routes = {}
        self.produced = {}
        self.margins = {}

    def columns(self, hidden):
        """The output columns this rank computes; a full-row run computes every column."""
        if not self.sharded:
            return 0, hidden
        return self.rank * hidden // self.world, (self.rank + 1) * hidden // self.world


def _run():
    return getattr(_CONTEXT, "run", Run())


def _layer_of(name):
    return int(name.split(".layer", 1)[1].split(".", 1)[0])


def _norm(value, weight, eps):
    value = value.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)) * weight.float()


class ReferenceMHC:
    """mHC pre or fused post/pre and the following RMSNorm, as row-local reference arithmetic.

    The four-stream residual, its post coefficients and the lagged mix stay per row, and `pre_out`
    receives the mix the next branch reads, which is the state the prepared mHC carries between a
    layer's attention and feed-forward branches.
    """

    def __init__(self, inputs, weights, scratch, *, pre_mix=None, pre_out=None, rms_eps=1e-6,
                 hc_eps=1e-6, sinkhorn_iters=20, norm_eps=1e-6, **options):
        self.operation = "pre" if len(inputs) == 1 else "post_pre"
        self.weights, self.eps, self.hc_eps = weights, rms_eps, hc_eps

    def functional(self):
        return self

    def __call__(self, *inputs, pre_mix=None, pre_out=None):
        if self.operation == "pre":
            (residual,) = inputs
        else:
            source, residual, post, comb = inputs
            diagonal = comb.diagonal(dim1=-2, dim2=-1)
            residual = residual.float() + source.float()[:, None, :] * (post * diagonal)[:, :, None]
        residual = residual.to(torch.bfloat16)
        mix = torch.sigmoid(residual.float().amax(-1) * self.weights.scale[0]
                            + self.weights.base[:4]) + self.hc_eps
        if pre_out is not None:
            pre_out.copy_(mix)
        used = mix if pre_mix is None else pre_mix
        normalized = _norm((residual.float() * used[:, :, None]).sum(1), self.weights.norm, self.eps)
        comb = torch.eye(4, dtype=torch.float32).expand(residual.shape[0], -1, -1).contiguous()
        return residual, mix, comb, normalized.to(torch.bfloat16)


class ReferenceMHCPost:
    """Write one branch's output into the four residual streams."""

    def __init__(self, source, residual, post, comb, output, **options):
        pass

    def functional(self):
        return self

    def __call__(self, source, residual, post, comb, out=None):
        diagonal = comb.diagonal(dim1=-2, dim2=-1)
        return (residual.float() + source.float()[:, None, :] * diagonal[:, :, None]).to(torch.bfloat16)


class ReferenceProjection:
    """A prepared projection as a row-local matrix product; the reference weights are narrow, so a
    projection reads the leading columns of its source when the two widths differ."""

    def __init__(self, source, weight, output, session=None, name=None, bias=None, **options):
        self.weight, self.dtype, self.plan = weight, output.dtype, object()

    def functional(self, *, session=None):
        return self

    def __call__(self, source):
        width = self.weight.shape[1]
        return (source[:, :width].float() @ self.weight.float().T).to(self.dtype)


class ReferenceRouter:
    """Reference top-k routing that allocates its routes inside the forward, as a functional router
    does: nothing writes the shard's prepared route record unless the layer copies the routes in.

    It records the routes it selected and the margin between the last selected and the first rejected
    expert, which the test uses to separate a route that changed from one that sat in a near tie.
    """

    def __init__(self, logits, scratch, *, name=None, score_func=None,
                 renormalize=True, routed_scaling_factor=1.0, correction_bias=None,
                 image_correction_bias=None, image_mask=None, session=None, **options):
        self.top_k, self.plan = scratch.ids.shape[1], object()
        self.name = name

    def functional(self, *, session=None):
        return self

    def __call__(self, logits, *, image_mask=None):
        probs = logits.float().softmax(-1)
        order = torch.argsort(-probs, dim=-1, stable=True)[:, :self.top_k]
        values = probs.gather(1, order)
        weights = values / values.sum(-1, keepdim=True)
        ids = order
        context = _run()
        sorted_probs = probs.sort(-1, descending=True).values
        if self.top_k < sorted_probs.shape[-1]:
            context.margins.setdefault(_layer_of(self.name or "x.layer0.x"), []).append(
                sorted_probs[:, self.top_k - 1] - sorted_probs[:, self.top_k])
        produced = context.produced
        if produced is not None:
            produced.setdefault(_layer_of(self.name or "x.layer0.x"), []).append(
                (ids.to(torch.int32).clone(), weights.clone()))
        return SimpleNamespace(ids=ids.to(torch.int32), weights=weights, selected_logits=values)


class ReferenceExperts:
    """Reference routed experts over this rank's output columns, which the mlp reduction sums.

    The route gate reads the whole branch input, so the selected IDs and weights decide the result,
    and a record that was gathered without being written changes it.
    """

    def __init__(self, source, experts, ids, weights, output, *, name="", **options):
        self.layer, self.hidden = _layer_of(name), output.shape[1]
        self.keys, self.directions = experts.keys, experts.directions

    def functional(self, *, session=None):
        return self

    def __call__(self, source, ids, weights, router_logits=None, selected_logits=None, image_mask=None):
        _run().routes.setdefault(self.layer, []).append((ids.clone(), weights.clone()))
        lo, hi = _run().columns(self.hidden)
        gather = ids.clamp_min(0)
        scores = (self.keys.float()[gather] * source.float()[:, None, :self.keys.shape[1]]).sum(-1)
        coefficient = torch.where(ids >= 0, weights.float() * scores, torch.zeros_like(scores))
        output = torch.zeros((source.shape[0], self.hidden), dtype=torch.bfloat16, device=source.device)
        if source.shape[0]:
            output[:, lo:hi] = torch.bmm(coefficient[:, None, :],
                                         self.directions.float()[gather][:, :, lo:hi]).squeeze(1)
        return output


class ReferenceMLP:
    """Reference shared expert over this rank's output columns, summed by the mlp reduction."""

    def __init__(self, source, weights, gate_up, intermediate, output, **options):
        self.up, self.down, self.hidden = weights.up, weights.down, output.shape[1]

    def functional(self, *, session=None):
        return self

    def __call__(self, source):
        lo, hi = _run().columns(self.hidden)
        hidden = torch.nn.functional.silu(source.float()[:, :self.up.shape[0]] @ self.up.float())
        output = torch.zeros((source.shape[0], self.hidden), dtype=torch.bfloat16, device=source.device)
        output[:, lo:hi] = (hidden @ self.down.float()[:, lo:hi]).to(torch.bfloat16)
        return output


class ReferenceEngram:
    """Reference Engram mix: a token-dependent addition to the full-row residual."""

    def __init__(self, layer, weights, tables, source, output, metadata, pool, **options):
        self.output = output

    def __call__(self, source, metadata):
        self.output.copy_(source)
        self.output[:, :, :64] += metadata.token_ids.view(-1, 1, 1).to(torch.bfloat16) * 0.01
        return self.output


class ReferenceTableRequests:
    """The request history the prepared table operations maintain; this chunk has no prefix."""

    def __init__(self, tables, metadata, pool, *, session=None, name=""):
        pass

    def __call__(self, metadata):
        return torch.zeros((metadata.starts.numel(), 3), dtype=torch.int64)

    def commit(self, counts, raw_history, metadata):
        pass


class ReferenceEmbedding:
    def __init__(self, weight, ids, output, *, session=None, name="", num_rows=None):
        self.weight, self.dtype = weight, output.dtype

    def functional(self, *, session=None, dynamic_inputs=False):
        return self

    def __call__(self, ids=None, *, token_ids=None, num_rows=None):
        ids = ids if ids is not None else token_ids
        return self.weight.float()[ids.clamp_min(0)].to(self.dtype)


class ReferenceVocabulary:
    """Reference head: a row-local projection onto the whole vocabulary."""

    def __init__(self, source, weight, local_output, output, *, runtime=None, session=None,
                 coordinator=None, name="", row_indices=None, gather=True, **options):
        self.weight, self.output, self.dtype = weight, output, output.dtype

    def functional(self, *, session=None, dynamic_indices=False):
        return self

    def __call__(self, source, row_indices=None):
        return (source[:, :self.weight.shape[1]].float() @ self.weight.float().T).to(self.dtype)


class ReferenceGlobalKV:
    """The decoder boundary's full-row KV production; the reference keeps no fused cache."""

    def __init__(self, layer, weights, source, scratch, metadata, pool, rotary, *, indices=None,
                 **options):
        self.ratio = pool.config.text["compress_ratios"][layer]

    def __call__(self, source, indexer, metadata, geometry):
        return None


class ReferenceAttention:
    """Reference MLA attention that keeps every prepared row-shard surface of its layer.

    Each row writes its KV latent; a shard gathers the latent so that every rank's cache holds every
    row; each attended row then reads a causal window of that cache. An index-source layer writes the
    carried selection in the rows it runs in, and the query reads that selection, so a moved
    selection that holds the wrong rows changes the result.
    """

    def __init__(self, layer, weights, source, scratch, metadata, pool, rotary, *, mode=None,
                 world=1, session=None, name="", workspace=None, replay_start=None,
                 window_staging=None, mla_compute_mode="auto", window_metadata=None, examples=None,
                 row_shard=None, window_target=None, runtime=None, coordinator=None, **options):
        from ignition.recipes.deepseek41 import attention as prepared

        self.layer, self.weights, self.scratch = layer, weights, scratch
        self.pool, self.mode, self.row_shard = pool, mode, row_shard
        self.ratio = pool.config.text["compress_ratios"][layer]
        self.eps = pool.config.text["rms_norm_eps"]
        device = source.device
        self.total = source.shape[0] if row_shard is None else metadata.positions.numel()
        self.window_values = scratch.window_values if row_shard is None else window_target
        if row_shard is None:
            if window_target is not None:
                raise ValueError("a full-row attention writes its own KV latent")
            self.window_gather = None
        else:
            if window_target is None or window_target.shape != (self.total, 512):
                raise ValueError("row-sharded attention gathers its KV latent into a full-row buffer")
            self.window_gather = GatherShards(scratch.window_values, window_target, runtime=runtime,
                                              session=session, coordinator=coordinator,
                                              name=name + ".window_gather")
        self.cache = torch.zeros((pool.tokens, 512), dtype=torch.bfloat16, device=device)
        self.selection = torch.zeros((source.shape[0], 512), dtype=torch.int32, device=device)
        self.selectors = [weights["selector"]] if self.ratio else []
        # The prepared compressor is the one this test keeps: its gathers, its operand flow and the
        # norm the compression kernel reads all run in the forward.
        self.compressor = None if "compressor" not in weights else prepared.PreparedCompressor(
            layer, weights["compressor"], source, metadata, pool,
            torch.arange(metadata.positions.numel(), dtype=torch.int64), session=session,
            name=f"{name}.compress", row_shard=row_shard, runtime=runtime, coordinator=coordinator)

    def functional(self, *, session=None):
        return self

    def release_examples(self):
        pass

    def commit_pending(self, counts, record, metadata):
        pass

    def _run(self, source, metadata, selection, *, indices=None, window_cache=None):
        rows_index = None if self.row_shard is None else self.row_shard.rows_index
        latent = torch.zeros((source.shape[0], 512), dtype=torch.bfloat16, device=source.device)
        latent.copy_((source[:, :512].float() @ self.weights["k"].float()).to(torch.bfloat16))
        if self.compressor is not None:
            geometry = indices.compressed(self.ratio)
            compressed, _, _ = self.compressor(source, metadata, geometry.slots)
            if rows_index is not None:
                compressed = compressed.index_select(0, rows_index)
            latent = latent + compressed
        self.scratch.window_values.copy_(latent)
        # Every rank holds every row's latent before any rank reads the cache, which is what makes a
        # cache entry of another rank's rows available here.
        full = latent if self.window_gather is None else self.window_gather(self.scratch.window_values)
        live = indices.position >= 0
        self.cache.index_copy_(0, indices.position.clamp_min(0)[live], full[live])
        position = indices.position if rows_index is None else indices.position.index_select(0, rows_index)
        selected, candidates, counts = selection
        if self.ratio:
            # An attention layer that selects writes the carried selection in the rows it runs in.
            self.selection.copy_((latent.float() * 4096).to(torch.int32))
            selected = self.selection
        if self.layer == self.pool.candidate:
            candidates = torch.zeros((source.shape[0], 16384), dtype=torch.int32, device=source.device)
            counts = torch.zeros(source.shape[0], dtype=torch.int32, device=source.device)
        # Production reads the carried selection only in a layer that selects, so a compression-free
        # layer never touches a selection another row representation wrote.
        output = self._attend(source, position, selected if self.ratio else None)
        record = (window_cache, None)
        return output, record, (selected, candidates, counts), {"selected": selected}

    def _attend(self, source, position, selected):
        """A causal window of the completed cache, gated by the carried selection."""
        offsets = torch.arange(128, device=source.device)
        starts = position[:, None] - 127 + offsets[None, :]
        inside = (starts >= 0) & (starts <= position[:, None])
        window = self.cache[starts.clamp_min(0)].float()
        query = source[:, :512].float() @ self.weights["q"].float()
        if selected is not None:
            query = query + selected.float().mean(-1, keepdim=True) * 1e-4
        # Cosine similarity keeps the reference scores bounded, so a one-ulp difference in a row
        # changes the softmax smoothly instead of flipping which key dominates it.
        unit_query = query / query.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        unit_keys = window / window.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scores = torch.bmm(unit_query[:, None, :], unit_keys.transpose(1, 2)).squeeze(1) * 4
        scores = scores.masked_fill(~inside, torch.finfo(torch.float32).min)
        attended = torch.bmm(scores.softmax(-1)[:, None, :], window).squeeze(1)
        attended = torch.where((position >= 0)[:, None], attended, torch.zeros_like(attended))
        output = torch.zeros((source.shape[0], HIDDEN), dtype=torch.bfloat16, device=source.device)
        # A row-sharded attention keeps every head and output group, so its rows are complete and no
        # reduction follows. A full-row layer computes one rank's output columns instead, and the
        # prepared all-reduce sums them across the ranks.
        values = attended.to(torch.bfloat16).repeat(1, HIDDEN // 512)
        lo, hi = (0, HIDDEN) if self.row_shard is not None else _run().columns(HIDDEN)
        output[:, lo:hi] = values[:, lo:hi]
        return output


def _install(monkeypatch):
    """Replace the GPU-only prepared components with the references above."""
    from ignition.recipes.deepseek41 import attention as prepared, decoder

    monkeypatch.setattr(decoder, "PreparedMHC", ReferenceMHC)
    monkeypatch.setattr(decoder, "PreparedMHCPost", ReferenceMHCPost)
    monkeypatch.setattr(decoder, "PreparedAttention", ReferenceAttention)
    monkeypatch.setattr(decoder, "PreparedGlobalKV", ReferenceGlobalKV)
    monkeypatch.setattr(decoder, "PreparedEngram", ReferenceEngram)
    monkeypatch.setattr(decoder, "PreparedTableRequests", ReferenceTableRequests)
    monkeypatch.setattr(decoder, "PreparedEmbedding", ReferenceEmbedding)
    monkeypatch.setattr(decoder, "PreparedVocabularyProjection", ReferenceVocabulary)
    monkeypatch.setattr(decoder, "UnquantizedProjection", ReferenceProjection)
    monkeypatch.setattr(decoder, "PreparedRouter", ReferenceRouter)
    monkeypatch.setattr(decoder, "PreparedRoutedExperts", ReferenceExperts)
    monkeypatch.setattr(decoder, "PreparedMLP", ReferenceMLP)
    monkeypatch.setattr(decoder, "PreparedAllReduce", AllReduce)
    monkeypatch.setattr(decoder, "PreparedReduceScatter", ReduceScatter)
    monkeypatch.setattr(decoder, "PreparedGatherShards", GatherShards)
    monkeypatch.setattr(decoder, "PreparedChunkedGatherShards", ChunkedGatherShards)
    monkeypatch.setattr(decoder, "register_finalizer", lambda *args, **options: None)
    # The prepared compressor keeps its own module's names for its gathers and its projections.
    monkeypatch.setattr(prepared, "PreparedGatherShards", GatherShards)
    monkeypatch.setattr(prepared, "UnquantizedProjection", ReferenceProjection)
    monkeypatch.setattr(prepared, "gather_rows_three", lambda *args: None)
    monkeypatch.setattr(prepared, "save_hidden", lambda *args: None)
    monkeypatch.setattr(prepared, "declare", lambda *args, **options: None)
    monkeypatch.setattr(prepared, "register_finalizer", lambda *args, **options: None)
    monkeypatch.setattr(prepared, "mla_compress", SimpleNamespace(
        Caps=lambda **options: SimpleNamespace(**options),
        plan=lambda caps: SimpleNamespace(handle=None, request=lambda **options: None)))
    import b12x.attention.mla_compress._impl as implementation

    monkeypatch.setattr(implementation, "_run_ratio1", _run_ratio1)
    monkeypatch.setattr(implementation, "_run_ratio2", _run_ratio2)


def _norm_kernel(weight):
    if weight is None:
        raise ValueError("the compression kernel reads the compressor's norm weight")
    return weight.float()


def _run_ratio1(handle, values, weight, query_start_loc, starts, state_ids, slots, live_counts,
                output, emitted, emitted_slots):
    """Reference ratio-one compression: every row emits its norm-weighted latent into its slot."""
    weight = _norm_kernel(weight)
    emitted.copy_(True)
    emitted_slots.copy_(slots)
    output.copy_((values.float() * weight).to(torch.bfloat16))


def _run_ratio2(handle, values, gates, weight, query_start_loc, starts, state_ids, slots, live_counts,
                pending_values, pending_gates, pending_position, output, emitted, emitted_slots):
    """Reference ratio-two compression: row pairs emit one gated, norm-weighted latent."""
    weight = _norm_kernel(weight)
    rows = values.shape[0]
    emit = (torch.arange(rows) % 2 == 1)
    combined = (values.float() + gates.float().flip(0)) * 0.5 * weight
    output.copy_(torch.where(emit[:, None], combined, torch.zeros_like(combined)).to(torch.bfloat16))
    emitted.copy_(emit)
    emitted_slots.copy_(torch.where(emit, slots, torch.full_like(slots, -1)))


# ---------------------------------------------------------------------------- the tiny V4.1 shape

LAYERS, HIDDEN, EXPERTS, TOP_K, VOCAB = 9, 5120, 8, 2, 129280
SAMPLES = 4


def _text():
    """A V4.1-shaped schedule: Engram layers, a ratio-two region, the decoder boundary at layer 7,
    a ratio-one region past it, and a draft target layer. `compress_ratios` runs three zeros past
    the backbone for the DSpark stages, as the checkpoint's does."""
    return {
        "num_hidden_layers": LAYERS,
        "compress_ratios": [0, 0, 2, 2, 2, 2, 2, 1, 1, 0, 0, 0],
        "kv_source_layer_ids": [2, 7],
        "index_source_layer_ids": [2, 7],
        "engram_layer_ids": [1, 4],
        "candidate_source_layer_id": 7,
        "rms_norm_eps": 1e-6, "hc_eps": 1e-6, "hc_sinkhorn_iters": 20,
        "scoring_func": "sigmoid", "norm_topk_prob": True, "routed_scaling_factor": 1.0,
        "swiglu_limit": 10.0,
        "num_nextn_predict_layers": 3, "dspark_block_size": 5, "dspark_noise_token_id": 3,
        "dspark_target_layer_ids": [8], "dspark_markov_rank": 256,
        "dspark_n_routed_experts": 128, "dspark_num_experts_per_tok": 3, "vocab_size": VOCAB,
        "sliding_window": 128,
        "rope_theta": 10000.0, "compress_rope_theta": 160000.0,
        "rope_scaling": {"rope_type": "yarn", "factor": 8.0, "beta_fast": 32, "beta_slow": 1,
                         "original_max_position_embeddings": 4096},
    }


def _config():
    from ignition.recipes.deepseek41.config import ModelConfig

    return ModelConfig(_text(), {})


def _weights(config, generator):
    """Reference weights shared by the full-row and the sharded run, so both compute the same math."""
    from ignition.recipes.deepseek41.weights import ExpertWeights, LayerWeights, ModelWeights

    def normal(*shape, dtype=torch.float32):
        return (torch.randn(shape, generator=generator) * 0.05).to(dtype)

    keys, queries = normal(512, 512, dtype=torch.bfloat16), normal(512, 512, dtype=torch.bfloat16)
    compressor = {"norm": torch.ones(512), "values": normal(512, HIDDEN, dtype=torch.bfloat16),
                  "gates": normal(512, HIDDEN, dtype=torch.bfloat16)}
    experts = ExpertWeights(router=normal(EXPERTS, 512, dtype=torch.bfloat16), correction=None,
        vision_correction=None,
        experts=SimpleNamespace(keys=normal(EXPERTS, 512, dtype=torch.bfloat16),
                                directions=normal(EXPERTS, HIDDEN, dtype=torch.bfloat16)),
        shared=SimpleNamespace(up=normal(512, 64, dtype=torch.bfloat16),
                               down=normal(64, HIDDEN, dtype=torch.bfloat16)))
    residual = {"hc_attn_fn": torch.zeros((24, 4 * HIDDEN)), "hc_attn_scale": torch.ones(3),
                "hc_attn_base": torch.zeros(24), "attn_norm.weight": torch.ones(HIDDEN),
                "hc_ffn_fn": torch.zeros((24, 4 * HIDDEN)), "hc_ffn_scale": torch.ones(3),
                "hc_ffn_base": torch.zeros(24), "ffn_norm.weight": torch.ones(HIDDEN)}
    layers = []
    for index in range(LAYERS):
        attention = {"k": keys, "q": queries, "selector": object()}
        if config.text["compress_ratios"][index]:
            attention["compressor"] = compressor
        layers.append(LayerWeights(attention=attention, residual=residual, experts=experts,
                                   engram=None))
    return ModelWeights(config=config, shard=None,
        embedding=normal(64, HIDDEN, dtype=torch.bfloat16), norm=torch.ones(HIDDEN),
        head=normal(VOCAB, 64), layers=tuple(layers), draft=object(), allocated_bytes=0)


def _metadata(tokens, samples):
    from ignition.components.paged_batch import Metadata

    blocks = tokens // 128 + 1
    metadata = Metadata.allocate(tokens, 1, blocks, "cpu", samples=samples, max_query_tokens=tokens)
    metadata.set(torch.zeros(1, dtype=torch.int64), torch.tensor([tokens], dtype=torch.int32),
                 torch.ones(1, dtype=torch.int64), torch.arange(blocks, dtype=torch.int32).view(1, -1),
                 torch.arange(tokens).view(1, -1), torch.ones(1, dtype=torch.int32))
    metadata.token_ids.copy_((torch.arange(tokens) % 64).to(torch.int64))
    return metadata


def _pool(config, tokens):
    return SimpleNamespace(config=config, tokens=tokens,
                           candidate=config.text["candidate_source_layer_id"],
                           region=lambda name: torch.zeros(1))


def _decoder(weights, tokens, world, rank, runtime, row_shard_attention, samples=SAMPLES):
    """One simulated rank's prepared prefill decoder."""
    from dataclasses import replace

    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.deepseek41 import decoder

    config = weights.config
    scratch = decoder.Scratch.allocate(tokens, samples, world, "cpu", candidate_tokens=16,
        experts=EXPERTS, top_k=TOP_K,
        row_shard=None if runtime is None else SimpleNamespace(rank=rank),
        attention_shard=row_shard_attention)
    metadata = _metadata(tokens, samples)
    local = replace(weights, shard=TensorParallelShard(rank, world))
    prepared = decoder.PreparedDecoder(local, None, scratch, metadata, _pool(config, tokens),
        mode="prefill", runtime=runtime, vocabulary_runtime=None, coordinator=None,
        session=SimpleNamespace(), name="test", workspace=None,
        row_shard_attention=row_shard_attention)
    # Preparation releases the example scratch before compilation, so a forward reads only what the
    # components kept.
    for layer in prepared.layers:
        layer.release_examples()
    return prepared, metadata


def _forward(decoder, metadata, run):
    _CONTEXT.run = run
    hidden, logits, auxiliary, _ = decoder(metadata)
    return hidden, logits, auxiliary


def _sharded(world, tokens, weights):
    """Run `world` simulated ranks of one packed prefill chunk in lock-step threads."""
    fabric = Fabric(world)
    runtimes = [SimulatedRuntime(fabric, rank, world) for rank in range(world)]
    results, errors = [None] * world, []

    def target(rank):
        try:
            prepared, metadata = _decoder(weights, tokens, world, rank, runtimes[rank], True)
            run = Run(rank, sharded=True, world=world)
            results[rank] = _forward(prepared, metadata, run) + (run,)
        except BaseException as error:  # noqa: BLE001 - re-raised on the test thread
            errors.append(error)
            fabric.abort()

    threads = [threading.Thread(target=target, args=(rank,), name=f"rank{rank}")
               for rank in range(world)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=600)
    if any(thread.is_alive() for thread in threads):
        fabric.abort()
        raise BrokenCollective(f"a simulated rank stopped at {fabric.summary()}")
    if errors:
        raise errors[0]
    return fabric, results


def _full_row(tokens, weights):
    prepared, metadata = _decoder(weights, tokens, 1, 0, None, False)
    run = Run(0, sharded=False, world=1)
    hidden, logits, auxiliary = _forward(prepared, metadata, run)
    return SimpleNamespace(hidden=hidden, logits=logits, auxiliary=auxiliary, run=run)


# ---------------------------------------------------------------------------- the checks


def test_the_fabric_reports_a_divergent_collective_sequence():
    """The lock-step exchange names the op and sequence at which the ranks disagree."""
    fabric = Fabric(2, timeout=10)
    seen = []

    def rank(index, name):
        try:
            fabric.collect(index, "gather_shards", name, torch.zeros(2))
        except BrokenCollective as error:
            seen.append(str(error))

    threads = [threading.Thread(target=rank, args=(0, "layer0.a")),
               threading.Thread(target=rank, args=(1, "layer0.b"))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert len(seen) == 2
    assert "layer0.a" in seen[0] and "layer0.b" in seen[0]


def _schedule():
    """The prepared schedule and selection moves this shape produces, for the coverage assertions."""
    from ignition.recipes.deepseek41 import decoder

    config = _config()
    schedule = decoder.shard_schedule(config, config.prefill_stop(False), object(),
                                      config.draft.target_layers)
    return config, schedule, decoder.selection_moves(config, schedule, attention_shard=True)


def test_the_tiny_shape_covers_every_row_transition_a_packed_prefill_has():
    config, schedule, moves = _schedule()
    from ignition.recipes.deepseek41.decoder import SelectionMove

    # The decoder boundary sits at layer 7, so the packed prefill stops at layer 6 and shards 0-5.
    assert (config.decoder_start, config.prefill_stop(False)) == (7, 6)
    assert [sharded for sharded, _, _ in schedule[:7]] == [True, False, True, True, False, True, False]
    # The second Engram layer splits the shard: layer 5 reseeds from full rows.
    assert schedule[5][2] is True and schedule[2][2] is True
    # The carried selection crosses to full rows at the Engram layer, back to the shard after it, and
    # gathers to full rows for the layer at the stop.
    assert moves[4] == SelectionMove(2, "shard_to_full")
    assert moves[5] == SelectionMove(2, "full_to_shard")
    assert moves[6] == SelectionMove(2, "shard_to_full")


def _routes_by_layer(run):
    """Per layer, the route IDs and weights the experts read, aligned by ID within each row."""
    aligned = {}
    for layer, records in run.routes.items():
        (ids, weights), = records
        order = torch.argsort(ids, dim=-1)
        aligned[layer] = (ids.gather(1, order), weights.gather(1, order))
    return aligned


@pytest.mark.parametrize("world", [2, 4])
def test_a_row_sharded_prefill_issues_one_collective_sequence_and_matches_full_rows(monkeypatch, world):
    """Every rank issues the same collectives, and the sharded forward reproduces the full-row one."""
    _install(monkeypatch)
    tokens = 512 * world
    weights = _weights(_config(), torch.Generator().manual_seed(20260412))
    full = _full_row(tokens, weights)
    fabric, results = _sharded(world, tokens, weights)
    logs = [fabric.sequence(rank) for rank in range(world)]

    # (a) Divergence check: every rank issued the same operations, in the same order, at the same
    # sequence numbers, over operands of the same shape.
    assert logs[0], "the row-sharded ranks issued no collectives"
    for rank, log in enumerate(logs):
        assert log == logs[0], f"rank {rank} diverged at {_first_divergence(logs[0], log)}"
    assert [entry[1] for entry in logs[0]] == list(range(len(logs[0])))
    kinds = {entry[2] for entry in logs[0]}
    assert {"reduce_scatter", "all_reduce", "gather_shards", "chunked_gather_shards"} <= kinds, kinds
    # The shard's own surfaces are collected: each sharded layer gathers its KV latent, its branch
    # input, its route records and its residual, reduces its attention and expert partials, and the
    # layer that crosses back to full rows gathers the carried selection.
    _, schedule, moves = _schedule()
    for index, (sharded, deferred, _) in enumerate(schedule):
        prefix = f"test.layer{index}."
        names = {entry[0] for entry in logs[0]}
        if sharded:
            # A layer that defers its post widens its rows in the next layer's attention mHC, so only
            # a layer that runs its own post gathers its residual and its lagged mix.
            suffixes = ["attention.window_gather", "ffn_gather", "route_gather", "mlp_reduce"]
            suffixes += [] if deferred else ["residual_gather", "mix_gather"]
            for suffix in suffixes:
                assert prefix + suffix in names, (index, suffix)
            assert prefix + "attention_reduce" not in names
        else:
            assert prefix + "ffn_gather" not in names
        if moves[index] is not None and moves[index].selected == "shard_to_full":
            assert prefix + "selection.selected" in names, index

    # (c) Route records. Each layer's experts must read routes this forward selected: a sharded layer
    # reads the rank-major concatenation of every rank's router output, restored by the record gather;
    # a full-row layer reads its own router's output. The comparison is exact, because the record
    # carries int32 IDs and FP32 weight bits without arithmetic.
    _, schedule, _ = _schedule()
    for rank, (_, _, _, run) in enumerate(results):
        for layer, produced in run.produced.items():
            (received_ids, received_weights), = run.routes[layer]
            if schedule[layer][0]:
                want_ids = torch.cat([results[r][3].produced[layer][0][0] for r in range(world)])
                want_weights = torch.cat([results[r][3].produced[layer][0][1] for r in range(world)])
            else:
                (want_ids, want_weights), = produced
            assert torch.equal(received_ids, want_ids), (rank, layer)
            assert torch.equal(received_weights, want_weights), (rank, layer)

    # (c) Against the full-row run, a row's route set must agree. Two runs differ by the rounding of
    # the chain, which moves a router logit by well under 0.5 here, so two experts closer than that
    # can swap places. Every row whose full-row top-k margin exceeds 0.5 is required to match exactly,
    # and matching rows must be the large majority.
    expected = _routes_by_layer(full.run)
    for rank, (_, _, _, run) in enumerate(results):
        actual = _routes_by_layer(run)
        assert sorted(actual) == sorted(expected)
        for layer, (ids, route_weights) in expected.items():
            sharded_ids, sharded_weights = actual[layer]
            agree = (sharded_ids == ids).all(-1)
            if layer in full.run.margins:
                margin, = full.run.margins[layer]
                assert agree[margin > 0.5].all(), (rank, layer, int((margin > 0.5).sum()))
            assert agree.float().mean() > 0.8, (rank, layer, float(agree.float().mean()))
            # The weights are the same softmax, so they move with the same near-tie drift.
            torch.testing.assert_close(sharded_weights[agree], route_weights[agree], rtol=0,
                                       atol=2e-2)

    # (b) The sampled hidden rows, their full-vocabulary logits and the draft feedback columns match.
    # The sharded ranks compute disjoint output columns and the reduction adds zero to every other
    # column, so the row arithmetic is the same arithmetic. What remains is the rounding of the chain,
    # one BF16 rounding being 2**-8 relative over nine layers, and the rows whose router swapped a
    # near-tie expert above. Both bound the drift well under 5e-2 of the tensor norm: the observed
    # relative errors are about 2.5e-2, while a wrong row, a stale gather or an unset route changes
    # these values by O(1), which is a relative error near one.
    for rank, (hidden, logits, auxiliary, _) in enumerate(results):
        for name, got, want in (("hidden", hidden, full.hidden), ("logits", logits, full.logits),
                                ("draft feedback", auxiliary, full.auxiliary)):
            error = _relative_error(got, want)
            assert error < 5e-2, (rank, name, error)
            # Every sampled row carries the same bound on its own, so one wrong row cannot hide
            # inside the tensor norm.
            rows = _relative_error_rows(got, want)
            assert rows.max() < 1e-1, (rank, name, rows.tolist())


def _relative_error_rows(got, want):
    """Each row's difference norm over that row's full-row norm."""
    got, want = got.float(), want.float()
    return (got - want).norm(dim=-1) / want.norm(dim=-1).clamp_min(1e-6)


def _relative_error(got, want):
    """The norm of the difference over the norm of the full-row value."""
    got, want = got.float(), want.float()
    return (got - want).norm() / want.norm().clamp_min(1e-9)
