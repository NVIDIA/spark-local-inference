# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""One rank's Inkling 2 decoder layer at tensor parallelism, token-sharded in prefill.

A rank holds whole query and KV heads and a slice of every expert's
intermediate channels. Prefill graphs hold one request whose `T` rows split
into `world` contiguous row shards; rank `r` owns rows `[r T / world,
(r + 1) T / world)`. Per layer:

1. The owner mixes its rows' attention-residual sources into the layer's
   attention input stream(s) and normalizes them; a shard gather gives every
   rank all rows.
2. Every rank runs its heads' attention on all rows and the attention-output
   convolution on its own partial sums with its own partial-sum history; a
   reduce-scatter returns the owned rows of the reduced output.
3. The owner mixes the MLP input, routes and quantizes its rows into expert-input
   records followed by the route IDs and weights; a shard gather of those packets
   gives every rank all rows.
4. Every rank runs its expert slice's first phase on all rows; a column gather of
   every rank's per-route activation amax gives the maximum for NVFP4 experts;
   the second phase, the expert-output convolution on partial sums and a
   reduce-scatter return the owned rows of the reduced output.
5. The owner updates its rows' residual state.

Decode keeps every row on every rank: all-reduces replace the reduce-scatters,
the gathers of normalized streams and packets disappear, and the amax gather
remains. Both modes apply the output convolutions before the reduction: the
convolutions are linear, so convolving each rank's partial sums with its own
history and summing equals convolving the sum, up to rounding.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ignition.components.collectives import PreparedAllGather, PreparedAllReduce, PreparedReduceScatter, PreparedShardGather

from .arithmetic import commit_history, residual_mix, rms_norm, short_convolution
from .attention import LayerCache, PreparedAttention
from .experts import PreparedExperts
from .layer import LayerWeights

ROUTE_BYTES = 8


@dataclass(frozen=True)
class ResidualState:
    """Committed attention-residual block summaries and the running sum of the current block (BF16 rows)."""

    blocks: tuple[torch.Tensor, ...]
    partial: torch.Tensor


def packet_layout(record, top_k):
    """Byte offsets of a row's expert-input record, int32 route IDs and FP32 route weights, and the packet width."""
    ids = record
    weights = ids + 4 * top_k
    width = weights + 4 * top_k
    if width % 16:
        raise ValueError("expert-input packets must keep 16-byte row alignment")
    return ids, weights, width


@dataclass(frozen=True)
class Collectives:
    """Collective operations one graph's layers share; layers run serially, so their storage is shared too.

    Prefill graphs hold shard gathers of normalized streams and expert-input
    packets and reduce-scatters of output partial sums; decode graphs hold
    all-reduces. Both hold the column gather of per-route activation amax.
    """

    streams: tuple | None
    packets: object | None
    attention: object
    experts: object
    amax: object | None
    world: int
    rank: int

    @classmethod
    def prepare(cls, text, *, mode, tokens, record, runtime, coordinator, session, name, examples=None):
        from ignition.preparation.examples import declaration_tensor

        examples = {} if examples is None else examples
        world = 1 if runtime is None else runtime.world_size
        rank = 0 if runtime is None else runtime.rank
        device = torch.device("cuda", torch.cuda.current_device())

        def buffer(role, shape, dtype=torch.bfloat16):
            return declaration_tensor(examples, role, shape, dtype, device)

        hidden, top_k = text.hidden, text.experts_per_token
        partial = buffer("inkling.collectives.partial", (tokens, hidden))
        source = buffer("inkling.collectives.amax", (tokens, top_k), torch.float32)
        output = buffer("inkling.collectives.amax_all", (tokens, top_k * world), torch.float32)
        amax = PreparedAllGather(source, output, runtime=runtime, session=session, coordinator=coordinator,
                                 name=f"{name}.amax").functional(session=session)
        if mode == "decode":
            reduce = lambda role: PreparedAllReduce(  # noqa: E731
                partial, buffer(f"inkling.collectives.{role}", (tokens, hidden)), runtime=runtime, session=session,
                coordinator=coordinator, name=f"{name}.{role}").functional(session=session)
            return cls(None, None, reduce("attention"), reduce("experts"), amax, world, rank)
        if tokens % world:
            raise ValueError("token-sharded prefill requires rows divisible by the world size")
        owned = tokens // world
        streams = tuple(PreparedShardGather(buffer(f"inkling.collectives.stream{index}_owned", (owned, hidden)),
            buffer(f"inkling.collectives.stream{index}", (tokens, hidden)), runtime=runtime, session=session,
            coordinator=coordinator, name=f"{name}.stream{index}") for index in range(3))
        width = packet_layout(record, top_k)[2]
        packets = PreparedShardGather(buffer("inkling.collectives.packets_owned", (owned, width // 2)),
            buffer("inkling.collectives.packets", (tokens, width // 2)), runtime=runtime, session=session,
            coordinator=coordinator, name=f"{name}.packets")
        scatter = lambda role: PreparedReduceScatter(  # noqa: E731
            partial, buffer(f"inkling.collectives.{role}_owned", (owned, hidden)), runtime=runtime, session=session,
            coordinator=coordinator, name=f"{name}.{role}")
        return cls(streams, packets, scatter("attention"), scatter("experts"), amax, world, rank)


class PreparedLayer(torch.nn.Module):
    """A rank's decoder layer for one graph mode over a shared continuation pool."""

    def __init__(self, layer, weights: LayerWeights, text, pool, metadata, collectives: Collectives, *, mode, tokens,
                 session, workspace, name, examples=None):
        super().__init__()
        self.layer, self.mode, self.text, self.weights = layer, mode, text, weights
        self.collectives = collectives
        self.attention = PreparedAttention(layer, weights.attention, text, LayerCache.of(pool, layer), metadata,
                                           mode=mode, tokens=tokens, session=session, workspace=workspace,
                                           name=f"{name}.attention", examples=examples)
        # In prefill the owner routes and quantizes only its rows; every rank runs the experts on all rows.
        owned = tokens // collectives.world if mode == "prefill" else tokens
        self.experts = PreparedExperts(weights.experts, text, tokens=tokens, route_tokens=owned, session=session,
                                       workspace=workspace, name=f"{name}.experts", examples=examples)
        self.histories = (pool.history(layer, "attention_convolution"), pool.history(layer, "mlp_convolution"))
        self.owned = owned

    def attention_input(self, state: ResidualState):
        w, t = self.weights, self.text
        if not state.blocks:
            streams = (state.partial,) * len(w.input_norms)
        else:
            streams = residual_mix((*state.blocks, state.partial), w.attention_query)
        return tuple(rms_norm(stream, norm, t.eps) for stream, norm in zip(streams, w.input_norms, strict=True))

    def mlp_input(self, state: ResidualState, attention):
        w, t = self.weights, self.text
        if self.layer % t.residual_block == 0:
            mixed = residual_mix((*state.blocks, state.partial, attention), w.mlp_query)[0]
        else:
            mixed = residual_mix((*state.blocks, state.partial), w.mlp_query, addend=attention)[0]
        return rms_norm(mixed, w.mlp_norm, t.eps)

    def update(self, state: ResidualState, attention, mlp):
        if self.layer % self.text.residual_block == 0:
            return ResidualState((*state.blocks, state.partial), attention + mlp)
        total = state.partial.float() + attention.float() + mlp.float()
        return ResidualState(state.blocks, total.to(state.partial.dtype))

    def owned_valid(self, metadata):
        """Live rows among this rank's owned rows of a prefill chunk."""
        first = self.collectives.rank * self.owned
        return (metadata.live_counts[:1] - first).clamp(0, self.owned).int()

    def forward(self, state: ResidualState, metadata):
        """The rank's residual state of its rows after this layer, and the layer's intermediates.

        Intermediates hold the convolved partial sums before each reduction, the
        reduced outputs of this rank's rows, the expert inputs, routes and
        per-route amax every rank holds, and this rank's expert partial sums.
        """
        c, m = self.collectives, metadata
        streams = self.attention_input(state)
        if self.mode == "prefill":
            streams = tuple(gather(stream) for gather, stream in zip(c.streams, streams, strict=False))
        attended, partial = self.attention.attend(streams, m)
        attention_history, mlp_history = self.histories
        attention_partial = short_convolution(partial, self.weights.attention_convolution, attention_history, m)
        commit_history(attention_history, partial, m)
        attention = c.attention(attention_partial)
        mlp_input = self.mlp_input(state, attention)
        if self.mode == "prefill":
            ids, weights, records = self._gather_routes(mlp_input, m)
        else:
            ids, weights = self.experts.route(mlp_input, m.live_counts[:1])
            records = self.experts.quantize(mlp_input)
        pending = self.experts.phase1(records, ids, weights)
        amax = None
        if self.experts.supplies_amax:
            tokens, top_k = ids.shape
            gathered = c.amax(pending.row_amax.view(tokens, top_k))
            amax = gathered.view(tokens, c.world, top_k).amax(1).reshape(-1)
        expert_partial = self.experts.phase2(pending, ids, weights, amax)
        mlp_partial = short_convolution(expert_partial, self.weights.mlp_convolution, mlp_history, m)
        commit_history(mlp_history, expert_partial, m)
        mlp = c.experts(mlp_partial)
        intermediates = dict(attended=attended, attention_partial=attention_partial, attention=attention,
                             mlp_input=mlp_input, records=records, ids=ids, weights=weights,
                             local_amax=pending.row_amax, amax=amax, expert_partial=expert_partial,
                             mlp_partial=mlp_partial, mlp=mlp)
        return self.update(state, attention, mlp), intermediates

    def _gather_routes(self, mlp_input, m):
        """Route and quantize the owned rows; gather every rank's records, IDs and weights."""
        experts, c = self.experts, self.collectives
        ids, weights = experts.route(mlp_input, self.owned_valid(m))
        offset_ids, offset_weights, width = packet_layout(experts.record, ids.shape[1])
        packets = torch.empty((self.owned, width), dtype=torch.uint8, device=mlp_input.device)
        experts.quantize(mlp_input, packets[:, :offset_ids])
        packets[:, offset_ids:offset_weights].view(torch.int32).copy_(ids)
        packets[:, offset_weights:].view(torch.float32).copy_(weights)
        gathered = c.packets(packets.view(torch.bfloat16)).view(torch.uint8)
        all_ids = gathered[:, offset_ids:offset_weights].view(torch.int32).contiguous()
        all_weights = gathered[:, offset_weights:].view(torch.float32).contiguous()
        return all_ids, all_weights, gathered[:, :offset_ids]


class PreparedDecoder:
    """A rank's text decoder for one graph: embedding, layers, output mixture and vocabulary logits.

    Each rank embeds every row from its vocabulary shard; the reduction yields
    the owned rows in prefill and every row in decode. After the last layer the
    owner mixes its rows' residual sources with the output query and normalizes
    them; prefill gathers all rows. The sampled rows' FP32 logits are computed
    per vocabulary shard with the softcap and gathered, and the sampler reads the
    valid vocabulary.
    """

    def __init__(self, weights, pool, metadata, *, mode, tokens, samples, vocabulary, runtime, vocabulary_runtime,
                 coordinator, session, workspace, name):
        from types import SimpleNamespace

        from b12x.moe import fused_moe as moe

        from ignition.preparation.examples import declaration_tensor

        from .vocabulary import PreparedLogits

        self.weights, self.pool, self.mode = weights, pool, mode
        text = weights.config.text
        self.text, self.vocabulary = text, vocabulary
        examples = {}
        formats = {layer.experts.format for layer in weights.layers}
        record = max(moe.quantized_input_record_layout(text.hidden, {"nvfp4": "nvfp4_four_over_six",
                                                                     "mxfp8": "mxfp8"}[fmt])[-1] for fmt in formats)
        self.collectives = Collectives.prepare(text, mode=mode, tokens=tokens, record=record, runtime=runtime,
                                               coordinator=coordinator, session=session, name=f"{name}.collectives",
                                               examples=examples)
        self.layers = [PreparedLayer(index, layer, text, pool, metadata, self.collectives, mode=mode, tokens=tokens,
                                     session=session, workspace=workspace, name=f"{name}.layer{index}",
                                     examples=examples)
                       for index, layer in enumerate(weights.layers)]
        self.logits = PreparedLogits(weights.unembedding, text, weights.rank, tokens=samples, session=session,
                                     name=f"{name}.logits", examples=examples)
        device = weights.unembedding.device
        local = declaration_tensor(examples, "inkling.decoder.logits_local", (samples, weights.unembedding.shape[0]),
                                   torch.float32, device)
        gathered = declaration_tensor(examples, "inkling.decoder.logits_all",
                                      (samples, weights.unembedding.shape[0] * self.collectives.world),
                                      torch.float32, device)
        self.gather = PreparedAllGather(local, gathered, runtime=vocabulary_runtime, session=session,
                                        coordinator=coordinator, name=f"{name}.logit_gather").functional(session=session)
        # Shared generation reads the sampler's logits geometry and the vocabulary channel.
        self.scratch = SimpleNamespace(logits=torch.zeros((samples, vocabulary), dtype=torch.float32, device=device))
        self.head = SimpleNamespace(gather=SimpleNamespace(runtime=vocabulary_runtime))
        self.auxiliary = None
        self.examples = examples

    def release_outputs(self):
        self.scratch = None
        self.examples.clear()

    def embed(self, metadata):
        from .vocabulary import embedding_rows

        w = self.weights
        rows = embedding_rows(w.embedding, metadata.token_ids, w.rank)
        rows = torch.where((metadata.positions >= 0)[:, None], rows, 0)
        return rms_norm(self.collectives.attention(rows), w.embedding_norm, self.text.eps)

    def __call__(self, metadata):
        w, t = self.weights, self.text
        state = ResidualState((), self.embed(metadata))
        for layer in self.layers:
            state, _ = layer(state, metadata)
        mixed = residual_mix((*state.blocks, state.partial), w.output_query)[0]
        hidden = rms_norm(mixed, w.final_norm, t.eps)
        if self.mode == "prefill":
            hidden = self.collectives.streams[0](hidden)
        hidden = hidden[metadata.sample_indices]
        gathered = self.gather(self.logits(hidden))
        logits = gathered[:, :self.vocabulary].contiguous()
        return hidden, logits, None, None
