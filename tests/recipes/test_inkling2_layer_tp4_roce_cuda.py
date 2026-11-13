# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inkling 2 decoder layers at TP4 across four DGX Sparks against the canonical four-rank layer.

Run on four assigned Spark hosts with FLARE_RUN_ROCE_TESTS=1, FLARE_TEST_WORLD=4,
FLARE_TEST_RANK=0 through 3 and the same FLARE_TEST_RENDEZVOUS=tcp://ADDRESS:PORT on
every host. FLARE_INKLING2_SNAPSHOT names the checkpoint on rank 0; on the other ranks
it may name a directory that holds only the snapshot's config.json and
hf_quant_config.json. Rank 0 reads every rank's tensors and the canonical activations
(tests/recipes/inkling2_checkpoint.py) and sends them over the process group.

Each layer prefills one request in chunks of 1100 and 305 rows through a token-sharded
1152-row graph (288 owned rows per rank), forks the request's continuation row after the
first chunk, and decodes three steps of both requests through a replicated three-row
graph whose third row is inactive. Rank 0 then checks:

- replicated values (decode rows, gathered expert inputs, routes and amax) are equal on
  every rank, the gathered amax is the elementwise maximum of the ranks' amax, and each
  reduced output lies within one rounding of the FP32 sum of the ranks' convolved
  partial sums;
- the reduced attention output lies within the sum over ranks of the attention bound of
  tests/recipes/inkling2_bounds.py, propagated through the output convolution;
- the MLP input lies within the mixture bound given the reduced attention output, the
  routes match the canonical router on that input away from ties, and the expert input
  records equal the canonical quantizer's bytes;
- the updated residual state equals the canonical update of the reduced outputs.
It reports the expert output against the canonical four-rank experts on the same input
and routes, and the layer against the canonical four-rank layer.
"""

from datetime import timedelta
import os
import socket
from urllib.parse import urlsplit

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_ROCE_TESTS") != "1",
                               reason="requires four assigned Spark hosts")

LAYERS = (0, 5, 40)
PREFILL_ROWS, DECODE_ROWS, BLOCKS = 1152, 3, 12
CHUNKS = ((0, 1100), (1100, 305))
DECODE_STEPS = 3


@pytest.fixture(scope="module")
def cluster():
    from ignition.preparation.collectives import CollectivePreparation, create_roce_runtime
    from ignition.preparation.host_ranks import _use_interface
    from ignition.recipes.inkling2.config import ModelConfig
    from ignition.recipes.inkling2.weights import checkpoint_layout
    from tests.recipes.inkling2_checkpoint import SNAPSHOT

    rank, world = int(os.environ["FLARE_TEST_RANK"]), int(os.environ.get("FLARE_TEST_WORLD", "4"))
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    rendezvous = os.environ["FLARE_TEST_RENDEZVOUS"]
    address = urlsplit(rendezvous)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
        route.connect((address.hostname, address.port))
        _use_interface(route.getsockname()[0])
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=30))
    coordinator = CollectivePreparation(dist.group.WORLD)
    runtime = create_roce_runtime(dist.group.WORLD, device, max_tokens=64, pipe_tokens=PREFILL_ROWS,
                                 hidden_size=4608, gather_bytes=4 << 20)
    config = ModelConfig.load(SNAPSHOT)
    try:
        yield dict(rank=rank, world=world, device=device, coordinator=coordinator, runtime=runtime, config=config,
                   layout=checkpoint_layout(config, world))
    finally:
        runtime.close()
        dist.destroy_process_group()


def counters():
    """Out-of-sequence, packet-sequence-error and ACK-timeout counts of every local RoCE port."""
    from pathlib import Path

    values = {}
    for port in Path("/sys/class/infiniband").glob("*/ports/*"):
        values[str(port)] = tuple(int((port / "hw_counters" / name).read_text()) for name in
                                  ("out_of_sequence", "packet_seq_err", "local_ack_timeout_err"))
    return values


def distribute(cluster, request, layer):
    """Every rank's placed tensors of `layer` and the canonical activations, read on rank 0."""
    from tests.recipes.inkling2_checkpoint import Checkpoint, reference_activations
    from tests.recipes.test_inkling2_components_cuda import load_rank

    rank, world, device, layout = cluster["rank"], cluster["world"], cluster["device"], cluster["layout"]
    placements = sorted(layout.layer(layer).items())
    if rank == 0:
        checkpoint = Checkpoint(device)
        placed = [load_rank(checkpoint, layer, r, experts=True) for r in range(world)]
        for peer in range(1, world):
            for role, _ in placements:
                dist.send(placed[peer][role].cpu(), dst=peer)
        mine = placed[0]
        # The components test's layer set shares one cached canonical run.
        activations = reference_activations(request, checkpoint, LAYERS)
        entry = activations[layer]
        state = [len(entry["blocks"])]
    else:
        mine = {}
        for role, placement in placements:
            value = torch.empty(placement.local_shape, dtype=_dtype(placement.dtype))
            dist.recv(value, src=0)
            mine[role] = value.to(device)
        placed, checkpoint, entry, state = None, None, None, [None]
    dist.broadcast_object_list(state, src=0)
    tensors = []
    for index in range(state[0] + 3):
        if rank == 0:
            value = (entry["blocks"] + [entry["partial"], entry["attention_output"], entry["mlp_input"]])[index].cpu()
        else:
            value = torch.empty((1408, 4608), dtype=torch.bfloat16)
        dist.broadcast(value, src=0)
        tensors.append(value.to(device))
    activations = dict(blocks=tensors[:state[0]], partial=tensors[state[0]], attention_output=tensors[state[0] + 1],
                       mlp_input=tensors[state[0] + 2])
    return mine, placed, checkpoint, activations


def _dtype(name):
    return {"BF16": torch.bfloat16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn, "U8": torch.uint8}[name]


class Run:
    """Prepared prefill and decode graphs of one rank's layer, with their input and output storage."""

    def __init__(self, cluster, layer, weights):
        from ignition.components.paged_batch import Metadata
        from ignition.preparation.compilation import compile_forward
        from ignition.preparation.scratch import ScratchArena
        from ignition.preparation.tuning import RecordedPreparationSession
        from ignition.recipes.inkling2.decoder import Collectives, PreparedLayer, ResidualState
        from ignition.recipes.inkling2.pool import ContinuationPool, RankGeometry

        text = cluster["config"].text
        device, runtime, coordinator = cluster["device"], cluster["runtime"], cluster["coordinator"]
        self.world, self.rank, self.layer, self.text = cluster["world"], cluster["rank"], layer, text
        self.owned = PREFILL_ROWS // self.world
        self.pool = ContinuationPool.allocate(text, RankGeometry.of(text, self.world), 4, 32, device)
        self.metadata = {"prefill": Metadata.allocate(PREFILL_ROWS, 1, BLOCKS, device),
                         "decode": Metadata.allocate(DECODE_ROWS, DECODE_ROWS, BLOCKS, device)}
        sources = (layer + text.residual_block - 1) // text.residual_block
        rows = {"prefill": self.owned, "decode": DECODE_ROWS}
        self.inputs = {mode: [torch.zeros(count, text.hidden, dtype=torch.bfloat16, device=device)
                              for _ in range(sources + 1)] for mode, count in rows.items()}
        from b12x.moe import fused_moe as moe

        record = moe.quantized_input_record_layout(text.hidden, {"nvfp4": "nvfp4_four_over_six",
                                                                 "mxfp8": "mxfp8"}[weights.experts.format])[-1]
        self.arena = ScratchArena(device)
        self.session = RecordedPreparationSession(device=device, checkpoint="inkling-2-small",
            workload_schema="inkling2-layer-tp4-v1", numerical_policy="default", coordinator=coordinator,
            compile_workers=2)
        self.session.__enter__()
        self.session.register_scratch(self.arena)
        tokens = {"prefill": PREFILL_ROWS, "decode": DECODE_ROWS}
        self.layers = {}
        for mode in ("prefill", "decode"):
            collectives = Collectives.prepare(text, mode=mode, tokens=tokens[mode], record=record, runtime=runtime,
                                              coordinator=coordinator, session=self.session, name=f"{mode}.collectives")
            self.layers[mode] = PreparedLayer(layer, weights, text, self.pool, self.metadata[mode], collectives,
                                              mode=mode, tokens=tokens[mode], session=self.session,
                                              workspace=self.arena, name=f"layer{layer}.{mode}")
        self.session.tune(show_progress=False)
        self.arena.release_preparation()
        self.graphs, self.outputs = {}, {}
        pool = torch.cuda.graph_pool_handle()
        for mode in ("prefill", "decode"):
            def forward(mode=mode):
                state = ResidualState(tuple(self.inputs[mode][:-1]), self.inputs[mode][-1])
                updated, intermediates = self.layers[mode](state, self.metadata[mode])
                return updated.blocks, updated.partial, intermediates

            compiled = compile_forward(forward)
            self.clear()
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                compiled()
            torch.cuda.synchronize(device)
            self.clear()
            graph = torch.cuda.CUDAGraph()
            with runtime.capture(stream=stream), self.session.capture(), torch.cuda.graph(graph, pool=pool, stream=stream):
                self.outputs[mode] = compiled()
            self.graphs[mode] = (graph, compiled)
        self.session.freeze()
        self.clear()

    def clear(self):
        self.pool.backing.zero_()
        for value in self.pool.pages.values():
            value.zero_()

    def replay(self, mode):
        torch.cuda.synchronize()
        dist.barrier()
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        self.graphs[mode][0].replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
        return self.outputs[mode]

    def close(self):
        for graph, _ in self.graphs.values():
            graph.reset()
        self.session.__exit__(None, None, None)


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(item) for item in value)
    return value


def run_sequence(run, activations):
    """Prefill chunks, the fork and decode steps; returns this rank's outputs per row and step."""
    device = run.inputs["prefill"][0].device
    rank, owned = run.rank, run.owned
    sources = activations["blocks"] + [activations["partial"]]
    pages = torch.full((BLOCKS,), -1, dtype=torch.int32, device=device)
    table_a, table_b = pages.clone(), pages.clone()
    table_a[:12] = torch.arange(12, dtype=torch.int32, device=device)
    table_b[:12] = torch.tensor([*range(8), 20, 21, 22, 23], dtype=torch.int32, device=device)
    results = {"prefill": [], "decode": []}
    for start, length in CHUNKS:
        first = start + rank * owned
        for buffer, source in zip(run.inputs["prefill"], sources, strict=True):
            buffer.zero_()
            count = max(min(owned, start + length - first), 0)
            buffer[:count].copy_(source[first:first + count])
        run.metadata["prefill"].set(torch.tensor([start], device=device),
            torch.tensor([length], dtype=torch.int32, device=device), torch.tensor([1], device=device),
            table_a.view(1, -1), torch.zeros(1, PREFILL_ROWS, dtype=torch.int64, device=device),
            torch.tensor([1], device=device))
        results["prefill"].append((start, length, _cpu(run.replay("prefill"))))
        if start == 0:
            run.pool.backing[2].copy_(run.pool.backing[1])
            for value in run.pool.pages.values():
                value[20].copy_(value[8])
    for step in range(DECODE_STEPS):
        positions = (1405 + step, 1100 + step, 0)
        for buffer, source in zip(run.inputs["decode"], sources, strict=True):
            for row, position in enumerate(positions):
                buffer[row] = source[position] if row < 2 else 0
        run.metadata["decode"].set(torch.tensor(positions, device=device),
            torch.tensor([1, 1, 0], dtype=torch.int32, device=device), torch.tensor([1, 2, 0], device=device),
            torch.stack([table_a, table_b, table_a]), torch.zeros(DECODE_ROWS, 1, dtype=torch.int64, device=device),
            torch.tensor([2], device=device))
        results["decode"].append((positions[:2], _cpu(run.replay("decode"))))
    return results


@pytest.mark.parametrize("layer", LAYERS)
@torch.no_grad()
def test_tp4_layer_matches_the_canonical_four_rank_layer(cluster, request, layer):
    from ignition.recipes.inkling2.layer import layer_weights

    before = counters()
    mine, placed, checkpoint, activations = distribute(cluster, request, layer)
    weights = layer_weights(mine, cluster["config"], layer, cluster["world"])
    run = Run(cluster, layer, weights)
    try:
        results = run_sequence(run, activations)
    finally:
        run.close()
    gathered = [None] * cluster["world"]
    dist.all_gather_object(gathered, results)
    after = counters()
    moved = {port: (before[port], after[port]) for port in after if after[port] != before.get(port)}
    flags = [None] * cluster["world"]
    dist.all_gather_object(flags, moved)
    assert not any(flags), f"RoCE error counters moved during the run: {flags}"
    if cluster["rank"] == 0:
        check(cluster, layer, placed, activations, gathered)
    dist.barrier()


def assemble(gathered, owned):
    """Per field: the prefill rows each owner computed and every rank's replicated rows, by position."""
    world = len(gathered)
    rows = {}
    for chunk in range(len(CHUNKS)):
        start, length, _ = gathered[0]["prefill"][chunk]
        for rank in range(world):
            _, _, (blocks, partial, intermediates) = gathered[rank]["prefill"][chunk]
            first = start + rank * owned
            # Every rank holds every row of the chunk's partial sums, gathered expert inputs and amax.
            for position in range(start, start + length):
                row = position - start
                entry = rows.setdefault(position, {"ranks": {}})
                for name in ("attention_partial", "mlp_partial", "expert_partial", "records", "ids", "weights"):
                    entry["ranks"].setdefault(name, {})[rank] = intermediates[name][row]
                for name in ("local_amax", "amax"):
                    if intermediates[name] is not None:
                        entry["ranks"].setdefault(name, {})[rank] = intermediates[name].view(-1, 8)[row]
                if first <= position < first + owned:
                    local = position - first
                    entry["partial"] = partial[local]
                    entry["blocks"] = [block[local] for block in blocks]
                    for name in ("attention", "mlp_input", "mlp"):
                        entry[name] = intermediates[name][local]
    decoded = {}
    for step in range(DECODE_STEPS):
        positions, _ = gathered[0]["decode"][step]
        for row, position in enumerate(positions):
            entry = decoded.setdefault(position, {"ranks": {}})
            for rank in range(world):
                _, (blocks, partial, intermediates) = gathered[rank]["decode"][step]
                values = dict(partial=partial[row], blocks=[block[row] for block in blocks],
                              **{name: intermediates[name][row] for name in ("attention", "mlp_input", "mlp")})
                for name, value in values.items():
                    if rank == 0:
                        entry[name] = value
                    elif name == "blocks":
                        assert all(torch.equal(a, b) for a, b in zip(entry[name], value, strict=True))
                    else:
                        assert torch.equal(entry[name], value), (position, name)
                for name in ("attention_partial", "mlp_partial", "expert_partial", "records", "ids", "weights"):
                    entry["ranks"].setdefault(name, {})[rank] = intermediates[name][row]
                for name in ("local_amax", "amax"):
                    if intermediates[name] is not None:
                        entry["ranks"].setdefault(name, {})[rank] = intermediates[name].view(-1, 8)[row]
    return rows, decoded


def check(cluster, layer, placed, activations, gathered):
    from ignition.reference import inkling2 as ref
    from tests.recipes.inkling2_bounds import FP32, attention_output_bound, partial_bound, reference_parts
    from tests.recipes.test_inkling2_components_cuda import (_spacing, reference_weights, report,
                                                             residual_bound)

    text, world = cluster["config"].text, cluster["world"]
    device = cluster["device"]
    owned = PREFILL_ROWS // world
    rows, decoded = assemble(gathered, owned)
    prefill_positions = sorted(rows)
    decode_positions = sorted(decoded)
    assert prefill_positions == list(range(1405))

    def stack(entries, positions, name):
        return torch.stack([entries[p][name] for p in positions]).to(device)

    canonical = [reference_weights(values) for values in placed]
    state = ref.ResidualState(tuple(activations["blocks"]), activations["partial"])
    streams = ref.normalized_streams(canonical[0], text, state)
    tokens = streams[0].shape[0]
    positions = torch.arange(tokens, device=device)
    local = text.local(layer)
    label = f"layer {layer}"

    # Replicated values: every rank holds the same gathered expert inputs, routes and amax.
    for entries in (rows, decoded):
        for position, entry in entries.items():
            for name in ("records", "ids", "weights", "amax"):
                values = entry["ranks"].get(name)
                if values:
                    assert all(torch.equal(values[0], values[r]) for r in range(world)), (position, name)
            if entry["ranks"].get("local_amax"):
                maximum = torch.stack([entry["ranks"]["local_amax"][r] for r in range(world)]).amax(0)
                assert torch.equal(maximum, entry["ranks"]["amax"][0]), position

    # Reductions: each reduced output is the sum of the ranks' convolved partial sums, rounded once.
    for entries, positions_ in ((rows, prefill_positions), (decoded, decode_positions)):
        for name, partial_name in (("attention", "attention_partial"), ("mlp", "mlp_partial")):
            reduced = stack(entries, positions_, name).double()
            parts = torch.stack([torch.stack([entries[p]["ranks"][partial_name][r] for p in positions_])
                                 for r in range(world)]).to(device).double()
            total = parts.sum(0)
            limit = 2 * _spacing(total.abs()) + world * 2 * FP32 * parts.abs().sum(0)
            assert bool(((reduced - total).abs() <= limit).all()), (name, float((reduced - total).abs().max()))

    # Attention: the sum over ranks of each rank's bound, through the convolution and the reduction.
    attention_parts = [reference_parts(weights, text, streams, positions) for weights in canonical]
    weight = canonical[0].attention_convolution
    expected_attention = ref.reduce_ranks([ref.short_convolution(parts.partial, weight)[0] for parts in attention_parts])
    for name, positions_, entries, fp8 in (("prefill", prefill_positions, rows, True),
                                           ("decode", decode_positions, decoded, False)):
        index = torch.tensor(positions_, device=device)
        bound = torch.zeros(len(positions_), text.hidden, dtype=torch.float64, device=device)
        convolved_magnitude = torch.zeros_like(bound)
        for weights, parts in zip(canonical, attention_parts, strict=True):
            # The convolution reads earlier rows, so the bound covers the whole sequence.
            gated = attention_output_bound(parts, text, positions, local=local, fp8_query=fp8, e4m3_probabilities=fp8)
            limit = partial_bound(gated, parts.gated, weights.output, parts.partial)
            convolved = ref.short_convolution(limit, weight.double().abs())[0]
            magnitude = ref.short_convolution(parts.partial.double().abs(), weight.double().abs())[0]
            bound += (convolved + 2 * _spacing(magnitude) + 8 * FP32 * magnitude)[index]
            convolved_magnitude += magnitude[index]
        bound += 2 * _spacing(expected_attention[index].double().abs() + bound) + world * 2 * FP32 * convolved_magnitude
        actual = stack(entries, positions_, "attention")
        difference = report(f"{label} {name} reduced attention output", actual, expected_attention[index], bound)
        assert bool((difference <= bound).all())

    caches = [ref.LayerCache.empty(weights, text.head_dim) for weights in canonical]
    canonical_partial = ref.decoder_layer_ranks(canonical, text, layer, state, positions, caches)[0].partial
    for name, positions_, entries in (("prefill", prefill_positions, rows), ("decode", decode_positions, decoded)):
        index = torch.tensor(positions_, device=device)
        attention = stack(entries, positions_, "attention")
        partial = activations["partial"][index]
        blocks = [block[index] for block in activations["blocks"]]
        if layer % text.residual_block == 0:
            sources = (*blocks, partial, attention)
        else:
            sources = (*blocks, partial.float() + attention.float())
        mixture = ref.residual_mix(sources, canonical[0].mlp_query)
        expected_input = ref.rms_norm(mixture[0], canonical[0].mlp_norm, text.eps)
        limit = residual_bound(sources, canonical[0].mlp_query, 1e-6, mixture, (expected_input,), canonical[0].mlp_norm)[0]
        mlp_input = stack(entries, positions_, "mlp_input")
        difference = report(f"{label} {name} MLP input given the reduced attention output", mlp_input, expected_input,
                            limit)
        assert bool((difference <= limit).all())
        # Expert inputs: the canonical quantizer's bytes, and canonical routes away from ties.
        records = stack({p: {"records": entries[p]["ranks"]["records"][0]} for p in positions_}, positions_, "records")
        expert_format = cluster["config"].expert_format(layer)
        quantized = ref._quantize(mlp_input, expert_format)
        from ignition.components.moe import record_views

        values, scales, amax = record_views(records, text.hidden, {"nvfp4": "nvfp4_four_over_six",
                                                                   "mxfp8": "mxfp8"}[expert_format])
        if expert_format == "nvfp4":
            table = torch.tensor(ref._E2M1, dtype=torch.float32, device=device)
            codes = torch.stack((values & 15, values >> 4), -1).flatten(-2).long()
            assert torch.equal(table[codes], quantized.values) and torch.equal(amax, quantized.amax)
            assert torch.equal(scales.view(torch.float8_e4m3fn).float(), quantized.scales.float())
        else:
            assert torch.equal(values.view(torch.float8_e4m3fn).float(), quantized.values.float())
            assert torch.equal(scales, quantized.exponents)
        ids = stack({p: {"ids": entries[p]["ranks"]["ids"][0]} for p in positions_}, positions_, "ids").long()
        gammas = stack({p: {"w": entries[p]["ranks"]["weights"][0]} for p in positions_}, positions_, "w")
        ids_ref, gamma_ref = ref.route(mlp_input, canonical[0].router, canonical[0].router_bias,
                                       top_k=text.experts_per_token, route_scale=text.route_scale)
        logits = ref.matmul(mlp_input, canonical[0].router)
        logit_limit = 2 * text.hidden * FP32 * (mlp_input.double().abs() @ canonical[0].router.double().abs().t())
        selection = (torch.sigmoid(logits.double()) + canonical[0].router_bias.double()).sort(-1, descending=True).values
        margin = selection[:, 7] - selection[:, 8]
        same = (ids.sort(-1).values == ids_ref.sort(-1).values).all(-1)
        assert bool((same | (margin <= 0.5 * logit_limit.amax(-1))).all())
        print(f"{label} {name}: {int(same.sum())}/{len(positions_)} rows select the canonical experts")
        # Experts: the canonical four-rank experts on the same input and routes.
        partials, _ = canonical_experts(canonical, text, mlp_input, ids, gammas)
        mlp_weight = canonical[0].mlp_convolution
        actual_mlp = stack(entries, positions_, "mlp")
        if name == "prefill":
            expected_mlp = ref.reduce_ranks([ref.short_convolution(value, mlp_weight)[0] for value in partials])
            report(f"{label} prefill reduced expert output against the canonical experts on its input and routes",
                   actual_mlp, expected_mlp)
        # The residual update of the reduced outputs.
        updated = ref.ResidualState(tuple(blocks), partial).update(layer, attention, actual_mlp, text.residual_block)
        assert torch.equal(stack(entries, positions_, "partial"), updated.partial)
        report(f"{label} {name} residual partial against the canonical four-rank layer",
               stack(entries, positions_, "partial"), canonical_partial[index])


def canonical_experts(rank_weights, text, x, ids, gammas):
    """`ref.routed_experts_ranks` with supplied routes: each rank's partial output."""
    from ignition.reference import inkling2 as ref

    first = rank_weights[0]
    expert_format = first.expert_format
    quantized = ref._quantize(x, expert_format)
    routes = [torch.zeros(*ids.shape, x.shape[1], dtype=torch.bfloat16, device=x.device) for _ in rank_weights]
    for expert in ids.unique().tolist():
        if expert < 0:
            continue
        tokens, slots = (ids == expert).nonzero(as_tuple=True)
        rows = ref._rows(quantized, tokens)
        activations = [ref.situ(ref._project(weights, "w13", expert, rows), gammas[tokens, slots], text.gate_softcap,
                                text.up_softcap) for weights in rank_weights]
        amax = None
        if expert_format == "nvfp4":
            amax = torch.stack([activation.float().abs().amax(-1) for activation in activations]).amax(0)
        for weights, activation, output in zip(rank_weights, activations, routes, strict=True):
            output[tokens, slots] = ref._project(weights, "w2", expert, ref._quantize(activation, expert_format, amax))
    return [ref._combine(output) for output in routes], routes

