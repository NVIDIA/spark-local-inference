# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Packed Kimi sparse verification with tuned and external GEMM candidates on TP16."""

from datetime import timedelta
import os
from types import SimpleNamespace

import pytest
import torch

from tests.recipes.kimi31_attention_observations import (
    allocate_observations, copy_observations, poison_observations,
)

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
                              reason="requires sixteen explicitly assigned CUDA devices")

WORLD, HEADS, HIDDEN = 16, 6, 7168
RECORD, TOPK, PAGE = 576, 2048, 128
REQUESTS, MAX_ROWS, EPS = 32, 129, 1e-5


def _queries(device):
    heads = torch.arange(WORLD * HEADS, device=device)[:, None]
    columns = torch.arange(192, device=device)[None]
    projected = ((heads + 1) * (columns.remainder(13) - 6) / 2048).bfloat16()
    absorbed = torch.zeros((WORLD * HEADS, RECORD), device=device, dtype=torch.bfloat16)
    absorbed[:, :128] = projected[:, :128]
    absorbed[:, 512:] = projected[:, 128:]
    return projected, absorbed


def _weights(rank, device):
    def zeros(*shape):
        return torch.zeros(shape, device=device, dtype=torch.bfloat16)

    weights = dict(latent_gate=zeros(1536 + RECORD + HEADS * 128, HIDDEN),
        query_norm=torch.ones(1536, device=device, dtype=torch.bfloat16),
        kv_norm=torch.ones(512, device=device, dtype=torch.bfloat16),
        q_b=zeros(HEADS * 192, 1536), key=zeros(HEADS, 128, 512),
        value=zeros(HEADS, 512, 128), output=zeros(HIDDEN, HEADS * 128))
    signs = (1 - 2 * torch.arange(512, device=device).remainder(2)).bfloat16()
    weights["latent_gate"][:1536, 0] = 1
    weights["latent_gate"][1536:1792, 0] = signs[:256]
    weights["latent_gate"][1792:2048, 1] = signs[256:]
    weights["latent_gate"][2048:2112, 2] = (torch.arange(64, device=device).remainder(7) - 3) / 4
    projected, _ = _queries(device)
    weights["q_b"][:, 0] = projected[rank * HEADS:(rank + 1) * HEADS].reshape(-1)
    diagonal = torch.arange(128, device=device)
    weights["key"][:, diagonal, diagonal] = 1
    weights["value"][:, diagonal, diagonal] = 1
    diagonal = torch.arange(HEADS * 128, device=device)
    weights["output"][diagonal, diagonal] = 1
    weights.update({"indexer.wq_b": zeros(4096, 1536), "indexer.wk": zeros(128, HIDDEN),
        "indexer.weights_proj": zeros(32, HIDDEN),
        "indexer.k_norm.weight": torch.ones(128, device=device, dtype=torch.bfloat16),
        "indexer.k_norm.bias": zeros(128)})
    weights["indexer.wq_b"][64:128, 0] = signs[64:128]
    weights["indexer.wk"][:64, 0] = signs[:64]
    weights["indexer.wk"][64:, 1] = signs[64:128]
    weights["indexer.weights_proj"][0, 0] = 1
    return weights


def _features(request, positions, revision=0):
    return torch.stack((torch.ones_like(positions),
        .25 + (positions * 17 + request * 29).remainder(127) / 64 + revision * 4,
        ((positions * 7 + request * 11).remainder(127) - 63) / 32 + revision * 2), -1).bfloat16()


def _reference_records(features):
    signs = (1 - 2 * torch.arange(512, device=features.device).remainder(2)).float()
    raw = torch.cat((features[:, :1].float() * signs[:256],
                     features[:, 1:2].float() * signs[256:]), -1).bfloat16().float()
    normalized = (raw * torch.rsqrt(raw.square().mean(-1, keepdim=True) + EPS)).bfloat16()
    tail = (features[:, 2:3].float()
        * (torch.arange(64, device=features.device).remainder(7) - 3) / 4).bfloat16()
    records = torch.cat((normalized, tail), -1).float().clamp(-448, 448).to(torch.float8_e4m3fn)
    raw = torch.cat((features[:, :1].float() * signs[:64],
                     features[:, 1:2].float() * signs[64:128]), -1).bfloat16().float()
    centered = raw - raw.mean(-1, keepdim=True)
    key = (centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + EPS)).bfloat16().float()
    scale = (key.abs().amax(-1).clamp_min(1e-4).double() / 448).float()
    inverse = (1 / scale.double()).float()
    quantized = (key * inverse[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return records, quantized, scale


def _write_reference(cache, index, table, rank, positions, features):
    owned = positions.remainder(WORLD) == rank
    positions, features = positions[owned], features[owned]
    if not positions.numel():
        return
    local = positions // WORLD
    slots = table[local // PAGE].long() * PAGE + local.remainder(PAGE)
    records, keys, scales = _reference_records(features)
    cache.view(-1, RECORD)[slots] = records
    subpages, offsets = slots // 64, slots.remainder(64)
    columns = offsets[:, None] * 128 + torch.arange(128, device=cache.device)[None]
    index[subpages[:, None], columns] = keys.view(torch.uint8)
    columns = 8192 + offsets[:, None] * 4 + torch.arange(4, device=cache.device)[None]
    index[subpages[:, None], columns] = scales.contiguous().view(torch.uint8).view(-1, 4)


class _TargetForward:
    def __init__(self, variant, source, *, observations=None):
        for field in ("starts", "lengths", "states", "tables", "tokens", "live", "operation", "width"):
            setattr(self, field, getattr(variant, field))
        self.source = source[:variant.capacity]
        self.output = torch.empty_like(self.source)
        self.observations = observations
        self.positions = None if observations is None else torch.empty_like(variant.metadata.positions)
        self.query_start_loc = None if observations is None else torch.empty_like(variant.metadata.query_start_loc)

    def __call__(self):
        from ignition.components.paged_batch import Metadata

        metadata = Metadata.pack(self.starts, self.lengths, self.states, self.tables, self.tokens, self.live,
            token_capacity=self.source.shape[0], sample_count=self.tokens.numel(), max_query_tokens=self.width)
        if self.observations is None:
            self.output.copy_(self.operation(self.source, metadata))
        else:
            result = self.operation._run(self.source, metadata)
            copy_observations(self.observations, result)
            self.positions.copy_(metadata.positions)
            self.query_start_loc.copy_(metadata.query_start_loc)
            self.output.copy_(result.output)
        return self.output


@torch.no_grad()
def _rank(rank, rendezvous, compile_decoder, force_external, cache_root):
    from functools import partial
    from unittest.mock import patch
    import torch.distributed as dist

    from ignition.components.paged_batch import Metadata
    from ignition.preparation import capture as capture_module
    from ignition.preparation.aot import BoundForward, ForwardCache
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.collectives import CollectivePreparation, create_head_exchange
    from ignition.preparation.compilation import _OPTIONS
    from ignition.preparation.execution import GraphSignature
    from ignition.preparation.core_storage import CoreStorage
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi31.attention import PreparedDSA
    from ignition.recipes.kimi31.dcp import AttentionConstants, ContextParallel, ContextShards

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=WORLD,
                            timeout=timedelta(minutes=15))
    heads = candidates = None
    graphs = []
    compiled = {}
    try:
        with CoreStorage(rank=rank, world=WORLD, device=rank) as storage:
            page_count = REQUESTS * 2 + 1
            tables = ((torch.arange(REQUESTS * 2, device=device) * 7 + 3) % page_count).int().view(REQUESTS, 2)
            cache = torch.zeros((page_count, PAGE, RECORD), device=device, dtype=torch.float8_e4m3fn)
            index = torch.zeros((page_count * 2, 64 * 132), device=device, dtype=torch.uint8)
            expected_cache, expected_index = torch.empty_like(cache), torch.empty_like(index)
            source = torch.zeros((MAX_ROWS, HIDDEN), device=device, dtype=torch.bfloat16)
            history = torch.zeros((REQUESTS, 4096, 3), device=device, dtype=torch.bfloat16)
            initial = [2047, 1021, 15, 0, *(3 + request % 7 for request in range(4, REQUESTS))]
            expected_cache.fill_(3)
            expected_index.fill_(0xA7)
            for request, length in enumerate(initial):
                positions = torch.arange(length, device=device)
                values = _features(request, positions)
                history[request, :length] = values
                _write_reference(expected_cache, expected_index, tables[request], rank, positions, values)
            coordinator = CollectivePreparation(dist.group.WORLD)
            heads = create_head_exchange(dist.group.WORLD, device, max_rows=MAX_ROWS,
                heads=WORLD * HEADS, head_dim=512, query_head_dim=RECORD, ipc=storage.peer_storage)
            candidates = create_head_exchange(dist.group.WORLD, device, max_rows=MAX_ROWS,
                heads=WORLD, head_dim=TOPK * 4, ipc=storage.peer_storage)
            constants = AttentionConstants.allocate(device, query_capacity=MAX_ROWS)
            context = ContextParallel(ContextShards.on(rank, WORLD, device), heads, coordinator, {}, constants,
                                      candidates=candidates)
            weights = _weights(rank, device)
            _, query = _queries(device)
            query = query.float().clamp(-448, 448).to(cache.dtype)
            local_query = query[rank * HEADS:(rank + 1) * HEADS].float()
            index_query = torch.zeros(128, device=device)
            index_query[64:] = (1 - 2 * torch.arange(64, device=device).remainder(2)) * 256
            head_weight = torch.tensor(32 ** -.5, device=device, dtype=torch.bfloat16).float()
            head_weight = ((head_weight * 32 ** .5) * (1 / 256)) * (1 / 64)
            with RecordedPreparationSession(device=device, checkpoint="kimi31-packed-target-verification",
                    workload_schema="sparse-target-dcp", numerical_policy="query-pow2-key-fp32",
                    autotune_effort="off", compile_workers=WORLD, coordinator=coordinator) as session:
                arena = ScratchArena(device)
                session.register_scratch(arena)
                buffers, variants = {}, {}
                for name, batch, width, capacity in (("verify129", 32, 8, 129), ("verify13", 4, 8, 13),
                                                    ("width1", 4, 1, 4), ("decode", 1, 1, 1)):
                    metadata = Metadata.allocate(capacity, batch, 2, device, samples=batch * width,
                                                 max_query_tokens=width)
                    variant = SimpleNamespace(name=name, batch=batch, width=width, capacity=capacity, metadata=metadata,
                        starts=torch.zeros(batch, device=device, dtype=torch.int64),
                        lengths=torch.zeros(batch, device=device, dtype=torch.int32),
                        states=torch.arange(1, batch + 1, device=device), tables=tables[:batch],
                        tokens=torch.arange(batch * width, device=device).view(batch, width),
                        live=torch.tensor([batch], device=device, dtype=torch.int32))
                    variant.operation = PreparedDSA(weights, source[:capacity], metadata, cache, index_cache=index,
                        eps=EPS, mode="decode" if width == 1 else "verify", session=session, name=name,
                        workspace=arena, buffers=buffers, context=context)
                    variants[name] = variant
                for variant in variants.values():
                    part = variant.operation.parts[variant.capacity]
                    assert part["starts"].numel() == variant.capacity + 1
                session.tune(show_progress=False)
                buffers.clear()
                for variant in variants.values():
                    result = variant.operation._run(source[:variant.capacity], variant.metadata)
                    variant.observations = allocate_observations(result)
                    variant.forward = _TargetForward(variant, source)
                    variant.diagnostic = _TargetForward(variant, source, observations=variant.observations)
                    variant.metadata = None
                del result, metadata
                functions = {name: variant.forward for name, variant in variants.items()}
                functions.update({name + "-observed": variant.diagnostic for name, variant in variants.items()})
                if compile_decoder:
                    signatures = {name: GraphSignature(
                        kind="decode" if name == "decode" else "speculative",
                        batch_capacity=variant.batch, token_capacity=variant.width,
                        inputs=("query_lengths", "request_state_rows", "page_tables", "live_rows"),
                        speculation_enabled=name != "decode",
                        packed_token_capacity=0 if name == "decode" else variant.capacity,
                        draft_tokens=None if name == "decode" else variant.width - 1)
                        for name, variant in variants.items()}
                    signatures.update({name + "-observed": value for name, value in tuple(signatures.items())})
                    options = dict(_OPTIONS)
                    if force_external:
                        # Exercise mutation dependencies when a deferred GEMM selects an external kernel.
                        options["test_configs.force_extern_kernel_in_multi_template"] = True
                    with patch.object(capture_module, "ForwardCache", partial(ForwardCache, options=options)):
                        captured, compiled = capture_forwards(functions, signatures, session=session,
                            device=device, stream=torch.cuda.Stream(device=device), pool=storage.graph_pool.id,
                            compile_decoder=True, coordinator=coordinator, cache_root=cache_root,
                            warmup_storage=storage)
                    graphs.extend(captured.values())
                    assert compiled.keys() == functions.keys()
                    for name, variant in variants.items():
                        assert isinstance(compiled[name], BoundForward)
                        assert compiled[name].owner is variant.forward
                        assert compiled[name].artifact is not None and compiled[name].pending is None
                        variant.graph = captured[name]
                        variant.observed_graph = captured[name + "-observed"]
                else:
                    captured = {}
                    for name, forward in functions.items():
                        forward()
                        torch.cuda.synchronize(device)
                        stream = torch.cuda.Stream(device=device)
                        stream.wait_stream(torch.cuda.current_stream(device))
                        graph = torch.cuda.CUDAGraph()
                        with session.capture(), torch.cuda.graph(graph, stream=stream):
                            forward()
                        captured[name] = graph
                        graphs.append(graph)
                    for name, variant in variants.items():
                        variant.graph = captured[name]
                        variant.observed_graph = captured[name + "-observed"]
                session.freeze()
                cache.copy_(expected_cache)
                index.copy_(expected_index)

                def replay(name, starts, lengths, *, revision=1, future_revision=None):
                    variant = variants[name]
                    variant.starts.copy_(torch.tensor(starts, device=device))
                    variant.lengths.copy_(torch.tensor(lengths, device=device, dtype=torch.int32))
                    source.zero_()
                    source[:, 0] = 1
                    source[:, 1] = 17
                    source[:, 2] = -19
                    offset, rows = 0, []
                    for request, (start, length) in enumerate(zip(starts, lengths, strict=True)):
                        positions = torch.arange(start, start + length, device=device)
                        values = _features(request, positions, revision)
                        if future_revision is not None and length > 2:
                            values[2:] = _features(request, positions[2:], future_revision)
                        history[request, start:start + length] = values
                        source[offset:offset + length, :3] = values
                        _write_reference(expected_cache, expected_index, tables[request], rank, positions, values)
                        rows.extend((request, position) for position in range(start, start + length))
                        offset += length
                    assert offset <= variant.capacity
                    arena.tensor.fill_(255)
                    poison_observations(variant.observations)
                    variant.forward.output.fill_(19)
                    observed = variant.observations.last_stage
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    before = torch.cuda.memory_stats(device)
                    variant.graph.replay()
                    torch.cuda.synchronize(device)
                    after = torch.cuda.memory_stats(device)
                    for field in ("allocation.all.allocated", "allocation.all.freed",
                                  "segment.all.allocated", "segment.all.freed"):
                        assert before[field] == after[field], (name, field)
                    torch.testing.assert_close(cache.view(torch.uint8), expected_cache.view(torch.uint8), rtol=0, atol=0)
                    torch.testing.assert_close(index, expected_index, rtol=0, atol=0)
                    before = torch.cuda.memory_stats(device)
                    variant.observed_graph.replay()
                    torch.cuda.synchronize(device)
                    after = torch.cuda.memory_stats(device)
                    for field in ("allocation.all.allocated", "allocation.all.freed",
                                  "segment.all.allocated", "segment.all.freed"):
                        assert before[field] == after[field], (name + "-observed", field)
                    selected = observed.selected.cpu()
                    root = [selected if rank == 0 else None]
                    dist.broadcast_object_list(root, src=0)
                    assert torch.equal(selected, root[0])
                    torch.testing.assert_close(cache.view(torch.uint8), expected_cache.view(torch.uint8), rtol=0, atol=0)
                    torch.testing.assert_close(index, expected_index, rtol=0, atol=0)
                    torch.testing.assert_close(observed.gathered.view(variant.capacity, WORLD * HEADS, RECORD), query[None].expand(variant.capacity, -1, -1),
                                              rtol=0, atol=0)
                    assert variant.diagnostic.positions.cpu().tolist() == [position for _, position in rows] + [-1] * (variant.capacity - offset)
                    ends, total = [0], 0
                    for length in lengths:
                        total += length
                        ends.append(total)
                    assert variant.diagnostic.query_start_loc.cpu().tolist() == ends
                    for row, (request, position) in enumerate(rows):
                        chosen = selected[row][selected[row] >= 0].long().to(device)
                        assert len(chosen) == len(set(chosen.cpu().tolist())) == min(position + 1, TOPK)
                        assert ((chosen >= 0) & (chosen <= position)).all(), (name, row)
                        records, keys, scales = _reference_records(history[request, :position + 1])
                        scores = torch.mv(keys.float() * scales[:, None], index_query).relu() * head_weight
                        score_mask = observed.selected[row] >= 0
                        # IDs follow rank zero; diagnostic scores retain each rank's local tie ordering.
                        if rank == 0:
                            torch.testing.assert_close(observed.selected_scores[row][score_mask], scores[chosen],
                                                      rtol=3e-5, atol=3e-4)
                        cutoff = scores.topk(len(chosen)).values[-1]
                        assert (scores[chosen] >= cutoff - 3e-4).all(), (name, row)
                        required = torch.where(scores > cutoff + 3e-4)[0]
                        assert set(required.cpu().tolist()) <= set(chosen.cpu().tolist())
                        local = chosen[chosen.remainder(WORLD) == rank]
                        count = len(local)
                        assert observed.counts[row].item() == count
                        physical = tables[request, local // (WORLD * PAGE)].long() * PAGE + (local // WORLD) % PAGE
                        torch.testing.assert_close(observed.slots[row, :count].long(), physical, rtol=0, atol=0)
                        assert (observed.slots[row, count:] == -1).all()
                        selected_records = records[chosen].float()
                        logits = local_query @ selected_records.T * 192 ** -.5
                        attended = logits.softmax(-1) @ selected_records[:, :512]
                        torch.testing.assert_close(variant.observations.attended[row].float(), attended, rtol=.03, atol=.002)
                        expected = (attended.bfloat16()[:, :128].float() * .5).bfloat16().reshape(-1)
                        output = variant.forward.output[row]
                        torch.testing.assert_close(output[:HEADS * 128].float(), expected.float(), rtol=.03, atol=.002)
                        torch.testing.assert_close(variant.observations.output[row, :HEADS * 128].float(),
                                                  expected.float(), rtol=.03, atol=.002)
                        assert not variant.observations.output[row, HEADS * 128:].count_nonzero()
                        assert not output[HEADS * 128:].count_nonzero()
                        if not count:
                            assert observed.lse[row].isneginf().all()
                        elif row == 0 or request != rows[row - 1][0]:
                            local_records = records[local].float()
                            local_logits = query.float() @ local_records.T * 192 ** -.5
                            torch.testing.assert_close(observed.lse[row], local_logits.logsumexp(-1),
                                                      rtol=3e-5, atol=3e-4)
                            expected_partial = local_logits.softmax(-1) @ local_records[:, :512]
                            torch.testing.assert_close(observed.partial[row].float(), expected_partial, rtol=.03, atol=.002)
                    assert (selected[offset:] == -1).all()
                    assert not observed.counts[offset:].count_nonzero()
                    assert observed.lse[offset:].isneginf().all()
                    assert not variant.observations.attended[offset:].count_nonzero()
                    assert not variant.forward.output[offset:].count_nonzero()
                    assert torch.isfinite(variant.forward.output).all()
                    assert not variant.observations.output[offset:].count_nonzero()
                    assert torch.isfinite(variant.observations.output).all()
                    dist.barrier()
                    return variant.observations.attended.clone(), selected, rows

                limits = [8, 5, 1, 0, *([4] * 28)]
                first, first_selection, rows = replay("verify129", initial, limits)
                changed, changed_selection, _ = replay("verify129", initial, limits, future_revision=2)
                assert not torch.equal(first, changed)
                previous = None
                for row, (request, _) in enumerate(rows):
                    if request != previous:
                        first_row = row
                    if row - first_row < 2 and rows[row][1] < TOPK:
                        torch.testing.assert_close(changed[row], first[row], rtol=0, atol=0)
                        assert torch.equal(changed_selection[row].sort().values, first_selection[row].sort().values)
                    previous = request
                accepted = [1, 3, 1, 0]
                starts = [start + count for start, count in zip(initial[:4], accepted, strict=True)]
                replay("width1", starts, [1, 1, 0, 1], revision=3)
                starts = [start + count for start, count in zip(starts, [1, 1, 0, 1], strict=True)]
                replay("decode", starts[:1], [1], revision=4)
                starts[0] += 1
                replay("verify13", starts, [2, 8, 1, 0], revision=5)
                replay("width1", [starts[0] + 1, starts[1] + 2, starts[2], starts[3]], [1, 0, 0, 0], revision=6)
                replay("verify129", initial, [0] * REQUESTS)
                assert constants.pages.shape == (128, TOPK)
                torch.testing.assert_close(constants.starts, torch.arange(MAX_ROWS + 1, device=device, dtype=torch.int32),
                                          rtol=0, atol=0)
                for graph in graphs:
                    graph.reset()
                graphs.clear()
                compiled.clear()
            heads.close()
            candidates.close()
            heads = candidates = None
    except BaseException:
        import traceback

        traceback.print_exc()
        os._exit(1)
    finally:
        for graph in graphs:
            graph.reset()
        compiled.clear()
        if heads is not None:
            heads.close()
        if candidates is not None:
            candidates.close()
        dist.destroy_process_group()


@pytest.mark.parametrize(("compile_decoder", "force_external"), [(False, False), (True, False), (True, True)],
                         ids=["eager", "compiled", "compiled-external"])
def test_packed_sparse_target_verification_and_rejected_suffix_continuation_on_tp16(tmp_path, compile_decoder,
                                                                                force_external):
    import torch.multiprocessing as mp

    assert torch.cuda.device_count() == WORLD, "assign exactly sixteen CUDA devices"
    mp.spawn(_rank, args=((tmp_path / "sparse-target-verification").as_uri(), compile_decoder, force_external,
                         str(tmp_path / "compiled-forwards")), nprocs=WORLD, join=True)
