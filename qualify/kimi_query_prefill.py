"""Qualify query-sharded sparse indexing with checkpoint projections on TP16.

The fixture compares an unordered distributed candidate fold followed by a
rank-zero byte broadcast with production query-row ownership. The reference
composition is explicit so production merge changes do not redefine it.
It uses checkpoint index weights and deterministic synthetic hidden states,
preserves stored key/scale bytes, and reports selection and attention errors.
Timings cover index projections, cache writes, selection and communication;
they are not model throughput measurements.
"""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import tempfile

WORLD, ROWS, TOPK, MAXIMUM = 16, 4096, 2048, 16384


def configure(checkpoint, path):
    from ignition.cli.config import Configuration
    from ignition.cli.environment import apply
    from ignition.cli.preparation import Host, collective_tuning_mode, resolve_model, tuning_effort

    configuration = Configuration.load(path)
    model, gpus = resolve_model(configuration, str(checkpoint)), Host.discover().gpus
    if model.recipe.name != "kimi31" or len(gpus) != WORLD:
        raise ValueError("the qualifier requires Kimi K3.1 and sixteen assigned GPUs")
    if collective_tuning_mode(configuration, model) != "static":
        raise ValueError("the qualifier requires static collective tuning")
    apply(configuration, gpus, model.recipe.b12x_environment)
    return dict(recipe=model.recipe.name, checkpoint=str(checkpoint), config=str(configuration.path),
        b12x={key: value for key, value in os.environ.items() if key.startswith("B12X_")},
        autotune_effort=tuning_effort(configuration), gpu_uuids=[gpu.uuid for gpu in gpus])


def load_weights(checkpoint, layer, device):
    from safetensors import safe_open

    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    fields = ("wq_b", "wk", "weights_proj", "k_norm.weight", "k_norm.bias")
    result = {}
    for field in fields:
        name = f"language_model.model.layers.{layer}.self_attn.indexer.{field}"
        if not field.endswith((".weight", ".bias")):
            name += ".weight"
        with safe_open(checkpoint / index[name], framework="pt", device=str(device)) as file:
            result["indexer." + field] = file.get_tensor(name)
    return result


class BroadcastIndexSelection:
    """Reference prefill selection with unordered top-k and rank-zero agreement.

    Rank-local FP8 scoring, underfull compaction, row top-k and byte broadcast
    are independently prepared components. This fixture deliberately does not
    instantiate the recipe's distributed-selection composition.
    """

    def __init__(self, query, weights, cache, metadata, shards, context, buffers, *,
                 start, end, session, name, workspace):
        import torch
        from ignition.components.candidate_rows import PreparedCandidateRows
        from ignition.components.collectives import PreparedAllGather, PreparedBroadcast
        from ignition.components.fp8_indexer import PreparedFP8Indexer
        from ignition.components.row_topk import PreparedRowTopK
        from ignition.recipes.kimi31.indexer import SharedBuffers

        rows, width = end - start, metadata.token_page_table.shape[1]
        self.positions, self.tables = metadata.positions[start:end], metadata.token_page_table[start:end]
        self.shards = shards
        storage = SharedBuffers(buffers, query.device)
        self.pages = storage.buffer("pages", (1, width * 2), torch.int32)
        self.lengths = storage.buffer("local_lengths", (rows,), torch.int32)
        self.local = storage.buffer("local_ids", (rows, TOPK), torch.int32)
        self.scores = storage.buffer("local_scores", (rows, TOPK), torch.float32)
        self.packed = storage.buffer("candidates", (rows, TOPK, 2), torch.float32)
        self.gathered = storage.buffer("gathered", (rows, TOPK * shards.world, 2), torch.float32)
        self.global_scores = storage.buffer("global_scores", (rows, TOPK * shards.world), torch.float32)
        self.global_ids = storage.buffer("global_ids", (rows, TOPK * shards.world), torch.int32)
        self.fold_lengths = storage.buffer("fold_lengths", (rows,), torch.int32)
        self.active = storage.buffer("active", (rows,), torch.bool)
        self.selected_scores = storage.buffer("selected_scores", (rows, TOPK), torch.float32)
        self.selected = storage.buffer("selected", (rows, TOPK), torch.int32)
        self.indexer = PreparedFP8Indexer(query[start:end].view(rows, 32, 128), weights[start:end],
            cache, self.pages.expand(rows, -1), self.lengths, None, self.local,
            mode="prefill", session=session, name=name + ".local", workspace=workspace,
            buffers=SharedBuffers(buffers, query.device, prefix="fp8."), topk=TOPK, scores=self.scores,
            prefill_records_per_call=131072, head_weight_scale=1., hadamard=False, key_power_of_two=False)
        self.gather = PreparedAllGather(self.packed.view(torch.bfloat16).view(rows, TOPK * 4),
            self.gathered.view(torch.bfloat16).view(rows, TOPK * 4 * shards.world), runtime=context.candidates,
            session=session, coordinator=context.coordinator, name=name + ".candidates", dependencies=(cache,))
        self.candidate_rows = PreparedCandidateRows(self.gathered, self.active,
            self.global_scores, self.global_ids, self.fold_lengths, topk=TOPK,
            buffers=SharedBuffers(buffers, query.device, prefix="candidate_rows."))
        self.fold = PreparedRowTopK(self.global_scores, self.fold_lengths, self.global_ids,
            self.selected_scores, self.selected, session=session, name=name + ".fold")
        self.broadcast = PreparedBroadcast((self.selected,), runtime=context.candidates, session=session,
            coordinator=context.coordinator, name=name + ".selected",
            buffers=SharedBuffers(buffers, query.device, prefix="consensus."))

    def prepare_local(self):
        import torch
        from ignition.recipes.kimi31.dcp import index_pages, pack_candidates

        index_pages(self.tables[:1], self.pages)
        self.lengths.copy_(self.shards.local_tokens((self.positions + 1).clamp(min=0)).to(torch.int32))
        self.indexer()
        pack_candidates(self.local, self.scores, self.packed, self.shards)
        return self.packed

    def __call__(self):
        import torch

        self.prepare_local()
        self.gather(self.packed.view(torch.bfloat16).view(self.packed.shape[0], -1))
        self.active.copy_(self.positions >= 0)
        self.candidate_rows()
        self.fold()
        self.broadcast()
        return self.selected


class SelectionFixture:
    def __init__(self, weights, source, latent, cache, metadata, context, session, arena, *, eps):
        import torch
        from ignition.recipes.kimi31.indexer import PreparedDSAIndexer

        self.source, self.metadata, self.cache = source, metadata, cache
        self.index = PreparedDSAIndexer(weights, source, latent, metadata, cache, context.shards,
            context=context, eps=eps, mode="prefill", session=session, name="index", workspace=arena,
            buffers={}, latent_cache=cache)
        # Head weights may be a strided slice of a fused key/gate projection.
        self.packed_weights = torch.empty((ROWS, 160), dtype=torch.bfloat16, device=source.device)
        self.index.sharded.weights = self.packed_weights[:, 128:]
        control_buffers = {}
        self.control = tuple(BroadcastIndexSelection(self.index.query, self.index.head_weights, cache,
            metadata, context.shards, context, control_buffers, start=begin, end=begin + 128,
            session=session, name=f"control.rows{begin}", workspace=arena) for begin in range(0, ROWS, 128))
        self.baseline = torch.empty((ROWS, TOPK), dtype=torch.int32, device=source.device)
        self.candidate = self.index.sharded.selected

    def run(self, candidate):
        self.index.project(self.source)
        if candidate:
            self.packed_weights[:, 128:].copy_(self.index.head_weights)
            return self.index.sharded()
        for part, begin in zip(self.control, range(0, ROWS, 128), strict=True):
            self.baseline[begin:begin + 128].copy_(part())
        return self.baseline


def _reference_module():
    """Reuse the independently specified MLA probability-encoding test oracle."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "tests/components/test_candidate_rows_cuda.py"
    spec = importlib.util.spec_from_file_location("kimi_selection_probability_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.probability_encoding_reference


def _keys(cache, visible):
    import torch

    ids = torch.arange(visible, device=cache.device)
    pages, rows = ids // 64, ids % 64
    keys = cache[:, :8192].reshape(-1, 64, 128)[pages, rows].view(torch.float8_e4m3fn).float()
    scales = cache[:, 8192:].contiguous().view(torch.float32)[pages, rows]
    return keys * scales[:, None]


def _scorer_row(indexer, row):
    """Read the prepared scorer's single-supertile FP32 values in token order."""
    scratch = indexer.selection.binding.runtime.scratch
    # FI's tiled layout is (query tile, key tile, query row, key column).
    # Read only a complete single-window binding; larger histories require
    # copying each window's scores before another window reuses the scratch.
    width = scratch.paged_tile_logits_k_rows
    if width != indexer.pages.shape[1] * 64:
        raise AssertionError("score diagnostics require one complete supertile")
    block = scratch.prefill_block_k
    values = scratch.get_indexer_contiguous_tile_logits()
    return values[:((indexer.query.shape[0] + 31) // 32) * width * 32].view(
        -1, width // block, 32, block)[row // 32, :, row % 32].flatten()


class WindowScores:
    """Inspect the existing prepared prefill scorer one complete supertile at a time.

    Diagnostic calls use the same BF16 query/head-weight values and stored key
    bytes. They avoid inferring earlier supertile scores from overwritten scratch.
    This operator is outside both captured performance variants.
    """

    def __init__(self, cache, *, session, arena, maximum):
        import torch
        from ignition.components.fp8_indexer import PreparedFP8Indexer
        from ignition.recipes.kimi31.indexer import SharedBuffers

        device = cache.device
        self.width = min(maximum, 131072)
        self.query = torch.empty((128, 32, 128), dtype=torch.bfloat16, device=device)
        self.weights = torch.empty((128, 32), dtype=torch.bfloat16, device=device)
        self.pages = torch.zeros((1, self.width // 64), dtype=torch.int32, device=device)
        self.lengths = torch.zeros(128, dtype=torch.int32, device=device)
        self.output = torch.empty((128, TOPK), dtype=torch.int32, device=device)
        self.indexer = PreparedFP8Indexer(self.query, self.weights, cache, self.pages.expand(128, -1),
            self.lengths, None, self.output, mode="prefill", session=session, name="diagnostic_score_window",
            workspace=arena, buffers=SharedBuffers({}, device), prefill_records_per_call=131072,
            topk=TOPK, head_weight_scale=1., hadamard=False, key_power_of_two=False)

    def __call__(self, original, row):
        import torch

        if original.pages.shape[1] * 64 <= 131072:
            return _scorer_row(original, row)
        self.query.copy_(original.query)
        self.weights.copy_(original.weights)
        pieces = []
        for begin in range(0, original.pages.shape[1], self.pages.shape[1]):
            end = min(begin + self.pages.shape[1], original.pages.shape[1])
            self.pages.fill_(-1)
            self.pages[:, :end - begin].copy_(original.pages[:1, begin:end])
            self.lengths.copy_((original.selection.inputs["cache_lengths"] - begin * 64).clamp(0, self.width))
            self.indexer()
            pieces.append(_scorer_row(self.indexer, row)[:(end - begin) * 64].clone())
        return torch.cat(pieces)


def _worker(rank, rendezvous, checkpoint, destination, policy, compiled, maximum):
    for key, value in policy["b12x"].items():
        if os.environ.get(key) != value:
            raise RuntimeError("child dependency policy differs from the configured recipe")
    import torch
    import torch.distributed as dist
    from types import SimpleNamespace
    from b12x.attention import dense_mla
    from ignition import runtime_patches
    from ignition.components.dense_mla import PreparedDenseMLA
    from ignition.components.fp8_indexer import write_fp8_index_keys
    from ignition.components.shared_pages import SharedPages
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime, create_head_exchange
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.core_storage import CoreStorage
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.kimi3.dcp import ContextShards
    from ignition.recipes.kimi31.dcp import ContextParallel, AttentionConstants

    runtime_patches.apply()
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD, init_method=rendezvous,
                            timeout=timedelta(minutes=20))
    graphs, runtimes, stage = [], [], None
    reports = []
    try:
        with CoreStorage(rank=rank, world=WORLD, device=rank) as storage:
            coordinator = CollectivePreparation(dist.group.WORLD)
            control_runtime = create_head_exchange(dist.group.WORLD, device,
                max_rows=128, heads=WORLD, head_dim=8192, ipc=storage.peer_storage)
            selected_runtime = create_head_exchange(dist.group.WORLD, device,
                max_rows=ROWS // WORLD, heads=WORLD, head_dim=4096, ipc=storage.peer_storage)
            barrier = create_all_reduce_runtime(dist.group.WORLD, device,
                max_tokens=1, hidden_size=8, collective_tuning="static", ipc=storage.peer_storage)
            runtimes = [control_runtime, selected_runtime, barrier]
            stage = SharedPages(dist.group.WORLD, device, maximum // WORLD * 132, ipc=storage.peer_storage)
            logical_pages, physical_pages = maximum // (WORLD * 128), maximum // (WORLD * 128) + 3
            cache = torch.zeros((physical_pages * 2, 64 * 132), dtype=torch.uint8, device=device)
            history = torch.empty((maximum // 64, 64 * 132), dtype=torch.uint8, device=device)
            table = ((torch.arange(logical_pages, device=device) * 3 + 2) % physical_pages).int()
            positions = torch.full((ROWS,), -1, dtype=torch.int64, device=device)
            metadata = SimpleNamespace(page_table=table[None], token_page_table=table[None].expand(ROWS, -1),
                positions=positions, starts=torch.zeros(1, dtype=torch.int64, device=device),
                lengths=torch.zeros(1, dtype=torch.int32, device=device))
            context = ContextParallel(ContextShards.on(rank, WORLD, device), heads=None,
                coordinator=coordinator, peers={}, attention_constants=AttentionConstants.allocate(device),
                candidates=control_runtime, stage_runtime=barrier, index_staging=stage,
                index_records=history, index_columns=torch.arange(maximum // 64, dtype=torch.int32, device=device),
                index_signals=tuple(torch.zeros((1, 8), dtype=torch.bfloat16, device=device) for _ in range(4)),
                selections=selected_runtime)
            weights = load_weights(Path(checkpoint), 3, device)
            config = json.loads((Path(checkpoint) / "config.json").read_text())
            text = config.get("text_config", config)
            eps = text["rms_norm_eps"]
            generator = torch.Generator(device=device).manual_seed(731)
            source = torch.randn((ROWS, 7168), dtype=torch.bfloat16, device=device, generator=generator)
            latent = torch.randn((ROWS, 1536), dtype=torch.bfloat16, device=device, generator=generator)
            with RecordedPreparationSession(device=device, checkpoint=str(checkpoint),
                    workload_schema="query-sharded-prefill-index-v1", numerical_policy="existing-indexer-mla-bounds",
                    autotune_effort=policy["autotune_effort"], coordinator=coordinator) as session:
                arena = ScratchArena(device)
                session.register_scratch(arena)
                fixture = SelectionFixture(weights, source, latent, cache, metadata, context, session, arena, eps=eps)
                score_windows = WindowScores(history, session=session, arena=arena, maximum=maximum)
                q = torch.randn((2, 6, 576), dtype=torch.bfloat16, device=device, generator=generator)
                records = torch.empty((2, TOPK, 576), dtype=torch.float8_e4m3fn, device=device)
                counts = torch.zeros(2, dtype=torch.int32, device=device)
                output = torch.empty((2, 6, 512), dtype=torch.bfloat16, device=device)
                caps = dense_mla.Caps(device=device, mode="decode", q_dtype=q.dtype, kv_dtype=records.dtype,
                    num_q_heads=6, max_total_q=2, max_batch=2, page_size=1, max_cache_tokens=TOPK,
                    max_page_table_width=TOPK, num_cache_pages=2 * TOPK,
                    use_cuda_graph=True, partial_dtype=torch.float32)
                attention = PreparedDenseMLA(q, records.view(-1, 1, 576), output,
                    torch.arange(2 * TOPK, dtype=torch.int32, device=device).view(2, TOPK), counts,
                    torch.arange(3, dtype=torch.int32, device=device), caps=caps, scale=192 ** -.5,
                    session=session, name="selected_order_attention", workspace=arena)
                session.tune(show_progress=False)
                calls = []
                for variant in (False, True):
                    def forward(candidate=variant):
                        return fixture.run(candidate)
                    call = compile_forward(forward) if compiled else forward
                    call()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with session.capture(), torch.cuda.graph(graph):
                        call()
                    graphs.append(graph)
                    calls.append(call)
                reference = _reference_module()
                all_keys = torch.arange(maximum // WORLD, dtype=torch.int64, device=device)
                physical = table.long()[all_keys // 128] * 128 + all_keys % 128
                # Transport preserves arbitrary bits, including nonfinite scale
                # encodings and signed zeros, independently of scoring validity.
                cache.copy_(torch.arange(cache.numel(), device=device).remainder(256).byte().view_as(cache))
                transfer = fixture.index.sharded.history
                transfer.end.fill_(maximum)
                transfer()
                torch.cuda.synchronize()
                tokens = all_keys * WORLD + rank
                raw_keys = cache[:, :8192].reshape(-1, 64, 128)[physical // 64, physical % 64]
                raw_scales = cache[:, 8192:].contiguous().view(torch.int32)[physical // 64, physical % 64]
                assert torch.equal(history[:, :8192].reshape(-1, 64, 128)[tokens // 64, tokens % 64], raw_keys)
                assert torch.equal(history[:, 8192:].contiguous().view(torch.int32)[tokens // 64, tokens % 64], raw_scales)
                generator.manual_seed(900 + rank)
                key_source = torch.randn((all_keys.numel(), 7168), device=device, dtype=torch.bfloat16,
                                         generator=generator)
                keys = (key_source @ weights["indexer.wk"].T).float()
                centered = keys - keys.mean(-1, keepdim=True)
                normalized = (centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + eps)
                    * weights["indexer.k_norm.weight"].float() + weights["indexer.k_norm.bias"].float()).bfloat16()
                history_values = torch.randn((maximum, 576), dtype=torch.bfloat16, device=device,
                    generator=torch.Generator(device=device).manual_seed(812)).to(torch.float8_e4m3fn)
                row_ids = torch.arange(ROWS, device=device)
                cases = [(0, 0), (0, 17), (0, 2047), (0, ROWS), (4096, ROWS - 3), (8192, ROWS)]
                if maximum > MAXIMUM:
                    cases.extend(((28672, ROWS), (maximum - ROWS, ROWS - 7),
                                  (maximum - ROWS, 0), (maximum - ROWS, ROWS)))
                for begin, active in cases:
                    write_fp8_index_keys(normalized, physical, cache, hadamard=False, power_of_two=False)
                    metadata.starts.fill_(begin)
                    metadata.lengths.fill_(active)
                    positions.copy_(torch.where(row_ids < active, begin + row_ids, -1))
                    stage.tensor.fill_(255)
                    history.fill_(255)
                    fixture.packed_weights.fill_(float("nan"))
                    dist.barrier()
                    graphs[0].replay()
                    graphs[1].replay()
                    torch.cuda.synchronize()
                    baseline, candidate = fixture.baseline, fixture.candidate
                    assert (baseline[active:] == -1).all() and (candidate[active:] == -1).all()
                    for ids in (baseline, candidate):
                        ordered = ids.sort(-1).values
                        expected_count = (positions + 1).clamp(min=0, max=TOPK)
                        assert torch.equal((ids >= 0).sum(-1), expected_count)
                        assert ((ids < 0) | (ids <= positions[:, None])).all()
                        assert not ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any()
                    if not active:
                        continue
                    # These are actual projection outputs, not assumptions about replicated weights.
                    replica = [None] * WORLD
                    import hashlib
                    digest = lambda tensor: hashlib.sha256(tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    dist.all_gather_object(replica, (digest(fixture.index.query), digest(fixture.index.head_weights),
                                                    digest(candidate)))
                    if len(set(replica)) != 1:
                        raise AssertionError(f"projection replicas differ; record a numerical transformation: {replica}")
                    # Verify original planar keys and scale payloads, including the chunk's cache writes.
                    owner_keys = cache[:, :8192].reshape(-1, 64, 128)[physical // 64, physical % 64]
                    owner_scales = cache[:, 8192:].contiguous().view(torch.int32)[physical // 64, physical % 64]
                    owner_tokens = all_keys * WORLD + rank
                    live = owner_tokens < begin + active
                    target_keys = history[:, :8192].reshape(-1, 64, 128)[owner_tokens[live] // 64, owner_tokens[live] % 64]
                    target_scales = history[:, 8192:].contiguous().view(torch.int32)[owner_tokens[live] // 64, owner_tokens[live] % 64]
                    assert torch.equal(target_keys, owner_keys[live]) and torch.equal(target_scales, owner_scales[live])
                    samples = sorted({min(rank, active - 1), min(active - 1, 2048 + rank)})
                    selected_metrics = []
                    global_keys = _keys(history, begin + active)
                    for row in samples:
                        indexer = fixture.index.sharded.indexers[(row // WORLD) // 128]
                        local_row = (row // WORLD) % 128
                        # Each rank checks its own query owner; the last partial row is checked by its owner.
                        if row % WORLD != rank:
                            continue
                        # Serial query slices reuse quantization scratch. Reproduce
                        # this slice before inspecting its prepared quantizer output.
                        indexer()
                        quantized = indexer.quantized[local_row].float()
                        gates = indexer.scaled_weights[local_row]
                        scores = (torch.einsum("hd,kd->hk", quantized, global_keys[:begin + row + 1]).relu()
                                  * gates[:, None]).sum(0)
                        actual_scores = score_windows(indexer, local_row)[:begin + row + 1]
                        # Bounds are those already used by Kimi's FP8 top-2048
                        # component test; membership below uses actual scorer bits.
                        torch.testing.assert_close(actual_scores, scores, rtol=3e-5, atol=3e-4)
                        cutoff = actual_scores.topk(min(TOPK, actual_scores.numel())).values[-1]
                        selected_sets = []
                        for label, ids in (("distributed", baseline[row]), ("query_owned", candidate[row])):
                            ids = ids[ids >= 0].long()
                            assert ids.numel() == min(TOPK, begin + row + 1) and ids.unique().numel() == ids.numel()
                            assert (ids <= begin + row).all()
                            assert actual_scores[ids].min() >= cutoff, (label, row, float(actual_scores[ids].min() - cutoff))
                            selected_sets.append(set(ids.tolist()))
                        difference = selected_sets[0] ^ selected_sets[1]
                        if difference:
                            values = actual_scores[torch.tensor(sorted(difference), device=device)]
                            assert (values == cutoff).all()
                        selected_metrics.append(dict(row=row, changed_members=len(difference),
                            identical_order=bool(torch.equal(baseline[row], candidate[row])),
                            reference_score_max_error=float((actual_scores - scores).abs().max()),
                            reference_score_rms=float((actual_scores - scores).square().mean().sqrt()),
                            minimum_selected_margin=float(actual_scores[
                                candidate[row][candidate[row] >= 0].long()].min() - cutoff)))
                    # Use the original selected order for both actual MLA and its probability oracle.
                    pair = (samples + samples)[:2]
                    attention_report = {}
                    for label, ids in (("distributed", baseline), ("query_owned", candidate)):
                        ideal_errors, contract_errors, repeats = [], [], []
                        records.zero_()
                        for slot, row in enumerate(pair):
                            selected = ids[row][ids[row] >= 0].long()
                            counts[slot] = selected.numel()
                            records[slot, :selected.numel()].copy_(history_values[selected])
                        for _ in range(2):
                            attention()
                            repeats.append(output.clone())
                        splits = attention.plan.prepared.state.layout.num_splits
                        for slot, row in enumerate(pair):
                            values = records[slot, :int(counts[slot])].float()
                            native_q = q[slot].float().clamp(-448, 448).to(torch.float8_e4m3fn).float()
                            expected = reference(native_q, values, splits=splits)
                            torch.testing.assert_close(output[slot].float(), expected, rtol=.03, atol=.002)
                            ideal = ((native_q.double() @ values.double().T) * 192 ** -.5).softmax(-1) @ values[:, :512].double()
                            ideal_errors.append(float((output[slot].double() - ideal).abs().max()))
                            contract_errors.append(float((output[slot].float() - expected).abs().max()))
                        attention_report[label] = dict(ideal_max=max(ideal_errors), contract_max=max(contract_errors),
                            repeatability=float((repeats[0].float() - repeats[1].float()).abs().max()))
                    case_timing = []
                    if active == ROWS:
                        for variant in (False, True, True, False):
                            dist.barrier()
                            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            start_event.record()
                            graphs[int(variant)].replay()
                            end_event.record()
                            end_event.synchronize()
                            case_timing.append(dict(query_owned=variant,
                                                    milliseconds=start_event.elapsed_time(end_event)))
                    reports.append(dict(rank=rank, begin=begin, active=active, projections_equal=True,
                        selected=selected_metrics, attention=attention_report, case_timing=case_timing))
                    print(json.dumps(reports[-1]), flush=True)
                # Repeated captured execution must neither allocate nor reuse a peer stage early.
                dist.barrier()
                if rank == WORLD - 1:
                    torch.cuda._sleep(2_000_000)
                snapshots = torch.empty((2, 4, ROWS, TOPK), dtype=torch.int32, device=device)
                reference_ids = torch.stack((fixture.baseline, fixture.candidate))
                before_query = fixture.index.query.clone()
                before_weights = fixture.index.head_weights.clone()
                before_cache = cache.clone()
                before = torch.cuda.memory_stats(device)
                for variant in range(2):
                    for iteration in range(4):
                        stage.tensor.fill_(255)
                        history.fill_(255)
                        graphs[variant].replay()
                        snapshots[variant, iteration].copy_(fixture.candidate if variant else fixture.baseline)
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats(device)
                for name in ("allocation.all.allocated", "allocation.all.freed"):
                    assert before[name] == after[name], name
                assert torch.equal(fixture.index.query, before_query)
                assert torch.equal(fixture.index.head_weights, before_weights)
                assert torch.equal(cache, before_cache)
                queue_report = {}
                changed_rows = set()
                for variant, label in enumerate(("distributed", "query_owned")):
                    different = (snapshots[variant].sort(-1).values
                        != reference_ids[variant].sort(-1).values[None]).any(-1).any(0)
                    changed_rows.update(different.nonzero().flatten().tolist())
                    queue_report[label] = dict(ordered_equal=bool(torch.equal(snapshots[variant],
                        reference_ids[variant][None].expand_as(snapshots[variant]))),
                        members_equal=bool(torch.equal(snapshots[variant].sort(-1).values,
                            reference_ids[variant].sort(-1).values[None].expand_as(snapshots[variant]))),
                        different_elements=int((snapshots[variant] != reference_ids[variant][None]).sum()))
                print(json.dumps(dict(rank=rank, queued=queue_report)), flush=True)
                replica_queue = [None] * WORLD
                dist.all_gather_object(replica_queue, digest(snapshots))
                if rank == 0:
                    torch.save(dict(selected=snapshots.cpu(), reference_ids=reference_ids.cpu(),
                        global_index_cache=history.cpu(), projected_query=fixture.index.query.cpu(),
                        projected_weights=fixture.index.head_weights.cpu(), page_table=table.cpu(),
                        positions=positions.cpu(), attention_query=q.cpu(), latent_records=history_values.cpu(),
                        replica_hashes=replica_queue, policy=policy, queued=queue_report),
                        Path(destination).with_suffix(".queued.pt"))
                assert len(set(replica_queue)) == 1, "queued replicas disagree"
                # Reconstruct native scores using each path's actual scorer.
                # Query and cache bytes were checked before these diagnostic calls.
                native_reports = []
                check_rows = sorted(changed_rows) or [0]
                for row in check_rows:
                    control = fixture.control[row // 128]
                    control.prepare_local()
                    native_local = _scorer_row(control.indexer, row % 128).cpu()
                    pieces = [None] * WORLD
                    dist.all_gather_object(pieces, native_local)
                    distributed_scores = torch.stack(pieces, dim=1).flatten()[:int(positions[row]) + 1]
                    candidate_part = fixture.index.sharded.indexers[(row // WORLD) // 128]
                    candidate_part()
                    local_scores = score_windows(candidate_part, (row // WORLD) % 128).cpu()
                    pieces = [None] * WORLD
                    dist.all_gather_object(pieces, local_scores if rank == row % WORLD else None)
                    query_scores = pieces[row % WORLD][:int(positions[row]) + 1]
                    native_report = dict(row=row, scorer_difference=float((distributed_scores - query_scores).abs().max()))
                    for variant, (label, scores) in enumerate((("distributed", distributed_scores),
                                                             ("query_owned", query_scores))):
                        cutoff = scores.topk(TOPK).values[-1]
                        margins = [float(scores[ids.long()].min() - cutoff)
                                   for ids in snapshots[variant, :, row].cpu()]
                        for ids in snapshots[variant, :, row].cpu():
                            assert ids.unique().numel() == TOPK and ids.min() >= 0 and ids.max() < scores.numel()
                        native_report[label] = dict(cutoff=float(cutoff), minimum_margins=margins,
                            tied_records=int((scores == cutoff).sum()))
                        if any(margin < 0 for margin in margins):
                            torch.save(dict(row=row, scores=scores, ids=snapshots[variant, :, row].cpu()),
                                Path(destination).with_suffix(f".rank{rank}.{label}.invalid.pt"))
                            raise AssertionError(native_report)
                    native_reports.append(native_report)
                print(json.dumps(dict(rank=rank, native_cutoffs=native_reports)), flush=True)
                # Evaluate the recorded orders themselves, including a row whose
                # membership changed in the unchanged distributed selector.
                checked = (check_rows + [ROWS - 1])[:2]
                queued_attention = []
                for variant, label in enumerate(("distributed", "query_owned")):
                    for iteration in range(4):
                        for slot, row in enumerate(checked):
                            ids = snapshots[variant, iteration, row].long()
                            counts[slot] = TOPK
                            records[slot].copy_(history_values[ids])
                        attention()
                        differences, ideals = [], []
                        for slot, row in enumerate(checked):
                            values = records[slot].float()
                            native_q = q[slot].float().clamp(-448, 448).to(torch.float8_e4m3fn).float()
                            expected = reference(native_q, values,
                                splits=attention.plan.prepared.state.layout.num_splits)
                            try:
                                torch.testing.assert_close(output[slot].float(), expected, rtol=.03, atol=.002)
                            except AssertionError:
                                torch.save(dict(ids=snapshots[variant, iteration, row].cpu(), query=q[slot].cpu(),
                                    records=records[slot].cpu(), actual=output[slot].cpu(), expected=expected.cpu()),
                                    Path(destination).with_suffix(f".rank{rank}.attention.pt"))
                                raise
                            ideal = ((native_q.double() @ values.double().T) * 192 ** -.5).softmax(-1) @ values[:, :512].double()
                            delta = output[slot].float() - expected
                            differences.append(dict(max_abs=float(delta.abs().max()),
                                                    rms=float(delta.square().mean().sqrt())))
                            ideals.append(float((output[slot].double() - ideal).abs().max()))
                        queued_attention.append(dict(path=label, iteration=iteration, rows=checked,
                            contract=differences, ideal_max=max(ideals)))
                print(json.dumps(dict(rank=rank, queued_attention=queued_attention)), flush=True)
                reports.append(dict(rank=rank, queued=queue_report, native_cutoffs=native_reports,
                                    queued_attention=queued_attention))
                timing = []
                for variant in (False, True, True, False):
                    dist.barrier()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[int(variant)].replay()
                    end.record()
                    end.synchronize()
                    timing.append(dict(query_owned=variant, milliseconds=start.elapsed_time(end)))
                reports.append(dict(rank=rank, timing=timing))
                gathered = [None] * WORLD
                dist.all_gather_object(gathered, reports)
                if rank == 0:
                    Path(destination).write_text(json.dumps(dict(policy=policy, compiled=compiled,
                        distributed_reference=dict(scoring="PreparedFP8Indexer", compaction="PreparedCandidateRows",
                            merge="PreparedRowTopK", agreement="PreparedBroadcast"),
                        checkpoint_layer=3, rows=ROWS, context_capacity=maximum, ranks=gathered), indent=2) + "\n")
    finally:
        for graph in graphs:
            graph.reset()
        if stage is not None:
            stage.close()
        for runtime in runtimes:
            runtime.close()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--context-capacity", type=int, default=MAXIMUM)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not already exist")
    if args.context_capacity < MAXIMUM or args.context_capacity % (WORLD * 128):
        parser.error("context capacity must be at least 16384 and contain whole TP16 model pages")
    policy = configure(args.checkpoint, args.config)
    import torch.multiprocessing as mp

    print(json.dumps(policy), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kimi-query-prefill-") as directory:
        mp.spawn(_worker, args=((Path(directory) / "gloo").as_uri(), str(args.checkpoint),
            str(args.output), policy, args.compiled, args.context_capacity), nprocs=WORLD, join=True)


if __name__ == "__main__":
    main()
