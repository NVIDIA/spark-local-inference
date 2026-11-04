# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare Inkling text graphs and complete continuation storage for native serving."""

import contextlib
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from ignition.components.capacity import batch_capacities, prefill_capacities, request_capacity
from ignition.components.completion import CompletionOutputs, VECTOR_TOKENS_FIELD
from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.components.paged_generation import PreparedGeneration
from ignition.components.paged_maintenance import PreparedMaintenance
from ignition.components.parallel import TensorParallelShard
from ignition.components.prompt_logprobs import PreparedScoredPrefill, PromptScoring, prefill_modes
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from ignition.components.state_transfer import StateCommands
from ignition.preparation.capture import capture_forwards
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.collectives import (
    CollectivePreparation, create_all_gather_runtime, create_all_reduce_runtime,
    create_dma_all_reduce_runtime,
)
from ignition.preparation.disposable import storage_ranges
from ignition.preparation.execution import GraphSignature
from ignition.preparation.host_memory import release_unused
from ignition.preparation.model import PreparedModel
from ignition.preparation.rank_group import minimum_across_ranks
from ignition.preparation.rebuild_manifest import declare_weights
from ignition.preparation.scratch import GraphBufferPool, ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession, with_autotune_effort
from ignition.preparation.tuning_replicas import TuningReplica
from ignition.progress import phase

from .contract import prepare as prepare_contract
from .decoder import PreparedDecoder, PreparedResume, Scratch
from .frontend import prepare_frontend, runtime_assets
from .pool import PAGE_TOKENS, ContinuationPool, continuation_layout, page_bytes
from .weights import ModelWeights


@dataclass(frozen=True)
class PreparationConfig:
    collective_tuning: str = "static"
    collective_ladder: dict | None = None
    prompt_logprobs: bool = False
    compile_workers: int | None = None
    native_dense_activations: bool = False
    lock_gpu_clocks: bool = False
    autotune_effort: str = "default"
    tuning_replica: TuningReplica | None = None
    devices: tuple[int, ...] = tuple(range(8))
    context: int = 262144
    prefill_chunk_size: int = 4096
    max_active: int = 4
    snapshot_budget_bytes: int = 4 << 30
    headroom_bytes: int = 4 << 30
    compile_decoder: bool = True
    retain_graphs: bool = False
    kernel_cache: Path | None = None
    stream_overlap: bool = True

    def __post_init__(self):
        if len(self.devices) != 8 or len(set(self.devices)) != 8 or any(d < 0 for d in self.devices):
            raise ValueError("Inkling-Small BF16 preparation requires eight distinct GPUs")
        request_capacity(self.max_active)
        if not PAGE_TOKENS <= self.context <= 262144 or self.context % PAGE_TOKENS:
            raise ValueError("Inkling context must be 128-aligned in [128,262144]")
        if not 64 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 64:
            raise ValueError("Inkling prefill chunk must be 64-aligned and fit the context")
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0:
            raise ValueError("snapshot budget must be positive and headroom nonnegative")


@torch.no_grad()
@with_autotune_effort
def prepare_rank(snapshot, config, *, rank, group, progress=None, package_storage):
    from .recipe import RECIPE

    snapshot = Path(snapshot).resolve()
    shard = TensorParallelShard(rank, len(config.devices))
    device = torch.device("cuda", config.devices[rank])
    torch.cuda.set_device(device)
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
        headers = [s.header_sha256 for s in manifest.shards]
    weights = ModelWeights.load(manifest, device, shard=shard, progress=progress, package_storage=package_storage)
    del manifest
    release_unused()
    transient = contextlib.ExitStack()
    transient.enter_context(package_storage.transient())
    layout = continuation_layout(weights.config, shard)
    snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
    if snapshots < 1:
        raise ValueError("snapshot budget must hold one complete Inkling continuation")
    slots, blocks = request_capacity(config.max_active), config.context // PAGE_TOKENS
    tokens = max(config.prefill_chunk_size, slots)
    state_rows, pages = slots + snapshots, (config.prefill_chunk_size + PAGE_TOKENS - 1) // PAGE_TOKENS + 2 * slots
    total_memory = minimum_across_ranks(torch.cuda.mem_get_info(device)[1], group)
    with phase(progress, "Allocate model state"):
        pool = ContinuationPool.allocate(weights.config, shard, state_rows + 1, pages, device,
            storage=package_storage, maximum_pages=total_memory // page_bytes(weights.config, shard),
            maximum_rows=total_memory // layout.row_bytes, backed_slots=min(state_rows + 1, 1 + 2 * slots))
        vocabulary = weights.config.vocabulary
        requests = DeviceRequestState.allocate(device, vocab_size=vocabulary, capacity=slots)
        inputs = BatchInputs.allocate(tokens, blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=vocabulary, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        completion = CompletionOutputs.allocate((2, slots, VECTOR_TOKENS_FIELD + 1), device=device)
        coordinator = CollectivePreparation(group)
        routed_width = weights.config.text["num_experts_per_tok"] * weights.config.hidden
        collective = create_all_reduce_runtime(group, device, collective_tuning=config.collective_tuning,
            max_tokens=slots, hidden_size=routed_width, ipc=package_storage.peer_storage)
        prefill_collective = create_dma_all_reduce_runtime(group, device, max_tokens=tokens,
            hidden_size=routed_width, ipc=package_storage.peer_storage)
        vocabulary_collective = create_all_gather_runtime(group, device, max_rows=slots,
            columns=weights.config.vocabulary_partition(shard)[2] * shard.world_size, ipc=package_storage.peer_storage)
        embedding_collective = create_all_gather_runtime(group, device, max_rows=min(tokens, 128),
            columns=weights.config.hidden, ipc=package_storage.peer_storage)
        identity = hashlib.sha256(json.dumps(dict(config=weights.config.text, headers=headers,
            kv_dtype="bfloat16", world_size=shard.world_size, contract="inkling.text.v1"), sort_keys=True).encode()).hexdigest()
        session = RecordedPreparationSession(compile_workers=config.compile_workers,
            collective_tuning=config.collective_tuning, collective_ladder=config.collective_ladder, autotune_effort=config.autotune_effort,
            tuning_replica=config.tuning_replica, native_dense_activations=config.native_dense_activations,
            lock_gpu_clocks=config.lock_gpu_clocks, device=device, checkpoint=identity,
            workload_schema="inkling", numerical_policy="default", coordinator=coordinator,
            headroom_bytes=config.headroom_bytes, benchmark_storage=package_storage)
        session.__enter__()
        arena = ScratchArena(device, overlap=config.stream_overlap)
        session.register_scratch(arena)
        graph_buffers, private_buffers = (GraphBufferPool(device, chunk_bytes=2 << 20) for _ in range(2))
        with package_storage.scratch_pool() as scratch_pool:
            prefill_storage = Scratch.allocate(weights.config, shard, config.prefill_chunk_size, 1, device,
                                               buffers=graph_buffers.cursor())
            decode_storage = Scratch.allocate(weights.config, shard, slots, slots, device, buffers=graph_buffers.cursor())
        buffers = dict(inputs.buffers(), **commands.buffers(),
            **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing, memory_capacity=pool.memory_capacity, request_pending_token=requests.pending_token,
            sampling_grammar_mask=requests.grammar_mask, frame_index=frame, submission_sequence=sequence,
            **completion.buffers())
    scoring = PromptScoring.allocate(config, device, buffers, frame=frame, group=group,
        ipc=package_storage.peer_storage, runtime=collective, coordinator=coordinator)
    forwards, signatures, metadata = {}, {}, {}
    prefills, batches = prefill_capacities(config.prefill_chunk_size), batch_capacities(config.max_active)
    with phase(progress, "Construct graph catalog"):
        def add(name, forward, kind, rows, tokens, fields):
            if kind == "scored_prefill":
                forward = PreparedScoredPrefill(forward, requests, inputs)
                fields = (*fields, "prompt_target_tokens")
            forwards[name] = forward
            signatures[name] = GraphSignature(kind, rows, tokens, tuple(fields))

        forward_inputs = (*inputs.buffers(), "sampling_grammar_mask")
        for kind, mode, sizes in prefill_modes(prefills, reversed(batches), scoring):
            for capacity in sizes:
                rows = 1 if mode == "prefill" else capacity
                name = f"{kind}{capacity}"
                private = private_buffers.cursor()
                m = metadata[name] = Metadata.allocate(capacity, rows, blocks, device, buffers=private)
                scratch = (prefill_storage if mode == "prefill" else decode_storage).rows(capacity, rows)
                decoder = PreparedDecoder(weights, scratch, m, pool, mode=mode,
                    runtime=prefill_collective if mode == "prefill" else collective,
                    embedding_runtime=embedding_collective, vocabulary_runtime=vocabulary_collective,
                    coordinator=coordinator, session=session, name=name, workspace=arena,
                    buffers=private, scoring=scoring if kind == "scored_prefill" else None)
                batch = PreparedBatch(inputs, requests, m, mode=mode, shard=shard, pool=pool, buffers=private)
                forward = PreparedGeneration(decoder, batch, completion, frame, sequence,
                    runtime=collective, coordinator=coordinator, session=session,
                    name=name + ".generation", workspace=arena, buffers=private)
                add(name, forward, kind, rows, capacity if mode == "prefill" else 1, forward_inputs)
        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for capacity in reversed(batches):
            name = f"resume{capacity}"
            forward = PreparedResume(weights, requests, pool, inputs, completion, frame, sequence, rows=capacity,
                runtime=collective, vocabulary_runtime=vocabulary_collective, coordinator=coordinator,
                session=session, name=name, buffers=private_buffers.cursor())
            add(name, forward, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"))
            for mode in ("init", "restore", "snapshot"):
                name = f"{mode}{capacity}"
                forward = PreparedMaintenance(mode, requests, pool, commands, inputs, completion, frame, sequence,
                    runtime=collective, coordinator=coordinator, session=session, name=name, capacity=capacity,
                    admission_buffers=admission, draft_state=False, buffers=private_buffers.cursor())
                fields = row_inputs if mode == "snapshot" else ("request_state_rows", *admission.__dataclass_fields__)
                add(name, forward, mode, capacity, 1, (*fields, "page_tables", *commands.buffers()))
    with phase(progress, "Prepare and tune kernels") as stage, transient:
        tuning = session.tune(show_progress=progress is not None and rank == 0, console=stage.console)
    stream, graph_pool = torch.cuda.Stream(device=device), package_storage.graph_pool.id
    if config.compile_decoder:
        scratch_pool.release_backing()

    def materialize():
        scratch_pool.materialize()
        pool.clear_for_preparation()

    try:
        graphs, compiled = capture_forwards(forwards, signatures, session=session, device=device, stream=stream,
            pool=graph_pool, collective=collective, coordinator=coordinator, compile_decoder=config.compile_decoder,
            progress=progress, cache_root=config.kernel_cache, retain_graphs=config.retain_graphs,
            materialize=materialize, warmup_storage=package_storage,
            compile_kinds=("prefill", "scored_prefill", "decode", "resume"))
    finally:
        scratch_pool.materialize()
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    release_unused()
    assets = runtime_assets(snapshot)
    frontend = prepare_frontend(snapshot, eos_token_ids=weights.config.text["eos_token_id"])
    server_config = dict(identity=identity, model_id=snapshot.name, max_sequence_tokens=config.context,
        prefill_chunk_size=config.prefill_chunk_size, pages=pages, state_rows=state_rows, snapshots=snapshots,
        page_tokens=PAGE_TOKENS, frontend=frontend, sampling_defaults=RECIPE.sampling_defaults.as_dict(), assets=assets,
        model_contract=prepare_contract(signatures, buffers, pool, vocabulary=vocabulary))
    free, total = torch.cuda.mem_get_info(device)
    memory = dict(rank=rank, world_size=shard.world_size, device=config.devices[rank], weights_bytes=weights.allocated_bytes,
        mapped_host_bytes=0, offloaded_bytes=0, continuation_bytes=pool.backing.nbytes, state_rows=state_rows,
        preparation_state_rows=int(pool.memory_capacity[0].item()),
        preparation_state_bytes=int(pool.memory_capacity[0].item()) * layout.row_bytes,
        pages=pages, page_bytes=page_bytes(weights.config, shard), page_tokens=PAGE_TOKENS,
        kv_bytes=pages * page_bytes(weights.config, shard), kv_dtype="bfloat16", completion_output_bytes=completion.nbytes,
        operator_scratch_bytes=arena.nbytes, graphs=len(graphs), graph_buffer_bytes=graph_buffers.nbytes,
        private_buffer_bytes=private_buffers.nbytes, prefill_capacities=prefills, free_device_bytes=free,
        total_device_bytes=total, headroom_bytes=config.headroom_bytes, tuning=tuning, compiled_decoder=config.compile_decoder)
    if scoring is not None:
        memory.update(scoring.memory_report())
    channels = collective, prefill_collective, vocabulary_collective, embedding_collective
    disposable = [prefill_storage, decode_storage, arena.tensor, metadata, graph_buffers.chunks, private_buffers.chunks]
    for forward in forwards.values():
        if isinstance(forward, PreparedScoredPrefill):
            forward = forward.forward
        if hasattr(forward, "scratch_tensors"):
            disposable.append(forward.scratch_tensors())
        if hasattr(forward, "completion") and hasattr(forward.completion, "scratch_tensors"):
            disposable.append(forward.completion.scratch_tensors)
        if hasattr(forward, "decoder"):
            disposable.append(forward.decoder.scratch_tensors())
    return PreparedModel(rebuild_weights=declare_weights(weights, components=("embedding", "embedding_norm", "norm", "head")),
        device=config.devices[rank], graphs=graphs, buffers=buffers, signatures=signatures, max_active=config.max_active,
        session=session, server_config=server_config, memory_report=memory,
        export_zero_ranges=(*(() if scoring is None else scoring.zero_ranges()), *storage_ranges(disposable), *completion.zero_ranges()),
        resources=(scoring, weights, pool, requests, inputs, admission, commands, arena, forwards, prefill_storage,
            decode_storage, metadata, compiled, stream, graph_pool, session, coordinator, *channels,
            scratch_pool, graph_buffers, private_buffers),
        close_resources=tuple(c for c in (*channels, scoring) if c is not None))
