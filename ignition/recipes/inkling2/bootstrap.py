# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare Inkling 2 text graphs on four GB10 ranks across hosts.

The catalog serves text without speculation or images: token-sharded prefill
chunks, replicated decode batches, resumed sampling from each request's last
hidden state, and continuation-row initialization, restore and snapshot
graphs. Ranks communicate through one RoCEnante runtime.
"""

import contextlib
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from ignition.components.capacity import batch_capacities, prefill_capacities, request_capacity
from ignition.components.completion import CompletionOutputs, VECTOR_TOKENS_FIELD
from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.components.paged_generation import PreparedGeneration
from ignition.components.paged_maintenance import PreparedMaintenance
from ignition.components.paged_resume import PreparedResume
from ignition.components.parallel import TensorParallelShard
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from ignition.components.state_transfer import StateCommands
from ignition.preparation.capture import capture_forwards
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.collectives import CollectivePreparation, create_roce_runtime
from ignition.preparation.execution import GraphSignature
from ignition.preparation.fixtures.recurrent import register_benchmark_state
from ignition.preparation.host_memory import COMPILE_HEADROOM_BYTES, release_unused
from ignition.preparation.model import PreparedModel
from ignition.preparation.rebuild_manifest import declare_weights
from ignition.preparation.rank_group import minimum_across_ranks
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession, with_autotune_effort
from ignition.preparation.tuning_replicas import TuningReplica
from ignition.progress import phase

from .config import WORLD_SIZE, ModelConfig
from .contract import prepare as prepare_contract
from .decoder import PreparedDecoder
from .model import ModelWeights
from .pool import PAGE_TOKENS, ContinuationPool, RankGeometry, continuation_layout, page_specs


@dataclass(frozen=True)
class PreparationConfig:
    collective_tuning: str = "static"
    collective_ladder: dict | None = None
    compile_workers: int | None = None
    compile_headroom_bytes: int = COMPILE_HEADROOM_BYTES
    native_dense_activations: bool = False
    lock_gpu_clocks: bool = False
    autotune_effort: str = "default"
    tuning_replica: TuningReplica | None = None
    devices: tuple[int, ...] = (0, 0, 0, 0)
    hosts: tuple[str, ...] | None = None
    roce_ring: bool = False
    context: int = 131072
    prefill_chunk_size: int = 4096
    max_active: int = 16
    snapshot_budget_bytes: int = 4 << 30
    headroom_bytes: int = 4 << 30
    compile_decoder: bool = True
    retain_graphs: bool = False
    kernel_cache: Path | None = None
    # Development preparation of the first N decoder layers; None prepares all of them.
    decoder_layers: int | None = None

    def __post_init__(self):
        if (not isinstance(self.devices, tuple) or len(self.devices) != WORLD_SIZE or self.hosts is None
                or not isinstance(self.hosts, tuple) or len(set(self.hosts)) != WORLD_SIZE):
            raise ValueError("Inkling 2 preparation requires four ranks on four hosts")
        request_capacity(self.max_active)
        if not PAGE_TOKENS <= self.context <= 1048576 or self.context % PAGE_TOKENS:
            raise ValueError("Inkling 2 context must be 128-aligned in [128, 1048576]")
        # 128-aligned chunks make every prefill capacity a multiple of 128 rows, which token-sharded
        # prefill splits into four equal row shards.
        if not PAGE_TOKENS <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % PAGE_TOKENS:
            raise ValueError("Inkling 2 prefill chunks must be 128-aligned and fit the context")
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0 or self.compile_headroom_bytes < 0:
            raise ValueError("snapshot budget must be positive and headroom nonnegative")
        if self.decoder_layers is not None and (type(self.decoder_layers) is not int or self.decoder_layers < 1):
            raise ValueError("decoder layer count must be a positive integer")


def _logit_gather_bytes(rows, width):
    """Per-rank bytes of an FP32 `[rows, width]` vocabulary-shard gather."""
    return rows * width * 4


class _Softcapped:
    def __init__(self, projection, softcap):
        self.projection, self.softcap = projection, softcap

    def __call__(self, source):
        return self.softcap * torch.tanh(self.projection(source) / self.softcap)


class InklingResume(PreparedResume):
    """Resumed sampling from the last hidden state through the softcapped FP32 vocabulary shards.

    The gathered logits stop at the valid vocabulary size, so the sampler never
    reads the masked IDs.
    """

    def __init__(self, weights, *args, **kwargs):
        heads = SimpleNamespace(head=weights.unembedding, shard=SimpleNamespace(world_size=weights.world))
        super().__init__(heads, *args, logits_dtype=torch.float32, **kwargs)
        self.projection = _Softcapped(self.projection, weights.config.text.softcap)


@torch.no_grad()
@with_autotune_effort
def prepare_rank(snapshot, config, *, rank, group, progress=None, core_storage):
    from .frontend import frontend as frontend_contract, prepare as prepare_frontend, runtime_assets
    from .recipe import RECIPE

    snapshot = Path(snapshot).resolve()
    world = len(config.devices)
    shard = TensorParallelShard(rank, world)
    device = torch.device("cuda", config.devices[rank])
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
        raise ValueError("Inkling 2 preparation requires SM120 or SM121 GPUs")
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
        headers = [entry.header_sha256 for entry in manifest.shards]
        model = ModelConfig.load(snapshot)
        model.validate_shard(shard)
    # Continuation rows and pages hold the prepared layers only.
    text = model.text if config.decoder_layers is None else model.text.prefix(config.decoder_layers)
    if not text.global_layers:
        raise ValueError("a decoder-layer prefix must include a global-attention layer, whose K/V use pages")
    with phase(progress, "Load weights"):
        weights = ModelWeights.load(manifest, device, config=model, rank=rank, world=world,
                                    layers=config.decoder_layers, progress=progress)
        weights_bytes = torch.cuda.memory_allocated(device)
    del manifest
    release_unused()
    transient = contextlib.ExitStack()
    transient.enter_context(core_storage.transient())
    geometry = RankGeometry.of(text, world)
    layout = continuation_layout(text, geometry)
    page_bytes = sum(shape[0] for _, shape in page_specs(text, geometry))
    snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
    if snapshots < 1:
        raise ValueError("snapshot budget must hold a complete Inkling 2 continuation")
    slots, blocks = request_capacity(config.max_active), config.context // PAGE_TOKENS
    tokens = max(config.prefill_chunk_size, slots)
    state_rows = slots + snapshots
    pages = -(-config.prefill_chunk_size // PAGE_TOKENS) + 2 * slots
    total_memory = minimum_across_ranks(torch.cuda.mem_get_info(device)[1], group)
    vocabulary = text.valid_vocabulary
    with phase(progress, "Allocate model state"):
        pool = ContinuationPool.allocate(text, geometry, state_rows + 1, pages, device, storage=core_storage,
            maximum_pages=total_memory // page_bytes, maximum_rows=total_memory // layout.row_bytes,
            backed_slots=min(state_rows + 1, 1 + 2 * slots))
        requests = DeviceRequestState.allocate(device, vocab_size=vocabulary, capacity=slots)
        inputs = BatchInputs.allocate(tokens, blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=vocabulary, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        completion = CompletionOutputs.allocate((2, slots, VECTOR_TOKENS_FIELD + 1), device=device)
        coordinator = CollectivePreparation(group)
        if not coordinator.spans_hosts:
            raise ValueError("Inkling 2 tensor parallelism runs over RoCEnante between hosts")
        shard_width = weights.unembedding.shape[0]
        gather_bytes = max(_logit_gather_bytes(slots, shard_width), tokens * text.experts_per_token * 4)
        runtime = create_roce_runtime(group, device, max_tokens=slots, pipe_tokens=tokens, hidden_size=text.hidden,
                                      gather_bytes=gather_bytes, ring=config.roce_ring, core_storage=core_storage)
        identity = hashlib.sha256(json.dumps({"text": model.text.__dict__ | {"local_layers": sorted(model.text.local_layers)},
            "quantization": dict(model.quantization.formats), "headers": headers, "world_size": world,
            "decoder_layers": len(weights.layers), "contract": "inkling2.paged-text.v1"},
            sort_keys=True, default=str).encode()).hexdigest()
        session = RecordedPreparationSession(compile_workers=config.compile_workers,
            native_dense_activations=config.native_dense_activations, lock_gpu_clocks=config.lock_gpu_clocks,
            compile_headroom_bytes=config.compile_headroom_bytes, tuning_replica=config.tuning_replica,
            device=device, checkpoint=identity, workload_schema="inkling2", numerical_policy="default",
            coordinator=coordinator, headroom_bytes=config.headroom_bytes, benchmark_storage=core_storage,
            collective_tuning=config.collective_tuning, collective_ladder=config.collective_ladder,
            autotune_effort=config.autotune_effort)
        session.__enter__()
        register_benchmark_state(session, pool.reserved_backing, storage=pool.reserved_backing,
                                 restore=pool.clear_for_preparation)
        arena = ScratchArena(device)
        session.register_scratch(arena)
        buffers = dict(inputs.buffers(), **commands.buffers(),
            **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing, memory_capacity=pool.memory_capacity,
            request_pending_token=requests.pending_token, sampling_grammar_mask=requests.grammar_mask,
            frame_index=frame, submission_sequence=sequence, **completion.buffers())
    forwards, signatures = {}, {}
    prefills, batches = prefill_capacities(config.prefill_chunk_size), batch_capacities(config.max_active)
    # Graphs share declaration storage, largest capacity first, so collective buffers are views of one allocation.
    examples = {}
    count = len(prefills) + 5 * len(batches)
    with phase(progress, "Construct graph catalog", total=count) as stage:
        def add(name, forward, kind, rows, tokens, fields):
            forwards[name] = forward
            signatures[name] = GraphSignature(kind, rows, tokens, tuple(fields), speculation_enabled=False)
            stage.update(advance=1)

        forward_inputs = (*inputs.buffers(), "sampling_grammar_mask")
        for mode, sizes in (("prefill", reversed(prefills)), ("decode", reversed(batches))):
            for capacity in sizes:
                rows = 1 if mode == "prefill" else capacity
                name = f"{mode}{capacity}"
                m = Metadata.allocate(capacity, rows, blocks, device)
                decoder = PreparedDecoder(weights, pool, m, mode=mode, tokens=capacity, samples=rows,
                    vocabulary=vocabulary, runtime=runtime, vocabulary_runtime=runtime, coordinator=coordinator,
                    session=session, workspace=arena, name=name, examples=examples)
                batch = PreparedBatch(inputs, requests, m, mode=mode, shard=shard, pool=pool)
                forward = PreparedGeneration(decoder, batch, completion, frame, sequence, runtime=runtime,
                    coordinator=coordinator, session=session, name=name + ".generation", workspace=arena)
                add(name, forward, mode, rows, capacity if mode == "prefill" else 1, forward_inputs)
        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for capacity in reversed(batches):
            name = f"resume{capacity}"
            forward = InklingResume(weights, requests, pool, inputs, completion, frame, sequence, rows=capacity,
                runtime=runtime, vocabulary_runtime=runtime, coordinator=coordinator, session=session, name=name,
                workspace=arena)
            add(name, forward, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"))
            for mode in ("init", "restore", "snapshot"):
                name = f"{mode}{capacity}"
                forward = PreparedMaintenance(mode, requests, pool, commands, inputs, completion, frame, sequence,
                    runtime=runtime, coordinator=coordinator, session=session, name=name, capacity=capacity,
                    admission_buffers=admission, draft_state=False)
                fields = row_inputs if mode == "snapshot" else ("request_state_rows", *admission.__dataclass_fields__)
                add(name, forward, mode, capacity, 1, (*fields, "page_tables", *commands.buffers()))
    examples.clear()
    m = decoder = batch = forward = None
    with phase(progress, "Prepare and tune kernels") as stage, transient:
        tuning = session.tune(show_progress=progress is not None and rank == 0, console=stage.console)
    stream, graph_pool = torch.cuda.Stream(device=device), core_storage.graph_pool.id
    arena.release_preparation()
    graphs, compiled = capture_forwards(forwards, signatures, session=session, device=device, stream=stream,
        pool=graph_pool, collective=runtime, coordinator=coordinator, compile_decoder=config.compile_decoder,
        progress=progress, cache_root=config.kernel_cache, retain_graphs=config.retain_graphs,
        materialize=pool.clear_for_preparation, warmup_storage=core_storage,
        compile_kinds=("prefill", "decode", "resume", "init", "restore", "snapshot"))
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    release_unused()
    server_config = dict(identity=identity, model_id=RECIPE.served_model_name, max_sequence_tokens=config.context,
        prefill_chunk_size=config.prefill_chunk_size, pages=pages, state_rows=state_rows, snapshots=snapshots,
        page_tokens=PAGE_TOKENS, frontend=prepare_frontend(snapshot, vision=False), assets=runtime_assets(snapshot),
        sampling_defaults=RECIPE.sampling_defaults.as_dict(),
        model_contract=prepare_contract(signatures, buffers, pool, vocabulary=vocabulary,
            input_vocabulary=vocabulary, frontend=frontend_contract(vision=False)))
    free, total = torch.cuda.mem_get_info(device)
    memory = dict(rank=rank, world_size=world, device=config.devices[rank], decoder_layers=len(weights.layers),
        weights_bytes=weights_bytes, mapped_host_bytes=0, offloaded_bytes=0, continuation_bytes=pool.backing.nbytes,
        state_rows=state_rows, preparation_state_rows=int(pool.memory_capacity[0].item()),
        preparation_state_bytes=int(pool.memory_capacity[0].item()) * layout.row_bytes, pages=pages,
        page_bytes=page_bytes, page_tokens=PAGE_TOKENS, kv_bytes=pages * page_bytes, kv_dtype="float8_e4m3fn",
        operator_scratch_bytes=arena.nbytes, graph_buffer_bytes=0, private_buffer_bytes=0, graphs=len(graphs),
        prefill_capacities=prefills, free_device_bytes=free, total_device_bytes=total,
        allocated_tensor_bytes=torch.cuda.memory_allocated(device),
        reserved_tensor_bytes=torch.cuda.memory_reserved(device), headroom_bytes=config.headroom_bytes,
        tuning=tuning, compiled_decoder=config.compile_decoder)
    # Each decoder layer's weights form one stable payload; vocabulary shards, norms and the output query share misc.
    rebuild_weights = declare_weights(weights, components=("embedding", "embedding_norm", "final_norm", "output_query",
                                                           "unembedding"))
    return PreparedModel(rebuild_weights=rebuild_weights, device=config.devices[rank], graphs=graphs, buffers=buffers,
        signatures={name: signatures[name] for name in graphs}, max_active=config.max_active, session=session,
        server_config=server_config, memory_report=memory,
        forwards={name: compiled.get(name, forwards[name]) for name in graphs},
        export_zero_ranges=completion.zero_ranges(),
        resources=(weights, pool, requests, inputs, admission, commands, completion, forwards, arena, session,
                   compiled, stream, graph_pool, coordinator, runtime), close_resources=(runtime,))
