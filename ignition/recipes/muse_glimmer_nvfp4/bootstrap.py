# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare single-GPU Muse Glimmer text, vision and speculative graphs with complete cache continuations.

The catalog is Muse Spark's (prefill, scored prefill, decode, resume, init, restore and
snapshot at every capacity; ``-no-mtp`` variants and ``speculative{batch}x{width}-tokens{packed}``
rounds when drafting) prepared at TP1 as ``gemma4/bootstrap.py`` prepares one rank: no
``CollectivePreparation`` and every runtime ``None``. ``vision{capacity}`` graphs per patch
capacity join it unless ``language_model_only``.
"""

import contextlib
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from ignition.components.capacity import batch_capacities, image_capacity, prefill_capacities, request_capacity
from ignition.components.completion import CompletionOutputs, VECTOR_TOKENS_FIELD
from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.components.paged_generation import PreparedGeneration
from ignition.components.paged_maintenance import PreparedMaintenance
from ignition.components.parallel import TensorParallelShard
from ignition.components.prompt_logprobs import PreparedScoredPrefill, PromptScoring, prefill_modes
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from ignition.components.state_transfer import StateCommands
from ignition.components.vision_state import PrefillVisionInputs, VisionState
from ignition.preparation.capture import capture_forwards
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.disposable import storage_ranges
from ignition.preparation.execution import GraphSignature
from ignition.preparation.host_memory import release_unused
from ignition.preparation.model import PreparedModel
from ignition.preparation.rebuild_manifest import declare_weights
from ignition.preparation.scratch import GraphBufferPool, ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession, with_autotune_effort
from ignition.preparation.tuning_replicas import TuningReplica
from ignition.progress import phase
from ignition.recipes.muse_spark.pool import PAGE_TOKENS, ContinuationPool, continuation_layout, page_bytes

from .contract import prepare as prepare_contract
from .decoder import PreparedDecoder, PreparedResume, Scratch
from .frontend import prepare_frontend, runtime_assets
from .weights import ModelWeights, vision_model

CONTEXT = 131072
MAX_WIDTH = 16  # the DFlash2 drafter's block: 15 drafts and the pending token


@dataclass(frozen=True)
class PreparationConfig:
    # One rank runs no collectives; the policy fields ride along so every recipe reads the plan alike.
    collective_tuning: str = "static"
    collective_ladder: dict | None = None
    prompt_logprobs: bool = False
    compile_workers: int | None = None
    native_dense_activations: bool = False
    lock_gpu_clocks: bool = False
    autotune_effort: str = "default"
    tuning_replica: TuningReplica | None = None
    device: int = 0
    context: int = CONTEXT
    prefill_chunk_size: int = 4096
    max_active: int = 4
    snapshot_budget_bytes: int = 4 << 30
    headroom_bytes: int = 4 << 30
    compile_decoder: bool = True
    retain_graphs: bool = False
    kernel_cache: Path | None = None
    stream_overlap: bool = True
    language_model_only: bool = True
    image_concurrency: int | None = None
    dflash: bool = False
    draft_snapshot: Path | None = None
    speculative_widths: tuple[int, ...] = tuple(range(1, MAX_WIDTH + 1))
    speculative_fixed_width: int | None = None
    speculative_token_capacity: int | None = None
    speculative_graph_budget: int = 64

    def __post_init__(self):
        if type(self.device) is not int or self.device < 0:
            raise ValueError("Muse Glimmer preparation names one CUDA device index")
        if not self.speculative_widths or max(self.speculative_widths) > MAX_WIDTH:
            raise ValueError("Muse Glimmer DFlash2 drafts at most fifteen tokens")
        if self.speculative_fixed_width is not None and self.speculative_fixed_width not in self.speculative_widths:
            raise ValueError("fixed verification width must belong to the captured widths")
        request_capacity(self.max_active)
        if not PAGE_TOKENS <= self.context <= CONTEXT or self.context % PAGE_TOKENS:
            raise ValueError(f"Muse Glimmer context must be 128-aligned in [128,{CONTEXT}]")
        if not 64 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 64:
            raise ValueError("Muse Glimmer prefill chunk must be 64-aligned and fit the context")
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0:
            raise ValueError("snapshot budget must be positive and headroom nonnegative")
        if self.dflash and self.draft_snapshot is None:
            raise ValueError("Muse Glimmer DFlash requires a resolved draft checkpoint")
        if type(self.language_model_only) is not bool:
            raise TypeError("language_model_only must be a boolean")
        object.__setattr__(self, "image_concurrency", None if self.language_model_only
                           else image_capacity(self.max_active, self.image_concurrency))
        _ = self.speculative_round_shapes

    @property
    def speculative_round_shapes(self):
        from ignition.components.capacity import parallel_speculative_shapes

        return parallel_speculative_shapes(self.max_active, self.speculative_widths,
            self.speculative_token_capacity, self.speculative_graph_budget,
            fixed_width=self.speculative_fixed_width) if self.dflash else ()

    @property
    def decode_tokens(self):
        return max((packed for _, _, packed in self.speculative_round_shapes),
                   default=request_capacity(self.max_active))

    @property
    def decode_samples(self):
        return max((batch * width for batch, width, _ in self.speculative_round_shapes),
                   default=request_capacity(self.max_active))

    @property
    def completion_token_capacity(self):
        return max((width for _, width, _ in self.speculative_round_shapes), default=1)


@torch.no_grad()
@with_autotune_effort
def prepare(snapshot, config, *, progress=None, package_storage):
    from .recipe import RECIPE

    snapshot = Path(snapshot).resolve()
    shard = TensorParallelShard(0, 1)
    device = torch.device("cuda", config.device)
    torch.cuda.set_device(device)
    vision = not config.language_model_only
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
        headers = [s.header_sha256 for s in manifest.shards]
        draft_headers = ([s.header_sha256 for s in inspect_indexed_checkpoint(config.draft_snapshot).shards]
                         if config.dflash else [])
    weights = ModelWeights.load(manifest, device, shard=shard, progress=progress, package_storage=package_storage,
                                draft_snapshot=config.draft_snapshot if config.dflash else None, vision=vision)
    del manifest
    release_unused()
    transient = contextlib.ExitStack()
    transient.enter_context(package_storage.transient())
    layout = continuation_layout(weights.config, shard, dflash=config.dflash)
    snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
    if snapshots < 1:
        raise ValueError("snapshot budget must hold one complete Muse Glimmer continuation")
    slots, blocks = request_capacity(config.max_active), config.context // PAGE_TOKENS
    draft_rows = slots * weights.config.draft.block_size if config.dflash else 0
    tokens = max(config.prefill_chunk_size, config.decode_tokens, draft_rows)
    state_rows, pages = slots + snapshots, (config.prefill_chunk_size + PAGE_TOKENS - 1) // PAGE_TOKENS + 2 * slots
    total_memory = torch.cuda.mem_get_info(device)[1]
    with phase(progress, "Allocate model state"):
        pool = ContinuationPool.allocate(weights.config, shard, state_rows + 1, pages, device,
            storage=package_storage, maximum_pages=total_memory // page_bytes(weights.config, shard),
            maximum_rows=total_memory // layout.row_bytes, backed_slots=min(state_rows + 1, 1 + 2 * slots),
            dflash=config.dflash)
        vocabulary = weights.config.vocabulary
        requests = DeviceRequestState.allocate(device, vocab_size=vocabulary, capacity=slots)
        inputs = BatchInputs.allocate(tokens, blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=vocabulary, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        completion = CompletionOutputs.allocate((2, slots, VECTOR_TOKENS_FIELD + config.completion_token_capacity), device=device)
        identity = hashlib.sha256(json.dumps(dict(config=weights.config.text, headers=headers,
            draft=None if weights.config.draft is None else weights.config.draft.text,
            draft_headers=draft_headers, kv_dtype="float8_e4m3fn",
            contract="muse-glimmer-nvfp4.text.v1"), sort_keys=True).encode()).hexdigest()
        session = RecordedPreparationSession(compile_workers=config.compile_workers,
            collective_tuning=config.collective_tuning, collective_ladder=config.collective_ladder,
            autotune_effort=config.autotune_effort, tuning_replica=config.tuning_replica,
            native_dense_activations=config.native_dense_activations, lock_gpu_clocks=config.lock_gpu_clocks,
            device=device, checkpoint=identity, workload_schema="muse_glimmer_nvfp4", numerical_policy="default",
            coordinator=None, headroom_bytes=config.headroom_bytes, benchmark_storage=package_storage)
        session.__enter__()
        arena = ScratchArena(device, overlap=config.stream_overlap)
        session.register_scratch(arena)
        graph_buffers, private_buffers = (GraphBufferPool(device, chunk_bytes=2 << 20) for _ in range(2))
        target_buffers = GraphBufferPool(device, chunk_bytes=2 << 20) if config.dflash else None
        context_buffers = GraphBufferPool(device, chunk_bytes=2 << 20) if config.dflash else None
        scratch_pool = package_storage.scratch_pool()
        with scratch_pool:
            prefill_storage = Scratch.allocate(weights.config, shard, config.prefill_chunk_size, 1, device,
                                               buffers=graph_buffers.cursor())
            decode_storage = Scratch.allocate(weights.config, shard, config.decode_tokens, config.decode_samples, device,
                                              buffers=graph_buffers.cursor())
        buffers = dict(inputs.buffers(), **commands.buffers(),
            **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing, memory_capacity=pool.memory_capacity, request_pending_token=requests.pending_token,
            sampling_grammar_mask=requests.grammar_mask, frame_index=frame, submission_sequence=sequence,
            **completion.buffers())
        vision_inputs = vision_prefill = vision_state = None
        if vision:
            from .vision_state import FEATURE_TOKEN, FEATURES, HIDDEN, VisionInputs

            with scratch_pool:
                vision_inputs = VisionInputs.allocate(device)
            vision_prefill = PrefillVisionInputs.allocate(config.prefill_chunk_size, device)
            vision_state = VisionState(inputs, vision_prefill, requests, image_token=FEATURE_TOKEN,
                                       features=FEATURES, hidden=HIDDEN, image_slots=config.image_concurrency)
            buffers.update(**vision_inputs.buffers(), **vision_prefill.buffers())
        speculation = None
        if config.dflash:
            from .dflash import SpeculativePreparation

            speculation = SpeculativePreparation(weights, pool, requests, inputs, decode_storage,
                completion=completion, frame=frame, sequence=sequence, session=session, workspace=arena,
                buffers=graph_buffers.cursor(), decoder_buffers=target_buffers.cursor(),
                context_buffers=context_buffers.cursor())
    scoring = PromptScoring.allocate(config, device, buffers, frame=frame)
    forwards, signatures, metadata = {}, {}, {}
    prefills, batches = prefill_capacities(config.prefill_chunk_size), batch_capacities(config.max_active)
    with phase(progress, "Construct graph catalog"):
        def add(name, forward, kind, rows, tokens, fields, *, enabled=None, packed_tokens=0):
            if kind == "scored_prefill":
                forward = PreparedScoredPrefill(forward, requests, inputs)
                fields = (*fields, "prompt_target_tokens")
            forwards[name] = forward
            signatures[name] = GraphSignature(kind, rows, tokens, tuple(fields), speculation_enabled=enabled,
                                              packed_token_capacity=packed_tokens)

        if vision:
            from .vision import PATCH_CAPACITIES, PreparedVision, VisionMetadata, VisionScratch
            from .vision_state import PreparedVisionGeneration

            for capacity in reversed(PATCH_CAPACITIES):
                name = f"vision{capacity}"
                with scratch_pool:
                    cursor = graph_buffers.cursor()
                    scratch = VisionScratch.allocate(capacity, device, buffers=cursor)
                    vision_metadata = VisionMetadata.allocate(capacity, 1, device, buffers=cursor)
                encoder = PreparedVision(vision_model(snapshot), scratch, vision_metadata, weights.vision,
                                         session=session, name=name, attention_workspace=arena)
                forward = PreparedVisionGeneration(encoder, vision_inputs, vision_state, completion, frame, sequence,
                                                   buffers=cursor)
                add(name, forward, "vision", 1, capacity, (*vision_inputs.buffers(), "vision_feature_slot"))
        forward_inputs = (*inputs.buffers(), "sampling_grammar_mask")
        prefill_inputs = (*forward_inputs, *(vision_prefill.buffers() if vision else ()))
        modes = (True, False) if config.dflash else (False,)
        for enabled in modes:
            suffix = "-no-mtp" if config.dflash and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            for kind, mode, sizes in prefill_modes(prefills, () if enabled else reversed(batches), scoring):
                for capacity in sizes:
                    rows = 1 if mode == "prefill" else capacity
                    name = f"{kind}{capacity}" + (suffix if mode == "prefill" else "")
                    private = private_buffers.cursor()
                    m = metadata[name] = Metadata.allocate(capacity, rows, blocks, device, buffers=private)
                    scratch = (prefill_storage if mode == "prefill" else decode_storage).rows(capacity, rows)
                    decoder = PreparedDecoder(mode_weights, scratch, m, pool, mode=mode, runtime=None,
                        embedding_runtime=None, vocabulary_runtime=None, coordinator=None, session=session,
                        name=name, workspace=arena, buffers=private,
                        scoring=scoring if kind == "scored_prefill" else None,
                        vision_state=vision_state if vision and mode == "prefill" else None)
                    batch = PreparedBatch(inputs, requests, m, mode=mode, shard=shard, pool=pool, buffers=private,
                        validate_prefill=vision_state.validate_prefill if vision and mode == "prefill" else None)
                    forward = PreparedGeneration(decoder, batch, completion, frame, sequence,
                        runtime=None, coordinator=None, session=session,
                        name=name + ".generation", workspace=arena, buffers=private,
                        context_factory=None if speculation is None else speculation.forwards.context)
                    add(name, forward, kind, rows, capacity if mode == "prefill" else 1,
                        prefill_inputs if mode == "prefill" else forward_inputs, enabled=enabled)
        if speculation is not None:
            from ignition.components.parallel_draft import PreparedParallelDraftRound

            for batch, width, packed in sorted(config.speculative_round_shapes,
                    key=lambda shape: (shape[2], shape[0]), reverse=True):
                name = f"speculative{batch}x{width}-tokens{packed}"
                forward = PreparedParallelDraftRound(speculation, batch, width, name,
                    buffers=private_buffers.cursor(), target_token_capacity=packed)
                forward.prepare()
                add(name, forward, "speculative", batch, width, forward_inputs, enabled=True,
                    packed_tokens=packed)
        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for enabled in modes:
            suffix = "-no-mtp" if config.dflash and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            for capacity in reversed(batches):
                name = f"resume{capacity}" + suffix
                forward = PreparedResume(mode_weights, requests, pool, inputs, completion, frame, sequence, rows=capacity,
                    runtime=None, vocabulary_runtime=None, coordinator=None,
                    session=session, name=name, buffers=private_buffers.cursor())
                add(name, forward, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"), enabled=enabled)
                for mode in ("init", "restore", "snapshot"):
                    name = f"{mode}{capacity}" + suffix
                    forward = PreparedMaintenance(mode, requests, pool, commands, inputs, completion, frame, sequence,
                        runtime=None, coordinator=None, session=session, name=name, capacity=capacity,
                        admission_buffers=admission, draft_state=enabled, buffers=private_buffers.cursor())
                    fields = row_inputs if mode == "snapshot" else ("request_state_rows", *admission.__dataclass_fields__)
                    add(name, forward, mode, capacity, 1, (*fields, "page_tables", *commands.buffers()), enabled=enabled)
    with phase(progress, "Prepare and tune kernels") as stage, transient:
        tuning = session.tune(show_progress=progress is not None, console=stage.console)
    stream, graph_pool = torch.cuda.Stream(device=device), package_storage.graph_pool.id
    if config.compile_decoder:
        scratch_pool.release_backing()

    def materialize():
        scratch_pool.materialize()
        if vision:
            vision_inputs.vision_slot.fill_(-1)
        pool.clear_for_preparation()

    try:
        graphs, compiled = capture_forwards(forwards, signatures, session=session, device=device, stream=stream,
            pool=graph_pool, collective=None, coordinator=None, compile_decoder=config.compile_decoder,
            progress=progress, cache_root=config.kernel_cache, retain_graphs=config.retain_graphs,
            materialize=materialize, warmup_storage=package_storage,
            compile_kinds=("prefill", "scored_prefill", "decode", "resume", "speculative", *(("vision",) if vision else ())))
    finally:
        scratch_pool.materialize()
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    release_unused()
    assets = runtime_assets(snapshot)
    frontend = prepare_frontend(snapshot, eos_token_ids=weights.config.text["eos_token_id"])
    if not vision:
        frontend.pop("image", None)
    server_config = dict(identity=identity, model_id=snapshot.name, max_sequence_tokens=config.context,
        prefill_chunk_size=config.prefill_chunk_size, pages=pages, state_rows=state_rows, snapshots=snapshots,
        page_tokens=PAGE_TOKENS, frontend=frontend, sampling_defaults=RECIPE.sampling_defaults.as_dict(), assets=assets,
        model_contract=prepare_contract(signatures, buffers, pool, vocabulary=vocabulary,
                                       draft_lag=0 if config.dflash else None,
                                       image_slots=config.image_concurrency if vision else None))
    free, total = torch.cuda.mem_get_info(device)
    memory = dict(rank=0, world_size=1, device=config.device, weights_bytes=weights.allocated_bytes,
        mapped_host_bytes=0, offloaded_bytes=0, continuation_bytes=pool.backing.nbytes, state_rows=state_rows,
        preparation_state_rows=int(pool.memory_capacity[0].item()),
        preparation_state_bytes=int(pool.memory_capacity[0].item()) * layout.row_bytes,
        pages=pages, page_bytes=page_bytes(weights.config, shard), page_tokens=PAGE_TOKENS,
        kv_bytes=pages * page_bytes(weights.config, shard), kv_dtype="float8_e4m3fn", completion_output_bytes=completion.nbytes,
        operator_scratch_bytes=arena.nbytes, graphs=len(graphs), graph_buffer_bytes=graph_buffers.nbytes,
        private_buffer_bytes=private_buffers.nbytes, prefill_capacities=prefills, free_device_bytes=free,
        total_device_bytes=total, headroom_bytes=config.headroom_bytes, tuning=tuning, compiled_decoder=config.compile_decoder,
        language_model_only=config.language_model_only, image_concurrency=config.image_concurrency)
    if scoring is not None:
        memory.update(scoring.memory_report())
    if config.dflash:
        from ignition.components.capacity import speculative_token_capacity

        server_config.update(
            speculative_widths=tuple(sorted({width for _, width, _ in config.speculative_round_shapes})),
            speculative_token_capacity=speculative_token_capacity(config.max_active,
                max(config.speculative_widths), config.speculative_token_capacity),
            speculative_fixed_width=config.speculative_fixed_width)
        memory.update(speculative_round_shapes=config.speculative_round_shapes,
                      speculative_graph_budget=config.speculative_graph_budget)
    disposable = [prefill_storage, decode_storage, arena.tensor, metadata, graph_buffers.chunks, private_buffers.chunks]
    if vision:
        disposable.append(vision_state.arena)
    if speculation is not None:
        disposable.extend((speculation.draft_scratch.scratch_tensors(), target_buffers.chunks, context_buffers.chunks))
        memory.update(speculative_buffer_bytes=target_buffers.nbytes, context_buffer_bytes=context_buffers.nbytes)
    for forward in forwards.values():
        if isinstance(forward, PreparedScoredPrefill):
            forward = forward.forward
        if hasattr(forward, "scratch_tensors"):
            disposable.append(forward.scratch_tensors())
        if hasattr(forward, "completion") and hasattr(forward.completion, "scratch_tensors"):
            disposable.append(forward.completion.scratch_tensors)
        if hasattr(forward, "decoder"):
            disposable.append(forward.decoder.scratch_tensors())
    return PreparedModel(rebuild_weights=declare_weights(weights, components=('embedding', 'norm', 'head', 'draft', 'vision')),
        device=config.device, graphs=graphs, buffers=buffers, signatures=signatures, max_active=config.max_active,
        session=session, server_config=server_config, memory_report=memory,
        export_zero_ranges=(*(() if scoring is None else scoring.zero_ranges()), *storage_ranges(disposable), *completion.zero_ranges()),
        resources=(scoring, weights, pool, requests, inputs, admission, commands, arena, forwards, prefill_storage,
            decode_storage, metadata, compiled, stream, graph_pool, session, scratch_pool, graph_buffers,
            private_buffers, target_buffers, context_buffers, speculation, vision_state, vision_inputs, vision_prefill),
        close_resources=() if scoring is None else (scoring,))
