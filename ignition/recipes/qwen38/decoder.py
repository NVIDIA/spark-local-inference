# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Exact-checkpoint decoder weight ownership and canonical module parameters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch

from ignition.components.parallel import TensorParallelShard
from ignition.planning import SUPPORTED_ARCHITECTURES, SUPPORTED_CAPABILITIES
from ignition.preparation.checkpoint import CheckpointManifest
from ignition.preparation.host_memory import device_memory_info
from ignition.preparation.persistent import allocated_tensor_bytes
from ignition.preparation.qwen38 import TEXT_PREFIX
from ignition.preparation.scratch import ScratchArena

from .hyperconnection import packed_down_projection, hyperconnection_extents
from .model import create_model_tree
from .moe import QwenMoeWeights
from .vision import VisionWeights
from .weights import QwenWeightLoader


@dataclass(frozen=True)
class DecoderLayerWeights:
    attention: dict[str, object]
    moe: QwenMoeWeights


@dataclass(frozen=True)
class ModelWeights:
    """Own the canonical tree and every packed GPU weight used by its recipes.

    Canonical quantized Linear and expert parameters remain on meta storage;
    their prepared replacements own the checkpoint's packed representations.
    PLE table ownership is separate because those tables reside in mapped host
    memory. Expert layers expand their CSF block scales once at load.
    """

    tree: torch.nn.Module
    layers: tuple[DecoderLayerWeights, ...]
    vision: VisionWeights
    allocated_bytes: int
    shard: TensorParallelShard = TensorParallelShard(0, 1)

    @classmethod
    def load(cls, manifest: CheckpointManifest, device: torch.device,
             *, progress: Callable[[str], None] | None = None,
             shard: TensorParallelShard = TensorParallelShard(0, 1),
             shard_hyperconnections: bool = False, force_moe_decode_a16: bool = False) -> ModelWeights:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("model weights require a CUDA device")
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
            raise ValueError(f"the Qwen recipe requires {SUPPORTED_ARCHITECTURES}")
        if force_moe_decode_a16 and torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
            raise ValueError("force_moe_decode_a16 requires SM120/121 packed NVFP4 experts")
        free_bytes, _ = device_memory_info(device)
        stored = sum(tensor.nbytes for tensor in manifest.tensors.values()
                     if tensor.component not in ("mtp", "ple_tables"))
        # Additional layouts, state, graph memory, and cache pages are planned separately.
        if free_bytes < stored // shard.world_size + (8 << 30):
            raise MemoryError(f"model weight loading requires at least {stored // shard.world_size + (8 << 30)} free CUDA bytes; found {free_bytes}")
        tree = create_model_tree(manifest)
        before = allocated_tensor_bytes(device)
        layers = []
        with QwenWeightLoader(manifest, device, progress=progress, shard=shard,
                              force_moe_decode_a16=force_moe_decode_a16) as loader:
            def parameters(paths, extents=None, *, temporary=False):
                for path, value in zip(paths, loader.load_tensors(paths, extents=extents, temporary=temporary), strict=True):
                    parent, _, attribute = path.rpartition(".")
                    setattr(tree.get_submodule(parent), attribute, torch.nn.Parameter(value, requires_grad=False))

            def hyperconnection(prefix: str):
                names = ("hc_norm", "input_mix_weight_down", "input_mix_weight_up")
                if tree.get_submodule(prefix).block_inject_weight is not None:
                    names += ("block_inject_weight",)
                extents = hyperconnection_extents(shard, shard_hyperconnections)
                packed = tree.get_submodule(prefix).block_inject_weight is not None
                for temporary in (False, True):
                    selected = tuple(name for name in names
                        if (packed and name in ("input_mix_weight_down", "block_inject_weight")) == temporary)
                    if selected:
                        parameters(tuple(f"{prefix}.{name}.weight" for name in selected),
                                   tuple(extents.get(f"{name}.weight") for name in selected), temporary=temporary)
                packed_down_projection(tree.get_submodule(prefix))

            parameters((f"{TEXT_PREFIX}.embed_tokens.weight", "lm_head.weight"),
                       (None, slice(*shard.partition(248320))))
            hyperconnection(f"{TEXT_PREFIX}.hyper_connection_mixer")
            for layer in range(48):
                prefix = f"{TEXT_PREFIX}.layers.{layer}"
                hyperconnection(f"{prefix}.attn_hyper_connection")
                hyperconnection(f"{prefix}.mlp_hyper_connection")
                if layer % 4 == 3:
                    attention_prefix = f"{prefix}.self_attn"
                    load_projections = loader.load_qsa_projections
                    names = ("q_norm.weight", "k_norm.weight", "indexer.q_layernorm.weight",
                             "indexer.k_layernorm.weight")
                else:
                    attention_prefix = f"{prefix}.linear_attn"
                    load_projections = loader.load_gdn_projections
                    names = ("A_log", "dt_bias", "conv1d.weight", "norm.weight")
                if layer % 4 == 3:
                    parameters(tuple(f"{attention_prefix}.{name}" for name in names))
                else:
                    heads = slice(*shard.partition(48))
                    parameters(tuple(f"{attention_prefix}.{name}" for name in ("A_log", "dt_bias", "norm.weight")),
                               (heads, heads, None))
                    channels = shard.fused_partitions((2048, 2048, 6144))
                    from ignition.preparation.persistent import persistent_cat

                    convolution = persistent_cat(loader.load_tensors((f"{attention_prefix}.conv1d.weight",) * 3,
                        extents=tuple(slice(*part) for part in channels), temporary=True))
                    module = tree.get_submodule(attention_prefix)
                    module.conv1d.weight = torch.nn.Parameter(convolution, requires_grad=False)
                    module.num_k_heads //= shard.world_size
                    module.num_v_heads //= shard.world_size
                    module.conv_dim //= shard.world_size
                attention = load_projections(attention_prefix)
                moe = QwenMoeWeights.load(loader, layer)
                module = tree.get_submodule(f"{prefix}.mlp")
                module.gate.weight = torch.nn.Parameter(moe.router, requires_grad=False)
                module.shared_expert_gate.weight = torch.nn.Parameter(moe.shared.gate, requires_grad=False)
                layers.append(DecoderLayerWeights(attention, moe))

            prefix = f"{TEXT_PREFIX}.layers.1.ple"
            parameters(tuple(f"{prefix}.{name}" for name in
                ("key_proj.weight", "value_proj.weight", "conv1d.weight", "norm_key.weight",
                 "norm_query.weight", "norm_conv.weight")))
            vision = VisionWeights.load(tree.model.visual, loader)
        torch.cuda.synchronize(device)
        return cls(tree, tuple(layers), vision,
                   allocated_tensor_bytes(device) - before, shard)


@dataclass(frozen=True)
class DecoderScratch:
    """Shared intermediates for serialized complete-forward graph variants."""

    embedding: torch.Tensor
    residual: torch.Tensor
    hidden: torch.Tensor
    sample_hidden: torch.Tensor
    logits: torch.Tensor
    hyperconnection: object
    gdn: object
    qsa: object
    ple: object
    moe: object
    local_logits: torch.Tensor | None = None

    @classmethod
    def allocate(cls, tokens: int, device, *, sample_rows: int = 4, world_size: int = 1, buffers=None,
                 hyperconnection_lowrank: int = 320):
        if type(tokens) is not int or tokens <= 0:
            raise ValueError("decoder token capacity must be a positive integer")
        if type(sample_rows) is not int or sample_rows <= 0:
            raise ValueError("decoder sampling capacity must be positive")
        from .attention import QSAScratch
        from ignition.components.gated_delta import GDNScratch
        from .hyperconnection import HyperconnectionScratch
        from .moe import QwenMoeScratch
        from .ple import PLEScratch

        def zeros(*shape):
            return (torch.zeros(shape, dtype=torch.bfloat16, device=device) if buffers is None
                    else buffers.zeros(shape, torch.bfloat16))

        if buffers is None:
            from ignition.preparation.scratch import GraphBufferPool

            buffers = GraphBufferPool(device, chunk_bytes=2 << 20).cursor()
        embedding, residual, hidden = zeros(tokens, 2560), zeros(tokens, 10240), zeros(tokens, 2560)
        sample_hidden, logits = zeros(sample_rows, 2560), zeros(sample_rows, 248320)
        local_logits = zeros(sample_rows, 248320 // world_size) if world_size > 1 else None
        block_input, block_output = zeros(tokens, 2560), zeros(tokens, 2560)
        projected = zeros(tokens, (hyperconnection_lowrank + 4 + 15) // 16 * 16)
        stages = [buffers.fork() for _ in range(5)]
        # Reserve PLE's wide buffers before other stages divide the shared chunks.
        ple = PLEScratch.allocate(tokens, device, buffers=stages[2])
        gdn = GDNScratch.allocate(tokens, device, world_size=world_size, buffers=stages[0], output=block_output)
        qsa = QSAScratch.allocate(tokens, device, world_size=world_size, buffers=stages[1], output=block_output)
        moe = QwenMoeScratch.allocate(tokens, device, intermediate=640 // world_size,
                                    buffers=stages[3], output=block_output)
        hc = HyperconnectionScratch.allocate(tokens, device, buffers=stages[4],
                                            projected=projected, block_input=block_input, lowrank=hyperconnection_lowrank)
        buffers.join(*stages)
        return cls(embedding, residual, hidden, sample_hidden, logits, hc, gdn, qsa, ple, moe, local_logits)

    def rows(self, tokens: int, sequences: int):
        if (type(tokens) is not int or type(sequences) is not int
                or not 0 < tokens <= self.embedding.shape[0]
                or not 1 <= sequences <= self.sample_hidden.shape[0]):
            raise ValueError("decoder view exceeds persistent scratch capacity")

        def rows(value):
            return type(value)(*(getattr(value, field)[:tokens] for field in value.__dataclass_fields__))

        return type(self)(self.embedding[:tokens], self.residual[:tokens], self.hidden[:tokens],
            self.sample_hidden[:sequences], self.logits[:sequences], self.hyperconnection.rows(tokens),
            self.gdn.rows(tokens), self.qsa.rows(tokens), rows(self.ple), self.moe.rows(tokens),
            None if self.local_logits is None else self.local_logits[:sequences])


class PreparedDecoder:
    """Capture the canonical 48-layer residual composition with prepared operators.

    Weights, hybrid state, paged attention caches and all escaping tensors have
    explicit owners. Scratch and workspaces are shared across serialized layer
    calls and graph variants. Finalize scratch bindings before capture.
    Python executes this object
    only during preparation and capture.
    """

    def __init__(self, weights: ModelWeights, scratch: DecoderScratch, state,
                 qsa_states, rotary, ple_table, workspaces: ScratchArena,
                 completion, frame_index, submission_sequence,
                 *, session, name: str, progress=None,
                 layer_outputs: torch.Tensor | None = None, embedding_injection=None,
                 stage_outputs: dict[str, torch.Tensor] | None = None,
                 max_cache_pages: int | None = None,
                 residual_output: torch.Tensor | None = None, verification: bool = False,
                 gdn_records=None, gdn_in_place: bool = False, runtime=None, vocabulary_runtime=None, coordinator=None,
                 sampling_workspace=None, scoring=None, gdn_buffers=None):
        from ignition.components.embedding import PreparedEmbedding
        from ignition.components.linear import UnquantizedProjection

        from .attention import PreparedAttention
        from ignition.components.gated_delta import PreparedGDN
        from .hyperconnection import PreparedHyperconnection
        from .moe import PreparedQwenMoe
        from .ple import PreparedPLE

        if verification and scratch.sample_hidden.shape == scratch.hidden.shape:
            scratch = replace(scratch, sample_hidden=scratch.hidden)
        self.weights, self.scratch, self.state = weights, scratch, state
        self.verification = verification
        if residual_output is not None and (
                residual_output.shape != scratch.residual.shape
                or residual_output.dtype != torch.bfloat16
                or residual_output.device != scratch.residual.device):
            raise ValueError("target feedback requires persistent BF16 [tokens,10240] storage")
        self.residual_output = residual_output
        if layer_outputs is not None and (layer_outputs.shape != (48, state.tokens, 10240)
                or layer_outputs.dtype != torch.bfloat16 or layer_outputs.device != scratch.hidden.device):
            raise ValueError("layer trace requires BF16 [48,token_capacity,10240] on the model device")
        self.layer_outputs = layer_outputs
        self.context_capture, self.context_layers = None, ()
        self.embedding_injection = embedding_injection
        self.stage_outputs = {} if stage_outputs is None else stage_outputs
        self.workspaces, self.ple_table, self.qsa_states = workspaces, ple_table, qsa_states
        source = scratch.hyperconnection.block_input
        language = weights.tree.model.language_model
        self.embedding = PreparedEmbedding(language.embed_tokens.weight, state.ple.token_ids,
            scratch.embedding, num_rows=state.gdn.num_tokens, session=session, name=f"{name}.embedding")
        self.layers, self.reductions = [], []
        for index, (module, layer_weights) in enumerate(zip(language.layers, weights.layers, strict=True)):
            prefix = f"{name}.layer{index}"
            attn_hc = PreparedHyperconnection(module.attn_hyper_connection, scratch.hyperconnection,
                priming_state=scratch.residual, session=session, name=f"{prefix}.attn_hc",
                runtime=runtime, coordinator=coordinator)
            mlp_hc = PreparedHyperconnection(module.mlp_hyper_connection, scratch.hyperconnection,
                priming_state=scratch.residual, session=session, name=f"{prefix}.mlp_hc",
                runtime=runtime, coordinator=coordinator)
            if index % 4 == 3:
                attention = PreparedAttention(module.self_attn, layer_weights.attention, source,
                    scratch.qsa, qsa_states[index], state.qsa, rotary, session=session, name=f"{prefix}.qsa",
                    workspace=workspaces, projection_workspace=workspaces,
                    max_cache_pages=max_cache_pages, draft_tokens=state.pool.layout.speculative_tokens)
            else:
                attention = PreparedGDN(module.linear_attn, layer_weights.attention, source,
                    scratch.gdn, state.pool.gdn[index], state.gdn, mode=state.mode,
                    session=session, name=f"{prefix}.gdn", checkpoint_export=False,
                    workspace=workspaces, projection_workspace=workspaces,
                    records=None if gdn_records is None else gdn_records[index], in_place=gdn_in_place,
                    partitions=getattr(state, "gdn_partitions", None),
                    gdn_buffers=None if gdn_buffers is None else gdn_buffers.fork())
            moe = PreparedQwenMoe(source, layer_weights.moe, session=session, name=f"{prefix}.moe",
                scratch=scratch.moe, workspace=workspaces,
                projection_workspace=workspaces, prefill=state.mode == "prefill")
            self.layers.append((attn_hc, attention, mlp_hc, moe))
            if runtime is not None:
                from ignition.components.collectives import PreparedAllReduce

                self.reductions.append(tuple(PreparedAllReduce(value, scratch.hidden,
                    runtime=runtime, session=session, coordinator=coordinator, name=f"{prefix}.{label}_reduce")
                    for label, value in (("attention", attention.scratch.output), ("moe", moe.output))))
            if progress is not None:
                progress(f"prepared {name} layer {index + 1}/48")
        module = language.layers[1].ple
        self.ple = PreparedPLE(module,
            {key: getattr(module, key).weight for key in ("key_proj", "value_proj")},
            scratch.residual, scratch.ple, state.pool.ple_convolution, state.ple, ple_table,
            mode=state.mode, session=session, name=f"{name}.ple", workspace=workspaces,
            partitions=getattr(state, "ple_partitions", None), runtime=runtime, coordinator=coordinator)
        self.final_mixer = PreparedHyperconnection(language.hyper_connection_mixer,
            replace(scratch.hyperconnection, block_input=scratch.hidden), priming_state=scratch.residual,
            session=session, name=f"{name}.final_mixer", runtime=runtime, coordinator=coordinator)
        for index, (attn_hc, _, mlp_hc, _) in enumerate(self.layers):
            attn_hc.prepare_combine_norm(mlp_hc, priming_state=scratch.residual,
                session=session, name=f"{name}.layer{index}.attn_combine_norm")
            if index != 0:
                following = self.layers[index + 1][0] if index + 1 < len(self.layers) else self.final_mixer
                mlp_hc.prepare_combine_norm(following, priming_state=scratch.residual,
                    session=session, name=f"{name}.layer{index}.mlp_combine_norm")
        state.prepare_hidden(scratch.hidden, scratch.sample_hidden)
        state.prepare_residual(scratch.residual)
        if vocabulary_runtime is None:
            self.lm_head = UnquantizedProjection(scratch.sample_hidden, weights.tree.lm_head.weight,
                scratch.logits, session=session, name=f"{name}.lm_head")
        else:
            from ignition.components.vocabulary import PreparedVocabularyProjection

            self.lm_head = PreparedVocabularyProjection(scratch.sample_hidden, weights.tree.lm_head.weight,
                scratch.local_logits, scratch.logits, runtime=vocabulary_runtime, session=session,
                coordinator=coordinator, name=f"{name}.lm_head")
        self.scoring = None if scoring is None else scoring.prepare(scratch.hidden, weights.tree.lm_head.weight,
            vocabulary=248320, session=session, coordinator=coordinator, name=name + ".prompt")
        self.sampler = None
        if not verification:
            from ignition.components.sampling import PreparedSampling

            state, inputs = self.state, self.state.inputs
            self.sampler = PreparedSampling(self.scratch.logits, state.requests,
                inputs.row_slot_ids[:state.rows], inputs.row_generations[:state.rows],
                inputs.live_rows, completion, frame_index, submission_sequence,
                sample_enabled=inputs.sample_enabled[:state.rows], workspace=sampling_workspace)
            self.sampler.prepare()
            self.sampler.prepare_broadcast(runtime=vocabulary_runtime, session=session,
                coordinator=coordinator, name=name + ".sampling")

    def capture_context(self, metadata, draft_config, *, buffers=None, compact_prefill=False):
        from ignition.components.dflash2 import PreparedContextCapture

        self.context_layers = draft_config.target_layers
        self.context_capture = PreparedContextCapture(metadata,
            2560 * len(self.context_layers), draft_config.attention.window,
            compact_prefill=compact_prefill, buffers=buffers)
        self.auxiliary = self.context_capture.output

    def __call__(self):
        sampler = self.sampler
        s = self.scratch
        if self.embedding_injection is not None:
            self.embedding_injection.validate()
        self.state.begin()
        if self.context_capture is not None:
            self.context_capture.begin()
        self.ple.embedding_lookup.publish()
        self.embedding()
        if self.embedding_injection is not None:
            self.embedding_injection.inject()
        if self.stage_outputs:
            self.stage_outputs["embedding"].copy_(s.embedding)
        s.residual.view(-1, 4, 2560).copy_(s.embedding[:, None, :])
        if self.stage_outputs:
            self.stage_outputs["residual_in"].copy_(s.residual)
        for index, (attn_hc, attention, mlp_hc, moe) in enumerate(self.layers):
            if index == 1:
                s.residual.add_(self.ple(s.residual))
            mixed = attn_hc.mix(s.residual, pre_normalized=index not in (0, 1))
            if index == 0 and self.stage_outputs:
                self.stage_outputs["attn_mix"].copy_(mixed)
                for field, value in attn_hc.trace_intermediates().items():
                    self.stage_outputs[f"attn_{field}"].copy_(value)
            block = attention(mixed)
            if self.reductions:
                block = self.reductions[index][0](block)
            if index == 0 and self.stage_outputs:
                self.stage_outputs["attn_output"].copy_(block)
                for field in ("projected_qkv", "convolved_qkv", "recurrence", "gated"):
                    self.stage_outputs[f"gdn_{field}"].copy_(getattr(s.gdn, field))
            attn_hc.combine_norm(s.residual, block)
            if index == 0 and self.stage_outputs:
                self.stage_outputs["attn_residual"].copy_(s.residual)
            mixed = mlp_hc.mix(s.residual, pre_normalized=True)
            if index == 0 and self.stage_outputs:
                self.stage_outputs["mlp_mix"].copy_(mixed)
                for field, value in mlp_hc.trace_intermediates().items():
                    self.stage_outputs[f"mlp_{field}"].copy_(value)
            block = moe(mixed)
            if self.reductions:
                block = self.reductions[index][1](block)
            if index == 0 and self.stage_outputs:
                self.stage_outputs["mlp_output"].copy_(block)
                self.stage_outputs["moe_router_logits"].copy_(moe.router_logits)
                self.stage_outputs["moe_ids"].copy_(moe.router.ids)
                self.stage_outputs["moe_weights"].copy_(moe.router.weights)
                self.stage_outputs["moe_shared"].copy_(moe.shared.finish())
                self.stage_outputs["moe_routed"].copy_(moe.routed_output)
            if index == 0:
                mlp_hc.combine(s.residual, block)
            else:
                mlp_hc.combine_norm(s.residual, block)
            if index == 0 and self.stage_outputs:
                self.stage_outputs["mlp_residual"].copy_(s.residual)
            if self.layer_outputs is not None:
                self.layer_outputs[index].copy_(s.residual)
            if index in self.context_layers:
                features = s.residual.view(-1, 4, 2560).mean(dim=1)
                self.context_capture(features, self.context_layers.index(index) * 2560)
        if self.residual_output is not None:
            self.residual_output.copy_(s.residual)
        self.final_mixer.mix(s.residual, pre_normalized=True)
        if self.scoring is not None:
            self.scoring()
        self.state.finish()
        self.lm_head(s.sample_hidden)
        if sampler is not None:
            sampler()
