# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared single-layer MTP feedback, sparse attention and W4A16 experts."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from b12x.preparation import PreparedCall
from b12x.sequence import mtp_feedback

from ignition.components.embedding import PreparedEmbedding
from ignition.components.linear import QuantizedProjection, UnquantizedProjection
from ignition.preparation.fixtures import mtp_feedback as feedback_fixtures
from ignition.preparation.scratch import ScratchArena, planned_scratch, prime_owners
from ignition.preparation.tuning import declare, register_finalizer

from .attention import PreparedAttention, QSAScratch
from .hyperconnection import HyperconnectionScratch, PreparedHyperconnection
from .moe import PreparedQwenMoe, QwenMoeScratch
from .mtp_weights import MTPWeights


@dataclass(frozen=True)
class MTPScratch:
    """Tensor examples used to declare and tune MTP operations."""

    embedding_ids: torch.Tensor
    embedding: torch.Tensor
    residual: torch.Tensor
    hidden: torch.Tensor
    selected_hidden: torch.Tensor
    logits: torch.Tensor
    hyperconnection: HyperconnectionScratch
    qsa: QSAScratch
    moe: QwenMoeScratch
    local_logits: torch.Tensor | None = None

    @classmethod
    def allocate(cls, tokens: int, output_rows: int, device, *, world_size=1,
                 hyperconnection_lowrank=320):
        if (type(tokens) is not int or type(output_rows) is not int
                or not 0 < output_rows <= tokens):
            raise ValueError("MTP requires positive output rows no greater than token capacity")

        def zeros(rows, width):
            return torch.zeros((rows, width), dtype=torch.bfloat16, device=device)

        ids = torch.zeros(tokens, dtype=torch.int64, device=device)
        embedding, residual, hidden = zeros(tokens, 2560), zeros(tokens, 10240), zeros(tokens, 2560)
        selected, logits = zeros(output_rows, 2560), zeros(output_rows, 248320)
        local_logits = zeros(output_rows, 248320 // world_size) if world_size > 1 else None
        block_input, block_output = zeros(tokens, 2560), zeros(tokens, 2560)
        projected = zeros(tokens, (hyperconnection_lowrank + 4 + 15) // 16 * 16)
        qsa = QSAScratch.allocate(tokens, device, world_size=world_size, output=block_output)
        moe = QwenMoeScratch.allocate(tokens, device, intermediate=640 // world_size,
                                    output=block_output)
        hc = HyperconnectionScratch.allocate(tokens, device,
                                            projected=projected, block_input=block_input, lowrank=hyperconnection_lowrank)
        return cls(ids, embedding, residual, hidden, selected, logits, hc, qsa, moe, local_logits)

    def rows(self, tokens: int, output_rows: int):
        if (type(tokens) is not int or type(output_rows) is not int
                or not 0 < output_rows <= min(tokens, self.logits.shape[0])
                or not tokens <= self.residual.shape[0]):
            raise ValueError("MTP view exceeds persistent token/output capacity")
        qsa = self.qsa.rows(tokens)
        return type(self)(self.embedding_ids[:tokens], self.embedding[:tokens], self.residual[:tokens],
            self.hidden[:tokens], self.selected_hidden[:output_rows], self.logits[:output_rows],
            self.hyperconnection.rows(tokens), qsa, self.moe.rows(tokens),
            None if self.local_logits is None else self.local_logits[:output_rows])


@dataclass(frozen=True)
class MTPOutput:
    residual: torch.Tensor
    hidden: torch.Tensor
    logits: torch.Tensor


torch.utils._pytree.register_dataclass(MTPOutput)


class PreparedMTPFeedback:
    """Qwen's flattened-state normalization and shared-stream BF16 projection."""

    def __init__(self, module, token_embedding, target_hidden, output, workspace, *, session, name):
        self.eps = module.config.rms_norm_eps
        self.kwargs = {
            "token_embedding": token_embedding,
            "multi_state": target_hidden.view(-1, 4, 2560),
            "token_norm_weight": module.pre_fc_norm_embedding.weight,
            "state_norm_weight": module.pre_fc_norm_hidden.weight,
            "embedding_fc_weight": module.fc_embedding.weight,
            "hidden_fc_weight": module.fc_hidden.weight,
            "output": output.view(-1, 4, 2560),
        }
        self.plan = mtp_feedback.plan(
            mtp_feedback.Caps(device=target_hidden.device, max_tokens=target_hidden.shape[0]),
            invocation=mtp_feedback.invocation_from_tensors(**self.kwargs),
        )

        def prepare(state):
            scratch = planned_scratch(state.layout.scratch_specs(), workspace)
            binding = state.bind(scratch=scratch, **self.kwargs)
            return PreparedCall(run=lambda: state.run(binding, eps=self.eps),
                                owners=prime_owners(workspace, scratch, binding))

        def benchmark(state):
            return feedback_fixtures.benchmark_call(
                state, arguments=self.kwargs, session=session, name=name,
                workspace=workspace, eps=self.eps,
            )

        declare(session, (self.plan.request(name=name, prepare_call=prepare, benchmark_call=benchmark),))

        def finalize():
            self.scratch = planned_scratch(self.plan.scratch_specs(), workspace)
            self.binding = mtp_feedback.bind(self.plan, scratch=self.scratch, **self.kwargs)

        register_finalizer(session, finalize, workspace=workspace)

    def functional(self, *, session):
        register_finalizer(session, self._finalize_functional)
        return self

    def _finalize_functional(self):
        self.constants = tuple(self.kwargs[name] for name in
            ("token_norm_weight", "state_norm_weight", "embedding_fc_weight", "hidden_fc_weight"))
        self.output_layout = (tuple(self.kwargs["output"].shape), self.kwargs["output"].dtype)
        self.scratch_layout = (tuple(self.scratch.shape), tuple(self.scratch.stride()), self.scratch.dtype)
        self.binding = self.kwargs = self.scratch = None

    def __call__(self, token_embedding, target_hidden):
        from b12x.sequence.mtp_feedback._kernels import run_mtp_feedback

        output = torch.empty(self.output_layout[0], dtype=self.output_layout[1], device=token_embedding.device)
        shape, stride, dtype = self.scratch_layout
        scratch = torch.empty_strided(shape, stride, dtype=dtype, device=token_embedding.device)
        run_mtp_feedback(token_embedding, target_hidden.view(-1, 4, 2560), *self.constants,
                         scratch, output, eps=self.eps, plan=self.plan)
        return output.flatten(1)


class PreparedMTP:
    """Fixed storage for absorption and recursive proposal graphs.

    Each input row supplies target/recursive residual ``h[i]``, token ``x[i+1]``
    and unshifted position ``i`` through QSA metadata. The caller owns the
    accepted-history alignment and all cache commit/rollback transitions.
    ``output_indices`` selects vocabulary rows without allocating prefill-sized
    logits. Returned residuals precede the final mixer and retain all streams.
    Cache-only instances prepare and execute feedback and attention cache writes,
    omitting selection, attention output, experts, and vocabulary projection.
    """

    def __init__(self, weights: MTPWeights, scratch: MTPScratch, token_ids, target_hidden,
                 qsa_state, metadata, rotary, workspace: ScratchArena, *, output_indices,
                 session, name: str, draft_tokens: int = 7, max_cache_pages=None, cache_only: bool = False,
                 runtime=None, vocabulary_runtime=None, coordinator=None):
        tokens, device = scratch.residual.shape[0], scratch.residual.device
        if (target_hidden.shape != (tokens, 10240) or target_hidden.dtype != torch.bfloat16
                or target_hidden.device != device or not target_hidden.is_contiguous()):
            raise ValueError("MTP feedback requires contiguous BF16 [tokens,10240] residuals")
        if (token_ids.shape != (tokens,) or token_ids.dtype != torch.int64
                or token_ids.device != device or not token_ids.is_contiguous()):
            raise ValueError("MTP token IDs require contiguous int64 [tokens] on the model device")
        if (output_indices.shape != (scratch.logits.shape[0],) or output_indices.dtype != torch.int64
                or output_indices.device != device or not output_indices.is_contiguous()):
            raise ValueError("MTP output indices require contiguous int64 [output_rows]")
        if metadata.request_ids.shape != (tokens,) or metadata.request_ids.device != device:
            raise ValueError("MTP QSA row metadata must match feedback capacity and device")
        source, output = target_hidden.data_ptr(), scratch.residual.data_ptr()
        if source < output + scratch.residual.nbytes and output < source + target_hidden.nbytes:
            raise ValueError("MTP feedback input must not alias its residual output")
        self.cache_only = cache_only
        self.weights, self.scratch, self.token_ids = weights, scratch, token_ids
        self.target_hidden, self.metadata, self.output_indices = target_hidden, metadata, output_indices
        self.workspace = workspace
        self.output = MTPOutput(scratch.residual, scratch.hidden, scratch.logits)
        self.embedding = PreparedEmbedding(weights.embedding, scratch.embedding_ids, scratch.embedding,
            session=session, name=f"{name}.embedding")
        self.feedback = PreparedMTPFeedback(weights.module, scratch.embedding, target_hidden,
            scratch.residual, workspace, session=session, name=f"{name}.feedback")
        layer = weights.module.layers[0]
        source = scratch.hyperconnection.block_input
        self.attn_hc = PreparedHyperconnection(layer.attn_hyper_connection, scratch.hyperconnection,
            priming_state=scratch.residual, session=session, name=f"{name}.attn_hc",
            runtime=runtime, coordinator=coordinator)
        self.attention = PreparedAttention(layer.self_attn, weights.attention, source, scratch.qsa,
            qsa_state, metadata, rotary, session=session, name=f"{name}.qsa", draft_tokens=draft_tokens,
            workspace=workspace, projection_workspace=workspace, max_cache_pages=max_cache_pages,
            cache_only=cache_only)
        if cache_only:
            return
        self.mlp_hc = PreparedHyperconnection(layer.mlp_hyper_connection, scratch.hyperconnection,
            priming_state=scratch.residual, session=session, name=f"{name}.mlp_hc",
            runtime=runtime, coordinator=coordinator)
        self.moe = PreparedQwenMoe(source, weights.moe, session=session, name=f"{name}.moe",
            scratch=scratch.moe, workspace=workspace, projection_workspace=workspace)
        self.final_mixer = PreparedHyperconnection(weights.module.hyper_connection_mixer,
            replace(scratch.hyperconnection, block_input=scratch.hidden),
            priming_state=scratch.residual, session=session, name=f"{name}.final_mixer",
            runtime=runtime, coordinator=coordinator)
        self.attn_hc.prepare_combine_norm(self.mlp_hc, priming_state=scratch.residual,
            session=session, name=f"{name}.attn_combine_norm")
        self.reductions = ()
        if runtime is not None:
            from ignition.components.collectives import PreparedAllReduce

            self.reductions = tuple(PreparedAllReduce(value, scratch.hidden, runtime=runtime,
                session=session, coordinator=coordinator, name=f"{name}.{label}_reduce")
                for label, value in (("attention", scratch.qsa.output), ("moe", self.moe.output)))
        if vocabulary_runtime is None and weights.draft_head is not None:
            self.lm_head = QuantizedProjection(scratch.selected_hidden, weights.draft_head, scratch.logits,
                                               session=session, name=f"{name}.lm_head", activation_mode="a16", workspace=workspace)
        elif vocabulary_runtime is None:
            self.lm_head = UnquantizedProjection(scratch.selected_hidden, weights.lm_head, scratch.logits,
                                         session=session, name=f"{name}.lm_head")
        else:
            from ignition.components.vocabulary import PreparedVocabularyProjection

            head = weights.draft_head if weights.draft_head is not None else weights.lm_head
            self.lm_head = PreparedVocabularyProjection(scratch.selected_hidden, head,
                scratch.local_logits, scratch.logits, runtime=vocabulary_runtime, session=session,
                coordinator=coordinator, name=f"{name}.lm_head", workspace=workspace)

    def rebind_pages_for_preparation(self, state):
        self.attention.rebind_pages_for_preparation(state)

    def functional(self, *, session):
        if getattr(self, "_functional_requested", False):
            return self
        self._functional_requested = True
        self.embedding.functional(session=session, dynamic_inputs=True)
        self.feedback.functional(session=session)
        self.attn_hc.functional(session=session)
        self.attention.functional(session=session)
        if not self.cache_only:
            self.mlp_hc.functional(session=session)
            self.moe.functional(session=session)
            self.final_mixer.functional(session=session)
            self.lm_head.functional(session=session)
            for reduction in self.reductions:
                reduction.functional(session=session)
        return self

    def release_outputs(self):
        self.scratch = self.output = self.token_ids = self.target_hidden = None
        self.metadata = self.output_indices = None

    def __call__(self, token_ids, target_hidden, metadata, output_indices):
        active = metadata.request_ids >= 0
        embedding_ids = torch.where(active, token_ids, 0)
        embedding = self.embedding(token_ids=embedding_ids)
        residual = self.feedback(embedding, target_hidden)
        residual.masked_fill_(~active[:, None], 0)
        mixed, injection = self.attn_hc.mix_runtime(residual)
        block = self.attention(mixed, metadata)
        if self.cache_only:
            return None
        if self.reductions:
            block = self.reductions[0](block)
        residual, normalized = self.attn_hc.combine_norm_runtime(residual, block, injection)
        mixed, injection = self.mlp_hc.mix_runtime(residual, normalized)
        block = self.moe(mixed)
        if self.reductions:
            block = self.reductions[1](block)
        residual = self.mlp_hc.combine_runtime(residual, block, injection)
        residual.masked_fill_(~active[:, None], 0)
        hidden, _ = self.final_mixer.mix_runtime(residual)
        return MTPOutput(residual, hidden, self.lm_head(torch.index_select(hidden, 0, output_indices)))
