# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Mixed prompt and decode execution through one packed Qwen target forward."""

from copy import copy
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from ignition.components.sampling import PreparedSampling

from .decoder import PreparedDecoder
from .decoder_state import PreparedDecoderState
from .mixed_state import MixedPacking, PreparedMixedState, role_inputs
from .speculative_model import PreparedPrefillAbsorption, PreparedSpeculativeRound
from .vision_state import PreparedEmbeddingInjection


@dataclass(frozen=True)
class MixedAuxiliary:
    residual: torch.Tensor
    mixed: tuple


class MixedEmbeddingInjection:
    def __init__(self, packing, features, state, embedding, scratch):
        self.packing, self.embedding, self.scratch = packing, embedding, scratch
        i = state.inputs
        self.prompt = PreparedEmbeddingInjection(features, state.requests, i.vision_token_indices,
            i.row_slot_ids, i.row_generations, i.query_lengths, i.live_rows, scratch)
        self.prompt.prepare()

    def functional(self):
        self.prompt.functional(dynamic_inputs=True)
        self.packing = self.embedding = self.scratch = None
        return self

    def validate_runtime(self, packing):
        return self.prompt.validate(packing.prompt)

    def inject(self, embedding, errors):
        errors, packing = errors
        scratch = embedding[packing.prompt_indices]
        self.prompt.inject(scratch, errors, packing.prompt)
        embedding.copy_(torch.where(packing.is_prompt[:, None], scratch[packing.prompt_source], embedding))


class PreparedMixedRound:
    """Keep role-specific sampling and commits around a shared target program."""

    def __init__(self, owner, batch, width, prefill, name, *, scratch, features, enabled):
        self.owner, self.enabled = owner, enabled
        self.prompt_inputs = role_inputs(owner.inputs, prefill)
        self.decode_inputs = role_inputs(owner.inputs, batch * width)
        self.packing = MixedPacking(owner.inputs, self.prompt_inputs, self.decode_inputs, batch, width, prefill)
        self.prompt_state = PreparedDecoderState(self.prompt_inputs, owner.requests, owner.pool,
            tokens=prefill, rows=1, mode="prefill", maintain_draft_state=enabled)
        if not hasattr(owner, "mixed_completions"):
            owner.mixed_completions = (owner.completion.private(), owner.completion.private())
        self.prompt_completion, self.decode_completion = owner.mixed_completions
        self.prompt_decoder = SimpleNamespace(state=self.prompt_state)
        self.absorb = None
        if enabled:
            prompt_owner = copy(owner)
            prompt_owner.inputs = self.prompt_inputs
            self.absorb = PreparedPrefillAbsorption(prompt_owner, self.prompt_decoder, f"{name}.prompt",
                                                    residual_output=scratch.prompt_residual[:prefill])
            owner.owners.append(self.absorb)
        else:
            self.prompt_decoder.residual_output = scratch.prompt_residual[:prefill]

        def decoder_factory(weights, decode_scratch, decode_state, cache, rotary, table, workspaces,
                            completion, frame, sequence, **kwargs):
            decode_residual = kwargs.pop("residual_output")
            kwargs.pop("verification", None)
            state = PreparedMixedState(self.packing, self.prompt_state, decode_state, scratch,
                prompt_residual=self.prompt_decoder.residual_output, decode_residual=decode_residual)
            sample_rows = batch * width + 1
            full_scratch = owner.scratch.rows(self.packing.tokens, sample_rows)
            injection = MixedEmbeddingInjection(self.packing, features, self.prompt_state,
                full_scratch.embedding, scratch.prompt_gdn.output[:prefill])
            self.decoder = PreparedDecoder(weights, full_scratch, state, cache, rotary, table, workspaces,
                completion, frame, sequence, verification=True, embedding_injection=injection, **kwargs)
            self.prompt_sampler = PreparedSampling(full_scratch.logits[-1:], owner.requests,
                self.prompt_inputs.row_slot_ids[:1], self.prompt_inputs.row_generations[:1],
                self.prompt_inputs.live_rows, self.prompt_completion, frame, sequence,
                sample_enabled=self.prompt_inputs.sample_enabled[:1], workspace=owner.sampling_workspace)
            self.prompt_sampler.prepare()
            self.prompt_sampler.prepare_broadcast(runtime=owner.collectives["vocabulary_runtime"],
                session=owner.session, coordinator=owner.collectives["coordinator"], name=name + ".prompt.sampling")
            return SimpleNamespace(scratch=decode_scratch, forward=self.decoder)

        if enabled:
            decode_owner = copy(owner)
            decode_owner.inputs, decode_owner.completion = self.decode_inputs, self.decode_completion
            # The speculative round requires a callable exposing only its verification logits.
            def speculative_decoder(*args, **kwargs):
                view = decoder_factory(*args, **kwargs)
                return DecodeTarget(view.scratch, view.forward, self.prompt_sampler)
            self.round = PreparedSpeculativeRound(decode_owner, batch, width, name,
                decoder_factory=speculative_decoder, residual_output=scratch.decode_residual[:batch * width])
            self.round.prepare()
            owner.owners.append(self.round)
        else:
            state = PreparedDecoderState(self.decode_inputs, owner.requests, owner.pool,
                tokens=batch, rows=batch, mode="decode", maintain_draft_state=False)
            residual = scratch.decode_residual[:batch]
            decoder_factory(owner.weights, owner.scratch.rows(batch, batch), state, owner.cache,
                owner.rotary, owner.table, owner.workspaces, self.decode_completion, owner.frame, owner.sequence,
                session=owner.session, name=name, max_cache_pages=owner.max_cache_pages,
                residual_output=residual, **owner.collectives)
            self.decode_sampler = PreparedSampling(self.decoder.scratch.logits[:batch], owner.requests,
                self.decode_inputs.row_slot_ids[:batch], self.decode_inputs.row_generations[:batch],
                self.decode_inputs.live_rows, self.decode_completion, owner.frame, owner.sequence,
                sample_enabled=self.decode_inputs.sample_enabled[:batch], workspace=owner.sampling_workspace)
            self.decode_sampler.prepare()
            self.decode_sampler.prepare_broadcast(runtime=owner.collectives["vocabulary_runtime"],
                session=owner.session, coordinator=owner.collectives["coordinator"], name=name + ".decode.sampling")

    def functional(self, *, session):
        self.packing.functional()
        if self.enabled:
            self.round.functional(session=session, dynamic_inputs=True)
        else:
            self.decoder.functional(session=session)
            self.prompt_sampler.functional(session=session, dynamic_inputs=True)
            self.decode_sampler.functional(session=session, dynamic_inputs=True)
        if self.absorb is not None:
            self.absorb.mtp.functional(session=session)
            self.absorb.metadata.functional()
            self.absorb.mtp.release_outputs()
            self.absorb.before = self.absorb.previous = self.absorb.previous_rope = None
            self.absorb.indices = self.absorb.target_residual = None
            self.absorb.owner.inputs = None
            self.absorb.owner.release_examples()
        if self.enabled:
            self.round.owner.inputs = None
            self.round.owner.release_examples()
        self.prompt_decoder.residual_output = None
        self.decoder.release_outputs()
        self.prompt_inputs = self.decode_inputs = None
        return self

    def __call__(self):
        from ignition.components.state import gather_rows

        o = self.owner
        packing = self.packing.runtime()
        saved = None
        if self.absorb is not None:
            slots = packing.prompt.row_slot_ids[:1]
            safe = slots.long().clamp(0, o.requests.capacity - 1)
            before = o.requests.consumed_tokens[safe]
            previous = torch.empty((1, 10240), dtype=torch.bfloat16, device=slots.device)
            rope = torch.empty((1, 3), dtype=torch.int64, device=slots.device)
            state_rows = packing.prompt.request_state_rows[safe, 0]
            gather_rows(o.pool.region("target_residual"), state_rows, previous)
            gather_rows(o.pool.region("target_residual_position"), state_rows, rope)
            saved = before, previous, rope
        if self.enabled:
            result = self.round.run(packing.decode, packing)
        else:
            errors = self.decoder.embedding_injection.validate_runtime(packing)
            prompt = self.prompt_state.begin_runtime(packing.prompt)
            decode = self.decoder.state.decode.begin_runtime(packing.decode)
            runtime = self.decoder.state.combine_runtime(packing, prompt, decode)
            runtime.injection_errors = (errors, packing)
            _, logits, auxiliary, _ = self.decoder(runtime)
            b = packing.batch
            self.decode_sampler.run(logits[:b], slots=packing.decode.row_slot_ids[:b],
                generations=packing.decode.row_generations[:b], live_rows=packing.decode.live_rows,
                sample_enabled=packing.decode.sample_enabled[:b])
            self.prompt_sampler.run(logits[b:], slots=packing.prompt.row_slot_ids[:1],
                generations=packing.prompt.row_generations[:1], live_rows=packing.prompt.live_rows,
                sample_enabled=packing.prompt.sample_enabled[:1])
            result = runtime, auxiliary
        if self.absorb is not None:
            runtime, auxiliary = result
            prompt_residual = runtime.gdn_partitions.parts[0].gather(auxiliary.residual)
            self.absorb.finish_runtime(packing.prompt, runtime.prompt, prompt_residual, *saved)
        o.completion.merge(self.prompt_completion, self.decode_completion,
            packing.prompt.row_slot_ids[0], o.frame)



class DecodeTarget:
    """Consume prompt logits before MTP absorption reuses target scratch."""

    def __init__(self, scratch, forward, prompt_sampler):
        self.scratch, self.forward, self.prompt_sampler = scratch, forward, prompt_sampler

    def functional(self, *, session):
        self.forward.functional(session=session)
        self.prompt_sampler.functional(session=session, dynamic_inputs=True)
        return self

    def __call__(self, decode, packing):
        state = self.forward.state
        errors = self.forward.embedding_injection.validate_runtime(packing)
        prompt = state.prompt.begin_runtime(packing.prompt)
        runtime = state.combine_runtime(packing, prompt, decode)
        runtime.injection_errors = (errors, packing)
        hidden, logits, auxiliary, records = self.forward(runtime)
        b = packing.batch * packing.width
        self.prompt_sampler.run(logits[b:], slots=packing.prompt.row_slot_ids[:1],
            generations=packing.prompt.row_generations[:1], live_rows=packing.prompt.live_rows,
            sample_enabled=packing.prompt.sample_enabled[:1])
        decode_residual = runtime.gdn_partitions.parts[1].gather(auxiliary.residual)
        return hidden[:b], logits[:b], MixedAuxiliary(decode_residual, (runtime, auxiliary)), records
