# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 KDA with fused input projections and a full-rank output gate."""

from types import SimpleNamespace

import torch
from b12x.sequence import gdn_decode, kda_prefill, short_conv

from ignition.components.gated_norm import PreparedGatedRMSNorm
from ignition.components.recurrent import prepare_sequence_binding
from ignition.components.kda_speculation import _record, _record_fused, register_recording
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.tuning import register_finalizer
from ignition.components.recurrent import run_short_conv
from b12x.sequence.gdn_decode._kernels import _gdn_decode_op as kda_decode_out
from b12x.sequence.kda_prefill._ops import _run as kda_prefill_out

from .projections import prepare_projection


class PreparedKDA:
    def __init__(self, weights, source, state, metadata, *, mode, eps, session, name, workspace, buffers, records=None):
        mode = "decode" if mode == "verify" else mode
        self.recording, self.state, self.metadata = records is not None, state, metadata
        self.weights, self.eps = weights, eps
        tokens, hidden = source.shape
        self.tokens, self.fused = tokens, None
        heads, width = state.recurrent.shape[1], state.recurrent.shape[1] * 128
        device = source.device

        def buffer(key, channels):
            role = 'kimi.hidden_output' if key == 'output' else 'kda.' + key
            return declaration_tensor(buffers, role, (tokens, channels), torch.bfloat16, device)

        self.heads, self.width = heads, width

        self.forget = buffer("forget_input", 128)
        raw_g = buffer("raw_g", width)
        mixed = buffer("mixed", 3 * width)
        recurrence, gated = buffer("recurrence", width), buffer("gated", width)
        columns = (4 * width + 128 + heads + 7) // 8 * 8
        self.input = buffer("input", columns)
        qkv, self.forget_source, gate, beta, _ = self.input.split(
            (3 * width, 128, width, heads, columns - 4 * width - 128 - heads), dim=-1)
        self.projections = {key: prepare_projection(x, weights[key], y, session=session,
            name=name + "." + key, workspace=workspace) for key, x, y in (
                ("input", source, self.input), ("forget", self.forget, raw_g),
                ("output", gated, buffer("output", hidden)))}
        sequences = metadata.query_start_loc.numel() - 1
        conv_args = dict(x=qkv, conv_weight=weights["convolution"], query_start_loc=metadata.query_start_loc,
            state_slot_ids=metadata.convolution_slots, state_is_fresh=metadata.fresh,
            num_accepted_tokens=metadata.accepted_tokens, num_seqs=metadata.num_seqs, num_tokens=metadata.num_tokens,
            conv_state=state.convolution, out=mixed)
        caps = short_conv.Caps(device=device, mode=mode, channels=3 * width, kernel_size=4,
            max_tokens=tokens, max_seqs=sequences, max_state_slots=state.convolution.shape[0],
            max_speculative_tokens=state.convolution.shape[-1] - 3)
        plan = short_conv.plan(caps, invocation=short_conv.invocation_from_tensors(**conv_args))
        self.conv_plan = plan
        prepare_sequence_binding(plan, conv_args, session=session, name=name + ".conv", mode=mode,
            public_bind=short_conv.bind, bind_plan=True, workspace=workspace,
            set_binding=lambda binding: setattr(self, "convolution", binding))
        recurrent = state.recurrent if records is None else state.recurrent[:records[0].shape[0]]
        self.recurrent_state = recurrent
        common = dict(raw_g=raw_g.view(tokens, heads, 128), raw_beta=beta, A_log=weights["A_log"],
            dt_bias=weights["dt_bias"].view(heads, 128), recurrent_state=recurrent,
            num_seqs=metadata.num_seqs, num_tokens=metadata.num_tokens,
            output=(gated if mode == "decode" else recurrence).view(tokens, heads, 128))
        self.options, self.mode = {"lower_bound": -5.0}, mode
        if mode == "decode":
            self.options["eps"] = eps
            caps = gdn_decode.Caps(device=device, max_tokens=tokens, max_seqs=sequences,
                max_state_slots=recurrent.shape[0], key_heads=heads, value_heads=heads,
                state_index_columns=metadata.state_indices.shape[1], gate_activation="sigmoid", null_state_index=0,
                recover_speculative_state=records is not None)
            args = dict(common, mixed_qkv=mixed, z=gate.view(tokens, heads, 128), norm_weight=weights["norm"],
                query_start_loc=metadata.query_start_loc, num_accepted_tokens=metadata.accepted_tokens,
                state_indices=metadata.state_indices)
            if records is not None:
                args.update(correction_cache=records[0], kg_cache=records[1])
                self.handle = register_recording(self)
            plan = gdn_decode.plan(caps, invocation=gdn_decode.invocation_from_tensors(caps, **args))
            bind, method = gdn_decode.bind_kda, "bind_kda"
        else:
            caps = kda_prefill.Caps(device=device, max_tokens=tokens, max_seqs=sequences,
                max_state_slots=state.recurrent.shape[0], heads=heads, null_state_index=0)
            q, k, v = (x.view(tokens, heads, 128) for x in mixed.split(width, dim=-1))
            args = dict(common, q=q, k=k, v=v, cu_seqlens=metadata.query_start_loc,
                initial_state_indices=metadata.initial_state_indices, final_state_indices=metadata.final_state_indices,
                checkpoint_state_indices=metadata.checkpoint_state_indices, checkpoint_offsets=metadata.checkpoint_offsets)
            plan = kda_prefill.plan(caps, invocation=kda_prefill.invocation_from_tensors(**args))
            bind, method = kda_prefill.bind, "bind"
            self.norm = PreparedGatedRMSNorm(recurrence.view(tokens, heads, 128), gate.view(tokens, heads, 128),
                weights["norm"], gated.view(tokens, heads, 128), metadata.num_tokens, eps=eps)
        self.plan = plan
        prepare_sequence_binding(plan, args, session=session, name=name + ".recurrence", mode=mode,
            public_bind=bind, bind_method=method, workspace=workspace, run_options=self.options,
            set_binding=lambda binding: setattr(self, "recurrence", binding))
        for projection in self.projections.values():
            projection.functional(session=session)
        if mode == "prefill":
            self.norm.functional(session=session, dynamic_live=True)
        register_finalizer(session, self._finalize)
        if workspace is not None:
            workspace.register_preparation_release(self._release_bindings)

    def _finalize(self):
        self.conv_bytes = self.conv_plan.scratch_specs()[0].nbytes
        self.recurrence_bytes = sum(spec.nbytes for spec in self.plan.scratch_specs())
        if self.recording and self.tokens <= 8:
            try:
                self.fused = gdn_decode.prepare_kda_fused_verify(
                    self.recurrence, self.convolution, retain_examples=False)
            except ValueError as error:
                import warnings

                warnings.warn(f"KDA fused verify disabled for this layer: {error}", stacklevel=2)
        self.input = self.forget = self.forget_source = self.metadata = None

    def _release_bindings(self):
        self.convolution = self.recurrence = None

    def __call__(self, source, metadata=None, *, records=None):
        if metadata is None:
            raise ValueError("KDA execution requires invocation metadata")
        if self.recording and records is None:
            raise ValueError("KDA verification requires invocation records")
        tokens, heads, width = source.shape[0], self.heads, self.width
        projected = self.projections["input"](source)
        qkv, forget, gate, beta, _ = projected.split(
            (3 * width, 128, width, heads, projected.shape[1] - 4 * width - 128 - heads), dim=-1)
        forget_input = torch.empty((tokens, 128), dtype=source.dtype, device=source.device)
        forget_input.copy_(forget)
        raw_g = self.projections["forget"](forget_input).view(tokens, heads, 128)
        gate = gate.view(tokens, heads, 128)
        if self.recording and self.fused is not None:
            output = torch.empty((tokens, heads, 128), dtype=source.dtype, device=source.device)
            _record_fused([qkv, raw_g, beta, gate, self.weights["A_log"], self.weights["dt_bias"].view(heads, 128),
                self.weights["norm"], self.weights["convolution"], metadata.query_start_loc,
                metadata.convolution_slots, metadata.fresh, metadata.accepted_tokens, metadata.state_indices,
                metadata.num_seqs, metadata.num_tokens], output, *records, self.state.convolution, self.handle)
            return self.projections["output"](output.reshape(tokens, -1))
        mixed = torch.empty((tokens, 3 * width), dtype=source.dtype, device=source.device)
        conv_scratch = torch.empty(self.conv_bytes, dtype=torch.uint8, device=source.device)
        run_short_conv(self.conv_plan, x=qkv, conv_weight=self.weights["convolution"],
            query_start_loc=metadata.query_start_loc, state_slot_ids=metadata.convolution_slots,
            state_is_fresh=metadata.fresh, num_accepted_tokens=metadata.accepted_tokens,
            num_seqs=metadata.num_seqs, num_tokens=metadata.num_tokens,
            conv_state=self.state.convolution, out=mixed, scratch=conv_scratch)
        gate = gate.view(tokens, heads, 128)
        output = torch.empty((tokens, heads, 128), dtype=source.dtype, device=source.device)
        dt_bias = self.weights["dt_bias"].view(heads, 128)
        if self.recording:
            scratch = torch.empty(self.recurrence_bytes, dtype=torch.uint8, device=source.device)
            _record([mixed, raw_g, beta, gate, self.weights["A_log"], dt_bias, self.weights["norm"],
                metadata.query_start_loc, metadata.accepted_tokens, metadata.state_indices,
                metadata.num_seqs, metadata.num_tokens], self.recurrent_state, output,
                *records, scratch, self.handle)
        elif self.mode == "decode":
            kda_decode_out(mixed, raw_g, beta, gate, self.weights["A_log"], dt_bias,
                self.weights["norm"], self.recurrent_state, metadata.query_start_loc,
                metadata.accepted_tokens, metadata.state_indices, metadata.num_seqs,
                metadata.num_tokens, output, self.eps, 128 ** -.5, -5.0, self.plan.handle)
        else:
            scratch = torch.empty(self.recurrence_bytes, dtype=torch.uint8, device=source.device)
            q, k, v = (value.view(tokens, heads, 128) for value in mixed.split(width, dim=-1))
            kda_prefill_out([q, k, v, raw_g, beta, self.weights["A_log"], dt_bias,
                metadata.query_start_loc, metadata.initial_state_indices, metadata.final_state_indices,
                metadata.checkpoint_state_indices, metadata.checkpoint_offsets, metadata.num_seqs,
                metadata.num_tokens], scratch, self.recurrent_state, output,
                self.plan.handle, -5.0, None, 1e-6, None, None)
            output = self.norm(output, gate, live=metadata.num_tokens)
        return self.projections["output"](output.reshape(tokens, -1))

    def run_fused(self, reads, output, corrections, keys_gates, convolution_state):
        (qkv, raw_g, raw_beta, z, A_log, dt_bias, norm_weight, conv_weight, query_start_loc,
         convolution_slots, fresh, accepted_tokens, state_indices, num_seqs, num_tokens) = reads
        binding = SimpleNamespace(raw_g=raw_g, raw_beta=raw_beta, A_log=A_log, dt_bias=dt_bias,
            recurrent_state=self.recurrent_state, query_start_loc=query_start_loc, state_indices=state_indices,
            num_seqs=num_seqs, num_tokens=num_tokens, output=output, z=z, norm_weight=norm_weight,
            correction_cache=corrections, kg_cache=keys_gates)
        convolution = SimpleNamespace(x=qkv, conv_weight=conv_weight, state_slot_ids=convolution_slots,
            state_is_fresh=fresh, num_accepted_tokens=accepted_tokens, conv_state=convolution_state)
        self.fused.run_dynamic(binding, convolution, scale=128 ** -.5, lower_bound=-5.0, eps=self.eps)

    def run_recorded(self, reads, recurrent_state, output, corrections, keys_gates, scratch):
        (mixed_qkv, raw_g, raw_beta, z, A_log, dt_bias, norm_weight, query_start_loc,
         num_accepted_tokens, state_indices, num_seqs, num_tokens) = reads
        binding = gdn_decode.bind_kda(self.plan, scratch=scratch,
            mixed_qkv=mixed_qkv, raw_g=raw_g, raw_beta=raw_beta, z=z, A_log=A_log,
            dt_bias=dt_bias, norm_weight=norm_weight, recurrent_state=recurrent_state,
            query_start_loc=query_start_loc, num_accepted_tokens=num_accepted_tokens,
            state_indices=state_indices, num_seqs=num_seqs, num_tokens=num_tokens,
            output=output, correction_cache=corrections, kg_cache=keys_gates)
        gdn_decode.run_kda(binding, lower_bound=-5.0, eps=self.eps)
