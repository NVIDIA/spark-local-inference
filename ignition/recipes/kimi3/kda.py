# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 KDA with fused input projections and a full-rank output gate."""

import torch
from b12x.sequence import gdn_decode, kda_prefill, short_conv
from b12x.sequence.gdn_decode._kernels import _gdn_decode_op as kda_decode_out
from b12x.sequence.kda_prefill._ops import _run as kda_prefill_out

from ignition.components.gated_norm import PreparedGatedRMSNorm
from ignition.components.recurrent import prepare_sequence_binding, run_short_conv
from ignition.components.kda_speculation import _record, register_recording
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.tuning import register_finalizer

from .projections import prepare_projection


class PreparedKDA:
    def __init__(self, weights, source, state, metadata, *, mode, eps, session, name, workspace, records=None, examples=None):
        mode = "decode" if mode == "verify" else mode
        self.state, self.weights = state, weights
        self.recording = records is not None
        self.tokens, self.hidden, self.eps = source.shape[0], source.shape[1], eps
        tokens, hidden = source.shape
        heads, width = state.recurrent.shape[1], state.recurrent.shape[1] * 128
        device = source.device

        def buffer(key, channels):
            return declaration_tensor(examples, "kda." + key, (tokens, channels), torch.bfloat16, device)

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
        self.recurrent_shape = tuple(recurrent.shape)
        self.width, self.heads, self.columns = width, heads, columns
        self.norm.functional(dynamic_live=True) if mode != "decode" else None
        for projection in self.projections.values():
            projection.functional(session=session)
        self.handle = register_recording(self) if self.recording else None

        def release():
            self.scratch_bytes = sum(spec.nbytes for spec in self.plan.scratch_specs())
            self.convolution = self.recurrence = None
            self.forget = self.forget_source = self.input = None
        register_finalizer(session, release, workspace=workspace)

    def __call__(self, source, metadata, *, records=None):
        m, w = metadata, self.weights
        tokens, heads, width = source.shape[0], self.heads, self.width
        projected = self.projections["input"](source)
        qkv, forget_source, gate, beta, _ = projected.split(
            (3 * width, 128, width, heads, self.columns - 4 * width - 128 - heads), dim=-1)
        raw_g = self.projections["forget"](forget_source.contiguous())
        mixed = torch.empty((tokens, 3 * width), dtype=source.dtype, device=source.device)
        conv_scratch = torch.empty(sum(spec.nbytes for spec in self.conv_plan.scratch_specs()), dtype=torch.uint8, device=source.device)
        run_short_conv(self.conv_plan, scratch=conv_scratch, x=qkv, conv_weight=w["convolution"],
            query_start_loc=m.query_start_loc, state_slot_ids=m.convolution_slots, state_is_fresh=m.fresh,
            num_accepted_tokens=m.accepted_tokens, num_seqs=m.num_seqs, num_tokens=m.num_tokens,
            conv_state=self.state.convolution, out=mixed)
        recurrent = self.state.recurrent[:self.recurrent_shape[0]]
        output = torch.empty((tokens, heads, 128), dtype=source.dtype, device=source.device)
        raw_g = raw_g.view(tokens, heads, 128)
        if self.mode == "decode":
            z = gate.view(tokens, heads, 128)
            reads = [mixed, raw_g, beta, z, w["A_log"], w["dt_bias"].view(heads, 128), w["norm"],
                     m.query_start_loc, m.accepted_tokens, m.state_indices, m.num_seqs, m.num_tokens]
            if self.recording:
                scratch = torch.empty(self.scratch_bytes, dtype=torch.uint8, device=source.device)
                _record(reads, recurrent, output, *records, scratch, self.handle)
            else:
                kda_decode_out(*reads[:7], recurrent, *reads[7:], output,
                               self.eps, 128 ** -.5, -5.0, self.plan.handle)
            gated = output
        else:
            q, k, v = (x.view(tokens, heads, 128) for x in mixed.split(width, dim=-1))
            scratch = torch.empty(self.scratch_bytes, dtype=torch.uint8, device=source.device)
            kda_prefill_out([q, k, v, raw_g, beta, w["A_log"], w["dt_bias"].view(heads, 128),
                m.query_start_loc, m.initial_state_indices, m.final_state_indices,
                m.checkpoint_state_indices, m.checkpoint_offsets, m.num_seqs, m.num_tokens],
                scratch, recurrent, output, self.plan.handle, -5.0, None, 1e-6, None, None)
            gated = self.norm(output, gate.view(tokens, heads, 128), live=m.num_tokens)
        return self.projections["output"](gated.view(tokens, width))

    def run_recorded(self, reads, recurrent_state, output, corrections, keys_gates, scratch):
        (mixed_qkv, raw_g, raw_beta, z, A_log, dt_bias, norm_weight, query_start_loc,
         num_accepted_tokens, state_indices, num_seqs, num_tokens) = reads
        binding = gdn_decode.bind_kda(self.plan, scratch=scratch, mixed_qkv=mixed_qkv,
            raw_g=raw_g, raw_beta=raw_beta, z=z, A_log=A_log, dt_bias=dt_bias,
            norm_weight=norm_weight, recurrent_state=recurrent_state, query_start_loc=query_start_loc,
            num_accepted_tokens=num_accepted_tokens, state_indices=state_indices,
            num_seqs=num_seqs, num_tokens=num_tokens, output=output,
            correction_cache=corrections, kg_cache=keys_gates)
        gdn_decode.run_kda(binding, lower_bound=-5.0, eps=self.eps)
