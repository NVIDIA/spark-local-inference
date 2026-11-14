# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mamba-2 projection, causal convolution, SSD, and grouped gated norm."""

import torch
from b12x.sequence import short_conv

from ignition.components.gated_norm import PreparedGatedRMSNorm
from ignition.components.linear import QuantizedProjection
from ignition.components.mamba import PreparedMambaScan, PreparedMambaUpdate
from ignition.components.mamba_speculation import PreparedMambaVerification
from ignition.components.recurrent import prepare_sequence_binding, run_short_conv
from ignition.preparation.examples import declaration_tensor


def _metadata(metadata):
    return {name: getattr(metadata, name) for name in (
        "query_start_loc", "convolution_slots", "fresh", "accepted_tokens", "state_indices",
        "initial_state_indices", "final_state_indices", "num_seqs", "num_tokens")}


class PreparedMamba:
    def __init__(self, weights, source, state, metadata, config, *, mode, session, name, workspace,
                 replay=None, records=None, verification_width=1, examples=None):
        c, tokens = config.text, source.shape[0]
        heads, dim, n, groups = c["mamba_num_heads"], c["mamba_head_dim"], c["ssm_state_size"], c["n_groups"]
        width, channels = config.mamba_width, config.convolution_width

        def buffer(role, columns):
            return declaration_tensor(examples, "nemotron.target.mamba." + role,
                                      (tokens, columns), torch.bfloat16, source.device)

        projected = buffer("projected", width + channels + heads)
        gate, xbc, dt = projected.split((width, channels, heads), dim=-1)
        mixed, output = buffer("mixed", channels), buffer("output", config.hidden)
        recurrence, gated = buffer("recurrence", width), buffer("gated", width)
        self.input = QuantizedProjection(source, weights["input"], projected, session=session,
            name=name + ".input", activation_mode="auto", workspace=workspace,
            examples=examples, example_role="nemotron.target.mamba.input")
        self.output = QuantizedProjection(gated, weights["output"], output, session=session,
            name=name + ".output", activation_mode="auto", workspace=workspace,
            examples=examples, example_role="nemotron.target.mamba.output")
        sequences = metadata.query_start_loc.numel() - 1
        args = dict(x=xbc, conv_weight=weights["convolution"], conv_bias=weights["convolution_bias"],
            query_start_loc=metadata.query_start_loc, state_slot_ids=metadata.convolution_slots,
            state_is_fresh=metadata.fresh, num_accepted_tokens=metadata.accepted_tokens,
            num_seqs=metadata.num_seqs, num_tokens=metadata.num_tokens,
            conv_state=state.convolution, out=mixed)
        conv_mode = "decode" if mode == "verify" else mode
        caps = short_conv.Caps(device=source.device, mode=conv_mode, channels=channels, kernel_size=c["conv_kernel"],
            max_tokens=tokens, max_seqs=sequences, max_state_slots=state.convolution.shape[0],
            max_speculative_tokens=state.convolution.shape[-1] - c["conv_kernel"] + 1, round_before_silu=False)
        self.convolution_plan = short_conv.plan(caps, invocation=short_conv.invocation_from_tensors(**args))
        prepare_sequence_binding(self.convolution_plan, args, session=session, name=name + ".convolution",
            mode=conv_mode, public_bind=short_conv.bind, bind_plan=True, workspace=workspace,
            set_binding=lambda binding: None)
        x, B, C = mixed.split((width, groups * n, groups * n), dim=-1)
        delta = _metadata(metadata)
        recurrence_args = dict(x=x.view(tokens, heads, dim), B=B.view(tokens, groups, n), C=C.view(tokens, groups, n),
            dt=dt, A=weights["A"], D=weights["D"], dt_bias=weights["dt_bias"], recurrent_state=state.recurrent,
            cu_seqlens=metadata.query_start_loc, initial_state_indices=delta["initial_state_indices"],
            final_state_indices=delta["final_state_indices"], state_is_fresh=delta["fresh"],
            num_seqs=delta["num_seqs"], num_tokens=delta["num_tokens"], out=recurrence.view(tokens, heads, dim))
        self.mode = mode
        if mode == "verify":
            if replay is None or records is None:
                raise ValueError("Mamba verification requires replay state and continuation records")
            ring_start = torch.empty((replay.backing.shape[0],), dtype=torch.int32, device=source.device)
            pending_count = torch.empty_like(ring_start)
            self.recurrence = PreparedMambaVerification(recurrence_args, records, ring_start, pending_count,
                delta["final_state_indices"], width=verification_width, session=session,
                workspace=workspace, name=name + ".verification")
        elif mode == "decode":
            self.recurrence = PreparedMambaUpdate(recurrence_args, session=session, name=name + ".state_update")
        else:
            self.recurrence = PreparedMambaScan(recurrence_args, session=session, name=name + ".scan", workspace=workspace)
        self.norm = PreparedGatedRMSNorm(recurrence.view(tokens, groups, width // groups),
            gate.view(tokens, groups, width // groups), weights["output_norm"].view(groups, width // groups),
            gated.view(tokens, groups, width // groups), metadata.num_tokens, eps=c["layer_norm_epsilon"],
            activation="silu", norm_before_gate=False, session=session, name=name + ".gated_norm")
        self.layer_weights = {key: weights[key] for key in ("convolution", "convolution_bias", "A", "D", "dt_bias")}
        self.heads, self.dim, self.groups, self.dstate = heads, dim, groups, n
        self.width, self.channels = width, channels

    def functional(self, *, session):
        self.input.functional(session=session)
        self.output.functional(session=session)
        self.norm.functional(session=session, dynamic_live=True)
        return self

    def __call__(self, source, metadata, state, *, records=None, replay_state=None):
        projection = self.input(source)
        tokens = projection.shape[0]
        c = self.layer_weights
        width, channels, heads, dim = self.width, self.channels, self.heads, self.dim
        xbc_width = channels
        gate, xbc, dt = projection.split((width, xbc_width, heads), dim=-1)
        mixed = torch.empty((tokens, channels), dtype=projection.dtype, device=projection.device)
        spec, = self.convolution_plan.scratch_specs()
        scratch = torch.empty(spec.nbytes, dtype=torch.uint8, device=projection.device)
        delta = _metadata(metadata)
        run_short_conv(self.convolution_plan, scratch=scratch, x=xbc,
            conv_weight=c["convolution"], conv_bias=c["convolution_bias"],
            query_start_loc=delta["query_start_loc"], state_slot_ids=delta["convolution_slots"],
            state_is_fresh=delta["fresh"], num_accepted_tokens=delta["accepted_tokens"],
            num_seqs=delta["num_seqs"], num_tokens=delta["num_tokens"], conv_state=state.convolution,
            out=mixed)
        groups, dstate = self.groups, self.dstate
        x, B, C = mixed.split((width, groups * dstate, groups * dstate), dim=-1)
        recurrence = torch.empty((tokens, heads, dim), dtype=projection.dtype, device=projection.device)
        recurrence_args = dict(x=x.view(tokens, heads, dim), B=B.view(tokens, groups, dstate),
            C=C.view(tokens, groups, dstate), dt=dt, A=c["A"], D=c["D"], dt_bias=c["dt_bias"],
            recurrent_state=state.recurrent, cu_seqlens=metadata.query_start_loc,
            initial_state_indices=delta["initial_state_indices"], final_state_indices=delta["final_state_indices"],
            state_is_fresh=delta["fresh"], num_seqs=delta["num_seqs"], num_tokens=delta["num_tokens"],
            out=recurrence)
        if self.mode == "verify":
            if records is None or replay_state is None:
                raise ValueError("Mamba verification requires replay records and round-local metadata")
            recurrent = self.recurrence(recurrence_args, records, replay_state[0], replay_state[1],
                                        delta["final_state_indices"])
        else:
            recurrent = self.recurrence(recurrence_args)
        norm_shape = (tokens, self.groups, width // self.groups)
        gated = self.norm(recurrent.view(norm_shape), gate.view(norm_shape), live=metadata.num_tokens)
        return self.output(gated.flatten(1))
