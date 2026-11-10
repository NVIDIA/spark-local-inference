# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed target metadata with independent prompt and decode state updates."""

from dataclasses import dataclass, fields, replace

import torch

from ignition.components.gated_delta import GDNScratch
from ignition.components.recurrent import DeltaMetadata

from .attention import QSAMetadata
from .ple import PLEMetadata, PLEScratch


@dataclass
class MixedRuntime:
    gdn: object
    qsa: object
    ple: object
    packing: object
    prompt: object
    decode: object
    gdn_partitions: object
    ple_partitions: object
    injection_errors: object = None


def role_inputs(inputs, tokens):
    return replace(inputs,
        row_slot_ids=torch.full_like(inputs.row_slot_ids, -1),
        row_generations=torch.zeros_like(inputs.row_generations),
        live_rows=torch.zeros_like(inputs.live_rows),
        query_lengths=torch.zeros_like(inputs.query_lengths),
        token_ids=torch.full_like(inputs.token_ids[:tokens], 248044),
        prefill_positions=torch.zeros_like(inputs.prefill_positions[:tokens]),
        vision_token_indices=torch.full_like(inputs.vision_token_indices[:tokens], -1),
        sample_enabled=torch.zeros_like(inputs.sample_enabled))


@dataclass
class MixedScratch:
    prompt_gdn: GDNScratch
    decode_gdn: GDNScratch
    prompt_ple: PLEScratch
    decode_ple: PLEScratch
    prompt_hidden: torch.Tensor
    decode_hidden: torch.Tensor
    prompt_residual: torch.Tensor
    decode_residual: torch.Tensor

    @classmethod
    def allocate(cls, prompt, decode, device, *, world_size):
        return cls(GDNScratch.allocate(prompt, device, world_size=world_size),
            GDNScratch.allocate(decode, device, world_size=world_size),
            PLEScratch.allocate(prompt, device), PLEScratch.allocate(decode, device),
            *(torch.zeros((n, width), dtype=torch.bfloat16, device=device)
              for width in (2560, 10240) for n in (prompt, decode)))


class MixedPacking:
    """Map native dense query offsets to fixed-capacity stateful operator operands."""

    def __init__(self, inputs, prompt, decode, batch, width, prefill):
        self.inputs, self.prompt, self.decode = inputs, prompt, decode
        self.batch, self.width, self.prefill = batch, width, prefill
        self.tokens = prefill + (batch - 1) * width
        device = inputs.token_ids.device
        self.rows = torch.arange(batch, device=device)
        self.p = torch.arange(prefill, device=device)
        self.d = torch.arange(batch * width, device=device)
        self.t = torch.arange(self.tokens, device=device)
        self.prompt_indices = torch.zeros(prefill, dtype=torch.int64, device=device)
        self.decode_indices = torch.zeros(batch * width, dtype=torch.int64, device=device)
        self.prompt_valid = torch.zeros(prefill, dtype=torch.bool, device=device)
        self.decode_valid = torch.zeros(batch * width, dtype=torch.bool, device=device)
        self.prompt_source = torch.zeros(self.tokens, dtype=torch.int64, device=device)
        self.decode_source = torch.zeros(self.tokens, dtype=torch.int64, device=device)
        self.is_prompt = torch.zeros(self.tokens, dtype=torch.bool, device=device)
        self.row_ids = torch.zeros(self.tokens, dtype=torch.int64, device=device)
        self.cu = torch.zeros(batch + 1, dtype=torch.int32, device=device)

    def functional(self):
        self.prompt = self.decode = None
        self.rows = self.p = self.d = self.t = self.cu = None
        self.prompt_indices = self.decode_indices = self.prompt_valid = self.decode_valid = None
        self.prompt_source = self.decode_source = self.is_prompt = self.row_ids = None
        return self

    def runtime(self):
        prompt = role_inputs(self.inputs, self.prefill)
        decode = role_inputs(self.inputs, self.batch * self.width)
        packing = MixedPacking(self.inputs, prompt, decode, self.batch, self.width, self.prefill)
        packing.upload_roles()
        return packing

    def upload_roles(self):
        i, p, d = self.inputs, self.prompt, self.decode
        lengths = torch.where(self.rows < i.live_rows[0], i.query_lengths[:self.batch], 0)
        self.cu[0].zero_()
        self.cu[1:].copy_(lengths.cumsum(0))
        row = i.prefill_row.long().clamp(0, self.batch - 1)
        present = (i.prefill_row >= 0) & (i.prefill_row < i.live_rows) & (i.prefill_row < self.batch)
        p.row_slot_ids[:1].copy_(torch.where(present, i.row_slot_ids[row], -1))
        p.row_generations[:1].copy_(i.row_generations[row])
        p.query_lengths[:1].copy_(torch.where(present, lengths[row], 0))
        p.sample_enabled[:1].copy_(i.sample_enabled[row])
        p.live_rows.copy_(present.int())
        self.prompt_indices.copy_((self.cu[row] + self.p).clamp(0, self.tokens - 1))
        self.prompt_valid.copy_(present & (self.p < p.query_lengths[0]))
        p.token_ids.copy_(torch.where(self.prompt_valid, i.token_ids[self.prompt_indices], 248044))
        p.prefill_positions.copy_(i.prefill_positions[self.prompt_indices])
        p.vision_token_indices.copy_(torch.where(self.prompt_valid, i.vision_token_indices[self.prompt_indices], -1))
        d.row_slot_ids[:self.batch].copy_(torch.where(self.rows == i.prefill_row[0], -1, i.row_slot_ids[:self.batch]))
        d.row_generations[:self.batch].copy_(i.row_generations[:self.batch])
        d.query_lengths[:self.batch].copy_(torch.where(self.rows == i.prefill_row[0], 0, lengths))
        d.sample_enabled[:self.batch].copy_(i.sample_enabled[:self.batch])
        d.live_rows.copy_(i.live_rows)

    def bind_metadata(self, prompt, decode):
        drow = decode.qsa.request_ids.long().clamp(0, self.batch - 1)
        local = self.d - decode.gdn.query_start_loc[drow]
        self.decode_indices.copy_((self.cu[drow] + local).clamp(0, self.tokens - 1))
        self.decode_valid.copy_(decode.qsa.request_ids >= 0)
        self.prompt_valid.logical_and_(prompt.qsa.request_ids >= 0)
        row = ((self.t[:, None] >= self.cu[:-1]) & (self.t[:, None] < self.cu[1:])).long()
        self.row_ids.copy_((row * self.rows).sum(1))
        offset = self.t - self.cu[self.row_ids]
        self.is_prompt.copy_((self.row_ids == self.inputs.prefill_row[0]) & (self.t < self.cu[-1]))
        self.prompt_source.copy_(offset.clamp(0, self.prefill - 1))
        self.decode_source.copy_((decode.gdn.query_start_loc[self.row_ids] + offset).clamp(0, self.d.numel() - 1))

    def combine(self, values, output):
        prompt, decode = values
        shape = (self.tokens,) + (1,) * (output.ndim - 1)
        result = torch.where(self.is_prompt.view(shape), prompt[self.prompt_source], decode[self.decode_source])
        output.copy_(result)


class StatePartition:
    def __init__(self, packing, mode, metadata, scratch):
        self.packing, self.mode, self.metadata, self.scratch = packing, mode, metadata, scratch

    def gather(self, source, output):
        p = self.packing
        indices, valid = ((p.prompt_indices, p.prompt_valid) if self.mode == "prefill"
                          else (p.decode_indices, p.decode_valid))
        output.copy_(torch.where(valid.view((-1,) + (1,) * (output.ndim - 1)), source[indices], 0))


class StatePartitions:
    def __init__(self, packing, prompt, decode, prompt_scratch, decode_scratch):
        self.packing = packing
        self.parts = (StatePartition(packing, "prefill", prompt, prompt_scratch),
                      StatePartition(packing, "decode", decode, decode_scratch))

    def combine(self, values, output):
        self.packing.combine(values, output)

    def functional(self):
        p = self.packing
        parts = tuple(FunctionalStatePartition(part.mode, part.metadata, indices, valid)
                      for part, indices, valid in zip(self.parts,
                          (p.prompt_indices, p.decode_indices), (p.prompt_valid, p.decode_valid), strict=True))
        return FunctionalStatePartitions(parts, p.is_prompt, p.prompt_source, p.decode_source)


@dataclass(frozen=True)
class FunctionalStatePartition:
    """Gather one sequence role through persistent row maps into graph-local storage."""

    mode: str
    metadata: object
    indices: torch.Tensor
    valid: torch.Tensor

    def gather(self, source):
        return torch.where(self.valid.view((-1,) + (1,) * (source.ndim - 1)), source[self.indices], 0)


@dataclass(frozen=True)
class FunctionalStatePartitions:
    """Combine prompt and decode results through persistent token-role maps."""

    parts: tuple
    is_prompt: torch.Tensor
    prompt_source: torch.Tensor
    decode_source: torch.Tensor

    def combine(self, values):
        prompt, decode = values
        shape = (self.is_prompt.numel(),) + (1,) * (prompt.ndim - 1)
        return torch.where(self.is_prompt.view(shape), prompt[self.prompt_source], decode[self.decode_source])


class PreparedMixedState:
    """One target forward; prompt consumption and decode acceptance remain separate."""

    def __init__(self, packing, prompt, decode, scratch, *, prompt_residual, decode_residual):
        self.packing, self.prompt, self.decode = packing, prompt, decode
        self.inputs, self.requests, self.pool = packing.inputs, prompt.requests, prompt.pool
        self.tokens, self.rows, self.mode = packing.tokens, packing.batch, "prefill"
        device = self.inputs.token_ids.device
        self.gdn = DeltaMetadata.allocate(self.rows, device)
        self.qsa = QSAMetadata.allocate(tokens=self.tokens, batch=self.rows,
            context=self.inputs.page_tables.shape[1] * 64, device=device)
        self.qsa = replace(self.qsa, compressed_block_table=self.qsa.main_block_table)
        self.token_ids = torch.full((self.tokens,), 248044, dtype=torch.int64, device=device)
        self.ple = PLEMetadata(self.token_ids, self.gdn.query_start_loc,
            torch.zeros((self.rows, 2), dtype=torch.int64, device=device),
            self.gdn.convolution_slots, self.gdn.fresh, self.gdn.accepted_tokens,
            self.gdn.num_seqs, self.gdn.num_tokens)
        self.gdn_partitions = StatePartitions(packing, prompt.gdn, decode.gdn,
            scratch.prompt_gdn.rows(prompt.tokens), scratch.decode_gdn.rows(decode.tokens))
        def ple_rows(s, n):
            return PLEScratch(*(getattr(s, f.name)[:n] for f in fields(s)))
        self.ple_partitions = StatePartitions(packing, prompt.ple, decode.ple,
            ple_rows(scratch.prompt_ple, prompt.tokens), ple_rows(scratch.decode_ple, decode.tokens))
        self.prompt_hidden = scratch.prompt_hidden[:prompt.tokens]
        self.decode_hidden = scratch.decode_hidden[:decode.tokens]
        self.prompt_residual, self.decode_residual = prompt_residual, decode_residual

    def prepare_hidden(self, hidden, sample):
        self.hidden = hidden
        self.decode.prepare_hidden(self.decode_hidden, sample[:-1])
        self.prompt.prepare_hidden(self.prompt_hidden, sample[-1:])

    def prepare_residual(self, residual):
        self.residual = residual
        self.prompt.prepare_residual(self.prompt_residual)
        self.decode.prepare_residual(self.decode_residual)

    def functional(self):
        self.prompt.functional(dynamic_inputs=True)
        self.decode.functional(dynamic_inputs=True)
        self.gdn = self.qsa = self.ple = self.token_ids = None
        self.gdn_partitions = self.ple_partitions = None
        self.prompt_hidden = self.decode_hidden = self.prompt_residual = self.decode_residual = None
        self.hidden = self.residual = None
        return self

    def combine_runtime(self, packing, prompt, decode):
        m, p, d = packing, prompt, decode
        m.bind_metadata(p, d)
        token_ids = torch.empty((self.tokens,), dtype=torch.int64, device=self.inputs.token_ids.device)
        m.combine((p.ple.token_ids, d.ple.token_ids), token_ids)
        live = m.t < m.cu[-1]
        token_ids.masked_fill_(~live, 248044)
        prompt_rows = m.rows == self.inputs.prefill_row[0]
        def roles(name):
            return torch.where(prompt_rows, getattr(p.gdn, name)[0], getattr(d.gdn, name))
        gdn = DeltaMetadata.allocate(self.rows, token_ids.device)
        gdn = replace(gdn, query_start_loc=m.cu, num_tokens=m.cu[-1:], num_seqs=self.inputs.live_rows,
                      convolution_slots=roles("convolution_slots"), fresh=roles("fresh"),
                      accepted_tokens=roles("accepted_tokens"))
        history = torch.where(prompt_rows[:, None], p.ple.committed_history, d.ple.committed_history)
        ple = PLEMetadata(token_ids, gdn.query_start_loc, history, gdn.convolution_slots,
                          gdn.fresh, gdn.accepted_tokens, gdn.num_seqs, gdn.num_tokens)
        def qsa_rows(name):
            pv, dv = getattr(p.qsa, name), getattr(d.qsa, name)
            return torch.where(prompt_rows.view((-1,) + (1,) * (dv.ndim - 1)), pv[0], dv)
        def qsa_tokens(name):
            pv, dv = getattr(p.qsa, name), getattr(d.qsa, name)
            output = torch.empty((self.tokens, *pv.shape[1:]), dtype=pv.dtype, device=pv.device)
            m.combine((pv, dv), output)
            return output
        request_ids = qsa_tokens("request_ids")
        active = torch.where(prompt_rows, p.values["active"][0], d.values["active"]).bool()
        valid = live & active[m.row_ids] & (request_ids >= 0)
        request_ids = torch.where(valid, m.row_ids, -1)
        positions = qsa_tokens("query_positions").masked_fill(~valid, -1)
        rope = qsa_tokens("rope_positions").masked_fill(~valid[:, None], -1)
        mapping = qsa_tokens("slot_mapping").masked_fill(~valid, -1)
        table = qsa_rows("main_block_table")
        qsa = QSAMetadata(table, table, qsa_rows("raw_state_slot_ids"), request_ids, positions, rope,
            qsa_rows("sequence_lengths"), m.cu, qsa_rows("num_accepted_tokens"),
            qsa_rows("is_prefilling"), mapping)
        def partitions(prompt_metadata, decode_metadata):
            parts = (FunctionalStatePartition("prefill", prompt_metadata, m.prompt_indices, m.prompt_valid),
                     FunctionalStatePartition("decode", decode_metadata, m.decode_indices, m.decode_valid))
            return FunctionalStatePartitions(parts, m.is_prompt, m.prompt_source, m.decode_source)
        return MixedRuntime(gdn, qsa, ple, m, p, d,
            partitions(p.gdn, d.gdn), partitions(p.ple, d.ple))

    def finish_runtime(self, runtime, hidden, residual):
        prompt, decode = runtime.gdn_partitions.parts
        prompt_hidden, decode_hidden = prompt.gather(hidden), decode.gather(hidden)
        prompt_residual, decode_residual = prompt.gather(residual), decode.gather(residual)
        prompt_sample = self.prompt.finish_runtime(runtime.prompt, prompt_hidden, prompt_residual)
        decode_sample = self.decode.finish_runtime(runtime.decode, decode_hidden, decode_residual)
        return torch.cat((decode_sample, prompt_sample), dim=0)
