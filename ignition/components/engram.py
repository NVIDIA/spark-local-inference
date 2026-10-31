# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compact b12x Engram lookup behind a captured external readiness event."""

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from b12x.preparation import PreparedCall
from b12x.sequence import engram
from b12x.sequence.engram._kernels import hash_op, lookup_op

from ignition.components import engram_trellis
from ignition.components.ngram_delivery import publish_requests
from ignition.components.readiness import PreparedReadinessWait, ReadinessTimings, wait_for_readiness

from ignition.preparation.tuning import declare


@torch.library.custom_op("flare::staged_engram", mutates_args=("out", "timings", "barrier_output"))
def _lookup(source: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor,
            hashes: torch.Tensor, count: torch.Tensor, out: torch.Tensor,
            timings: torch.Tensor, frame: torch.Tensor, barrier_input: torch.Tensor,
            barrier_output: torch.Tensor, interval: int, barrier: int, plan: int, event: int) -> None:
    wait_for_readiness(source, event, timings, frame, interval, barrier_input, barrier_output, barrier)
    lookup_op(plan, weight.view(torch.float8_e4m3fn), scales, hashes, count, out, hashes.shape[0], True)


@_lookup.register_fake
def _lookup_fake(source, weight, scales, hashes, count, out, timings, frame,
                 barrier_input, barrier_output, interval, barrier, plan, event):
    pass


@torch.library.custom_op("flare::staged_engram_trellis", mutates_args=("out", "timings", "barrier_output"))
def _lookup_trellis(source: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor,
                    hashes: torch.Tensor, count: torch.Tensor, out: torch.Tensor,
                    lut: torch.Tensor, permutation: torch.Tensor,
                    timings: torch.Tensor, frame: torch.Tensor, barrier_input: torch.Tensor,
                    barrier_output: torch.Tensor, interval: int, barrier: int, rows: int, event: int) -> None:
    wait_for_readiness(source, event, timings, frame, interval, barrier_input, barrier_output, barrier)
    engram_trellis.lookup_k4(weight, scales, hashes, count, out, shard_start=0, shard_end=rows,
                             rows=rows, tables={"lut": lut, "perm": permutation},
                             prepared_tokens=hashes.shape[0], compact_rows=True)


@_lookup_trellis.register_fake
def _lookup_trellis_fake(source, weight, scales, hashes, count, out, lut, permutation,
                         timings, frame, barrier_input, barrier_output, interval, barrier, rows, event):
    pass


@dataclass(frozen=True)
class EngramInputs:
    weights: torch.Tensor
    scales: torch.Tensor
    hashes: torch.Tensor
    ready: object
    timings: ReadinessTimings
    interval: int

    @classmethod
    def allocate(cls, tokens, device, *, timings, interval=0, row_width=256):
        ready = torch.cuda.Event(external=True)
        ready.record(torch.cuda.current_stream(device))
        return cls(torch.zeros((tokens * 24, row_width), dtype=torch.uint8, device=device),
                   torch.zeros((tokens * 24, 8), dtype=torch.uint8, device=device),
                   torch.full((tokens, 24), -1, dtype=torch.int64, device=device), ready, timings, interval)

    def buffers(self, layer):
        return {f"engram_{layer}_{name}": getattr(self, name) for name in ("weights", "scales", "hashes")}


class PreparedEngramLookup:
    def __init__(self, inputs, output, count, *, layer, token_map, geometry, session, name,
                 runtime=None, coordinator=None, trellis=False, table_rows=0):
        tokens = output.shape[0]
        self.weights = inputs.weights[:tokens * 24]
        self.scales = inputs.scales[:tokens * 24]
        self.hashes = inputs.hashes[:tokens]
        self.count, self.output = count, output
        self.handle = inputs.ready.cuda_event
        self.wait = PreparedReadinessWait(inputs.timings, inputs.interval, runtime=runtime, session=session,
                                         coordinator=coordinator, name=name + ".readiness")
        self.trellis, self.table_rows = trellis, int(table_rows)
        if self.trellis:
            if self.weights.shape[1] != engram_trellis.ROW_BYTES or table_rows <= 0:
                raise ValueError("Trellis Engram rows need 128-byte staging and a global row count")
            tables = engram_trellis.engram_tables(output.device)
            self.lut, self.permutation = tables["lut"], tables["perm"]
            return
        self.plan = engram.plan(engram.Caps(device=output.device, max_tokens=tokens,
            max_seqs=tokens, max_requests=tokens, vocab_size=len(token_map), layer_id=layer,
            tp_size=1, tp_rank=0), token_map=token_map, geometry=geometry,
            invocation={"operation": "lookup", "compact_rows": True})
        binding = SimpleNamespace(weight=self.weights.view(torch.float8_e4m3fn),
            scale_bytes=self.scales, hash_ids=self.hashes, num_tokens=count, out=output)
        declare(session, (self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
            run=lambda: state.run_lookup(binding, tokens, clear_tail=True), owners=(inputs, binding))),))

    def __call__(self, source):
        if self.trellis:
            _lookup_trellis(source, self.weights, self.scales, self.hashes, self.count,
                            self.output, self.lut, self.permutation,
                            self.wait.timings.timestamps, self.wait.timings.frame,
                            self.wait.input, self.wait.output, self.wait.interval,
                            self.wait.handle, self.table_rows, self.handle)
            return self.output
        _lookup(source, self.weights, self.scales, self.hashes, self.count,
                self.output, self.wait.timings.timestamps, self.wait.timings.frame,
                self.wait.input, self.wait.output, self.wait.interval, self.wait.handle,
                self.plan.handle, self.handle)
        return self.output


class PreparedEngramRequests:
    """Hash packed token rows on the GPU and release their mapped delivery mailbox."""

    def __init__(self, inputs, token_ids, starts, history, counts, mailbox, sequence, *,
                 layer, token_map, geometry, session, name, row_mapping=None):
        tokens, rows = token_ids.numel(), history.shape[0]
        device = token_ids.device
        self.mask = torch.ones(tokens, dtype=torch.bool, device=device)
        self.slots = torch.arange(rows, dtype=torch.int32, device=device)
        self.compressed = torch.empty(tokens, dtype=torch.int64, device=device)
        self.requests = torch.empty(tokens, dtype=torch.int32, device=device)
        self.hashes = inputs.hashes[:tokens]
        self.mailbox, self.sequence = mailbox, sequence
        if row_mapping is not None and (row_mapping.shape != (self.hashes.shape[1], 3)
                or row_mapping.dtype != torch.int64 or row_mapping.device != device):
            raise ValueError("embedding row mapping must have int64 source bounds and a destination per head")
        self.row_mapping = row_mapping
        self.args = (token_ids, self.mask, starts, self.slots, history, counts[1:], counts[:1],
                     self.hashes, self.compressed, self.requests, tokens)
        self.plan = engram.plan(engram.Caps(device=device, max_tokens=tokens, max_seqs=rows,
            max_requests=rows, vocab_size=len(token_map), layer_id=layer, tp_size=1, tp_rank=0),
            token_map=token_map, geometry=geometry, invocation={"operation": "hash"})
        binding = SimpleNamespace(**dict(zip(("token_ids", "token_mask", "query_start_loc", "request_slots",
            "committed_history", "num_seqs", "num_tokens", "hash_ids", "compressed", "request_ids"), self.args[:-1])))
        declare(session, (self.plan.request(name=name, prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(binding, tokens), owners=(binding,))),))

    def __call__(self):
        hash_op(self.plan.handle, *self.args)
        if self.row_mapping is not None:
            # Preserve canonical hash geometry; remap only the storage rows of owned heads.
            start, end, destination = self.row_mapping.unbind(1)
            valid = (destination >= 0) & (self.hashes >= start) & (self.hashes < end)
            self.hashes.copy_(torch.where(valid, self.hashes - start + destination, -1))
        publish_requests(self.hashes, self.mailbox, self.sequence)
