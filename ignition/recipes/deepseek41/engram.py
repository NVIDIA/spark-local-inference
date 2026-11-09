# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""File-backed n-gram tables (FP8 or 4-bit trellis), compact GPU inputs, and residual injection."""

from dataclasses import dataclass
import json
from pathlib import Path
import struct

import torch
from b12x.norm import hyperconnection
from b12x.preparation import PreparedCall
from b12x.sequence import engram

from ignition.components import engram_trellis
from ignition.components.engram import EngramInputs, PreparedEngramLookup, PreparedEngramRequests
from ignition.components.readiness import ReadinessTimings
from ignition.components.linear import QuantizedProjection
from ignition.components.collectives import PreparedChunkedAllReduce
from ignition.components.state import gather_rows, save_hidden
from ignition.preparation.runtime_package import FileSource
from ignition.preparation.tuning import declare
from ignition.progress import phase


def _tensor_span(path, name):
    """Byte span, shape and dtype of one tensor in a single-tensor safetensors file."""
    with open(path, "rb") as source:
        header_bytes = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_bytes))
    record = header[name]
    start, end = record["data_offsets"]
    return 8 + header_bytes + start, end - start, tuple(record["shape"]), record["dtype"]


@dataclass(frozen=True)
class Tables:
    geometry: object
    token_map: tuple[int, ...]
    sources: dict
    inputs: dict
    mailboxes: dict
    sequence: torch.Tensor
    shard: object
    columns: torch.Tensor
    readiness_runtime: object = None
    trellis: bool = False

    @classmethod
    def prepare(cls, manifest, config, *, tokens, device, sequence, frame, host_allocator, progress=None,
                shard=None, root=None):
        from transformers import PreTrainedTokenizerFast
        from ignition.components.parallel import TensorParallelShard

        shard = shard or TensorParallelShard(0, 1)

        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(manifest.metadata_path / "tokenizer.json"))
        token_map, size = engram.build_compressed_token_map(tokenizer)
        text = config.text
        if size != text["engram_compressed_vocab_size"]:
            raise ValueError(f"Engram compressed vocabulary differs: {size}")
        geometry = engram.build_geometry(layer_ids=tuple(text["engram_layer_ids"]),
            base_table_size=text["engram_vocab_size"], compressed_vocab_size=size)
        if geometry.num_embeddings != tuple(text["engram_num_embeddings"]):
            raise ValueError("Engram hash geometry differs from checkpoint tables")
        # A multi-host peer stages indexed tensor ranges into a transfer cache;
        # external Engram files live in the checkpoint directory.
        root = Path(root) if root is not None else manifest.snapshot
        if not (root / "engram").is_dir():
            root = manifest.snapshot
        sources, inputs, mailboxes = {}, {}, {}
        timings = ReadinessTimings.allocate(frame, len(geometry.layer_ids))
        with phase(progress, "Prepare embedding tables") as stage:
            for index, layer in enumerate(geometry.layer_ids):
                rows = geometry.num_embeddings[index]
                if config.trellismx:
                    trellis_path = root / f"engram/layer-{layer:03d}.trellis.safetensors"
                    scale_path = root / f"engram/layer-{layer:03d}.scale.safetensors"
                    pair = []
                    for path, suffix, columns, dtype, width in (
                            (trellis_path, "trellis", 128, "U8", 128),
                            (scale_path, "scale", 8, "F8_E8M0", 8)):
                        name = f"layers.{layer}.engram.embed.{suffix}"
                        offset, nbytes, shape, actual = _tensor_span(path, name)
                        if shape != (rows, columns) or actual != dtype:
                            raise ValueError(f"{name}: invalid trellis Engram geometry or dtype")
                        pair.append(FileSource(path, offset, rows * width))
                    sources[layer] = tuple(pair)
                    inputs[layer] = EngramInputs.allocate(tokens, device, timings=timings, interval=index,
                                                          row_width=128)
                else:
                    pair = []
                    for suffix, columns, dtype in (("weight", 256, "F8_E4M3"), ("scale", 8, "F8_E8M0")):
                        name = f"layers.{layer}.engram.embed.{suffix}"
                        metadata = manifest.tensors[name]
                        if metadata.shape != (rows, columns) or metadata.dtype != dtype:
                            raise ValueError(f"{name}: invalid Engram table geometry or dtype")
                        pair.append(FileSource(manifest.snapshot / metadata.shard, metadata.offset, rows * columns))
                    sources[layer] = tuple(pair)
                    inputs[layer] = EngramInputs.allocate(tokens, device, timings=timings, interval=index)
                mailboxes[layer] = host_allocator(f"engram_{layer}_requests", (tokens * 24 + 1,), torch.int64)
                mailboxes[layer].host_view.zero_()
            stage.update(f"{sum(source.nbytes for pair in sources.values() for source in pair) / (1 << 30):.1f} GiB in checkpoint spans")
        columns = torch.tensor([head * 256 + column for head in range(shard.rank, 24, shard.world_size)
                                for column in range(256)], dtype=torch.int64, device=device)
        return cls(geometry, tuple(token_map), sources, inputs, mailboxes, sequence, shard, columns,
                   trellis=config.trellismx)

    def buffers(self):
        return {**{name: tensor for layer, inputs in self.inputs.items() for name, tensor in inputs.buffers(layer).items()},
                **{f"engram_{layer}_requests": mailbox.device_view for layer, mailbox in self.mailboxes.items()},
                **next(iter(self.inputs.values())).timings.buffers()}

    def events(self):
        return {f"engram_{layer}_ready": inputs.ready.cuda_event for layer, inputs in self.inputs.items()}

    def resident_catalog(self):
        """Source spans and native bindings for shared RAM delivery without package export."""
        return {"vocabulary": len(self.token_map), "tables": [
            {"planes": [{"segments": [
                            {"path": str(source.path), "offset": source.offset, "nbytes": source.nbytes}],
                         "row_bytes": row_bytes}
                        for source, row_bytes in zip(self.sources[layer],
                                                      (128, 8) if self.trellis else (256, 8), strict=True)],
             "heads": 24, "weights": f"engram_{layer}_weights", "scales": f"engram_{layer}_scales",
             "hashes": f"engram_{layer}_hashes", "event": f"engram_{layer}_ready",
             "request": f"engram_{layer}_requests"}
            for layer in self.geometry.layer_ids]}

    def export_metadata(self):
        """Describe table geometry and checkpoint spans for native publication."""
        tables = []
        for ordinal, layer in enumerate(self.geometry.layer_ids):
            table = dict(layer=layer, rows=self.geometry.num_embeddings[ordinal],
                embedding_width=256, scale_block=32,
                encoding="TRELLIS_K4_E4M3_E8M0" if self.trellis else "FP8_E4M3_E8M0",
                weight_buffer=f"engram_{layer}_weights", scale_buffer=f"engram_{layer}_scales",
                hash_buffer=f"engram_{layer}_hashes", ready_event=f"engram_{layer}_ready",
                request_buffer=f"engram_{layer}_requests")
            for name, source in zip(("weights", "scales"), self.sources[layer], strict=True):
                table[name] = [dict(path=str(source.path), offset=source.offset, nbytes=source.nbytes)]
            table["hash"] = dict(method="XOR_MODULO", orders=[order for order in (2, 3, 4) for _ in range(8)],
                multipliers=list(self.geometry.multipliers[ordinal]), primes=list(self.geometry.primes[ordinal]),
                offsets=list(self.geometry.offsets[ordinal]))
            tables.append(table)
        return dict(token_map=list(self.token_map), pad_token_id=2,
                    compressed_vocabulary=self.geometry.compressed_vocab_size, tables=tables)


class PreparedTableRequests:
    def __init__(self, tables, metadata, pool, *, session, name):
        self.metadata, self.pool = metadata, pool
        rows, device = metadata.starts.numel(), metadata.starts.device
        self.raw_history = torch.zeros((rows, 3), dtype=torch.int64, device=device)
        self.history = torch.zeros_like(self.raw_history)
        self.columns = torch.arange(3, device=device)
        self.token_map = torch.tensor(tables.token_map, dtype=torch.int64, device=device)
        self.operations = tuple(PreparedEngramRequests(inputs, metadata.token_ids, metadata.query_start_loc,
            self.history, metadata.live_counts, tables.mailboxes[layer].device_view, tables.sequence,
            layer=layer, token_map=tables.token_map, geometry=tables.geometry, session=session,
            name=f"{name}.{layer}") for layer, inputs in tables.inputs.items())

    def __call__(self):
        m = self.metadata
        gather_rows(self.pool.region("engram_history"), m.state_rows, self.raw_history)
        raw = torch.where(self.columns >= 3 - m.starts[:, None], self.raw_history, 2)
        # Image token types use IDs outside the text vocabulary. A dead marker
        # stops every n-gram at the image boundary, including across prefill chunks.
        text = (raw >= 0) & (raw < self.token_map.numel())
        self.history.copy_(torch.where(text, self.token_map[raw.clamp(0, self.token_map.numel() - 1)], -1))
        for operation in self.operations:
            operation()

    def commit(self, counts):
        m = self.metadata
        relative = counts[:, None].long() + self.columns - 3
        indices = (m.query_start_loc[:-1, None] + relative).clamp(0, m.token_ids.numel() - 1)
        history = torch.where(relative >= 0, m.token_ids[indices],
                              self.raw_history.gather(1, (relative + 3).clamp(0, 2)))
        save_hidden(history, self.pool.region("engram_history"), m.state_rows, counts > 0)


@dataclass(frozen=True)
class EngramScratch:
    """Intermediate lookup and projection storage for serialized Engram layers."""

    mask: torch.Tensor
    gathered: torch.Tensor
    local: torch.Tensor
    partial: torch.Tensor
    projected: torch.Tensor

    @classmethod
    def allocate(cls, tokens, device, *, buffers=None, world=1):
        def zeros(shape, dtype):
            return (torch.zeros(shape, dtype=dtype, device=device) if buffers is None
                    else buffers.zeros(shape, dtype))
        gathered = zeros((tokens, 6144), torch.bfloat16)
        local = gathered if world == 1 else zeros((tokens, 6144 // world), torch.bfloat16)
        partial = zeros((tokens, 25600), torch.bfloat16)
        projected = partial if world == 1 else zeros((tokens, 25600), torch.bfloat16)
        return cls(zeros(tokens, torch.bool), gathered, local, partial, projected)


class PreparedEngram:
    def __init__(self, layer, weights, tables, source, output, metadata, pool, *, scratch, session, name, workspace,
                 runtime=None, coordinator=None):
        self.source, self.output, self.metadata, self.pool = source, output, metadata, pool
        tokens, device = metadata.token_ids.numel(), source.device
        self.mask, self.gathered, self.projected = scratch.mask, scratch.gathered, scratch.projected
        self.local, self.partial = scratch.local, scratch.partial
        self.columns = tables.columns
        self.sharded = tables.shard.world_size > 1
        if self.sharded and runtime is None:
            raise ValueError("sharded Engram projection requires a collective runtime")
        for tensor, shape, dtype in ((self.mask, (tokens,), torch.bool),
                (self.gathered, (tokens, 6144), torch.bfloat16),
                (self.local, (tokens, 6144 // tables.shard.world_size), torch.bfloat16),
                (self.partial, (tokens, 25600), torch.bfloat16),
                (self.projected, (tokens, 25600), torch.bfloat16)):
            if tensor.shape != shape or tensor.dtype != dtype or tensor.device != device or not tensor.is_contiguous():
                raise ValueError("Engram scratch must match the contiguous lookup and projection geometry")
        ordinal = tables.geometry.layer_ids.index(layer)
        self.lookup = PreparedEngramLookup(tables.inputs[layer], self.gathered, metadata.live_counts[:1],
            layer=layer, token_map=tables.token_map, geometry=tables.geometry,
            session=session, name=name + ".lookup", runtime=tables.readiness_runtime, coordinator=coordinator,
            trellis=tables.trellis, table_rows=tables.geometry.num_embeddings[ordinal])
        self.projection = QuantizedProjection(self.local, weights["projection"], self.partial,
            session=session, name=name + ".projection", workspace=workspace, activation_mode="auto")
        self.reduce = (PreparedChunkedAllReduce(self.partial.view(-1, 5120), self.projected.view(-1, 5120), runtime=runtime,
            session=session, coordinator=coordinator, name=name + ".reduce") if self.sharded else None)
        self.norm_weight = weights["weight"].view(-1)
        self.eps = pool.config.text["rms_norm_eps"]
        self.mix_plan = hyperconnection.plan(hyperconnection.Caps(device=device, max_tokens=tokens, hidden_size=5120),
            invocation={"operation": "engram_mix", "eps": self.eps, "token_mask": True})
        declare(session, (self.mix_plan.request(name=name + ".mix", prepare_call=lambda state: PreparedCall(
            run=lambda: state.launch(source.view(tokens, -1), self.projected, self.norm_weight,
                                      self.mask, output.view(tokens, -1), eps=self.eps))),))

    def __call__(self, source):
        m = self.metadata
        self.mask.copy_((m.positions >= 0) & (m.token_ids < 129280))
        self.lookup(source)
        if self.sharded:
            self.local.copy_(torch.index_select(self.gathered, 1, self.columns))
        self.projection(self.local)
        if self.sharded:
            self.reduce(self.partial.view(-1, 5120))
        hyperconnection.run_engram_mix(source.view(source.shape[0], -1), self.projected, self.norm_weight,
            eps=self.eps, plan=self.mix_plan, out=self.output.view(source.shape[0], -1), token_mask=self.mask)
        return self.output
