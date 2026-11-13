# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw generation logprobs from vocabulary shards and bounded statistic exchanges."""

import torch
import triton
import triton.language as tl

from ignition.preparation.triton import launch_triton, prepare_triton
from ignition.preparation.persistent import persistent_copy

from .collectives import PreparedAllGather
from .completion import (
    LOGPROB_FIELDS, LOGPROB_PAIR_FIELDS, MAX_LOGPROB_TOKEN_IDS,
    MAX_TOP_LOGPROBS, validate_completion_buffers,
)
from .logprobs import _position, _sort_key, _sort_pair

_TILE = 1024
_CANDIDATES = 32
_SUMMARY = 256  # FP32 words; the first 64 words also carry 32 uint64 sort keys.
_CHOSEN = 64
_REQUESTED = 65


@triton.jit
def _statistics(Logits, Records, Frame, Requested, Slots, Epochs, Live, Stats,
                LOCAL: tl.constexpr, V: tl.constexpr, Offset,
                WIDTH: tl.constexpr, TILES: tl.constexpr, SLOTS: tl.constexpr,
                FIELDS: tl.constexpr, BLOCK: tl.constexpr):
    row, col, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offset = tl.load(Offset)
    _, _, valid = _position(Records, Frame, Requested, Slots, Epochs, Live, row, col, SLOTS, FIELDS)
    if valid:
        ids = tile * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(Logits + (row * WIDTH + col) * LOCAL + ids,
                    (ids < LOCAL) & (ids + offset < V), other=-float("inf")).to(tl.float32)
        maximum = tl.max(x, 0)
        mass = tl.sum(tl.exp(x - tl.where(maximum == -float("inf"), 0., maximum)), 0)
        index = (row * WIDTH + col) * TILES + tile
        tl.store(Stats + index * 2, maximum)
        tl.store(Stats + index * 2 + 1, mass)


@triton.jit
def _local_normalization(Records, Frame, Requested, Slots, Epochs, Live, Stats, Summary,
                         WIDTH: tl.constexpr, TILES: tl.constexpr, SLOTS: tl.constexpr,
                         FIELDS: tl.constexpr, BLOCK: tl.constexpr, WORDS: tl.constexpr):
    row, col = tl.program_id(0), tl.program_id(1)
    _, _, valid = _position(Records, Frame, Requested, Slots, Epochs, Live, row, col, SLOTS, FIELDS)
    maximum = -float("inf")
    mass = 0.
    if valid:
        tiles = tl.arange(0, BLOCK)
        base = ((row * WIDTH + col) * TILES + tiles) * 2
        maxima = tl.load(Stats + base, tiles < TILES, other=-float("inf"))
        masses = tl.load(Stats + base + 1, tiles < TILES, other=0.)
        maximum = tl.max(maxima, 0)
        mass = tl.sum(masses * tl.exp(maxima - tl.where(maximum == -float("inf"), 0., maximum)), 0)
    tl.store(Summary + (row * WIDTH + col) * WORDS, maximum)
    tl.store(Summary + (row * WIDTH + col) * WORDS + 1, mass)


@triton.jit
def _global_normalization(Records, Frame, Requested, Slots, Epochs, Live, Gathered, Lse,
                          WIDTH: tl.constexpr, SLOTS: tl.constexpr, FIELDS: tl.constexpr,
                          WORLD: tl.constexpr, BLOCK: tl.constexpr, WORDS: tl.constexpr):
    row, col = tl.program_id(0), tl.program_id(1)
    _, _, valid = _position(Records, Frame, Requested, Slots, Epochs, Live, row, col, SLOTS, FIELDS)
    if valid:
        ranks = tl.arange(0, BLOCK)
        base = ((row * WIDTH + col) * WORLD + ranks) * WORDS
        maxima = tl.load(Gathered + base, ranks < WORLD, other=-float("inf"))
        masses = tl.load(Gathered + base + 1, ranks < WORLD, other=0.)
        maximum = tl.max(maxima, 0)
        lse = maximum + tl.log(tl.sum(masses * tl.exp(maxima - maximum), 0))
        tl.store(Lse + row * WIDTH + col, lse)


@triton.jit
def _candidates(Logits, Records, Frame, Requested, Slots, Epochs, Live, Counts, Lse, Keys,
                LOCAL: tl.constexpr, V: tl.constexpr, Offset,
                WIDTH: tl.constexpr, TILES: tl.constexpr, SLOTS: tl.constexpr,
                FIELDS: tl.constexpr, BLOCK: tl.constexpr, CANDIDATES: tl.constexpr):
    row, col, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offset = tl.load(Offset)
    slot, _, valid = _position(Records, Frame, Requested, Slots, Epochs, Live, row, col, SLOTS, FIELDS)
    if valid:
        if tl.load(Counts + slot) == 0:
            ids = tile * BLOCK + tl.arange(0, BLOCK)
            inside = (ids < LOCAL) & (ids + offset < V)
            x = tl.load(Logits + (row * WIDTH + col) * LOCAL + ids, inside, other=-float("inf")).to(tl.float32)
            x -= tl.load(Lse + row * WIDTH + col)
            # Sort the reported FP32 values, including ties created by subtraction.
            keys = tl.sort(tl.where(inside, _sort_key(x, ids + offset), 0), descending=True)
            index = (row * WIDTH + col) * TILES + tile
            k = tl.arange(0, BLOCK)
            tl.store(Keys + index * CANDIDATES + k, keys, k < CANDIDATES)


@triton.jit
def _pack(Logits, Records, Frame, Requested, Slots, Epochs, Live, Counts, TokenIds, Keys, Summary,
          LOCAL: tl.constexpr, V: tl.constexpr, Offset, WIDTH: tl.constexpr,
          TILES: tl.constexpr, SLOTS: tl.constexpr, FIELDS: tl.constexpr,
          BLOCK: tl.constexpr, CANDIDATES: tl.constexpr, IDS: tl.constexpr,
          WORDS: tl.constexpr, CHOSEN: tl.constexpr, REQUESTED: tl.constexpr):
    row, col = tl.program_id(0), tl.program_id(1)
    offset = tl.load(Offset)
    slot, token, valid = _position(Records, Frame, Requested, Slots, Epochs, Live, row, col, SLOTS, FIELDS)
    if valid:
        index = row * WIDTH + col
        owned = (token >= offset) & (token < offset + LOCAL) & (token < V)
        chosen = tl.load(Logits + index * LOCAL + token - offset, owned, other=0.).to(tl.float32)
        tl.store(Summary + index * WORDS + CHOSEN, chosen)
        count = tl.load(Counts + slot)
        if count > 0:
            requested_lane = tl.arange(0, IDS)
            ids = tl.load(TokenIds + slot * IDS + requested_lane, requested_lane < count, other=-1)
            requested_owned = ((requested_lane < count) & (ids >= offset)
                               & (ids < offset + LOCAL) & (ids < V))
            values = tl.load(Logits + index * LOCAL + ids - offset, requested_owned, other=0.).to(tl.float32)
            tl.store(Summary + index * WORDS + REQUESTED + requested_lane, values, requested_lane < count)
        else:
            k = tl.arange(0, BLOCK)
            keys = tl.load(Keys + index * TILES * CANDIDATES + k, k < TILES * CANDIDATES, other=0)
            keys = tl.sort(keys, descending=True)
            packed = Summary.to(tl.pointer_type(tl.uint64)) + index * (WORDS // 2)
            tl.store(packed + k, keys, k < CANDIDATES)


@triton.jit
def _finish(Records, Frame, Requested, Slots, Epochs, Live, Counts, TokenIds, Gathered, Lse, Output,
            LOCAL: tl.constexpr, V: tl.constexpr, WIDTH: tl.constexpr, SLOTS: tl.constexpr,
            FIELDS: tl.constexpr, WORLD: tl.constexpr, WORDS: tl.constexpr,
            CANDIDATES: tl.constexpr, BLOCK: tl.constexpr, TOP: tl.constexpr, IDS: tl.constexpr,
            CHOSEN: tl.constexpr, REQUESTED: tl.constexpr, OUTPUT_TOKENS: tl.constexpr,
            OUTPUT_FIELDS: tl.constexpr, PAIR_FIELDS: tl.constexpr):
    row, col = tl.program_id(0), tl.program_id(1)
    slot, token, valid = _position(Records, Frame, Requested, Slots, Epochs, Live, row, col, SLOTS, FIELDS)
    if valid:
        index = row * WIDTH + col
        lse = tl.load(Lse + index)
        owner = token // LOCAL
        chosen = tl.load(Gathered + (index * WORLD + owner) * WORDS + CHOSEN)
        out = Output + ((tl.load(Frame) * SLOTS + slot) * OUTPUT_TOKENS + col) * OUTPUT_FIELDS
        tl.store(out, token.to(tl.int32))
        tl.store(out + 1, (chosen - lse).to(tl.int32, bitcast=True))
        count = tl.load(Counts + slot)
        if count > 0:
            requested_lane = tl.arange(0, IDS)
            queried_ids = tl.load(TokenIds + slot * IDS + requested_lane, requested_lane < count, other=-1)
            owners = queried_ids // LOCAL
            queried_values = tl.load(Gathered + (index * WORLD + owners) * WORDS + REQUESTED + requested_lane,
                (requested_lane < count) & (queried_ids >= 0) & (queried_ids < V), other=float("nan"))
            tl.store(out + (1 + requested_lane) * PAIR_FIELDS, queried_ids, requested_lane < count)
            tl.store(out + (1 + requested_lane) * PAIR_FIELDS + 1,
                     (queried_values - lse).to(tl.int32, bitcast=True), requested_lane < count)
        else:
            k = tl.arange(0, BLOCK)
            rank, local = k // CANDIDATES, k % CANDIDATES
            packed = Gathered.to(tl.pointer_type(tl.uint64))
            keys = tl.load(packed + (index * WORLD + rank) * (WORDS // 2) + local,
                           rank < WORLD, other=0)
            keys = tl.sort(keys, descending=True)
            ids, values = _sort_pair(keys)
            tl.store(out + (1 + k) * PAIR_FIELDS, ids, k < TOP)
            tl.store(out + (1 + k) * PAIR_FIELDS + 1, values.to(tl.int32, bitcast=True), k < TOP)


class PreparedVocabularyParallelLogprobs:
    """Exchange raw normalizers and requested scores while retaining vocabulary shards."""

    def __init__(self, logits, completion, slots, epochs, live, frame, *, vocabulary,
                 offset, runtime, session, coordinator, name):
        validate_completion_buffers(completion.buffers(), completion.records.shape[1])
        if logits.ndim == 2:
            logits = logits[:, None, :]
        if logits.ndim != 3 or not logits.is_contiguous() or logits.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("vocabulary-sharded raw logits require contiguous BF16 or FP32 rows")
        batch, width, local = logits.shape
        world = 1 if runtime is None else runtime.world_size
        if (min(batch, width, local, vocabulary) <= 0 or width > completion.token_capacity
                or offset < 0 or offset % local
                or offset >= world * local or vocabulary > world * local
                or (world - 1) * local >= vocabulary):
            raise ValueError("raw logprob shards require equal partitions with padding on the final rank")
        tiles = triton.cdiv(local, _TILE)
        self.offset = persistent_copy(torch.tensor([offset], dtype=torch.int32, device=logits.device))
        self.stats = torch.zeros((batch, width, tiles, 2), dtype=torch.float32, device=logits.device)
        self.lse = torch.zeros((batch, width), dtype=torch.float32, device=logits.device)
        self.keys = torch.zeros((batch, width, tiles, _CANDIDATES), dtype=torch.uint64, device=logits.device)
        self.summary = torch.zeros((batch * width, _SUMMARY), dtype=torch.float32, device=logits.device)
        self.gathered = torch.zeros((batch * width, world * _SUMMARY), dtype=torch.float32, device=logits.device)
        self.summary_bits = self.summary.view(torch.bfloat16)
        self.gathered_bits = self.gathered.view(torch.bfloat16)
        self.exchange = PreparedAllGather(self.summary_bits, self.gathered_bits,
            runtime=runtime, session=session, coordinator=coordinator, name=name + ".exchange")
        self.logits = logits
        self.row_inputs = dict(slots=slots, generations=epochs, live_rows=live)
        common = (logits, completion.records, frame, completion.requested, slots, epochs, live)
        geometry = (local, vocabulary, self.offset, width, tiles, completion.records.shape[1], completion.records.shape[2])
        rows = (batch, width, 1)
        self._preparation = (
            (_statistics, (*common, self.stats, *geometry, _TILE), (batch, width, tiles), 4),
            (_local_normalization, (*common[1:], self.stats, self.summary, *geometry[3:],
                triton.next_power_of_2(tiles), _SUMMARY), rows, 4),
            (_global_normalization, (*common[1:], self.gathered, self.lse, width, *geometry[5:],
                world, triton.next_power_of_2(world), _SUMMARY), rows, 4),
            (_candidates, (*common, completion.token_id_counts, self.lse, self.keys, *geometry,
                _TILE, _CANDIDATES), (batch, width, tiles), 4),
            (_pack, (*common, completion.token_id_counts, completion.token_ids, self.keys, self.summary,
                *geometry, triton.next_power_of_2(tiles * _CANDIDATES), _CANDIDATES,
                MAX_LOGPROB_TOKEN_IDS, _SUMMARY, _CHOSEN, _REQUESTED), rows, 8),
            (_finish, (*common[1:], completion.token_id_counts, completion.token_ids,
                self.gathered, self.lse, completion.logprobs, local, vocabulary, width,
                *geometry[5:], world, _SUMMARY, _CANDIDATES,
                triton.next_power_of_2(world * _CANDIDATES), MAX_TOP_LOGPROBS, MAX_LOGPROB_TOKEN_IDS,
                _CHOSEN, _REQUESTED, completion.token_capacity, LOGPROB_FIELDS, LOGPROB_PAIR_FIELDS), rows, 8),
        )
        self.operations = ()

    def prepare(self):
        if not self.operations:
            self.operations = tuple(prepare_triton(kernel, args, grid, num_warps=warps)
                                    for kernel, args, grid, warps in self._preparation)
            self._preparation = ()

    def functional(self, *, session, dynamic_inputs=False):
        if not self.operations:
            raise RuntimeError("vocabulary logprobs require preparation before functional conversion")
        values = {name: getattr(self, name) for name in ("logits", "stats", "lse", "keys", "summary", "gathered")}
        self.local_layout = {name: (tuple(value.shape), value.dtype) for name, value in values.items()}
        names = {id(value): name for name, value in {**values, **self.row_inputs}.items()}
        for operation in self.operations:
            operation.local_slots = tuple((index, names[id(value)]) for index, value in enumerate(operation.args)
                                          if id(value) in names)
            operation.args = tuple(None if id(value) in names else value for value in operation.args)
        self.exchange.functional(session=session)
        if dynamic_inputs:
            self.row_inputs = dict.fromkeys(self.row_inputs)
        for name in (*values, "summary_bits", "gathered_bits"):
            setattr(self, name, None)
        return self

    def run(self, logits, *, slots=None, generations=None, live_rows=None):
        if logits.ndim == 2:
            logits = logits[:, None, :]
        local = {name: torch.empty(shape, dtype=dtype, device=logits.device)
                 for name, (shape, dtype) in self.local_layout.items() if name != "logits"}
        local["logits"] = logits
        for name, value in (("slots", slots), ("generations", generations), ("live_rows", live_rows)):
            local[name] = self.row_inputs[name] if value is None else value
            if local[name] is None:
                raise ValueError("vocabulary logprobs require runtime row operands")
        for index, operation in enumerate(self.operations):
            if index in (2, 5):
                local["gathered"] = self.exchange(local["summary"].view(torch.bfloat16)).view(torch.float32)
            args = list(operation.args)
            for slot, name in operation.local_slots:
                args[slot] = local[name]
            launch_triton(operation.kernel, operation.compiled, tuple(args), operation.grid, **operation.options)

    def __call__(self):
        if not self.operations:
            raise RuntimeError("vocabulary-parallel logprob preparation must finish before capture")
        stats, local, global_lse, candidates, pack, finish = self.operations
        stats()
        local()
        self.exchange(self.summary_bits)
        global_lse()
        candidates()
        pack()
        self.exchange(self.summary_bits)
        finish()
