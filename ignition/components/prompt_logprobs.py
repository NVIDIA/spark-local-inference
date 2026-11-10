# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Blocked vocabulary-shard projection and raw prompt-token statistics."""

import torch
import triton
import triton.language as tl

from ignition.preparation.scratch import GraphBufferPool
from ignition.preparation.triton import launch_triton, prepare_triton

from .collectives import PreparedAllGather
from .completion import MAX_TOP_LOGPROBS
from .linear import QuantizedProjection, UnquantizedProjection
from .logprobs import _sort_key, _sort_pair

BLOCK_ROWS = 64
_TILE = 1024
_CANDIDATES = 32
# BF16 communication transports these int32 words without arithmetic.
_PARTIAL_WORDS = 64
PROMPT_FIELDS = 3 + 2 * MAX_TOP_LOGPROBS


@triton.jit
def _statistics(Logits, Stats, V: tl.constexpr, LOCAL: tl.constexpr, Offset,
                TILES: tl.constexpr, TILE: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    offset = tl.load(Offset)
    ids = tile * TILE + tl.arange(0, TILE)
    x = tl.load(Logits + row * LOCAL + ids, (ids < LOCAL) & (ids + offset < V), other=-float('inf')).to(tl.float32)
    maximum = tl.max(x, 0)
    mass = tl.sum(tl.exp(x - tl.where(maximum == -float('inf'), 0., maximum)), 0)
    tl.store(Stats + (row * TILES + tile) * 2, maximum)
    tl.store(Stats + (row * TILES + tile) * 2 + 1, mass)


@triton.jit
def _local_stats(Logits, Targets, Stats, Partial, V: tl.constexpr, LOCAL: tl.constexpr,
                 Offset, START, TILES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offset = tl.load(Offset)
    tiles = tl.arange(0, BLOCK)
    maxima = tl.load(Stats + (row * TILES + tiles) * 2, tiles < TILES, other=-float('inf'))
    masses = tl.load(Stats + (row * TILES + tiles) * 2 + 1, tiles < TILES, other=0.)
    maximum = tl.max(maxima, 0)
    mass = tl.sum(masses * tl.exp(maxima - tl.where(maximum == -float('inf'), 0., maximum)), 0)
    target = tl.load(Targets + START + row)
    local = target - offset
    chosen = tl.load(Logits + row * LOCAL + local,
        (target >= 0) & (target < V) & (local >= 0) & (local < LOCAL), other=-float('inf')).to(tl.float32)
    tl.store(Partial + row * 64, maximum.to(tl.int32, bitcast=True))
    tl.store(Partial + row * 64 + 1, mass.to(tl.int32, bitcast=True))
    tl.store(Partial + row * 64 + 2, chosen.to(tl.int32, bitcast=True))


@triton.jit
def _global_stats(Gathered, Norm, WORLD: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    ranks = tl.arange(0, BLOCK)
    base = Gathered + row * WORLD * 64 + ranks * 64
    maxima = tl.load(base, ranks < WORLD, other=-8388608).to(tl.float32, bitcast=True)
    masses = tl.load(base + 1, ranks < WORLD, other=0).to(tl.float32, bitcast=True)
    chosen = tl.max(tl.load(base + 2, ranks < WORLD, other=-8388608).to(tl.float32, bitcast=True), 0)
    maximum = tl.max(maxima, 0)
    lse = maximum + tl.log(tl.sum(masses * tl.exp(maxima - maximum), 0))
    tl.store(Norm + row * 2, lse)
    tl.store(Norm + row * 2 + 1, chosen - lse)


@triton.jit
def _candidates(Logits, Norm, Keys, Counts, V: tl.constexpr, LOCAL: tl.constexpr,
                Offset, TILES: tl.constexpr, TILE: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    offset = tl.load(Offset)
    ids = tile * TILE + tl.arange(0, TILE)
    valid = (ids < LOCAL) & (ids + offset < V)
    x = tl.load(Logits + row * LOCAL + ids, valid, other=-float('inf')).to(tl.float32)
    x -= tl.load(Norm + row * 2)
    chosen = tl.load(Norm + row * 2 + 1)
    tl.store(Counts + row * TILES + tile, tl.sum((valid & (x > chosen)).to(tl.int32), 0))
    keys = tl.sort(tl.where(valid, _sort_key(x, ids + offset), 0), descending=True)
    k = tl.arange(0, TILE)
    tl.store(Keys + (row * TILES + tile) * 32 + k, keys, k < 32)


@triton.jit
def _local_candidates(Keys, Counts, Partial, TILES: tl.constexpr, BLOCK: tl.constexpr,
                       COUNT_BLOCK: tl.constexpr, TOP: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    keys = tl.sort(tl.load(Keys + row * TILES * 32 + k, k < TILES * 32, other=0), descending=True)
    ids, values = _sort_pair(keys)
    out = Partial + row * 64
    tl.store(out + 1 + 2 * k, tl.where(keys != 0, ids, -1), k < TOP)
    tl.store(out + 2 + 2 * k, values.to(tl.int32, bitcast=True), k < TOP)
    tiles = tl.arange(0, COUNT_BLOCK)
    tl.store(out, tl.sum(tl.load(Counts + row * TILES + tiles, tiles < TILES, other=0), 0))


@triton.jit
def _finish(Targets, Gathered, Norm, Output, Frame, V: tl.constexpr, START,
            CAPACITY: tl.constexpr, WORLD: tl.constexpr, TOP: tl.constexpr,
            BLOCK: tl.constexpr, RANK_BLOCK: tl.constexpr, FIELDS: tl.constexpr):
    row = tl.program_id(0)
    target = tl.load(Targets + START + row)
    valid = (target >= 0) & (target < V)
    k = tl.arange(0, BLOCK)
    rank, col = k // TOP, k % TOP
    base = Gathered + row * WORLD * 64
    ids = tl.load(base + rank * 64 + 1 + col * 2, rank < WORLD, other=-1)
    values = tl.load(base + rank * 64 + 2 + col * 2, rank < WORLD, other=0).to(tl.float32, bitcast=True)
    keys = tl.sort(tl.where((rank < WORLD) & (ids >= 0), _sort_key(values, ids), 0), descending=True)
    ids, values = _sort_pair(keys)
    ranks = tl.arange(0, RANK_BLOCK)
    actual_rank = 1 + tl.sum(tl.load(base + ranks * 64, ranks < WORLD, other=0), 0)
    out = Output + (tl.load(Frame) * CAPACITY + START + row) * FIELDS
    tl.store(out, tl.where(valid, target, -1))
    tl.store(out + 1, tl.where(valid, tl.load(Norm + row * 2 + 1), float('nan')).to(tl.int32, bitcast=True))
    tl.store(out + 2, tl.where(valid, actual_rank, 0))
    tl.store(out + 3 + k * 2, tl.where(valid & (keys != 0), ids, -1), k < TOP)
    tl.store(out + 4 + k * 2, tl.where(valid, values, float('nan')).to(tl.int32, bitcast=True), k < TOP)


class PromptOutputs:
    def __init__(self, capacity, device):
        self.targets = torch.full((capacity,), -1, dtype=torch.int32, device=device)
        self.values = torch.zeros((2, capacity, PROMPT_FIELDS), dtype=torch.int32, device=device)
        self.scratch = GraphBufferPool(device, chunk_bytes=2 << 20)

    def buffers(self):
        return dict(prompt_target_tokens=self.targets, prompt_logprobs=self.values)


class PreparedPromptReduction:
    """Reduce one bounded row block; partial exchanges never contain vocabulary rows."""

    def __init__(self, logits, targets, output, frame, *, start, vocabulary, offset,
                 runtime, session, coordinator, name, buffers):
        rows, local = logits.shape
        if isinstance(offset, int):
            offset = torch.tensor([offset], dtype=torch.int32, device=logits.device)
        world = 1 if runtime is None else runtime.world_size
        tiles = triton.cdiv(local, _TILE)
        def zeros(shape, dtype):
            return (torch.zeros(shape, dtype=dtype, device=logits.device) if buffers is None
                    else buffers.zeros(shape, dtype))
        stats = zeros((rows, tiles, 2), torch.float32)
        norm = zeros((rows, 2), torch.float32)
        keys = zeros((rows, tiles, _CANDIDATES), torch.uint64)
        counts = zeros((rows, tiles), torch.int32)
        partial = zeros((rows, _PARTIAL_WORDS), torch.int32)
        gathered = zeros((rows, world * _PARTIAL_WORDS), torch.int32)
        self.gather = PreparedAllGather(partial.view(torch.bfloat16), gathered.view(torch.bfloat16),
            runtime=runtime, session=session, coordinator=coordinator, name=name + '.exchange')
        self.partial = partial
        self.geometry = rows, local, tiles, world, vocabulary, start
        self.outputs = targets, output, frame, offset
        self.operations = tuple(prepare_triton(kernel, args, grid, num_warps=warps)
            for kernel, args, grid, warps in (
                (_statistics, (logits, stats, vocabulary, local, offset, tiles, _TILE), (rows, tiles, 1), 4),
                (_local_stats, (logits, targets, stats, partial, vocabulary, local, offset, start,
                    tiles, triton.next_power_of_2(tiles)), (rows, 1, 1), 4),
                (_global_stats, (gathered, norm, world, triton.next_power_of_2(world)), (rows, 1, 1), 4),
                (_candidates, (logits, norm, keys, counts, vocabulary, local, offset, tiles, _TILE), (rows, tiles, 1), 4),
                (_local_candidates, (keys, counts, partial, tiles, triton.next_power_of_2(tiles * _CANDIDATES),
                    triton.next_power_of_2(tiles), MAX_TOP_LOGPROBS), (rows, 1, 1), 8),
                (_finish, (targets, gathered, norm, output, frame, vocabulary, start, output.shape[1], world,
                    MAX_TOP_LOGPROBS, triton.next_power_of_2(world * MAX_TOP_LOGPROBS),
                    triton.next_power_of_2(world), PROMPT_FIELDS), (rows, 1, 1), 4),
            ))

    def functional(self, *, session):
        return _FunctionalPromptReduction(self, session=session)

    def __call__(self):
        statistics, local_stats, global_stats, candidates, local_candidates, finish = self.operations
        statistics()
        local_stats()
        self.gather(self.partial.view(torch.bfloat16))
        global_stats()
        candidates()
        local_candidates()
        self.gather(self.partial.view(torch.bfloat16))
        finish()


class _FunctionalPromptReduction:
    def __init__(self, prepared, *, session):
        self.geometry, self.outputs = prepared.geometry, prepared.outputs
        self.gather = prepared.gather.functional(session=session)
        self.operations = tuple((op.kernel, op.compiled, op.grid, op.options) for op in prepared.operations)

    def __call__(self, logits):
        rows, local, tiles, world, vocabulary, start = self.geometry
        targets, output, frame, offset = self.outputs
        def zeros(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=logits.device)
        stats, norm = zeros((rows, tiles, 2), torch.float32), zeros((rows, 2), torch.float32)
        keys, counts = zeros((rows, tiles, _CANDIDATES), torch.uint64), zeros((rows, tiles), torch.int32)
        partial = zeros((rows, _PARTIAL_WORDS), torch.int32)
        def run(index, args):
            kernel, compiled, grid, options = self.operations[index]
            launch_triton(kernel, compiled, args, grid, **options)
        run(0, (logits, stats, vocabulary, local, offset, tiles, _TILE))
        run(1, (logits, targets, stats, partial, vocabulary, local, offset, start,
                tiles, triton.next_power_of_2(tiles)))
        gathered = self.gather(partial.view(torch.bfloat16)).view(torch.int32)
        run(2, (gathered, norm, world, triton.next_power_of_2(world)))
        run(3, (logits, norm, keys, counts, vocabulary, local, offset, tiles, _TILE))
        run(4, (keys, counts, partial, tiles, triton.next_power_of_2(tiles * _CANDIDATES),
                triton.next_power_of_2(tiles), MAX_TOP_LOGPROBS))
        gathered = self.gather(partial.view(torch.bfloat16)).view(torch.int32)
        run(5, (targets, gathered, norm, output, frame, vocabulary, start, output.shape[1], world,
                MAX_TOP_LOGPROBS, triton.next_power_of_2(world * MAX_TOP_LOGPROBS),
                triton.next_power_of_2(world), PROMPT_FIELDS))


class PreparedPromptScoring:
    def __init__(self, source, weight, outputs, frame, *, vocabulary, rank=0,
                 runtime=None, session, coordinator=None, name, copy_source=False, logits_dtype=torch.bfloat16,
                 local_outputs=False):
        local = weight.shape[0] if isinstance(weight, torch.Tensor) else weight.out_features
        buffers = None if local_outputs else outputs.scratch.cursor()
        def zeros(shape, dtype):
            return (torch.zeros(shape, dtype=dtype, device=source.device) if buffers is None
                    else buffers.zeros(shape, dtype))
        if copy_source:
            source = zeros(source.shape, source.dtype)
        self.source = source
        rows = min(BLOCK_ROWS, source.shape[0])
        logits = zeros((rows, local), logits_dtype)
        stage = None if buffers is None else buffers.fork()
        self.offset = torch.tensor([rank * local], dtype=torch.int32, device=source.device)
        self.parts = []
        projection_type = UnquantizedProjection if isinstance(weight, torch.Tensor) else QuantizedProjection
        options = {} if isinstance(weight, torch.Tensor) else dict(activation_mode="auto")
        for start in range(0, source.shape[0], rows):
            count = min(rows, source.shape[0] - start)
            view = source[start:start + count]
            projection = projection_type(view, weight, logits[:count], session=session,
                name=f"{name}.head{start}", **options)
            reduction = PreparedPromptReduction(logits[:count], outputs.targets, outputs.values, frame,
                start=start, vocabulary=vocabulary, offset=self.offset, runtime=runtime,
                session=session, coordinator=coordinator, name=f'{name}.reduce{start}',
                buffers=None if stage is None else stage.fork())
            self.parts.append((view, projection, reduction))

    def functional(self, *, session):
        parts = tuple((start, view.shape[0], projection.functional(session=session), reduction.functional(session=session))
                      for start, (view, projection, reduction) in zip(
                          range(0, self.source.shape[0], BLOCK_ROWS), self.parts, strict=True))
        return _FunctionalPromptScoring(parts)

    def __call__(self):
        for source, projection, reduction in self.parts:
            projection(source)
            reduction()


class _FunctionalPromptScoring:
    def __init__(self, parts):
        self.parts = parts

    def __call__(self, source):
        for start, count, projection, reduction in self.parts:
            reduction(projection(source[start:start + count]))


class PromptScoring:
    """One package's optional outputs, communication channel, and serial scratch pool."""

    def __init__(self, capacity, device, *, frame, group=None, ipc=None, remote_runtime=None):
        self.outputs = PromptOutputs(capacity, device)
        self.frame = frame
        self.runtime = remote_runtime
        self.owns_runtime = False
        self.rank = 0
        if group is not None:
            import torch.distributed as dist
            from ignition.preparation.collectives import create_all_gather_runtime

            self.rank = dist.get_rank(group)
            if self.runtime is None:
                self.owns_runtime = True
                self.runtime = create_all_gather_runtime(group, device, max_rows=BLOCK_ROWS,
                    columns=128 * dist.get_world_size(group), ipc=ipc)

    def close(self):
        if self.owns_runtime:
            self.runtime.close()

    @classmethod
    def allocate(cls, config, device, buffers, *, frame, group=None, ipc=None, runtime=None, coordinator=None):
        if not config.prompt_logprobs:
            return None
        scoring = cls(config.prefill_chunk_size, device, frame=frame, group=group, ipc=ipc,
            remote_runtime=runtime if coordinator is not None and coordinator.spans_hosts else None)
        buffers.update(scoring.outputs.buffers())
        return scoring

    def memory_report(self):
        return dict(prompt_scoring_scratch_bytes=self.outputs.scratch.nbytes,
                    prompt_output_bytes=sum(t.nbytes for t in self.outputs.buffers().values()))

    def zero_ranges(self):
        from ignition.preparation.disposable import storage_ranges

        return storage_ranges((self.outputs.buffers(), self.outputs.scratch.chunks))

    def prepare(self, source, weight, *, vocabulary, session, coordinator, name, copy_source=False,
                logits_dtype=torch.bfloat16, functional=False):
        operation = PreparedPromptScoring(source, weight, self.outputs, self.frame, vocabulary=vocabulary,
            rank=self.rank, runtime=self.runtime, session=session, coordinator=coordinator,
            name=name, copy_source=copy_source, logits_dtype=logits_dtype, local_outputs=functional)
        return operation.functional(session=session) if functional else operation


def scored_prefill_capacities(chunk):
    values = []
    capacity = 128
    while capacity < chunk:
        values.append(capacity)
        capacity *= 4
    return (*values, chunk)


def prefill_modes(prefills, decodes, scoring):
    result = [("prefill", "prefill", tuple(reversed(prefills)))]
    if scoring:
        result.append(("scored_prefill", "prefill", tuple(reversed(scored_prefill_capacities(max(prefills))))))
    result.append(("decode", "decode", decodes))
    return result


def prefill_graph_count(prefills, scoring):
    return sum(len(sizes) for _, _, sizes in prefill_modes(prefills, (), scoring))


@triton.jit
def _begin_scoring(Slots, Epochs, Live, Generations, Budget, Status):
    slot = tl.program_id(0)
    if (tl.load(Live) == 1) & (tl.load(Slots) == slot):
        if (tl.load(Epochs) == tl.load(Generations + slot)) & (tl.load(Budget + slot) == 0) & (tl.load(Status + slot) == 2):
            tl.store(Status + slot, 1)


class PreparedScoredPrefill:
    def __init__(self, forward, requests, inputs):
        self.forward = forward
        self.compilation_variant = getattr(forward, "compilation_variant", "shared")
        self.begin = prepare_triton(_begin_scoring, (inputs.row_slot_ids, inputs.row_generations, inputs.live_rows,
            requests.generation, requests.max_new_tokens, requests.status),
            (requests.capacity, 1, 1), num_warps=4)

    def __call__(self):
        self.begin()
        self.forward()
