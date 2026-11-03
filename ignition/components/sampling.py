# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Request-owned stochastic sampling and admission for native CUDA graph replay.

Physical slots retain RNG counters, token counts, penalties, and grammar masks.
Recipes own model history and supply vocabulary size and EOS token IDs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .completion import CompletionOutputs, completion_record_capacity
from .logprobs import PreparedLogprobs
from .capacity import BATCH_CAPACITIES
from .flashinfer_sampling import filter_probabilities_out, probability_filter_workspace, sample_probabilities
from ignition.preparation.scratch import bind_scratch

EOS_CAPACITY = 8
INACTIVE, RUNNING, FINISHED, ERROR = 0, 1, 2, 3
INVALID_DISTRIBUTION, UNCONSUMED_PENDING, INVALID_CONSUMPTION, DUPLICATE_SLOT = 1, 2, 3, 4
INVALID_ADMISSION = 5


@dataclass(frozen=True)
class RequestAdmission:
    generation: int
    seed: int
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 20
    top_p: float = .95
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    eos_token_ids: tuple[int, ...] = ()

    def validate(self, vocab_size: int) -> None:
        for name, lower, upper in (("generation", 1, 2**63 - 1), ("seed", 0, 2**64 - 1),
                                   ("max_new_tokens", 0, 2**31 - 1), ("top_k", -1, vocab_size)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
        for name in ("temperature", "top_p", "min_p", "repetition_penalty", "presence_penalty", "frequency_penalty"):
            value = getattr(self, name)
            if not math.isfinite(value) or abs(value) > torch.finfo(torch.float32).max:
                raise ValueError(f"{name} must be finite FP32")
            if name in ("temperature", "top_p", "repetition_penalty") and 0 < value < torch.finfo(torch.float32).tiny:
                raise ValueError(f"{name} must be a positive normal FP32 value")
        if self.temperature <= 0 or not 0 < self.top_p <= 1 or not 0 <= self.min_p <= 1 or self.repetition_penalty <= 0:
            raise ValueError("require temperature > 0, 0 < top_p <= 1, 0 <= min_p <= 1 and repetition_penalty > 0")
        if len(self.eos_token_ids) > EOS_CAPACITY or any(type(x) is not int or not 0 <= x < vocab_size for x in self.eos_token_ids):
            raise ValueError(f"EOS requires at most {EOS_CAPACITY} vocabulary token IDs")


@dataclass
class DeviceRequestState:
    generation: torch.Tensor
    consumed_tokens: torch.Tensor
    generated_tokens: torch.Tensor
    pending_token: torch.Tensor
    status: torch.Tensor
    error_code: torch.Tensor
    seed: torch.Tensor
    rng_counter: torch.Tensor
    histogram: torch.Tensor
    parameters: torch.Tensor
    top_k: torch.Tensor
    max_new_tokens: torch.Tensor
    eos_token_ids: torch.Tensor
    eos_count: torch.Tensor
    grammar_mask: torch.Tensor
    grammar_enabled: torch.Tensor

    @classmethod
    def allocate(cls, device, *, vocab_size: int,
                 consumed_tokens: torch.Tensor | None = None, capacity: int = 4) -> DeviceRequestState:
        """Allocate physical request slots for the supplied vocabulary.

        ``parameters`` columns are temperature, top-p, min-p, repetition,
        presence and frequency. Grammar words are little-bit-order int32 words:
        bit ``token % 32`` of word ``token // 32`` permits that token. Seeds and
        counters are uint64. Other field dtypes are explicit in this allocator.
        """
        if type(vocab_size) is not int or not 1 <= vocab_size <= 2**31 - 1:
            raise ValueError("vocab_size must be a positive int32 value")
        def zeros(shape, dtype=torch.int64):
            return torch.zeros(shape, dtype=dtype, device=device)
        if capacity not in BATCH_CAPACITIES:
            raise ValueError("request capacity must be a power of two through 64")
        consumed = zeros(capacity) if consumed_tokens is None else consumed_tokens
        actual_device = torch.empty(0, device=device).device
        _check_tensor(consumed, (capacity,), torch.int64, actual_device, "consumed_tokens", contiguous=False)
        if consumed.stride(0) <= 0:
            raise ValueError("consumed_tokens requires positive row stride")
        return cls(zeros(capacity), consumed, zeros(capacity),
            torch.full((capacity,), -1, dtype=torch.int64, device=device),
            zeros(capacity, torch.int32), zeros(capacity, torch.int32),
            zeros(capacity, torch.uint64), zeros(capacity, torch.uint64),
            zeros((capacity, vocab_size), torch.int32), zeros((capacity, 6), torch.float32),
            zeros(capacity, torch.int32), zeros(capacity),
            torch.full((capacity, EOS_CAPACITY), -1, dtype=torch.int64, device=device),
            zeros(capacity, torch.int32),
            torch.full((capacity, triton.cdiv(vocab_size, 32)), -1, dtype=torch.int32, device=device),
            zeros(capacity, torch.int32))

    @property
    def capacity(self) -> int:
        return self.generation.numel()

    @property
    def vocab_size(self) -> int:
        return self.histogram.shape[1]

    def admit(self, slot: int, admission: RequestAdmission, prompt_ids, *, consumed_tokens: int = 0) -> None:
        """Bootstrap a slot outside replay; Rust admission uploads the same fields.

        The complete prompt initializes penalties even when consumed_tokens
        restores a cached prefix. Epoch reuse protection belongs to the native
        slot allocator. No hidden Python state participates in admission.
        """
        admission.validate(self.vocab_size)
        if type(slot) is not int or not 0 <= slot < self.capacity:
            raise ValueError("slot must be within request capacity")
        ids = torch.as_tensor(prompt_ids, dtype=torch.int64, device="cpu")
        if ids.ndim != 1 or bool(((ids < 0) | (ids >= self.vocab_size)).any()):
            raise ValueError("prompt_ids must be a one-dimensional vocabulary token list")
        if type(consumed_tokens) is not int or not 0 <= consumed_tokens <= len(ids):
            raise ValueError("consumed_tokens must lie within the complete prompt")
        self.generation[slot] = admission.generation
        self.consumed_tokens[slot] = consumed_tokens
        self.generated_tokens[slot] = 0
        self.pending_token[slot] = -1
        self.status[slot] = RUNNING if admission.max_new_tokens else FINISHED
        self.error_code[slot] = 0
        self.seed[slot:slot + 1].copy_(torch.tensor([admission.seed], dtype=torch.uint64))
        self.rng_counter[slot] = 0
        self.histogram[slot].copy_(torch.bincount(ids, minlength=self.vocab_size).to(torch.int32))
        self.parameters[slot].copy_(torch.tensor((admission.temperature, admission.top_p, admission.min_p,
            admission.repetition_penalty, admission.presence_penalty, admission.frequency_penalty), dtype=torch.float32))
        self.top_k[slot] = admission.top_k
        self.max_new_tokens[slot] = admission.max_new_tokens
        self.eos_token_ids[slot].fill_(-1)
        if admission.eos_token_ids:
            self.eos_token_ids[slot, :len(admission.eos_token_ids)].copy_(torch.tensor(admission.eos_token_ids))
        self.eos_count[slot] = len(admission.eos_token_ids)
        self.grammar_mask[slot].fill_(-1)
        self.grammar_enabled[slot] = 0


@dataclass(frozen=True)
class AdmissionBuffers:
    """Contiguous staging for packed admissions uploaded by Rust.

    Each active packed row identifies one physical request slot. The native
    worker orders uploads, hybrid-state reset/restore, this initialization
    graph, and forward execution on its CUDA stream. Seed bits use int64 ABI
    storage and are bitcast to the resident uint64 Philox key.
    """

    admission_count: torch.Tensor
    admission_slots: torch.Tensor
    admission_generations: torch.Tensor
    admission_seed_bits: torch.Tensor
    admission_parameters: torch.Tensor
    admission_top_k: torch.Tensor
    admission_max_new_tokens: torch.Tensor
    admission_eos_token_ids: torch.Tensor
    admission_eos_count: torch.Tensor
    admission_histogram: torch.Tensor
    admission_grammar_mask: torch.Tensor
    admission_grammar_enabled: torch.Tensor

    @classmethod
    def allocate(cls, device, *, vocab_size: int, capacity: int = 4) -> AdmissionBuffers:
        if type(vocab_size) is not int or not 1 <= vocab_size <= 2**31 - 1:
            raise ValueError("vocab_size must be a positive int32 value")
        if capacity not in BATCH_CAPACITIES:
            raise ValueError("admission capacity must be a power of two through 64")
        def zeros(shape, dtype=torch.int32):
            return torch.zeros(shape, dtype=dtype, device=device)
        return cls(zeros(1), torch.full((capacity,), -1, dtype=torch.int32, device=device),
            zeros(capacity, torch.int64), zeros(capacity, torch.int64), zeros((capacity, 6), torch.float32),
            zeros(capacity), zeros(capacity, torch.int64),
            torch.full((capacity, 8), -1, dtype=torch.int64, device=device), zeros(capacity),
            zeros((capacity, vocab_size)),
            torch.full((capacity, triton.cdiv(vocab_size, 32)), -1, dtype=torch.int32, device=device), zeros(capacity))

    def stage(self, row: int, slot: int, admission: RequestAdmission, prompt_ids) -> None:
        """Populate one row for qualification; production uses native H2D uploads.

        ``admission_count`` is set by the caller after staging the complete
        packed batch. This helper never changes live request state.
        """
        vocab = self.admission_histogram.shape[1]
        admission.validate(vocab)
        if type(row) is not int or not 0 <= row < self.admission_slots.numel() or type(slot) is not int or not 0 <= slot < self.admission_slots.numel():
            raise ValueError("admission row and slot must be within request capacity")
        ids = torch.as_tensor(prompt_ids, dtype=torch.int64, device="cpu")
        if ids.ndim != 1 or bool(((ids < 0) | (ids >= vocab)).any()):
            raise ValueError("prompt_ids must be a one-dimensional vocabulary token list")
        self.admission_slots[row] = slot
        self.admission_generations[row] = admission.generation
        self.admission_seed_bits[row:row + 1].copy_(torch.tensor([admission.seed], dtype=torch.uint64).view(torch.int64))
        self.admission_parameters[row].copy_(torch.tensor((admission.temperature, admission.top_p, admission.min_p,
            admission.repetition_penalty, admission.presence_penalty, admission.frequency_penalty), dtype=torch.float32))
        self.admission_top_k[row] = admission.top_k
        self.admission_max_new_tokens[row] = admission.max_new_tokens
        self.admission_eos_token_ids[row].fill_(-1)
        if admission.eos_token_ids:
            self.admission_eos_token_ids[row, :len(admission.eos_token_ids)].copy_(torch.tensor(admission.eos_token_ids))
        self.admission_eos_count[row] = len(admission.eos_token_ids)
        self.admission_histogram[row].copy_(torch.bincount(ids, minlength=vocab).to(torch.int32))
        self.admission_grammar_mask[row].fill_(-1)
        self.admission_grammar_enabled[row] = 0


def philox4x32_reference(seed: int, counter: int) -> tuple[int, int, int, int]:
    """Philox4x32-10 with a uint64 block counter and uint64 key.

    The counter words are (low32, high32, 0, 0), matching CUDA's Philox round
    constants and Triton's randint4x. FlashInfer samples from the first word
    with CUDA curand_uniform conversion and an explicit offset of 4*counter.
    """
    mask = 2**32 - 1
    c0, c1, c2, c3 = counter & mask, (counter >> 32) & mask, 0, 0
    k0, k1 = seed & mask, (seed >> 32) & mask
    for _ in range(10):
        p0, p1 = 0xD2511F53 * c0, 0xCD9E8D57 * c2
        c0, c1, c2, c3 = (p1 >> 32) ^ c1 ^ k0, p1 & mask, (p0 >> 32) ^ c3 ^ k1, p0 & mask
        k0, k1 = (k0 + 0x9E3779B9) & mask, (k1 + 0xBB67AE85) & mask
    return c0, c1, c2, c3


def _compile(kernel, args, grid):
    from ignition.preparation.triton import prepare_triton

    return prepare_triton(kernel, args, grid, num_warps=4, enable_fp_fusion=False)


@triton.jit
def _scores(Logits, Scores, Slots, Epochs, Live, Generation, Status, Counts, Params, Grammar, GrammarEnabled,
            V: tl.constexpr, WORDS: tl.constexpr, BLOCK: tl.constexpr, SLOTS: tl.constexpr = 4):
    row, block = tl.program_id(0), tl.program_id(1)
    token = block * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(Slots + row)
    valid = (row < tl.load(Live)) & (slot >= 0) & (slot < SLOTS)
    valid &= tl.load(Generation + slot, valid, other=-1) == tl.load(Epochs + row)
    valid &= tl.load(Status + slot, valid, other=0) == 1
    x = tl.load(Logits + row * V + token, token < V, other=-float("inf")).to(tl.float32)
    temperature = tl.load(Params + slot * 6, valid, other=1.)
    repetition = tl.load(Params + slot * 6 + 3, valid, other=1.)
    presence = tl.load(Params + slot * 6 + 4, valid, other=0.)
    frequency = tl.load(Params + slot * 6 + 5, valid, other=0.)
    # Positive-zero penalties and nonnegative counts preserve signed logits when their arithmetic is omitted.
    neutral = ((repetition == 1.) & (presence.to(tl.int32, bitcast=True) == 0)
        & (frequency.to(tl.int32, bitcast=True) == 0))
    enabled = tl.load(GrammarEnabled + slot, valid, other=0)
    allowed = tl.full((BLOCK,), True, tl.int1)
    if not (neutral & (enabled == 0)):
        counts = tl.load(Counts + slot * V + token, valid & (token < V), other=0)
        x = tl.where(counts > 0, tl.where(x > 0., x / repetition, x * repetition) - presence, x)
        x = x - frequency * counts
        bits = tl.load(Grammar + slot * WORDS + token // 32, valid & (token < V), other=0).to(tl.uint32)
        allowed = (enabled == 0) | ((bits >> (token % 32)) & 1 != 0)
    x = x / tl.where(valid, temperature, 1.)
    tl.store(Scores + row * V + token, tl.where(valid & allowed, x, -float("inf")), token < V)


@triton.jit
def _sampling_metadata(Slots, Epochs, Live, Generation, Status, Params, TopK, Seed, Counter,
                       PackedK, PackedP, PackedMinP, PackedSeed, PackedOffset, V: tl.constexpr, SLOTS: tl.constexpr = 4):
    row = tl.program_id(0)
    slot = tl.load(Slots + row)
    valid = (row < tl.load(Live)) & (slot >= 0) & (slot < SLOTS)
    valid &= tl.load(Generation + slot, valid, other=-1) == tl.load(Epochs + row)
    valid &= tl.load(Status + slot, valid, other=0) == 1
    k = tl.load(TopK + slot, valid, other=0)
    tl.store(PackedK + row, tl.where(k > 0, tl.minimum(k, V), V))
    tl.store(PackedP + row, tl.load(Params + slot * 6 + 1, valid, other=1.))
    tl.store(PackedMinP + row, tl.load(Params + slot * 6 + 2, valid, other=0.))
    tl.store(PackedSeed + row, tl.load(Seed + slot, valid, other=0))
    tl.store(PackedOffset + row, tl.load(Counter + slot, valid, other=0) * 4)


@triton.jit
def _sanitize_scores(Scores, Maximum, V: tl.constexpr, BLOCK: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    token = block * BLOCK + tl.arange(0, BLOCK)
    maximum = tl.load(Maximum + row)
    valid = (maximum == maximum) & (maximum != float("inf")) & (maximum != -float("inf"))  # noqa: PLR0124
    value = tl.load(Scores + row * V + token, token < V, other=-float("inf"))
    dummy = tl.where(token == 0, 0., -float("inf"))
    tl.store(Scores + row * V + token, tl.where(valid, value, dummy), token < V)


@triton.jit
def _mask_min_p(Input, Output, Maximum, MinP, V: tl.constexpr, BLOCK: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    token = block * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Input + row * V + token, token < V, other=0.)
    threshold = tl.load(MinP + row) * tl.load(Maximum + row)
    tl.store(Output + row * V + token, tl.where(value >= threshold, value, 0.), token < V)


@triton.jit
def _normalize_probs(Probability, Mass, V: tl.constexpr, BLOCK: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    token = block * BLOCK + tl.arange(0, BLOCK)
    mass = tl.load(Mass + row)
    valid = (mass > 0.) & (mass != float("inf"))
    value = tl.load(Probability + row * V + token, token < V, other=0.)
    value = tl.where(valid, value / mass, (token == 0).to(tl.float32))
    tl.store(Probability + row * V + token, value, token < V)


@triton.jit
def _write_completion(Records, Frame, Sequence, slot, epoch, consumed, generated,
                      token, status, error, active,
                      RECORD_FIELDS: tl.constexpr, SLOTS: tl.constexpr = 4):
    fields = tl.arange(0, triton.next_power_of_2(RECORD_FIELDS))
    value = tl.full((triton.next_power_of_2(RECORD_FIELDS),), 0, tl.int64)
    value = tl.where(fields == 0, tl.where(active, epoch, 0), value)
    value = tl.where(fields == 1, tl.where(active, tl.load(Sequence), 0), value)
    value = tl.where(fields == 2, tl.where(active, consumed, 0), value)
    value = tl.where(fields == 3, tl.where(active, generated, 0), value)
    value = tl.where(fields == 4, tl.where(active, token, -1), value)
    value = tl.where(fields == 5, tl.where(active, status, 0), value)
    value = tl.where(fields == 6, tl.where(active, error, 0), value)
    value = tl.where(fields == 8, (active & (token >= 0)).to(tl.int64), value)
    value = tl.where(fields == 12, -1, value)
    value = tl.where(fields == 13, tl.where(active, tl.maximum(consumed - 1, 0), 0), value)
    value = tl.where(fields == 15, 1, value)
    value = tl.where(fields >= 16, -1, value)
    value = tl.where(fields == 16, tl.where(active, token, -1), value)
    tl.store(Records + (tl.load(Frame) * SLOTS + slot) * RECORD_FIELDS + fields,
             value, fields < RECORD_FIELDS)


@triton.jit
def _complete(Maximum, Sampled, SampleValid, Mass, Slots, Epochs, Live, SampleEnabled, Generation, Consumed, Generated, Pending, Status, Error,
              Counter, Counts, Budget, Eos, EosCount, Records, Frame, Sequence,
              V: tl.constexpr, ROWS: tl.constexpr, CS: tl.constexpr,
              RECORD_FIELDS: tl.constexpr, SAMPLE: tl.constexpr, SLOTS: tl.constexpr = 4):
    slot = tl.program_id(0)
    rows = tl.arange(0, SLOTS)
    matches = (rows < ROWS) & (rows < tl.load(Live))
    matches &= tl.load(Slots + rows, rows < ROWS, other=-1) == slot
    epoch = tl.load(Generation + slot)
    matches &= tl.load(Epochs + rows, rows < ROWS, other=-1) == epoch
    matches_count = tl.sum(matches.to(tl.int32), 0)
    row = tl.max(tl.where(matches, rows, 0), 0)
    state = tl.load(Status + slot)
    code = tl.load(Error + slot)
    token = tl.full((), -1, tl.int64)
    generated = tl.load(Generated + slot)
    if matches_count > 1:
        state, code = 3, 4
        tl.store(Status + slot, state)
        tl.store(Error + slot, code)
    if SAMPLE:
        sample_enabled = tl.load(SampleEnabled + row) != 0
        if (matches_count == 1) & (state == 1) & SAMPLE & sample_enabled:
            pending = tl.load(Pending + slot)
            if pending >= 0:
                state, code = 3, 2
            elif generated >= tl.load(Budget + slot):
                state = 2
            else:
                maximum = tl.load(Maximum + row)
                finite = (maximum == maximum) & (maximum != float("inf")) & (maximum != -float("inf"))  # noqa: PLR0124
                counter = tl.load(Counter + slot)
                total = tl.load(Mass + row)
                finite &= (total > 0.) & (total == total) & (total != float("inf"))  # noqa: PLR0124
                selected = tl.load(Sampled + row).to(tl.int64)
                finite &= tl.load(SampleValid + row) & (selected >= 0) & (selected < V)
                if finite:
                    token = selected
                    generated += 1
                    tl.store(Pending + slot, token)
                    tl.store(Generated + slot, generated)
                    tl.store(Counter + slot, counter + 1)
                    count = tl.load(Counts + slot * V + token)
                    tl.store(Counts + slot * V + token, count + 1)
                    eos_index = tl.arange(0, 8)
                    eos = tl.load(Eos + slot * 8 + eos_index)
                    stop = tl.sum(((eos_index < tl.load(EosCount + slot)) & (eos == token)).to(tl.int32), 0) > 0
                    state = tl.where(stop | (generated >= tl.load(Budget + slot)), 2, 1)
                else:
                    state, code = 3, 1
            tl.store(Status + slot, state)
            tl.store(Error + slot, code)
    consumed = tl.load(Consumed + slot * CS)
    _write_completion(Records, Frame, Sequence, slot, epoch, consumed, generated,
                      token, state, code, matches_count > 0, RECORD_FIELDS, SLOTS)


def _check_tensor(tensor, shape, dtype, device, name, *, contiguous=True):
    if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype or tensor.device != device or (contiguous and not tensor.is_contiguous()):
        raise ValueError(f"{name} requires contiguous {dtype} {tuple(shape)} on {device}")


def _completion_fields(records, device, capacity=4):
    completion_record_capacity(records, capacity)
    _check_tensor(records, records.shape, torch.int64, device, "completion_records")
    return records.shape[2]


class PreparedCompletion:
    """Record consumption and errors without vocabulary outputs or RNG work."""

    def __init__(self, state: DeviceRequestState, row_slot_ids, row_generations, live_rows,
                 completion: CompletionOutputs, frame_index, submission_sequence):
        records = completion.records
        rows, device = row_slot_ids.numel(), state.generation.device
        if device.type != "cuda" or not 1 <= rows <= state.capacity:
            raise ValueError("completion requires CUDA state and rows within request capacity")
        fields = _completion_fields(records, device, state.capacity)
        for tensor, shape, dtype, name in ((row_slot_ids, (rows,), torch.int32, "row_slot_ids"),
                (row_generations, (rows,), torch.int64, "row_generations"),
                (live_rows, (1,), torch.int32, "live_rows"), (frame_index, (1,), torch.int32, "frame_index"),
                (submission_sequence, (1,), torch.int64, "submission_sequence")):
            _check_tensor(tensor, shape, dtype, device, name)
        self._args = (None, None, None, None, row_slot_ids, row_generations, live_rows, None,
            state.generation, state.consumed_tokens, state.generated_tokens, state.pending_token, state.status,
            state.error_code, state.rng_counter, state.histogram, state.max_new_tokens, state.eos_token_ids,
            state.eos_count, records, frame_index, submission_sequence,
            state.vocab_size, rows, state.consumed_tokens.stride(0), fields, False, state.capacity)
        self.capacity = state.capacity
        self._record = None

    def prepare(self):
        self._record = _compile(_complete, self._args, (self.capacity, 1, 1))

    def __call__(self):
        if self._record is None:
            raise RuntimeError("completion prepare() must finish before capture")
        self._record()


class PreparedSampling:
    """Captured FlashInfer sampling with explicit request-owned Philox state.

    Penalties and grammar precede temperature and FlashInfer top-k, softmax,
    and top-p. Both cutoff filters retain boundary ties. Min-p keeps equality
    against the largest remaining probability, then renormalizes. Temperature
    must be positive.

    Each row uses a separate FlashInfer call with device seed and
    offset=4*counter. CUDA's Philox subsequence is therefore zero regardless
    of packed row or physical slot. One word is drawn per generated token;
    the logical counter advances once. FlashInfer's temporary outputs remain
    inside the captured graph. ``scratch_tensors``
    enumerates mutable storage retained through graph replay.
    """

    @staticmethod
    def scratch_layout(rows, vocab):
        return dict(scores=((rows, vocab), torch.float32), maximum=((rows,), torch.float32),
            probabilities=((rows, vocab), torch.float32), probability_maximum=((rows,), torch.float32),
            probability_mass=((rows, 1), torch.float32), packed_top_k=((rows,), torch.int32),
            packed_top_p=((rows,), torch.float32), packed_min_p=((rows,), torch.float32),
            packed_seed=((rows,), torch.uint64), packed_offset=((rows,), torch.uint64),
            sampled_tokens=((rows,), torch.int32), sample_valid=((rows,), torch.bool))

    def __init__(self, logits, state: DeviceRequestState, row_slot_ids, row_generations, live_rows,
                 completion: CompletionOutputs, frame_index, submission_sequence, *, sample_enabled: torch.Tensor | None = None,
                 workspace=None):
        records = completion.records
        rows, vocab = logits.shape
        if not 1 <= rows <= state.capacity or vocab != state.vocab_size or logits.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("sampling logits require rows within request capacity and the state vocabulary in FP32 or BF16")
        device = logits.device
        if device.type != "cuda" or not logits.is_contiguous() or state.generation.device != device:
            raise ValueError("sampling requires contiguous CUDA logits on the state device")
        record_fields = _completion_fields(records, device, state.capacity)
        for tensor, shape, dtype, name in ((row_slot_ids, (rows,), torch.int32, "row_slot_ids"),
                (row_generations, (rows,), torch.int64, "row_generations"), (live_rows, (1,), torch.int32, "live_rows"),
                (frame_index, (1,), torch.int32, "frame_index"), (submission_sequence, (1,), torch.int64, "submission_sequence")):
            _check_tensor(tensor, shape, dtype, device, name)
        self.state, self.logits, self.slots = state, logits, row_slot_ids
        self.sample_enabled = torch.ones(rows, dtype=torch.int32, device=device) if sample_enabled is None else sample_enabled
        _check_tensor(self.sample_enabled, (rows,), torch.int32, device, "sample_enabled")
        self.scratch_tensors = bind_scratch(self.scratch_layout(rows, vocab), device=device, workspace=workspace)
        for name, tensor in self.scratch_tensors.items():
            setattr(self, name, tensor)
        self._rows = rows
        self._grid = (rows, triton.cdiv(vocab, 1024), 1)
        self._score_args = (logits, self.scores, row_slot_ids, row_generations, live_rows, state.generation,
            state.status, state.histogram, state.parameters, state.grammar_mask, state.grammar_enabled,
            vocab, state.grammar_mask.shape[1], 1024, state.capacity)
        from flashinfer import sampling
        self._flashinfer = sampling
        self._metadata_args = (row_slot_ids, row_generations, live_rows, state.generation, state.status,
            state.parameters, state.top_k, state.seed, state.rng_counter, self.packed_top_k,
            self.packed_top_p, self.packed_min_p, self.packed_seed, self.packed_offset, vocab, state.capacity)
        self._sanitize_args = (self.scores, self.maximum, vocab, 1024)
        self._normalize_args = (self.probabilities, self.probability_mass, vocab, 1024)
        # The top-p output address is graph-private and supplied during capture.
        self._min_p_kernel = None
        sampled, valid, mass = self.sampled_tokens, self.sample_valid, self.probability_mass
        self._complete_args = (self.maximum, sampled, valid, mass,
            row_slot_ids, row_generations, live_rows, self.sample_enabled,
            state.generation, state.consumed_tokens, state.generated_tokens, state.pending_token, state.status,
            state.error_code, state.rng_counter, state.histogram, state.max_new_tokens,
            state.eos_token_ids, state.eos_count, records, frame_index, submission_sequence,
            vocab, rows, state.consumed_tokens.stride(0), record_fields, True, state.capacity)
        self._logprob_args = (logits, completion, row_slot_ids, row_generations, live_rows, frame_index)
        self._prepared = False
        self.broadcast = None

    def prepare_broadcast(self, **kwargs):
        from .collectives import PreparedBroadcast

        self.broadcast = PreparedBroadcast(
            (self.sampled_tokens, self.sample_valid, self.maximum, self.probability_mass), **kwargs)

    def prepare(self) -> None:
        """Compile and warm all operators before CUDA graph capture."""
        self.logprobs = PreparedLogprobs(*self._logprob_args)
        self.logprobs.prepare()
        self._scores = _compile(_scores, self._score_args, self._grid)
        self._complete = _compile(_complete, self._complete_args, (self.state.capacity, 1, 1))
        self._record = _compile(_complete, (*self._complete_args[:-2], False, self.state.capacity), (self.state.capacity, 1, 1))
        self._metadata = _compile(_sampling_metadata, self._metadata_args, (self._rows, 1, 1))
        self._sanitize = _compile(_sanitize_scores, self._sanitize_args, self._grid)
        self._normalize = _compile(_normalize_probs, self._normalize_args, self._grid)
        from ignition.preparation.triton import warmup_triton

        self._min_p_kernel = warmup_triton(_mask_min_p,
            (self.probabilities, self.probabilities, self.probability_maximum,
             self.packed_min_p, self.state.vocab_size, 1024), self._grid,
            num_warps=4, enable_fp_fusion=False)
        # FlashInfer's public operators share these buffers on the model stream.
        self.scratch_tensors["flashinfer_top_k_row_states"] = self._flashinfer._get_cache_buf(
            f"top_k_mask_logits_row_states_{self.logits.device}", 1024 * 1024,
            self.logits.device, zero_init=True)
        self.scratch_tensors["flashinfer_softmax_workspace"] = self._flashinfer._get_cache_buf(
            "softmax_workspace", 1024 * 1024, self.logits.device)
        self.scratch_tensors["flashinfer_top_p_workspace"] = probability_filter_workspace(
            self._rows, self.state.vocab_size, self.logits.device)
        self._scores()
        self._filter_scores()
        self._prepared = True

    def _filter_scores(self):
        torch.amax(self.scores, dim=-1, out=self.maximum)
        self._metadata()
        self._sanitize()
        filter_probabilities_out(self.scores, self.packed_top_k, self.packed_top_p,
            self.scratch_tensors["flashinfer_top_k_row_states"],
            self.scratch_tensors["flashinfer_softmax_workspace"], self.probabilities,
            self.scratch_tensors["flashinfer_top_p_workspace"])
        probabilities = self.probabilities
        torch.amax(probabilities, dim=-1, out=self.probability_maximum)
        from ignition.preparation.triton import launch_triton

        launch_triton(_mask_min_p, self._min_p_kernel,
            (probabilities, self.probabilities, self.probability_maximum,
             self.packed_min_p, self.state.vocab_size, 1024), self._grid,
            num_warps=4, enable_fp_fusion=False)
        torch.sum(self.probabilities, dim=-1, keepdim=True, out=self.probability_mass)
        self._normalize()
        for row in range(self._rows):
            sampled, valid = sample_probabilities(self.probabilities[row:row + 1],
                seed=self.packed_seed[row:row + 1], offset=self.packed_offset[row:row + 1],
                return_valid=True)
            self.sampled_tokens[row:row + 1].copy_(sampled)
            self.sample_valid[row:row + 1].copy_(valid)

    def __call__(self) -> None:
        if not self._prepared:
            raise RuntimeError("sampling prepare() must complete before capture")
        self._scores()
        self._filter_scores()
        if self.broadcast is not None:
            self.broadcast()
        self._complete()
        self.logprobs()

    def record_completion(self) -> None:
        """Record a consumption-only prefill without a vocabulary projection."""
        if not self._prepared:
            raise RuntimeError("sampling prepare() must complete before capture")
        self._record()


@triton.jit
def _admission_validate(Count, Slots, Epochs, Params, TopK, Budget, Eos, EosCount, Histogram, GrammarEnabled,
                        Generation, Errors, V: tl.constexpr, CAPACITY: tl.constexpr, BLOCK: tl.constexpr, SLOTS: tl.constexpr = 4):
    row, block = tl.program_id(0), tl.program_id(1)
    count = tl.load(Count)
    if row < count:
        slot = tl.load(Slots + row)
        epoch = tl.load(Epochs + row)
        slot_valid = (slot >= 0) & (slot < SLOTS)
        scalar_valid = slot_valid & (count <= CAPACITY) & (epoch > 0)
        scalar_valid &= epoch > tl.load(Generation + slot, slot_valid, other=0)
        pindex = tl.arange(0, 8)
        params = tl.load(Params + row * 6 + pindex, pindex < 6, other=0.)
        finite = (params == params) & (tl.abs(params) != float("inf"))  # noqa: PLR0124
        scalar_valid &= tl.sum((~finite).to(tl.int32), 0) == 0
        temperature, top_p, min_p = tl.load(Params + row * 6), tl.load(Params + row * 6 + 1), tl.load(Params + row * 6 + 2)
        repetition = tl.load(Params + row * 6 + 3)
        scalar_valid &= (temperature > 0.) & (top_p > 0.) & (top_p <= 1.) & (min_p >= 0.) & (min_p <= 1.) & (repetition > 0.)
        scalar_valid &= (temperature >= 1.1754943508222875e-38)
        scalar_valid &= (top_p >= 1.1754943508222875e-38) & (repetition >= 1.1754943508222875e-38)
        k, budget = tl.load(TopK + row), tl.load(Budget + row)
        scalar_valid &= (k >= -1) & (k <= V) & (budget >= 0) & (budget <= 2147483647)
        eos_count = tl.load(EosCount + row)
        eos = tl.load(Eos + row * 8 + pindex)
        scalar_valid &= (eos_count >= 0) & (eos_count <= 8)
        scalar_valid &= tl.sum(((pindex < eos_count) & ((eos < 0) | (eos >= V))).to(tl.int32), 0) == 0
        enabled = tl.load(GrammarEnabled + row)
        scalar_valid &= (enabled == 0) | (enabled == 1)
        other_rows = tl.arange(0, SLOTS)
        other_slots = tl.load(Slots + other_rows)
        duplicates = tl.sum(((other_rows < count) & (other_rows < CAPACITY) & (other_slots == slot)).to(tl.int32), 0) > 1
        tokens = block * BLOCK + tl.arange(0, BLOCK)
        histogram = tl.load(Histogram + row * V + tokens, tokens < V, other=0)
        bad_histogram = tl.sum((histogram < 0).to(tl.int32), 0) > 0
        error = tl.where(~scalar_valid | bad_histogram, 1, 0) | tl.where(duplicates, 2, 0)
        tl.atomic_or(Errors + row, error, sem="relaxed")


@triton.jit
def _admission_copy(Count, Slots, Errors, InputHistogram, InputGrammar, Histogram, Grammar,
                    V: tl.constexpr, WORDS: tl.constexpr, BLOCK: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    if (row < tl.load(Count)) & (tl.load(Errors + row) == 0):
        slot = tl.load(Slots + row)
        index = block * BLOCK + tl.arange(0, BLOCK)
        histogram = tl.load(InputHistogram + row * V + index, index < V, other=0)
        tl.store(Histogram + slot * V + index, histogram, index < V)
        grammar = tl.load(InputGrammar + row * WORDS + index, index < WORDS, other=0)
        tl.store(Grammar + slot * WORDS + index, grammar, index < WORDS)


@triton.jit
def _admission_finish(Count, Slots, Epochs, InputSeed, InputParams, InputTopK, InputBudget, InputEos, InputEosCount,
                      InputGrammarEnabled, Errors, Generation, Seed, Counter, Params, TopK, Budget, Eos, EosCount,
                      GrammarEnabled, Generated, Pending, Status, Error, Consumed, Records, Frame, Sequence,
                      CAPACITY: tl.constexpr, CS: tl.constexpr, RECORD_FIELDS: tl.constexpr, SLOTS: tl.constexpr = 4):
    slot = tl.program_id(0)
    rows = tl.arange(0, SLOTS)
    matches = (rows < tl.load(Count)) & (rows < CAPACITY) & (tl.load(Slots + rows) == slot)
    active = tl.sum(matches.to(tl.int32), 0) > 0
    row = tl.min(tl.where(matches, rows, SLOTS), 0)
    epoch, consumed = tl.full((), 0, tl.int64), tl.full((), 0, tl.int64)
    status, code = tl.full((), 0, tl.int32), tl.full((), 0, tl.int32)
    if active:
        epoch = tl.load(Epochs + row)
        errors = tl.load(Errors + row)
        consumed = tl.load(Consumed + slot * CS)
        if errors == 0:
            fields = tl.arange(0, 8)
            parameters = tl.load(InputParams + row * 6 + fields, fields < 6, other=0.)
            tl.store(Params + slot * 6 + fields, parameters, fields < 6)
            tl.store(Eos + slot * 8 + fields, tl.load(InputEos + row * 8 + fields))
            tl.store(EosCount + slot, tl.load(InputEosCount + row))
            tl.store(Generation + slot, epoch)
            tl.store(Seed + slot, tl.load(InputSeed + row).to(tl.uint64, bitcast=True))
            tl.store(Counter + slot, 0)
            tl.store(TopK + slot, tl.load(InputTopK + row))
            budget = tl.load(InputBudget + row)
            tl.store(Budget + slot, budget)
            tl.store(GrammarEnabled + slot, tl.load(InputGrammarEnabled + row))
            tl.store(Generated + slot, 0)
            tl.store(Pending + slot, -1)
            status = tl.where(budget > 0, 1, 2)
            tl.store(Status + slot, status)
            tl.store(Error + slot, 0)
        else:
            status = 3
            code = tl.where(errors & 2 != 0, 4, 5)
            code = tl.where(errors >> 8 != 0, 64 + (errors >> 8), code)
    _write_completion(Records, Frame, Sequence, slot, epoch, consumed, 0,
                      -1, status, code, active, RECORD_FIELDS, SLOTS)


class PreparedAdmission:
    """Validate packed admissions, scatter session state, and finalize records.

    Validation precedes mutation. Invalid rows and duplicate physical slots
    leave resident session state unchanged and emit error completion records.
    Epochs must increase on slot reuse. Consumed counts and model state are owned by reset/restore and are untouched.
    """

    def __init__(self, state: DeviceRequestState, buffers: AdmissionBuffers,
                 completion: CompletionOutputs, frame_index, submission_sequence, *, capacity: int = 4):
        if capacity not in BATCH_CAPACITIES or capacity > state.capacity:
            raise ValueError("admission graph capacity must fit request storage")
        device, vocab = state.generation.device, state.vocab_size
        if device.type != "cuda":
            raise ValueError("prepared admission requires CUDA request state")
        slots = state.capacity
        shapes = ((1,), (slots,), (slots,), (slots,), (slots, 6), (slots,), (slots,), (slots, 8), (slots,),
                  (slots, vocab), (slots, triton.cdiv(vocab, 32)), (slots,))
        dtypes = (torch.int32, torch.int32, torch.int64, torch.int64, torch.float32, torch.int32, torch.int64,
                  torch.int64, torch.int32, torch.int32, torch.int32, torch.int32)
        for field, shape, dtype in zip(buffers.__dataclass_fields__, shapes, dtypes, strict=True):
            _check_tensor(getattr(buffers, field), shape, dtype, device, field)
        records = completion.records
        record_fields = _completion_fields(records, device, state.capacity)
        for tensor, shape, dtype, name in ((frame_index, (1,), torch.int32, "frame_index"), (submission_sequence, (1,), torch.int64, "submission_sequence")):
            _check_tensor(tensor, shape, dtype, device, name)
        self.state, self.buffers = state, buffers
        self.errors = torch.zeros(slots, dtype=torch.int32, device=device)
        b = buffers
        self._validate_args = (b.admission_count, b.admission_slots, b.admission_generations, b.admission_parameters,
            b.admission_top_k, b.admission_max_new_tokens, b.admission_eos_token_ids, b.admission_eos_count,
            b.admission_histogram, b.admission_grammar_enabled, state.generation, self.errors, vocab, capacity, 1024, slots)
        self._copy_args = (b.admission_count, b.admission_slots, self.errors, b.admission_histogram, b.admission_grammar_mask,
            state.histogram, state.grammar_mask, vocab, state.grammar_mask.shape[1], 1024)
        self._finish_args = (b.admission_count, b.admission_slots, b.admission_generations, b.admission_seed_bits,
            b.admission_parameters, b.admission_top_k, b.admission_max_new_tokens, b.admission_eos_token_ids,
            b.admission_eos_count, b.admission_grammar_enabled, self.errors, state.generation, state.seed, state.rng_counter,
            state.parameters, state.top_k, state.max_new_tokens, state.eos_token_ids, state.eos_count, state.grammar_enabled,
            state.generated_tokens, state.pending_token, state.status, state.error_code, state.consumed_tokens,
            records, frame_index, submission_sequence, capacity, state.consumed_tokens.stride(0), record_fields, slots)
        self._grid = (capacity, triton.cdiv(vocab, 1024), 1)
        self._prepared = False

    def prepare(self) -> None:
        self._validate = _compile(_admission_validate, self._validate_args, self._grid)
        self._copy = _compile(_admission_copy, self._copy_args, self._grid)
        self._finish = _compile(_admission_finish, self._finish_args, (self.state.capacity, 1, 1))
        self._prepared = True

    def __call__(self) -> None:
        self.validate()
        self.apply()

    def validate(self) -> None:
        """Validate staging before a composed hybrid-state reset/restore."""
        if not self._prepared:
            raise RuntimeError("admission prepare() must complete before capture")
        self.errors.zero_()
        self._validate()

    def apply(self) -> None:
        """Apply validated rows; composed maintenance may add transfer errors.

        Bits8.. encode state-transfer errors. Nonzero error rows never modify
        session state and finalize error_code=64+transfer_error.
        """
        if not self._prepared:
            raise RuntimeError("admission prepare() must complete before capture")
        self._copy()
        self._finish()
