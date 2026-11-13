# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Captured vision staging, feature publication, and decoder embedding injection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from ignition.components.sampling import _completion_fields, _write_completion
from ignition.preparation.triton import prepare_triton

from .sampling import DeviceRequestState
from .vision import VisionMetadata

FEATURES = 16384
WIDTH = 2560
PATCHES = 16384
VISION_ERROR_BASE = 128
INJECTION_ERROR_BASE = 256

_JIT_FEATURES = tl.constexpr(FEATURES)
_JIT_WIDTH = tl.constexpr(WIDTH)
_JIT_PATCHES = tl.constexpr(PATCHES)
_JIT_VISION_ERROR_BASE = tl.constexpr(VISION_ERROR_BASE)
_JIT_INJECTION_ERROR_BASE = tl.constexpr(INJECTION_ERROR_BASE)


@dataclass(frozen=True)
class VisionFeatureState:
    """Per-request feature storage; only rows below count in its epoch are valid."""

    arena: torch.Tensor
    generation: torch.Tensor
    count: torch.Tensor
    feature_slot: torch.Tensor

    @classmethod
    def allocate(cls, device, *, capacity=4, image_slots=None, feature_slot=None):
        from ignition.components.capacity import image_capacity

        slots = image_capacity(capacity, image_slots)
        return cls(
            torch.empty((slots, FEATURES, WIDTH), dtype=torch.bfloat16, device=device),
            torch.zeros(capacity, dtype=torch.int64, device=device),
            torch.zeros(capacity, dtype=torch.int32, device=device),
            torch.full((1,), -1, dtype=torch.int32, device=device) if feature_slot is None else feature_slot,
        )


@dataclass(frozen=True)
class VisionInputs:
    vision_pixel_values: torch.Tensor
    vision_interpolation_indices: torch.Tensor
    vision_interpolation_weights: torch.Tensor
    vision_position_ids: torch.Tensor
    vision_cu_seqlens: torch.Tensor
    vision_num_patches: torch.Tensor
    vision_slot: torch.Tensor
    vision_generation: torch.Tensor
    vision_feature_offset: torch.Tensor

    @classmethod
    def allocate(cls, device):
        def zeros(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=device)

        return cls(
            zeros((PATCHES, 1536), torch.float32),
            zeros((PATCHES, 4), torch.int64),
            zeros((PATCHES, 4), torch.float32),
            zeros((PATCHES, 2), torch.int64),
            zeros(2, torch.int32),
            zeros(1, torch.int32),
            torch.full((1,), -1, dtype=torch.int32, device=device),
            zeros(1, torch.int64),
            zeros(1, torch.int32),
        )

    def buffers(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _require(tensor, shape, dtype, device, name):
    if (
        tensor.shape != shape
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_contiguous()
    ):
        raise ValueError(f"{name} requires contiguous {dtype} {shape} on {device}")


def _compile(kernel, args, grid):
    return prepare_triton(kernel, args, grid, num_warps=4)


@triton.jit
def _vision_validate(
    Pixels,
    Indices,
    Weights,
    Positions,
    Cu,
    Num,
    Slot,
    Epoch,
    Offset,
    Generation,
    Status,
    FeatureGeneration,
    FeatureCount,
    FeatureSlot,
    Errors,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
    SLOTS: tl.constexpr = 4,
    FEATURE_SLOTS: tl.constexpr = 4,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot, epoch, count, offset = (
        tl.load(Slot),
        tl.load(Epoch),
        tl.load(Num),
        tl.load(Offset),
    )
    active = (slot != -1) | (count != 0)
    slot_valid = (slot >= 0) & (slot < SLOTS)
    epoch_valid = (epoch > 0) & (
        tl.load(Generation + slot, slot_valid, other=0) == epoch
    )
    running = tl.load(Status + slot, slot_valid, other=0) == 1
    feature_slot = tl.load(FeatureSlot)
    feature_valid = (feature_slot >= 0) & (feature_slot < FEATURE_SLOTS)
    error = tl.where(active & (~slot_valid | ~feature_valid), 1, 0)
    error |= tl.where(active & (~epoch_valid | ~running), 2, 0)
    geometry = (count > 0) & (count <= CAPACITY) & (count % 4 == 0)
    geometry &= (tl.load(Cu) == 0) & (tl.load(Cu + 1) == count)
    error |= tl.where(active & ~geometry, 4, 0)
    same_features = tl.load(FeatureGeneration + slot, slot_valid, other=0) == epoch
    previous_count = tl.load(FeatureCount + slot, slot_valid, other=0)
    append = offset == tl.where(same_features, previous_count, 0)
    append &= (offset >= 0) & (offset + count // 4 <= _JIT_FEATURES)
    error |= tl.where(active & ~append, 8, 0)
    live_patches = tl.minimum(tl.maximum(count, 0), CAPACITY)
    metadata_mask = active & (index < live_patches * 4)
    interpolation = tl.load(Indices + index, metadata_mask, other=0)
    weight = tl.load(Weights + index, metadata_mask, other=0.0)
    bad_metadata = metadata_mask & ((interpolation < 0) | (interpolation >= 2304))
    bad_metadata |= metadata_mask & (~((weight >= 0.0) & (weight <= 1.0)))
    position_mask = active & (index < live_patches * 2)
    position = tl.load(Positions + index, position_mask, other=0)
    bad_metadata |= position_mask & ((position < 0) | (position >= _JIT_PATCHES))
    error |= tl.where(tl.sum(bad_metadata.to(tl.int32), 0) > 0, 16, 0)
    pixel_mask = active & (index < live_patches * 1536)
    pixel = tl.load(Pixels + index, pixel_mask, other=0.0)
    bad_pixel = pixel_mask & ~(tl.abs(pixel) < float("inf"))
    error |= tl.where(tl.sum(bad_pixel.to(tl.int32), 0) > 0, 32, 0)
    if error != 0:
        tl.atomic_or(Errors, error, sem="relaxed")


@triton.jit
def _vision_sanitize(
    Pixels,
    Indices,
    Weights,
    Positions,
    Num,
    Errors,
    SafePixels,
    SafeIndices,
    SafeWeights,
    SafePositions,
    SafeCu,
    SafeNum,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    count = tl.where(tl.load(Errors) == 0, tl.load(Num), 0)
    pixel = tl.load(Pixels + index, index < count * 1536, other=0.0)
    tl.store(SafePixels + index, pixel, index < CAPACITY * 1536)
    value = tl.load(Indices + index, index < count * 4, other=0)
    tl.store(SafeIndices + index, value, index < CAPACITY * 4)
    weight = tl.load(Weights + index, index < count * 4, other=0.0)
    tl.store(SafeWeights + index, weight, index < CAPACITY * 4)
    position = tl.load(Positions + index, index < count * 2, other=0)
    tl.store(SafePositions + index, position, index < CAPACITY * 2)
    if tl.program_id(0) == 0:
        tl.store(SafeNum, count)
        tl.store(SafeCu, 0)
        tl.store(SafeCu + 1, count)


@triton.jit
def _feature_validate(Output, Num, Errors, CAPACITY: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    count = tl.minimum(tl.maximum(tl.load(Num), 0), CAPACITY) // 4
    valid = (tl.load(Errors) == 0) & (index < count * _JIT_WIDTH)
    value = tl.load(Output + index, valid, other=0.0).to(tl.float32)
    bad = valid & ~(tl.abs(value) < float("inf"))
    if tl.sum(bad.to(tl.int32), 0) > 0:
        tl.atomic_or(Errors, 64, sem="relaxed")


@triton.jit
def _feature_copy(
    Output,
    Arena,
    Num,
    Slot,
    Offset,
    Errors,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
    SLOTS: tl.constexpr = 4,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    slot, count, offset = (
        tl.load(Slot).to(tl.int64),
        tl.load(Num),
        tl.load(Offset).to(tl.int64),
    )
    valid = (
        (tl.load(Errors) == 0)
        & (slot >= 0)
        & (slot < SLOTS)
        & (index < count // 4 * _JIT_WIDTH)
    )
    value = tl.load(Output + index, valid, other=0)
    tl.store(Arena + (slot * _JIT_FEATURES + offset) * _JIT_WIDTH + index, value, valid)


@triton.jit
def _vision_finish(
    Slot,
    Epoch,
    Num,
    Offset,
    Errors,
    FeatureGeneration,
    FeatureCount,
    Generation,
    Consumed,
    Generated,
    Status,
    StateError,
    Records,
    Frame,
    Sequence,
    CS: tl.constexpr,
    RECORD_FIELDS: tl.constexpr,
    SLOTS: tl.constexpr = 4,
):
    slot = tl.program_id(0)
    active = tl.load(Slot) == slot
    epoch = tl.load(Epoch)
    current = active & (epoch == tl.load(Generation + slot))
    error = tl.load(Errors)
    status, code = tl.full((), 0, tl.int32), tl.full((), 0, tl.int32)
    if active:
        if error == 0:
            tl.store(FeatureGeneration + slot, epoch)
            tl.store(FeatureCount + slot, tl.load(Offset) + tl.load(Num) // 4)
            status = 1
        else:
            status, code = 3, _JIT_VISION_ERROR_BASE + error
            if current & (tl.load(Status + slot) == 1):
                tl.store(Status + slot, status)
                tl.store(StateError + slot, code)
    consumed = tl.load(Consumed + slot * CS, current, other=0)
    generated = tl.load(Generated + slot, current, other=0)
    _write_completion(Records, Frame, Sequence, slot, epoch, consumed, generated,
                      -1, status, code, active, RECORD_FIELDS, SLOTS)


class PreparedVisionHandoff:
    """Validate/sanitize before encoding, then publish complete image features.

    Call ``prepare()`` before capture. Record ``begin()``, the encoder over
    ``pixels`` and ``metadata``, and ``finish()`` in that order. Invalid input
    sanitizes every interpolation index and position as well as counts/pixels;
    no malformed indexing payload reaches the encoder. Publication appends to
    the feature count and never consumes tokens or samples. All graphs sharing
    this arena and request state execute in one ordered stream.

    Error bits: 1 slot, 2 epoch/status, 4 geometry, 8 feature append/bounds,
    16 interpolation/position metadata, 32 nonfinite pixels, 64 nonfinite output.
    Stale epochs do not alter a reused request's resident state. Slot -1 with
    zero patches is inactive. Native admission must reject other invalid slots
    because no physical completion lane can represent them.
    """

    def __init__(
        self,
        inputs: VisionInputs,
        features: VisionFeatureState,
        requests: DeviceRequestState,
        output: torch.Tensor,
        completion,
        frame_index,
        submission_sequence,
        *,
        pixels,
    ):
        device = features.arena.device
        if device.type != "cuda" or output.ndim != 2 or output.shape[1] != WIDTH:
            raise ValueError("vision handoff requires CUDA BF16 [features,2560] output")
        capacity = output.shape[0] * 4
        if capacity not in (256, 1024, 4096, PATCHES):
            raise ValueError(
                "vision handoff patch bucket must be 256, 1024, 4096 or 16384"
            )
        _require(
            features.arena,
            (features.arena.shape[0], FEATURES, WIDTH),
            torch.bfloat16,
            device,
            "feature arena",
        )
        if not 1 <= features.arena.shape[0] <= requests.capacity:
            raise ValueError("feature slots must fit within request capacity")
        _require(features.feature_slot, (1,), torch.int32, device, "feature slot")
        _require(features.generation, (requests.capacity,), torch.int64, device, "feature generation")
        _require(features.count, (requests.capacity,), torch.int32, device, "feature count")
        _require(
            output, (capacity // 4, WIDTH), torch.bfloat16, device, "vision output"
        )
        _require(pixels, (capacity, 1536), torch.bfloat16, device, "vision patch staging")
        shapes = (
            ((PATCHES, 1536), torch.float32),
            ((PATCHES, 4), torch.int64),
            ((PATCHES, 4), torch.float32),
            ((PATCHES, 2), torch.int64),
            ((2,), torch.int32),
            ((1,), torch.int32),
            ((1,), torch.int32),
            ((1,), torch.int64),
            ((1,), torch.int32),
        )
        for (name, tensor), (shape, dtype) in zip(
            inputs.buffers().items(), shapes, strict=True
        ):
            _require(tensor, shape, dtype, device, name)
        records = completion.records
        record_fields = _completion_fields(records, device, requests.capacity)
        for tensor, shape, dtype, name in (
            (frame_index, (1,), torch.int32, "frame index"),
            (submission_sequence, (1,), torch.int64, "submission sequence"),
        ):
            _require(tensor, shape, dtype, device, name)
        self.inputs, self.features, self.requests, self.output = (
            inputs,
            features,
            requests,
            output,
        )
        self.metadata = VisionMetadata.allocate(capacity, 1, device)
        def zeros(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=device)
        self.pixels = pixels
        self.errors = zeros((1,), torch.int32)
        self._prepared = False
        i, m, f, r = inputs, self.metadata, features, requests
        grid = (triton.cdiv(capacity * 1536, 1024), 1, 1)
        self._kernels = [
            (
                _vision_validate,
                (
                    i.vision_pixel_values,
                    i.vision_interpolation_indices,
                    i.vision_interpolation_weights,
                    i.vision_position_ids,
                    i.vision_cu_seqlens,
                    i.vision_num_patches,
                    i.vision_slot,
                    i.vision_generation,
                    i.vision_feature_offset,
                    r.generation,
                    r.status,
                    f.generation,
                    f.count,
                    f.feature_slot,
                    self.errors,
                    capacity,
                    1024,
                ),
                grid,
            ),
            (
                _vision_sanitize,
                (
                    i.vision_pixel_values,
                    i.vision_interpolation_indices,
                    i.vision_interpolation_weights,
                    i.vision_position_ids,
                    i.vision_num_patches,
                    self.errors,
                    self.pixels,
                    m.interpolation_indices,
                    m.interpolation_weights,
                    m.position_ids,
                    m.cu_seqlens,
                    m.num_patches,
                    capacity,
                    1024,
                ),
                grid,
            ),
            (
                _feature_validate,
                (output, i.vision_num_patches, self.errors, capacity, 1024),
                (triton.cdiv(capacity // 4 * WIDTH, 1024), 1, 1),
            ),
            (
                _feature_copy,
                (
                    output,
                    f.arena,
                    i.vision_num_patches,
                    f.feature_slot,
                    i.vision_feature_offset,
                    self.errors,
                    capacity,
                    1024,
                ),
                (triton.cdiv(capacity // 4 * WIDTH, 1024), 1, 1),
            ),
            (
                _vision_finish,
                (
                    i.vision_slot,
                    i.vision_generation,
                    i.vision_num_patches,
                    i.vision_feature_offset,
                    self.errors,
                    f.generation,
                    f.count,
                    r.generation,
                    r.consumed_tokens,
                    r.generated_tokens,
                    r.status,
                    r.error_code,
                    records,
                    frame_index,
                    submission_sequence,
                    r.consumed_tokens.stride(0),
                    record_fields,
                ),
                (requests.capacity, 1, 1),
            ),
        ]

    def prepare(self):
        self._calls = tuple(_compile(kernel, (*args, self.requests.capacity, self.features.arena.shape[0]) if kernel is _vision_validate
            else (*args, self.features.arena.shape[0]) if kernel is _feature_copy
            else (*args, self.requests.capacity) if kernel is _vision_finish else args, grid)
            for kernel, args, grid in self._kernels)
        self._prepared = True

    def functional(self):
        self.capacity = self.pixels.shape[0]
        values = {"errors": self.errors, "pixels": self.pixels, "output": self.output,
                  **{name: getattr(self.metadata, name) for name in self.metadata.__dataclass_fields__}}
        self._dynamic_calls = []
        for call in self._calls:
            slots = {index: name for index, value in enumerate(call.args)
                     for name, candidate in values.items() if value is candidate}
            args = tuple(None if index in slots else value for index, value in enumerate(call.args))
            self._dynamic_calls.append((call.kernel, call.compiled, args, call.grid, call.options, slots))
            call.args = ()
        self._kernels = self._calls = None
        self.output = self.pixels = self.metadata = self.errors = None
        return self

    def _run_dynamic(self, index, values):
        from ignition.preparation.triton import launch_triton

        kernel, compiled, args, grid, options, slots = self._dynamic_calls[index]
        args = tuple(values[slots[i]] if i in slots else value for i, value in enumerate(args))
        launch_triton(kernel, compiled, args, grid, **options)

    def begin(self):
        if not self._prepared:
            raise RuntimeError("vision handoff prepare() must complete before capture")
        device = self.features.arena.device
        metadata = VisionMetadata.allocate(self.capacity, 1, device)
        values = {"errors": torch.zeros((1,), dtype=torch.int32, device=device),
                  "pixels": torch.empty((self.capacity, 1536), dtype=torch.bfloat16, device=device),
                  **{name: getattr(metadata, name) for name in metadata.__dataclass_fields__}}
        self._run_dynamic(0, values)
        self._run_dynamic(1, values)
        return values, metadata

    def finish(self, values, output):
        if not self._prepared:
            raise RuntimeError("vision handoff prepare() must complete before capture")
        values["output"] = output
        for index in range(2, len(self._dynamic_calls)):
            self._run_dynamic(index, values)


class PreparedVisionGeneration:
    def __init__(self, handoff, encoder):
        self.handoff, self.encoder = handoff, encoder

    def __call__(self):
        values, metadata = self.handoff.begin()
        output = self.encoder(values["pixels"], metadata)
        self.handoff.finish(values, output)


@triton.jit
def _injection_validate(
    Indices,
    Slots,
    Epochs,
    Lengths,
    Live,
    Generation,
    Status,
    StateError,
    FeatureGeneration,
    FeatureCount,
    FeatureSlot,
    Errors,
    TOKENS: tl.constexpr,
    BLOCK: tl.constexpr,
    SLOTS: tl.constexpr = 4,
    FEATURE_SLOTS: tl.constexpr = 4,
):
    slot, epoch, length, live = (
        tl.load(Slots),
        tl.load(Epochs),
        tl.load(Lengths),
        tl.load(Live),
    )
    physical = (slot >= 0) & (slot < SLOTS)
    current = (
        physical
        & (tl.load(Generation + slot, physical, other=0) == epoch)
        & (epoch > 0)
    )
    running = tl.load(Status + slot, physical, other=0) == 1
    active = (live > 0) & current & running
    geometry = (live == 1) & (length > 0) & (length <= TOKENS)
    index = tl.arange(0, BLOCK)
    rows = active & (index < length) & (index < TOKENS)
    feature = tl.load(Indices + index, rows, other=-1)
    visual = rows & (feature >= 0)
    count = tl.load(FeatureCount + slot, physical, other=0)
    same_epoch = tl.load(FeatureGeneration + slot, physical, other=0) == epoch
    invalid = rows & (feature < -1)
    feature_slot = tl.load(FeatureSlot)
    invalid |= visual & ((feature >= _JIT_FEATURES) | (feature >= count) | ~same_epoch
                         | (feature_slot < 0) | (feature_slot >= FEATURE_SLOTS))
    error = tl.where(active & ~geometry, 1, 0)
    error |= tl.where(tl.sum(invalid.to(tl.int32), 0) > 0, 2, 0)
    tl.store(Errors, error)
    if active & (error != 0):
        tl.store(Status + slot, 3)
        tl.store(StateError + slot, _JIT_INJECTION_ERROR_BASE + error)


@triton.jit
def _inject(
    Embedding,
    Arena,
    Indices,
    Slots,
    Epochs,
    Lengths,
    Live,
    Generation,
    Status,
    FeatureGeneration,
    FeatureCount,
    FeatureSlot,
    Errors,
    TOKENS: tl.constexpr,
    BLOCK: tl.constexpr,
    SLOTS: tl.constexpr = 4,
    FEATURE_SLOTS: tl.constexpr = 4,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    slot, epoch = tl.load(Slots), tl.load(Epochs)
    physical = (slot >= 0) & (slot < SLOTS)
    valid = (
        physical
        & (tl.load(Live) == 1)
        & (tl.load(Errors) == 0)
        & (row < tl.load(Lengths))
    )
    valid &= tl.load(Generation + slot, physical, other=0) == epoch
    valid &= tl.load(Status + slot, physical, other=0) == 1
    valid &= tl.load(FeatureGeneration + slot, physical, other=0) == epoch
    feature = tl.load(Indices + row)
    valid &= (
        (feature >= 0)
        & (feature < tl.load(FeatureCount + slot, physical, other=0))
        & (feature < _JIT_FEATURES)
    )
    feature_slot = tl.load(FeatureSlot)
    valid &= (feature_slot >= 0) & (feature_slot < FEATURE_SLOTS)
    channel = tile * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(
        Arena
        + (feature_slot.to(tl.int64) * _JIT_FEATURES + feature.to(tl.int64)) * _JIT_WIDTH
        + channel,
        valid & (channel < _JIT_WIDTH),
        other=0,
    )
    tl.store(
        Embedding + row * _JIT_WIDTH + channel, value, valid & (channel < _JIT_WIDTH)
    )


class PreparedEmbeddingInjection:
    """Validate one packed prefill request before decoder-state gathering.

    Call ``prepare()`` before capture. Record ``validate()`` before
    ``PreparedDecoderState.begin_runtime()``, then ``inject()`` after token embedding and
    before hyperconnection broadcast. Invalid visual indices set the matching
    running request's error status, which the subsequent gather excludes from
    every model-state update. Stale generations leave reused slots unchanged.
    Text rows use -1 and do not require a feature epoch. No token is consumed.
    """

    def __init__(
        self,
        features: VisionFeatureState,
        requests: DeviceRequestState,
        vision_token_indices,
        row_slot_ids,
        row_generations,
        query_lengths,
        live_rows,
        embedding,
        *,
        rows: int = 1,
    ):
        device = features.arena.device
        if (
            rows != 1
            or device.type != "cuda"
            or embedding.ndim != 2
            or embedding.shape[1] != WIDTH
        ):
            raise ValueError(
                "vision embedding injection requires one CUDA prefill request"
            )
        tokens = embedding.shape[0]
        if tokens <= 0:
            raise ValueError("vision embedding injection requires a positive prefill capacity")
        _require(embedding, (tokens, WIDTH), torch.bfloat16, device, "embedding")
        _require(
            vision_token_indices, (tokens,), torch.int32, device, "vision token indices"
        )
        for tensor, dtype, name in (
            (row_slot_ids, torch.int32, "row slots"),
            (row_generations, torch.int64, "row generations"),
            (query_lengths, torch.int32, "query lengths"),
        ):
            if tensor.ndim != 1 or tensor.numel() < 1:
                raise ValueError(f"{name} must hold the prefill request")
            _require(tensor, tensor.shape, dtype, device, name)
        _require(live_rows, (1,), torch.int32, device, "live rows")
        self.features, self.requests, self.embedding = features, requests, embedding
        self.errors = torch.zeros(1, dtype=torch.int32, device=device)
        self._prepared = False
        f, r = features, requests
        self._validate_args = (
            vision_token_indices,
            row_slot_ids,
            row_generations,
            query_lengths,
            live_rows,
            r.generation,
            r.status,
            r.error_code,
            f.generation,
            f.count,
            f.feature_slot,
            self.errors,
            tokens,
            triton.next_power_of_2(tokens),
            requests.capacity,
            f.arena.shape[0],
        )
        self._inject_args = (
            embedding,
            f.arena,
            vision_token_indices,
            row_slot_ids,
            row_generations,
            query_lengths,
            live_rows,
            r.generation,
            r.status,
            f.generation,
            f.count,
            f.feature_slot,
            self.errors,
            tokens,
            512,
            requests.capacity,
            f.arena.shape[0],
        )
        self._grid = (tokens, triton.cdiv(WIDTH, 512), 1)

    def prepare(self):
        self._validate = _compile(_injection_validate, self._validate_args, (1, 1, 1))
        self._inject = _compile(_inject, self._inject_args, self._grid)
        self._prepared = True

    def functional(self, *, dynamic_inputs=False):
        validate_slots = {11} | (set(range(5)) if dynamic_inputs else set())
        inject_slots = {0, 12} | (set(range(2, 7)) if dynamic_inputs else set())
        self._runtime_validate_args = tuple(None if index in validate_slots else value
                                            for index, value in enumerate(self._validate_args))
        self._runtime_inject_args = tuple(None if index in inject_slots else value
                                          for index, value in enumerate(self._inject_args))
        self.dynamic_inputs = dynamic_inputs
        self._validate_args = self._inject_args = None
        self._validate.args = self._inject.args = ()
        self.errors = self.embedding = None
        return self

    def validate(self, inputs=None):
        if not self._prepared:
            raise RuntimeError("vision injection prepare() must complete before capture")
        from ignition.preparation.triton import launch_triton
        errors = torch.empty((1,), dtype=torch.int32, device=self.features.arena.device)
        args = list(self._runtime_validate_args)
        args[11] = errors
        if self.dynamic_inputs:
            if inputs is None:
                raise ValueError("vision injection requires runtime role inputs")
            args[:5] = (inputs.vision_token_indices, inputs.row_slot_ids, inputs.row_generations,
                        inputs.query_lengths, inputs.live_rows)
        launch_triton(self._validate.kernel, self._validate.compiled, tuple(args), self._validate.grid,
                      **self._validate.options)
        return errors

    def inject(self, embedding, errors, inputs=None):
        if not self._prepared:
            raise RuntimeError("vision injection prepare() must complete before capture")
        from ignition.preparation.triton import launch_triton
        args = list(self._runtime_inject_args)
        args[0], args[12] = embedding, errors
        if self.dynamic_inputs:
            if inputs is None:
                raise ValueError("vision injection requires runtime role inputs")
            args[2:7] = (inputs.vision_token_indices, inputs.row_slot_ids, inputs.row_generations,
                         inputs.query_lengths, inputs.live_rows)
        launch_triton(self._inject.kernel, self._inject.compiled, tuple(args), self._inject.grid,
                      **self._inject.options)


__all__ = [
    "PreparedEmbeddingInjection",
    "PreparedVisionHandoff",
    "VisionFeatureState",
    "VisionInputs",
]
