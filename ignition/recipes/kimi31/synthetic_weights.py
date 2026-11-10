# SPDX-License-Identifier: Apache-2.0

"""Executable synthetic Kimi weights with checkpoint geometry but no payload reads."""

from contextlib import ExitStack
import math
import zlib

import torch

from ignition.components.weights import DirectWeightLoader
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.recipes.kimi3.weights import KimiWeightLoader


SYNTHETIC_SCHEMA = "kimi31-dflash2-synthetic-v2"
_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "U8": torch.uint8}


def _is_norm(name):
    return "norm" in name.rsplit("/", 1)[-1].lower()


def _write_scales(destination, name, seed):
    """Generate native E8M0 scales with a measured, nondegenerate CSF exception density."""
    if destination.ndim < 2 or destination.shape[-1] < 8 or not destination.is_contiguous():
        destination.fill_(127)
        return
    columns = destination.shape[-1]
    flat = destination.view(-1, columns)
    salt = (seed + zlib.crc32(name.encode())) % 1000
    rows_per_chunk = max(1, (1 << 20) // columns)
    column = torch.arange(columns, device=destination.device)
    for first in range(0, flat.shape[0], rows_per_chunk):
        last = min(first + rows_per_chunk, flat.shape[0])
        rows = torch.arange(first, last, dtype=torch.int64, device=destination.device)
        base = 126 + (rows + salt).remainder(2)
        flat[first:last].copy_((base[:, None] + column.remainder(2)).to(torch.uint8))
        selector = (rows * 17 + salt).remainder(1000)
        exceptions = torch.zeros_like(selector)
        exceptions += (selector >= 708)
        exceptions += (selector >= 940)
        exceptions += (selector >= 987)
        exceptions += (selector >= 997)
        start = (rows + salt).remainder(columns)
        for ordinal in range(4):
            selected = exceptions > ordinal
            indices = selected.nonzero().squeeze(1)
            position = (start[indices] + ordinal * 37).remainder(columns)
            flat[first:last][indices, position] = (base[indices] - 2).to(torch.uint8)


def _write(destination, name, seed=0):
    """Fill one destination on its device without materializing a full host tensor."""
    if destination.dtype == torch.uint8:
        # Native MXFP4 payloads use small finite nibbles. Two-value E8M0 windows
        # carry exceptions at the measured aggregate density of real Kimi scales.
        if "scale" in name:
            _write_scales(destination, name, seed)
        else:
            destination.fill_(0x11)
        return
    if destination.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"synthetic tensor {name} has unsupported dtype {destination.dtype}")
    if "scale" in name:
        destination.fill_(1.0)
        return
    if name.endswith(".A_log"):
        destination.fill_(-1.0)
        return
    if _is_norm(name):
        destination.fill_(1.0)
        return
    if "bias" in name:
        destination.zero_()
        return
    if destination.ndim == 0:
        destination.fill_(0.001)
        return
    row_values = max(1, destination[0].numel())
    rows_per_chunk = max(1, (1 << 20) // row_values)
    salt = (seed + zlib.crc32(name.encode())) % 31
    for first in range(0, destination.shape[0], rows_per_chunk):
        last = min(first + rows_per_chunk, destination.shape[0])
        rows = torch.arange(first, last, dtype=torch.float32, device=destination.device)
        values = ((rows + salt).remainder(31) - 15) * 0.0001
        destination[first:last].copy_(values.view(-1, *([1] * (destination.ndim - 1))))


class _SyntheticSession:
    """Only the flush boundary used by DirectWeightLoader's tensor helpers."""

    payload_bytes = 0

    def flush(self):
        return None


class SyntheticWeightLoader:
    """Replace source views and transfers while retaining the recipe's layout logic."""

    def _initialize_synthetic(self, seed):
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("synthetic seed must be a uint63 integer")
        self.synthetic_seed = seed
        self._source_names = {}

    def __enter__(self):
        if self._stack is not None:
            raise RuntimeError("weight loader context is already open")
        device = torch.device(self.device)
        if device.type != "cuda":
            raise ValueError(f"{type(self).__name__} requires a CUDA device")
        self.device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        selected = self._selected_names()
        sources = {}
        for name in sorted(selected):
            entry = self.manifest.tensors[name]
            if entry.dtype not in _DTYPES:
                raise ValueError(f"{name}: unsupported synthetic source dtype {entry.dtype}")
            dtype = _DTYPES[entry.dtype]
            # Scalar calibration consumers call item() and finite checks before transfer.
            device = "cpu" if math.prod(entry.shape) == 1 else "meta"
            sources[name] = torch.empty(entry.shape, dtype=dtype, device=device)
            if device == "cpu":
                _write(sources[name], name, self.synthetic_seed)
        self._sources = sources
        self._source_names = {value.untyped_storage()._cdata: name for name, value in sources.items()}
        self._session = _SyntheticSession()
        self._stack = ExitStack()
        self._stack.callback(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        return self

    def __exit__(self, *exc):
        try:
            if self._stack is not None:
                self._stack.__exit__(*exc)
        finally:
            self._stack = None
            self._session = None
            self._sources.clear()
            self._source_names.clear()
        return False

    def _source_name(self, source):
        if not isinstance(source, torch.Tensor):
            raise ValueError("synthetic transfer source is not a tensor")
        return self._source_names.get(source.untyped_storage()._cdata)

    def _copy(self, destination, source, *, numeric=True):
        name = self._source_name(source)
        if name is None:
            # Small computed CPU vectors in the expert loader contain generated scales,
            # not checkpoint data. Large or device-resident unknown sources are rejected.
            if source.device.type != "cpu" or source.numel() > 4096:
                raise ValueError("synthetic transfer has no generated source identity")
            destination.copy_(source)
        else:
            _write(destination, name, self.synthetic_seed)
        if numeric:
            record_numeric_tensor(destination)

    def _queue_tensor(self, name, *, dtype=None, extent=None):
        self._require_open()
        source = self._sources[name]
        if extent is not None:
            source = source[extent]
        target = torch.empty(source.shape, dtype=source.dtype if dtype is None else dtype, device=self.device)
        _write(target, name, self.synthetic_seed)
        record_numeric_tensor(target)
        return target


class SyntheticKimiWeightLoader(SyntheticWeightLoader, KimiWeightLoader):
    """Kimi target loader supporting native MXFP4 scales compressed by b12x."""

    def __init__(self, *args, synthetic_seed, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_synthetic(synthetic_seed)

    def load_fp4_experts(self, *args, **kwargs):
        if (kwargs.get("format") != "mxfp4" or kwargs.get("scale_compression") != "csf"
                or kwargs.get("shared_columns") is not None):
            raise ValueError("synthetic Kimi experts require native MXFP4, direct CSF compression and no shared reads")
        return super().load_fp4_experts(*args, **kwargs)

    def load_mxfp4_csf_experts(self, *args, **kwargs):
        raise ValueError("synthetic CSF source streams are not supported; use native MXFP4 checkpoint geometry")


class SyntheticDraftWeightLoader(SyntheticWeightLoader, DirectWeightLoader):
    """Metadata-only source loader for the BF16 DFlash2 draft."""

    def __init__(self, *args, synthetic_seed, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_synthetic(synthetic_seed)
