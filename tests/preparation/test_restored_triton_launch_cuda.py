# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The GDN prefill state initializer restores fixed launches before graph replay."""

import os

import pytest
import torch
import triton
from triton.runtime.jit import JITFunction

from ignition.components.gdn_prefill_sm103 import (
    _ZERO_BLOCK,
    _ZERO_CONFIG,
    _zero_fresh_rows,
)
from ignition.preparation.aot import aot_forward
from ignition.preparation.device_binding import bind_device
from ignition.preparation.triton import PreparedTriton, warmup_triton

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_CUDA_TESTS=1 and an explicitly assigned CUDA device",
)


class _GdnZeroRowsForward:
    """Bounded recurrent rows with padding outside each head's state matrix."""

    slots = 8
    sequences = 4
    heads = 2
    head_elements = 128 * 128
    padding_elements = 128

    def __init__(self, device):
        self.owner = torch.full(
            (self.slots, self.heads * self.head_elements + self.padding_elements),
            -777.0,
            dtype=torch.float32,
            device=device,
        )
        self.state = self.owner[:, : self.heads * self.head_elements].view(
            self.slots, self.heads, 128, 128
        )
        self.fresh = torch.zeros(self.sequences, dtype=torch.bool, device=device)
        self.final = torch.arange(
            1, self.sequences + 1, dtype=torch.int64, device=device
        )
        self.num_seqs = torch.full(
            (1,), self.sequences, dtype=torch.int32, device=device
        )
        args = (
            self.state,
            self.fresh,
            self.final,
            self.num_seqs,
            self.state.stride(0),
        )
        grid = (self.sequences, self.heads, 1)
        compiled = warmup_triton(
            _zero_fresh_rows.fn,
            args,
            grid,
            num_warps=_ZERO_CONFIG.num_warps,
            num_stages=_ZERO_CONFIG.num_stages,
            BLOCK=_ZERO_BLOCK,
        )
        self.operation = PreparedTriton(
            kernel=_zero_fresh_rows,
            compiled=compiled,
            args=args,
            grid=grid,
            options={"BLOCK": _ZERO_BLOCK},
        )

    def __call__(self):
        self.operation()

    def tensors(self):
        return self.owner, self.fresh, self.final, self.num_seqs


@torch.no_grad()
def test_gdn_zero_rows_restores_fixed_options_and_replays_live_masks(
    tmp_path, monkeypatch
):
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("assign one CUDA device explicitly")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))

    original = _GdnZeroRowsForward(device)
    compiled = aot_forward(original)
    compiled(original)
    path = tmp_path / "gdn-zero-rows.bin"
    compiled.save_compiled_function(str(path))
    fresh = _GdnZeroRowsForward(device)
    assert fresh.owner.data_ptr() != original.owner.data_ptr()
    assert fresh.state.stride(0) > fresh.heads * fresh.head_elements
    # Captured writes must bind the restored owner's storage.
    original.owner.fill_(-991.0)
    torch.cuda.synchronize(device)

    def unexpected(*args, **kwargs):
        raise AssertionError("restored GDN execution must not trace or compile")

    cases = (
        # Every live fresh non-null slot is cleared.
        ((1, 2, 3, 4), (0, 12, 0, 0), 4),
        # Slot zero and rows beyond NumSeqs preserve their exact state.
        ((4, 0, 2, 5), (0, 0, 13, 0), 3),
        ((6, 7, 1, 3), (0, 0, 0, 0), 0),
        # Nonzero positions preserve rows even when all sequences are live.
        ((2, 5, 1, 6), (16, 8, 32, 64), 4),
        ((6, 3, 0, 1), (0, 0, 0, 0), 2),
    )
    with monkeypatch.context() as frozen:
        frozen.setattr(torch, "compile", unexpected)
        frozen.setattr(triton, "compile", unexpected)
        frozen.setattr(JITFunction, "_do_compile", unexpected)
        with bind_device(device), path.open("rb") as file:
            restored = torch.compiler.load_compiled_function(file)
        assert restored.guard_check(fresh)
        restored(fresh)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.graph(graph, stream=stream):
            restored(fresh)
        torch.cuda.synchronize(device)
        try:
            addresses = tuple(tensor.data_ptr() for tensor in fresh.tensors())
            for generation, (row_ids, positions, live) in enumerate(cases):
                before = torch.arange(fresh.owner.numel(), dtype=torch.float32).view(
                    fresh.owner.shape
                )
                before.add_(1000 * (generation + 1))
                expected = before.clone()
                for index in range(live):
                    slot = row_ids[index]
                    if positions[index] == 0 and slot != 0:
                        expected[slot, : fresh.heads * fresh.head_elements].zero_()
                fresh.final.copy_(torch.tensor(row_ids, dtype=torch.int64))
                fresh.fresh.copy_(torch.tensor(positions, dtype=torch.int64).eq(0))
                fresh.num_seqs.fill_(live)
                for _ in range(2):
                    # Poison includes live matrices, inactive rows, null row, and padding.
                    fresh.owner.copy_(before)
                    allocated = torch.cuda.memory_stats(device)[
                        "allocation.all.allocated"
                    ]
                    graph.replay()
                    torch.cuda.synchronize(device)
                    assert (
                        torch.cuda.memory_stats(device)["allocation.all.allocated"]
                        == allocated
                    )
                    assert (
                        tuple(tensor.data_ptr() for tensor in fresh.tensors())
                        == addresses
                    )
                    torch.testing.assert_close(
                        fresh.owner.cpu(), expected, rtol=0, atol=0
                    )
                    assert torch.all(original.owner == -991.0)
        finally:
            graph.reset()
