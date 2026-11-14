"""Expanded projection matrices and GEMM workspaces occupy separate byte ranges."""

from contextlib import nullcontext

import torch

from b12x.gemm.blockscaled import _a16
from flashinfer import gemm
from flashinfer.gemm import gemm_base
from ignition.components import sm103_projection as projection


def test_expanded_nvfp4_separates_matrix_and_aligned_gemm_workspace(monkeypatch):
    query = projection.PackedProjectionQuery(recipe="nvfp4", rows=2, in_features=8,
        padded_in_features=16, out_features=8, global_scale_kind="multiplier",
        workspace_form="provided")
    monkeypatch.setattr(gemm_base, "DEFAULT_WORKSPACE_SIZE", 512)
    monkeypatch.setattr(torch.cuda, "device", lambda *args: nullcontext())
    monkeypatch.setattr(_a16, "_stream_context", lambda *args: nullcontext())
    expanded_shapes, calls = [], []
    class Expansion:
        def __getitem__(self, grid):
            return lambda values, scales, gain, output, *args: expanded_shapes.append(output.shape)
    monkeypatch.setattr(gemm, "mm_bf16", lambda source, weight, **kwargs: calls.append((weight, kwargs)))
    size = projection.workspace_bytes(query)
    state = projection.PreparedPackedProjection(query, "cublaslt", torch.device("cpu"),
        {"expand": Expansion()}, size, None)
    source = torch.empty((2, 8), dtype=torch.bfloat16)
    weight = torch.empty((8, 8), dtype=torch.uint8)
    scales = torch.empty(512, dtype=torch.uint8)
    gain = torch.ones(1)
    output = torch.empty_like(source)
    workspace = torch.empty(size, dtype=torch.uint8)
    assert state.run(source, weight, scales, gain, out=output, workspace=workspace) is output
    assert expanded_shapes == [torch.Size((8, 8))]
    matrix, options = calls[0]
    assert matrix.shape == (8, 8) and matrix.data_ptr() == workspace.data_ptr()
    offset = options["workspace_buffer"].data_ptr() - workspace.data_ptr()
    assert offset >= matrix.nbytes and offset % 256 == 0
    assert options["workspace_buffer"].nbytes == 512
