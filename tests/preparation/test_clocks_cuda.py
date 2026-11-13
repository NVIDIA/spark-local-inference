# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Privileged clock locking around real b12x tuning and CUDA graph replay."""

import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an explicitly selected GPU and clock-control permission")


def test_locked_tuning_resets_clocks_before_graph_capture(tmp_path, monkeypatch):
    import torch
    from ignition.components.linear import UnquantizedProjection
    from ignition.preparation import clocks
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda", 0)
    source = torch.randn((4, 768), device=device, dtype=torch.bfloat16)
    weight = torch.randn((512, 768), device=device, dtype=torch.bfloat16)
    output = torch.empty((4, 512), device=device, dtype=torch.bfloat16)
    changes = []
    set_clocks = clocks._set_clocks

    def record(uuid, option):
        set_clocks(uuid, option)
        changes.append((uuid, option))

    monkeypatch.setattr(clocks, "_set_clocks", record)
    with RecordedPreparationSession(device=device, checkpoint="clock-control-test",
            workload_schema="projection", numerical_policy="bf16", compile_workers=2,
            namespace={"case": str(tmp_path)}, lock_gpu_clocks=True) as session:
        projection = UnquantizedProjection(source, weight, output, session=session, name="projection")

        def progress(status):
            assert len(changes) == 1

        report = session.tune(progress=progress)
        assert report["benchmarked_candidates"] > 0
        assert len(changes) == 2
        assert changes[0][0] == changes[1][0] == f"GPU-{session.device.uuid.removeprefix('GPU-')}"
        assert changes[0][1].startswith("--lock-gpu-clocks=")
        assert changes[1][1] == "--reset-gpu-clocks"
        projection(source)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            projection(source)
        output.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(output.float(), source.float() @ weight.float().T,
                                   rtol=0.01, atol=0.25)
