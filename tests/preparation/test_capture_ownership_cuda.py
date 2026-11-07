# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Captured forwards keep guards and live kernels without serialized compiler state."""

from contextlib import nullcontext
import gc
import os
import weakref

import pytest
import torch

from ignition.components.linear import UnquantizedProjection
from ignition.preparation.aot import BindingMismatch, BoundForward, aot_forward
from ignition.preparation.capture import capture_forwards
from ignition.preparation.execution import GraphSignature
from ignition.preparation.tuning import RecordedPreparationSession


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


class Forward:
    def __init__(self, source, output, operation=None):
        self.source, self.output, self.operation = source, output, operation
        self.offset = True

    def __call__(self):
        value = self.source.square() if self.operation is None else self.operation(self.source)
        self.output.copy_(value + (1 if self.offset else 0))


@pytest.mark.parametrize("compile_decoder", [False, True])
@pytest.mark.parametrize("autotune_effort", ["off", "default"])
@torch.no_grad()
def test_forward_warmup_captures_unprimed_plans_and_cached_restart(tmp_path, compile_decoder, autotune_effort):
    device = torch.device("cuda:0")
    source = (torch.randn((4, 128), device=device) * .1).bfloat16()
    weight = (torch.randn((128, 128), device=device) * .1).bfloat16()
    projection_output = torch.empty_like(source)
    output = torch.empty_like(source)
    signature = GraphSignature(kind="decode", batch_capacity=4, token_capacity=1)
    for restart in (False, True):
        projection_output.fill_(float("nan"))
        with RecordedPreparationSession(device=device, checkpoint="unprimed-capture",
                workload_schema="projection", numerical_policy="bf16", compile_workers=1,
                autotune_effort=autotune_effort) as session:
            session.cache_dir = tmp_path / "kernels"
            assert session.prime is False
            operation = UnquantizedProjection(source, weight, projection_output,
                session=session, name="projection")
            session.tune(show_progress=False)
            assert projection_output.isnan().all()
            forward = Forward(source, output, operation)
            graphs, compiled = capture_forwards({"decode": forward}, {"decode": signature},
                session=session, device=device, stream=torch.cuda.Stream(device=device),
                pool=torch.cuda.graph_pool_handle(), compile_decoder=compile_decoder,
                cache_root=tmp_path / "forwards")
            try:
                if compile_decoder:
                    assert compiled["decode"].cache_hit is restart
                session.freeze()
                for value in (.125, -.25):
                    source.fill_(value)
                    before = torch.cuda.memory_stats(device)
                    graphs["decode"].replay()
                    torch.cuda.synchronize(device)
                    after = torch.cuda.memory_stats(device)
                    for key in ("allocation.all.allocated", "allocation.all.freed",
                                "segment.all.allocated", "segment.all.freed"):
                        assert before[key] == after[key]
                    expected = (source.float() @ weight.float().T).bfloat16() + 1
                    torch.testing.assert_close(output, expected, atol=.01, rtol=.01)
            finally:
                graphs["decode"].reset()
                compiled.clear()


@pytest.mark.parametrize("opaque", [False, True])
@torch.no_grad()
def test_capture_releases_serialized_artifact_and_preserves_guards_and_replay(tmp_path, opaque):
    device = torch.device("cuda:0")
    source = (torch.randn((4, 128), device=device) * .1).bfloat16()
    output = torch.empty_like(source)
    scope = (RecordedPreparationSession(device=device, checkpoint="capture-ownership",
        workload_schema="projection", numerical_policy="bf16", compile_workers=1) if opaque else nullcontext())
    with scope as session:
        weight = (torch.randn((128, 128), device=device) * .1).bfloat16() if opaque else None
        operation = None
        if opaque:
            operation = UnquantizedProjection(source, weight, torch.empty_like(source),
                session=session, name="projection")
            session.tune(show_progress=False)
        forward = Forward(source, output, operation)
        artifact = aot_forward(forward)
        artifact(forward)
        path = tmp_path / "compiled.bin"
        artifact.save_compiled_function(str(path))
        del artifact
        with path.open("rb") as file:
            artifact = torch.compiler.load_compiled_function(file)
        bound = BoundForward(artifact, forward, True)
        bound()
        reference = weakref.ref(artifact)
        del artifact
        assert bound.prepare_capture() > 0
        assert bound.prepare_capture() == 0
        gc.collect()
        assert reference() is None
        assert bound.artifact.backend._serialized_artifact is None
        graph = torch.cuda.CUDAGraph()
        try:
            bound()
            with (session.capture() if opaque else nullcontext()), torch.cuda.graph(graph):
                bound()
            for value in (.125, -.25, .5):
                source.fill_(value)
                before = torch.cuda.memory_stats(device)
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                expected = ((source.float() @ weight.float().T).bfloat16() if opaque else source.square()) + 1
                torch.testing.assert_close(output, expected, atol=.01 if opaque else 0, rtol=.01 if opaque else 0)
            forward.offset = False
            with pytest.raises(BindingMismatch, match="compiled guards"):
                bound()
            forward.offset = True
            forward.source = source[:, :64]
            with pytest.raises(BindingMismatch, match="compiled guards"):
                bound()
        finally:
            graph.reset()
