# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Block-FP8 tracing keeps process-local plan handles as runtime inputs."""

from dataclasses import replace

import pytest
import torch
from b12x.gemm import blockscaled
from torch._subclasses.fake_tensor import FakeTensorMode

from ignition.components.block_fp8 import BlockFP8Projection, BlockFP8Weight
from ignition.components.block_fp8_sm103 import plan as sm103_plan


@pytest.mark.parametrize("sm103, rows", [(False, 4), (True, 4), (True, 33)])
def test_block_fp8_traces_with_unbacked_plan_handles(sm103, rows):
    graphs = []
    handles = []

    def backend(module, examples):
        graphs.append(module)

        def run(*arguments):
            handles.extend(value for value in arguments if type(value) is int)
            return module.forward(*arguments)

        return run

    projection = object.__new__(BlockFP8Projection)
    projection.plan = (sm103_plan(rows, 128, 256, "cpu") if sm103 else
        blockscaled.plan(blockscaled.FixedBlockscaledQuery(
            recipe="block_fp8", call_kind="serialized", max_rows=rows,
            in_features=128, padded_in_features=128, out_features=256,
            input_dtype="float8_e4m3fn", output_dtype="bfloat16", expected_m=rows)))
    expected_handles = [projection.plan.handle]
    projection.workspace_nbytes = 64
    projection.output = projection.output_layout = None
    with FakeTensorMode(), torch.compiler.config.patch(unbacked_sources=r".*\._?handle$"):
        source = torch.empty((rows, 128), dtype=torch.bfloat16)
        projection.weight = BlockFP8Weight(torch.empty((256, 128), dtype=torch.float8_e4m3fn),
                                           torch.empty((2, 1), dtype=torch.float32))
        compiled = torch.compile(lambda value: projection(value), backend=backend,
                                 fullgraph=True, dynamic=False)
        result = compiled(source)
        projection.plan = replace(projection.plan)
        expected_handles.append(projection.plan.handle)
        rebound = compiled(source)
    assert result.shape == rebound.shape == (rows, 256)
    assert result.dtype == rebound.dtype == torch.bfloat16
    assert len(graphs) == 1
    assert handles == expected_handles
    calls = [node for node in graphs[0].graph.nodes
             if node.target is torch.ops.flare.block_fp8_projection_out.default]
    assert len(calls) == 1
    handle = calls[0].args[-1]
    assert handle.op == "placeholder"
    assert isinstance(handle.meta["example_value"], torch.SymInt)
