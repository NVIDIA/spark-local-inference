# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""External matrix-multiply replacements retain dependencies on storage writes."""

import pytest
import torch

from ignition import runtime_patches
from ignition.runtime_patches import scheduler as patches


def _clean_replace_node(function):
    from inspect import getsource
    from textwrap import dedent

    applied = """    # Rebuild unmet dependencies after renaming inputs to their mutation outputs.
    # Initial pruning treats the original graph inputs as already available.
    new_scheduler_node.update_mutated_names(mutation_renames)
"""
    if applied not in dedent(getsource(function)):
        return function
    clean = runtime_patches._replace(function, ((applied, """    def rename_deps(deps: OrderedSet[Dep]) -> OrderedSet[Dep]:
        return OrderedSet(dep.rename(mutation_renames) for dep in deps)

    new_scheduler_node.unmet_dependencies = rename_deps(
        new_scheduler_node.unmet_dependencies
    )
    new_scheduler_node.read_writes.reads = rename_deps(
        new_scheduler_node.read_writes.reads
    )
"""),))
    del clean.__wrapped__
    return clean


@torch.library.custom_op("test_inductor_template_mutations::copy", mutates_args=("writes",))
def _copy(reads: list[torch.Tensor], writes: list[torch.Tensor]) -> None:
    writes[0].copy_(reads[0])


@_copy.register_fake
def _fake_copy(reads, writes):
    pass


@torch.no_grad()
@pytest.mark.parametrize("implementation", ["clean", "runtime_clean", "runtime_installed"])
def test_external_bmm_replacement_retains_input_mutation_dependency(monkeypatch, implementation):
    from torch._inductor import ir, standalone_compile
    from torch._inductor.scheduler import Scheduler
    from torch.fx.experimental.proxy_tensor import make_fx

    if implementation != "runtime_installed":
        monkeypatch.setattr(Scheduler, "_replace_node", _clean_replace_node(Scheduler._replace_node))
    if implementation != "clean":
        for owner, name, replacement in patches.replacements():
            if owner is Scheduler and name == "_replace_node":
                monkeypatch.setattr(owner, name, replacement)

    attended = torch.empty(13 * 6 * 512, dtype=torch.bfloat16)
    source = torch.ones_like(attended)
    weight = torch.ones(6, 512, 128, dtype=torch.bfloat16)
    output = torch.empty(13, 768, dtype=torch.bfloat16)
    chain = [torch.empty_like(attended) for _ in range(4)]

    def forward(attended, source, weight, output, *chain):
        view = attended.view(13, 6, 512).transpose(0, 1)
        previous = source
        for buffer in chain:
            _copy([previous], [buffer])
            previous = buffer
        _copy([previous], [attended])
        output.copy_(torch.bmm(view, weight).transpose(0, 1).flatten(1))
        return (output,)

    replacements = []
    finalize = Scheduler.finalize_multi_template_buffers

    def replace_external_bmm(scheduler):
        finalize(scheduler)
        for index, node in enumerate(scheduler.nodes):
            original = node.node
            if (not isinstance(original, ir.ExternKernelOut)
                    or original.op_overload is not torch.ops.aten.bmm.out):
                continue
            input_name = original.inputs[0].get_name()
            writers = {dep.name for dep in node.unmet_dependencies
                       if scheduler.mutation_real_name.get(dep.name) == input_name}
            assert writers
            with ir.IRNode.current_origins(original.origins):
                replacement = ir.ExternKernelOut(layout=original.layout, inputs=original.inputs,
                    constant_args=original.constant_args, kwargs=original.kwargs,
                    python_kernel_name=original.python_kernel_name,
                    cpp_kernel_name=original.cpp_kernel_name,
                    ordered_kwargs_for_cpp_kernel=original.ordered_kwargs_for_cpp_kernel,
                    op_overload=original.op_overload)
            # CPU compilation selects its external kernel before deferred-template finalization.
            scheduler._replace_node(replacement, original, index, node)
            replaced = scheduler.nodes[index]
            assert writers <= {dep.name for dep in replaced.read_writes.reads}
            assert writers <= {dep.name for dep in replaced.unmet_dependencies}, "template input writer dependency was pruned"
            replacements.append(writers)

    monkeypatch.setattr(Scheduler, "finalize_multi_template_buffers", replace_external_bmm)
    arguments = (attended, source, weight, output, *chain)
    module = make_fx(forward, tracing_mode="fake")(*arguments)

    def compile_forward():
        return standalone_compile(module, arguments, aot=True, options={"config_patches": {
            "max_autotune": False, "compile_threads": 1, "force_disable_caches": True,
        }})

    if implementation == "clean":
        with pytest.raises(torch._inductor.exc.InductorError, match="template input writer dependency was pruned"):
            compile_forward()
        return
    compiled = compile_forward()
    assert len(replacements) == 1

    for value, scale, poison in ((1.0, 1.0, float("nan")), (-0.5, 2.0, 19.0), (3.0, 0.25, -23.0)):
        source.fill_(value)
        weight.fill_(scale)
        for buffer in (attended, output, *chain):
            buffer.fill_(poison)
        result, = compiled(*arguments)
        expected = torch.full_like(output, 512 * value * scale)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
