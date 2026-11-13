# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Restored storage calls retain traced Triton launch options without decorators."""

import sys
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_flatten
from triton.runtime.autotuner import Autotuner, Heuristics

from ignition.preparation import aot
from ignition.preparation.compilation import _Invocation


@triton.jit
def _plain_copy(source, output, N: tl.constexpr):
    columns = tl.arange(0, N)
    tl.store(output + columns, tl.load(source + columns))


@triton.autotune(configs=[triton.Config({}, num_warps=4, num_stages=2)], key=[])
@triton.jit
def _fixed_copy(source, output, N: tl.constexpr):
    columns = tl.arange(0, N)
    tl.store(output + columns, tl.load(source + columns))


def _first_config(configs, named_args, **kwargs):
    return configs[:1]


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=2),
    ],
    key=[],
    prune_configs_by={"early_config_prune": _first_config},
)
@triton.jit
def _pruned_copy(source, output, N: tl.constexpr):
    columns = tl.arange(0, N)
    tl.store(output + columns, tl.load(source + columns))


@triton.autotune(configs=[triton.Config({}, num_warps=4, num_stages=2)], key=[])
@triton.heuristics({"BLOCK": lambda arguments: arguments["N"]})
@triton.jit
def _heuristic_copy(source, output, N: tl.constexpr, BLOCK: tl.constexpr):
    columns = tl.arange(0, BLOCK)
    tl.store(output + columns, tl.load(source + columns, columns < N), columns < N)


@pytest.fixture(autouse=True)
def no_cuda_or_shared_kernel_ids(monkeypatch):
    initialized = torch.cuda.is_initialized()
    monkeypatch.setattr(
        torch.cuda,
        "_lazy_init",
        lambda *args, **kwargs: pytest.fail(
            "restored launch tests must not initialize CUDA"
        ),
    )
    for name in ("id_to_kernel", "kernel_to_id", "constant_args"):
        monkeypatch.setattr(kernel_side_table, name, {})
    yield
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("warps", [1, 8])
@pytest.mark.parametrize(
    "decorated",
    [_fixed_copy, _pruned_copy, _heuristic_copy],
    ids=["fixed", "pruned", "heuristic"],
)
def test_storage_roundtrip_bypasses_imported_global_autotuner(
    monkeypatch, warps, decorated
):
    source = torch.arange(32, dtype=torch.float32)
    output = torch.zeros_like(source)

    def forward():
        torch.library.wrap_triton(decorated)[(1,)](
            source, output, N=32, num_warps=warps, enable_fp_fusion=False
        )

    graph = make_fx(forward, tracing_mode="fake", _allow_non_fake_inputs=True)()
    (node,) = [
        node
        for node in graph.graph.nodes
        if node.target is torch.ops.higher_order.triton_kernel_wrapper_mutation
    ]
    traced = kernel_side_table.get_kernel(node.kwargs["kernel_idx"])
    assert type(traced) is Autotuner and len(traced.configs) == 1
    assert traced.configs[0].num_warps == warps
    assert decorated.configs[0].num_warps == 4
    leaves, tree = tree_flatten((node.args, node.kwargs))
    operands = []
    for key, mutable in (("source", False), ("output", True)):
        tensor_node = node.kwargs["kwargs"][key]
        position = next(
            index for index, leaf in enumerate(leaves) if leaf is tensor_node
        )
        operands.append((position, mutable, 0, None))
        leaves[position] = None
    owner = SimpleNamespace(
        artifact=SimpleNamespace(serialize=lambda: b"inner artifact"),
        inputs=[],
        _serialized_artifact=None,
        _tuning_frozen=False,
        invocations=[_Invocation(node.target, leaves, tree, operands)],
        signature=(((32,), (1,), 0, torch.float32, torch.device("cuda:0"), 0),),
    )
    payload = aot.StorageBackend.serialize_compile_artifacts(owner)
    monkeypatch.setattr(aot, "_load_inductor", lambda artifact, device, *, frozen: (object(), None))
    restored = aot.StorageBackend.deserialize_compile_artifacts(payload)
    launches = []
    monkeypatch.setattr(
        traced.fn, "run", lambda *args, **kwargs: launches.append((args, kwargs))
    )
    monkeypatch.setattr(
        decorated,
        "_bench",
        lambda *args, **kwargs: pytest.fail(
            "restored fixed launches must not autotune"
        ),
    )
    restored.invocations[0].run([source], [output])
    assert len(launches) == 1
    args, options = launches[0]
    assert len(args) == 2 and args[0] is source and args[1] is output
    assert options["N"] == 32
    assert options["num_warps"] == warps
    assert options["num_stages"] == 2
    assert options["enable_fp_fusion"] is False
    assert options["grid"] == (1, 1, 1) and options["warmup"] is False
    assert not decorated.cache


def test_restored_raw_jit_has_no_decorator_requirements():
    kernel_info = (__name__, "_plain_copy")
    assert aot._deserialized_triton_launch(kernel_info) is _plain_copy


def test_heuristics_outside_autotuner_are_unwrapped_without_running(monkeypatch):
    raw = _fixed_copy.fn
    wrapper = Heuristics(
        _fixed_copy,
        raw.arg_names,
        {
            "N": lambda arguments: pytest.fail(
                "restoration must not evaluate heuristics"
            )
        },
    )
    monkeypatch.setattr(sys.modules[__name__], "_fixed_copy", wrapper)
    assert aot._deserialized_triton_launch((__name__, "_fixed_copy")) is raw


@pytest.mark.parametrize("problem", ["unknown_wrapper", "cycle", "different_function"])
def test_invalid_imported_fixed_launch_is_rejected(monkeypatch, problem):
    if problem == "unknown_wrapper":
        monkeypatch.setattr(
            sys.modules[__name__], "_fixed_copy", SimpleNamespace(fn=_fixed_copy.fn)
        )
        message = "does not resolve to a JIT function"
    elif problem == "cycle":
        monkeypatch.setattr(_fixed_copy, "fn", _fixed_copy)
        message = "cyclic decorators"
    else:
        monkeypatch.setattr(sys.modules[__name__], "_fixed_copy", _pruned_copy)
        message = "different function"
    with pytest.raises(ValueError, match=message):
        aot._deserialized_triton_launch((__name__, "_fixed_copy"))
