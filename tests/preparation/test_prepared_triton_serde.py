# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Wrapped fixed Triton launches retain their actual prepared launch config."""

from types import SimpleNamespace
import pickle

import pytest
import torch
import triton
import triton.language as tl
from triton.runtime.autotuner import Autotuner, Config
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_flatten, tree_unflatten, treespec_loads

from ignition.preparation import aot
from ignition.preparation.compilation import _Invocation
from ignition.preparation.triton import _launch_options, prepare_triton, warmup_triton


@triton.jit
def _copy_launch(source, output, N: tl.constexpr):
    offsets = tl.arange(0, N)
    tl.store(output + offsets, tl.load(source + offsets))


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    initialized = torch.cuda.is_initialized()
    monkeypatch.setattr(torch.cuda, "_lazy_init",
        lambda *args, **kwargs: pytest.fail("launch-config CPU tests must not initialize CUDA"))
    for name in ("id_to_kernel", "kernel_to_id", "constant_args"):
        monkeypatch.setattr(kernel_side_table, name, {})
    yield
    assert torch.cuda.is_initialized() == initialized


def tuner(config=None):
    return Autotuner(_copy_launch, _copy_launch.arg_names,
        [Config({}) if config is None else config], [], None, None)


@pytest.mark.parametrize("warps", [1, 4, 8])
def test_warmup_normalizes_fixed_config_and_preserves_trace_options(warps):
    original = dict(N=32, num_warps=warps, enable_fp_fusion=False, maxnreg=64)
    expected = dict(N=32, enable_fp_fusion=False, maxnreg=64)
    expected.update(Config({}, num_warps=warps).all_kwargs())
    assert list(_launch_options(original).items()) == list(expected.items())
    assert original == dict(N=32, num_warps=warps, enable_fp_fusion=False, maxnreg=64)
    assert list(_launch_options(expected).items()) == list(expected.items())

    warmups = []
    compiled = object()
    kernel = SimpleNamespace(warmup=lambda *args, **kwargs:
        warmups.append((args, kwargs)) or compiled)
    prepared = prepare_triton(kernel, ("source", "output"), (1,), **original)
    assert prepared.compiled is compiled
    assert list(prepared.options.items()) == list(original.items())
    assert warmups == [(("source", "output"), dict(grid=(1,), **expected))]
    assert warmup_triton(kernel, ("source", "output"), (1,), **original) is compiled
    assert warmups[-1] == (("source", "output"), dict(grid=(1,), **expected))


@pytest.mark.parametrize("warps", [1, 8])
def test_prepared_launch_traces_through_torch_fixed_config_without_cuda(monkeypatch, warps):
    warmups = []
    compiled = object()
    monkeypatch.setattr(_copy_launch, "warmup", lambda *args, **kwargs:
        warmups.append(kwargs) or compiled)
    source = torch.arange(32, dtype=torch.float32)
    output = torch.zeros_like(source)
    options = dict(N=32, num_warps=warps, enable_fp_fusion=False)
    prepared = prepare_triton(_copy_launch, (source, output), (1,), **options)

    graph = make_fx(prepared, tracing_mode="fake", _allow_non_fake_inputs=True)()
    calls = [node for node in graph.graph.nodes
        if node.target is torch.ops.higher_order.triton_kernel_wrapper_mutation]
    assert len(calls) == 1
    traced = kernel_side_table.get_kernel(calls[0].kwargs["kernel_idx"])
    assert type(traced) is Autotuner and len(traced.configs) == 1
    assert traced.fn is _copy_launch
    assert traced.configs[0].all_kwargs() == Config({}, num_warps=warps).all_kwargs()
    assert warmups == [dict(grid=(1,), **_launch_options(options))]


def test_plain_backend_options_are_not_promoted_to_an_autotuner_config():
    options = dict(N=32, maxnreg=64, enable_fp_fusion=False)
    normalized = _launch_options(options)
    assert normalized is not options
    assert list(normalized.items()) == list(options.items())
    kernel, copied = aot._serialized_triton_launch(_copy_launch, options, {})
    assert kernel is _copy_launch and copied == options and copied is not options


@pytest.mark.parametrize("warps", [1, 8])
def test_single_config_preserves_constexprs_and_backend_options_without_aliasing(warps):
    config = Config(dict(N=32), num_warps=warps, num_stages=2, maxnreg=64)
    constants = dict(enable_fp_fusion=False)
    kernel, combined = aot._serialized_triton_launch(tuner(config), constants,
        dict(source=None, output=None))
    assert kernel is _copy_launch
    assert list(combined.items()) == list(dict(constants, **config.all_kwargs()).items())
    assert constants == dict(enable_fp_fusion=False)
    assert config.kwargs == dict(N=32)


@pytest.mark.parametrize("conflict_in", ["constant", "argument"])
def test_config_collisions_reject_even_equal_values(conflict_in):
    constants = dict(num_warps=8) if conflict_in == "constant" else {}
    arguments = dict(num_warps=8) if conflict_in == "argument" else {}
    with pytest.raises(ValueError, match="conflicts.*num_warps"):
        aot._serialized_triton_launch(tuner(Config({}, num_warps=8)), constants, arguments)


@pytest.mark.parametrize("field,value", [
    ("user_defined_pre_hook", True), ("user_defined_post_hook", True),
    ("reset_to_zero", ["output"]), ("restore_value", ["output"]),
    ("early_config_prune", lambda *args: None), ("perf_model", lambda *args: None),
    ("_do_bench", lambda *args: None), ("num_warmups", 3), ("num_reps", 5),
    ("use_cuda_graph", True),
])
def test_unsupported_autotuner_behavior_is_not_silently_discarded(field, value):
    kernel = tuner()
    setattr(kernel, field, value)
    with pytest.raises(ValueError, match="cannot discard"):
        aot._serialized_triton_launch(kernel, {}, {})


def test_multiple_configs_nested_wrappers_and_config_hooks_reject():
    kernel = tuner()
    kernel.configs.append(Config({}, num_warps=8))
    with pytest.raises(ValueError, match="one fixed launch config"):
        aot._serialized_triton_launch(kernel, {}, {})
    kernel = tuner()
    kernel.fn = tuner()
    with pytest.raises(ValueError, match="one fixed launch config"):
        aot._serialized_triton_launch(kernel, {}, {})
    with pytest.raises(ValueError, match="cannot discard"):
        aot._serialized_triton_launch(tuner(Config({}, pre_hook=lambda *args: None)), {}, {})


@pytest.mark.parametrize("warps", [1, 8])
def test_real_storage_serde_preserves_launch_options_and_dynamic_leaf_positions(monkeypatch, warps):
    traced = tuner(Config({}, num_warps=warps))
    constants = dict(N=32, enable_fp_fusion=False)
    kernel_id = kernel_side_table.add_kernel(traced)
    constant_id = kernel_side_table.add_constant_args(constants)
    arguments = dict(source=None, output=None)
    leaves, tree = tree_flatten(((), dict(kernel_idx=kernel_id, constant_args_idx=constant_id,
        grid=[(1, 1, 1)], tma_descriptor_metadata={}, kwargs=arguments)))
    positions = [index for index, value in enumerate(leaves) if value is None]
    assert len(positions) == 2
    operands = [(positions[0], False, 0, None), (positions[1], True, 0, None)]
    call = _Invocation(torch.ops.higher_order.triton_kernel_wrapper_mutation,
        leaves, tree, operands, [])
    signature = (((32,), (1,), 0, torch.float32, torch.device("cuda:0"), 0),)
    owner = SimpleNamespace(artifact=SimpleNamespace(serialize=lambda: b"inner artifact"),
        inputs=[], invocations=[call], signature=signature, _serialized_artifact=None, _tuning_frozen=False)
    payload = aot.StorageBackend.serialize_compile_artifacts(owner)
    _, _, stored, _, frozen = pickle.loads(payload)
    assert frozen is False
    name, restored_leaves, restored_tree, restored_operands, restored_integers = stored[0]
    assert name[1] == (__name__, "_copy_launch")
    assert list(name[2].items()) == list(dict(constants, **traced.configs[0].all_kwargs()).items())
    assert restored_operands == operands and restored_integers == []
    assert len(restored_leaves) == len(leaves)
    _, stored_kwargs = tree_unflatten(restored_leaves, treespec_loads(restored_tree))
    assert stored_kwargs['kernel_idx'] is None and stored_kwargs['constant_args_idx'] is None
    assert stored_kwargs['kwargs'] == dict(source=None, output=None)
    assert all(restored_leaves[position] is None for position in positions)
    assert constants == dict(N=32, enable_fp_fusion=False)

    monkeypatch.setattr(aot, "_load_inductor", lambda artifact, device, *, frozen: (object(), None))
    restored = aot.StorageBackend.deserialize_compile_artifacts(payload)
    actual_launches = []
    monkeypatch.setattr(_copy_launch, "run",
        lambda *args, **kwargs: actual_launches.append((args, kwargs)))
    source, output = torch.arange(32, dtype=torch.float32), torch.zeros(32)
    restored.invocations[0].run([source], [output])
    assert len(actual_launches) == 1
    args, kwargs = actual_launches[0]
    assert args[0] is source and args[1] is output and args[2] == 32
    options = {name: value for name, value in kwargs.items() if name not in ("grid", "warmup")}
    expected = _launch_options(dict(enable_fp_fusion=False, num_warps=warps))
    assert list(options.items()) == list(expected.items())
    assert kwargs["warmup"] is False and kwargs["grid"] == (1, 1, 1)
    assert output.count_nonzero() == 0
