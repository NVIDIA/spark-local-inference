# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Restored forwards bind live storage and native handles before graph capture."""

import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

from ignition.preparation.aot import ForwardCache, _guard_code_cache, aot_forward

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="explicit CUDA assignment required")


@torch.library.custom_op("test_aot::fill", mutates_args=("output",))
def _fill(output: torch.Tensor, handle: int) -> None:
    output.fill_(handle + .25)


@_fill.register_fake
def _fake_fill(output, handle):
    pass


@triton.jit
def _copy(source, output):
    index = tl.arange(0, 32)
    tl.store(output + index, tl.load(source + index))


@triton.jit
def _unrelated(output):
    tl.store(output + tl.arange(0, 32), -19.)


class BufferForward:
    def __init__(self, handle):
        from b12x.preparation.device import DetectedDevice

        self.handle = handle
        self.storage = torch.zeros(1024, dtype=torch.uint8, device="cuda")
        self.prepared_device = DetectedDevice(ordinal=self.storage.device.index, identity=None)
        self.part = self.storage.view(torch.float32)[16:48]
        self.alias = self.storage.view(torch.bfloat16)
        self.before = torch.empty_like(self.alias)
        self.source = torch.randn(32, device="cuda")
        self.output = torch.empty_like(self.source)
        self.negate = False

    def __call__(self):
        if self.source.device.index != self.prepared_device.ordinal:
            raise ValueError("prepared device differs from source")
        self.storage.zero_()
        _fill(self.part, self.handle)
        self.before.copy_(self.alias)
        torch.library.wrap_triton(_copy)[(1,)](self.source, self.output)
        self.output.add_(self.part)
        if self.negate:
            self.output.neg_()
        return self.output


class ElementwiseForward:
    def __init__(self):
        self.source = torch.randn(4096, device="cuda")
        self.output = torch.empty_like(self.source)

    def __call__(self):
        self.output.copy_(self.source.sin() * self.source)


class ReductionForward:
    def __init__(self):
        self.source = torch.empty((128, 4096), device="cuda")
        self.output = torch.empty(128, device="cuda")

    def __call__(self):
        self.output.copy_(self.source.sum(dim=1))


@torch.no_grad()
def test_compile_time_reduction_winner_restores_without_search(tmp_path, monkeypatch):
    from ignition.preparation import aot
    from ignition.preparation.compilation import _OPTIONS
    from ignition.preparation.execution import GraphSignature
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    cache = ForwardCache(namespace={"model": "prepared-reduction"}, device="cuda:0",
        root=tmp_path, options=dict(_OPTIONS, **{"triton.autotune_at_compile_time": True}))
    signature = {"reduce": GraphSignature("prefill", 1, 128)}
    original = ReductionForward()
    original.source.fill_(1)
    first = cache.prepare_many({"reduce": original}, signature)["reduce"]
    assert not first.cache_hit
    torch.testing.assert_close(original.output, torch.full_like(original.output, 4096), rtol=0, atol=0)

    def unexpected(*args, **kwargs):
        raise AssertionError("restoring a completed reduction must not compile or search")

    fresh = ReductionForward()
    fresh.source.fill_(2)
    monkeypatch.setattr(aot, "aot_forward", unexpected)
    monkeypatch.setattr(triton, "compile", unexpected)
    monkeypatch.setattr(CachingAutotuner, "benchmark_all_configs", unexpected)
    restored = cache.prepare_many({"reduce": fresh}, signature)["reduce"]
    assert restored.cache_hit
    backend = restored.backend
    entry = backend.artifact.inner_fn.compiled_fn.serialize()
    tuners = entry.compiled_fw.result._triton_bundle.static_autotuners
    assert tuners
    for selected in tuners:
        assert len(selected.kernel.compile_results) == 1
        assert selected.kernel.inductor_meta["dynamic_scale_rblock"] is False

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
        restored()
    try:
        for value in (2, -3, 0):
            fresh.source.fill_(value)
            fresh.output.fill_(float("nan"))
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            torch.testing.assert_close(fresh.output,
                torch.full_like(fresh.output, 4096 * value), rtol=0, atol=0)
    finally:
        graph.reset()


class SharedResidualForward:
    def __init__(self, triton_write=False):
        self.triton_write = triton_write
        self.source = torch.randn((128, 256), device="cuda", dtype=torch.bfloat16)
        self.attention = torch.empty((4096, 256), device="cuda", dtype=torch.bfloat16)[:128]
        self.feed_forward = torch.empty_like(self.attention)
        self.observed = torch.empty((4, 128, 256), device="cuda", dtype=torch.bfloat16)
        self.output = torch.empty_like(self.source)

    def fill(self, output, value):
        if self.triton_write:
            torch.library.wrap_triton(_fill_shared)[(triton.cdiv(output.numel(), 256),)](
                output, output.numel(), value + .25)
        else:
            _fill(output, value)

    def __call__(self):
        hidden = self.source
        for layer in range(4):
            self.fill(self.attention, layer + 1)
            hidden = hidden + self.attention
            self.observed[layer].copy_(hidden.square())
            self.fill(self.feed_forward, layer + 5)
            hidden = hidden + self.feed_forward
        self.output.copy_(hidden)


@triton.jit
def _fill_shared(output, count: tl.constexpr, value: tl.constexpr):
    index = tl.program_id(0) * 256 + tl.arange(0, 256)
    tl.store(output + index, value, index < count)


@torch.no_grad()
@pytest.mark.parametrize("triton_write", [False, True])
def test_residual_reads_survive_shared_scratch_overwrites(tmp_path, triton_write):
    forward = SharedResidualForward(triton_write)
    forward()
    expected = forward.output.clone()
    observed = forward.observed.clone()
    compiled = aot_forward(forward)
    compiled(forward)
    torch.testing.assert_close(forward.output, expected, rtol=0, atol=0)
    torch.testing.assert_close(forward.observed, observed, rtol=0, atol=0)
    path = tmp_path / "residual.bin"
    compiled.save_compiled_function(str(path))
    with path.open("rb") as file:
        restored = torch.compiler.load_compiled_function(file)
    restored(forward)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
        restored(forward)
    for _ in range(3):
        forward.source.normal_()
        forward()
        expected = forward.output.clone()
        observed = forward.observed.clone()
        forward.output.fill_(float("nan"))
        forward.observed.fill_(float("nan"))
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        torch.testing.assert_close(forward.output, expected, rtol=0, atol=0)
        torch.testing.assert_close(forward.observed, observed, rtol=0, atol=0)


@torch.no_grad()
def test_generated_kernels_rebind_without_compilation(tmp_path, monkeypatch):
    from ignition.preparation.device_binding import bind_device

    if torch.cuda.device_count() < 2:
        pytest.skip("requires two assigned GPUs")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    with torch.cuda.device(0):
        original = ElementwiseForward()
        compiled = aot_forward(original)
        compiled(original)
        path = tmp_path / "elementwise.bin"
        compiled.save_compiled_function(str(path))

    def unexpected(*args, **kwargs):
        raise AssertionError("binding compiled kernels must not invoke the compiler")

    with torch.cuda.device(1):
        fresh = ElementwiseForward()
        with monkeypatch.context() as loading:
            loading.setattr(torch, "compile", unexpected)
            loading.setattr(triton, "compile", unexpected)
            with bind_device("cuda:1"), path.open("rb") as file:
                restored = torch.compiler.load_compiled_function(file)
        restored(fresh)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.Stream(device=1)):
            restored(fresh)
        fresh.source.normal_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(fresh.output, fresh.source.sin() * fresh.source)

        backend = restored._artifacts.compiled_fn
        entry = backend.artifact.inner_fn.compiled_fn.serialize()
        assert not entry.compiled_fw.result._triton_bundle.kernel_artifacts
        saved = tmp_path / "rebound.bin"
        restored.save_compiled_function(str(saved))
        with monkeypatch.context() as loading:
            loading.setattr(torch, "compile", unexpected)
            loading.setattr(triton, "compile", unexpected)
            loading.setenv("TRITON_CACHE_DIR", str(tmp_path / "reloaded-triton"))
            with bind_device("cuda:1"), saved.open("rb") as file:
                reloaded = torch.compiler.load_compiled_function(file)
            assert list((tmp_path / "reloaded-triton").rglob("*.cubin"))
            fresh.source.normal_()
            reloaded(fresh)
            torch.testing.assert_close(fresh.output, fresh.source.sin() * fresh.source)
            source = fresh.source
            fresh.source = source[:2048]
            assert not reloaded.guard_check(fresh)
            fresh.source = source


@torch.no_grad()
def test_compile_before_materializing_storage(tmp_path, monkeypatch):
    import ignition.preparation.aot as aot
    from ignition.preparation.compile_pool import CompilePool
    from ignition.preparation.deferred import deferred_mode, deferred_tensor
    from ignition.preparation.execution import GraphSignature

    cache = ForwardCache(namespace={"model": "deferred-buffer"}, device="cuda:0", root=tmp_path)
    pools = []
    shutdown = CompilePool.__exit__
    load = torch.compiler.load_compiled_function
    original_threads = torch._inductor.config.compile_threads
    bindings = []

    def bind(*args, **kwargs):
        assert torch._inductor.config.compile_threads == 1
        bindings.append(True)
        return load(*args, **kwargs)

    monkeypatch.setattr(torch.compiler, "load_compiled_function", bind)

    def close(pool, *args):
        shutdown(pool, *args)
        pools.append(pool)

    monkeypatch.setattr(CompilePool, "__exit__", close)
    for warm in (False, True):
        forward = BufferForward(73)
        forward.output = deferred_tensor((32,), dtype=torch.float32, device="cuda:0", mode=deferred_mode())
        def materialize():
            assert len(pools) == 1 + int(warm)
            forward.output = torch.empty(32, device="cuda")
        if warm:
            def unexpected(*args, **kwargs):
                raise AssertionError("cached deferred compilation must not trace or start workers")
            monkeypatch.setattr(aot, "aot_forward", unexpected)
            monkeypatch.setattr(CompilePool, "submit", unexpected)
        compiled = cache.prepare_many({"decode": forward}, {"decode": GraphSignature("decode", 1, 1)},
                                      materialize=materialize)
        assert bindings and torch._inductor.config.compile_threads == original_threads
        torch.testing.assert_close(forward.output, forward.source + 73.25)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
            compiled["decode"]()
        forward.source.fill_(2)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(forward.output, torch.full_like(forward.output, 75.25))
        graph.reset()


@torch.no_grad()
def test_compiled_forward_rebinds_cuda_device_without_tracing(tmp_path, monkeypatch):
    from ignition.preparation.device_binding import bind_device

    if torch.cuda.device_count() < 2:
        pytest.skip("requires two assigned GPUs")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    with torch.cuda.device(0):
        original = BufferForward(19)
        compiled = aot_forward(original)
        compiled(original)
        path = tmp_path / "forward.bin"
        compiled.save_compiled_function(str(path))

    def unexpected(*args, **kwargs):
        raise AssertionError("device rebinding must not trace or compile")

    monkeypatch.setattr(torch, "compile", unexpected)
    with torch.cuda.device(1):
        fresh = BufferForward(73)
        with monkeypatch.context() as loading:
            loading.setattr(triton, "compile", unexpected)
            with bind_device("cuda:1"), _guard_code_cache(tmp_path / "guards"), path.open("rb") as file:
                restored = torch.compiler.load_compiled_function(file)
        restored(fresh)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.Stream(device=1)):
            restored(fresh)
        for value in (2., -3.):
            fresh.source.fill_(value)
            allocations = [torch.cuda.memory_stats(device)["allocation.all.allocated"] for device in (0, 1)]
            graph.replay()
            torch.cuda.synchronize()
            assert allocations == [torch.cuda.memory_stats(device)["allocation.all.allocated"] for device in (0, 1)]
            torch.testing.assert_close(fresh.output, torch.full_like(fresh.output, value + 73.25))
            torch.testing.assert_close(fresh.before, fresh.alias, rtol=0, atol=0)


@torch.no_grad()
def test_uncompiled_capture_bypasses_forward_cache_and_replays_without_allocating(monkeypatch):
    from ignition.preparation import capture

    def unexpected(*args, **kwargs):
        raise AssertionError("uncompiled capture must not load or build compiled forwards")

    monkeypatch.setattr(capture, "ForwardCache", unexpected)
    monkeypatch.setattr(torch, "compile", unexpected)
    forward = BufferForward(73)
    graphs, compiled = capture.capture_forwards(
        {"decode1": forward}, {"decode1": SimpleNamespace(kind="decode")},
        session=SimpleNamespace(capture=nullcontext), device=torch.device("cuda:0"),
        stream=torch.cuda.Stream(), pool=torch.cuda.graph_pool_handle(), compile_decoder=False,
    )
    assert compiled == {}
    for value in (1., -2.):
        forward.source.fill_(value)
        forward.output.fill_(float("nan"))
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
        graphs["decode1"].replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
        torch.testing.assert_close(forward.output, torch.full_like(forward.output, value + 73.25))


@pytest.mark.parametrize("handle", [0, 1, 73])
@torch.no_grad()
def test_restored_handles_aliases_and_triton_registration(handle, tmp_path):
    original = BufferForward(handle)
    compiled = aot_forward(original)
    compiled(original)
    path = tmp_path / "forward.bin"
    compiled.save_compiled_function(str(path))
    unrelated = kernel_side_table.add_kernel(_unrelated)
    with path.open("rb") as file:
        restored = torch.compiler.load_compiled_function(file)
    assert kernel_side_table.get_kernel(unrelated) is _unrelated
    for handle in (127, 1, 0):
        fresh = BufferForward(handle)
        returned = restored(fresh)
        assert returned.data_ptr() == fresh.output.data_ptr()
        assert returned.shape == fresh.output.shape and returned.stride() == fresh.output.stride()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            restored(fresh)
        for _ in range(2):
            fresh.source.normal_()
            graph.replay()
            torch.testing.assert_close(fresh.output, fresh.source + handle + .25)
            torch.testing.assert_close(fresh.before, fresh.alias, rtol=0, atol=0)
        fresh.negate = True
        assert not restored.guard_check(fresh)
        graph.reset()


@torch.no_grad()
@pytest.mark.parametrize("rebind", [False, True])
@pytest.mark.parametrize("cached_guards", [False, True])
def test_restored_forward_rejects_changed_storage_aliases(tmp_path, rebind, cached_guards):
    original = BufferForward(19)
    compiled = aot_forward(original)
    compiled(original)
    path = tmp_path / "forward.bin"
    compiled.save_compiled_function(str(path))
    restoration = _guard_code_cache(tmp_path / "guards") if cached_guards else nullcontext()
    with restoration, path.open("rb") as file:
        restored = torch.compiler.load_compiled_function(file)
    fresh = BufferForward(23)
    different = torch.empty_like(fresh.storage).view(torch.float32)[16:48]
    if rebind:
        restored(fresh)
        fresh.part.set_(different)
    else:
        fresh.part = different
    with pytest.raises((ValueError, RuntimeError), match="storage aliases changed|GuardManager check failed"):
        restored(fresh)


@torch.no_grad()
def test_forward_cache_restores_without_tracing_and_rebuilds_changed_guards(tmp_path, monkeypatch):
    import ignition.preparation.aot as aot
    from ignition.preparation.execution import GraphSignature
    from torch._dynamo.guards import PyExprCSEPass

    cache = ForwardCache(namespace={'model': 'buffer'}, device='cuda:0', root=tmp_path)
    signature = GraphSignature('decode', 1, 1)
    from types import SimpleNamespace

    events = []
    progress = SimpleNamespace(update=events.append)
    first = cache.prepare(BufferForward(0), name='decode', signature=signature, progress=progress)
    assert any("tracing and compiling" in event for event in events)
    assert not first.cache_hit

    def forbid_trace(*args, **kwargs):
        raise AssertionError('warm forward cache must not trace')

    fresh = BufferForward(73)
    with monkeypatch.context() as patch:
        patch.setattr(aot, 'aot_forward', forbid_trace)
        patch.setattr(aot, 'make_fx', forbid_trace)
        patch.setattr(PyExprCSEPass, 'count', forbid_trace)
        events.clear()
        restored = cache.prepare(fresh, name='decode', signature=signature, progress=progress)
        assert any("restoring compiled" in event for event in events)
        assert any("warming restored" in event for event in events)
        assert not any("tracing and compiling" in event for event in events)
        assert restored.cache_hit
        torch.testing.assert_close(fresh.output, fresh.source + 73.25)

    fresh.negate = True
    rebuilt = cache.prepare(fresh, name='decode', signature=signature)
    assert not rebuilt.cache_hit
    torch.testing.assert_close(fresh.output, -(fresh.source + 73.25))


class BranchForward:
    """A layer whose optional concurrent-branch owner is absent when stream overlap is disabled."""

    def __init__(self, chain):
        self.source = torch.randn(32, device="cuda")
        self.output = torch.empty_like(self.source)
        self.chain = chain

    def __call__(self):
        self.output.copy_(self.source)
        if self.chain is not None:
            self.output.mul_(2)


@torch.no_grad()
def test_forward_cache_rebuilds_forwards_whose_branch_composition_changed(tmp_path):
    from ignition.components.projection_overlap import ConcurrentProjectionChain
    from ignition.preparation.execution import GraphSignature

    cache = ForwardCache(namespace={"model": "branches"}, device="cuda:0", root=tmp_path)
    signature = GraphSignature("decode", 1, 1)
    hits = []
    # Serial and overlapped preparations share a cache path; guards keep either from restoring the other.
    for overlap in (True, True, False, False, True):
        forward = BranchForward(object.__new__(ConcurrentProjectionChain) if overlap else None)
        hits.append(cache.prepare(forward, name="decode", signature=signature).cache_hit)
        torch.testing.assert_close(forward.output, forward.source * (2 if overlap else 1), rtol=0, atol=0)
    assert hits == [False, True, False, True, False]


@torch.no_grad()
def test_parallel_forward_preparation_captures_and_restores_without_workers(tmp_path, monkeypatch):
    import ignition.preparation.aot as aot
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.compile_pool import CompilePool
    from ignition.preparation.execution import GraphSignature

    functions = {f"decode{index}": BufferForward(handle) for index, handle in enumerate((0, 73, 127), 1)}
    functions["decode2"].negate = True
    functions["decode1"].source = torch.nn.Parameter(functions["decode1"].source, requires_grad=False)
    signatures = {name: GraphSignature("decode", index, 1) for index, name in enumerate(functions, 1)}
    session = SimpleNamespace(capture=nullcontext, namespace=SimpleNamespace(to_dict=lambda: {"model": "parallel-buffer"}))
    options = dict(session=session, device=torch.device("cuda:0"), stream=torch.cuda.Stream(),
                   pool=torch.cuda.graph_pool_handle(), cache_root=tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("parent must not trace or lower prepared forwards")

    monkeypatch.setattr(aot, "aot_forward", forbidden)
    monkeypatch.setattr(aot, "make_fx", forbidden)
    graphs, compiled = capture_forwards(functions, signatures, **options)
    assert len(compiled) == 3 and all(not value.cache_hit for value in compiled.values())
    for _ in range(2):
        for name, forward in functions.items():
            forward.source.normal_()
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graphs[name].replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            expected = forward.source + forward.handle + .25
            torch.testing.assert_close(forward.output, -expected if forward.negate else expected)
            torch.testing.assert_close(forward.before, forward.alias, rtol=0, atol=0)
    for graph in graphs.values():
        graph.reset()

    def unexpected(*args, **kwargs):
        raise AssertionError("warm preparation must not trace or submit compilation")

    monkeypatch.setattr(aot, "aot_forward", unexpected)
    monkeypatch.setattr(CompilePool, "submit", unexpected)
    fresh = {name: BufferForward(19) for name in functions}
    fresh["decode2"].negate = True
    fresh["decode1"].source = torch.nn.Parameter(fresh["decode1"].source, requires_grad=False)
    restored_graphs, restored = capture_forwards(fresh, signatures, **options)
    assert all(value.cache_hit for value in restored.values())
    for name, forward in fresh.items():
        restored_graphs[name].replay()
        expected = forward.source + 19.25
        torch.testing.assert_close(forward.output, -expected if forward.negate else expected)


@torch.no_grad()
def test_explicit_worker_budget_compiles_and_replays_a_catalog(tmp_path, monkeypatch):
    from ignition.preparation import compile_pool
    from ignition.preparation.execution import GraphSignature

    workers, submitted = [], []
    initialize = compile_pool.CompilePool.__init__
    submit = compile_pool.CompilePool.submit

    def record(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        workers.append((self.workers, self.options['compile_threads']))

    def dispatch(self, forward, path):
        assert not path.exists()
        submitted.append(path)
        return submit(self, forward, path)

    monkeypatch.setattr(compile_pool.CompilePool, '__init__', record)
    monkeypatch.setattr(compile_pool.CompilePool, 'submit', dispatch)
    functions = {}
    for rows in range(1, 10):
        forward = ElementwiseForward()
        forward.source = torch.randn(rows * 1024, device='cuda')
        forward.output = torch.empty_like(forward.source)
        functions[f'rows{rows}'] = forward
    signatures = {name: GraphSignature('prefill', 1, forward.source.numel())
                  for name, forward in functions.items()}
    cache = ForwardCache(namespace={'model': 'worker-catalog'}, device='cuda:0',
                         root=tmp_path, compile_workers=9)
    compiled = cache.prepare_many(functions, signatures, materialize=lambda: None)
    assert workers == [(9, 1)]
    assert len(submitted) == len(functions)
    assert all(path.is_file() for path in submitted)
    graph = torch.cuda.CUDAGraph()
    for name, forward in functions.items():
        with torch.cuda.graph(graph):
            compiled[name]()
        for _ in range(3):
            forward.source.normal_()
            forward.output.fill_(float('nan'))
            allocated = torch.cuda.memory_stats()['allocation.all.allocated']
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
            torch.testing.assert_close(forward.output, forward.source.sin() * forward.source)
        graph.reset()


@torch.no_grad()
@pytest.mark.parametrize("deferred", [False, True])
def test_restore_checks_storage_before_gpu_work(tmp_path, deferred):
    from ignition.preparation.execution import GraphSignature

    cache = ForwardCache(namespace={"model": "alias-restore"}, device="cuda:0", root=tmp_path)
    signature = GraphSignature("decode", 1, 1)
    cache.prepare(BufferForward(0), name="decode", signature=signature)
    fresh = BufferForward(23)
    fresh.output.fill_(float("nan"))
    restored, path = cache._restore(fresh, name="decode", signature=signature, deferred=deferred)
    assert restored is not None
    assert torch.isnan(fresh.output).all()
    fresh.part = torch.empty_like(fresh.storage).view(torch.float32)[16:48]
    restored, rejected_path = cache._restore(fresh, name="decode", signature=signature, deferred=deferred)
    assert restored is None
    assert rejected_path == path
    assert torch.isnan(fresh.output).all()
    rebuilt = cache.prepare(fresh, name="decode", signature=signature)
    assert not rebuilt.cache_hit
    torch.testing.assert_close(fresh.output, fresh.source + 23.25)


@torch.no_grad()
@pytest.mark.parametrize("coordinated", [False, True])
def test_catalog_recompiles_changed_aliases_before_warmup(tmp_path, monkeypatch, coordinated):
    import torch.distributed as dist
    from ignition.preparation.collectives import CollectivePreparation
    from ignition.preparation.compile_pool import CompilePool
    from ignition.preparation.execution import GraphSignature

    submitted = []
    submit = CompilePool.submit
    def record(pool, forward, path):
        submitted.append(path)
        return submit(pool, forward, path)
    monkeypatch.setattr(CompilePool, "submit", record)
    dist.init_process_group("gloo", init_method="file://" + str(tmp_path / "rank"), rank=0, world_size=1)
    try:
        coordinator = CollectivePreparation(dist.group.WORLD)
        cache = ForwardCache(namespace={"model": "catalog-aliases"}, device="cuda:0", root=tmp_path,
                             compile_workers=1)
        signatures = {"decode": GraphSignature("decode", 1, 1)}
        hits, compilations = [], []
        for separate in [False, False, True, True, False]:
            forward = BufferForward(23)
            if separate:
                forward.part = torch.empty_like(forward.storage).view(torch.float32)[16:48]
            compiled = cache.prepare_many({"decode": forward}, signatures,
                coordinator=coordinator if coordinated else None, materialize=lambda: None)
            hits.append(compiled["decode"].cache_hit)
            compilations.append(len(submitted))
            torch.testing.assert_close(forward.output, forward.source + 23.25)
        assert compilations == [1, 1, 2, 2, 3]
        if coordinated:
            assert hits == [False, True, False, True, False]
    finally:
        dist.destroy_process_group()
