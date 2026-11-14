# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Owner-process tuning freezes actual generated kernels across TP ranks and reloads."""

import copy
import gc
import os
import pickle
import weakref
from contextlib import contextmanager
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
import torch

from ignition.preparation.aot import BindingMismatch, ForwardCache
from ignition.preparation.compilation import _OPTIONS
from ignition.preparation.execution import GraphSignature
from ignition.preparation import inductor_tuning


pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1',
                               reason='requires explicitly assigned CUDA devices')


class ArithmeticForward:
    def __init__(self):
        self.source = torch.full((3072,), .25, device='cuda')
        self.output = torch.empty_like(self.source)
        self.rows = torch.ones((128, 4096), device='cuda')
        self.sums = torch.empty(128, device='cuda')

    def __call__(self):
        self.output.copy_(self.source.sin() * self.source)
        self.sums.copy_(self.rows.sum(dim=1))


class VariantArithmeticForward(ArithmeticForward):
    def __init__(self, rank):
        super().__init__()
        self.role_bias = rank
        self.compilation_variant = f'rank-{rank}'

    def __call__(self):
        self.output.copy_(self.source.sin() * self.source)
        self.sums.copy_(self.rows.sum(dim=1) + self.role_bias)


@torch.library.custom_op('test_inductor_tuning::opaque', mutates_args=('output',))
def opaque(source: torch.Tensor, output: torch.Tensor) -> None:
    output.copy_(source)


@opaque.register_fake
def fake_opaque(source, output):
    pass


class OpaqueForward:
    def __init__(self):
        self.source = torch.ones(32, device='cuda')
        self.output = torch.empty_like(self.source)

    def __call__(self):
        opaque(self.source, self.output)


class VisionPixelValidationForward:
    """Nemotron-H's 1024-patch RADIO pixel validation and inactive-row zeroing."""

    def __init__(self):
        self.pixels = torch.full((1024, 3 * 16 * 16), .25, device='cuda')
        self.count = torch.full((1,), 1024, dtype=torch.int32, device='cuda')
        self.errors = torch.empty(1, dtype=torch.int32, device='cuda')
        self.patches = torch.empty_like(self.pixels)

    def __call__(self):
        capacity = self.pixels.shape[0]
        live = torch.arange(capacity, device=self.count.device) < self.count
        valid = (self.count > 0) & (self.count <= capacity)
        valid &= (~live[:, None] | torch.isfinite(self.pixels)).all()
        self.errors.copy_(~valid)
        self.patches.copy_(torch.where(valid & live[:, None], self.pixels, 0))


@torch.no_grad()
@pytest.mark.parametrize('distributed', [False, True], ids=['single', 'tp2'])
def test_cooperative_vision_validation_freezes_and_restores(tmp_path, monkeypatch, request, distributed):
    import ctypes as C

    from torch._inductor.async_compile import CompiledTritonKernels
    from torch._inductor.codecache import PyCodeCache
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from ignition.preparation.runtime_core import _Cuda

    coordinator = None
    root = tmp_path
    if distributed:
        import torch.distributed as dist
        from ignition.preparation.collectives import CollectivePreparation

        rank = request.getfixturevalue('distributed_group')
        roots = [str(tmp_path) if rank == 0 else None]
        dist.broadcast_object_list(roots, src=0)
        root = Path(roots[0])
        coordinator = CollectivePreparation(dist.group.WORLD)
    device = torch.device('cuda', torch.cuda.current_device())
    inventories = []

    class Probe(inductor_tuning.InductorSelections):
        def register(self, bound):
            graph = bound.backend.artifact._flare_graph
            tuners = [value for value in inductor_tuning._kernel_globals(graph).values()
                      if isinstance(value, CachingAutotuner)]
            cooperative = [tuner for tuner in tuners if tuner.triton_meta.get('launch_cooperative_grid')]
            assert cooperative, 'the vision validation must generate a cooperative reduction'
            assert all(not tuner.triton_meta.get('launch_pdl', False) for tuner in cooperative)
            super().register(bound)

        def freeze(self, bound):
            super().freeze(bound)
            results = {}
            for identity, _, tuner in self.forwards[id(bound)][2].values():
                result = inductor_tuning._selected(tuner)
                results[identity] = result._flare_config_state, result.kernel.cubin_raw
            assert results, 'the vision validation must retain a nonempty kernel inventory'
            if coordinator is not None:
                peers = [None] * dist.get_world_size()
                dist.all_gather_object(peers, results)
                assert all(peer == results for peer in peers)
            inventories.append(results)

    monkeypatch.setattr(inductor_tuning, 'InductorSelections', Probe)
    cache = ForwardCache(namespace={'model': 'cooperative-vision-validation'}, device=device,
        root=root, compile_workers=1 if coordinator is None else coordinator.local_ranks,
        lock_gpu_clocks=True, options=dict(_OPTIONS, autotune_local_cache=False, autotune_remote_cache=False))
    signature = GraphSignature('vision', 1024, 1)

    def prepare(forward):
        if coordinator is None:
            return cache.prepare(forward, name='vision1024', signature=signature)
        return cache.prepare_many({'vision1024': forward}, {'vision1024': signature},
            coordinator=coordinator, materialize=lambda: None)['vision1024']

    def replay(bound, forward):
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            bound()
        cuda, count = _Cuda(), C.c_size_t()
        raw = C.c_void_p(graph.raw_cuda_graph())
        cuda.call('cuGraphGetNodes', raw, None, C.byref(count))
        nodes = (C.c_void_p * count.value)()
        cuda.call('cuGraphGetNodes', raw, nodes, C.byref(count))
        cooperative = 0
        for node in nodes:
            kind = C.c_int()
            cuda.call('cuGraphNodeGetType', C.c_void_p(node), C.byref(kind))
            if kind.value == 0:  # CU_GRAPH_NODE_TYPE_KERNEL.
                attribute = (C.c_uint64 * 8)()
                # CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE uses CUlaunchAttributeValue.
                cuda.call('cuGraphKernelNodeGetAttribute', C.c_void_p(node), C.c_int(2), C.byref(attribute))
                cooperative += attribute[0] == 1
        assert cooperative > 0, 'capture must retain the cooperative launch attribute'
        for live_count, nan_row in ((1024, None), (511, 1000), (511, 1), (0, None), (1024, None)) * 2:
            pixels = torch.full((1024, 768), .25)
            if nan_row is not None:
                pixels[nan_row, 0] = float('nan')
            live = torch.arange(1024) < live_count
            valid = (0 < live_count <= 1024) and bool((~live[:, None] | torch.isfinite(pixels)).all())
            expected = torch.where(valid & live[:, None], pixels, 0)
            forward.pixels.copy_(pixels)
            forward.count.fill_(live_count)
            forward.errors.fill_(-1)
            forward.patches.fill_(float('nan'))
            allocated = torch.cuda.memory_stats()['allocation.all.allocated']
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
            assert forward.errors.item() == int(not valid)
            torch.testing.assert_close(forward.patches.cpu(), expected, rtol=0, atol=0)

    forward = VisionPixelValidationForward()
    bound = prepare(forward)
    winners, = inventories
    inventories.clear()
    replay(bound, forward)
    CompiledTritonKernels.cache_clear()
    PyCodeCache.cache_clear()

    def forbidden(*args, **kwargs):
        raise AssertionError('frozen cooperative kernel compiled or benchmarked on restore')

    monkeypatch.setattr(CachingAutotuner, 'bench', forbidden)
    monkeypatch.setattr(CachingAutotuner, '_precompile_config', forbidden)
    fresh = VisionPixelValidationForward()
    restored = prepare(fresh)
    assert restored.cache_hit
    assert inventories == [winners]
    replay(restored, fresh)


def _portable(result):
    result = copy.copy(result)
    result.kernel = copy.copy(result.kernel)
    result._flare_config_state = copy.deepcopy(vars(result.config))
    result.reload_cubin_path()
    if result.kernel.cubin_raw is None:
        result.kernel.cubin_raw = Path(result.kernel.cubin_path).read_bytes()
    return pickle.dumps(result)


def _replay(bound, forward):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        bound()
    for value in (.125, .5):
        forward.source.fill_(value)
        forward.rows.fill_(value)
        forward.output.fill_(float('nan'))
        forward.sums.fill_(float('nan'))
        allocated = torch.cuda.memory_stats()['allocation.all.allocated']
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocated
        torch.testing.assert_close(forward.output, forward.source.sin() * forward.source)
        expected_sum = 4096 * value + getattr(forward, 'role_bias', 0)
        torch.testing.assert_close(forward.sums, torch.full_like(forward.sums, expected_sum), rtol=0, atol=0)


class MutatingForward:
    def __init__(self):
        self.state = torch.empty(4096, device='cuda')
        self.output = torch.empty_like(self.state)

    def __call__(self):
        self.state.add_(1)
        self.output.copy_(self.state * 3)


@torch.no_grad()
@pytest.mark.parametrize('deferred', [False, True])
@pytest.mark.parametrize('distributed', [False, True], ids=['single', 'tp2'])
def test_input_reset_survives_tuning_and_frozen_warmup(tmp_path, request, deferred, distributed):
    coordinator = None
    root = tmp_path
    if distributed:
        import torch.distributed as dist
        from ignition.preparation.collectives import CollectivePreparation

        rank = request.getfixturevalue('distributed_group')
        roots = [str(tmp_path) if rank == 0 else None]
        dist.broadcast_object_list(roots, src=0)
        root = Path(roots[0])
        coordinator = CollectivePreparation(dist.group.WORLD)
    cache = ForwardCache(namespace={'model': 'mutable-warmup-inputs'},
        device=torch.device('cuda', torch.cuda.current_device()), root=root,
        compile_workers=1 if coordinator is None else coordinator.local_ranks)
    signatures = {'decode': GraphSignature('decode', 1, 1)}
    for restored in (False, True):
        forward = MutatingForward()
        bound = cache.prepare_many({'decode': forward}, signatures, coordinator=coordinator,
            materialize=(lambda: None) if deferred else None,
            before_run=lambda name: forward.state.fill_(1))['decode']
        if restored:
            assert bound.cache_hit
        assert bound.backend._tuning_frozen
        torch.testing.assert_close(forward.state, torch.full_like(forward.state, 2), rtol=0, atol=0)
        torch.testing.assert_close(forward.output, torch.full_like(forward.output, 6), rtol=0, atol=0)


@torch.no_grad()
def test_single_rank_default_tuning_is_frozen_on_reload(tmp_path, monkeypatch):
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    cache = ForwardCache(namespace={'model': 'frozen-owner-tuning'}, device='cuda:0', root=tmp_path,
                         compile_workers=1, options=dict(_OPTIONS, autotune_local_cache=False))
    signature = GraphSignature('decode', 1, 1)
    forward = ArithmeticForward()
    bound = cache.prepare(forward, name='decode', signature=signature)
    assert bound.backend._tuning_frozen
    _replay(bound, forward)

    def forbidden(*args, **kwargs):
        raise AssertionError('cached frozen kernel reopened autotuning')

    monkeypatch.setattr(CachingAutotuner, 'bench', forbidden)
    fresh = ArithmeticForward()
    restored = cache.prepare(fresh, name='decode', signature=signature)
    assert restored.cache_hit
    _replay(restored, fresh)
    opaque_forward = OpaqueForward()
    opaque_bound = cache.prepare(opaque_forward, name='opaque', signature=signature)
    opaque_bound()
    torch.testing.assert_close(opaque_forward.output, opaque_forward.source, rtol=0, atol=0)


@torch.no_grad()
def test_frozen_restore_ignores_conflicting_enabled_local_cache(tmp_path, monkeypatch, request):
    from torch._inductor.async_compile import CompiledTritonKernels
    from torch._inductor.codecache import PyCodeCache
    from torch._inductor.runtime.autotune_cache import AutotuneCache
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner, hash_configs

    probes = []

    class Probe(inductor_tuning.InductorSelections):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.candidates = {}
            probes.append(self)

        def register(self, bound):
            super().register(bound)
            for identity, _, tuner in self.forwards[id(bound)][2].values():
                self.candidates[identity] = [_portable(result) for result in tuner.compile_results]

    monkeypatch.setattr(inductor_tuning, 'InductorSelections', Probe)
    cache = ForwardCache(namespace={'model': 'frozen-local-cache-conflict'}, device='cuda:0',
        root=tmp_path, compile_workers=1, options=dict(_OPTIONS, autotune_local_cache=True,
        autotune_remote_cache=False, coordinate_descent_tuning=True))
    signature = GraphSignature('decode', 1, 1)
    forward = ArithmeticForward()
    bound = cache.prepare(forward, name='decode', signature=signature)
    probe = probes[-1]
    winners = dict(probe.winners)
    conflicts = 0
    for identity, instances in probe.instances.items():
        tuner = next(iter(instances.values()))
        selected = inductor_tuning._selected(tuner)
        alternatives = [pickle.loads(data).config for data in probe.candidates[identity]
                        if pickle.loads(data).config.all_kwargs() != selected.config.all_kwargs()]
        if not alternatives:
            continue
        metadata = dict(tuner.inductor_meta, autotune_local_cache=True,
                        autotune_remote_cache=False, coordinate_descent_tuning=True)
        local_cache = AutotuneCache.create(metadata, tuner.filename, hash_configs([selected.config]))
        assert local_cache is not None and local_cache.local_cache is not None
        cache_path = Path(local_cache.local_cache[1])
        original = cache_path.read_bytes() if cache_path.exists() else None

        def restore_cache(path=cache_path, contents=original):
            if contents is None:
                path.unlink(missing_ok=True)
            else:
                from torch._inductor.codecache import write_atomic
                write_atomic(str(path), contents)

        request.addfinalizer(restore_cache)
        local_cache.save(alternatives[0], 0, found_by_coordesc=True)
        cached = local_cache.read_best(metadata, [selected.config])
        assert cached is not None and cached.all_kwargs() == alternatives[0].all_kwargs()
        assert cached.all_kwargs() != selected.config.all_kwargs()
        conflicts += 1
    assert conflicts > 0, 'fixture requires a real alternative compiled config'
    _replay(bound, forward)

    # Reload generated modules and static tuners from the serialized artifact.
    CompiledTritonKernels.cache_clear()
    PyCodeCache.cache_clear()

    def forbidden(*args, **kwargs):
        raise AssertionError('frozen artifact compiled or benchmarked against a local cache choice')

    monkeypatch.setattr(CachingAutotuner, 'bench', forbidden)
    monkeypatch.setattr(CachingAutotuner, '_precompile_config', forbidden)
    fresh = ArithmeticForward()
    restored = cache.prepare(fresh, name='decode', signature=signature)
    assert restored.cache_hit
    for identity, encoded in probes[-1].winners.items():
        expected, actual = pickle.loads(winners[identity]), pickle.loads(encoded)
        assert actual._flare_config_state == expected._flare_config_state
        assert actual.kernel.cubin_raw == expected.kernel.cubin_raw
    _replay(restored, fresh)


@torch.no_grad()
def test_clock_lock_covers_only_actual_inductor_trials(tmp_path, monkeypatch):
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from ignition.preparation import clocks

    original_lock, original_bench = clocks.locked_gpu_clocks, CachingAutotuner.bench
    active, locks, trials = 0, [], []

    @contextmanager
    def observed_lock(uuid):
        nonlocal active
        assert active == 0
        with original_lock(uuid):
            active += 1
            locks.append(uuid)
            try:
                yield
            finally:
                active -= 1

    def observed_bench(self, *args, **kwargs):
        assert active == 1
        trials.append(self.inductor_meta['kernel_name'])
        return original_bench(self, *args, **kwargs)

    class Probe(inductor_tuning.InductorSelections):
        @contextmanager
        def _tuning(self, bindings, *, frozen):
            assert active == 0
            before = len(locks)
            with super()._tuning(bindings, frozen=frozen):
                yield
            assert active == 0
            if frozen:
                assert len(locks) == before

    monkeypatch.setattr(clocks, 'locked_gpu_clocks', observed_lock)
    monkeypatch.setattr(CachingAutotuner, 'bench', observed_bench)
    monkeypatch.setattr(inductor_tuning, 'InductorSelections', Probe)
    cache = ForwardCache(namespace={'model': 'inductor-clock-scope'}, device='cuda:0', root=tmp_path,
        compile_workers=1, lock_gpu_clocks=True, options=dict(_OPTIONS, autotune_local_cache=False,
        autotune_remote_cache=False))
    signature = GraphSignature('decode', 1, 1)
    forward = ArithmeticForward()
    bound = cache.prepare(forward, name='decode', signature=signature)
    assert locks and trials and active == 0
    _replay(bound, forward)
    count = len(locks)
    fresh = ArithmeticForward()
    restored = cache.prepare(fresh, name='decode', signature=signature)
    assert restored.cache_hit and len(locks) == count and active == 0
    _replay(restored, fresh)


@torch.no_grad()
def test_long_generated_kernel_names_freeze_and_restore(tmp_path, monkeypatch):
    from torch._inductor.async_compile import CompiledTritonKernels
    from torch._inductor.choices import InductorChoices
    from torch._inductor.codecache import PyCodeCache
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner, StaticTritonCompileResult
    from torch._inductor.virtualized import V
    from ignition.preparation.aot import BoundForward, _guard_code_cache, aot_forward

    class LongNames(InductorChoices):
        def customize_fused_kernel_name(self, fused_name, src_code):
            return 'long_generated_kernel_name_' * 10 + fused_name

    cache = ForwardCache(namespace={'model': 'long-generated-kernel-names'}, device='cuda:0', root=tmp_path,
        compile_workers=1, lock_gpu_clocks=True, options=dict(_OPTIONS, compile_threads=1,
        fx_graph_cache=False, autotune_local_cache=False, autotune_remote_cache=False))
    signature = GraphSignature('decode', 1, 1)
    forward = ArithmeticForward()
    path = cache._path(forward, name='decode', signature=signature)
    with V.set_choices_handler(LongNames()), _guard_code_cache(cache.root / 'guards'):
        artifact = aot_forward(forward, options=cache.options)
        bound = cache._warm(BoundForward(artifact, forward, False), path, name='decode')

    def selected(bound):
        selections = inductor_tuning.InductorSelections('cuda:0')
        selections.register(bound)
        results = {}
        for identity, instances in selections.instances.items():
            tuner = next(iter(instances.values()))
            assert tuner.is_statically_launchable()
            result = inductor_tuning._selected(tuner)
            assert isinstance(result, StaticTritonCompileResult)
            assert len(result.kernel.name) > 255
            result.reload_cubin_path()
            assert Path(result.kernel.cubin_path).name == result.kernel.name[:150] + '.cubin'
            results[identity] = result._flare_config_state, result.kernel.cubin_raw
        assert results
        return results

    winners = selected(bound)
    _replay(bound, forward)
    CompiledTritonKernels.cache_clear()
    PyCodeCache.cache_clear()

    def forbidden(*args, **kwargs):
        raise AssertionError('frozen long-name kernel compiled or benchmarked on restore')

    monkeypatch.setattr(CachingAutotuner, 'bench', forbidden)
    monkeypatch.setattr(CachingAutotuner, '_precompile_config', forbidden)
    fresh = ArithmeticForward()
    restored = cache.prepare(fresh, name='decode', signature=signature)
    assert restored.cache_hit
    assert selected(restored) == winners
    _replay(restored, fresh)


@torch.no_grad()
def test_unreferenced_bundled_candidates_are_dropped_and_named_kernels_still_bind(tmp_path, monkeypatch):
    """Epilogue-fusion benchmarking can bundle a template candidate that final code never names."""
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from torch._inductor.utils import Placeholder

    bundled = {}

    def probe(candidate):
        class Probe(inductor_tuning.InductorSelections):
            def register(self, bound):
                entries = bound.backend.artifact._flare_graph._triton_bundle.static_autotuners
                if not bound.cache_hit:
                    entries.append(candidate(entries[0]))
                bundled[bound.cache_hit] = {entry.cache_key for entry in entries}
                super().register(bound)
        return Probe

    def cache(name):
        return ForwardCache(namespace={'model': name}, device='cuda:0', root=tmp_path / name, compile_workers=1,
            options=dict(_OPTIONS, autotune_local_cache=False, autotune_remote_cache=False))

    signature = GraphSignature('decode', 1, 1)
    monkeypatch.setattr(inductor_tuning, 'InductorSelections', probe(lambda entry: replace(
        entry, cache_key='unreferenced-candidate', kernel_name=str(Placeholder.DESCRIPTIVE_NAME))))
    accepted = cache('unreferenced-candidate')
    forward = ArithmeticForward()
    bound = accepted.prepare(forward, name='decode', signature=signature)
    assert 'unreferenced-candidate' in bundled[False]
    _replay(bound, forward)

    def forbidden(*args, **kwargs):
        raise AssertionError('frozen kernel compiled or benchmarked on restore')

    with monkeypatch.context() as frozen:
        frozen.setattr(CachingAutotuner, 'bench', forbidden)
        frozen.setattr(CachingAutotuner, '_precompile_config', forbidden)
        fresh = ArithmeticForward()
        restored = accepted.prepare(fresh, name='decode', signature=signature)
    assert restored.cache_hit
    assert bundled[True] == bundled[False] - {'unreferenced-candidate'}
    _replay(restored, fresh)

    monkeypatch.setattr(inductor_tuning, 'InductorSelections', probe(lambda entry: replace(
        entry, cache_key='named-unbound-kernel')))
    with pytest.raises(ValueError, match='unbound static Inductor kernels'):
        cache('named-unbound-kernel').prepare(ArithmeticForward(), name='decode', signature=signature)


@pytest.fixture(scope='module')
def distributed_group():
    import torch.distributed as dist

    if int(os.environ.get('WORLD_SIZE', '1')) < 2:
        pytest.skip('launch with torchrun and at least two ranks')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('gloo', timeout=timedelta(minutes=5))
    try:
        yield dist.get_rank()
    finally:
        dist.destroy_process_group()


@torch.no_grad()
@pytest.mark.parametrize('distinct_variants', [False, True])
def test_distributed_catalog_reconciles_real_conflicting_candidates(
        tmp_path, monkeypatch, distinct_variants, distributed_group):
    """Run under torchrun across the assigned GPUs on one or more hosts."""
    import torch.distributed as dist
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from ignition.preparation.collectives import CollectivePreparation

    rank = distributed_group
    root = [str(tmp_path) if rank == 0 else None]
    dist.broadcast_object_list(root, src=0)
    coordinator = CollectivePreparation(dist.group.WORLD)
    probes = []

    class Probe(inductor_tuning.InductorSelections):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.candidates = {}
            self.injected = 0
            self.portable_artifacts = []
            self.guard_wrappers = []
            self.diff_guard_roots = []
            probes.append(self)

        def register(self, bound):
            super().register(bound)
            self.portable_artifacts.append(weakref.ref(bound.artifact))
            wrapper = bound.artifact._artifacts.guard_manager
            root = wrapper.root
            self.guard_wrappers.append(weakref.ref(wrapper))
            assert wrapper.diff_guard_root is not None
            self.diff_guard_roots.append(weakref.ref(wrapper.diff_guard_root))
            arguments = {"forward": bound.owner}
            assert wrapper.check(arguments) == root.check(arguments) is True
            source = bound.owner.source
            try:
                bound.owner.source = source[:-1]
                assert wrapper.check(arguments) == root.check(arguments) is False
                assert str(wrapper.check_verbose(arguments)) == str(root.check_verbose(arguments))
            finally:
                bound.owner.source = source
            with torch.enable_grad():
                assert wrapper.check(arguments) == root.check(arguments)
            for identity, _, tuner in self.forwards[id(bound)][2].values():
                self.candidates.setdefault(identity, [_portable(result) for result in tuner.compile_results])

        def _reconcile(self, bindings):
            local = {key: tuner for key, _, tuner in bindings.values()}
            choices = self._gather({key: vars(inductor_tuning._selected(tuner).config)
                                   for key, tuner in local.items()})
            if rank == 1 and not self.winners:
                for key in sorted(local.keys() & choices[0].keys()):
                    alternatives = [data for data in self.candidates[key]
                                    if vars(pickle.loads(data).config) != choices[0][key]]
                    if alternatives:
                        # Seed a conflicting valid compiled cache choice; the production
                        # boundary must replace it with the canonical rank's measured winner.
                        inductor_tuning._install(local[key], alternatives[0])
                        self.injected += 1
                        break
                assert self.injected == 1, 'fixture requires a real multi-candidate kernel'
            super()._reconcile(bindings)

    monkeypatch.setattr(inductor_tuning, 'InductorSelections', Probe)
    cache = ForwardCache(namespace={'model': 'shared-owner-tuning'},
        device=torch.device('cuda', torch.cuda.current_device()), root=Path(root[0]),
        compile_workers=coordinator.local_ranks,
        options=dict(_OPTIONS, autotune_local_cache=False, autotune_remote_cache=False))
    signatures = {name: GraphSignature('decode', 1, 1) for name in ('decode', 'resume')}
    make_forward = (lambda: VariantArithmeticForward(rank)) if distinct_variants else ArithmeticForward
    forwards = {name: make_forward() for name in signatures}
    compiled = cache.prepare_many(forwards, signatures, coordinator=coordinator, materialize=lambda: None)
    assert len(probes[-1].winners) > 0
    gc.collect()
    assert all(reference() is None for reference in probes[-1].portable_artifacts)
    assert all(reference() is None for reference in probes[-1].diff_guard_roots)
    for reference in probes[-1].guard_wrappers:
        wrapper = reference()
        assert wrapper is not None
        assert not wrapper.code_parts and not wrapper.diff_guard_sources
        assert wrapper.verbose_code_parts is None
    for name in signatures:
        assert compiled[name].backend._tuning_frozen
        _replay(compiled[name], forwards[name])
        source = forwards[name].source
        try:
            forwards[name].source = source[:-1]
            with pytest.raises(BindingMismatch, match='compiled guards'):
                compiled[name]()
            forwards[name].source = forwards[name].output.view_as(source)
            with pytest.raises(BindingMismatch, match='storage aliases changed|compiled guards'):
                compiled[name]()
        finally:
            forwards[name].source = source

    def forbidden(*args, **kwargs):
        raise AssertionError('distributed frozen restore reopened autotuning')

    monkeypatch.setattr(CachingAutotuner, 'bench', forbidden)
    monkeypatch.setattr(inductor_tuning, 'InductorSelections', Probe.__bases__[0])
    fresh = {name: make_forward() for name in signatures}
    restored = cache.prepare_many(fresh, signatures, coordinator=coordinator, materialize=lambda: None)
    assert all(bound.cache_hit for bound in restored.values())
    for name in signatures:
        _replay(restored[name], fresh[name])
