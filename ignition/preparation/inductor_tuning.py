# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Agree and retain compiled Inductor kernel selections before graph capture."""

import ast
import copy
import inspect
import pickle
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path

import torch

from .device_binding import _device_source, _freeze_tuning_winners


def _selected(tuner):
    from torch._inductor.runtime.triton_heuristics import StaticTritonCompileResult

    if len(tuner.launchers) != 1:
        raise RuntimeError(f"Inductor kernel {tuner.inductor_meta['kernel_name']} has no final launcher")
    tuner._prune_compile_results_to_launcher(tuner.launchers[0])
    result, = tuner.compile_results
    if not isinstance(result, StaticTritonCompileResult):
        raise TypeError("prepared Inductor selections require a portable static kernel")
    result = copy.copy(result)
    result.kernel = copy.copy(result.kernel)
    result._flare_config_state = copy.deepcopy(vars(result.config))
    if result.kernel.cubin_raw is None:
        result.reload_cubin_path()
        result.kernel.cubin_raw = Path(result.kernel.cubin_path).read_bytes()
    return result


def _materialize_selected_cubin(result):
    from torch._inductor.codecache import write_atomic

    raw = result.kernel.cubin_raw
    if raw is None:
        raise ValueError('selected Inductor kernel lacks its compiled cubin bytes')
    result.reload_cubin_path()
    path = result.kernel.cubin_path
    if Path(path).read_bytes() != raw:
        write_atomic(path, raw)


def _install(tuner, encoded):
    result = pickle.loads(encoded)
    result.config.__dict__.clear()
    result.config.__dict__.update(result._flare_config_state)
    result.compile_meta = dict(result.compile_meta, device=tuner.triton_meta['device'])
    _materialize_selected_cubin(result)
    tuner.compile_results = [result]
    tuner.launchers = []
    tuner._cached_launcher = None
    tuner.inductor_meta = {**tuner.inductor_meta, 'dynamic_scale_rblock': False}
    tuner.__dict__.pop('_could_rblock_scale', None)
    tuner._make_launchers()


def _snapshot(entry, tuner):
    state = tuner.prepare_for_pickle()
    try:
        saved = copy.deepcopy(tuner)
    finally:
        tuner.restore_after_unpickle(state)
    saved._reload_kernel = None
    for result, live in zip(saved.compile_results, tuner.compile_results, strict=True):
        result._flare_config_state = copy.deepcopy(vars(live.config))
        result.config.__dict__.clear()
        result.config.__dict__.update(result._flare_config_state)
        if result.kernel.cubin_raw is None:
            result.reload_cubin_path()
            result.kernel.cubin_raw = Path(result.kernel.cubin_path).read_bytes()
    return replace(entry, kernel=saved)


def _kernel_globals(graph):
    pending, visited, matches = [graph.current_callable], set(), {}
    while pending:
        function = pending.pop()
        if inspect.ismethod(function):
            function = function.__func__
        if id(function) in visited or not inspect.isfunction(function):
            continue
        visited.add(id(function))
        if Path(function.__code__.co_filename).stem == graph.cache_key:
            matches[id(function.__globals__)] = function.__globals__
        for cell in function.__closure__ or ():
            pending.append(cell.cell_contents)
    if len(matches) != 1:
        raise RuntimeError("compiled forward does not retain one generated Python module")
    return next(iter(matches.values()))


class InductorSelections:
    """One canonical winner per exact generated kernel across a preparation catalog."""

    def __init__(self, device, *, coordinator=None, lock_gpu_clocks=False):
        self.device = torch.device(device)
        self.coordinator = coordinator
        self.lock_gpu_clocks = lock_gpu_clocks
        self.winners = {}
        self.instances = {}
        self.forwards = {}

    def _gather(self, value):
        if self.coordinator is None:
            return [value]
        import torch.distributed as dist

        values = [None] * len(self.coordinator.ranks)
        dist.all_gather_object(values, value, group=self.coordinator.group)
        return values

    def _check(self, failure):
        failures = self._gather(failure)
        if any(failures):
            raise RuntimeError(f"Inductor preparation failed across ranks: {failures}")

    def register(self, bound):
        from torch._inductor.async_compile import CompiledTritonKernels
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        backend = bound.artifact._artifacts.compiled_fn
        graph = backend.artifact._flare_graph
        original, = graph.device_idxs
        entries = ({} if graph._triton_bundle is None else
                   {entry.cache_key: entry for entry in graph._triton_bundle.static_autotuners})
        module = _kernel_globals(graph) if entries else {}
        bindings = {}
        for node in ast.walk(ast.parse(graph.source_code)):
            if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and ast.unparse(node.value.func) == 'async_compile.triton'):
                continue
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise ValueError("unsupported generated Inductor kernel assignment")
            source = ast.literal_eval(node.value.args[1])
            cache_key = CompiledTritonKernels.key(source)
            name = node.targets[0].id
            tuner = module.get(name)
            if not isinstance(tuner, CachingAutotuner) or cache_key not in entries:
                kind = f'{type(tuner).__module__}.{type(tuner).__qualname__}'
                raise TypeError(
                    f'generated Inductor kernel {name!r} in graph {graph.cache_key} lacks its static '
                    f'autotuner binding: live_type={kind}, source_key={cache_key}, '
                    f'bundled={cache_key in entries}, static_entries={len(entries)}')
            identity = CompiledTritonKernels.key(_device_source(source, original, 0))
            bindings[cache_key] = identity, entries[cache_key], tuner
            instances = self.instances.setdefault(identity, {})
            if id(tuner) not in instances:
                instances[id(tuner)] = tuner
                if identity in self.winners:
                    _install(tuner, self.winners[identity])
        # Epilogue-fusion benchmarking bundles template candidates that final code never names.
        if any(entries[key].kernel_name in graph.source_code for key in set(entries) - set(bindings)):
            raise ValueError("compiled forward has unbound static Inductor kernels")
        self.forwards[id(bound)] = backend, graph, bindings

    @contextmanager
    def _tuning(self, bindings, *, frozen):
        stack = None
        saved = []
        tuners = {id(tuner): tuner for _, _, tuner in bindings.values()}

        def selection(method):
            def run(*args, **kwargs):
                nonlocal stack
                if stack is not None:
                    return method(*args, **kwargs)
                with ExitStack() as active:
                    stack = active
                    try:
                        return method(*args, **kwargs)
                    finally:
                        stack = None
            return run

        def benchmark(method):
            def run(*args, **kwargs):
                if frozen:
                    raise RuntimeError("frozen Inductor kernel attempted to benchmark")
                if self.lock_gpu_clocks:
                    if stack is None:
                        raise RuntimeError("Inductor benchmark ran outside kernel selection")
                    if not getattr(stack, '_flare_clocks_locked', False):
                        from .clocks import locked_gpu_clocks
                        uuid = str(torch.cuda.get_device_properties(self.device).uuid)
                        stack.enter_context(locked_gpu_clocks(uuid))
                        stack._flare_clocks_locked = True
                return method(*args, **kwargs)
            return run

        def compile_result(tuner, method):
            def run(*args, **kwargs):
                result = method(*args, **kwargs)
                # Refinement searches otherwise retain only their launcher, which
                # cannot carry the selected cubin into another process.
                if result not in tuner.compile_results:
                    tuner.compile_results.append(result)
                return result
            return run

        try:
            for tuner in tuners.values():
                for name in ('autotune_to_one_config', '_combo_sequential_autotune',
                             'coordinate_descent_tuning', 'bench', '_precompile_config'):
                    previous = tuner.__dict__.get(name)
                    saved.append((tuner, name, previous, name in tuner.__dict__))
                    method = getattr(tuner, name)
                    wrapper = (compile_result(tuner, method) if name == '_precompile_config' else
                               benchmark(method) if name == 'bench' else selection(method))
                    setattr(tuner, name, wrapper)
            yield
        finally:
            for tuner, name, previous, present in reversed(saved):
                if present:
                    setattr(tuner, name, previous)
                else:
                    delattr(tuner, name)

    def warm(self, bound, warmup, *, before_run=None):
        bindings = self.forwards[id(bound)][2]
        inventory = sorted({identity for identity, _, _ in bindings.values()})
        variants = {}
        for variant, identities in self._gather((getattr(bound.owner, 'compilation_variant', 'shared'), inventory)):
            if variant in variants and variants[variant] != identities:
                raise RuntimeError(f"equivalent compiled forward variant {variant!r} has different Inductor kernels")
            variants[variant] = identities
        failure = None
        try:
            with warmup(), self._tuning(bindings, frozen=self.forwards[id(bound)][0]._tuning_frozen):
                if before_run is not None:
                    before_run()
                bound()
            torch.cuda.current_stream(self.device).synchronize()
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        self._check(failure)
        self._reconcile(bindings)
        failure = None
        try:
            with warmup(), self._tuning(bindings, frozen=True):
                if before_run is not None:
                    before_run()
                bound()
            torch.cuda.current_stream(self.device).synchronize()
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        self._check(failure)
        actual, failure = {}, None
        try:
            for identity, _, tuner in bindings.values():
                selected = _selected(tuner)
                actual[identity] = vars(selected.config), selected.kernel.cubin_raw
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        self._check(failure)
        gathered = self._gather(actual)
        agreed = {}
        for rank in gathered:
            for identity, selection in rank.items():
                if identity in agreed and agreed[identity] != selection:
                    raise RuntimeError("equivalent Inductor kernels disagree after frozen warmup")
                agreed[identity] = selection

    def _reconcile(self, bindings):
        import torch.distributed as dist

        local = {identity: tuner for identity, _, tuner in bindings.values()}
        inventories = self._gather(sorted(local))
        owners = {}
        for rank, identities in enumerate(inventories):
            for identity in identities:
                if identity not in self.winners:
                    owners.setdefault(identity, rank)
        me = 0 if self.coordinator is None else self.coordinator.ranks.index(dist.get_rank())
        payload, failure = {}, None
        try:
            for identity, owner in owners.items():
                if owner == me:
                    payload[identity] = pickle.dumps(_selected(local[identity]))
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        self._check(failure)
        for owner in sorted(set(owners.values())):
            message = [payload if owner == me else None]
            if self.coordinator is not None:
                dist.broadcast_object_list(message, src=self.coordinator.ranks[owner],
                                           group=self.coordinator.group)
            self.winners.update(message[0])
        failure = None
        try:
            for identity in set(local) | set(owners):
                for tuner in self.instances.get(identity, {}).values():
                    _install(tuner, self.winners[identity])
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        self._check(failure)

    def freeze(self, bound):
        import zlib

        backend, graph, bindings = self.forwards[id(bound)]
        snapshots = [_snapshot(entry, tuner) for _, entry, tuner in bindings.values()]
        bundle = graph._triton_bundle
        if bundle is not None:
            bundle = replace(bundle, static_autotuners=snapshots)
            _freeze_tuning_winners(bundle)
        entry, kernels, constants = pickle.loads(zlib.decompress(backend._serialized_artifact))
        entry.compiled_fw.result._triton_bundle = bundle
        backend._serialized_artifact = zlib.compress(pickle.dumps((entry, kernels, constants)), level=1)
        backend._tuning_frozen = True
