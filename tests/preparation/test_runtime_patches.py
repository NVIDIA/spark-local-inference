# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compiler fixes install independently in every preparation process."""

import subprocess
import sys
import textwrap

import pytest

from ignition import runtime_patches
from ignition.errors import PreparationError


_FACTOR = 3
_DEFAULT = object()


def _example(value, default=_DEFAULT, *, offset=2):
    def scale():
        return value * _FACTOR
    return scale() + offset


def test_checked_replacement_retains_live_globals_defaults_and_nested_functions(monkeypatch):
    replacement = runtime_patches._replace(_example, (("value * _FACTOR", "value * _FACTOR + 1"),))
    assert replacement(4) == 15
    assert _example(4) == 14
    monkeypatch.setattr(sys.modules[__name__], "_FACTOR", 5)
    assert replacement(4, offset=7) == 28
    assert replacement.__globals__ is _example.__globals__
    assert replacement.__defaults__ is _example.__defaults__
    assert replacement.__kwdefaults__ is _example.__kwdefaults__


@pytest.mark.parametrize("before", ["missing context", "value"])
def test_missing_or_ambiguous_context_is_rejected(before):
    with pytest.raises(PreparationError, match="context is missing or ambiguous"):
        runtime_patches._replace(_example, ((before, "replacement"),))


def test_incompatible_triton_version_is_rejected_before_installation(monkeypatch):
    monkeypatch.setattr(runtime_patches, "_TRITON_APPLIED", False)
    monkeypatch.setattr(runtime_patches, "version", lambda name: "3.7.0")
    with pytest.raises(PreparationError, match="require triton==3.8.0"):
        runtime_patches.apply_triton()
    assert not runtime_patches._TRITON_APPLIED


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_missing_triton_is_allowed_only_on_windows(monkeypatch, platform):
    from types import SimpleNamespace

    def missing(name):
        raise runtime_patches.PackageNotFoundError(name)

    monkeypatch.setattr(runtime_patches, "_TRITON_APPLIED", False)
    monkeypatch.setattr(runtime_patches, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(runtime_patches, "version", missing)
    if platform == "win32":
        runtime_patches.apply_triton()
    else:
        with pytest.raises(PreparationError, match="require the preparation dependencies"):
            runtime_patches.apply_triton()
    assert not runtime_patches._TRITON_APPLIED


def _run(source):
    subprocess.run([sys.executable, "-c", textwrap.dedent(source)], check=True, timeout=180)


def test_apply_is_idempotent_and_keeps_dependency_sources_unchanged():
    _run('''
        from pathlib import Path
        import torch
        from torch._dynamo import output_graph, symbolic_convert
        from torch._inductor import scheduler, pattern_matcher
        from torch._inductor.fx_passes import post_grad
        from torch._inductor.runtime import static_triton_launcher, triton_heuristics
        from triton.runtime import build
        from ignition import runtime_patches

        modules = (output_graph, symbolic_convert, scheduler, pattern_matcher, post_grad,
                   static_triton_launcher, triton_heuristics, build)
        before = {Path(module.__file__): Path(module.__file__).read_bytes() for module in modules}
        generic_matcher = pattern_matcher.PatternMatcherPass.apply
        torch.compiler.config.cache_key_tag = "caller-tag"
        runtime_patches.apply()
        targets = (post_grad.decompose_auto_functionalized, scheduler.BaseSchedulerNode.decide_inplace_update,
                   output_graph.OutputGraph.check_event_record_after_input_mutation,
                   output_graph.OutputGraph.check_input_mutation_on_current_stream)
        expected_tag = "caller-tag;" + runtime_patches.identity()
        assert torch.compiler.config.cache_key_tag == expected_tag
        runtime_patches.apply()
        assert targets == (post_grad.decompose_auto_functionalized, scheduler.BaseSchedulerNode.decide_inplace_update,
                           output_graph.OutputGraph.check_event_record_after_input_mutation,
                           output_graph.OutputGraph.check_input_mutation_on_current_stream)
        assert torch.compiler.config.cache_key_tag == expected_tag
        assert pattern_matcher.PatternMatcherPass.apply is generic_matcher
        assert all(path.read_bytes() == content for path, content in before.items())
        assert not torch.cuda.is_initialized()
    ''')


def test_incompatible_version_and_context_publish_no_partial_patch():
    _run('''
        import torch
        from torch._dynamo.output_graph import OutputGraph
        from torch._inductor import scheduler
        from torch._inductor.fx_passes import post_grad
        from ignition import runtime_patches
        from ignition.errors import PreparationError

        original = (OutputGraph.check_event_record_after_input_mutation,
                    post_grad.decompose_auto_functionalized, scheduler.BaseSchedulerNode.decide_inplace_update)
        version = torch.__version__
        torch.__version__ = "unsupported"
        try:
            runtime_patches.apply()
        except PreparationError as error:
            assert "require torch==" in str(error)
        else:
            raise AssertionError("unsupported Torch accepted")
        torch.__version__ = version
        from ignition.runtime_patches import scheduler as patches
        builder = patches.replacements
        def incompatible():
            raise PreparationError("missing scheduler context")
        patches.replacements = incompatible
        try:
            runtime_patches.apply()
        except PreparationError as error:
            assert "missing scheduler context" in str(error)
        else:
            raise AssertionError("incompatible source accepted")
        assert original == (OutputGraph.check_event_record_after_input_mutation,
                            post_grad.decompose_auto_functionalized, scheduler.BaseSchedulerNode.decide_inplace_update)
        assert not runtime_patches._APPLIED
        patches.replacements = builder
        runtime_patches.apply()
    ''')


@pytest.mark.parametrize("kind", ["SPAWN", "FORK"])
def test_inductor_worker_installs_matching_patch_identity_before_first_job(kind):
    _run('''
        import torch
        from torch._inductor.compile_worker.subproc_pool import SubprocKind, SubprocPool
        from ignition import runtime_patches

        runtime_patches.apply()
        pool = SubprocPool(1, kind=SubprocKind.KIND)
        try:
            worker_tag = pool.submit(eval, "__import__('torch').compiler.config.cache_key_tag").result(timeout=30)
            assert worker_tag == torch.compiler.config.cache_key_tag
            pool.submit(runtime_patches.apply).result(timeout=30)
            repeated = pool.submit(eval, "__import__('torch').compiler.config.cache_key_tag").result(timeout=30)
            assert repeated == worker_tag
        finally:
            pool.shutdown()
    '''.replace("SubprocKind.KIND", f"SubprocKind.{kind}"))


def test_triton_bootstrap_does_not_import_torch():
    _run('''
        import sys
        from ignition import runtime_patches
        runtime_patches.apply_triton()
        from triton.runtime.build import platform_key
        assert platform_key.__module__ == "ignition.runtime_patches.triton"
        assert "torch" not in sys.modules
    ''')


def test_direct_spawned_inductor_worker_installs_before_first_job():
    _run('''
        import multiprocessing
        import os
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial
        from ignition import runtime_patches
        from torch._inductor import async_compile

        runtime_patches.apply()
        with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"),
                initializer=partial(async_compile._async_compile_initializer, os.getpid())) as pool:
            tag = pool.submit(eval, "__import__('torch').compiler.config.cache_key_tag").result(timeout=120)
            assert tag == runtime_patches.identity()
    ''')


def test_b12x_offline_worker_installs_triton_fix_before_first_job():
    _run('''
        from ignition import runtime_patches
        from b12x._lib.compile_pool import CompilePool

        runtime_patches.apply()
        with CompilePool(device_ordinal=0, compute_capability=(12, 0), device_uuid="runtime-patch-test",
                         product_name="NVIDIA RTX PRO 6000 Blackwell", sm_count=188,
                         max_shared_memory_per_block=101376, max_shared_memory_per_multiprocessor=102400,
                         workers=1) as pool:
            result = pool._pool.apply_async(eval, ("(__import__('triton.runtime.build', fromlist=['platform_key'])."
                "platform_key.__module__, __import__('torch').cuda.is_initialized())",)).get(timeout=120)
            assert result == ("ignition.runtime_patches.triton", False)
    ''')
