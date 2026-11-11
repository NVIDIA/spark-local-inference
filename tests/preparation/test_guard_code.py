# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persist guard source while binding each process's closure values."""

import subprocess
import sys

import torch
from torch._dynamo import guards

from ignition.preparation.aot import _guard_code_cache


class GuardedForward:
    def __init__(self):
        self.value = torch.ones(2)
        self.offset = 2

    def __call__(self):
        return self.value + self.offset


def _execute(forward):
    return forward()


def test_loaded_guards_preserve_predicates_without_construction_stacks(tmp_path):
    from torch._dynamo import package
    from torch._dynamo.aot_compile_types import GraphModuleSerializableCallable

    forward = GuardedForward()
    compiled = torch.compile(_execute,
        backend=lambda graph, inputs: GraphModuleSerializableCallable(graph),
        fullgraph=True, dynamic=False).aot_compile(((forward,), {}))
    data = type(compiled).serialize(compiled).serialized_data
    reference = type(compiled).deserialize(data)
    original = package.load_guard_manager
    with _guard_code_cache(tmp_path):
        loaded = type(compiled).deserialize(data)
    assert package.load_guard_manager is original
    torch.testing.assert_close(loaded(forward), reference(forward))
    for value, offset, valid in ((torch.ones(2), 2, True), (torch.ones(2), 3, False),
                                 (torch.ones(3), 2, False), (torch.ones(2, dtype=torch.float64), 2, False)):
        forward.value, forward.offset = value, offset
        assert loaded.guard_check(forward) is valid
        assert reference.guard_check(forward) is valid
        if not valid:
            reason = str(loaded._artifacts.guard_manager.check_verbose({"forward": forward}))
            assert "forward" in reason
    forward.value, forward.offset = torch.ones(2), 2
    torch.testing.assert_close(loaded(forward), reference(forward))


def test_cached_guard_source_binds_fresh_closures_in_another_process(tmp_path):
    parts = ["len(L['values']) == expected", "L['enabled'] == enabled"]
    expected = guards.build_guard_function(parts, 'expected, enabled')
    original = guards.build_guard_function
    with _guard_code_cache(tmp_path):
        assert guards.build_guard_function(parts, 'expected, enabled') == expected
    assert guards.build_guard_function is original

    subprocess.run([sys.executable, '-c', '''
import sys
from pathlib import Path
from torch._dynamo import guards
from ignition.preparation.aot import _guard_code_cache

def forbid_build(*args, **kwargs):
    raise AssertionError('cached guard source must not be regenerated')

guards.build_guard_function = forbid_build
with _guard_code_cache(Path(sys.argv[1])):
    _, source = guards.build_guard_function(
        ["len(L['values']) == expected", "L['enabled'] == enabled"], 'expected, enabled')
scope = {}
exec(source, scope)
for size, enabled in ((3, True), (1, False)):
    check = scope['___make_guard_fn'](size, enabled)
    assert check(dict(values=list(range(size)), enabled=enabled))
    assert not check(dict(values=list(range(size + 1)), enabled=enabled))
    assert not check(dict(values=list(range(size)), enabled=not enabled))
assert guards.build_guard_function is forbid_build
''', str(tmp_path)], check=True)
