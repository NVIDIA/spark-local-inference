# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind guard-manager lifetime to wrappers and defer input-mutation stream scans."""

from functools import wraps

from . import _replace


def replacements():
    from torch._dynamo import exc
    from torch._dynamo.guards import CheckFunctionManager
    from torch._dynamo.output_graph import OutputGraph
    from torch._dynamo.symbolic_convert import SpeculationLog

    initialize_guards = _replace(CheckFunctionManager.__init__, ((
        "        self.guard_manager = guard_manager\n"
        "        self.compile_check_fn(builder, sorted_guards, guard_fail_fn)",
        "        self.guard_manager = guard_manager\n"
        "        guard_manager._check_function_manager = self\n"
        "        self.compile_check_fn(builder, sorted_guards, guard_fail_fn)",
    ),))
    id_ref = _replace(CheckFunctionManager.id_ref, ((
        "            weakref.finalize(\n"
        "                obj, functools.partial(self.invalidate, obj_str=obj_str)\n"
        "            )",
        "            manager_ref = weakref.ref(self)\n"
        "\n"
        "            def invalidated():\n"
        "                manager = manager_ref()\n"
        "                if manager is not None:\n"
        "                    manager.invalidate(obj_str=obj_str)\n"
        "\n"
        "            weakref.finalize(obj, invalidated)",
    ),))
    initialize = SpeculationLog.__init__
    check_mutation = OutputGraph.check_input_mutation_on_current_stream
    check_event = OutputGraph.check_event_record_after_input_mutation

    @wraps(initialize)
    def initialize_speculation_log(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        self.requires_input_mutation_stream_checks = False

    @wraps(check_mutation)
    def check_input_mutation_on_current_stream(self, tx):
        if tx.speculation_log.requires_input_mutation_stream_checks:
            return check_mutation(self, tx)

    @wraps(check_event)
    def check_event_record_after_input_mutation(self, stream_index):
        speculation_log = self.root_tx.speculation_log
        if not speculation_log.requires_input_mutation_stream_checks:
            speculation_log.requires_input_mutation_stream_checks = True
            raise exc.SpeculationRestartAnalysis(
                restart_reason="event recording requires input-mutation stream checks"
            )
        return check_event(self, stream_index)

    return [
        (CheckFunctionManager, "__init__", initialize_guards),
        (CheckFunctionManager, "id_ref", id_ref),
        (SpeculationLog, "__init__", initialize_speculation_log),
        (OutputGraph, "check_input_mutation_on_current_stream", check_input_mutation_on_current_stream),
        (OutputGraph, "check_event_record_after_input_mutation", check_event_record_after_input_mutation),
    ]
