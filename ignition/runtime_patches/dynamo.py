# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Defer Dynamo input-mutation stream scans until a trace records an event."""

from functools import wraps


def replacements():
    from torch._dynamo import exc
    from torch._dynamo.output_graph import OutputGraph
    from torch._dynamo.symbolic_convert import SpeculationLog

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
        (SpeculationLog, "__init__", initialize_speculation_log),
        (OutputGraph, "check_input_mutation_on_current_stream", check_input_mutation_on_current_stream),
        (OutputGraph, "check_event_record_after_input_mutation", check_event_record_after_input_mutation),
    ]
