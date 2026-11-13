# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Completion bundles for tests that keep direct assertions on record tensors."""

from dataclasses import replace

from ignition.components.completion import CompletionOutputs


def completion_outputs(records):
    return replace(CompletionOutputs.allocate(records.shape, device=records.device), records=records)


class RecordOnlyLogprobs:
    """CPU state tests leave the CUDA-only, unrequested reduction unbound."""
    def __init__(self, logits, outputs, *args):
        assert outputs.records.device.type == 'cpu'
        assert not outputs.requested.any()

    def prepare(self):
        pass

    def __call__(self):
        pass
