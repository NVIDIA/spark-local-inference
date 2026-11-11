# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Qualification copies around the prepared projection's unchanged result."""


class OutputReadback:
    def __init__(self, operation, output):
        self.operation, self.output = operation, output

    def __call__(self, source):
        output = self.operation(source)
        self.output.copy_(output)
        return output


class ProjectionReadback:
    def __init__(self, operation, output, *, source=None):
        self.operation, self.output, self.source = operation, output, source
        self.weight, self.plan = operation.weight, operation.plan

    def __call__(self, source):
        if self.source is not None:
            self.source.copy_(source)
        output = self.operation(source)
        self.output.copy_(output)
        return output


class KVReadback:
    def __init__(self, operation, key, value):
        self.operation, self.key, self.value = operation, key, value

    def __call__(self, key, value, slots=None, **kwargs):
        self.key.copy_(key)
        self.value.copy_(value)
        return self.operation(key, value, slots, **kwargs)
