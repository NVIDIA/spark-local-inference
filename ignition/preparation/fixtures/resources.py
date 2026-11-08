# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Share serial candidate fixtures without retaining them after a tuning race."""

from weakref import ref


def restore_seed(value, seed):
    """Restore matching private storage without a strided host-copy device temporary."""
    size = value.untyped_storage().nbytes() // value.element_size()
    value.as_strided((size,), (1,), 0).copy_(seed.as_strided((size,), (1,), 0), non_blocking=True)


def candidate_inputs(factory, *, session=None):
    """Acquire one fixture whose lifetime belongs to the live candidate calls.

    b12x runs each candidate's producer and invocation in stream order. Calls
    can share mutable operands when the producer restores every input and the
    kernel fully writes its output. Each call must retain the returned owner.
    """
    reference = None

    def acquire():
        nonlocal reference
        value = None if reference is None else reference()
        if value is None:
            from ignition.preparation.tuning import RecordedPreparationSession

            if isinstance(session, RecordedPreparationSession):
                # A preceding operator can leave its fixture in the session cache.
                # Live trials own their storage independently of that cache entry.
                session.release_benchmark_cache()
            value = factory()
            reference = ref(value)
        return value

    return acquire
