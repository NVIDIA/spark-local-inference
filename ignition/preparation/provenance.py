# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact checkpoint provenance recorded where preparation copies weights.

An entry exists only for transfers that know their logical source and ranges.
Nothing is inferred from shape, name, or recipe identity. Entries are held by
weak reference to the prepared tensor and are dropped with it, so loading a
model never retains its GPU tensors here.
"""

import weakref

_ENTRIES = {}


def _drop(key):
    _ENTRIES.pop(key, None)


def _valid(part, axis_size):
    if not isinstance(part, slice):
        return None
    if part.step not in (None, 1):
        return None
    begin = 0 if part.start is None else part.start
    end = axis_size if part.stop is None else part.stop
    if not isinstance(begin, int) or not isinstance(end, int):
        return None
    if begin < 0 or end < 0 or begin > end or end > axis_size:
        return None
    return (begin, end)


def ranges_of(extent, shape):
    """One half-open range per source axis, in axis order.

    Returns None when the extent is not fully representable (integer indices,
    negative or non-unit steps, out-of-bounds slices); such a view stays
    explicitly unprovenanced.
    """
    parts = (extent,) if not isinstance(extent, tuple) else extent
    if extent is None:
        parts = ()
    if len(parts) > len(shape):
        return None
    ranges = []
    for axis, size in enumerate(shape):
        if axis < len(parts):
            part = _valid(parts[axis], size)
            if part is None:
                return None
            ranges.append(part)
        else:
            ranges.append((0, size))
    return tuple(ranges)


def record(tensor, *, tensor_name, ranges=(), packing='', statistics_domain='tensor',
           packing_parameters=None):
    """Record how one prepared tensor's bytes were produced."""
    if ranges is None:
        return
    record_sources(tensor, [dict(tensor=tensor_name, ranges=tuple(ranges))],
                   packing=packing, statistics_domain=statistics_domain,
                   packing_parameters=packing_parameters)


def record_sources(tensor, sources, *, packing='', statistics_domain='tensor',
                   concatenation_axis=None, packing_parameters=None):
    """Record one or more ordered logical sources for a prepared tensor."""
    if not sources:
        return
    normalized = tuple(
        dict(tensor=source['tensor'],
             ranges=tuple((int(begin), int(end)) for begin, end in source['ranges']),
             concatenation_axis=(source.get('concatenation_axis')
                                 if source.get('concatenation_axis') is not None
                                 else concatenation_axis),
             piece_encoding=str(source.get('piece_encoding', '')),
             pieces=tuple(
                 dict(source_offset=int(piece['source_offset']),
                      destination_offset=int(piece['destination_offset']),
                      contiguous_bytes=int(piece['contiguous_bytes']),
                      extents=tuple(int(value) for value in piece['extents']),
                      source_strides=tuple(int(value) for value in piece['source_strides']),
                      destination_strides=tuple(int(value) for value in piece['destination_strides']))
                 for piece in source.get('pieces', ())))
        for source in sources)
    if not any(source['pieces'] or source['piece_encoding'] for source in normalized):
        normalized = tuple({key: value for key, value in source.items()
                            if key not in ('pieces', 'piece_encoding')}
                           for source in normalized)
    key = id(tensor)
    fingerprint = (tensor.data_ptr(), tensor.numel(), tensor.element_size(),
                   tuple(tensor.shape))
    _ENTRIES[key] = dict(
        reference=weakref.ref(tensor, lambda _ref, key=key: _drop(key)),
        fingerprint=fingerprint,
        sources=normalized,
        packing=packing,
        packing_parameters=(None if packing_parameters is None
                            else dict(packing_parameters)),
        statistics_domain=statistics_domain,
    )


def lookup(tensor):
    """Return recorded provenance for `tensor`, or None when unknown."""
    entry = _ENTRIES.get(id(tensor))
    if entry is None:
        return None
    owner = entry['reference']()
    if owner is not tensor:
        # A recycled identity never inherits another tensor's provenance.
        _drop(id(tensor))
        return None
    if entry['fingerprint'] != (tensor.data_ptr(), tensor.numel(), tensor.element_size(),
                                tuple(tensor.shape)):
        return None
    return dict(sources=entry['sources'], packing=entry['packing'],
                packing_parameters=entry.get('packing_parameters'),
                statistics_domain=entry['statistics_domain'])
