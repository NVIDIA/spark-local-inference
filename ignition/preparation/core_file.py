# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Metadata adapters for the native core publisher and reader."""

from pathlib import Path
import json

from . import runtime_core_pb2 as proto

ALIGNMENT = 4096
MAX_HEADER_BYTES = 64 << 10
MAX_INDEX_BYTES = 128 << 20


def align(value):
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def validate_filename(filename):
    import flare_heliopause
    flare_heliopause.export_validate_layout(json.dumps({'version': 1, 'files': {'': [], filename: []}}))
    return filename


def container_header(length, index_size=0, format='flare-data'):
    import flare_heliopause
    return flare_heliopause.export_container_header(length, index_size, format)


def _path(source):
    if isinstance(source, (str, Path)):
        return str(source)
    import os
    return f'/proc/self/fd/{source.fileno()}' if os.name == 'posix' else source.name


def _payload(payload):
    return json.dumps([(destination, _path(source), offset, size) for destination, source, offset, size in payload])


def _options(path):
    """Forward explicit export options; absent cache stays absent for native."""
    from .core_snapshot import _OPTIONS
    options = _OPTIONS.get()
    return dict(cache=None if options is None else str(options.cache),
                no_weights=False if options is None else options.no_weights)


def read_index(path):
    import flare_heliopause
    encoded, base = (flare_heliopause.export_read_index(str(path)) if isinstance(path, (str, Path))
                     else flare_heliopause.export_read_index_fd(path.fileno()))
    return proto.ModelIndex.FromString(encoded), base


def core_form(path, form):
    """Select the canonical form from an explicit request.

    The default is the directory form at exactly the requested path, even when
    that path ends in `.safetensors`. `form='packed'` explicitly selects the
    packed file form at exactly the requested path.
    """
    if form is None:
        form = 'directory'
    if form not in ('packed', 'directory'):
        raise ValueError(f'unknown core form {form!r}')
    return form


def _export_native(path, form, publish, cache_only=False):
    """Forward the requested form; the native exporter owns all file management."""
    destination = ''
    if not cache_only:
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        destination = str(path)
    return publish(destination, form)


def write_sources(path, index, payload, *, hardware=None, weight_payload=None):
    """Write one staged rank source, carrying captured side metadata forward."""
    import json

    import flare_heliopause
    flare_heliopause.export_write_sources(
        str(path), index.SerializeToString(deterministic=True), _payload(payload),
        hardware, None if weight_payload is None else json.dumps(weight_payload))


def read_sources_record(path):
    """Read one staged rank source with its captured side metadata."""
    import flare_heliopause
    record = json.loads(flare_heliopause.export_read_sources(str(path)))
    index = proto.ModelIndex.FromString(bytes(record['index']))
    def locate(span):
        return tuple((Path(source), offset, size) for source, offset, size in
                     json.loads(flare_heliopause.export_locate_sources(str(path), span.offset, span.nbytes)))
    metadata = dict(hardware=record.get('hardware'),
                    weight_payload=record.get('weight_payload'))
    return index, locate, metadata


def read_sources(path):
    index, locate, _metadata = read_sources_record(path)
    return index, locate


def combine(path, ranks, peer_memory, *, publisher=None,
            cluster=None, form=None, cache_only=False):
    import flare_heliopause
    from types import SimpleNamespace
    options = _options(path)
    if cluster is not None:
        cluster = dict(world=cluster[0], export_id=cluster[1])
    sources = [str(rank) for rank in ranks]
    peers = json.dumps(peer_memory)
    publisher_request = None if publisher is None else json.dumps(publisher.native_request)
    cluster_request = None if cluster is None else json.dumps(cluster)
    form = core_form(path if path is not None else 'cache', form)
    value = _export_native(path, form,
        lambda destination, selected: flare_heliopause.export_combine(
            destination, sources, peers, publisher_request, cluster_request,
            options['cache'], options['no_weights'], selected),
        cache_only=cache_only)
    result = json.loads(value)
    result.setdefault('form', form)
    return SimpleNamespace(**result)


class _PayloadReader:
    def __init__(self, files, filename):
        self.files, self.filename, self.position = files, filename, 0

    def read_at(self, size, offset):
        if size < 0 or offset < 0:
            raise ValueError('invalid logical read')
        return self.files.read_at(self.filename, offset, size)

    def seek(self, offset, whence=0):
        if whence == 1:
            offset += self.position
        elif whence == 2:
            import flare_heliopause
            offset += self.files.length(self.filename)
        elif whence:
            raise ValueError('invalid seek')
        if offset < 0:
            raise ValueError('negative seek')
        self.position = offset
        return offset

    def read(self, size=-1):
        if size < 0:
            import flare_heliopause
            size = max(0, self.files.length(self.filename) - self.position)
        data = self.read_at(size, self.position)
        self.position += len(data)
        return data


class CoreFiles:
    def __init__(self, path, primary, base, length):
        import flare_heliopause
        self.files, self.readers = flare_heliopause.NativeCoreFiles(str(path)), {}

    def get(self, filename):
        import flare_heliopause
        reader = self.readers.setdefault(filename, _PayloadReader(self.files, filename))
        return reader, 0, self.files.length(filename)

    def close(self):
        self.files.close()
        self.readers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
