# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Publishing options and adapters for native immutable runtime snapshots."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import json

_OPTIONS = ContextVar('core_snapshot_options', default=None)


@dataclass(frozen=True)
class ExportOptions:
    cache: Path
    no_weights: bool = False
    reuse_weights: dict | None = None


@contextmanager
def export_options(cache, *, no_weights=False, reuse_weights=None):
    token = _OPTIONS.set(ExportOptions(Path(cache), no_weights, reuse_weights))
    try:
        yield
    finally:
        _OPTIONS.reset(token)


def read_container(source):
    import flare_heliopause
    record = json.loads(flare_heliopause.export_read_container(str(source)) if isinstance(source, (str, Path))
                        else flare_heliopause.export_read_container_fd(source.fileno()))
    return record['header'], record['base'], bytes(record['envelope']), record['layout']



def validate_layout(layout):
    import flare_heliopause
    flare_heliopause.export_validate_layout(json.dumps(layout))


def payload_convert_cache(source, cache, destination=None, *, profiles, form='directory'):
    """Install a payload core as a shared cache entry, optionally exporting it.

    An omitted destination leaves the committed leased entry in the shared cache;
    the returned `entry_path` is that entry's actual directory.
    """
    import flare_heliopause
    from types import SimpleNamespace
    if destination is not None and Path(destination).exists():
        raise FileExistsError(destination)
    value = flare_heliopause.export_payload_convert_cache(
        str(source), str(cache), '' if destination is None else str(destination),
        json.dumps(profiles), form)
    return SimpleNamespace(**json.loads(value))


def payload_combine(sources, destination, *, publisher, profiles, accepted, target_accepted,
                    form='directory'):
    """Combine prepared TP cores into one signed payload directory."""
    import flare_heliopause
    from types import SimpleNamespace
    if Path(destination).exists():
        raise FileExistsError(destination)
    value = flare_heliopause.export_payload_combine(
        [str(source) for source in sources], str(destination), json.dumps(publisher.native_request),
        json.dumps(profiles), accepted, target_accepted, form)
    return SimpleNamespace(**json.loads(value))


def payload_promote(source, destination, *, publisher, profiles, accepted, target_accepted, target_publisher):
    """Re-sign one payload core under a fresh identity while preserving compressed executables."""
    import flare_heliopause
    from types import SimpleNamespace
    if Path(destination).exists():
        raise FileExistsError(destination)
    value = flare_heliopause.export_payload_promote(
        str(source), str(destination), json.dumps(publisher.native_request),
        json.dumps(profiles), accepted, target_accepted, json.dumps(target_publisher))
    return SimpleNamespace(**json.loads(value))


def payload_edit(source, destination, *, publisher, profiles, served_model_name=None, chat_template=None):
    """Edit frontend metadata while preserving stored weights and executables."""
    import flare_heliopause
    if Path(destination).exists():
        raise FileExistsError(destination)
    flare_heliopause.export_payload_edit(
        str(source), str(destination), json.dumps(publisher.native_request), json.dumps(profiles),
        served_model_name, chat_template)


def payload_form(source):
    """Classify a core as payload packed, payload directory, or legacy."""
    import flare_heliopause
    return flare_heliopause.export_payload_form(str(source))


def convert(source, destination, *, profiles=(), cache=None,
            form='directory'):
    """Convert a payload core without changing its signed bytes."""
    import flare_heliopause
    from types import SimpleNamespace
    if form not in ('packed', 'directory'):
        raise ValueError(f'unknown payload core form {form!r}')
    if Path(destination).exists():
        raise FileExistsError(destination)
    if payload_form(source) == 'legacy':
        raise ValueError('legacy core format is unsupported; re-export the model with Ignition')
    value = flare_heliopause.export_payload_convert(str(source), str(destination), form)
    return SimpleNamespace(**json.loads(value))
