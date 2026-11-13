# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File, checkout, and JSON records shared by qualification entrypoints."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import subprocess


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 << 20):
            result.update(block)
    return result.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}


def plain(value):
    if isinstance(value, Enum):
        return plain(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def checkout_record(path, expected):
    path = Path(path).resolve()

    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(path), *arguments], text=True).strip()

    record = {'path': str(path), 'head': git('rev-parse', 'HEAD'),
              'tree': git('rev-parse', 'HEAD^{tree}'),
              'status': git('status', '--porcelain', '--untracked-files=all'),
              'submodules': git('submodule', 'status', '--recursive')}
    if any(record[key] != expected[key] for key in ('head', 'tree')) or record['status']:
        raise ValueError(f'checkout does not match its clean source pin: {record}')
    if any(line[:1] in ('+', 'U') for line in record['submodules'].splitlines()):
        raise ValueError(f'checkout has an unpinned submodule: {record}')
    return record
