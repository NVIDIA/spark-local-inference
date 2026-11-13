# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stamp publisher verification profiles into a linked Flare ELF executable."""

import argparse
import json
import os
from pathlib import Path
import struct

from ignition.preparation.protection import validate_namespace

MAGIC = b'FLARE-PUBLISHERS-V3'.ljust(32, b'\0')
CAPACITY = 65536


def stamp(source, destination, profiles):
    source, destination = Path(source), Path(destination)
    seen = set()
    if not profiles:
        raise ValueError('at least one publisher profile is required')
    for profile in profiles:
        if set(profile) not in ({'name', 'namespace', 'public_key'}, {'name', 'namespace', 'public_key', 'development'}):
            raise ValueError('publisher profiles require name, namespace, and public_key')
        validate_namespace(profile['namespace'])
        if type(profile.get('development', False)) is not bool:
            raise ValueError('development must be a boolean')
        if profile.get('development') and profile['namespace'] != 'local-development':
            raise ValueError('development publisher namespace must be local-development')
        name = profile['name']
        if not isinstance(name, str) or not name or len(name.encode()) > 256 or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in name):
            raise ValueError('invalid publisher display name')
        raw = profile['public_key']
        if not isinstance(raw, list) or any(type(v) is not int or not 0 <= v <= 255 for v in raw):
            raise ValueError('public_key must be an array of 32 bytes')
        key = bytes(raw)
        if len(key) != 32 or key == bytes(32) or key in seen:
            raise ValueError('invalid or duplicate publisher verification key')
        seen.add(key)
    encoded = json.dumps(profiles, ensure_ascii=False, separators=(',', ':')).encode()
    if len(encoded) > CAPACITY - 36:
        raise ValueError('publisher profiles exceed ELF section capacity')
    binary = bytearray(source.read_bytes())
    if binary[:6] != b'\x7fELF\x02\x01':
        raise ValueError('stamping requires a little-endian ELF64 executable')
    section_offset = struct.unpack_from('<Q', binary, 40)[0]
    entry_size, count, names_index = struct.unpack_from('<HHH', binary, 58)
    if entry_size != 64 or not 0 < names_index < count or section_offset + count * entry_size > len(binary):
        raise ValueError('invalid ELF section table')
    def section(index):
        return struct.unpack_from('<IIQQQQIIQQ', binary, section_offset + index * entry_size)
    names = section(names_index)
    name_bytes = binary[names[4]:names[4] + names[5]]
    slots = []
    for i in range(count):
        entry = section(i)
        end = name_bytes.find(b'\0', entry[0])
        if end >= 0 and name_bytes[entry[0]:end] == b'.flare_publishers':
            slots.append(entry)
    if len(slots) != 1:
        raise ValueError('expected exactly one .flare_publishers ELF section')
    _, kind, flags, _, offset, size, *_ = slots[0]
    if kind != 1 or flags & 4 or not flags & 2 or size != CAPACITY or offset + size > len(binary):
        raise ValueError('invalid publisher data section')
    if binary[offset:offset + 32] != MAGIC or any(binary[offset + 32:offset + size]):
        raise ValueError('publisher section is not an empty release slot')
    binary[offset + 32:offset + 36] = struct.pack('<I', len(encoded))
    binary[offset + 36:offset + 36 + len(encoded)] = encoded
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, source.stat().st_mode & 0o777)
    try:
        with os.fdopen(descriptor, 'wb') as output:
            output.write(binary)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--publishers', required=True, type=Path, help='JSON array of publisher verification profiles')
    args = parser.parse_args()
    stamp(args.source, args.output, json.loads(args.publishers.read_text()))


if __name__ == '__main__':
    main()
