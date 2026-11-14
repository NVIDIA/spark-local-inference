# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Safetensors interoperability, bounded metadata reads, and stable storage placement."""

import json
import struct

import pytest

from ignition.preparation import runtime_core_pb2 as proto
from ignition.preparation.core_file import (
    ALIGNMENT, MAX_HEADER_BYTES, MAX_INDEX_BYTES, align, read_index,
)


def index():
    return proto.ModelIndex(version=4, ranks=[proto.BundleIndex()])


def header(payload_size, index_size):
    return {
        "__metadata__": {"format": "flare-staging"},
        "flare.payload": {"dtype": "U8", "shape": [payload_size], "data_offsets": [0, payload_size]},
        "flare.metadata": {"dtype": "U8", "shape": [index_size],
                           "data_offsets": [payload_size, payload_size + index_size]},
    }


def write_container(path, description, payload, metadata):
    encoded = json.dumps(description, separators=(",", ":")).encode()
    encoded = encoded.ljust(align(8 + len(encoded)) - 8, b" ")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload + metadata)


def test_index_read_does_not_touch_the_payload(tmp_path):
    import re
    import shutil
    import subprocess
    import sys

    trace_program = shutil.which("strace")
    if trace_program is None:
        pytest.skip("requires strace to observe native file reads")
    description = index().SerializeToString()
    path = tmp_path / "model.safetensors"
    write_container(path, header(2 << 20, len(description)), bytes(2 << 20), description)
    trace = tmp_path / "reads.txt"
    result = subprocess.run([trace_program, "-yy", "-e", "trace=read,pread64", "-o", str(trace),
        sys.executable, "-c", "from ignition.preparation.core_file import read_index; "
        "import sys; index,base=read_index(sys.argv[1]); assert index.version==4 and base==4096",
        str(path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    reads = []
    for line in trace.read_text().splitlines():
        if f"<{path}>" not in line:
            continue
        assert line.startswith("pread64("), line
        match = re.search(r", (\d+), (\d+)\) = (\d+)$", line)
        assert match, line
        size, offset, returned = map(int, match.groups())
        assert returned == size
        reads.append((offset, size))
    assert reads == [(0, 8), (8, ALIGNMENT - 8), (ALIGNMENT + (2 << 20), len(description))]


@pytest.mark.parametrize("field,value", [
    (("__metadata__", "format"), "pt"),
    (("flare.payload", "dtype"), "F32"),
    (("flare.payload", "shape"), [2, 2]),
    (("flare.payload", "shape"), [True]),
    (("flare.payload", "data_offsets"), [1, 5]),
    (("flare.payload", "data_offsets"), [-1, 3]),
    (("flare.metadata", "data_offsets"), [3, 7]),
    (("flare.metadata", "data_offsets"), [4, 1 << 64]),
    (("flare.metadata", "shape"), [MAX_INDEX_BYTES + 1]),
])
def test_invalid_safetensors_header_is_rejected(tmp_path, field, value):
    description = index().SerializeToString()
    container = header(4, len(description))
    container[field[0]][field[1]] = value
    path = tmp_path / "invalid.safetensors"
    write_container(path, container, b"data", description)
    with pytest.raises(ValueError):
        read_index(path)


def test_resources_cannot_address_protobuf_bytes(tmp_path):
    description = index()
    description.ranks[0].resources["outside"].CopyFrom(proto.Span(offset=4, nbytes=1))
    encoded = description.SerializeToString()
    path = tmp_path / "invalid.safetensors"
    write_container(path, header(4, len(encoded)), b"data", encoded)
    with pytest.raises(ValueError, match="resource exceeds"):
        read_index(path)


def test_index_size_is_bounded_before_reading(tmp_path):
    path = tmp_path / "invalid.safetensors"
    write_container(path, header(0, MAX_INDEX_BYTES + 1), b"", b"")
    with pytest.raises(ValueError, match="invalid model index size"):
        read_index(path)


@pytest.mark.parametrize("prefix", [b"", struct.pack("<Q", 0), struct.pack("<Q", MAX_HEADER_BYTES + 1)])
def test_header_size_is_bounded_before_reading(tmp_path, prefix):
    path = tmp_path / "invalid.safetensors"
    path.write_bytes(prefix)
    with pytest.raises(ValueError):
        read_index(path)
