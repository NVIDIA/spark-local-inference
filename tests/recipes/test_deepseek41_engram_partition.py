# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek table payloads retain canonical rows independently of TP placement."""

from dataclasses import replace
from pathlib import Path

import pytest
from b12x.sequence import engram

from ignition.components.parallel import TensorParallelShard
from ignition.preparation.runtime_core import FileSource
from ignition.recipes.deepseek41.engram import Tables


@pytest.mark.parametrize("world", [1, 2, 3, 4, 8])
def test_core_planes_and_resident_segments_preserve_full_table_bytes(tmp_path, world):
    geometry = engram.build_geometry(layer_ids=(1,), base_table_size=101, compressed_vocab_size=32)
    rows = geometry.num_embeddings[0]
    sources, contents = [], []
    for width in (256, 8):
        data = bytes((row * 13 + column) % 251 for row in range(rows) for column in range(width))
        path = tmp_path / str(width)
        path.write_bytes(bytes(37) + data)
        sources.append(FileSource(path, 37, len(data)))
        contents.append(data)
    tables = Tables(geometry, tuple(range(32)), {1: tuple(sources)}, {}, {}, None,
                    TensorParallelShard(0, 1), None)
    canonical = tables.export_metadata()
    table = canonical["tables"][0]
    assert "shards" not in table
    assert table["hash"]["primes"] == list(geometry.primes[0])
    assert table["hash"]["multipliers"] == list(geometry.multipliers[0])
    assert table["hash"]["offsets"] == list(geometry.offsets[0])
    for rank in range(world):
        rank_tables = replace(tables, shard=TensorParallelShard(rank, world))
        assert rank_tables.export_metadata() == canonical
        resident = rank_tables.resident_catalog()["tables"][0]["planes"]
        for ordinal, (name, width, source) in enumerate(zip(("weights", "scales"), (256, 8), contents)):
            chunks = table[name]
            actual = b"".join(Path(chunk["path"]).read_bytes()[chunk["offset"]:chunk["offset"] + chunk["nbytes"]]
                              for chunk in chunks)
            assert actual == source and len(actual) == rows * width
            assert resident[ordinal]["segments"] == chunks
            assert resident[ordinal]["row_bytes"] == width
        assert rank_tables.resident_catalog() == tables.resident_catalog()
    assert sum(chunk["nbytes"] for name in ("weights", "scales") for chunk in table[name]) == rows * 264
