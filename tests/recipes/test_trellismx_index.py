# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

import json

from ignition.recipes import _stored_bytes


def test_trellismx_stored_bytes_sums_snapshot(tmp_path):
    (tmp_path / "trellismx").mkdir()
    (tmp_path / "trellismx" / "manifest.json").write_text("{}")
    (tmp_path / "model-00001.safetensors").write_bytes(b"x" * 32)
    (tmp_path / "engram").mkdir()
    (tmp_path / "engram" / "layer-001.trellis.safetensors").write_bytes(b"y" * 64)
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"metadata": {"total_size": 32}, "weight_map": {}}))

    assert _stored_bytes(index, "test") == sum(
        path.stat().st_size for path in tmp_path.rglob("*") if path.is_file())


def test_stored_bytes_uses_index_total_for_non_trellis(tmp_path):
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"metadata": {"total_size": 1234}, "weight_map": {}}))

    assert _stored_bytes(index, "test") == 1234
