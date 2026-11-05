# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint shards load through b12x's pinned-host reader on every filesystem."""
import os
from pathlib import Path

import pytest

from ignition.components import weights


@pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='requires an assigned CUDA GPU')
@pytest.mark.parametrize('location', ('shared-memory', 'temporary-directory'))
@pytest.mark.parametrize('dtype', ('bfloat16', 'uint8', 'float32'))
def test_checkpoint_reads_preserve_strided_tensor_bytes(tmp_path, location, dtype):
    import tempfile
    from types import MappingProxyType

    import torch
    from safetensors.torch import save_file

    from ignition.preparation.checkpoint import IndexedCheckpoint, _read_header

    if location == 'shared-memory' and not Path('/dev/shm').is_dir():
        pytest.skip('requires Linux shared memory')
    parent = '/dev/shm' if location == 'shared-memory' else tmp_path
    with tempfile.TemporaryDirectory(prefix='flare-checkpoint-', dir=parent) as directory:
        root = Path(directory)
        source = torch.arange(17 * 65, dtype=torch.float32).reshape(17, 65).to(getattr(torch, dtype))
        save_file({'weight': source}, str(root / 'weights.safetensors'))
        shard, tensors = _read_header(root / 'weights.safetensors', 'weights.safetensors')
        manifest = IndexedCheckpoint(root, MappingProxyType(tensors), (shard,))
        with weights.DirectWeightLoader(manifest, 'cuda:0') as loader:
            actual = loader.load_tensor('weight', extent=(slice(2, 13), slice(3, 62)))
            assert loader._session.read_mode == 'bounce'
            assert loader._session.torch_copy_bytes == 0
        torch.testing.assert_close(actual.cpu(), source[2:13, 3:62], rtol=0, atol=0)
