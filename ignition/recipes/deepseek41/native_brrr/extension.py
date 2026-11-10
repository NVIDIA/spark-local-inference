# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Load the preparation-only CUDA extension; native serving uses exported graphs."""
from functools import lru_cache
import hashlib
from pathlib import Path


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parent
    source = root / 'prologue.cu'
    digest = hashlib.sha256(b''.join(path.read_bytes() for path in
        (source, *sorted(root.glob('*.cuh'))))).hexdigest()[:12]
    return load(name='flare_ds41_brrr_' + digest, sources=[str(source)],
                extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_120a,code=sm_120a',
                                   '--expt-relaxed-constexpr', '--expt-extended-lambda'],
                extra_cflags=['-O3', '-std=c++20'], verbose=False)
