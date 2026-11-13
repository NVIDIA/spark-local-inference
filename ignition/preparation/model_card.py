# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The model license as the checkpoint's model card declares it."""

from pathlib import Path

from .protection import License


def model_license(snapshot, repository=None, revision=None):
    """Return the license the model card at `snapshot/README.md` declares, or None.

    The front matter's `license_name` names the license, else its `license` identifier, which the
    Hugging Face hub constrains to an SPDX identifier or `other`. The link resolves, in order, an
    absolute HTTPS `license_link`, a repository-relative `license_link`, a LICENSE file in the
    snapshot, or the model page; the repository forms are available only for a checkpoint resolved
    from a repository, pinned to `revision`.
    """
    card = Path(snapshot) / 'README.md'
    if not card.is_file():
        return None
    from huggingface_hub import ModelCard

    try:
        data = ModelCard.load(card).data
    except Exception as error:  # noqa: BLE001 - Any malformed card is one preparation error.
        raise ValueError(f'unreadable model card {card}: {error}') from error
    declared = getattr(data, 'license', None)
    if isinstance(declared, list):
        declared = ', '.join(str(item) for item in declared)
    name = getattr(data, 'license_name', None) or declared
    if not name:
        return None
    link = getattr(data, 'license_link', None)
    base = None if repository is None else f'https://huggingface.co/{repository}'
    if isinstance(link, str) and link.startswith('https://'):
        pass
    elif isinstance(link, str) and link and base is not None and not link.startswith(('http://', '/')):
        link = f'{base}/blob/{revision}/{link}'
    else:
        link = ''
        stored = sorted(p.name for p in Path(snapshot).iterdir()
                        if p.is_file() and p.name.lower().startswith(('license', 'licence')))
        if stored and base is not None:
            link = f'{base}/blob/{revision}/{stored[0]}'
        elif base is not None:
            link = base
    return License(str(name), link)
