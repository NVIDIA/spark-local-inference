# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-user development keys remain stable and require exact runtime acceptance."""

import json
import os
import subprocess

import pytest

from ignition.cli.config import Configuration
from ignition.development import publisher


def config(path):
    return Configuration(path / 'config.toml', {}, False)


def test_development_key_is_private_stable_and_has_a_separate_public_profile(tmp_path):
    configuration = config(tmp_path / 'first')
    signer = publisher(configuration)
    assert signer.fingerprint == publisher(configuration).fingerprint
    private = configuration.path.parent / 'development-signing-key.pem'
    if os.name == 'nt':
        from ignition._windows_permissions import owner_only

        assert owner_only(private)
    else:
        assert private.stat().st_mode & 0o777 == 0o600
    profiles = json.loads(private.with_name('development-publisher.json').read_text())
    assert profiles[0]['namespace'] == signer.namespace == 'local-development'
    assert bytes(profiles[0]['public_key']) == signer.public_key
    assert profiles[0]['development'] and signer.development
    other = publisher(config(tmp_path / 'second'))
    assert other.fingerprint != signer.fingerprint


def test_insecure_existing_development_key_is_not_silently_replaced(tmp_path):
    configuration = config(tmp_path)
    publisher(configuration)
    private = tmp_path / 'development-signing-key.pem'
    contents = private.read_bytes()
    if os.name == 'nt':
        subprocess.run(['icacls', str(private), '/grant', '*S-1-1-0:R'], check=True, capture_output=True)
    else:
        private.chmod(0o644)
    with pytest.raises(ValueError, match='owner-only permissions'):
        publisher(configuration)
    assert private.read_bytes() == contents
