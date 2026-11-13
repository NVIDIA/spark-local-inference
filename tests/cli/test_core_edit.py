# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Core edits preserve executable payloads and use development signing."""

import json

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from ignition.cli import main
from ignition.cli.config import Configuration
from ignition.development import publisher
from tests.preparation.payload_fixture import manifest, publish, schema, verify_signatures


def test_edit_cli_preserves_model_and_rank_body_with_separate_development_key(tmp_path):
    source_config = Configuration(tmp_path / 'source-config/config.toml', {}, False)
    signer = publisher(source_config)
    source = publish(tmp_path / 'source', signer)
    source_profile = source_config.path.parent / 'development-publisher.json'
    config = tmp_path / 'destination-config/config.toml'
    config.parent.mkdir()
    config.write_text('[publishing]\nnamespace="unused"\n')
    destination = tmp_path / 'edited'
    before = manifest(source)
    assert main(['edit', str(source), '--output', str(destination), '--config', str(config),
                 '--publisher-keys', str(source_profile), '--served-model-name', 'nvidia/model']) == 0
    profile = json.loads((config.parent / 'development-publisher.json').read_text())[0]
    after = verify_signatures(destination, Ed25519PublicKey.from_public_bytes(bytes(profile['public_key'])))
    core = schema()('CoreBody').FromString(after.core_body)
    old_core = schema()('CoreBody').FromString(before.core_body)
    deployment = schema()('DeploymentBody').FromString(after.deployments[0].body)
    assert list(deployment.contract.served_names) == ['nvidia/model']
    assert core.model_identity == old_core.model_identity
    assert core.publisher.namespace == 'local-development'
    assert core.publisher.development
    assert after.ranks[0].body == before.ranks[0].body
    assert after.signer != before.signer
    assert manifest(source) == before


def test_edit_cli_requires_an_edit_and_preserves_existing_destination(tmp_path, capsys):
    output = tmp_path / 'existing'
    output.mkdir()
    marker = output / 'keep'
    marker.write_bytes(b'user data')
    assert main(['edit', 'missing', '--output', str(output), '--served-model-name', 'name']) == 2
    assert marker.read_bytes() == b'user data'
    assert main(['edit', 'missing', '--output', str(tmp_path / 'absent')]) == 2
    assert '--served-model-name or --chat-template' in capsys.readouterr().err
    assert not (tmp_path / 'absent').exists()
