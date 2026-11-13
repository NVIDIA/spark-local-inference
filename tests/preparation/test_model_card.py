# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model licenses read from checkpoint model cards for the acceptance disclosure."""

from ignition.preparation.model_card import model_license
from ignition.preparation.protection import License


def card(directory, front_matter):
    (directory / 'README.md').write_text(f'---\n{front_matter}---\n# Model\n')
    return directory


def test_spdx_license_links_the_stored_license_file(tmp_path):
    card(tmp_path, 'license: apache-2.0\n')
    (tmp_path / 'LICENSE').write_text('Apache License')
    assert model_license(tmp_path, 'org/model', 'abc123') == License(
        'apache-2.0', 'https://huggingface.co/org/model/blob/abc123/LICENSE')


def test_custom_license_uses_its_name_and_relative_link(tmp_path):
    card(tmp_path, 'license: other\nlicense_name: example-community\nlicense_link: LICENSE.md\n')
    assert model_license(tmp_path, 'org/model', 'abc123') == License(
        'example-community', 'https://huggingface.co/org/model/blob/abc123/LICENSE.md')


def test_absolute_links_are_kept_and_an_unknown_repository_links_nowhere(tmp_path):
    card(tmp_path, 'license: other\nlicense_name: terms\nlicense_link: https://example.com/terms\n')
    assert model_license(tmp_path, None, None) == License('terms', 'https://example.com/terms')
    card(tmp_path, 'license: mit\n')
    assert model_license(tmp_path, None, None) == License('mit', '')
    assert model_license(tmp_path, 'org/model', 'abc') == License('mit', 'https://huggingface.co/org/model')


def test_missing_card_or_license_records_none(tmp_path):
    assert model_license(tmp_path) is None
    card(tmp_path, 'language: en\n')
    assert model_license(tmp_path, 'org/model', 'abc') is None
