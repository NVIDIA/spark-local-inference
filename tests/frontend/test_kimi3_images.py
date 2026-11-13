# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoonViT native preprocessing against the checkpoint's image processor."""
import io
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = Path('/data/models/Kimi-K3-MXFP4-CSF')


@pytest.fixture(scope='module')
def reference(tmp_path_factory):
    if not CHECKPOINT.is_dir():
        pytest.skip('Kimi K3 checkpoint is unavailable')
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from ignition.recipes.kimi3.tokenizer import convert
    cls = get_class_from_dynamic_module('kimi_k3_vision_processing.KimiK3VisionProcessor', str(CHECKPOINT))
    processor = cls.from_pretrained(CHECKPOINT)
    assets = tmp_path_factory.mktemp('kimi3-assets')
    tokenizer = convert(CHECKPOINT)
    tokenizer.save(str(assets / 'tokenizer.json'))
    shutil.copyfile(CHECKPOINT / 'tokenizer_config.json', assets / 'tokenizer_config.json')
    shutil.copyfile(ROOT / 'ignition/recipes/kimi3/assets/chat_template.jinja', assets / 'chat_template.jinja')
    subprocess.run(['cargo', 'build', '--quiet', '--manifest-path', str(ROOT / 'helios/Cargo.toml'),
        '--features', 'server,test-tools', '--bin', 'test_kimi3_images'], check=True)
    return processor, assets, tokenizer


@pytest.mark.parametrize('mode,size', [('RGB', (1, 1)), ('RGB', (403, 207)), ('RGBA', (83, 127)),
    ('RGBA', (8000, 37)), ('RGB', (8000, 37)), ('P', (8000, 37)), ('LA', (8000, 37)),
    ('RGB_transparency', (83, 127)), ('L', (503, 701)), ('1', (8000, 37)), ('JPEG', (701, 503)), ('CMYK', (8000, 37)),
    ('WEBP_RGB', (403, 207)), ('WEBP_RGBA', (83, 127))])
def test_native_processor(reference, tmp_path, mode, size):
    processor, assets, tokenizer = reference
    width, height = size
    rng = np.random.default_rng(41)
    rgba = rng.integers(0, 256, (height, width, 4), dtype=np.uint8)
    image = Image.fromarray(rgba, 'RGBA').convert('RGB' if mode in ('JPEG', 'RGB_transparency') else mode.removeprefix('WEBP_'))
    if mode == 'RGB_transparency':
        image.info['transparency'] = image.getpixel((0, 0))
    if mode == 'P':
        image.info['transparency'] = bytes(range(256))
    if mode.startswith('WEBP_'):
        exif = Image.Exif(); exif[274] = 6
        path = tmp_path / 'image.webp'; image.save(path, lossless=True, exact=True, exif=exif)
    elif mode in ('JPEG', 'CMYK'):
        exif = Image.Exif(); exif[274] = 6
        path = tmp_path / 'image.jpg'; image.save(path, exif=exif)
    else:
        path = tmp_path / 'image.png'; image.save(path)
    encoded = path.read_bytes()
    expected = processor.preprocess([{'type': 'image', 'image': encoded}], return_tensors='pt')
    case = tmp_path / 'case.json'
    case.write_text(json.dumps({'images': [{'path': str(path), 'format': 'webp' if mode.startswith('WEBP_') else 'jpeg' if mode in ('JPEG', 'CMYK') else 'png'}],
        'prompt': 'Look: <|media_pad|> describe.'}))
    subprocess.run([str(ROOT / 'helios/target/debug/test_kimi3_images'), str(assets), str(assets), str(case), str(tmp_path)], check=True)
    actual = json.loads((tmp_path / 'metadata.json').read_text())
    pixels = np.fromfile(tmp_path / 'pixels.f32', dtype=np.float32)
    np.testing.assert_array_equal(pixels, expected['pixel_values'].flatten().numpy())
    assert actual['grids'] == expected['grid_thws'].tolist()
    _, h, w = actual['grids'][0]
    prefix = f'Look: <|media_begin|>image {width}x{height}<|media_content|>'
    suffix = '<|media_end|> describe.'
    count = h * w // 4
    tokens = tokenizer.encode(prefix, add_special_tokens=False).ids + [163605] * count + tokenizer.encode(suffix, add_special_tokens=False).ids
    assert actual['tokens'] == tokens
    assert actual['indices'] == [-1] * len(tokenizer.encode(prefix, add_special_tokens=False).ids) + list(range(count)) + [-1] * len(tokenizer.encode(suffix, add_special_tokens=False).ids)
    table = torch.randn(64, 64, 3)
    wanted = torch.nn.functional.interpolate(table.permute(2, 0, 1).unsqueeze(0), (h, w), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0).reshape(-1, 3)
    indices, weights = torch.tensor(actual['interpolation_indices']), torch.tensor(actual['interpolation_weights'])
    actual_positions = (table.reshape(-1, 3)[indices] * weights[:, :, None]).sum(1)
    torch.testing.assert_close(actual_positions, wanted, atol=2e-5, rtol=2e-5)
