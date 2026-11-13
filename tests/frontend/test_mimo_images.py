# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native MiMo patches and prompt expansion match the Transformers image processor."""

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

from ignition.recipes.mimo_v2.frontend import chat_template_asset
from ignition.recipes.mimo_v2.recipe import RECIPE

ROOT = Path(__file__).resolve().parents[2]
HUB = Path.home() / '.cache/huggingface/hub'


def snapshot(alias):
    checkpoint = next(c for c in RECIPE.checkpoints if c.alias == alias)
    return HUB / f"models--{checkpoint.repository.replace('/', '--')}" / 'snapshots' / checkpoint.revision


CHECKPOINTS = pytest.mark.parametrize('alias', ['mimo-v2.6-pro-mopd', 'mimo-v2.6-flash-rl'])


@CHECKPOINTS
def test_native_mimo_chat_matches_transformers(tmp_path, alias):
    SNAPSHOT = snapshot(alias)
    if not all((SNAPSHOT / name).is_file() for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')):
        pytest.skip('MiMo checkpoint frontend is not cached')
    subprocess.run(['cargo', 'build', '--quiet', '--manifest-path', str(ROOT / 'helios/Cargo.toml'),
                    '--features', 'frontend,test-tools', '--bin', 'test_frontend'], check=True)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    user = dict(role='user', content='What is 17 times 23?')
    cases = []

    def add(name, messages, **options):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, **options)
        cases.append(dict(name=name, messages=messages, options=options, rendered=rendered,
                          token_ids=tokenizer.encode(rendered, add_special_tokens=False)))

    add('default', [user], add_generation_prompt=False)
    add('thinking_disabled', [user], add_generation_prompt=True, enable_thinking=False)
    add('thinking_enabled', [user], add_generation_prompt=True, enable_thinking=True)
    for tools in (None, [], [dict(type='function', function=dict(name='lookup', parameters={}))]):
        add(f'tools_{tools}', [user], add_generation_prompt=True, tools=tools)
    add('assistant_null_fields', [user, dict(role='assistant', content=None, tool_calls=None), user],
        add_generation_prompt=True)
    add('images', [dict(role='user', content=[dict(type='image'), dict(type='text', text='Describe.')])],
        add_generation_prompt=True)
    add('tool_history', [user, dict(role='assistant', content='', tool_calls=[dict(type='function',
        function=dict(name='lookup', arguments=dict(city='Paris', count=2)))]),
        dict(role='tool', content='Sunny.')], add_generation_prompt=True)
    annotated = chat_template_asset(SNAPSHOT)
    native_assets = tmp_path / 'assets'
    native_assets.mkdir()
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        (native_assets / name).symlink_to(SNAPSHOT / name)
    (native_assets / 'chat_template.jinja').write_text(annotated['text'])
    assets = {name: hashlib.sha256((native_assets / name).read_bytes()).hexdigest()
              for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')}
    fixture = tmp_path / 'chat.json'
    fixture.write_text(json.dumps(dict(assets=assets, cases=cases, encode_cases=[], decode_cases=[])))
    subprocess.run([str(ROOT / 'helios/target/debug/test_frontend'), str(native_assets), str(fixture)], check=True)


@pytest.fixture(scope='module')
def image_tool():
    subprocess.run(['cargo', 'build', '--quiet', '--manifest-path', str(ROOT / 'helios/Cargo.toml'),
                    '--features', 'server,test-tools', '--bin', 'test_mimo_images'], check=True)
    return ROOT / 'helios/target/debug/test_mimo_images'


@CHECKPOINTS
@pytest.mark.parametrize('sizes', [[(1, 1)], [(64, 64)], [(701, 503)], [(3000, 47)],
                                  [(17, 3000)], [(503, 701), (149, 31)]])
def test_native_mimo_images_match_transformers(image_tool, tmp_path, alias, sizes):
    SNAPSHOT = snapshot(alias)
    if not all((SNAPSHOT / name).is_file() for name in ('preprocessor_config.json', 'tokenizer.json', 'chat_template.jinja')):
        pytest.skip('MiMo checkpoint frontend is not cached')
    processor = Qwen2VLImageProcessorPil.from_pretrained(SNAPSHOT, local_files_only=True)
    # The native tool reads the preprocessor settings and the annotated chat template from one directory.
    native_assets = tmp_path / 'assets'
    native_assets.mkdir()
    (native_assets / 'preprocessor_config.json').symlink_to(SNAPSHOT / 'preprocessor_config.json')
    (native_assets / 'chat_template.jinja').write_text(chat_template_asset(SNAPSHOT)['text'])
    tokenizer = Tokenizer.from_file(str(SNAPSHOT / 'tokenizer.json'))
    prompt = 'Describe ' + '<|vision_start|><|image_pad|><|vision_end|>' * len(sizes)
    inputs, images = [], []
    for index, (width, height) in enumerate(sizes):
        rgb = np.random.default_rng(index + 51).integers(0, 256, (height, width, 3), dtype=np.uint8)
        path = tmp_path / f'image{index}.rgb'
        path.write_bytes(rgb.tobytes())
        inputs.append(dict(width=width, height=height, pixels=str(path)))
        images.append(Image.fromarray(rgb))
    expected = processor(images=images, return_tensors='pt')
    grids = expected['image_grid_thw'].tolist()
    counts = [h * w for _, h, w in grids]
    positions, cu = [], [0]
    for _, h, w in grids:
        positions.extend([[y + dy, x + dx] for y in range(0, h, 2) for x in range(0, w, 2)
                          for dy in range(2) for dx in range(2)])
        cu.append(cu[-1] + h * w)
    tokens, indices, offset, image = [], [], 0, 0
    for token in tokenizer.encode(prompt, add_special_tokens=False).ids:
        if token != 151655:
            tokens.append(token)
            indices.append(-1)
        else:
            count = counts[image] // 4
            tokens.extend([token] * count)
            indices.extend(range(offset, offset + count))
            offset += count
            image += 1
    case = tmp_path / 'case.json'
    case.write_text(json.dumps(dict(images=inputs, prompt=prompt)))
    subprocess.run([str(image_tool), str(SNAPSHOT), str(native_assets), str(case), str(tmp_path)], check=True)
    actual = json.loads((tmp_path / 'metadata.json').read_text())
    assert actual == dict(tokens=tokens, indices=indices, grids=grids, patches=counts,
                          positions=positions, cu_seqlens=cu)
    pixels = torch.from_numpy(np.fromfile(tmp_path / 'pixels.f32', dtype=np.float32))
    reference = expected['pixel_values'].flatten()
    # The native frontend scales each byte by a float32 reciprocal of 255; the reference divides in
    # float64 and narrows to float32. The scaled values differ by at most one float32 ulp, which the
    # channel's standard deviation amplifies, before each side rounds the normalized value once more.
    eps = torch.finfo(torch.float32).eps
    atol = eps / min(processor.image_std) + 2 * eps * float(reference.abs().max())
    torch.testing.assert_close(pixels, reference, rtol=0, atol=atol)
