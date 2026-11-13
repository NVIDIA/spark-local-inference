# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal typed frontend metadata for CUDA execution fixtures."""

import hashlib
import json


def server_config(root):
    tokenizer = json.dumps({'version': '1.0', 'model': {'type': 'WordLevel',
        'vocab': {'<unk>': 0, '<eos>': 1}, 'unk_token': '<unk>'}}).encode()
    assets = {}
    for name, contents in [('tokenizer', tokenizer), ('chat_template', b'{{ messages[0].content }}')]:
        path = root / f'{name}.txt'
        path.write_bytes(contents)
        assets[name] = {'path': str(path), 'sha256': hashlib.sha256(contents).hexdigest()}
    return {'assets': assets, 'frontend': {'tokenizer': {'eos_token': '<eos>'}, 'eos_token_ids': [1]},
            'sampling_defaults': {'seed': 0, 'temperature': 1.0, 'top_p': 1.0, 'top_k': -1,
                'min_p': 0.0, 'repetition_penalty': 1.0, 'presence_penalty': 0.0, 'frequency_penalty': 0.0}}
