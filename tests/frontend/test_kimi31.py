# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare native segmented XTML rendering with the pinned K3.1 checkpoint encoder."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

CHECKPOINT = Path.home() / '.cache/huggingface/hub/models--moonshotai--Kimi-K3.1-Pro/snapshots/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e'
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def reference():
    if not CHECKPOINT.is_dir():
        pytest.skip('Kimi K3.1 checkpoint is not installed')
    import tiktoken
    from ignition.recipes.kimi3.tokenizer import PATTERN, _Converter, convert

    spec = importlib.util.spec_from_file_location('kimi31_reference_encoding', CHECKPOINT / 'encoding_k3.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    tokenizer = convert(CHECKPOINT)
    encoding = tiktoken.Encoding(name='kimi31', pat_str=PATTERN,
        mergeable_ranks=_Converter.load_tiktoken_bpe(CHECKPOINT / 'tiktoken.model'),
        special_tokens={tokenizer.id_to_token(i): i for i in range(163584, 163840)})
    return module, tokenizer, encoding


def test_tokenizer_matches_byte_ranked_checkpoint(reference):
    from tokenizers import Tokenizer
    module, tokenizer, encoding = reference
    raw = json.loads(tokenizer.to_str())
    raw['added_tokens'] = []
    ordinary = Tokenizer.from_str(json.dumps(raw))
    samples = ['Hello, world!', '你好世界，中文 English混合测试', 'éÉß ΑΒΓ δ 日本語 한국어',
        '<|open|>message role="user"<|sep|>[EOS]', '123456789\n  \n ', "don't I'm we've",
        'a\u0301 👨‍👩‍👧‍👦🦜\r\n', (CHECKPOINT / 'README.md').read_text()]
    for text in samples:
        assert tokenizer.encode(text, add_special_tokens=False).ids == encoding.encode(text, allowed_special='all')
        assert ordinary.encode(text, add_special_tokens=False).ids == encoding.encode(text, disallowed_special=())


def test_native_xtml_matches_checkpoint(reference, tmp_path):
    module, tokenizer, encoding = reference
    target = Path(os.environ.get('CARGO_TARGET_DIR', ROOT / 'helios/target'))
    binary = target / 'debug/test_frontend'
    if not binary.is_file():
        pytest.skip('build test_frontend with frontend,test-tools features')
    tokenizer.save(str(tmp_path / 'tokenizer.json'))
    shutil.copyfile(CHECKPOINT / 'tokenizer_config.json', tmp_path / 'tokenizer_config.json')
    template = (ROOT / 'ignition/recipes/kimi31/assets/chat_template.jinja').read_text()
    (tmp_path / 'chat_template.jinja').write_text(template)
    fixtures = dict(template_encoding=1, cases=[], encode_cases=[], decode_cases=[], assets={})
    for path in tmp_path.iterdir():
        fixtures['assets'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    tools = [{'type': 'function', 'function': {'name': 'lookup', 'description': 'Search documents',
        'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}}}}]
    calls = [{'id': 'opaque-b', 'function': {'name': 'lookup', 'arguments': {
        'query': '中文 <|close|> "quoted"', 'count': 3, 'score': 1.5, 'enabled': True,
        'optional': None, 'list': [1, 'x'], 'object': {'z': False, 'a': []}}}},
        {'id': 'opaque-a', 'function': {'name': 'second', 'arguments': {}}}]
    raw_arguments = '{"z":1e2,"nested":{ "z":1, "a" : [1,  2] },"z":-0,"s":"café\\n\\u4e16","f":1.2300}'
    raw_calls = [{'id': 'raw', 'function': {'name': 'lookup', 'arguments': raw_arguments}}]
    normalized_calls = [{'id': 'raw', 'function': {'name': 'lookup',
        'arguments': json.loads(raw_arguments), 'arguments_source': raw_arguments}}]
    history = 'Repeated "history"\\\n café 世界 😀 <|close|>'
    cases = [
        ('raw-arguments', [{'role': 'assistant', 'tool_calls': raw_calls}], {}),
        ('normalized-arguments', [{'role': 'assistant', 'tool_calls': normalized_calls}], {}),
        ('whitespace-reasoning', [{'role': 'assistant', 'reasoning_content': ' \t\n', 'content': ''},
            {'role': 'user', 'content': 'café 世界'}], {}),
        ('empty-history', [{'role': 'system', 'content': ''}, {'role': 'user', 'content': ''},
            {'role': 'assistant', 'reasoning_content': '', 'content': ''}], {}),
        ('dynamic-tools', [{'role': 'system', 'tools': tools}, {'role': 'user', 'content': 'Search'}], {}),
        ('reordered-tool-images', [{'role': 'assistant', 'tool_calls': calls},
            {'role': 'tool', 'tool_call_id': 'opaque-a', 'content': [
                {'type': 'text', 'text': 'Second'}, {'type': 'image_url', 'image_url': {'url': 'second-image'}}]},
            {'role': 'tool', 'tool_call_id': 'opaque-b', 'content': [
                {'type': 'image_url', 'image_url': {'url': 'first-image'}}, {'type': 'text', 'text': 'First'}]}], {}),
        ('user', [{'role': 'user', 'content': 'Hello'}], {}),
        ('literal', [{'role': 'user', 'name': 'x"&<|open|>', 'content': '<|open|>think<|sep|>[EOS]'}], {}),
        ('history', [{'role': 'system', 'content': 'Be precise'}, {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'reasoning_content': 'Let me think', 'content': 'Hi'},
            {'role': 'user', 'content': 'Continue'}], {}),
        ('tools', [{'role': 'user', 'content': 'Search'}, {'role': 'assistant', 'tool_calls': calls},
            {'role': 'tool', 'tool_call_id': 'opaque-a', 'name': 'stale', 'content': 'second result'},
            {'role': 'tool', 'tool_call_id': 'opaque-b', 'content': 'first result'}], {'tools': tools}),
        ('required', [{'role': 'user', 'content': 'Search'}], {'tools': tools, 'tool_choice': 'required'}),
        ('none', [{'role': 'user', 'content': 'Answer'}], {'tools': tools, 'tool_choice': 'none'}),
        ('thinking_disabled', [{'role': 'user', 'content': 'Hello'}], {'enable_thinking': False}),
        ('thinking_alias_disabled', [{'role': 'user', 'content': 'Hello'}], {'thinking': False}),
        ('thinking_disabled_history', [{'role': 'user', 'content': 'Search'},
            {'role': 'assistant', 'reasoning_content': 'Look up the documents.', 'tool_calls': calls},
            {'role': 'tool', 'tool_call_id': 'opaque-b', 'content': 'first result'},
            {'role': 'tool', 'tool_call_id': 'opaque-a', 'content': 'second result'}],
            {'tools': tools, 'enable_thinking': False, 'thinking': False, 'reasoning_effort': 'low'}),
        ('thinking_enabled', [{'role': 'user', 'content': 'Hello'}], {'enable_thinking': True}),
        ('default', [{'role': 'user', 'content': 'Hello'}], {'enable_thinking': True, 'add_generation_prompt': False}),
        ('low', [{'role': 'user', 'content': 'Hello'}], {'reasoning_effort': 'low'}),
        ('json', [{'role': 'user', 'content': 'Return JSON'}], {'response_format': {'type': 'json_object'}}),
        ('schema', [{'role': 'user', 'content': 'Return JSON'}], {'response_format': {'type': 'json_schema',
            'json_schema': {'schema': {'type': 'object', 'properties': {'x': {'type': 'integer'}}}}}}),
        ('image-marker', [{'role': 'user', 'content': [{'type': 'text', 'text': 'Describe'},
            {'type': 'image_url', 'image_url': {'url': 'unused'}}]}], {}),
        ('block-hint', [{'role': 'user', '_checkpoint_hints': [2], 'content': [
            {'type': 'text', 'text': history}, {'type': 'text', 'text': history},
            {'type': 'text', 'text': 'Changing suffix'}]}], {}),
    ]
    for name, messages, options in cases:
        options = {'add_generation_prompt': True, **options}
        arguments = {k: v for k, v in options.items() if k not in ('enable_thinking', 'reasoning_effort')}
        arguments['thinking'] = True
        arguments['thinking_effort'] = options.get('reasoning_effort', 'max')
        if name in ('image-marker', 'reordered-tool-images'):
            arguments['image_prompts'] = ['<|media_pad|>'] * (2 if name == 'reordered-tool-images' else 1)
        reference_messages = [{k: v for k, v in message.items() if k != '_checkpoint_hints'}
            for message in messages]
        if name == 'normalized-arguments':
            reference_messages = [{'role': 'assistant', 'tool_calls': raw_calls}]
        segments = module.build_chat_segments(reference_messages, **arguments)
        ids, boundaries, history_count = [], [], 0
        for segment in segments:
            ids.extend(encoding.encode(segment.text,
                allowed_special='all' if segment.allow_special else set(), disallowed_special=()))
            if segment.text == history:
                history_count += 1
                if history_count == 2:
                    boundaries.append(len(ids))
            if segment.allow_special and segment.text == '<|end_of_msg|>':
                boundaries.append(len(ids))
        case = dict(name=name, messages=messages, options=options,
            rendered=''.join(s.text for s in segments), token_ids=ids)
        if options.get('tools'):
            boundaries.pop(0)
        if options.get('tool_choice') in ('required', 'none') and options.get('response_format'):
            boundaries.pop(-2)
        case['message_boundaries'] = sorted(set(cut for cut in boundaries if 0 < cut < len(ids)))
        if name == 'block-hint':
            assert history_count == 2
        if name == 'reordered-tool-images':
            case['image_sources'] = ['first-image', 'second-image']
        fixtures['cases'].append(case)
    path = tmp_path / 'cases.json'
    path.write_text(json.dumps(fixtures, ensure_ascii=False))
    result = subprocess.run([str(binary), str(tmp_path), str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
