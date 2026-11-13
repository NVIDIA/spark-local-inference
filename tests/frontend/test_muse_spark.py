# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint tokenizer parity for recipient headers, escaping, and ATEM history."""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = Path(os.environ.get('FLARE_MUSE_SPARK_SNAPSHOT', '/data/cache/huggingface/hub/models--meta-models--Muse-Spark-Open-816B-A42B-NVFP4/snapshots/4239e4db19376904f2e82c167e78f70abb5455fe'))


def test_native_muse_chat_matches_checkpoint(tmp_path):
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    from transformers.utils.chat_template_utils import _compile_jinja_template
    from ignition.recipes.muse_spark.frontend import prepare_frontend, runtime_assets

    if not (SNAPSHOT / 'tokenizer.json').is_file():
        pytest.skip('requires pinned Muse Spark checkpoint frontend assets')
    assets = runtime_assets(SNAPSHOT)
    special = prepare_frontend(SNAPSHOT, eos_token_ids=[200001, 200008])['tokenizer']
    configured_eos = json.loads((SNAPSHOT / 'config.json').read_text())['text_config']['eos_token_id']
    assert prepare_frontend(SNAPSHOT, eos_token_ids=configured_eos) == prepare_frontend(SNAPSHOT, eos_token_ids=[200001, 200008])
    tokenizer = Tokenizer.from_file(str(SNAPSHOT / 'tokenizer.json'))
    hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(SNAPSHOT)
    source = (SNAPSHOT / 'chat_template.jinja').read_text()
    annotated = assets['chat_template']['text']
    assert annotated.replace('{%- mark -%}', '') == source
    template = _compile_jinja_template(source)
    marked_template = _compile_jinja_template(annotated.replace('{%- mark -%}', '{{- message_position() -}}'))
    cases = []

    def add(name, messages, **options):
        options = dict(dict(current_date='2026-10-27', add_generation_prompt=True), **options)
        if options.get('current_date') == 'clock':
            del options['current_date']
        rendered = template.render(messages=messages, **special, **options)
        marker = '\0message-position\0'
        marked = marked_template.render(messages=messages, **special, **options, message_position=lambda: marker)
        chunks = marked.split(marker)
        assert ''.join(chunks) == rendered
        if options.get('continue_final_message'):
            rendered = hf_tokenizer.apply_chat_template(messages, tokenize=False, chat_template=source, **options)
        positions, position = [], 0
        for chunk in chunks[:-1]:
            position += len(chunk)
            positions.append(position)
        hint_positions = []
        for mi, message in enumerate(messages):
            for block in message.get('_checkpoint_hints', []):
                hinted = copy.deepcopy(messages)
                sentinel = '\0hint-position\0'
                hinted[mi]['content'][block - 1]['text'] += sentinel
                hinted_text = template.render(messages=hinted, **special, **options)
                assert hinted_text.replace(sentinel, '') == rendered
                hint_positions.append(hinted_text.index(sentinel))
        encoding = tokenizer.encode(rendered, add_special_tokens=False)
        prefix_ends = [0]
        for _, end in encoding.offsets:
            prefix_ends.append(max(prefix_ends[-1], end))
        suffix_starts = [len(rendered)] * (len(encoding.ids) + 1)
        for index in range(len(encoding.ids) - 1, -1, -1):
            suffix_starts[index] = min(suffix_starts[index + 1], encoding.offsets[index][0])
        boundaries = sorted({index for position in positions for index in range(1, len(encoding.ids))
                             if prefix_ends[index] <= position <= suffix_starts[index]})
        for position in hint_positions:
            safe = max(index for index, end in enumerate(prefix_ends) if end <= position)
            if 0 < safe < len(encoding.ids):
                boundaries = sorted(set([*boundaries, safe]))
        assert boundaries
        cases.append(dict(name=name, messages=messages, options=options, rendered=rendered,
                          token_ids=encoding.ids, message_boundaries=boundaries))

    user = dict(role='user', content='Hello 世界 café á 👨‍👩‍👧‍👦 <|eot|> <image> <audio> <video>')
    add('default', [user])
    add('closed_header', [user], add_generation_prompt=False)
    add('clock_date', [user], current_date='clock')
    for content in ['café 世界', ' repeated repeated ']:
        add('continuation_'+content, [user, dict(role='assistant', content=content)],
            continue_final_message=True, add_generation_prompt=False)
    for effort in ['low', 'medium', 'high', 'xhigh']:
        add(effort, [user], reasoning_effort=effort)
    for role in ['system', 'developer']:
        add(role, [dict(role=role, content='Reasoning effort should be precise.'), user])
    add('history', [user, dict(role='assistant', reasoning_content='Plan 世界', content='Answer'), user])
    add('channels', [user, dict(role='assistant', recipient='commentary', content='Checking'),
                     dict(role='assistant', recipient='self', content='Plan'), dict(role='assistant', content='Done'), user])
    tools = [dict(type='function', function=dict(name='lookup', description='Search <|start|>',
        parameters=dict(type='object', properties=dict(query=dict(type='string'), n=dict(type='integer')))))]
    add('tools', [user], tools=tools)
    history = 'Repeated 世界 <|eom|> & café '
    for suffix in ['Suffix one', 'Suffix two']:
        for hinted in [False, True]:
            message = dict(role='user', content=[dict(type='text', text=history),
                dict(type='text', text=history), dict(type='text', text=suffix)])
            if hinted:
                message['_checkpoint_hints'] = [2]
            add(f'hints_{suffix}_{hinted}', [message])
    add('empty_schema', [user], tools=[dict(type='function', function=dict(name='empty'))])
    calls = [dict(id='one', type='function', function=dict(name='lookup', arguments=dict(
        query=' 世界 <|eom|> & "quotes" \n', n=3, enabled=True, absent=None,
        nested=dict(text='中文'), items=[1, False], empty=[]))),
        dict(id='two', type='function', function=dict(name='lookup', arguments=dict(query='again')))]
    add('tool_results', [user, dict(role='assistant', content='Checking', reasoning_content='Plan', tool_calls=calls),
        dict(role='tool', tool_call_id='one', content='Result <|start|>'),
        dict(role='tool', tool_call_id='two', content='Second')], tools=tools)
    add('escaping_headers', [user, dict(role='assistant', content='', tool_calls=[dict(id='x', type='function',
        function=dict(name='a\r\n"<&', arguments={'k"<&': '<|start|>'}))]),
        dict(role='tool', tool_call_id='x', content='')], tools=tools)
    for effort in ['none', 'minimal', 'max']:
        cases.append(dict(name='invalid_'+effort, messages=[user], options=dict(reasoning_effort=effort),
                          error=f'Unsupported reasoning_effort: {effort}. Expected one of: low, medium, high, xhigh.'))

    incompatible = [user, dict(role='assistant', content='<|eom|> content')]
    with pytest.raises(ValueError, match='final message does not appear'):
        hf_tokenizer.apply_chat_template(incompatible, tokenize=False, chat_template=source,
            continue_final_message=True, add_generation_prompt=False)
    cases.append(dict(name='escaped_continuation_error', messages=incompatible,
        options=dict(continue_final_message=True, add_generation_prompt=False),
        error='continue_final_message is set but the final message does not appear in the chat after applying the chat template!'))
    samples = ['Hello café 世界 á 👨‍👩‍👧‍👦', ' to=self<|message|>Plan<|eom|><|start|>assistant to=user<|message|>Done']
    encode_cases = [dict(name=f'encode{i}', text=text, add_special_tokens=False,
        token_ids=tokenizer.encode(text, add_special_tokens=False).ids) for i, text in enumerate(samples)]
    decode_cases = [dict(name=f'decode{i}-{skip}', token_ids=case['token_ids'], skip_special_tokens=skip,
        text=tokenizer.decode(case['token_ids'], skip_special_tokens=skip))
        for i, case in enumerate(encode_cases) for skip in [True, False]]

    manifest = str(ROOT / 'helios/Cargo.toml')
    subprocess.run(['cargo', 'build', '--quiet', '--manifest-path', manifest,
                    '--features', 'frontend,test-tools', '--bin', 'test_frontend'], check=True)
    target = json.loads(subprocess.check_output(['cargo', 'metadata', '--format-version', '1', '--no-deps',
                                                '--manifest-path', manifest], text=True))['target_directory']
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    for name in ['tokenizer.json', 'tokenizer_config.json']:
        (runtime / name).symlink_to(SNAPSHOT / name)
    (runtime / 'chat_template.jinja').write_text(assets['chat_template']['text'])
    fingerprints = {'tokenizer.json': assets['tokenizer']['sha256'],
                    'chat_template.jinja': assets['chat_template']['sha256'],
                    'tokenizer_config.json': hashlib.sha256((runtime / 'tokenizer_config.json').read_bytes()).hexdigest()}
    fixture = tmp_path / 'chat.json'
    fixture.write_text(json.dumps(dict(assets=fingerprints, cases=cases, encode_cases=encode_cases, decode_cases=decode_cases,
        template_defaults=dict(reasoning_effort='high', current_date='2026-10-27', add_generation_prompt=True),
        default_overrides=[['high', {}], ['low', dict(reasoning_effort='low')],
                           ['closed_header', dict(reasoning_effort='xhigh', add_generation_prompt=False)]])))
    result = subprocess.run([str(Path(target) / 'debug/test_frontend'), str(runtime), str(fixture)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
