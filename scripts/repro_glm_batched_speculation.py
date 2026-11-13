#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce GLM-5.3 Flash answer corruption under batched speculative decoding.

Standard library only; point it at a Flare server's OpenAI-compatible endpoint.

``cached`` publishes one retrieval prompt (a single-token request), then sends
``--concurrency`` copies at once with distinct seeds. Every copy restores the
same checkpoint, so all rows of the speculative batch are live and decode
together. ``cold`` sends prompts that cannot reuse a cached prefix (a nonce
leads each prompt), one at a time, so each request decodes right after its own
prefill. Full requests and replies are saved with ``--output``.

    python3 scripts/repro_glm_batched_speculation.py cached --url http://host:8000 --concurrency 8
    python3 scripts/repro_glm_batched_speculation.py cold --url http://host:8000 --size 70000 --count 3

The answer to the default ``--size 8000`` prompt (about 10.5K tokens) is
``2U2GYCZF, 198``. Sampling uses the model card's T=1, top-p .95. Each reply is
classified as:

- ``correct``: the code and record 198, and no other code-like token;
- ``neighbour``: the code with record 197 or 199;
- ``misplaced``: the code with another record (record 99 is the usual one);
- ``corrupt``: empty, or a wrong, garbled or additional invented code.

Neighbour and misplaced answers are model-level misses that also occur without
speculation and on unaffected servers. Corrupt answers are the defect.

The script also reads the server's speculative round counters before and after
the test and reports how many rounds ran in graphs with more than 16 query rows
(batch capacity times verification width); the corruption appears in those.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import re
import time
import urllib.error
import urllib.request

WORDS = 'the of and to in is was for on with as by at from that this which are be or an it not'.split()
CODE_LIKE = re.compile(r'\b(?=[A-Z0-9-]*\d)(?=[A-Z0-9-]*[A-Z])[A-Z0-9][A-Z0-9-]{3,13}\b')
ROUNDS = re.compile(r'^flare_speculative_rounds_total\{batch="(\d+)",width="(\d+)"\} (\d+)', re.M)


def fixture(size, nonce, seed, max_tokens, model):
    rng = random.Random(4242 + size)
    code = ''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for _ in range(8))
    records = [f'Record {i}: ' + ' '.join(rng.choice(WORDS) for _ in range(10)) + '.' for i in range(size // 12)]
    at = int(len(records) * .3)
    records.insert(at, f'IMPORTANT: the vault access code is {code}. Remember it.')
    question = ('What is the vault access code, and what is the number of the record written immediately before the'
                ' line that states it? Answer as: code, record number.')
    prompt = f'Session {nonce}.\n' + '\n'.join(records) + '\n\n' + question
    body = dict(model=model, messages=[dict(role='user', content=prompt)], max_tokens=max_tokens,
                temperature=1.0, top_p=0.95, seed=seed)
    return body, code, at - 1


def send(url, body):
    request = urllib.request.Request(url + '/v1/chat/completions', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
    try:
        return json.load(urllib.request.urlopen(request, timeout=3600))
    except urllib.error.HTTPError as error:
        return {'error': error.read().decode(errors='replace')[:500]}


def rounds(url):
    """Speculative rounds by (batch, width), or None when the server has no Flare metrics."""
    try:
        text = urllib.request.urlopen(url + '/metrics', timeout=10).read().decode()
    except (OSError, ValueError):
        return None
    return {(int(b), int(w)): int(n) for b, w, n in ROUNDS.findall(text)}


def classify(response, code, record):
    if 'error' in response:
        return 'error', response['error']
    content = response['choices'][0]['message'].get('content') or ''
    others = {token for token in CODE_LIKE.findall(content) if token != code}
    numbers = {int(n) for n in re.findall(r'\b\d{1,6}\b', content)}
    if code in content and not others:
        if record in numbers:
            return 'correct', content
        if numbers & {record - 1, record + 1}:
            return 'neighbour', content
        return 'misplaced', content
    return 'corrupt', content


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('mode', choices=('cached', 'cold'))
    parser.add_argument('--url', default='http://127.0.0.1:8000', help='server origin, without /v1')
    parser.add_argument('--model', default='glm-5.3-flash')
    parser.add_argument('--size', type=int, default=8000, help='fixture size; 8000 gives about 10.5K prompt tokens')
    parser.add_argument('--concurrency', type=int, default=8, help='cached: simultaneous requests')
    parser.add_argument('--count', type=int, default=3, help='cold: sequential requests')
    parser.add_argument('--max-tokens', type=int, default=1024, help='a correct answer takes about 150 tokens')
    parser.add_argument('--seed', type=int, default=1101)
    parser.add_argument('--nonce', default=None, help='prompt prefix; default derives one from the clock')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    nonce = args.nonce or f'r{int(time.time())}'
    results = []
    started = time.monotonic()
    if args.mode == 'cached':
        send(args.url, fixture(args.size, nonce, args.seed, 1, args.model)[0])
        before = rounds(args.url)
        bodies = [fixture(args.size, nonce, args.seed + i, args.max_tokens, args.model)[0]
                  for i in range(args.concurrency)]
        _, code, record = fixture(args.size, nonce, args.seed, 1, args.model)
        with ThreadPoolExecutor(args.concurrency) as pool:
            responses = list(pool.map(lambda body: send(args.url, body), bodies))
        results = list(zip(bodies, responses))
    else:
        before = rounds(args.url)
        for i in range(args.count):
            body, code, record = fixture(args.size, f'{nonce}-{i}', args.seed, args.max_tokens, args.model)
            results.append((body, send(args.url, body)))
    after = rounds(args.url)
    counts = Counter()
    saved = []
    for i, (body, response) in enumerate(results):
        verdict, content = classify(response, code, record)
        counts[verdict] += 1
        usage = response.get('usage', {})
        print(f'{i:2d} {verdict:9s} completion_tokens={usage.get("completion_tokens")} {content[-120:]!r}', flush=True)
        saved.append(dict(request=body, response=response, verdict=verdict))
    summary = ', '.join(f'{counts[k]} {k}' for k in ('correct', 'neighbour', 'misplaced', 'corrupt', 'error') if counts[k])
    print(f'{summary} of {len(results)} (answer {code}, {record}) in {time.monotonic() - started:.1f}s')
    shapes = {}
    if before is not None and after is not None:
        shapes = {key: after[key] - before.get(key, 0) for key in after if after[key] > before.get(key, 0)}
        wide = sum(n for (b, w), n in shapes.items() if b * w > 16)
        listed = ' '.join(f'{b}x{w}:{n}' for (b, w), n in sorted(shapes.items()))
        print(f'speculative rounds by batch x width: {listed or "none"}; '
              f'{wide} of {sum(shapes.values())} in graphs above 16 query rows')
    if args.output:
        args.output.write_text(json.dumps(dict(results=saved, rounds={f'{b}x{w}': n for (b, w), n in shapes.items()},
                                               counts=dict(counts)), indent=2))


if __name__ == '__main__':
    main()
