# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure complete GPU graph envelopes from Flare Nsight captures.

CUDA API correlation IDs associate GPU activities with NVTX submission ranges.
Every measured decode group contains 255 steps, each including sampling.
"""

import argparse
import bisect
import json
import re
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def summarize(values):
    return {'count': len(values), 'mean_ms': statistics.mean(values) / 1e6,
            'median_ms': statistics.median(values) / 1e6,
            'p90_ms': sorted(values)[int(.9 * (len(values)-1))] / 1e6}

def analyze(root, batches=(1, 2, 4)):
    completed = json.loads((root / 'flare/results.json').read_text())
    groups = [g for g in completed['groups'] if g['batch'] in batches]
    expected = {(b, r, r > 0) for b in batches for r in range(4)}
    assert len(groups) == len(expected) and {(g['batch'], g['repeat'], g['measured']) for g in groups} == expected, \
        'require all selected warmup and measured groups to complete'
    db = sqlite3.connect(f'file:{root / "flare-trace.sqlite"}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    def rows(table):
        return [dict(r) for r in db.execute('select * from ' + table)] if table in tables else []
    strings = {r['id']: r['value'] for r in rows('StringIds')}
    scopes = defaultdict(list)
    for r in rows('NVTX_EVENTS'):
        name = r.get('text') or strings.get(r.get('textId'), '')
        if name.startswith('flare.') and r.get('end') is not None:
            r['name'] = name
            scopes[r['globalTid']].append(r)
    for rs in scopes.values():
        rs.sort(key=lambda r: r['start'])
    starts = {tid: [r['start'] for r in rs] for tid, rs in scopes.items()}
    corr = {}
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
        for api in rows(table):
            tid = api['globalTid']
            if tid not in scopes:
                continue
            i = bisect.bisect_right(starts[tid], api['start']) - 1
            if i >= 0 and api['start'] < scopes[tid][i]['end']:
                key = (tid >> 24, api['correlationId'])
                name = scopes[tid][i]['name']
                assert key not in corr or corr[key] == name
                corr[key] = name
    by_scope = defaultdict(lambda: defaultdict(list))
    unmatched = Counter()
    for table in ('GRAPH_TRACE', 'KERNEL', 'MEMCPY', 'MEMSET'):
        for gpu in rows('CUPTI_ACTIVITY_KIND_' + table):
            name = corr.get((gpu['globalPid'] >> 24, gpu['correlationId']))
            if name:
                by_scope[name][table].append(gpu)
            else:
                unmatched[table] += 1
    print('flare', 'scopes', sum(map(len, scopes.values())), 'matched', len(by_scope), 'unmatched GPU records', dict(unmatched))
    samples = defaultdict(list)
    graphs = json.loads((root / 'flare/graphs.json').read_text())
    decode = {g['graph_exec_id']: g['batch_capacity'] for g in graphs['graphs'] if g['name'] in ('decode1', 'decode2', 'decode4')}
    for name, types in by_scope.items():
        m = re.fullmatch(r'flare.decode.b([124]).r([123])', name)
        if not m:
            continue
        b, repeat = map(int, m.groups())
        if b not in batches:
            continue
        gs = sorted(types['GRAPH_TRACE'], key=lambda g: g['start'])
        assert len(gs) == 255, (name, len(gs))
        assert all(decode[g['graphExecId']] == b for g in gs)
        for pos, g in enumerate(gs, 512):
            samples[b].append({'group': repeat, 'position': pos, 'graph_ns': g['end']-g['start'], 'total_ns': g['end']-g['start'], 'graphs': 1})
    report = {}
    for b, rs in sorted(samples.items()):
        assert len(rs) == 765, (b, len(rs))
        groups = defaultdict(list)
        for r in rs:
            groups[str(r['group'])].append(r['graph_ns'])
        report[b] = {'graph': summarize([r['graph_ns'] for r in rs]), 'forward_sample_gpu': summarize([r['total_ns'] for r in rs]),
                     'graphs_per_step': dict(Counter(r['graphs'] for r in rs)), 'repeat_mean_ms': [statistics.mean(v)/1e6 for v in groups.values()]}
    assert set(report) == set(batches), 'require all selected measured batches'
    db.close()
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--batches', type=int, nargs='+', choices=(1, 2, 4), default=(1, 2, 4))
    args = parser.parse_args()
    report = {'flare': analyze(args.directory.resolve(), args.batches)}
    text = json.dumps(report, indent=2) + '\n'
    (args.directory / 'timings.json').write_text(text)
    print(text, end='')
