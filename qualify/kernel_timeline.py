# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analyze captured node activities without substituting graph envelopes."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import quote

from qualify.performance_trace import Intervals, distribution


GPU_TABLES = {'CUPTI_ACTIVITY_KIND_KERNEL': 0, 'CUPTI_ACTIVITY_KIND_MEMCPY': 1,
              'CUPTI_ACTIVITY_KIND_MEMSET': 2}
LAUNCH = re.compile(r'^cu(?:da)?GraphLaunch(?:$|[_\d])')


def clone_parents(db, tables, pid):
    parents = {}
    if 'CUDA_GRAPH_NODE_EVENTS' not in tables:
        return parents
    for event in db.execute('SELECT globalTid,graphNodeId,originalGraphNodeId FROM CUDA_GRAPH_NODE_EVENTS'):
        tid, node, original = event
        if tid is None or ((tid >> 24) & 0xffffff) != pid or original is None:
            continue
        if node in parents and parents[node] != original:
            raise ValueError('ambiguous graph node clone provenance')
        parents[node] = original
    return parents


def captured_node(node, by_id, parents):
    path = []
    while node not in by_id and node in parents:
        if node in path:
            raise ValueError('cyclic graph node clone provenance')
        path.append(node)
        node = parents[node]
    return by_id.get(node), path


def analyze(database, mapping):
    if mapping['schema'] != 'flare-capture-node-map-v1' or mapping['unassigned_node_ids']:
        raise ValueError('require complete captured node attribution')
    if len({node['node_id'] for node in mapping['nodes']}) != len(mapping['nodes']):
        raise ValueError('duplicate mapped node IDs')
    by_id = {node['node_id']: node for node in mapping['nodes']}
    unsupported = {node['node_type'] for node in by_id.values()} - {0, 1, 2, 5, 6, 7}
    if unsupported:
        raise ValueError(f'unsupported captured node types: {sorted(unsupported)}')
    expected = Counter(node_id for node_id, node in by_id.items() if node['node_type'] in (0, 1, 2))
    if not expected:
        raise ValueError('captured graph has no timed device nodes')
    uri = 'file:' + quote(str(Path(database).resolve()), safe='/') + '?mode=ro'
    with sqlite3.connect(uri, uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'CUPTI_ACTIVITY_KIND_KERNEL' not in tables:
            raise ValueError('node-mode kernel table missing; graph envelopes are not substituted')
        strings = dict(db.execute('SELECT id,value FROM StringIds'))
        parents = clone_parents(db, tables, mapping['pid'])
        rows = []
        for table, kind in GPU_TABLES.items():
            if table not in tables:
                continue
            columns = {row[1] for row in db.execute(f'PRAGMA table_info({table})')}
            required = {'start', 'end', 'globalPid', 'deviceId', 'contextId', 'streamId',
                        'correlationId', 'graphNodeId'}
            if not required <= columns:
                raise ValueError(f'{table} missing columns {sorted(required - columns)}')
            for sql in db.execute(f'SELECT * FROM {table}'):
                row = dict(sql)
                if row['globalPid'] is None or ((row['globalPid'] >> 24) & 0xffffff) != mapping['pid']:
                    continue
                node, clone_path = captured_node(row['graphNodeId'], by_id, parents)
                if node is None:
                    if row.get('graphId') in (mapping['graph_id'], mapping['graph_exec_id']):
                        raise ValueError('mapped graph contains an activity with an unqueried node ID')
                    continue
                if node['node_type'] != kind:
                    raise ValueError('activity kind differs from queried node type')
                if row['end'] < row['start'] or row['start'] == row['end'] == 0:
                    raise ValueError('invalid timed node interval')
                activity_graph = mapping['graph_exec_id'] if clone_path else mapping['graph_id']
                if row.get('graphId') not in (None, activity_graph):
                    raise ValueError('queried graph ID differs from node activity graph ID')
                if row.get('graphExecId') not in (None, mapping['graph_exec_id']):
                    raise ValueError('queried executable ID differs from activity executable ID')
                name_id = row.get('demangledName', row.get('shortName'))
                name = strings.get(name_id) if kind == 0 else ('memcpy' if kind == 1 else 'memset')
                if not name:
                    raise ValueError('kernel activity lacks a resolved name')
                rows.append({**row, **node, 'name': name, 'activity_table': table,
                             'clone_path': clone_path})
        launches = defaultdict(list)
        for table in ('CUPTI_ACTIVITY_KIND_DRIVER', 'CUPTI_ACTIVITY_KIND_RUNTIME'):
            if table not in tables:
                continue
            for sql in db.execute(f'SELECT * FROM {table}'):
                row = dict(sql)
                if row['globalTid'] is None or ((row['globalTid'] >> 24) & 0xffffff) != mapping['pid']:
                    continue
                name = strings.get(row['nameId'], '')
                if LAUNCH.match(name):
                    launches[(row['globalTid'] >> 24, row['correlationId'])].append({**row, 'name': name, 'table': table})
        diagnostics = []
        if 'DIAGNOSTIC_EVENT' in tables:
            diagnostics = [dict(row) for row in db.execute('SELECT * FROM DIAGNOSTIC_EVENT')]
    groups = defaultdict(list)
    for row in rows:
        key = (row['globalPid'] >> 24, row['deviceId'], row['contextId'], row['correlationId'])
        groups[key].append(row)
    if not groups:
        raise ValueError('no activities match queried graph-node IDs')
    executions, checks = [], []
    for key, activities in sorted(groups.items(), key=lambda item: min(row['start'] for row in item[1])):
        actual = Counter(row['node_id'] for row in activities)
        host = launches[(key[0], key[3])]
        checks.append({'name': f'execution-{len(executions)}.node_coverage', 'passed': actual == expected,
                       'missing': list((expected - actual).elements()), 'extra': list((actual - expected).elements())})
        checks.append({'name': f'execution-{len(executions)}.host_launch', 'passed': len(host) == 1,
                       'candidates': host})
        activities.sort(key=lambda row: (row['start'], row['end'], row['node_id']))
        kernels = [row for row in activities if row['node_type'] == 0]
        start, end = min(row['start'] for row in activities), max(row['end'] for row in activities)
        kernel_union = Intervals((row['start'], row['end']) for row in kernels).covered(start, end)
        all_union = Intervals((row['start'], row['end']) for row in activities).covered(start, end)
        by_operator = defaultdict(list)
        for row in kernels:
            by_operator[(row['layer'], row['operator'])].append(row)
        operators = [{'layer': key[0], 'operator': key[1], 'kernel_count': len(values),
                      'kernel_sum_ns': sum(row['end'] - row['start'] for row in values),
                      'kernel_names': dict(Counter(row['name'] for row in values))}
                     for key, values in by_operator.items()]
        executions.append({'process_key': key[0], 'device_id': key[1], 'context_id': key[2],
                           'launch_correlation_id': key[3], 'start': start, 'end': end,
                           'node_span_ns': end - start, 'kernel_count': len(kernels),
                           'kernel_sum_ns': sum(row['end'] - row['start'] for row in kernels),
                           'kernel_union_ns': kernel_union, 'device_node_union_ns': all_union,
                           'span_without_observed_nodes_ns': end - start - all_union,
                           'operators': operators, 'sequence': activities})
    if 'expected_replays' in mapping:
        checks.append({'name': 'replay_count', 'passed': len(executions) == mapping['expected_replays'],
                       'expected': mapping['expected_replays'], 'observed': len(executions)})
    return {'schema': 'flare-nsys-node-timeline-v1', 'checks': checks,
            'passed': all(check['passed'] for check in checks), 'executions': executions,
            'summaries': {key: distribution(item[key] for item in executions) for key in
                          ('node_span_ns', 'kernel_sum_ns', 'kernel_union_ns', 'span_without_observed_nodes_ns')},
            'trace_diagnostics': diagnostics,
            'scope': 'Node span runs from first observed device node to last; it is not a whole-graph envelope. '
                     'Uncovered intervals have no observed mapped device node; no host-delay cause is inferred. '
                     'Kernel sums may double-count overlap; node tracing overhead requires a separate paired measurement.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    result = analyze(args.database, json.loads(args.map.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
