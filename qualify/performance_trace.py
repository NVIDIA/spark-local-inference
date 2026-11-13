# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analyze exported Nsight graph timelines without inferring graph identities.

The SQLite file is opened read-only. Times remain in the trace's nanosecond
clock domain. Graph intervals describe complete graph envelopes, not summed
kernel time; gaps exclude other observed GPU work from the same process/device.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import ctypes
import json
import os
from pathlib import Path
import re
import sqlite3
import statistics
from urllib.parse import quote

from .provenance import utc, write_json

GRAPH = 'CUPTI_ACTIVITY_KIND_GRAPH_TRACE'
GPU_TABLES = (GRAPH, 'CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_MEMCPY',
              'CUPTI_ACTIVITY_KIND_MEMSET', 'CUPTI_ACTIVITY_KIND_MEM_DECOMPRESS')
API_TABLES = ('CUPTI_ACTIVITY_KIND_DRIVER', 'CUPTI_ACTIVITY_KIND_RUNTIME')
LAUNCH = re.compile(r'^cu(?:da)?GraphLaunch(?:$|[_\d])')


def _loaded_cupti_selection(maps_text: str):
    paths, injectors = set(), set()
    for line in maps_text.splitlines():
        columns = line.split(maxsplit=5)
        if len(columns) != 6:
            continue
        path = Path(columns[5])
        if re.fullmatch(r'libcupti\.so(?:\.[0-9]+)*', path.name):
            paths.add(path.resolve(strict=True))
        elif path.name == 'libToolsInjection64.so':
            injectors.add(path.resolve(strict=True))
    if injectors:
        directories = {path.parent for path in injectors}
        selected = {path for path in paths if path.parent in directories}
        if len(directories) != 1 or len(selected) != 1:
            raise RuntimeError('require exactly one CUPTI beside the loaded Nsight injector; '
                               f'injectors={sorted(map(str, injectors))}, CUPTI={sorted(map(str, paths))}')
        return selected.pop(), sorted(paths), sorted(injectors)
    if len(paths) != 1:
        raise RuntimeError(f'require exactly one already-loaded CUPTI library, found {sorted(map(str, paths))}')
    return next(iter(paths)), sorted(paths), []


def loaded_cupti_path(maps_text: str) -> Path:
    return _loaded_cupti_selection(maps_text)[0]


def prepared_graph_map(manifest: dict) -> dict:
    """Query CUPTI IDs before handoff, using only its already-loaded library.

    Call explicitly inside an Nsight-launched process. No library is loaded if
    CUPTI is absent. CUDA graph handles are pointer values; CUPTI executable IDs
    are separate uint32 identifiers and must never be equated with those values.
    """
    path, candidates, injectors = _loaded_cupti_selection(Path('/proc/self/maps').read_text())
    library = ctypes.CDLL(str(path), mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
    query = library.cuptiGetGraphExecId
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    query.restype = ctypes.c_int
    rows, names, ids = [], set(), set()
    for graph in manifest['graphs']:
        name, handle = graph['name'], graph['handle']
        if not isinstance(name, str) or not name or name in names or type(handle) is not int or handle <= 0:
            raise ValueError('manifest requires unique graph names and positive executable handles')
        value = ctypes.c_uint32()
        status = query(ctypes.c_void_p(handle), ctypes.byref(value))
        if status != 0:
            raise RuntimeError(f'cuptiGetGraphExecId({name}) returned CUPTI status {status}')
        if value.value in ids:
            raise ValueError('distinct named graphs resolve to the same CUPTI executable ID')
        names.add(name)
        ids.add(value.value)
        rows.append({key: graph[key] for key in
                     ('name', 'kind', 'batch_capacity', 'token_capacity', 'handle')})
        rows[-1]['graph_exec_id'] = value.value
    if not rows:
        raise ValueError('graph catalog is empty')
    return {'schema': 'flare-nsys-graph-map-v1', 'pid': os.getpid(), 'created_at': utc(),
            'cupti_library': str(path),
            'loaded_cupti_libraries': [str(candidate) for candidate in candidates],
            'nsight_injectors': [str(injector) for injector in injectors],
            'cupti_selection': 'loaded_nsight_injector_directory' if injectors else 'sole_loaded_library',
            'graphs': rows}


class Intervals:
    """Union intervals with logarithmic clipped-duration queries."""

    def __init__(self, intervals):
        merged = []
        for start, end in sorted(intervals):
            if end < start:
                raise ValueError('negative activity duration')
            if end == start:
                continue
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        self.starts = [row[0] for row in merged]
        self.ends = [row[1] for row in merged]
        self.prefix = [0]
        for start, end in merged:
            self.prefix.append(self.prefix[-1] + end - start)

    def covered(self, start, end):
        if end <= start:
            return 0
        first = bisect_right(self.ends, start)
        stop = bisect_left(self.starts, end)
        if stop <= first:
            return 0
        return (self.prefix[stop] - self.prefix[first]
                - max(0, start - self.starts[first]) - max(0, self.ends[stop - 1] - end))


def distribution(values):
    values = sorted(values)
    if not values:
        return {'count': 0}
    return {'count': len(values), 'minimum': values[0], 'maximum': values[-1],
            'median': statistics.median(values),
            'p90': values[max(0, (9 * len(values) + 9) // 10 - 1)],
            'p99': values[max(0, (99 * len(values) + 99) // 100 - 1)], 'sum': sum(values)}


def _identifier(value):
    return '"' + value.replace('"', '""') + '"'


def _process(value):
    return int(value) >> 24


def _pid(value):
    return _process(value) & 0xffffff


def _map_labels(graph_map):
    if graph_map is None:
        return {}
    if graph_map.get('schema') != 'flare-nsys-graph-map-v1' or type(graph_map.get('pid')) is not int:
        raise ValueError('graph map schema or PID is invalid')
    labels, names = {}, set()
    for graph in graph_map['graphs']:
        key = (graph_map['pid'], graph['graph_exec_id'])
        if type(key[1]) is not int or not 0 <= key[1] < 2**32:
            raise ValueError('CUPTI executable IDs must be uint32 values')
        if key in labels or graph['name'] in names:
            raise ValueError('graph map has duplicate names or executable IDs')
        labels[key] = graph
        names.add(graph['name'])
    return labels


def analyze(database: Path, *, graph_map=None, pid=None, device=None, start_ns=None, end_ns=None):
    if start_ns is not None and end_ns is not None and start_ns >= end_ns:
        raise ValueError('trace window must have positive duration')
    labels = _map_labels(graph_map)
    connection = sqlite3.connect(f'file:{quote(str(database.resolve()), safe="/")}?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute('PRAGMA query_only=ON')
        schema = {row['name']: [dict(column) for column in connection.execute(
            f'PRAGMA table_info({_identifier(row["name"])})')]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}

        def rows(table, required=(), where='', params=()):
            if table not in schema:
                return []
            missing = set(required) - {col['name'] for col in schema[table]}
            if missing:
                raise ValueError(f'{table} lacks required columns: {sorted(missing)}')
            query = f'SELECT * FROM {_identifier(table)}'
            if where:
                query += ' WHERE ' + where
            return [dict(row) for row in connection.execute(query, params)]

        def name_filter(ids):
            return ('nameId IN (' + ','.join('?' for _ in ids) + ')', tuple(ids)) if ids else ('0', ())

        if GRAPH not in schema:
            raise ValueError('no whole-graph table: export a CUDA graph-mode capture; node spans are not substituted')
        strings = {row['id']: row['value'] for row in rows('StringIds', ('id', 'value'))}
        copy_names = {row['id']: row['label'] for row in rows('ENUM_CUDA_MEMCPY_OPER', ('id', 'label'))}
        apis = []
        api_index = defaultdict(list)
        launch_filter, launch_params = name_filter([key for key, name in strings.items() if LAUNCH.match(name)])
        for table in API_TABLES:
            for row in rows(table, ('start', 'end', 'nameId', 'globalTid', 'correlationId'), launch_filter, launch_params):
                if row['globalTid'] is None:
                    continue
                row['name'] = strings.get(row['nameId'], f'unknown-name-id:{row["nameId"]}')
                row['table'] = table
                if LAUNCH.match(row['name']):
                    apis.append(row)
                    api_index[(_process(row['globalTid']), row['correlationId'])].append(row)

        activities, graphs = [], []
        for table in GPU_TABLES:
            required = ('start', 'end', 'globalPid', 'deviceId', 'contextId', 'streamId')
            if table == GRAPH:
                required += ('correlationId', 'graphId')
            for row in rows(table, required):
                if row['globalPid'] is None:
                    raise ValueError(f'{table} activity has no process identity')
                if pid is not None and _pid(row['globalPid']) != pid:
                    continue
                if device is not None and row['deviceId'] != device:
                    continue
                if row['end'] < row['start']:
                    raise ValueError(f'{table} activity has negative duration')
                if row['start'] == row['end'] == 0:
                    continue
                if start_ns is not None and row['end'] <= start_ns:
                    continue
                if end_ns is not None and row['start'] >= end_ns:
                    continue
                row['table'] = table
                activities.append(row)
                if table == GRAPH and (start_ns is None or row['start'] >= start_ns) and (end_ns is None or row['end'] <= end_ns):
                    graphs.append(row)
        if not graphs:
            raise ValueError('no complete graph executions in the selected window')

        by_device, copies = defaultdict(list), defaultdict(list)
        transfers = []
        activity_counts = Counter()
        for row in activities:
            key = (_process(row['globalPid']), row['deviceId'])
            by_device[key].append((row['start'], row['end']))
            activity_counts[row['table']] += 1
            if row['table'] == 'CUPTI_ACTIVITY_KIND_MEMCPY':
                direction = copy_names.get(row['copyKind'], f'copyKind:{row["copyKind"]}')
                copies[(*key, direction)].append((row['start'], row['end']))
                transfers.append({**row, 'direction': direction})
        unions = {key: Intervals(value) for key, value in by_device.items()}
        copy_unions = {key: Intervals(value) for key, value in copies.items()}
        records, previous, previous_submit = [], {}, {}
        errors = Counter()
        for row in sorted(graphs, key=lambda row: (row['start'], row['end'])):
            process = _process(row['globalPid'])
            key = (process, row['deviceId'], row['contextId'], row['streamId'])
            record = {name: row.get(name) for name in
                      ('start', 'end', 'deviceId', 'contextId', 'streamId', 'correlationId', 'graphId', 'graphExecId')}
            record.update(pid=_pid(row['globalPid']), process_key=process, graph_duration_ns=row['end'] - row['start'])
            label = labels.get((record['pid'], row.get('graphExecId')))
            record['identity'] = label
            if not label:
                errors['unmapped_graph_executions'] += 1
            candidates = api_index[(process, row['correlationId'])]
            candidates = list({(value['name'], value['start'], value['end'], value['globalTid']): value
                               for value in candidates}.values())
            if len(candidates) == 1:
                api = candidates[0]
                record['host_launch'] = {name: api[name] for name in ('name', 'table', 'start', 'end', 'globalTid')}
                record['host_api_duration_ns'] = api['end'] - api['start']
                record['gpu_start_minus_api_end_ns'] = row['start'] - api['end']
                host_key = (process, api['globalTid'], row['deviceId'], row['contextId'], row['streamId'])
                prior = previous_submit.get(host_key)
                record['host_submission_interval_ns'] = None if prior is None else api['start'] - prior
                previous_submit[host_key] = api['start']
            else:
                errors['missing_host_launch' if not candidates else 'ambiguous_host_launch'] += 1
                record['host_launch_candidates'] = candidates
            prior = previous.get(key)
            if prior is not None:
                begin, end = prior['end'], row['start']
                span = max(0, end - begin)
                record['previous_graph_identity'] = prior['identity']
                record['previous_graph_exec_id'] = prior['graphExecId']
                record['inter_graph_gap_ns'] = end - begin
                if row.get('graphExecId') is not None and row['graphExecId'] == prior['graphExecId']:
                    record['same_graph_gap_ns'] = end - begin
                record['gap_other_observed_gpu_work_ns'] = unions[(process, row['deviceId'])].covered(begin, end)
                record['gap_without_observed_gpu_work_ns'] = span - record['gap_other_observed_gpu_work_ns']
                record['gap_copy_activity_ns'] = {direction: intervals.covered(begin, end)
                    for (proc, dev, direction), intervals in copy_unions.items()
                    if proc == process and dev == row['deviceId']}
            previous[key] = record
            records.append(record)

        grouped = defaultdict(list)
        for record in records:
            grouped[(record['pid'], record['deviceId'], record['contextId'], record['streamId'],
                     record['graphExecId'], record['identity']['name'] if record['identity'] else None)].append(record)
        summaries = []
        for key, values in grouped.items():
            summary = dict(zip(('pid', 'device', 'context', 'stream', 'graph_exec_id', 'name'), key, strict=True))
            summary['executions'] = len(values)
            for metric in ('graph_duration_ns', 'host_api_duration_ns', 'gpu_start_minus_api_end_ns',
                           'host_submission_interval_ns', 'inter_graph_gap_ns', 'same_graph_gap_ns',
                           'gap_without_observed_gpu_work_ns'):
                summary[metric] = distribution([row[metric] for row in values if row.get(metric) is not None])
            summaries.append(summary)
        window = (min(row['start'] for row in graphs), max(row['end'] for row in graphs))
        query_pattern = re.compile(r'^cu(?:da)?(?:Event|Stream)Query(?:$|_)')
        dependency_pattern = re.compile(r'^cu(?:da)?(?:Memcpy|Memset|Event|StreamWait|StreamSynchronize|CtxSynchronize)')
        query_ids = [key for key, name in strings.items() if query_pattern.match(name)]
        dependency_ids = [key for key, name in strings.items()
                          if dependency_pattern.match(name) and key not in query_ids]
        host_dependencies, query_summaries = [], []
        for table in API_TABLES:
            if table not in schema:
                continue
            scope, parameters = 'end >= ? AND start <= ?', [*window]
            if pid is not None:
                scope += ' AND ((globalTid >> 24) & 16777215) = ?'
                parameters.append(pid)
            dependency_filter, dependency_params = name_filter(dependency_ids)
            for row in rows(table, where=f'{scope} AND {dependency_filter}',
                            params=(*parameters, *dependency_params)):
                host_dependencies.append({**row, 'name': strings.get(row['nameId'], ''), 'table': table})
            query_filter, query_params = name_filter(query_ids)
            columns = {column['name'] for column in schema[table]}
            return_value = 'returnValue' if 'returnValue' in columns else 'NULL'
            query = (f'SELECT nameId,globalTid,{return_value} AS returnValue,COUNT(*) AS count,'
                     'SUM(end-start) AS total_duration_ns,MIN(end-start) AS minimum_duration_ns,'
                     'MAX(end-start) AS maximum_duration_ns '
                     f'FROM {_identifier(table)} WHERE {scope} AND {query_filter} '
                     'GROUP BY nameId,globalTid,returnValue')
            for row in connection.execute(query, (*parameters, *query_params)):
                query_summaries.append({**dict(row), 'name': strings.get(row['nameId'], ''), 'table': table})
        synchronization, synchronization_summary = {}, {}
        dependency_correlations = {row['correlationId'] for row in host_dependencies}
        for table in ('CUPTI_ACTIVITY_KIND_SYNCHRONIZATION', 'CUPTI_ACTIVITY_KIND_CUDA_EVENT'):
            if table not in schema:
                synchronization[table] = []
                continue
            columns = {column['name'] for column in schema[table]}
            start_column, end_column = ('start', 'end') if 'start' in columns else ('timestamp', 'timestamp')
            scope = f'{end_column} >= ? AND {start_column} <= ?'
            parameters = [*window]
            if pid is not None:
                scope += ' AND ((globalPid >> 24) & 16777215) = ?'
                parameters.append(pid)
            if device is not None:
                scope += ' AND deviceId = ?'
                parameters.append(device)
            if table == 'CUPTI_ACTIVITY_KIND_SYNCHRONIZATION':
                type_column = 'syncType' if 'syncType' in columns else 'deprecatedSyncType'
                query = (f'SELECT globalPid,deviceId,contextId,streamId,{type_column} AS syncType,'
                         'COUNT(*) AS count,SUM(end-start) AS total_duration_ns,'
                         'MIN(end-start) AS minimum_duration_ns,MAX(end-start) AS maximum_duration_ns '
                         f'FROM {_identifier(table)} WHERE {scope} '
                         f'GROUP BY globalPid,deviceId,contextId,streamId,{type_column}')
                synchronization_summary[table] = [dict(row) for row in connection.execute(query, parameters)]
                # Event polls can produce millions of event-sync records, including
                # records with a zero exported returnValue. Retain non-poll dependencies.
                synchronization[table] = [dict(row) for row in connection.execute(
                    f'SELECT * FROM {_identifier(table)} WHERE {scope}', parameters)
                    if row[type_column] != 1 or row['correlationId'] in dependency_correlations]
            else:
                synchronization[table] = rows(table, where=scope, params=parameters)
        return {'schema': 'flare-nsys-performance-v1', 'created_at': utc(), 'database': str(database),
                'graph_map': graph_map, 'selection': {'pid': pid, 'device': device, 'start_ns': start_ns, 'end_ns': end_ns},
                'schema_columns': schema, 'activity_counts': dict(activity_counts), 'graph_executions': len(records),
                'identity_complete': not errors['unmapped_graph_executions'],
                'host_correlation_complete': not (errors['missing_host_launch'] or errors['ambiguous_host_launch']),
                'diagnostics': dict(errors), 'observed_graph_window_ns': list(window),
                'definitions': {
                    'graph_duration_ns': 'GPU whole-graph end minus start; includes internal dependencies and gaps.',
                    'host_submission_interval_ns': 'Host API start interval between consecutive observed launches on the same thread and stream; may span another graph kind or requests.',
                    'gpu_start_minus_api_end_ns': 'Signed GPU start minus host API return; overlap may make this negative.',
                    'gap_without_observed_gpu_work_ns': 'Inter-graph gap minus union of all captured GPU activities for the same process/device; not proof of device-wide idle or CPU causation.',
                    'gap_copy_activity_ns': 'Union transfer activity intersecting gap by exported copy-kind label; overlap establishes timing, not dependency causation.',
                    'identity': 'Exact PID and CUPTI executable-ID map only; raw CUgraphExec handles and durations are never used as IDs.',
                    'scope': 'Graph mode has no node/kernel breakdown inside graphs. Profiling overhead and untraced processes are not measured here.'},
                'summaries': summaries, 'graphs': records,
                'transfers': transfers,
                'synchronization_activities': synchronization,
                'synchronization_activity_summary': synchronization_summary,
                'host_query_api_summary': query_summaries,
                'query_retention': 'All event/stream query calls, including exported returnValue=0, are aggregated by name, thread and recorded return value without individual payloads. Raw event-sync records are retained only when correlated with non-poll dependency APIs; all synchronization records remain in aggregate counts/durations. Exported return values are not interpreted as completion evidence.',
                'trace_diagnostics': rows('DIAGNOSTIC_EVENT'),
                'host_memory_and_sync_apis': host_dependencies}
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--graph-map', type=Path)
    parser.add_argument('--pid', type=int)
    parser.add_argument('--device', type=int)
    parser.add_argument('--start-ns', type=int)
    parser.add_argument('--end-ns', type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    mapping = json.loads(args.graph_map.read_text()) if args.graph_map else None
    result = analyze(args.database, graph_map=mapping, pid=args.pid, device=args.device,
                     start_ns=args.start_ns, end_ns=args.end_ns)
    result['graph_map_file'] = str(args.graph_map) if args.graph_map else None
    write_json(args.output, result)
    print(json.dumps({key: result[key] for key in ('graph_executions', 'identity_complete', 'host_correlation_complete', 'diagnostics')}))


if __name__ == '__main__':
    main()
