# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic CPU records test trace accounting; they are not measured evidence."""

import sqlite3

import pytest

from qualify.performance_trace import Intervals, analyze, loaded_cupti_path


def test_interval_union_clips_overlap_and_does_not_double_count():
    intervals = Intervals([(10, 20), (15, 25), (30, 40), (40, 50), (19, 19)])
    assert intervals.covered(0, 60) == 35
    assert intervals.covered(15, 45) == 25
    assert intervals.covered(25, 30) == 0
    assert intervals.covered(20, 20) == 0
    assert intervals.covered(50, 70) == 0
    assert intervals.covered(0, 15) == 5
    assert intervals.covered(45, 60) == 5
    with pytest.raises(ValueError, match='negative'):
        Intervals([(20, 10)])


def database(path):
    with sqlite3.connect(path) as db:
        db.executescript('''
            CREATE TABLE StringIds(id INTEGER, value TEXT);
            INSERT INTO StringIds VALUES(1,'cuGraphLaunch'),(2,'cuMemcpyHtoDAsync_v2'),(3,'cuStreamWaitEvent');
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,nameId INTEGER,
                globalTid INTEGER,correlationId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE(start INTEGER,end INTEGER,globalPid INTEGER,
                deviceId INTEGER,contextId INTEGER,streamId INTEGER,correlationId INTEGER,
                graphId INTEGER,graphExecId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,globalPid INTEGER,
                deviceId INTEGER,contextId INTEGER,streamId INTEGER,copyKind INTEGER,bytes INTEGER);
            CREATE TABLE ENUM_CUDA_MEMCPY_OPER(id INTEGER,label TEXT);
            INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES(1,'HtoD'),(2,'DtoH');
        ''')
        db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,?)', [
            (80, 110, 1, (111 << 24) | 7, 1),
            (145, 150, 1, (111 << 24) | 7, 2),
            (81, 99, 1, (222 << 24) | 7, 1),
            (138, 149, 2, (111 << 24) | 7, 10),
            (149, 151, 3, (111 << 24) | 7, 11),
        ])
        db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES(?,?,?,?,?,?,?,?,?)', [
            (100, 140, 111 << 24, 0, 1, 9, 1, 5, 42),
            (170, 210, 111 << 24, 0, 1, 9, 2, 5, 42),
            (100, 169, 222 << 24, 0, 1, 9, 1, 5, 42),
        ])
        db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(?,?,?,?,?,?,?,?)', [
            (142, 147, 111 << 24, 0, 1, 8, 2, 256),
            (144, 156, 111 << 24, 0, 1, 9, 1, 80),
        ])
    return path


def mapping():
    return {'schema': 'flare-nsys-graph-map-v1', 'pid': 111,
            'graphs': [{'name': 'decode1', 'kind': 'decode', 'graph_exec_id': 42,
                        'handle': 0x12340000}]}


def test_real_schema_shape_correlates_by_process_and_computes_union_gaps(tmp_path):
    path = database(tmp_path / 'synthetic.sqlite')
    before = path.read_bytes()
    result = analyze(path, graph_map=mapping(), pid=111)
    assert path.read_bytes() == before
    assert result['identity_complete'] and result['host_correlation_complete']
    assert result['graph_executions'] == 2
    first, second = result['graphs']
    assert first['gpu_start_minus_api_end_ns'] == -10
    assert second['host_submission_interval_ns'] == 65
    assert second['inter_graph_gap_ns'] == 30
    assert second['same_graph_gap_ns'] == 30
    assert second['gap_other_observed_gpu_work_ns'] == 14
    assert second['gap_without_observed_gpu_work_ns'] == 16
    assert second['gap_copy_activity_ns'] == {'DtoH': 5, 'HtoD': 12}
    assert len(result['host_memory_and_sync_apis']) == 2
    assert result['summaries'][0]['graph_duration_ns']['sum'] == 80


def test_unknown_graph_ids_are_not_inferred_from_handle_or_duration(tmp_path):
    path = database(tmp_path / 'synthetic.sqlite')
    wrong = mapping()
    wrong['graphs'][0]['graph_exec_id'] = wrong['graphs'][0]['handle']
    result = analyze(path, graph_map=wrong, pid=111)
    assert not result['identity_complete']
    assert all(row['identity'] is None for row in result['graphs'])
    assert result['diagnostics']['unmapped_graph_executions'] == 2


def test_duplicate_host_correlations_are_explicitly_ambiguous(tmp_path):
    path = database(tmp_path / 'synthetic.sqlite')
    with sqlite3.connect(path) as db:
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,?)',
                   (70, 120, 1, (111 << 24) | 8, 1))
    result = analyze(path, graph_map=mapping(), pid=111)
    assert not result['host_correlation_complete']
    assert result['diagnostics']['ambiguous_host_launch'] == 1
    assert len(result['graphs'][0]['host_launch_candidates']) == 2


def test_window_excludes_partial_graphs_and_node_only_capture_is_rejected(tmp_path):
    path = database(tmp_path / 'synthetic.sqlite')
    result = analyze(path, pid=111, start_ns=105)
    assert result['graph_executions'] == 1
    with sqlite3.connect(path) as db:
        db.execute('DROP TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE')
    with pytest.raises(ValueError, match='node spans are not substituted'):
        analyze(path)


def test_cupti_lookup_requires_one_existing_loaded_library(tmp_path):
    library = tmp_path / 'libcupti.so.13.3'
    library.write_bytes(b'CPU path fixture, never loaded')
    line = f'1-2 r-xp 0 00:00 0 {library}\n'
    assert loaded_cupti_path(line + line) == library
    with pytest.raises(RuntimeError, match='exactly one'):
        loaded_cupti_path('')
    other = tmp_path / 'libcupti.so.12.9'
    other.touch()
    with pytest.raises(RuntimeError, match='exactly one'):
        loaded_cupti_path(line + f'3-4 r-xp 0 00:00 1 {other}\n')


def test_cupti_selects_loaded_nsight_library_beside_injector(tmp_path):
    profiler = tmp_path / 'nsight' / 'target-linux-x64'
    torch = tmp_path / 'torch' / 'lib'
    profiler.mkdir(parents=True)
    torch.mkdir(parents=True)
    candidates = [profiler / 'libcupti.so.13.3', torch / 'libcupti.so.13']
    injector = profiler / 'libToolsInjection64.so'
    for path in [*candidates, injector]:
        path.touch()
    maps = ''.join(f'1-2 r-xp 0 00:00 0 {path}\n' for path in [*candidates, injector])
    assert loaded_cupti_path(maps) == candidates[0]
    second = profiler / 'libcupti.so.12.9'
    second.touch()
    with pytest.raises(RuntimeError, match='exactly one CUPTI beside'):
        loaded_cupti_path(maps + f'1-2 r-xp 0 00:00 0 {second}\n')
    candidates[0].unlink()
    maps = ''.join(f'1-2 r-xp 0 00:00 0 {path}\n' for path in [candidates[1], injector])
    with pytest.raises(RuntimeError, match='exactly one CUPTI beside'):
        loaded_cupti_path(maps)


def test_all_poll_records_are_aggregated_and_nonpoll_dependencies_retained(tmp_path):
    path = database(tmp_path / 'polls.sqlite')
    with sqlite3.connect(path) as db:
        db.executescript("""
            ALTER TABLE CUPTI_ACTIVITY_KIND_RUNTIME ADD COLUMN returnValue INTEGER;
            INSERT INTO StringIds VALUES(4,'cuEventQuery'),(5,'cuEventSynchronize');
            CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION(start INTEGER,end INTEGER,
                globalPid INTEGER,deviceId INTEGER,contextId INTEGER,streamId INTEGER,
                correlationId INTEGER,syncType INTEGER);
            CREATE TABLE DIAGNOSTIC_EVENT(message TEXT);
            INSERT INTO DIAGNOSTIC_EVENT VALUES('synthetic diagnostic retained');
        """)
        polls = [(110, 111, 4, (111 << 24) | 7, 100 + index, 600) for index in range(2000)]
        polls += [(141, 142, 4, (111 << 24) | 7, 2100, 0),
                  (142, 143, 5, (111 << 24) | 7, 2101, 0)]
        db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,?,?)', polls)
        db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION VALUES(?,?,?,?,?,?,?,?)',
                       [(row[0], row[1], 111 << 24, 0, 1, 9, row[4], 1) for row in polls])
    result = analyze(path, graph_map=mapping(), pid=111)
    queries = {row['returnValue']: row for row in result['host_query_api_summary']}
    assert queries[600]['count'] == queries[600]['total_duration_ns'] == 2000
    assert queries[0]['count'] == 1
    assert sum(row['count'] for row in result['synchronization_activity_summary']['CUPTI_ACTIVITY_KIND_SYNCHRONIZATION']) == 2002
    assert {row['correlationId'] for row in result['synchronization_activities']['CUPTI_ACTIVITY_KIND_SYNCHRONIZATION']} == {2101}
    assert not any(row['name'] == 'cuEventQuery' for row in result['host_memory_and_sync_apis'])
    assert result['trace_diagnostics'] == [{'message': 'synthetic diagnostic retained'}]
    assert result['graph_executions'] == 2 and result['host_correlation_complete']
