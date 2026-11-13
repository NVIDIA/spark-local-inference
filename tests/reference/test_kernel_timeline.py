# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic CPU checks for mapping and interval accounting; no GPU evidence."""
import copy
import sqlite3

import pytest

from qualify.graph_nodes import NodeScopes
from qualify.kernel_timeline import analyze


class Query:
    graph, capture_id = 0xDEAD0000, 9

    def __init__(self):
        self.values = {}

    def capture(self, stream):
        return self.graph, self.capture_id

    def graph_id(self, graph):
        assert graph == self.graph
        return 24

    def nodes(self, graph):
        assert graph == self.graph
        return copy.deepcopy(self.values)

    def add(self, node, kind=0):
        self.values[node] = {'node_id': node, 'node_type': kind}


def test_nested_scopes_assign_innermost_operator_and_keep_inclusive_parent():
    query = Query()
    mapping = NodeScopes(query, 12)
    with mapping.scope('decoder'):
        query.add(10)
        with mapping.scope('qsa.qkv', layer=3):
            query.add(61)
        query.add(90, 2)
    result = mapping.finish_capture()
    assert result['graph_id'] == 24
    assert not result['unassigned_node_ids']
    assert [(n['node_id'], n['operator'], n['layer']) for n in result['nodes']] == [
        (10, 'decoder', None), (61, 'qsa.qkv', 3), (90, 'decoder', None)]
    assert result['calls'][0]['inclusive_node_ids'] == [10, 61, 90]
    assert result['calls'][1]['parent'] == 0


def test_capture_boundary_change_and_node_removal_reject():
    query = Query()
    mapping = NodeScopes(query, 12)
    with pytest.raises(RuntimeError, match='capture boundary'):
        with mapping.scope('qsa'):
            query.capture_id += 1
    query = Query()
    query.add(99)
    mapping = NodeScopes(query, 12)
    with pytest.raises(RuntimeError, match='removed'):
        with mapping.scope('qsa'):
            query.values.clear()


def test_initial_nodes_remain_explicitly_unassigned():
    query = Query()
    query.add(99)
    mapping = NodeScopes(query, 12)
    with mapping.scope('attention'):
        query.add(12)
    result = mapping.finish_capture()
    assert result['initial_node_ids'] == result['unassigned_node_ids'] == [99]


def fixture(path):
    with sqlite3.connect(path) as db:
        db.executescript('''
        CREATE TABLE StringIds(id INTEGER,value TEXT);
        INSERT INTO StringIds VALUES(1,'cuGraphLaunch'),(2,'qkv_kernel'),(3,'pv_kernel');
        CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER(start INTEGER,end INTEGER,nameId INTEGER,
          globalTid INTEGER,correlationId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER,
          deviceId INTEGER,contextId INTEGER,streamId INTEGER,correlationId INTEGER,
          graphNodeId INTEGER,graphId INTEGER,demangledName INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET(start INTEGER,end INTEGER,globalPid INTEGER,
          deviceId INTEGER,contextId INTEGER,streamId INTEGER,correlationId INTEGER,
          graphNodeId INTEGER);
        ''')
        for index, offset in enumerate((0, 1000)):
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES(?,?,?,?,?)',
                       (80+offset,90+offset,1,(77<<24)|4,index+12))
            for start,end,stream,node,name in [(100,130,7,101,2),(120,160,8,401,3)]:
                db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?,?,?,?,?)',
                           (start+offset,end+offset,77<<24,0,1,stream,index+12,node,24,name))
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES(?,?,?,?,?,?,?,?)',
                       (165+offset,170+offset,77<<24,0,1,7,index+12,900))
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?,?,?,?,?)',
                   (100,900,88<<24,0,1,7,12,101,24,2))
    mapping = {'schema':'flare-capture-node-map-v1','pid':77,'graph_id':24,
               'graph_exec_id':97,'unassigned_node_ids':[],'expected_replays':2,
               'nodes':[{'node_id':n,'node_type':t,'layer':3,'operator':op,'call_index':i}
                        for i,(n,t,op) in enumerate([(101,0,'qsa.qkv'),(401,0,'qsa.attend'),(900,2,'metadata')])]}
    return mapping


def test_exact_node_ids_launches_overlap_and_non_kernel_gap_accounting(tmp_path):
    path=tmp_path/'nodes.sqlite'
    mapping=fixture(path)
    before=path.read_bytes()
    result=analyze(path,mapping)
    assert result['passed']
    assert path.read_bytes()==before
    assert len(result['executions'])==2
    first=result['executions'][0]
    assert first['node_span_ns']==70
    assert first['kernel_sum_ns']==70
    assert first['kernel_union_ns']==60
    assert first['device_node_union_ns']==65
    assert first['span_without_observed_nodes_ns']==5
    assert [n['name'] for n in first['sequence']]==['qkv_kernel','pv_kernel','memset']
    assert [n['layer'] for n in first['operators']]==[3,3]


@pytest.mark.parametrize('mutation',['missing','duplicate','ambiguous_launch','replay_count'])
def test_missing_or_duplicate_nodes_and_ambiguous_launch_fail(tmp_path,mutation):
    path=tmp_path/'nodes.sqlite'
    mapping=fixture(path)
    with sqlite3.connect(path) as db:
        if mutation=='missing':
            db.execute('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphNodeId=401 AND correlationId=12')
        elif mutation=='duplicate':
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphNodeId=401 AND correlationId=12')
        elif mutation=='ambiguous_launch':
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER SELECT * FROM CUPTI_ACTIVITY_KIND_DRIVER WHERE correlationId=12')
        else:
            mapping['expected_replays']=3
    result=analyze(path,mapping)
    assert not result['passed']
    assert any(not check['passed'] for check in result['checks'])


def test_graph_envelopes_and_guessed_handle_identity_are_rejected(tmp_path):
    path=tmp_path/'nodes.sqlite'
    mapping=fixture(path)
    wrong=copy.deepcopy(mapping)
    for row in wrong['nodes']:
        row['node_id']+=0xDEAD0000
    with pytest.raises(ValueError,match='unqueried node ID'):
        analyze(path,wrong)
    with sqlite3.connect(path) as db:
        db.execute('DROP TABLE CUPTI_ACTIVITY_KIND_KERNEL')
        db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE(start INTEGER,end INTEGER)')
    with pytest.raises(ValueError,match='graph envelopes are not substituted'):
        analyze(path,mapping)


def test_wrong_graph_id_and_unsupported_child_graph_fail_closed(tmp_path):
    path=tmp_path/'nodes.sqlite'
    mapping=fixture(path)
    mapping['graph_id']=44
    with pytest.raises(ValueError,match='graph ID differs'):
        analyze(path,mapping)
    mapping['nodes'][0]['node_type']=4
    with pytest.raises(ValueError,match='unsupported captured node types'):
        analyze(path,mapping)


def test_empty_and_event_nodes_remain_mapped_without_device_duration(tmp_path):
    path=tmp_path/'nodes.sqlite'
    mapping=fixture(path)
    for kind in (5,6,7):
        mapping['nodes'].append({'node_id':1000+kind,'node_type':kind,'layer':None,
                                 'operator':'synchronization','call_index':kind})
    result=analyze(path,mapping)
    assert result['passed']
    assert result['executions'][0]['node_span_ns']==70
    mapping['nodes'].append({'node_id':1200,'node_type':8})
    with pytest.raises(ValueError,match='unsupported captured node types'):
        analyze(path,mapping)


def clone_fixture(path):
    mapping = fixture(path)
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE CUDA_GRAPH_NODE_EVENTS(globalTid INTEGER,graphNodeId INTEGER,originalGraphNodeId INTEGER)')
        for node in (101, 401, 900):
            db.execute('INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES(?,?,?)', (77<<24, node+10000, node))
            for table in ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_MEMSET'):
                db.execute(f'UPDATE {table} SET graphNodeId=? WHERE graphNodeId=? AND globalPid=?',
                           (node+10000, node, 77<<24))
        db.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET graphId=97 WHERE globalPid=?', (77<<24,))
    return mapping


def test_executable_clones_join_through_recorded_provenance(tmp_path):
    path = tmp_path/'clone.sqlite'
    mapping = clone_fixture(path)
    result = analyze(path, mapping)
    assert result['passed']
    assert [(n['node_id'], n['clone_path']) for n in result['executions'][0]['sequence']] == [
        (101, [10101]), (401, [10401]), (900, [10900])]


@pytest.mark.parametrize('mutation', ['missing', 'foreign', 'ambiguous', 'cycle', 'wrong_graph'])
def test_missing_or_invalid_clone_provenance_rejects(tmp_path, mutation):
    path = tmp_path/'clone.sqlite'
    mapping = clone_fixture(path)
    with sqlite3.connect(path) as db:
        if mutation == 'missing':
            db.execute('DELETE FROM CUDA_GRAPH_NODE_EVENTS WHERE graphNodeId=10101')
        elif mutation == 'foreign':
            db.execute('UPDATE CUDA_GRAPH_NODE_EVENTS SET globalTid=? WHERE graphNodeId=10101', (88<<24,))
        elif mutation == 'ambiguous':
            db.execute('INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES(?,?,?)', (77<<24, 10101, 401))
        elif mutation == 'cycle':
            db.execute('UPDATE CUDA_GRAPH_NODE_EVENTS SET originalGraphNodeId=10101 WHERE graphNodeId=10101')
        else:
            db.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET graphId=24 WHERE graphNodeId=10101')
    with pytest.raises(ValueError):
        analyze(path, mapping)
