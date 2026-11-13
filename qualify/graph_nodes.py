# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only CUDA graph-node attribution during explicitly scoped capture."""
from contextlib import contextmanager
import ctypes as C
import os
from pathlib import Path

from qualify.performance_trace import loaded_cupti_path


class CaptureQuery:
    def __init__(self):
        path = loaded_cupti_path(Path('/proc/self/maps').read_text())
        self.cupti_path = str(path)
        self.driver = C.CDLL('libcuda.so.1', mode=os.RTLD_LOCAL)
        self.cupti = C.CDLL(str(path), mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
        self._info = self._bind(self.driver, 'cuStreamGetCaptureInfo_v2',
                               [C.c_void_p, C.POINTER(C.c_int), C.POINTER(C.c_uint64),
                                C.POINTER(C.c_void_p), C.c_void_p, C.c_void_p])
        self._nodes = self._bind(self.driver, 'cuGraphGetNodes',
                                [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)])
        self._driver_ids = {name: self._bind(self.driver, name, [C.c_void_p, C.POINTER(dtype)])
                            for name, dtype in (('cuGraphNodeGetLocalId', C.c_uint32),
                                                ('cuGraphNodeGetToolsId', C.c_uint64))
                            if hasattr(self.driver, name)}
        self._type = self._bind(self.driver, 'cuGraphNodeGetType', [C.c_void_p, C.POINTER(C.c_int)])
        self._node_id = self._bind(self.cupti, 'cuptiGetGraphNodeId', [C.c_void_p, C.POINTER(C.c_uint64)])
        self._graph_id = self._bind(self.cupti, 'cuptiGetGraphId', [C.c_void_p, C.POINTER(C.c_uint32)])
        self._exec_id = self._bind(self.cupti, 'cuptiGetGraphExecId', [C.c_void_p, C.POINTER(C.c_uint32)])

    @staticmethod
    def _bind(library, name, arguments):
        fn = getattr(library, name)
        fn.argtypes, fn.restype = arguments, C.c_int
        return fn

    @staticmethod
    def _check(function, *args):
        status = function(*args)
        if status:
            raise RuntimeError(f'{function.__name__} returned status {status}')

    def capture(self, stream):
        status, capture_id, graph = C.c_int(), C.c_uint64(), C.c_void_p()
        self._check(self._info, C.c_void_p(stream), C.byref(status), C.byref(capture_id),
                    C.byref(graph), None, None)
        if status.value != 1 or not graph.value:
            raise RuntimeError(f'expected active capture; status={status.value}')
        return graph.value, capture_id.value

    def nodes(self, graph):
        count = C.c_size_t()
        self._check(self._nodes, C.c_void_p(graph), None, C.byref(count))
        if not count.value:
            return {}
        storage = (C.c_void_p * count.value)()
        self._check(self._nodes, C.c_void_p(graph), storage, C.byref(count))
        result = {}
        for handle in storage[:count.value]:
            node_id, kind = C.c_uint64(), C.c_int()
            self._check(self._node_id, C.c_void_p(handle), C.byref(node_id))
            self._check(self._type, C.c_void_p(handle), C.byref(kind))
            if node_id.value in result:
                raise RuntimeError('CUPTI returned duplicate graph node IDs')
            extra = {}
            for name, dtype, key in (('cuGraphNodeGetLocalId', C.c_uint32, 'driver_local_id'),
                                     ('cuGraphNodeGetToolsId', C.c_uint64, 'driver_tools_id')):
                if name in self._driver_ids:
                    value = dtype()
                    self._check(self._driver_ids[name], C.c_void_p(handle), C.byref(value))
                    extra[key] = value.value
            result[node_id.value] = {'node_id': node_id.value, 'node_type': kind.value, **extra}
        return result

    def graph_id(self, graph):
        value = C.c_uint32()
        self._check(self._graph_id, C.c_void_p(graph), C.byref(value))
        return value.value

    def exec_id(self, handle):
        value = C.c_uint32()
        self._check(self._exec_id, C.c_void_p(handle), C.byref(value))
        return value.value


class NodeScopes:
    def __init__(self, query, stream):
        self.query, self.stream = query, stream
        self.graph, self.capture_id = query.capture(stream)
        self.graph_id = query.graph_id(self.graph)
        self.initial = query.nodes(self.graph)
        self.assigned, self.calls = {}, []
        self.stack = []

    def _nodes(self):
        if self.query.capture(self.stream) != (self.graph, self.capture_id):
            raise RuntimeError('scope crossed a capture boundary')
        return self.query.nodes(self.graph)

    @contextmanager
    def scope(self, operator, *, layer=None):
        before = self._nodes()
        call = {'operator': operator, 'layer': layer, 'call_index': len(self.calls),
                'parent': self.stack[-1] if self.stack else None}
        self.calls.append(call)
        self.stack.append(call['call_index'])
        try:
            yield
        finally:
            self.stack.pop()
            after = self._nodes()
            if not before.keys() <= after.keys():
                raise RuntimeError('capture removed existing graph nodes')
            created = after.keys() - before.keys()
            call['inclusive_node_ids'] = sorted(created)
            for node in created:
                self.assigned.setdefault(node, {**after[node], **{key: call[key] for key in
                                           ('operator', 'layer', 'call_index', 'parent')}})

    def finish_capture(self):
        if self.stack:
            raise RuntimeError('operator scopes are still active')
        nodes = self._nodes()
        unassigned = nodes.keys() - self.assigned.keys()
        return {'schema': 'flare-capture-node-map-v1', 'pid': os.getpid(),
                'graph_id': self.graph_id, 'capture_id': self.capture_id,
                'nodes': [self.assigned[key] for key in sorted(self.assigned)],
                'calls': self.calls, 'unassigned_node_ids': sorted(unassigned),
                'initial_node_ids': sorted(self.initial), 'node_count': len(nodes)}
