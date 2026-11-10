# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Export a graph with ordinary and programmatic kernel dependencies."""

import ctypes as C
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from types import SimpleNamespace

from ignition.preparation.runtime_core import (
    Allocation, Argument, Kernel, Reference, Relocation, RuntimeCore, _Cuda,
)
from cuda_graph_fixture import KernelParams
from fixture_frontend import server_config
from signing import publisher


def prepare(root: Path) -> None:
    cuda = _Cuda()
    cuda.call("cuInit", C.c_uint(0))
    device, context = C.c_int(), C.c_void_p()
    cuda.call("cuDeviceGet", C.byref(device), C.c_int(0))
    cuda.call("cuCtxCreate_v2", C.byref(context), C.c_uint(0), device)
    major, minor = C.c_int(), C.c_int()
    cuda.call("cuDeviceGetAttribute", C.byref(major), C.c_int(75), device)
    cuda.call("cuDeviceGetAttribute", C.byref(minor), C.c_int(76), device)

    source = root / "programmatic_edges.cu"
    source.write_text('''
#include <cuda_runtime.h>
__device__ __forceinline__ unsigned long long timer() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}
extern "C" __global__ void before(unsigned long long *times, int *value) {
    times[4] = timer();
    while (timer() - times[4] < 100000ULL) {}
    *value = 5;
    times[5] = timer();
}
extern "C" __global__ void producer(unsigned long long *times, int *value) {
    times[0] = timer();
    times[6] = *value;
    cudaTriggerProgrammaticLaunchCompletion();
    while (timer() - times[0] < 300000ULL) {}
    *value = 7;
    times[1] = timer();
}
extern "C" __global__ void consumer(unsigned long long *times, int *value, int *result) {
    times[2] = timer();
    cudaGridDependencySynchronize();
    times[3] = timer();
    *result = *value;
}
''')
    cubin = root / "programmatic_edges.cubin"
    command = ["nvcc", "--cubin", "-std=c++17",
               f"-arch=sm_{major.value}{minor.value}"]
    host_compiler = shutil.which("g++-13") or shutil.which("g++")
    if host_compiler:
        command.extend(["-ccbin", host_compiler])
    subprocess.run([*command, str(source), "-o", str(cubin)], check=True)

    module = C.c_void_p()
    image = C.create_string_buffer(cubin.read_bytes())
    cuda.call("cuModuleLoadData", C.byref(module), image)
    functions = {}
    for name in ("before", "producer", "consumer"):
        function = C.c_void_p()
        cuda.call("cuModuleGetFunction", C.byref(function), module, name.encode())
        functions[name] = function

    state = C.c_uint64()
    cuda.call("cuMemAlloc_v2", C.byref(state), C.c_size_t(64))
    times, value, result = (C.c_uint64(state.value + offset) for offset in (0, 56, 60))
    graph, executable = C.c_void_p(), C.c_void_p()
    cuda.call("cuGraphCreate", C.byref(graph), C.c_uint(0))
    nodes = []
    for name, pointers in (
        ("before", (times, value)),
        ("producer", (times, value)),
        ("consumer", (times, value, result)),
    ):
        arguments = (C.c_void_p * len(pointers))(*(C.addressof(p) for p in pointers))
        params = KernelParams(
            func=functions[name].value,
            grid=(C.c_uint * 3)(1, 1, 1),
            block=(C.c_uint * 3)(1, 1, 1),
            params=arguments if pointers else None,
        )
        node = C.c_void_p()
        cuda.call("cuGraphAddKernelNode_v2", C.byref(node), graph, None,
                  C.c_size_t(0), C.byref(params))
        nodes.append(node)
    for source_node, destination_node, annotation in (
        (nodes[0], nodes[1], (0, 0, 0, 0, 0, 0, 0, 0)),
        (nodes[1], nodes[2], (1, 0, 1, 0, 0, 0, 0, 0)),
    ):
        sources = (C.c_void_p * 1)(source_node.value)
        destinations = (C.c_void_p * 1)(destination_node.value)
        data = (C.c_ubyte * 8)(*annotation)
        cuda.call("cuGraphAddDependencies_v2", graph, sources, destinations,
                  data, C.c_size_t(1))
    cuda.call("cuGraphInstantiateWithFlags", C.byref(executable), graph, C.c_uint64(1))

    stream = C.c_void_p()
    cuda.call("cuStreamCreate", C.byref(stream), C.c_uint(1))
    cuda.call("cuGraphLaunch", executable, stream)
    cuda.call("cuStreamSynchronize", stream)
    captured = (C.c_ubyte * 64)()
    cuda.call("cuMemcpyDtoH_v2", captured, state, C.c_size_t(64))
    times_seen = struct.unpack_from("<7Q", bytes(captured))
    assert times_seen[4] < times_seen[5] < times_seen[0]
    assert times_seen[0] < times_seen[2] < times_seen[1] < times_seen[3]
    assert times_seen[6] == 5
    assert struct.unpack_from("<i", bytes(captured), 60)[0] == 7

    def pointer(offset):
        return Argument(8, (Relocation(0, Reference("state", offset)),))

    core = RuntimeCore(
        allocations=[Allocation("state", state.value, 64, initialize="zero")],
        modules={"programmatic_edges": cubin},
        kernels=[
            Kernel(functions["before"].value, "programmatic_edges",
                   (pointer(0), pointer(56))),
            Kernel(functions["producer"].value, "programmatic_edges",
                   (pointer(0), pointer(56))),
            Kernel(functions["consumer"].value, "programmatic_edges",
                   (pointer(0), pointer(56), pointer(60))),
        ],
    )
    graphs = {"programmatic": SimpleNamespace(
        raw_cuda_graph=lambda: graph.value,
        raw_cuda_graph_exec=lambda: executable.value,
    )}
    core.write(root / "core", graphs, publisher=publisher(),
                  server=server_config(root))
    cuda.call("cuCtxDestroy_v2", context)


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    prepare(directory)
