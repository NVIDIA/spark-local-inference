/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Build: cc -O2 -Wall -Wextra -I/opt/cuda/include qualify/pcie_peer_probe.c
 *        -L/opt/cuda/lib64 -Wl,-rpath,/opt/cuda/lib64 -lcuda -lnvrtc -o /tmp/pcie_peer_probe
 * Run each case in a fresh process: /tmp/pcie_peer_probe legacy|vmm directed|full [gpu_count]
 * Directed pairs use one mapping for both a remote read and a remote write.
 */
#include <cuda.h>
#include <nvrtc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_GPUS 32
#define WORDS (1 + 2 * MAX_GPUS)

static void cuda_check(CUresult result, const char *call) {
    if (result == CUDA_SUCCESS) return;
    const char *name = "unknown", *message = "unknown";
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &message);
    fprintf(stderr, "%s: %s (%d): %s\n", call, name, result, message);
    exit(1);
}
#define CUDA(call) cuda_check((call), #call)

static void rtc_check(nvrtcResult result, const char *call) {
    if (result == NVRTC_SUCCESS) return;
    fprintf(stderr, "%s: %s\n", call, nvrtcGetErrorString(result));
    exit(1);
}
#define RTC(call) rtc_check((call), #call)

static int owns_pair(int source, int peer, int count) {
    int distance = (peer - source + count) % count;
    return distance > 0 && (2 * distance < count ||
                            (2 * distance == count && source < peer));
}

static unsigned value(int rank) { return 0x12340000u + (unsigned)rank; }

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4 ||
        (strcmp(argv[1], "legacy") && strcmp(argv[1], "vmm")) ||
        (strcmp(argv[2], "directed") && strcmp(argv[2], "full"))) {
        fprintf(stderr, "usage: %s legacy|vmm directed|full [gpu_count]\n", argv[0]);
        return 2;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    int vmm = !strcmp(argv[1], "vmm"), directed = !strcmp(argv[2], "directed");
    int count = argc == 4 ? atoi(argv[3]) : 16, available, driver;
    CUDA(cuInit(0));
    CUDA(cuDeviceGetCount(&available));
    CUDA(cuDriverGetVersion(&driver));
    if (count < 2 || count > MAX_GPUS || count > available) {
        fprintf(stderr, "GPU count must be 2..%d and no greater than %d available devices\n",
                MAX_GPUS, available);
        return 2;
    }
    printf("mode=%s topology=%s GPUs=%d CUDA_driver=%d\n", argv[1], argv[2], count, driver);
    CUdevice devices[MAX_GPUS];
    CUcontext contexts[MAX_GPUS];
    CUdeviceptr memory[MAX_GPUS];
    CUmemGenericAllocationHandle physical[MAX_GPUS];
    size_t bytes[MAX_GPUS];
    CUmodule modules[MAX_GPUS];
    CUfunction kernels[MAX_GPUS];
    unsigned char mapped[MAX_GPUS][MAX_GPUS] = {{0}};
    int outgoing[MAX_GPUS] = {0}, incoming[MAX_GPUS] = {0};
    int major, minor;
    for (int rank = 0; rank < count; ++rank) {
        CUDA(cuDeviceGet(&devices[rank], rank));
        CUDA(cuDevicePrimaryCtxRetain(&contexts[rank], devices[rank]));
        CUDA(cuCtxSetCurrent(contexts[rank]));
        if (vmm) {
            CUmemAllocationProp properties = {0};
            properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            properties.location.id = devices[rank];
            CUDA(cuMemGetAllocationGranularity(&bytes[rank], &properties,
                                               CU_MEM_ALLOC_GRANULARITY_MINIMUM));
            CUDA(cuMemCreate(&physical[rank], bytes[rank], &properties, 0));
            CUDA(cuMemAddressReserve(&memory[rank], bytes[rank], 0, 0, 0));
            CUDA(cuMemMap(memory[rank], bytes[rank], 0, physical[rank], 0));
            CUmemAccessDesc access = {0};
            access.location = properties.location;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            CUDA(cuMemSetAccess(memory[rank], bytes[rank], &access, 1));
        } else {
            bytes[rank] = WORDS * sizeof(unsigned);
            CUDA(cuMemAlloc(&memory[rank], bytes[rank]));
        }
        unsigned initial[WORDS] = {0};
        initial[0] = value(rank);
        CUDA(cuMemcpyHtoD(memory[rank], initial, sizeof(initial)));
    }

    int successes = 0, failures = 0, expected_edges = directed ? count * (count - 1) / 2 : count * (count - 1);
    /* Round-robin ordering spreads mappings across all initiators. */
    for (int distance = 1; distance < count; ++distance) {
        for (int source = 0; source < count; ++source) {
            int peer = (source + distance) % count, supported;
            if (directed && !owns_pair(source, peer, count)) continue;
            CUDA(cuDeviceCanAccessPeer(&supported, devices[source], devices[peer]));
            CUDA(cuCtxSetCurrent(contexts[vmm ? peer : source]));
            CUresult result;
            if (!supported) {
                result = CUDA_ERROR_PEER_ACCESS_UNSUPPORTED;
            } else if (vmm) {
                CUmemAccessDesc access = {0};
                access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = devices[source];
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                result = cuMemSetAccess(memory[peer], bytes[peer], &access, 1);
            } else {
                result = cuCtxEnablePeerAccess(contexts[peer], 0);
            }
            if (result != CUDA_SUCCESS) {
                const char *name = "unknown";
                cuGetErrorName(result, &name);
                printf("mapping FAILED %d -> %d: %s (%d); existing source out=%d in=%d, peer out=%d in=%d\n",
                       source, peer, name, result, outgoing[source], incoming[source], outgoing[peer], incoming[peer]);
                ++failures;
            } else {
                mapped[source][peer] = 1;
                ++outgoing[source]; ++incoming[peer]; ++successes;
            }
        }
    }
    printf("mappings: %d/%d succeeded, %d failed\n", successes, expected_edges, failures);
    for (int rank = 0; rank < count; ++rank)
        printf("GPU %2d: outgoing=%2d incoming=%2d\n", rank, outgoing[rank], incoming[rank]);

    const char *source =
        "extern \"C\" __global__ void exchange(unsigned *local, unsigned *remote, int rank, int peer) {"
        " local[1 + peer] = remote[0];"
        " remote[1 + 32 + rank] = local[0];"
        "}";
    nvrtcProgram program;
    RTC(nvrtcCreateProgram(&program, source, "exchange.cu", 0, NULL, NULL));
    CUDA(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, devices[0]));
    CUDA(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, devices[0]));
    char architecture[64];
    snprintf(architecture, sizeof(architecture), "--gpu-architecture=compute_%d%d", major, minor);
    const char *options[] = {architecture};
    nvrtcResult compile = nvrtcCompileProgram(program, 1, options);
    if (compile != NVRTC_SUCCESS) {
        size_t length;
        RTC(nvrtcGetProgramLogSize(program, &length));
        char *log = malloc(length);
        RTC(nvrtcGetProgramLog(program, log));
        fputs(log, stderr);
        free(log);
        rtc_check(compile, "nvrtcCompileProgram");
    }
    size_t ptx_size;
    RTC(nvrtcGetPTXSize(program, &ptx_size));
    char *ptx = malloc(ptx_size);
    if (!ptx) return 1;
    RTC(nvrtcGetPTX(program, ptx));
    RTC(nvrtcDestroyProgram(&program));
    for (int rank = 0; rank < count; ++rank) {
        CUDA(cuCtxSetCurrent(contexts[rank]));
        CUDA(cuModuleLoadData(&modules[rank], ptx));
        CUDA(cuModuleGetFunction(&kernels[rank], modules[rank], "exchange"));
        for (int peer = 0; peer < count; ++peer) {
            if (!mapped[rank][peer]) continue;
            void *arguments[] = {&memory[rank], &memory[peer], &rank, &peer};
            CUDA(cuLaunchKernel(kernels[rank], 1, 1, 1, 1, 1, 1, 0, NULL, arguments, NULL));
        }
    }
    for (int rank = 0; rank < count; ++rank) {
        CUDA(cuCtxSetCurrent(contexts[rank]));
        CUDA(cuCtxSynchronize());
    }
    int bad = 0;
    for (int rank = 0; rank < count; ++rank) {
        unsigned actual[WORDS];
        CUDA(cuCtxSetCurrent(contexts[rank]));
        CUDA(cuMemcpyDtoH(actual, memory[rank], sizeof(actual)));
        for (int peer = 0; peer < count; ++peer) {
            if (mapped[rank][peer] && actual[1 + peer] != value(peer)) {
                printf("READ mismatch: GPU %d reading GPU %d\n", rank, peer); ++bad;
            }
            if (mapped[peer][rank] && actual[1 + MAX_GPUS + peer] != value(peer)) {
                printf("WRITE mismatch: GPU %d writing GPU %d\n", peer, rank); ++bad;
            }
        }
    }
    printf("GPU read/write verification: %s (%d directed edges, %d errors)\n",
           bad ? "FAIL" : "PASS", successes, bad);
    free(ptx);
    for (int rank = 0; rank < count; ++rank) {
        CUDA(cuCtxSetCurrent(contexts[rank]));
        CUDA(cuModuleUnload(modules[rank]));
        if (!vmm)
            for (int peer = 0; peer < count; ++peer)
                if (mapped[rank][peer]) CUDA(cuCtxDisablePeerAccess(contexts[peer]));
    }
    for (int rank = 0; rank < count; ++rank) {
        CUDA(cuCtxSetCurrent(contexts[rank]));
        if (vmm) {
            CUDA(cuMemUnmap(memory[rank], bytes[rank]));
            CUDA(cuMemAddressFree(memory[rank], bytes[rank]));
            CUDA(cuMemRelease(physical[rank]));
        } else {
            CUDA(cuMemFree(memory[rank]));
        }
    }
    for (int rank = 0; rank < count; ++rank) CUDA(cuDevicePrimaryCtxRelease(devices[rank]));
    return failures || bad ? 1 : 0;
}
