#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the b12x dense MXFP8 GEMM stage-reuse race (local-inference-lab/flashinfer#16) on SM120/SM121.

The GEMM's producer warp refills a reclaimed shared-memory stage with TMA (async proxy) once the
consumer warps release it. Consumers on the non-a16 ("quantized") path read stages through the
generic proxy, and before #16 only the a16 path issued fence.proxy.async before the refill, so a
refill can overwrite a stage that is still being read. Whole 64x128 output tiles then come out wrong
but finite. On a quiet GPU it practically never happens; it needs SM scheduling to be disturbed.

This harness, with no model or checkpoint:
- builds b12x blockscaled MXFP8 GEMMs with random weights at GLM-5.3 Flash TP4 KDA projection shapes
  (input_proj 4096 -> 6288 and o_proj 2048 -> 4096, 96 rows = 16 requests x MTP width 6), pinned to
  the quantized plan GLM selects at that row count;
- captures them in one CUDA graph, replays it, and compares every replay bit for bit with the first;
- in the "noise" arm, a host thread launches highest-priority "hog" kernels with random CTA counts,
  durations and shared-memory footprints on a second stream, so the GEMM CTAs start and finish
  unevenly (the same perturbation that reproduced the race at about 3% of replays on DGX Spark).

``--variant`` picks the GEMM source: ``unfenced`` (the pre-#16 mainloop), ``fenced`` (#16), or
``installed`` (whatever the installed FlashInfer has). Patched variants compile into their own
kernel cache, because b12x's compile-cache keys do not hash kernel source.

Needs an SM120/SM121 GPU, torch, nvidia-cutlass-dsl, FlashInfer with b12x (Flare's environment
has all three), and nvcc to build the noise kernel. No model, checkpoint or server is involved.

    .venv/bin/python scripts/b12x_gemm_stage_race.py --variant unfenced   # mismatches under noise
    .venv/bin/python scripts/b12x_gemm_stage_race.py --variant fenced     # none

On each of four DGX Sparks (GB10, 48 SMs), 30 s per arm: unfenced, 0 of 100,000 quiet replays and
3.5-3.7% of about 61,000 replays under noise, whole 64x128 tiles wrong in both GEMMs; fenced, 0 of
about 61,000 under noise. On an RTX PRO 4000 Blackwell (SM120, 70 SMs), 30 s per arm: unfenced, 0 of
100,000 quiet and 1,177 of 100,000 (1.2%) under noise; fenced, 0 of 100,000 under noise.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

PATCH_ENV = "B12X_STAGE_RACE_DENSE_GEMM"

FENCED = """                    # Order consumer reads before async-proxy stage reuse. Every mainloop path can
                    # read a stage through the generic proxy before TMA refills it, not only a16.
                    cute.arch.fence_view_async_shared()
"""
UNFENCED = """                    if cutlass.const_expr(self.a16):
                        # Order consumer reads before async-proxy stage reuse.
                        cute.arch.fence_view_async_shared()
"""

NOISE_CU = r"""
#include <cuda_runtime.h>
#include <stdint.h>
#include <unistd.h>

static __device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Each CTA holds either a whole SM's shared memory or a light footprint, spins on the global timer
// for its duration, and optionally strides random loads over a buffer to thrash TLB and L2.
__global__ void hog(unsigned long long ns, const unsigned long long *buf, unsigned long long words,
                    unsigned long long seed, int touch, unsigned long long *sink) {
  extern __shared__ unsigned long long smem[];
  unsigned long long start = gtimer();
  unsigned long long x = seed ^ (blockIdx.x * 0x9E3779B97F4A7C15ull) ^ threadIdx.x;
  unsigned long long acc = 0;
  smem[threadIdx.x] = x;
  while (gtimer() - start < ns) {
    if (touch && words) {
      #pragma unroll 4
      for (int i = 0; i < 4; ++i) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        acc += __ldcg(buf + (x >> 17) % words);
      }
    }
  }
  if (acc == 0x5555555555555555ull) sink[0] = acc + smem[threadIdx.x ^ 1];
}

static unsigned long long rng(unsigned long long *s) {
  *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s;
}

extern "C" long long noise_loop(int device, volatile int *stop, void *buf, unsigned long long bytes,
                                unsigned long long seed, int max_us, int max_gap_us, int touch_percent,
                                volatile unsigned long long *launched) {
  if (cudaSetDevice(device) != cudaSuccess) return -1;
  int least, greatest;
  cudaDeviceGetStreamPriorityRange(&least, &greatest);
  cudaStream_t stream;
  if (cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest) != cudaSuccess) return -1;
  int sms = 0, smem = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  cudaFuncSetAttribute(hog, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  unsigned long long *sink;
  cudaMalloc(&sink, 64);
  unsigned long long s = seed | 1, count = 0, words = bytes / 8;
  while (!*stop) {
    int ctas = 1 + (int)(rng(&s) % (unsigned long long)(2 * sms));
    unsigned long long us = 2 + rng(&s) % (unsigned long long)max_us;
    int threads = (rng(&s) & 1) ? 1024 : 256;
    int shared = (rng(&s) % 3) ? smem : threads * 8;
    int touch = (int)(rng(&s) % 100) < touch_percent;
    hog<<<ctas, threads, shared, stream>>>(us * 1000ull, (const unsigned long long *)buf, words, rng(&s),
                                           touch, sink);
    if (cudaGetLastError() != cudaSuccess) return -1;
    ++count;
    __atomic_store_n(launched, count, __ATOMIC_RELAXED);
    if (count % 8 == 0) cudaStreamSynchronize(stream);
    unsigned long long gap = rng(&s) % (unsigned long long)(max_gap_us + 1);
    if (gap) usleep((useconds_t)gap);
  }
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  cudaFree(sink);
  return (long long)count;
}
"""


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def _install_patch():
    """Load the patched dense_gemm before b12x imports it. Also runs in b12x's spawned compile workers,
    which import this script as their main module; the environment variable only carries the
    patched file's path to them and is set by --variant, not by users."""
    path = os.environ.get(PATCH_ENV)
    if not path or "b12x._lib.dense_gemm" in sys.modules:
        return
    package = importlib.import_module("b12x._lib")
    spec = importlib.util.spec_from_file_location("b12x._lib.dense_gemm", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["b12x._lib.dense_gemm"] = module
    spec.loader.exec_module(module)
    package.dense_gemm = module


_install_patch()


def prepare_variant(variant, work):
    """Write the requested dense_gemm variant and point b12x at a kernel cache of its own."""
    if variant == "installed":
        return "installed"
    spec = importlib.util.find_spec("b12x._lib.dense_gemm")
    source = Path(spec.origin).read_text()
    if source.count(FENCED) + source.count(UNFENCED) != 1:
        raise SystemExit(f"{spec.origin}: mainloop fence block not recognized; cannot build variants")
    installed = "fenced" if FENCED in source else "unfenced"
    text = source if variant == installed else source.replace(
        FENCED if installed == "fenced" else UNFENCED, UNFENCED if variant == "unfenced" else FENCED)
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    path = work / f"dense_gemm_{variant}_{digest}.py"
    path.write_text(text)
    cache = work / f"kernel-cache-{variant}-{digest}"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["B12X_COMPILE_CACHE_DIR"] = str(cache)
    os.environ[PATCH_ENV] = str(path)
    _install_patch()
    return f"{variant} (installed FlashInfer is {installed}; source {path}, kernel cache {cache})"


class Noise:
    def __init__(self, torch, work, device, buffer_gib, seed, nvcc):
        major, minor = torch.cuda.get_device_capability(device)
        lib = work / f"libnoise_sm{major}{minor}.so"
        if not lib.exists():
            src = work / "noise.cu"
            src.write_text(NOISE_CU)
            subprocess.run([nvcc, "-O3", f"-arch=sm_{major}{minor}", "-shared", "-Xcompiler", "-fPIC",
                            str(src), "-o", str(lib)], check=True)
        self.lib = ctypes.CDLL(str(lib))
        self.lib.noise_loop.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_void_p,
                                        ctypes.c_ulonglong, ctypes.c_ulonglong, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int, ctypes.POINTER(ctypes.c_ulonglong))
        self.lib.noise_loop.restype = ctypes.c_longlong
        self.buffer = torch.full((int(buffer_gib * (1 << 30)),), 0x3C, dtype=torch.uint8, device=device)
        self.device, self.seed = device.index or 0, seed
        self.stop, self.launched = ctypes.c_int(0), ctypes.c_ulonglong(0)

    def __enter__(self):
        self.stop.value = 0
        self.result = None

        def run():
            self.result = self.lib.noise_loop(self.device, ctypes.byref(self.stop),
                                              ctypes.c_void_p(self.buffer.data_ptr()), self.buffer.numel(),
                                              self.seed, 400, 300, 50, ctypes.byref(self.launched))

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.value = 1
        self.thread.join()
        if self.result is not None and self.result < 0:
            raise RuntimeError("noise kernel launch failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=("unfenced", "fenced", "installed"), default="unfenced")
    ap.add_argument("--arms", default="baseline,noise", help="comma list of baseline, noise")
    ap.add_argument("--seconds", type=float, default=60, help="per arm")
    ap.add_argument("--max-replays", type=int, default=100_000, help="per arm")
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--shapes", default="6288x4096,4096x2048", help="comma list of OUTxIN")
    ap.add_argument("--noise-gib", type=float, default=2.0, help="buffer the noise kernels read at random")
    ap.add_argument("--work-dir", default=str(Path(tempfile.gettempdir()) / "b12x-stage-race"))
    ap.add_argument("--nvcc", default=shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc")
    ap.add_argument("--json", default=None, help="write a summary here")
    args = ap.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    described = prepare_variant(args.variant, work)

    import torch
    from b12x.gemm import blockscaled
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.gemm.blockscaled._a16 import _weight_parts

    device = torch.device("cuda", torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability(device)
    if major != 12 or minor not in (0, 1):
        raise SystemExit(f"needs SM120/SM121, found sm_{major}{minor}")
    props = torch.cuda.get_device_properties(device)
    log(f"{props.name} sm_{major}{minor}, {props.multi_processor_count} SMs; torch {torch.__version__}")
    log(f"dense_gemm variant: {described}")
    torch.set_grad_enabled(False)

    gen = torch.Generator(device="cpu").manual_seed(5)
    gemms = []
    for shape in args.shapes.split(","):
        n, k = (int(v) for v in shape.lower().split("x"))
        weight = (torch.randn((n, k), generator=gen) * 64).clamp(-448, 448).to(device).to(torch.float8_e4m3fn)
        scales = torch.randint(122, 127, (n, k // 32), generator=gen, dtype=torch.int32).to(torch.uint8).to(device)
        packed = blockscaled.pack_weight(weight, scales)
        source = (torch.randn((args.rows, k), generator=gen) * 0.5).to(torch.bfloat16).to(device)
        out = torch.zeros((args.rows, n), dtype=torch.bfloat16, device=device)
        gemms.append(dict(name=f"{n}x{k}", source=source, packed=packed, out=out))

    session = PreparationSession(device=device, autotune=False, compile_workers=2)
    session.__enter__()
    requests = []
    for g in gemms:
        query = blockscaled.query_from_call(g["source"], g["packed"], activation_mode="quantized", out=g["out"])
        g["plan"] = blockscaled.plan(query)
        values, scales, gain, _ = _weight_parts(g["packed"])

        def call(state, g=g, values=values, scales=scales, gain=gain):
            return PreparedCall(run=lambda: state.run(g["source"], values, scales, gain, out=g["out"]))

        requests.append(g["plan"].request(name=g["name"], prepare_call=call))
    t0 = time.time()
    session.prepare(tuple(requests))
    session.freeze()
    log(f"prepared {len(gemms)} GEMMs in {time.time() - t0:.0f}s")
    for g in gemms:
        try:
            state = g["plan"].prepared.state
            state = getattr(state, "capacity", state)
            low = getattr(getattr(state, "dense", None), "lowering", None)
            log(f"  {g['name']}: {state.config}; tile {getattr(low, 'mma_tiler_mn', None)} "
                f"load {getattr(low, 'load_path', None)}")
        except Exception as error:  # noqa: BLE001 - the plan description is informational
            log(f"  {g['name']}: plan description unavailable ({type(error).__name__})")

    stream = torch.cuda.Stream(device)

    def forward():
        for g in gemms:
            g["out"].zero_()
            blockscaled.mm(g["source"], g["packed"], g["out"], plan=g["plan"])

    with torch.cuda.stream(stream):
        forward()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        forward()
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    refs = [g["out"].clone() for g in gemms]
    for g, ref in zip(gemms, refs):
        if not torch.isfinite(ref).all():
            raise SystemExit(f"{g['name']}: reference output is not finite")

    counts = torch.zeros(len(gemms), dtype=torch.int64, device=device)
    host = torch.zeros(len(gemms), dtype=torch.int64, pin_memory=True)

    def compare_ops():
        torch.stack([(g["out"].view(torch.int16) != r.view(torch.int16)).sum() for g, r in zip(gemms, refs)],
                    out=counts)

    with torch.cuda.stream(stream):
        compare_ops()
    stream.synchronize()
    compare = torch.cuda.CUDAGraph()
    with torch.cuda.graph(compare, stream=stream):
        compare_ops()
        host.copy_(counts, non_blocking=True)

    def run_arm(arm):
        bad = replays = 0
        elements = [0] * len(gemms)
        first = []
        deadline = time.time() + args.seconds
        last_report = time.time()
        while time.time() < deadline and replays < args.max_replays:
            with torch.cuda.stream(stream):
                graph.replay()
                compare.replay()
            stream.synchronize()
            replays += 1
            row = host.tolist()
            if any(row):
                bad += 1
                elements = [a + b for a, b in zip(elements, row)]
                if len(first) < 5:
                    first.append({"replay": replays, "mismatched_elements": dict(zip([g["name"] for g in gemms], row))})
                    tiles = {}
                    for g, r in zip(gemms, refs):
                        diff = (g["out"].view(torch.int16) != r.view(torch.int16)).nonzero()
                        if diff.numel():
                            t = sorted({(int(a) // 64, int(b) // 128) for a, b in diff[:4096].tolist()})
                            tiles[g["name"]] = t[:8]
                    log(f"[{arm}] MISMATCH replay {replays}: {first[-1]['mismatched_elements']} 64x128 tiles {tiles}")
            if time.time() - last_report > 15:
                log(f"[{arm}] {replays} replays, {bad} bad")
                last_report = time.time()
        return {"replays": replays, "bad": bad, "mismatched_elements": dict(zip([g["name"] for g in gemms], elements)),
                "first": first}

    results = {}
    noise = None
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if arm == "baseline":
            results[arm] = run_arm(arm)
        elif arm == "noise":
            if noise is None:
                noise = Noise(torch, work, device, args.noise_gib, seed=12345, nvcc=args.nvcc)
            with noise:
                results[arm] = run_arm(arm)
            results[arm]["noise_kernels"] = noise.launched.value
        else:
            raise SystemExit(f"unknown arm {arm}")
        r = results[arm]
        log(f"SUMMARY {arm}: {r['bad']} of {r['replays']} replays bad ({100 * r['bad'] / max(1, r['replays']):.2f}%)")

    summary = {"device": props.name, "sm": f"{major}{minor}", "sms": props.multi_processor_count,
               "variant": args.variant, "rows": args.rows, "shapes": args.shapes, "results": results}
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1))
    session.__exit__(None, None, None)


if __name__ == "__main__":
    main()
