// SPDX-FileCopyrightText: Copyright (c) 2026 Fraser Price
// SPDX-License-Identifier: Apache-2.0
// Adapted from BRRR; see NOTICE for the upstream project and contributors.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

constexpr int RANKS = 4;
constexpr float AMAX_FLOOR = 1e-4f;
constexpr float MX_AMAX_FLOOR = 1e-30f;

__host__ __device__ __forceinline__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__ unsigned smem_u32(const void* p) { return static_cast<unsigned>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void cp_async4(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(smem_u32(dst)), "l"(src) : "memory");
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src) : "memory");
}

__device__ __forceinline__ unsigned ld_volatile(const unsigned* p) {
  unsigned v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ unsigned ld_acquire_gpu(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ unsigned ld_acquire_sys(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ unsigned long long ld_volatile64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void red_release(unsigned* p, unsigned v) { asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }

__device__ __forceinline__ void st_release_sys(unsigned* p, unsigned v) { asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }

__device__ __forceinline__ void st_relaxed_sys(unsigned* p, unsigned v) { asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }

__device__ __forceinline__ void st_relaxed_sys64(unsigned long long* p, unsigned long long v) {
  asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ unsigned atom_add_acq_rel(unsigned* p, unsigned v) {
  unsigned old;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}

__device__ __forceinline__ void pdl_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;"); }

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }

__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }

__device__ __forceinline__ bool before(unsigned a, unsigned b) { return int(a - b) < 0; }

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}

__device__ __forceinline__ float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const unsigned*>(&v);
}

__device__ __forceinline__ float2 unpack_bf16(unsigned v) { return make_float2(__uint_as_float(v << 16), __uint_as_float(v & 0xffff0000u)); }

__device__ __forceinline__ uint4 pack8(const float (&f)[8]) {
  return make_uint4(pack_bf16(f[0], f[1]), pack_bf16(f[2], f[3]), pack_bf16(f[4], f[5]), pack_bf16(f[6], f[7]));
}

__device__ __forceinline__ void add8(float (&acc)[8], uint4 v) {
  const unsigned u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = unpack_bf16(u[i]);
    acc[2 * i] += f.x;
    acc[2 * i + 1] += f.y;
  }
}

__device__ __forceinline__ float pow2(int e) { return __uint_as_float(unsigned(e + 127) << 23); }

__device__ __forceinline__ float e8m0(unsigned e) { return __uint_as_float(e << 23); }

__device__ __forceinline__ int quant_exp(float amax, float floor) {
  const unsigned bits = __float_as_uint(fmaxf(amax, floor) * (1.0f / 448.0f));
  return int((bits >> 23) & 0xff) - 127 + ((bits & 0x7fffffu) != 0);
}

__device__ __forceinline__ unsigned e4m3x2(float lo, float hi) {
  unsigned short r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}

__device__ __forceinline__ uint8_t quant_byte(float x, int e) { return uint8_t(e4m3x2(x * pow2(-e), 0.0f) & 0xffu); }

__device__ __forceinline__ unsigned float_order(float v) {
  const unsigned bits = __float_as_uint(v);
  return (bits & 0x80000000u) ? ~bits : bits | 0x80000000u;
}

__device__ __forceinline__ int block_exp(const float (&x)[8], float floor) {
  float amax = 0.0f;
#pragma unroll
  for (int e = 0; e < 8; ++e) amax = fmaxf(amax, fabsf(x[e]));
  return quant_exp(quad_max(amax), floor);
}

__device__ __forceinline__ uint2 quant8(const float (&x)[8], int e) {
  const float s = pow2(-e);
  return make_uint2(e4m3x2(x[0] * s, x[1] * s) | (e4m3x2(x[2] * s, x[3] * s) << 16), e4m3x2(x[4] * s, x[5] * s) | (e4m3x2(x[6] * s, x[7] * s) << 16));
}

__device__ __forceinline__ unsigned e4m3x2_f16x2(unsigned v) {
  unsigned h;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(static_cast<unsigned short>(v)));
  return h;
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_f16(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
