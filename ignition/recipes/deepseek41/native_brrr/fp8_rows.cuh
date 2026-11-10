// SPDX-FileCopyrightText: Copyright (c) 2026 Fraser Price
// SPDX-License-Identifier: Apache-2.0
// Adapted from BRRR; see NOTICE for the upstream project and contributors.
#pragma once

#include "common.cuh"

__device__ __forceinline__ uint4 ld_stream(const uint8_t* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

__device__ __forceinline__ unsigned ld_scale(const uint8_t* p) {
  unsigned v;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

template <int BA, int TA, int BB>
__device__ __forceinline__ void mma_mx(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1, unsigned sa, unsigned sb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "h"((short)BA), "h"((short)TA), "r"(sb), "h"((short)BB), "h"((short)0));
}

__device__ __forceinline__ unsigned scale_index(int grp, int tok) { return (grp >> 2) * 512 + tok * 16 + (grp & 3); }

template <int K, int UNITS, int STRIDE>
struct Rows {
  static constexpr int PAIRS = K / 64;
  uint4 w[UNITS];
  unsigned s[UNITS];

  __device__ __forceinline__ void load(const uint8_t* weight, const uint8_t* scales, long tile, int kp0, int lane) {
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int j = 0; j < UNITS; ++j) {
      const int kp = kp0 + j * STRIDE;
      w[j] = ld_stream(weight + (tile * 8 + g) * K + kp * 64 + t * 16);
      s[j] = ld_scale(scales + ((tile * PAIRS + kp) * 4 + t) * 4);
    }
  }

  __device__ __forceinline__ void ready(int lane) {
    const bool low = (lane & 3) < 2;
#pragma unroll
    for (int j = 0; j < UNITS; ++j) {
      const uint4 v = w[j];
      const unsigned r0 = __shfl_xor_sync(0xffffffffu, low ? v.z : v.x, 2), r1 = __shfl_xor_sync(0xffffffffu, low ? v.w : v.y, 2);
      w[j] = make_uint4(low ? v.x : r0, low ? v.y : r1, low ? r0 : v.z, low ? r1 : v.w);
    }
  }

  template <int TOKENS>
  __device__ __forceinline__ void accumulate(const uint8_t* xq, int pitch, const uint8_t* xs, int xs_ld, int kp0, int kb0, int lane, float (&acc)[4]) const {
    static_assert(TOKENS == 8 || TOKENS == 16);
    const int g = lane >> 2, t = lane & 3, piece = 2 * (t & 1) + (t >> 1);
    const uint8_t* lo = xq + g * pitch + piece * 8;
    const uint8_t* sa = xs + (TOKENS == 16 && (t & 1) ? g + 8 : g) * xs_ld;
    float odd[4] = {};
#pragma unroll
    for (int j = 0; j < UNITS; ++j) {
      const int kb = (kp0 + j * STRIDE) * 2 - kb0;
      const uint2 x0 = *reinterpret_cast<const uint2*>(lo + kb * 32), x1 = *reinterpret_cast<const uint2*>(lo + kb * 32 + 32);
      uint2 y0 = make_uint2(0u, 0u), y1 = y0;
      if constexpr (TOKENS == 16) {
        y0 = *reinterpret_cast<const uint2*>(lo + 8 * pitch + kb * 32);
        y1 = *reinterpret_cast<const uint2*>(lo + 8 * pitch + kb * 32 + 32);
      }
      mma_mx<0, 0, 0>(acc, x0.x, y0.x, x0.y, y0.y, w[j].x, w[j].y, sa[kb], s[j]);
      mma_mx<0, 0, 2>(odd, x1.x, y1.x, x1.y, y1.y, w[j].z, w[j].w, sa[kb + 1], s[j]);
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[e] += odd[e];
  }
};
