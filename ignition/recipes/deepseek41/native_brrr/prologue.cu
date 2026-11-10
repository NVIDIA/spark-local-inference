// SPDX-FileCopyrightText: Copyright (c) 2026 Fraser Price
// SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
// SPDX-License-Identifier: Apache-2.0
// BRRR decoder kernels adapted for Flare preparation. See NOTICE and README.md.
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "fp8_rows.cuh"
#include "host.cuh"

#include <vector>

namespace {

constexpr int HC = 4;
constexpr int MIX = 24;
constexpr int HID = 5120;
constexpr int HCD = HC * HID;
constexpr int MAXM = 128;
constexpr int TILE = 8;
constexpr int TOPK = 6;
constexpr int ROUTES = MAXM * TOPK;
constexpr int EXPERTS = 384;
constexpr int SPAN = 32;
constexpr int ENTRIES = EXPERTS + ROUTES / SPAN;
constexpr int HDIM = 512;
constexpr int ROPE = 64;
constexpr int OA = 2048;
constexpr int OAK = 4096;
constexpr int INTER = 576;
constexpr int SLICES = HID / 32;
constexpr int SH2_LD = 20;
constexpr int HQ_LD = 20;
constexpr float LIMIT = 10.0f;
constexpr int SINKHORN = 20;
constexpr float HC_EPS = 1e-6f;
constexpr float NORM_EPS = 1e-20f;
constexpr float ROUTE_SCALE = 1.5f;

constexpr int GW = 4;
constexpr int GT = GW * 32;
constexpr int THREADS = GT + 32;
constexpr int PER_SM = 2;
constexpr int STAGES = 3;
constexpr int FRAG = 512;
constexpr int FRAGS = 16;
constexpr int SCALE_OFF = FRAGS * FRAG;
constexpr int STAGE = FRAGS * FRAG + 512;
constexpr int OWNA_ROWS = 16;
constexpr int TWO_SHOT_MIN = 17;
constexpr int BT = 18432;
constexpr int BS = 640;

constexpr int CHUNKS = 40;
constexpr int CHUNK_COLS = HID / CHUNKS;
constexpr int CHUNK_VECS = CHUNK_COLS / 8;
constexpr int OTOK = 32;
constexpr int GROUPS = MAXM / OTOK;
constexpr int TILES = MAXM / TILE;
constexpr int MIXW = 32;
constexpr int KPARTS = 10;
constexpr int XNORM_MIN = 16;
constexpr int WOA_BLOCKS = OA / 64;
constexpr int WOB_BLOCKS = HID / 64;
constexpr int GATE_BLOCKS = EXPERTS / 64;
constexpr int SH1_BLOCKS = INTER / 32;
constexpr int SH2_BLOCKS = HID / 64;
constexpr int KS_WOA = 8;
constexpr int KS_WOB = 16;
constexpr int KS_GATE = 8;
constexpr int KS_SH1 = 16;
constexpr int KP_WOB = OA / 32 / KS_WOB;
constexpr int KP_GATE = HID / 16 / KS_GATE;
constexpr int KP_SH1 = HID / 32 / KS_SH1;
constexpr int KS_INTER = INTER / 32;
constexpr int KW_INTER = HQ_LD / 4;
constexpr int INTER_STAGES = cdiv(KS_INTER, 4);
constexpr int R1_BLOCKS = INTER / 64;
constexpr int R1_PARTS = 10;
constexpr int R2_BLOCKS = HID / 128;
constexpr int KG_WOA = OAK / 128;
constexpr int KG_WOB = OA / 128;
constexpr int KG_GATE = HID * 2 / 128;
constexpr int KG_SH1 = HID / 128;
static_assert(SCALE_OFF + 512 <= STAGE && CHUNK_COLS == 128 && FRAGS * FRAG == 8192 && R2_BLOCKS == CHUNKS);
static_assert(OWNA_ROWS * HC * 32 * 4 <= STAGE && (MIX - OWNA_ROWS) * HC * 32 * 4 <= STAGE);
static_assert(OTOK * KS_WOA * 64 <= BT && OTOK * KS_WOB * 32 <= BT && OTOK * KS_SH1 * 32 <= BT && OTOK * KS_INTER * 32 <= BT && OTOK * KS_GATE * 32 <= BT);

__host__ __device__ __forceinline__ int tiles_of(int m) { return cdiv(m, TILE); }

__host__ __device__ __forceinline__ int token_groups(int m) { return cdiv(m, OTOK); }

__host__ __device__ __forceinline__ bool split_norm(int m) { return m > XNORM_MIN; }

__host__ __device__ __forceinline__ int quads(int m) { return cdiv(m, GW); }

constexpr int WOA_SMALL = 16;

__host__ __device__ __forceinline__ int woa_ks(int m) { return m <= WOA_SMALL ? 2 * KS_WOA : KS_WOA; }

struct Span {
  int first, count, tiles, ts;
};

__host__ __device__ __forceinline__ Span span_of(int first, int count) {
  const int tiles = tiles_of(count);
  return {first, count, tiles, tiles == 1 ? 1 : tiles == 2 ? 2 : 4};
}

__host__ __device__ __forceinline__ Span group_of(int m, int tg) { return span_of(tg * OTOK, m - tg * OTOK < OTOK ? m - tg * OTOK : OTOK); }

__host__ __device__ __forceinline__ Span tile_span(int m, int t) { return span_of(t * TILE, m - t * TILE < TILE ? m - t * TILE : TILE); }

constexpr int OWNA_SPANS = 4;

__host__ __device__ __forceinline__ int owna_tiles(int m) { return cdiv(tiles_of(m), OWNA_SPANS); }

__host__ __device__ __forceinline__ int owna_spans(int m) { return cdiv(tiles_of(m), owna_tiles(m)); }

struct Layout {
  int m, tg, amix, wob, owna, xnorm, gate, route, sh1, mix, routed;
};

__host__ __device__ __forceinline__ Layout layout(int m) {
  Layout l{};
  l.m = m;
  l.tg = token_groups(m);
  l.amix = WOA_BLOCKS * (OAK / 32 / woa_ks(m)) * l.tg;
  l.wob = l.amix + quads(m);
  l.owna = l.wob + CHUNKS * (2 * KP_WOB * l.tg + 1);
  l.xnorm = l.owna + SLICES * owna_spans(m);
  l.gate = l.xnorm + (split_norm(m) ? KPARTS * tiles_of(m) : 0);
  l.route = l.gate + GATE_BLOCKS * KP_GATE * l.tg;
  l.sh1 = l.route + m;
  l.mix = l.sh1 + SH1_BLOCKS * KP_SH1 * l.tg;
  l.routed = l.mix + quads(m);
  return l;
}

constexpr int R1_JOBS = R1_BLOCKS * R1_PARTS;

__host__ __device__ __forceinline__ int chunk_jobs(const Layout& L, int n) { return 2 * L.tg + n + 1; }

__device__ __forceinline__ int total_jobs(const Layout& L, int n) { return L.routed + n * R1_JOBS + CHUNKS * chunk_jobs(L, n); }

enum : int { J_WOA, J_AMIX, J_WOB, J_XA, J_OWNA, J_XNORM, J_GATE, J_ROUTE, J_SH1, J_SH2, J_MIX, J_R1, J_R2, J_XF, J_END };

struct Job {
  int type, a, b, c, stages, e, f, n;
};

struct Slot {
  int chunk, item;
};

__device__ __forceinline__ int exchange_lag(int per) { return min(CHUNKS, max(1, cdiv(int(gridDim.x), per))); }

__device__ __forceinline__ Slot lagged(int r, int per) {
  const int lag = exchange_lag(per);
  if (r < lag * per) return {r / per, r % per};
  r -= lag * per;
  if (r < (CHUNKS - lag) * (per + 1)) {
    const int k = lag + r / (per + 1), i = r % (per + 1);
    return i < per ? Slot{k, i} : Slot{k - lag, per};
  }
  return {CHUNKS - lag + r - (CHUNKS - lag) * (per + 1), per};
}

__host__ __device__ __forceinline__ Job blocked(int type, int r, int parts, int tg, int stages) { return {type, r / (parts * tg), r / tg % parts, r % tg, stages}; }

__device__ __forceinline__ Job decode(const Layout& L, int n, int j) {
  if (j < L.amix) return blocked(J_WOA, j, OAK / 32 / woa_ks(L.m), L.tg, woa_ks(L.m) / 4);
  if (j < L.wob) return {J_AMIX, j - L.amix, 0, 0, 1};
  if (j < L.owna) {
    const int per = 2 * KP_WOB * L.tg;
    const Slot sl = lagged(j - L.wob, per);
    if (sl.item == per) return {J_XA, sl.chunk, 0, 0, 1};
    const Job w = blocked(J_WOB, sl.item, KP_WOB, L.tg, KS_WOB / 4);
    return {J_WOB, 2 * sl.chunk + w.a, w.b, w.c, w.stages};
  }
  if (j < L.xnorm) return {J_OWNA, (j - L.owna) % SLICES, (j - L.owna) / SLICES, 0, 2};
  if (j < L.gate) return {J_XNORM, (j - L.xnorm) % KPARTS, (j - L.xnorm) / KPARTS, 0, 1};
  if (j < L.route) return blocked(J_GATE, j - L.gate, KP_GATE, L.tg, KS_GATE / 4);
  if (j < L.sh1) return {J_ROUTE, j - L.route, 0, 0, 1};
  if (j < L.mix) return blocked(J_SH1, j - L.sh1, KP_SH1, L.tg, KS_SH1 / 4);
  if (j < L.routed) return {J_MIX, j - L.mix, 0, 0, 1};
  int r = j - L.routed;
  if (r < n * R1_JOBS) return {J_R1, r / R1_JOBS, r / R1_PARTS % R1_BLOCKS, r % R1_PARTS, 4};
  r -= n * R1_JOBS;
  if (r < CHUNKS * chunk_jobs(L, n)) {
    const int shared = 2 * L.tg;
    const Slot sl = lagged(r, shared + n);
    if (sl.item < shared) return {J_SH2, 2 * sl.chunk + sl.item / L.tg, 0, sl.item % L.tg, INTER_STAGES};
    return sl.item < shared + n ? Job{J_R2, sl.item - shared, sl.chunk, 0, INTER_STAGES} : Job{J_XF, sl.chunk, 0, 0, 1};
  }
  return {J_END, 0, 0, 0, 1};
}

constexpr int C_CLAIM = 0;
constexpr int C_ZA = C_CLAIM + 1;
constexpr int C_Z = C_ZA + WOA_BLOCKS * GROUPS;
constexpr int C_YA = C_Z + KP_WOB * GROUPS;
constexpr int C_YC = C_YA + WOB_BLOCKS * GROUPS;
constexpr int C_YR = C_YC + CHUNKS;
constexpr int C_AMIX = C_YR + CHUNKS;
constexpr int C_A = C_AMIX + 1;
constexpr int C_XK = C_A + TILES;
constexpr int C_G = C_XK + KPARTS * GROUPS;
constexpr int C_ROUTE = C_G + GROUPS;
constexpr int C_SA = C_ROUTE + 1;
constexpr int C_HS = C_SA + SH1_BLOCKS * GROUPS;
constexpr int C_SF = C_HS + GROUPS;
constexpr int C_FIN = C_SF + CHUNKS;
constexpr int CACHED = C_FIN;
constexpr int C_PUB = C_FIN + 1;
constexpr int C_R1A = C_PUB + 4 * CHUNKS;
constexpr int C_R1D = C_R1A + ENTRIES * R1_BLOCKS;
constexpr int C_NENT = C_R1D + ENTRIES;
constexpr int C_ECOUNT = C_NENT + 1;
constexpr int COUNTERS = C_ECOUNT + EXPERTS;

struct Peers {
  __nv_bfloat16* pub[RANKS];
  unsigned* flags[RANKS];
};

constexpr long PUB_SLOT = long(MAXM) * HID;
constexpr long FLAG_BYTES = 8192;
static_assert(2 * 2 * RANKS * CHUNKS * sizeof(unsigned) <= FLAG_BYTES);

struct Table {
  int expert[ENTRIES];
  int chunk[ENTRIES];
  int routes[EXPERTS * MAXM];
};

struct Scratch {
  unsigned* epoch;
  unsigned* counters;
  Table* table;
  float* zacc;
  uint8_t* zq;
  uint8_t* zse;
  float* yacc;
  __nv_bfloat16* yattn;
  float* mixp;
  float* mixq;
  __nv_bfloat16* xc;
  float* xsq;
  __nv_bfloat16* xbf;
  uint8_t* xq;
  uint8_t* xse;
  float* gacc;
  float* topk_w;
  float* sacc;
  uint8_t* hsq;
  uint8_t* hse;
  float* shacc;
  float* hacc;
  uint8_t* hq;
  uint8_t* hqe;
  float* yracc;
  uint8_t* aq;
  uint8_t* as;
  float* qkp;
  float* qkv;
  float* qsq;
  unsigned* tiles;
  unsigned* pc;
};

struct Weights {
  const uint8_t* woa;
  const uint8_t* woa_s;
  const uint8_t* wob;
  const uint8_t* wob_s;
  const uint8_t* gate;
  const float* gate_b;
  const float* bias_vl;
  const uint8_t* sh13;
  const uint8_t* sh13_s;
  const uint8_t* sh2;
  const uint8_t* sh2_s;
  const uint4* w13;
  const unsigned* s13;
  const uint4* w2;
  const unsigned* s2;
  long w13_stride;
  long s13_stride;
  long w2_stride;
  long s2_stride;
  const float* fn;
  const float* hc_scale;
  const float* hc_base;
  const __nv_bfloat16* norm;
};

struct Params {
  Peers peers;
  Scratch s;
  Weights w;
  int rank;
  int m;
  const __nv_bfloat16* o;
  long o_stride;
  const long* positions;
  const void* input_ids;
  int ids64;
  long image_lo;
  const float* cos_sin;
  const __nv_bfloat16* resid0;
  float* attn_post;
  float* attn_comb;
  float* attn_pre;
  const float* attn_scale;
  const float* attn_base;
  int attn_mix;
  __nv_bfloat16* y_ffn;
  __nv_bfloat16* resid1;
  float* ffn_post;
  float* ffn_comb;
  float* ffn_pre;
};

struct Smem {
  __align__(128) uint8_t ring[STAGES][STAGE];
  unsigned long long full[STAGES];
  unsigned long long empty[STAGES];
  Job job[STAGES];
  unsigned seen[CACHED];
  __align__(16) uint8_t bt[BT];
  __align__(16) uint8_t bs[BS];
  float rs[OTOK];
  int routes[SPAN];
  float cand_key[GW * TOPK];
  float cand_score[GW * TOPK];
  int cand_expert[GW * TOPK];
  int flag;
  int last;
  int ready;
};

constexpr unsigned SPIN_NS = 200;

__device__ __forceinline__ void mbar_init(unsigned long long* bar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_arrive(unsigned long long* bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared::cta.b64 st, [%0];\n}" ::"r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void mbar_wait(unsigned long long* bar, unsigned parity) {
  asm volatile(
      "{\n .reg .pred p;\n LAB_WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra LAB_WAIT;\n}" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}

__device__ __forceinline__ void cp_async16_hint(void* dst, const void* src, unsigned long long policy) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(smem_u32(dst)), "l"(src), "l"(policy) : "memory");
}

__device__ __forceinline__ unsigned long long evict_first_policy() {
  unsigned long long policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  return policy;
}

__device__ __forceinline__ void cp_async_arrive(unsigned long long* bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }

__device__ __forceinline__ float lane_sums(float (&v)[32], int lane) {
#pragma unroll
  for (int w = 16; w >= 1; w >>= 1) {
    const bool hi = lane & w;
#pragma unroll
    for (int i = 0; i < w; ++i) {
      const float keep = hi ? v[i + w] : v[i], send = hi ? v[i] : v[i + w];
      v[i] = keep + __shfl_xor_sync(0xffffffffu, send, w);
    }
  }
  return v[0];
}

__device__ __forceinline__ void mma_e4m3x(int q, float (&d)[4], uint4 v, uint2 b, unsigned sa, unsigned sb) {
  switch (q) {
    case 0: mma_mx<0, 0, 0>(d, v.x, v.z, v.y, v.w, b.x, b.y, sa, sb); break;
    case 1: mma_mx<1, 0, 1>(d, v.x, v.z, v.y, v.w, b.x, b.y, sa, sb); break;
    case 2: mma_mx<2, 0, 2>(d, v.x, v.z, v.y, v.w, b.x, b.y, sa, sb); break;
    default: mma_mx<3, 0, 3>(d, v.x, v.z, v.y, v.w, b.x, b.y, sa, sb); break;
  }
}

__device__ __forceinline__ unsigned bf16x2_f16x2(unsigned v) {
  unsigned h;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(h) : "f"(__uint_as_float(v & 0xffff0000u)), "f"(__uint_as_float(v << 16)));
  return h;
}

__device__ __forceinline__ unsigned fp4_lo(unsigned w) { return ((w << 4) & 0x80808080u) | ((w << 2) & 0x1c1c1c1cu); }

__device__ __forceinline__ unsigned fp4_hi(unsigned w) { return (w & 0x80808080u) | ((w >> 2) & 0x1c1c1c1cu); }

template <int BA>
__device__ __forceinline__ void mma_fp4x32(float (&t0)[4], float (&t1)[4], uint4 w, unsigned b0, unsigned b1, unsigned sa, unsigned sb) {
  mma_mx<BA, 0, BA>(t0, fp4_lo(w.x), fp4_lo(w.y), fp4_hi(w.x), fp4_hi(w.y), b0, b1, sa, sb);
  mma_mx<BA, 1, BA>(t1, fp4_lo(w.z), fp4_lo(w.w), fp4_hi(w.z), fp4_hi(w.w), b0, b1, sa, sb);
}

__device__ __forceinline__ void mma_fp4(int q, float (&t0)[4], float (&t1)[4], uint4 w, unsigned b0, unsigned b1, unsigned sa, unsigned sb) {
  switch (q) {
    case 0: mma_fp4x32<0>(t0, t1, w, b0, b1, sa, sb); break;
    case 1: mma_fp4x32<1>(t0, t1, w, b0, b1, sa, sb); break;
    case 2: mma_fp4x32<2>(t0, t1, w, b0, b1, sa, sb); break;
    default: mma_fp4x32<3>(t0, t1, w, b0, b1, sa, sb); break;
  }
}

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float swiglu(float gate, float up) {
  gate = fminf(gate, LIMIT);
  up = fminf(fmaxf(up, -LIMIT), LIMIT);
  return gate / (1.0f + expf(-gate)) * up;
}

struct Ctx {
  const Params& p;
  Smem& sm;
  Layout L;
  int warp;
  int lane;
  int tid;
  int it;
  unsigned epoch;

  __device__ __forceinline__ void sync() const { asm volatile("bar.sync 1, %0;" ::"n"(GT) : "memory"); }

  __device__ void wait(int index, unsigned target) {
    if (tid == 0 && (index >= CACHED || sm.seen[index] < target)) {
      unsigned seen = ld_acquire_gpu(p.s.counters + index);
      while (seen < target) {
        __nanosleep(SPIN_NS);
        seen = ld_acquire_gpu(p.s.counters + index);
      }
      if (index < CACHED) sm.seen[index] = seen;
    }
    sync();
  }

  __device__ void signal(int index) {
    sync();
    if (tid == 0) {
      red_release(p.s.counters + index, 1u);
    }
  }

  __device__ bool arrive(int index, unsigned target) {
    sync();
    if (tid == 0) {
      sm.flag = atom_add_acq_rel(p.s.counters + index, 1u) == target - 1;
    }
    sync();
    return sm.flag;
  }

  __device__ __forceinline__ const uint8_t* stage() const { return sm.ring[it % STAGES]; }

  __device__ __forceinline__ void full(int k) const {
    unsigned long long* bar = &sm.full[k % STAGES];
    const unsigned parity = (k / STAGES) & 1;
    mbar_wait(bar, parity);
  }

  __device__ __forceinline__ void acquire() const { full(it); }

  __device__ __forceinline__ void release() {
    __syncwarp();
    if (lane == 0) mbar_arrive(&sm.empty[it % STAGES]);
    ++it;
  }

  __device__ __forceinline__ void next() {
    release();
    acquire();
  }

  __device__ __forceinline__ const uint8_t* held(int s) const {
    full(it + s);
    return sm.ring[(it + s) % STAGES];
  }

  __device__ __forceinline__ void drop(int stages) {
    for (int s = 0; s < stages; ++s) release();
  }
};

__device__ __forceinline__ const unsigned* stage_words(const uint8_t* stage) { return reinterpret_cast<const unsigned*>(stage + SCALE_OFF); }

struct Acc8 {
  float4 lo, hi;
};

__device__ __forceinline__ Acc8 load8(const float* src, bool live) {
  const float4* a = reinterpret_cast<const float4*>(src);
  const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  return {live ? __ldcg(a) : zero, live ? __ldcg(a + 1) : zero};
}

__device__ __forceinline__ void unpack8(const Acc8& a, float (&f)[8]) {
  f[0] = a.lo.x, f[1] = a.lo.y, f[2] = a.lo.z, f[3] = a.lo.w, f[4] = a.hi.x, f[5] = a.hi.y, f[6] = a.hi.z, f[7] = a.hi.w;
}

__device__ __forceinline__ void clear8(float* dst) {
  float4* a = reinterpret_cast<float4*>(dst);
  const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  __stcg(a, zero);
  __stcg(a + 1, zero);
}

__device__ __forceinline__ long chunk_off(int i, int k) { return long(i / CHUNK_VECS) * HID + k * CHUNK_COLS + (i % CHUNK_VECS) * 8; }

__device__ __forceinline__ __nv_bfloat16* inbox(const Params& p, int r, int which, int stage, unsigned epoch) {
  return p.peers.pub[r] + ((long(which) * 2 + stage) * 2 + (epoch & 1)) * PUB_SLOT;
}

__device__ __forceinline__ int flag_index(int which, int stage, int src, int g) { return ((which * 2 + stage) * RANKS + src) * CHUNKS + g; }

__device__ void post(Ctx& c, int which, int stage, int g, unsigned target, unsigned epoch, unsigned mask) {
  if (target > 1) {
    if (!c.arrive(C_PUB + (which * 2 + stage) * CHUNKS + g, target)) return;
  } else {
    c.sync();
  }
  if (c.tid == 0) {
    __threadfence_system();
    for (int r = 0; r < RANKS; ++r)
      if (mask >> r & 1) st_relaxed_sys(c.p.peers.flags[r] + flag_index(which, stage, c.p.rank, g), epoch);
  }
}

__device__ void await(Ctx& c, int which, int stage, int g, unsigned epoch, unsigned mask) {
  if (c.tid < RANKS && (mask >> c.tid & 1)) {
    const unsigned* flag = c.p.peers.flags[c.p.rank] + flag_index(which, stage, c.tid, g);
    while (before(ld_acquire_sys(flag), epoch)) __nanosleep(SPIN_NS);
  }
  c.sync();
}

constexpr int XB = 4;

__device__ void reduce4(Ctx& c, int which, int k, unsigned epoch, int nv, __nv_bfloat16* out, __nv_bfloat16* keep) {
  const Params& p = c.p;
  const __nv_bfloat16* src[RANKS];
#pragma unroll
  for (int r = 0; r < RANKS; ++r) src[r] = inbox(p, r, which, 0, epoch);
  for (int base = 0; base < nv; base += GT * XB) {
    uint4 v[XB][RANKS];
#pragma unroll
    for (int u = 0; u < XB; ++u) {
      const int i = base + c.tid + u * GT;
#pragma unroll
      for (int r = 0; r < RANKS; ++r)
        v[u][r] = i >= nv ? make_uint4(0, 0, 0, 0)
                  : r == p.rank ? __ldcg(reinterpret_cast<const uint4*>(src[r] + chunk_off(i, k)))
                                : __ldcv(reinterpret_cast<const uint4*>(src[r] + chunk_off(i, k)));
    }
#pragma unroll
    for (int u = 0; u < XB; ++u) {
      const int i = base + c.tid + u * GT;
      if (i >= nv) continue;
      float acc[8] = {};
#pragma unroll
      for (int r = 0; r < RANKS; ++r) add8(acc, v[u][r]);
      const uint4 sum = pack8(acc);
      *reinterpret_cast<uint4*>(out + chunk_off(i, k)) = sum;
      if (keep) *reinterpret_cast<uint4*>(keep + chunk_off(i, k)) = sum;
    }
  }
}

__device__ void gather(Ctx& c, const __nv_bfloat16* src, int k, int nv, __nv_bfloat16* out) {
  constexpr int GB = 2 * XB;
  for (int base = 0; base < nv; base += GT * GB) {
    uint4 v[GB];
#pragma unroll
    for (int u = 0; u < GB; ++u) {
      const int i = base + c.tid + u * GT;
      v[u] = i < nv ? __ldcv(reinterpret_cast<const uint4*>(src + chunk_off(i, k))) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < GB; ++u) {
      const int i = base + c.tid + u * GT;
      if (i < nv) *reinterpret_cast<uint4*>(out + chunk_off(i, k)) = v[u];
    }
  }
}

__device__ void mix_finish(int lane, int tok, float mine, const float* scale, const float* base, float* pre, float* post, float* comb) {
  const float sq = __shfl_sync(0xffffffffu, mine, MIX);
  const float rs = rsqrtf(sq / HCD + NORM_EPS);
  const float mix = mine * rs;
  if (lane < HC) {
    pre[tok * HC + lane] = sigmoid(mix * __ldg(scale) + __ldg(base + lane)) + HC_EPS;
  } else if (lane < 2 * HC) {
    post[tok * HC + lane - HC] = 2.0f * sigmoid(mix * __ldg(scale + 1) + __ldg(base + lane));
  }
  const int idx = 2 * HC + (lane & 15);
  float cm = __shfl_sync(0xffffffffu, mix, idx) * scale[2] + base[idx];
  cm = expf(cm - quad_max(cm));
  cm = cm / quad_sum(cm) + HC_EPS;
  for (int it = 0; it < SINKHORN; ++it) {
    if (it > 0) cm /= quad_sum(cm) + HC_EPS;
    float col = cm + __shfl_xor_sync(0xffffffffu, cm, 4);
    col += __shfl_xor_sync(0xffffffffu, col, 8);
    cm /= col + HC_EPS;
  }
  if (lane < HC * HC) comb[tok * HC * HC + lane] = cm;
}

__device__ void share(Ctx& c, int which, int k, __nv_bfloat16* out) {
  const Params& p = c.p;
  const unsigned epoch = c.epoch;
  const int nv = p.m * CHUNK_VECS;
  const unsigned peers = ((1u << RANKS) - 1) & ~(1u << p.rank);
  const int per = which == 0 && p.m < TWO_SHOT_MIN ? CHUNKS : 1, g = k / per;
  if (p.m < TWO_SHOT_MIN) {
    post(c, which, 0, g, per, epoch, peers);
    await(c, which, 0, g, epoch, peers);
    reduce4(c, which, k, epoch, nv, out, nullptr);
    return;
  }
  const int owner = k % RANKS;
  post(c, which, 0, g, per, epoch, per == 1 ? 1u << owner : peers);
  if (p.rank != owner) {
    await(c, which, 1, g, epoch, 1u << owner);
    gather(c, inbox(p, owner, which, 1, epoch), k, nv, out);
    return;
  }
  await(c, which, 0, g, epoch, peers);
  reduce4(c, which, k, epoch, nv, out, inbox(p, p.rank, which, 1, epoch));
  post(c, which, 1, g, per == 1 ? 1 : per / RANKS, epoch, peers);
}

__device__ __forceinline__ void red4(float* dst, float4 v) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(dst), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}

__device__ __forceinline__ float4 transpose4(const float* a, int lane) {
  const int i = (lane >> 2) & 3;
  const bool p = i & 1, q = i & 2;
  const float r0 = __shfl_xor_sync(0xffffffffu, p ? a[0] : a[1], 4), r1 = __shfl_xor_sync(0xffffffffu, p ? a[2] : a[3], 4);
  const float b0 = p ? r0 : a[0], b1 = p ? a[1] : r0, b2 = p ? r1 : a[2], b3 = p ? a[3] : r1;
  const float w0 = __shfl_xor_sync(0xffffffffu, q ? b0 : b2, 8), w1 = __shfl_xor_sync(0xffffffffu, q ? b1 : b3, 8);
  return q ? make_float4(w0, w1, b2, b3) : make_float4(b0, b1, w0, w1);
}

__device__ __forceinline__ int quad_col(int tt, int lane) { return tt * TILE + 2 * (lane & 3) + ((lane >> 2) & 1); }

__device__ __forceinline__ int quad_row(int lane) { return (lane >> 4) * 4 + ((lane >> 3) & 1) * 8; }

__device__ __forceinline__ void scatter(float* dst, int ld, int row0, const Span& sp, int tt, const float (&a)[4], int lane) {
  const float4 v = transpose4(a, lane);
  const int tok = quad_col(tt, lane);
  if (tok < sp.count) red4(dst + long(sp.first + tok) * ld + row0 + quad_row(lane), v);
}

__device__ __forceinline__ uint4 frag(const uint8_t* stage, int f, int lane) { return reinterpret_cast<const uint4*>(stage + f * FRAG)[lane]; }

struct Bq {
  uint2 x;
  unsigned s;
};

__device__ __forceinline__ uint8_t* bt_row(Ctx& c, int tok, int ks, int k, int bytes) { return c.sm.bt + ((tok >> 3) * ks + k) * bytes * 8 + (tok & 7) * bytes; }

__device__ __forceinline__ Bq load_q(const Ctx& c, int ks, int kw, int tt, int k) {
  return {reinterpret_cast<const uint2*>(c.sm.bt + (tt * ks + k) * 256)[c.lane], reinterpret_cast<const unsigned*>(c.sm.bs)[(tt * TILE + (c.lane >> 2)) * kw + (k >> 2)]};
}

template <class Row, class Scale>
__device__ __forceinline__ void stage_q(Ctx& c, int count, int ks, int kw, Row row, Scale scale) {
  for (int tok = c.warp; tok < count; tok += GW) {
    const uint8_t* src = row(tok);
    for (int r = c.lane; r < ks * 2; r += 32) cp_async16(bt_row(c, tok, ks, r >> 1, 32) + (r & 1) * 16, src + r * 16);
    if (c.lane < kw) cp_async4(c.sm.bs + (tok * kw + c.lane) * 4, scale(tok) + c.lane * 4);
  }
  cp_async_wait_all();
  c.sync();
}

template <class Row, class Scale>
__device__ __forceinline__ void fill_q(Ctx& c, int count, int ks, int kw, Row row, Scale scale) {
  c.sync();
  stage_q(c, count, ks, kw, row, scale);
}

template <int A, bool HOLD = false, class Load, class Step>
__device__ __forceinline__ void gemm(Ctx& c, const Span& sp, int ksteps, float (&acc)[GW][A], Load load, Step step) {
  const int rs = GW / sp.ts, rt0 = c.warp / sp.ts, tt = c.warp % sp.ts;
  const bool live = tt < sp.tiles;
#pragma unroll
  for (int r = 0; r < GW; ++r)
#pragma unroll
    for (int e = 0; e < A; ++e) acc[r][e] = 0.0f;
  const int stages = (ksteps + 3) >> 2;
#pragma unroll 1
  for (int s = 0; s < stages; ++s) {
    if (!HOLD && s) c.next();
    const uint8_t* stage = HOLD ? c.held(s) : c.stage();
    if (live) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        if (s * 4 + q < ksteps) {
          const auto b = load(tt, s * 4 + q);
#pragma unroll
          for (int r = 0; r < GW; ++r)
            if (r < sp.ts) step(acc[r], stage, q, rt0 + r * rs, b, s);
        }
      }
    }
  }
  if (!HOLD) c.release();
}

template <int A, class Store>
__device__ __forceinline__ void stores(const Ctx& c, const Span& sp, const float (&acc)[GW][A], Store store) {
  const int rs = GW / sp.ts, rt0 = c.warp / sp.ts, tt = c.warp % sp.ts;
  if (tt >= sp.tiles) return;
#pragma unroll
  for (int r = 0; r < GW; ++r)
    if (r < sp.ts) store(rt0 + r * rs, tt, acc[r]);
}

template <class Store>
__device__ __forceinline__ void dense_q(Ctx& c, const Span& sp, int ks, int kw, Store store) {
  float acc[GW][4];
  gemm<4>(
      c, sp, ks, acc, [&](int tt, int k) { return load_q(c, ks, kw, tt, k); },
      [&](float (&a)[4], const uint8_t* stage, int q, int rt, const Bq& b, int) { mma_e4m3x(q, a, frag(stage, q * 4 + rt, c.lane), b.x, stage_words(stage)[rt >> 1], b.s); });
  stores(c, sp, acc, store);
}

__device__ __forceinline__ float2 rope_cs(const float* row, int pair) { return make_float2(__ldg(row + pair), __ldg(row + ROPE / 2 + pair)); }

__device__ __forceinline__ float rot_add(float a, float b, float2 cs) { return a * cs.x + b * cs.y; }

__device__ __forceinline__ float rot_sub(float a, float b, float2 cs) { return a * cs.x - b * cs.y; }

__device__ __forceinline__ uint4 rope(uint4 v, const float* cs, int d) {
  if (d < HDIM - ROPE) return v;
  unsigned u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const float2 r = rope_cs(cs, (d + 2 * q - (HDIM - ROPE)) >> 1);
    const float2 x = unpack_bf16(u[q]);
    u[q] = pack_bf16(rot_add(x.x, x.y, r), rot_sub(x.y, x.x, r));
  }
  return make_uint4(u[0], u[1], u[2], u[3]);
}

__device__ void group_rs(Ctx& c, const Span& sp, float* rs) {
  constexpr int V = SLICES / 4, QMIN = GT / OTOK;
  const int q = sp.count <= GW ? 32 : sp.count <= 2 * GW ? 16 : sp.count <= 4 * GW ? 8 : QMIN;
  const int i = c.tid / q, l = c.tid % q;
  float v = 0.0f;
  if (i < sp.count) {
    const float4* x = reinterpret_cast<const float4*>(c.p.s.xsq + (sp.first + i) * SLICES);
#pragma unroll
    for (int k = 0; k < cdiv(V, QMIN); ++k)
      if (l + k * q < V) {
        const float4 f = __ldcg(x + l + k * q);
        v += (f.x + f.y) + (f.z + f.w);
      }
  }
  for (int o = q / 2; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if (l == 0 && i < sp.count) rs[i] = rsqrtf(v / HID + NORM_EPS);
  c.sync();
}

template <int PER>
__device__ __forceinline__ void select_topk(const float (&score)[PER], float (&key)[PER], const int (&expert)[PER], int lane, float& chosen_key, float& chosen_score, int& chosen_expert) {
  for (int k = 0; k < TOPK; ++k) {
    float best = -INFINITY;
    int bi = 0;
#pragma unroll
    for (int i = 0; i < PER; ++i)
      if (key[i] > best) best = key[i], bi = i;
    int be_local = 0;
#pragma unroll
    for (int i = 0; i < PER; ++i)
      if (i == bi) be_local = expert[i];
    const unsigned ord = float_order(best);
    const unsigned top = __reduce_max_sync(0xffffffffu, ord);
    const int be = int(__reduce_min_sync(0xffffffffu, ord == top ? unsigned(be_local) : 0xffffffffu));
    const unsigned owner = __ffs(__ballot_sync(0xffffffffu, ord == top && be_local == be)) - 1;
    float sv = 0.0f;
#pragma unroll
    for (int i = 0; i < PER; ++i)
      if (i == bi) sv = score[i];
    sv = __shfl_sync(0xffffffffu, sv, owner);
    const float kv = __shfl_sync(0xffffffffu, best, owner);
    if (lane == int(owner))
#pragma unroll
      for (int i = 0; i < PER; ++i)
        if (i == bi) key[i] = -INFINITY;
    if (lane == k) chosen_key = kv, chosen_score = sv, chosen_expert = be;
  }
}

__device__ void job_route(Ctx& c, int tok) {
  const Params& p = c.p;
  constexpr int EW = EXPERTS / GW, PER = EW / 32;
  c.release();
  const long id = !p.input_ids ? -1 : p.ids64 ? __ldg(static_cast<const long*>(p.input_ids) + tok) : long(__ldg(static_cast<const int*>(p.input_ids) + tok));
  const float* bias = p.w.bias_vl && id >= p.image_lo && id < p.image_lo + 5 ? p.w.bias_vl : p.w.gate_b;
  float score[PER], key[PER];
  int expert[PER];
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    expert[i] = c.warp * EW + c.lane + 32 * i;
    float* a = p.s.gacc + tok * EXPERTS + expert[i];
    const float v = __ldcg(a);
    __stcg(a, 0.0f);
    score[i] = sqrtf(v > 20.0f ? v : logf(1.0f + expf(v)));
    key[i] = score[i] + __ldg(bias + expert[i]);
    if (key[i] != key[i]) key[i] = -1e30f;
  }
  float kk = -INFINITY, ks = 0.0f;
  int ke = 0;
  select_topk(score, key, expert, c.lane, kk, ks, ke);
  if (c.lane < TOPK) {
    c.sm.cand_key[c.warp * TOPK + c.lane] = kk;
    c.sm.cand_score[c.warp * TOPK + c.lane] = ks;
    c.sm.cand_expert[c.warp * TOPK + c.lane] = ke;
  }
  c.sync();
  if (c.warp == 0) {
    const bool live = c.lane < GW * TOPK;
    const float cand[1] = {live ? c.sm.cand_score[c.lane] : 0.0f};
    float ckeys[1] = {live ? c.sm.cand_key[c.lane] : -INFINITY};
    const int cexp[1] = {live ? c.sm.cand_expert[c.lane] : EXPERTS + c.lane};
    float k2 = -INFINITY, s2 = 0.0f;
    int e2 = 0;
    select_topk(cand, ckeys, cexp, c.lane, k2, s2, e2);
    float total = 0.0f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) total += __shfl_sync(0xffffffffu, s2, k);
    if (c.lane < TOPK) {
      const int r = tok * TOPK + c.lane;
      p.s.topk_w[r] = s2 * (ROUTE_SCALE / (total > 0.0f ? total : 1.0f));
      Table* tb = p.s.table;
      const int pos = int(atomicAdd(p.s.counters + C_ECOUNT + e2, 1u));
      tb->routes[e2 * MAXM + pos] = r;
      if (pos % SPAN == 0) {
        const int ent = int(atomicAdd(p.s.counters + C_NENT, 1u));
        tb->expert[ent] = e2;
        tb->chunk[ent] = pos / SPAN;
      }
    }
  }
  __threadfence();
  c.signal(C_ROUTE);
}

__device__ void finish_woa(Ctx& c, int rb, int tg) {
  constexpr int PER = OTOK * 8 / GT;
  const Params& p = c.p;
  const Span sp = group_of(p.m, tg);
  float z[PER][8];
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int i = c.tid + u * GT, tok = sp.first + i / 8;
    float v[8];
    unpack8(load8(p.s.zacc + long(tok) * OA + rb * 64 + i % 8 * 8, i / 8 < sp.count), v);
#pragma unroll
    for (int e = 0; e < 8; ++e) z[u][e] = bf(v[e]);
  }
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int i = c.tid + u * GT, tok = sp.first + i / 8, col = rb * 64 + i % 8 * 8;
    const int e = block_exp(z[u], AMAX_FLOOR);
    if (i / 8 >= sp.count) continue;
    clear8(p.s.zacc + long(tok) * OA + col);
    *reinterpret_cast<uint2*>(p.s.zq + long(tok) * OA + col) = quant8(z[u], e);
    if (i % 4 == 0) p.s.zse[tok * 64 + col / 32] = uint8_t(e + 127);
  }
  c.signal(C_Z + rb * 2 / KS_WOB * GROUPS + tg);
}

template <int KS, int TOK>
__device__ void woa_group(Ctx& c, int rb, int kp, int tg) {
  const Params& p = c.p;
  const Span sp = group_of(p.m, tg);
  constexpr int VECS = KS * 4, PER = TOK * VECS / GT;
  static_assert(TOK * KS * 64 <= BT);
  const __nv_bfloat16* o = p.o + rb / 16 * 8 * HDIM + kp * KS * 32;
  const int r = c.tid % VECS, d = (kp * KS * 32 + r * 8) % HDIM;
  uint4 v[PER];
  long pos[PER];
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int tok = (c.tid + u * GT) / VECS;
    const long t = sp.first + tok;
    v[u] = tok < sp.count ? __ldcg(reinterpret_cast<const uint4*>(o + t * p.o_stride + r * 8)) : make_uint4(0, 0, 0, 0);
    pos[u] = p.cos_sin && tok < sp.count && d >= HDIM - ROPE ? __ldg(p.positions + t) : 0;
  }
  c.sync();
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int tok = (c.tid + u * GT) / VECS;
    if (tok >= sp.count) continue;
    if (p.cos_sin) v[u] = rope(v[u], p.cos_sin + pos[u] * ROPE, d);
    *reinterpret_cast<uint4*>(bt_row(c, tok, KS, r >> 2, 64) + (r & 3) * 16) = make_uint4(bf16x2_f16x2(v[u].x), bf16x2_f16x2(v[u].y), bf16x2_f16x2(v[u].z), bf16x2_f16x2(v[u].w));
  }
  c.sync();
  float acc[GW][4];
  gemm<4>(
      c, sp, KS, acc, [&](int tt, int k) { return reinterpret_cast<const uint4*>(c.sm.bt + (tt * KS + k) * 512)[c.lane]; },
      [&](float (&a)[4], const uint8_t* stage, int q, int rt, const uint4& b, int) {
        const uint4 v = frag(stage, q * 4 + rt, c.lane);
        float d[4] = {};
        mma_f16(d, e4m3x2_f16x2(v.x), e4m3x2_f16x2(v.z), e4m3x2_f16x2(v.x >> 16), e4m3x2_f16x2(v.z >> 16), b.x, b.y);
        mma_f16(d, e4m3x2_f16x2(v.y), e4m3x2_f16x2(v.w), e4m3x2_f16x2(v.y >> 16), e4m3x2_f16x2(v.w >> 16), b.z, b.w);
        const float sw = e8m0(stage[SCALE_OFF + (rt >> 1) * 4 + q]);
#pragma unroll
        for (int e = 0; e < 4; ++e) a[e] += d[e] * sw;
      });
  stores(c, sp, acc, [&](int rt, int tt, const float (&a)[4]) { scatter(p.s.zacc, OA, rb * 64 + rt * 16, sp, tt, a, c.lane); });
}

__device__ void job_woa(Ctx& c, int rb, int kp, int tg, int stages) {
  if (stages * 4 == KS_WOA) woa_group<KS_WOA, OTOK>(c, rb, kp, tg);
  else woa_group<2 * KS_WOA, WOA_SMALL>(c, rb, kp, tg);
  if (!c.arrive(C_ZA + rb * GROUPS + tg, OAK / 32 / (stages * 4))) return;
  finish_woa(c, rb, tg);
}

__device__ void finish_wob(Ctx& c, int rb, int tg) {
  const Params& p = c.p;
  const Span sp = group_of(p.m, tg);
  __nv_bfloat16* mine = inbox(p, p.rank, 0, 0, c.epoch);
  constexpr int PER = OTOK * 8 / GT;
  Acc8 v[PER];
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int i = c.tid + u * GT;
    v[u] = load8(p.s.yacc + long(sp.first + i / 8) * HID + rb * 64 + (i % 8) * 8, i < sp.count * 8);
  }
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int i = c.tid + u * GT, tok = sp.first + i / 8, col = rb * 64 + (i % 8) * 8;
    if (i >= sp.count * 8) break;
    clear8(p.s.yacc + long(tok) * HID + col);
    float y[8];
    unpack8(v[u], y);
    *reinterpret_cast<uint4*>(mine + long(tok) * HID + col) = pack8(y);
  }
  c.signal(C_YC + rb / 2);
}

__device__ void job_wob(Ctx& c, int rb, int kp, int tg) {
  const Params& p = c.p;
  const Span sp = group_of(p.m, tg);
  fill_q(
      c, sp.count, KS_WOB, KS_WOB / 4, [&](int tok) { return p.s.zq + long(sp.first + tok) * OA + kp * KS_WOB * 32; },
      [&](int tok) { return p.s.zse + (sp.first + tok) * 64 + kp * KS_WOB; });
  dense_q(c, sp, KS_WOB, KS_WOB / 4, [&](int rt, int tt, const float (&a)[4]) { scatter(p.s.yacc, HID, rb * 64 + rt * 16, sp, tt, a, c.lane); });
  if (c.arrive(C_YA + rb * GROUPS + tg, KP_WOB)) finish_wob(c, rb, tg);
}

__device__ void job_xa(Ctx& c, int k) {
  c.release();
  share(c, 0, k, c.p.s.yattn);
  c.signal(C_YR + k);
}

__device__ void job_amix(Ctx& c, int q) {
  const Params& p = c.p;
  c.release();
  const int tok = q * GW + c.warp;
  if (p.attn_mix && tok < p.m) {
    const float mine = c.lane <= MIX ? __ldcg(p.s.mixq + tok * (MIX + 1) + c.lane) : 0.0f;
    mix_finish(c.lane, tok, mine, p.attn_scale, p.attn_base, p.attn_pre, p.attn_post, p.attn_comb);
  }
  c.signal(C_AMIX);
}

__device__ void owna_tile(Ctx& c, int s, int t8, const float* fn0, const float* fn1) {
  constexpr int PER = TILE / GW;
  const Params& p = c.p;
  const Span sp = tile_span(p.m, t8);
  const int col = s * 32 + c.lane;
  float ya[PER], rk[PER][HC];
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int i = c.warp + u * GW, tok = sp.first + (i < sp.count ? i : 0);
    ya[u] = __bfloat162float(__ldcg(p.s.yattn + tok * HID + col));
#pragma unroll
    for (int k = 0; k < HC; ++k) rk[u][k] = __bfloat162float(__ldg(p.resid0 + (long(tok) * HC + k) * HID + col));
  }
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int i = c.warp + u * GW, tok = sp.first + i;
    if (i >= sp.count) break;
    float r1[HC];
    float x = 0.0f, sq = 0.0f;
#pragma unroll
    for (int j = 0; j < HC; ++j) {
      float v = __ldg(p.attn_post + tok * HC + j) * ya[u];
#pragma unroll
      for (int k = 0; k < HC; ++k) v += __ldg(p.attn_comb + tok * HC * HC + k * HC + j) * rk[u][k];
      r1[j] = bf(v);
      p.resid1[(long(tok) * HC + j) * HID + col] = __float2bfloat16(r1[j]);
      sq += r1[j] * r1[j];
      x += __ldg(p.attn_pre + tok * HC + j) * r1[j];
    }
    static_assert(MIX + 2 <= MIXW);
    float v[32] = {};
#pragma unroll
    for (int r = 0; r < MIX; ++r)
#pragma unroll
      for (int j = 0; j < HC; ++j) v[r] += (r < OWNA_ROWS ? fn0[(r * HC + j) * 32 + c.lane] : fn1[((r - OWNA_ROWS) * HC + j) * 32 + c.lane]) * r1[j];
    x = bf(x);
    p.s.xc[tok * HID + col] = __float2bfloat16(x);
    v[MIX] = sq;
    v[MIX + 1] = x * x;
    const float total = lane_sums(v, c.lane);
    p.s.mixp[(long(tok) * SLICES + s) * MIXW + c.lane] = total;
    if (c.lane == MIX + 1) p.s.xsq[tok * SLICES + s] = total;
  }
  c.signal(C_A + t8);
}

__device__ void job_owna(Ctx& c, int s, int span) {
  const float* fn0 = reinterpret_cast<const float*>(c.stage());
  const float* fn1 = reinterpret_cast<const float*>(c.held(1));
  const int per = owna_tiles(c.p.m), last = min(tiles_of(c.p.m), (span + 1) * per);
  for (int t8 = span * per; t8 < last; ++t8) owna_tile(c, s, t8, fn0, fn1);
  c.drop(2);
}

__device__ void job_xnorm(Ctx& c, int kp, int t8) {
  constexpr int PER = TILE * 64 / GT;
  const Params& p = c.p;
  const Span sp = tile_span(p.m, t8);
  c.release();
  const int k0 = kp * 512 + c.tid % 64 * 8;
  uint4 xv[PER];
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int row = (c.tid + u * GT) / 64;
    xv[u] = row < sp.count ? __ldcg(reinterpret_cast<const uint4*>(p.s.xc + (sp.first + row) * HID + k0)) : make_uint4(0, 0, 0, 0);
  }
  float nw[8] = {};
  add8(nw, __ldg(reinterpret_cast<const uint4*>(p.w.norm + k0)));
  float* rs = c.sm.rs;
  group_rs(c, sp, rs);
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int row = (c.tid + u * GT) / 64, tok = sp.first + row;
    float xc[8] = {}, x8[8];
    add8(xc, xv[u]);
    const float r = row < sp.count ? rs[row] : 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) x8[e] = bf(nw[e] * (xc[e] * r));
    const int ex = block_exp(x8, AMAX_FLOOR);
    if (row < sp.count) {
      *reinterpret_cast<uint2*>(p.s.xq + tok * HID + k0) = quant8(x8, ex);
      *reinterpret_cast<uint4*>(p.s.xbf + tok * HID + k0) = pack8(x8);
      if ((c.tid & 3) == 0) p.s.xse[tok * SLICES + k0 / 32] = uint8_t(ex + 127);
    }
  }
  c.signal(C_XK + kp * GROUPS + t8 * TILE / OTOK);
}

__device__ void gate_group(Ctx& c, int rb, int kp, int tg) {
  const Params& p = c.p;
  const Span sp = group_of(p.m, tg);
  const int k0 = kp * KS_GATE * 16;
  const bool split = split_norm(p.m);
  if (split) {
    c.wait(C_XK + k0 / 512 * GROUPS + tg, sp.tiles);
    stage_q(
        c, sp.count, KS_GATE, 0, [&](int tok) { return reinterpret_cast<const uint8_t*>(p.s.xbf + long(sp.first + tok) * HID + k0); },
        [](int) -> const uint8_t* { return nullptr; });
  } else {
    group_rs(c, sp, c.sm.rs);
    for (int i = c.tid; i < sp.count * (KS_GATE / 2); i += GT) {
      const int tok = i / (KS_GATE / 2), j = i % (KS_GATE / 2), k = k0 + j * 32;
      const long t = sp.first + tok;
      const float r = c.sm.rs[tok];
      float v[32];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        float xf[8] = {}, nf[8] = {};
        add8(xf, __ldcg(reinterpret_cast<const uint4*>(p.s.xc + t * HID + k + u * 8)));
        add8(nf, __ldg(reinterpret_cast<const uint4*>(p.w.norm + k + u * 8)));
#pragma unroll
        for (int e = 0; e < 8; ++e) v[u * 8 + e] = bf(nf[e] * (xf[e] * r));
      }
      uint4 packed[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) packed[u] = pack8(reinterpret_cast<const float(&)[8]>(v[u * 8]));
      uint4* d0 = reinterpret_cast<uint4*>(bt_row(c, tok, KS_GATE, 2 * j, 32));
      uint4* d1 = reinterpret_cast<uint4*>(bt_row(c, tok, KS_GATE, 2 * j + 1, 32));
      d0[0] = packed[0];
      d0[1] = packed[1];
      d1[0] = packed[2];
      d1[1] = packed[3];
      if (rb == 0) {
        float amax = 0.0f;
#pragma unroll
        for (int e = 0; e < 32; ++e) amax = fmaxf(amax, fabsf(v[e]));
        const int ex = quant_exp(amax, AMAX_FLOOR);
        uint2* q = reinterpret_cast<uint2*>(p.s.xq + t * HID + k);
#pragma unroll
        for (int u = 0; u < 4; ++u) q[u] = quant8(reinterpret_cast<const float(&)[8]>(v[u * 8]), ex);
        p.s.xse[t * SLICES + k / 32] = uint8_t(ex + 127);
      }
    }
    c.sync();
  }
  float acc[GW][4];
  gemm<4, true>(
      c, sp, KS_GATE, acc, [&](int tt, int k) { return reinterpret_cast<const uint2*>(c.sm.bt + (tt * KS_GATE + k) * 256)[c.lane]; },
      [&](float (&a)[4], const uint8_t* stage, int q, int rt, const uint2& b, int) {
        const uint4 v = frag(stage, q * 4 + rt, c.lane);
        mma_bf16(a, v.x, v.z, v.y, v.w, b.x, b.y);
      });
  stores(c, sp, acc, [&](int rt, int tt, const float (&a)[4]) { scatter(p.s.gacc, EXPERTS, rb * 64 + rt * 16, sp, tt, a, c.lane); });
  if (!split && rb == 0) c.signal(C_XK + k0 / 512 * GROUPS);
  c.signal(C_G + tg);
}

__device__ void job_gate(Ctx& c, int rb, int kp, int tg) {
  gate_group(c, rb, kp, tg);
  c.drop(KS_GATE / 4);
}

__device__ void finish_sh1(Ctx& c, int b, int tg) {
  const Params& p = c.p;
  const Span sp = group_of(p.m, tg);
  constexpr int PER = OTOK / GW;
  const int col = b * 32 + c.lane;
  float g[PER], u[PER];
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int t = c.warp + i * GW;
    const float* h = p.s.sacc + long(sp.first + t) * 2 * INTER + col;
    g[i] = t < sp.count ? __ldcg(h) : 0.0f;
    u[i] = t < sp.count ? __ldcg(h + INTER) : 0.0f;
  }
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int t = c.warp + i * GW, tok = sp.first + t;
    if (t >= sp.count) break;
    float* h = p.s.sacc + long(tok) * 2 * INTER + col;
    __stcg(h, 0.0f);
    __stcg(h + INTER, 0.0f);
    const float a = bf(swiglu(bf(g[i]), bf(u[i])));
    const int e = quant_exp(warp_max(fabsf(a)), AMAX_FLOOR);
    p.s.hsq[tok * INTER + col] = quant_byte(a, e);
    if (c.lane == 0) p.s.hse[tok * HQ_LD + b] = uint8_t(e + 127);
  }
  c.signal(C_HS + tg);
}

__device__ void job_sh1(Ctx& c, int b, int kp, int tg) {
  const Params& p = c.p;
  const Span sp = group_of(p.m, tg);
  fill_q(
      c, sp.count, KS_SH1, KS_SH1 / 4, [&](int tok) { return p.s.xq + long(sp.first + tok) * HID + kp * KS_SH1 * 32; },
      [&](int tok) { return p.s.xse + (sp.first + tok) * SLICES + kp * KS_SH1; });
  dense_q(c, sp, KS_SH1, KS_SH1 / 4,
          [&](int rt, int tt, const float (&a)[4]) { scatter(p.s.sacc, 2 * INTER, (rt < 2 ? b * 32 : INTER + b * 32 - 32) + rt * 16, sp, tt, a, c.lane); });
  if (c.arrive(C_SA + b * GROUPS + tg, KP_SH1)) finish_sh1(c, b, tg);
}

__device__ void job_sh2(Ctx& c, int rb, int tg) {
  const Params& p = c.p;
  const Span sp = group_of(p.m, tg);
  fill_q(
      c, sp.count, KS_INTER, KW_INTER, [&](int tok) { return p.s.hsq + long(sp.first + tok) * INTER; }, [&](int tok) { return p.s.hse + (sp.first + tok) * HQ_LD; });
  dense_q(c, sp, KS_INTER, KW_INTER, [&](int rt, int tt, const float (&a)[4]) { scatter(p.s.shacc, HID, rb * 64 + rt * 16, sp, tt, a, c.lane); });
  c.signal(C_SF + rb / 2);
}

__device__ __forceinline__ float mix_reduce(const float* mixp, int tok, int lane) {
  constexpr int ROWS = SLICES / 4, BATCH = 10;
  const float4* src = reinterpret_cast<const float4*>(mixp + long(tok) * SLICES * MIXW) + (lane >> 3) * (MIXW / 4) + (lane & 7);
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
  for (int b = 0; b < ROWS; b += BATCH) {
    float4 v[BATCH];
#pragma unroll
    for (int i = 0; i < BATCH; ++i) v[i] = __ldcg(src + (b + i) * MIXW);
#pragma unroll
    for (int i = 0; i < BATCH; ++i) acc.x += v[i].x, acc.y += v[i].y, acc.z += v[i].z, acc.w += v[i].w;
  }
#pragma unroll
  for (int o = 8; o < 32; o <<= 1) {
    acc.x += __shfl_xor_sync(0xffffffffu, acc.x, o);
    acc.y += __shfl_xor_sync(0xffffffffu, acc.y, o);
    acc.z += __shfl_xor_sync(0xffffffffu, acc.z, o);
    acc.w += __shfl_xor_sync(0xffffffffu, acc.w, o);
  }
  const int from = (lane >> 2) & 7, part = lane & 3;
  const float a0 = __shfl_sync(0xffffffffu, acc.x, from), a1 = __shfl_sync(0xffffffffu, acc.y, from);
  const float a2 = __shfl_sync(0xffffffffu, acc.z, from), a3 = __shfl_sync(0xffffffffu, acc.w, from);
  return part == 0 ? a0 : part == 1 ? a1 : part == 2 ? a2 : a3;
}

__device__ void job_mix(Ctx& c, int q) {
  const Params& p = c.p;
  c.release();
  const int tok = q * GW + c.warp;
  if (tok >= p.m) return;
  mix_finish(c.lane, tok, mix_reduce(p.s.mixp, tok, c.lane), p.w.hc_scale, p.w.hc_base, p.ffn_pre, p.ffn_post, p.ffn_comb);
}

__device__ __forceinline__ float (&lo4(float (&a)[8]))[4] { return reinterpret_cast<float(&)[4]>(a[0]); }

__device__ __forceinline__ float (&hi4(float (&a)[8]))[4] { return reinterpret_cast<float(&)[4]>(a[4]); }

__device__ __forceinline__ Span entry(Ctx& c, const Job& job) {
  const int* routes = c.p.s.table->routes + job.f;
  c.sync();
  if (c.tid < job.n) c.sm.routes[c.tid] = __ldcg(routes + c.tid);
  return span_of(0, job.n);
}

template <class Store>
__device__ __forceinline__ void routed(Ctx& c, const Span& en, int ks, int kw, Store store) {
  const int g = c.lane >> 2, t = c.lane & 3;
  float acc[GW][8];
  gemm<8>(
      c, en, ks, acc, [&](int tt, int k) { return load_q(c, ks, kw, tt, k); },
      [&](float (&a)[8], const uint8_t* stage, int q, int rt, const Bq& b, int s) {
        const unsigned* words = stage_words(stage);
        const int idx = rt * 32 + t * 8 + g;
        const unsigned sa = (s * 4 + 4 <= ks ? words[idx] : unsigned(reinterpret_cast<const unsigned short*>(words)[idx])) + 0x06060606u;
        mma_fp4(q, lo4(a), hi4(a), frag(stage, q * 4 + rt, c.lane), __byte_perm(b.x.x, b.x.y, 0x6420), __byte_perm(b.x.x, b.x.y, 0x7531), sa, b.s);
      });
  stores(c, en, acc, [&](int rt, int tt, const float (&a)[8]) {
    const float4 lo = transpose4(a, c.lane), hi = transpose4(a + 4, c.lane);
    const int i = quad_col(tt, c.lane), row = rt * 32 + quad_row(c.lane);
    if (i < en.count) {
      store(c.sm.routes[i], row, lo);
      store(c.sm.routes[i], row + 16, hi);
    }
  });
}

__device__ void finish_r1(Ctx& c, int ei, int bb, const Span& en) {
  const Params& p = c.p;
  for (int item = c.warp; item < en.count * 2; item += GW) {
    const int route = c.sm.routes[item / 2], col = bb * 64 + (item & 1) * 32 + c.lane;
    float* h = p.s.hacc + long(route) * 2 * INTER + col;
    const float g = __ldcg(h), u = __ldcg(h + INTER);
    __stcg(h, 0.0f);
    __stcg(h + INTER, 0.0f);
    const float a = swiglu(g, u);
    const int e = quant_exp(warp_max(fabsf(a)), AMAX_FLOOR);
    p.s.hq[long(route) * INTER + col] = quant_byte(a, e);
    if (c.lane == 0) p.s.hqe[route * HQ_LD + col / 32] = uint8_t(e + 127);
  }
  c.signal(C_R1D + ei);
}

__device__ void job_r1(Ctx& c, const Job& job) {
  constexpr int KS = SLICES / R1_PARTS;
  const Params& p = c.p;
  const int ei = job.a, bb = job.b, kp = job.c;
  const Span en = entry(c, job);
  fill_q(
      c, en.count, KS, KS / 4, [&](int i) { return p.s.xq + long(c.sm.routes[i] / TOPK) * HID + kp * KS * 32; },
      [&](int i) { return p.s.xse + (c.sm.routes[i] / TOPK) * SLICES + kp * KS; });
  routed(c, en, KS, KS / 4, [&](int route, int r, float4 v) { red4(p.s.hacc + long(route) * 2 * INTER + (r < 64 ? 0 : INTER - 64) + bb * 64 + r, v); });
  if (c.arrive(C_R1A + ei * R1_BLOCKS + bb, R1_PARTS)) finish_r1(c, ei, bb, en);
}

__device__ void job_r2(Ctx& c, const Job& job) {
  const Params& p = c.p;
  const int rb = job.b;
  const Span en = entry(c, job);
  fill_q(
      c, en.count, KS_INTER, KW_INTER, [&](int i) { return p.s.hq + long(c.sm.routes[i]) * INTER; }, [&](int i) { return p.s.hqe + c.sm.routes[i] * HQ_LD; });
  routed(c, en, KS_INTER, KW_INTER, [&](int route, int r, float4 v) {
    const float w = __ldcg(p.s.topk_w + route);
    red4(p.s.yracc + long(route / TOPK) * HID + rb * 128 + r, make_float4(w * v.x, w * v.y, w * v.z, w * v.w));
  });
  c.signal(C_SF + rb);
}

__device__ void job_xf(Ctx& c, int k) {
  const Params& p = c.p;
  c.release();
  __nv_bfloat16* mine = inbox(p, p.rank, 1, 0, c.epoch);
  constexpr int XU = 2;
  for (int base = 0; base < p.m * CHUNK_VECS; base += GT * XU) {
    Acc8 rv[XU], sv[XU];
#pragma unroll
    for (int u = 0; u < XU; ++u) {
      const int i = base + c.tid + u * GT;
      const bool live = i < p.m * CHUNK_VECS;
      const long off = chunk_off(live ? i : 0, k);
      rv[u] = load8(p.s.yracc + off, live);
      sv[u] = load8(p.s.shacc + off, live);
    }
#pragma unroll
    for (int u = 0; u < XU; ++u) {
      const int i = base + c.tid + u * GT;
      if (i >= p.m * CHUNK_VECS) break;
      const long off = chunk_off(i, k);
      clear8(p.s.yracc + off);
      clear8(p.s.shacc + off);
      float routed[8], shared[8], y[8];
      unpack8(rv[u], routed);
      unpack8(sv[u], shared);
#pragma unroll
      for (int e = 0; e < 8; ++e) shared[e] = bf(shared[e]);
#pragma unroll
      for (int e = 0; e < 8; ++e) y[e] = bf(routed[e]) + shared[e];
      *reinterpret_cast<uint4*>(mine + off) = pack8(y);
    }
  }
  share(c, 1, k, p.y_ffn);
}

__device__ void deps(Ctx& c, const Job& job) {
  const Params& p = c.p;
  switch (job.type) {
    case J_WOB: return c.wait(C_Z + job.b * GROUPS + job.c, KS_WOB / 2);
    case J_XA: return c.wait(C_YC + job.a, 2 * c.L.tg);
    case J_OWNA:
      if (p.attn_mix) c.wait(C_AMIX, quads(p.m));
      return c.wait(C_YR + job.a / (CHUNK_COLS / 32), 1);
    case J_XNORM: return c.wait(C_A + job.b, SLICES);
    case J_GATE:
      if (!split_norm(p.m))
        for (int t = 0; t < tiles_of(p.m); ++t) c.wait(C_A + t, SLICES);
      return;
    case J_ROUTE: return c.wait(C_G + job.a / OTOK, GATE_BLOCKS * KP_GATE);
    case J_SH1: {
      const int k0 = job.b * KS_SH1 * 32;
      const unsigned target = split_norm(p.m) ? group_of(p.m, job.c).tiles : 512 / (KS_GATE * 16);
      for (int part = k0 / 512; part < (k0 + KS_SH1 * 32 + 511) / 512; ++part) c.wait(C_XK + part * GROUPS + job.c, target);
      return;
    }
    case J_SH2: return c.wait(C_HS + job.c, SH1_BLOCKS);
    case J_MIX: return c.wait(C_A + job.a * GW / TILE, SLICES);
    case J_R2: return c.wait(C_R1D + job.a, R1_BLOCKS);
    case J_XF: return c.wait(C_SF + job.a, __ldcg(p.s.counters + C_NENT) + 2 * c.L.tg);
    default: return;
  }
}

__device__ void run_job(Ctx& c, const Job& job) {
  switch (job.type) {
    case J_WOA: return job_woa(c, job.a, job.b, job.c, job.stages);
    case J_AMIX: return job_amix(c, job.a);
    case J_WOB: return job_wob(c, job.a, job.b, job.c);
    case J_XA: return job_xa(c, job.a);
    case J_OWNA: return job_owna(c, job.a, job.b);
    case J_XNORM: return job_xnorm(c, job.a, job.b);
    case J_GATE: return job_gate(c, job.a, job.b, job.c);
    case J_ROUTE: return job_route(c, job.a);
    case J_SH1: return job_sh1(c, job.a, job.b, job.c);
    case J_SH2: return job_sh2(c, job.a, job.c);
    case J_MIX: return job_mix(c, job.a);
    case J_R1: return job_r1(c, job);
    case J_R2: return job_r2(c, job);
    default: return job_xf(c, job.a);
  }
}

__device__ int route_table(const Params& p, int lane) {
  int n = 0;
  if (lane == 0) {
    while (ld_acquire_gpu(p.s.counters + C_ROUTE) < unsigned(p.m)) __nanosleep(SPIN_NS);
    n = int(__ldcg(p.s.counters + C_NENT));
  }
  return __shfl_sync(0xffffffffu, n, 0);
}

struct Issue {
  uint8_t* stage;
  int lane;
  unsigned long long policy;

  __device__ __forceinline__ void block(const uint8_t* src) const {
#pragma unroll
    for (int f = 0; f < FRAGS; ++f) cp_async16_hint(stage + f * FRAG + lane * 16, src + f * FRAG + lane * 16, policy);
  }

  __device__ __forceinline__ void frag(int f, const uint4* src) const { cp_async16_hint(stage + f * FRAG + lane * 16, src + lane, policy); }

  __device__ __forceinline__ void scales(const uint8_t* src, long ld) const {
    if (lane < 2) cp_async4(stage + SCALE_OFF + lane * 4, src + lane * ld);
  }

  __device__ __forceinline__ void words(int offset, const unsigned* src, int bytes) const {
    for (int o = lane * 16; o < bytes; o += 32 * 16) cp_async16(stage + SCALE_OFF + offset + o, reinterpret_cast<const uint8_t*>(src) + o);
  }
};

__device__ void issue(const Params& p, const Job& job, int s, const Issue& is) {
  const Weights& w = p.w;
  switch (job.type) {
    case J_WOA: {
      const int kg = job.b * job.stages + s;
      is.block(w.woa + (long(job.a) * KG_WOA + kg) * FRAGS * FRAG);
      is.scales(w.woa_s + long(job.a * 2) * (OAK / 32) + kg * 4, OAK / 32);
      return;
    }
    case J_WOB: {
      const int kg = job.b * (KS_WOB / 4) + s;
      is.block(w.wob + (long(job.a) * KG_WOB + kg) * FRAGS * FRAG);
      is.scales(w.wob_s + long(job.a * 2) * (OA / 32) + kg * 4, OA / 32);
      return;
    }
    case J_GATE: is.block(w.gate + (long(job.a) * KG_GATE + job.b * (KS_GATE / 4) + s) * FRAGS * FRAG); return;
    case J_SH1: {
      const int kg = job.b * (KS_SH1 / 4) + s;
      is.block(w.sh13 + (long(job.a) * KG_SH1 + kg) * FRAGS * FRAG);
      is.scales(w.sh13_s + long(job.a * 2) * SLICES + kg * 4, SLICES);
      return;
    }
    case J_SH2:
      is.block(w.sh2 + (long(job.a) * INTER_STAGES + s) * FRAGS * FRAG);
      is.scales(w.sh2_s + long(job.a * 2) * SH2_LD + s * 4, SH2_LD);
      return;
    case J_OWNA:
      for (int i = is.lane; i < (s ? MIX - OWNA_ROWS : OWNA_ROWS) * HC * 8; i += 32) {
        const int piece = s * OWNA_ROWS * HC + i / 8, part = i % 8, r = piece / HC, hc = piece % HC;
        cp_async16(is.stage + (i / 8) * 128 + part * 16, w.fn + long(r) * HCD + hc * HID + job.a * 32 + part * 4);
      }
      return;
    case J_R1: {
      const int e = job.e, g4 = job.c * 4 + s, nt = job.b / 2, rows = nt < INTER / 128 ? 128 : INTER % 128, n0 = (job.b % 2) * 2;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int half = 1 - h;
        const uint4* base = w.w13 + e * w.w13_stride + long(half) * (INTER * HID / 32) + long(nt) * (128 * HID / 32) + long(g4) * rows * 4 + n0 * 32;
        for (int q = 0; q < 4; ++q) {
          is.frag(q * 4 + h * 2, base + q * rows);
          is.frag(q * 4 + h * 2 + 1, base + q * rows + 32);
        }
        is.words(h * 256, w.s13 + e * w.s13_stride + long(half) * (INTER * SLICES / 4) + long(nt) * (128 * SLICES / 4) + n0 * 32 + long(g4) * rows, 256);
      }
      return;
    }
    case J_R2: {
      const int e = job.e, kq = s == INTER / 128 ? (INTER % 128) / 32 : 4;
      const uint4* base = w.w2 + e * w.w2_stride + long(job.b) * (128 * INTER / 32) + s * 512;
      for (int q = 0; q < kq; ++q)
#pragma unroll
        for (int rt = 0; rt < 4; ++rt) is.frag(q * 4 + rt, base + q * 128 + rt * 32);
      is.words(0, w.s2 + e * w.s2_stride + long(job.b) * (128 * INTER / 32 / 4) + s * 128, kq * 128);
      return;
    }
    default: return;
  }
}

__device__ void producer(const Params& p, Smem& sm) {
  const int lane = threadIdx.x & 31;
  const Layout L = layout(p.m);
  const unsigned long long policy = evict_first_policy();
  const volatile int* ready = &sm.ready;
  int n = -1, it = 0, j = int(blockIdx.x);
  for (int seq = 0;; ++seq) {
    if (n < 0 && j >= L.routed) n = route_table(p, lane);
    Job job = n < 0 || j < total_jobs(L, n) ? decode(L, n, j) : Job{J_END, 0, 0, 0, 1};
    if (job.type == J_R1 || job.type == J_R2) {
      const int e = __ldcg(p.s.table->expert + job.a), ch = __ldcg(p.s.table->chunk + job.a);
      job.e = e;
      job.f = e * MAXM + ch * SPAN;
      job.n = min(SPAN, int(__ldcg(p.s.counters + C_ECOUNT + e)) - ch * SPAN);
    }
    bool claimed = false;
    unsigned next = 0;
    for (int s = 0; s < job.stages; ++s, ++it) {
      const int slot = it % STAGES;
      mbar_wait(&sm.empty[slot], ((it / STAGES) & 1) ^ 1);
      issue(p, job, s, Issue{sm.ring[slot], lane, policy});
      if (lane == 0 && s == 0) sm.job[slot] = job;
      cp_async_arrive(&sm.full[slot]);
      if (lane == 0) {
        mbar_arrive(&sm.full[slot]);
        if (!claimed && seq > 0 && job.type != J_END && *ready >= seq) {
          next = atomicAdd(p.s.counters + C_CLAIM, 1u);
          claimed = true;
        }
      }
    }
    if (job.type == J_END) return;
    if (seq == 0) pdl_wait();
    if (lane == 0 && !claimed) {
      while (*ready < seq) __nanosleep(32);
      next = atomicAdd(p.s.counters + C_CLAIM, 1u);
    }
    j = int(gridDim.x) + int(__shfl_sync(0xffffffffu, next, 0));
  }
}

__global__ void __launch_bounds__(THREADS, PER_SM) layer_kernel(Params p) {
  extern __shared__ __align__(128) uint8_t raw[];
  Smem& sm = *reinterpret_cast<Smem*>(raw);
  for (int i = threadIdx.x; i < CACHED; i += THREADS) sm.seen[i] = 0;
  if (threadIdx.x == 0) {
    sm.ready = -1;
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&sm.full[s], 33);
      mbar_init(&sm.empty[s], GW);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x >= GT) {
    producer(p, sm);
  } else {
    const int tid = threadIdx.x;
    Ctx c{p, sm, layout(p.m), tid >> 5, tid & 31, tid, 0, 0u};
    pdl_wait();
    c.epoch = ld_volatile(p.s.epoch) + 1;
    for (int seq = 0;; ++seq) {
      c.acquire();
      const Job job = sm.job[c.it % STAGES];
      if (job.type == J_END) break;
      deps(c, job);
      const bool exchange = job.type == J_XA || job.type == J_XF;
      if (tid == 0 && !exchange) *reinterpret_cast<volatile int*>(&sm.ready) = seq;
      run_job(c, job);
      if (tid == 0 && exchange) *reinterpret_cast<volatile int*>(&sm.ready) = seq;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    sm.last = atomicAdd(p.s.counters + C_FIN, 1u) == gridDim.x - 1;
  }
  __syncthreads();
  if (!sm.last) return;
  __threadfence();
  for (int i = threadIdx.x; i < COUNTERS; i += THREADS) p.s.counters[i] = 0;
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    atomicAdd(p.s.epoch, 1u);
    __threadfence();
  }
}

struct AttnIn {
  const __nv_bfloat16* y;
  const __nv_bfloat16* resid;
  const float* post;
  const float* comb;
  const float* pre;
  const float* fn;
  const __nv_bfloat16* norm;
  __nv_bfloat16* resid_out;
  __nv_bfloat16* input_out;
  int m;
  const float* mix_scale = nullptr;
  const float* mix_base = nullptr;
  float* post_out = nullptr;
  float* comb_out = nullptr;
  float* pre_out = nullptr;
  float quant_floor = MX_AMAX_FLOOR;
};

constexpr int QR = 1280, KVW = 512, QKV = QR + KVW, QKV_TILES = QKV / 8, Q_TILES = QR / 8;
constexpr int KSPLIT = 5, KPART = HID / KSPLIT, WQB_ROWS = 16 * HDIM, WQB_TILES = WQB_ROWS / 8;
constexpr int NORM_COLS = GT, Q_CHUNKS = QR / NORM_COLS, KV_CHUNKS = KVW / NORM_COLS;
constexpr int QKV_CTAS = KSPLIT * cdiv(QKV_TILES, GW), WQB_CTAS = WQB_TILES / GW;
constexpr int P1 = 0, P3 = 1, P3Q = 2, P3K = 3, P4 = 4, PFIN = 5, PCOUNTERS = 8;
constexpr int PT = 16, PRO_FN = MIX * HC * 32 * 4, PRO_XQ = PT * (QR + 16), PRO_XS = PT * (QR / 32);
constexpr int PRO_SMEM = (PRO_FN > PRO_XQ ? PRO_FN : PRO_XQ) + PRO_XS;

__host__ __device__ __forceinline__ int act_stride(int m) { return m <= 32 ? 512 : 16 * MAXM; }
__host__ __device__ __forceinline__ unsigned act_scale(int grp, int tok, int m) { return (grp >> 2) * act_stride(m) + tok * 16 + (grp & 3); }
__host__ __device__ __forceinline__ int slice_tokens(int m) { return m <= TILE ? GW : 4 * GW; }
__host__ __device__ __forceinline__ int input_tokens(int m) { return m <= TILE ? 2 * GW : 8 * GW; }
__host__ __device__ __forceinline__ int norm_span(int m) { return m < 32 ? 1 : m / 16; }

struct Prologue {
  AttnIn in;
  const uint8_t* wqa;
  const uint8_t* wqa_s;
  const __nv_bfloat16* q_norm;
  const __nv_bfloat16* kv_norm;
  float eps;
  __nv_bfloat16* qr;
  __nv_bfloat16* kv;
  uint8_t* qr_q;
  uint8_t* qr_s;
  const uint8_t* wqb;
  const uint8_t* wqb_s;
  const long* positions;
  const float* cos_sin;
  __nv_bfloat16* q;
  uint8_t* swa;
  long swa_stride;
  long swa_capacity;
  const long* swa_slots;
  int swa_page;
  Scratch s;
};

constexpr int SWA_RECORD = 528;

__device__ __forceinline__ void write_swa(const Prologue& a, int tok, int col, float y, int lane) {
  const long pos = __ldg(a.positions + tok);
  const float partner = __shfl_xor_sync(0xffffffffu, y, 1);
  float r = y;
  if (col >= KVW - ROPE) {
    const int local = col - (KVW - ROPE);
    const float2 cs = rope_cs(a.cos_sin + pos * ROPE, local / 2);
    r = local % 2 ? rot_add(r, partner, cs) : rot_sub(r, partner, cs);
  }
  if (pos < 0) r = 0.0f;
  r = bf(r);
  const int e = quant_exp(warp_max(fabsf(r)), AMAX_FLOOR);
  float inv;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(pow2(e)));
  const long slot = __ldg(a.swa_slots + tok);
  if (slot < 0 || slot >= a.swa_capacity) return;
  uint8_t* rec = a.swa + (slot / a.swa_page) * a.swa_stride + (slot % a.swa_page) * SWA_RECORD;
  rec[col] = uint8_t(e4m3x2(r * inv, 0.0f) & 0xffu);
  if (lane == 0) rec[KVW + col / 32] = uint8_t(e + 127);
}

template <int PERW>
__device__ __forceinline__ void attn_slice(const AttnIn& a, const Scratch& sc, float* fn, int s, int tid, int tok0, int tok_end) {
  const int warp = tid >> 5, lane = tid & 31, col = s * 32 + lane, end = min(a.m, tok_end);
  __syncthreads();
  for (int i = tid; i < MIX * HC * 32; i += GT) fn[i] = __ldg(a.fn + long(i / 32) * HID + s * 32 + i % 32);
  __syncthreads();
  float y[PERW], rk[PERW][HC];
#pragma unroll
  for (int u = 0; u < PERW; ++u) {
    const int tok = min(tok0 + warp + u * GW, a.m - 1);
    y[u] = __bfloat162float(__ldg(a.y + long(tok) * HID + col));
#pragma unroll
    for (int k = 0; k < HC; ++k) rk[u][k] = __bfloat162float(__ldg(a.resid + (long(tok) * HC + k) * HID + col));
  }
#pragma unroll
  for (int u = 0; u < PERW; ++u) {
    const int tok = tok0 + warp + u * GW;
    if (tok >= end) break;
    float r[HC], x = 0.0f, v[32] = {};
#pragma unroll
    for (int j = 0; j < HC; ++j) {
      float t = __ldg(a.post + tok * HC + j) * y[u];
#pragma unroll
      for (int k = 0; k < HC; ++k) t += __ldg(a.comb + tok * HC * HC + k * HC + j) * rk[u][k];
      r[j] = bf(t);
      a.resid_out[(long(tok) * HC + j) * HID + col] = __float2bfloat16(r[j]);
      x += __ldg(a.pre + tok * HC + j) * r[j];
      v[MIX] += r[j] * r[j];
    }
    x = bf(x);
    sc.xc[long(tok) * HID + col] = __float2bfloat16(x);
#pragma unroll
    for (int q = 0; q < MIX; ++q)
#pragma unroll
      for (int j = 0; j < HC; ++j) v[q] += fn[(q * HC + j) * 32 + lane] * r[j];
    v[MIX + 1] = x * x;
    const float total = lane_sums(v, lane);
    sc.mixp[(long(tok) * SLICES + s) * MIXW + lane] = total;
    if (lane == MIX + 1) sc.xsq[tok * SLICES + s] = total;
  }
}

__device__ __forceinline__ void attn_reduce(const AttnIn& a, const Scratch& sc, int tok, int lane) {
  const float mine = mix_reduce(sc.mixp, tok, lane);
  if (tok >= a.m) return;
  if (a.mix_scale) {
    mix_finish(lane, tok, mine, a.mix_scale, a.mix_base, a.pre_out, a.post_out, a.comb_out);
    return;
  }
  if (lane <= MIX) sc.mixq[tok * (MIX + 1) + lane] = mine;
}

__device__ __forceinline__ float token_rs(const float* xsq, int tok, int lane) {
  constexpr int V = SLICES / 4;
  const float4* x = reinterpret_cast<const float4*>(xsq + tok * SLICES);
  float v = 0.0f;
#pragma unroll
  for (int i = 0; i < cdiv(V, 32); ++i)
    if (lane + i * 32 < V) {
      const float4 f = __ldcg(x + lane + i * 32);
      v += (f.x + f.y) + (f.z + f.w);
    }
  return rsqrtf(warp_sum(v) / HID + NORM_EPS);
}

template <int PERW>
__device__ __forceinline__ void attn_input(const AttnIn& a, const Scratch& sc, int s, int tok0, int tid) {
  const int warp = tid >> 5, lane = tid & 31, col = s * 32 + lane;
  const float nw = __bfloat162float(__ldg(a.norm + col));
  float xv[PERW], rs[PERW];
#pragma unroll
  for (int u = 0; u < PERW; ++u) {
    const int tok = min(tok0 + warp + u * GW, a.m - 1);
    xv[u] = __bfloat162float(__ldcg(sc.xc + long(tok) * HID + col));
    rs[u] = token_rs(sc.xsq, tok, lane);
  }
#pragma unroll
  for (int u = 0; u < PERW; ++u) {
    const int tok = tok0 + warp + u * GW;
    if (tok >= a.m) break;
    const float v = bf(nw * (xv[u] * rs[u]));
    a.input_out[long(tok) * HID + col] = __float2bfloat16(v);
    const int e = quant_exp(warp_max(fabsf(v)), a.quant_floor);
    sc.aq[long(tok) * HID + col] = quant_byte(v, e);
    if (lane == 0) sc.as[act_scale(s, tok, a.m)] = uint8_t(e + 127);
  }
}

__device__ __forceinline__ void stage_mx(uint8_t* xq, int pitch, uint8_t* xs, const uint8_t* q, int q_ld, const uint8_t* scales, int k0, int width, int t0, int m, int tid) {
  for (int v = tid; v < PT * width / 16; v += GT) {
    const int r = v / (width / 16), c = (v % (width / 16)) * 16, tok = t0 + r;
    *reinterpret_cast<uint4*>(xq + r * pitch + c) = tok < m ? __ldcg(reinterpret_cast<const uint4*>(q + long(tok) * q_ld + k0 + c)) : make_uint4(0, 0, 0, 0);
  }
  for (int i = tid; i < PT * (width / 32); i += GT) {
    const int r = i / (width / 32), grp = i % (width / 32), tok = t0 + r;
    xs[i] = tok < m ? scales[act_scale(k0 / 32 + grp, tok, m)] : uint8_t(127);
  }
}

__device__ __forceinline__ void store_q(const Prologue& a, int tile, int tok, float v0, float v1, int t) {
  if (tok >= a.in.m) return;
  const int r = tile * 8 + 2 * t, d = r % HDIM;
  const long pos = a.positions ? __ldg(a.positions + tok) : 0;
  v0 = bf(v0), v1 = bf(v1);
  if (a.cos_sin && d >= HDIM - ROPE) {
    const float2 cs = rope_cs(a.cos_sin + pos * ROPE, (d - (HDIM - ROPE)) / 2);
    const float e = rot_sub(v0, v1, cs), o = rot_add(v1, v0, cs);
    v0 = e, v1 = o;
  }
  if (pos < 0) v0 = v1 = 0.0f;
  *reinterpret_cast<__nv_bfloat162*>(a.q + long(tok) * WQB_ROWS + r) = __floats2bfloat162_rn(v0, v1);
}

template <bool KV, int B>
__device__ __forceinline__ void norm_batch(const Prologue& a, int col, float nw, int t0, int end, int lane) {
  constexpr int N = KV ? KVW : QR, FIRST = KV ? Q_TILES : 0, TILES = KV ? QKV_TILES - Q_TILES : Q_TILES, PER = cdiv(TILES, 32);
  const int m = a.in.m;
  float part[B][PER], v[B];
#pragma unroll
  for (int u = 0; u < B; ++u) {
    const int tok = min(t0 + u, end - 1);
#pragma unroll
    for (int i = 0; i < PER; ++i) part[u][i] = lane + i * 32 < TILES ? __ldcg(a.s.qsq + (FIRST + lane + i * 32) * MAXM + tok) : 0.0f;
    v[u] = __ldcg(a.s.qkv + tok * QKV + (KV ? QR : 0) + col);
  }
#pragma unroll
  for (int u = 0; u < B; ++u) {
    const int tok = t0 + u;
    if (tok >= end) break;
    float total = 0.0f;
#pragma unroll
    for (int i = 0; i < PER; ++i) total += part[u][i];
    const float scale = rsqrtf(warp_sum(total) / N + a.eps);
    const __nv_bfloat16 y = __float2bfloat16(v[u] * scale * nw);
    (KV ? a.kv : a.qr)[long(tok) * N + col] = y;
    if (KV && a.swa) write_swa(a, tok, col, __bfloat162float(y), lane);
    if (!KV) {
      const float f = __bfloat162float(y);
      const int e = quant_exp(warp_max(fabsf(f)), a.in.quant_floor);
      a.qr_q[long(tok) * QR + col] = quant_byte(f, e);
      if (lane == 0) a.qr_s[act_scale(col / 32, tok, m)] = uint8_t(e + 127);
    }
  }
}

template <bool KV>
__device__ __forceinline__ void norm_tokens(const Prologue& a, int chunk, int t0, int n, int tid) {
  const int lane = tid & 31, col = chunk * NORM_COLS + tid, end = min(a.in.m, t0 + n);
  const float nw = __bfloat162float((KV ? a.kv_norm : a.q_norm)[col]);
  if (n == 1) return norm_batch<KV, 1>(a, col, nw, t0, end, lane);
#pragma unroll 1
  for (int b = t0; b < end; b += 4) norm_batch<KV, 4>(a, col, nw, b, end, lane);
}

__global__ void __launch_bounds__(GT) prologue_kernel(Prologue a) {
  extern __shared__ __align__(16) uint8_t raw[];
  float* fn = reinterpret_cast<float*>(raw);
  uint8_t* xq = raw;
  uint8_t* xs = raw + (PRO_FN > PRO_XQ ? PRO_FN : PRO_XQ);
  __shared__ int last;
  pdl_launch_dependents();
  pdl_wait();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3, m = a.in.m;
  unsigned* pc = a.s.pc;
  auto wait = [&](int index, unsigned target) {
    if (tid == 0)
      while (ld_acquire_gpu(pc + index) < target) {
      }
    __syncthreads();
  };
  auto signal = [&](int index) {
    __syncthreads();
    if (tid == 0) {
      red_release(pc + index, 1u);
    }
  };

  const int slice_jobs = SLICES * cdiv(m, slice_tokens(m)), reduce_jobs = cdiv(m, GW), input_jobs = SLICES * cdiv(m, input_tokens(m)), span = norm_span(m), spans = cdiv(m, span);
  for (int job = blockIdx.x; job < slice_jobs; job += gridDim.x) {
    const int s = job % SLICES, tok0 = job / SLICES * slice_tokens(m);
    if (m <= TILE) attn_slice<1>(a.in, a.s, fn, s, tid, tok0, tok0 + slice_tokens(m));
    else attn_slice<4>(a.in, a.s, fn, s, tid, tok0, tok0 + slice_tokens(m));
    signal(P1);
  }
  for (int job = blockIdx.x; job < input_jobs; job += gridDim.x) {
    wait(P1, slice_jobs);
    const int s = job % SLICES, tok0 = job / SLICES * input_tokens(m);
    if (m <= TILE) attn_input<2>(a.in, a.s, s, tok0, tid);
    else attn_input<8>(a.in, a.s, s, tok0, tid);
    signal(P3);
  }
  for (int job = gridDim.x - 1 - blockIdx.x; job < reduce_jobs; job += gridDim.x) {
    wait(P1, slice_jobs);
    attn_reduce(a.in, a.s, job * GW + warp, lane);
  }
  if (blockIdx.x < QKV_CTAS) {
    const int ks = blockIdx.x % KSPLIT, tile = (blockIdx.x / KSPLIT) * GW + warp;
    Rows<HID, KPART / 64, 1> rows;
    if (tile < QKV_TILES) rows.load(a.wqa, a.wqa_s, tile, ks * (KPART / 64), lane);
    wait(P3, input_jobs);
    if (tile < QKV_TILES) rows.ready(lane);
    for (int t0 = 0; t0 < m; t0 += PT) {
      stage_mx(xq, KPART + 16, xs, a.s.aq, HID, a.s.as, ks * KPART, KPART, t0, m, tid);
      __syncthreads();
      if (tile < QKV_TILES) {
        float acc[4] = {};
        rows.accumulate<PT>(xq, KPART + 16, xs, KPART / 32, ks * (KPART / 64), ks * (KPART / 32), lane, acc);
        float* part = a.s.qkp + (long(ks) * MAXM + t0 + g) * QKV + tile * 8 + 2 * t;
        *reinterpret_cast<float2*>(part) = make_float2(acc[0], acc[1]);
        *reinterpret_cast<float2*>(part + 8 * QKV) = make_float2(acc[2], acc[3]);
      }
      __syncthreads();
    }
    if (tile < QKV_TILES) {
      __syncwarp();
      unsigned arrived = 0;
      if (lane == 0) {
        __threadfence();
        arrived = atomicAdd(a.s.tiles + tile, 1u);
      }
      if (__shfl_sync(0xffffffffu, arrived, 0) == KSPLIT - 1) {
        __threadfence();
        for (int base = 0; base < m; base += TILE) {
          const int tok = base + g;
          float v[2] = {};
#pragma unroll
          for (int k = 0; k < KSPLIT; ++k) {
            const float2 p = __ldcg(reinterpret_cast<const float2*>(a.s.qkp + (long(k) * MAXM + tok) * QKV + tile * 8 + 2 * t));
            v[0] += p.x, v[1] += p.y;
          }
          v[0] = bf(v[0]), v[1] = bf(v[1]);
          if (tok < m) *reinterpret_cast<float2*>(a.s.qkv + tok * QKV + tile * 8 + 2 * t) = make_float2(v[0], v[1]);
          const float sq = quad_sum(v[0] * v[0] + v[1] * v[1]);
          if (t == 0 && tok < m) a.s.qsq[tile * MAXM + tok] = sq;
        }
        __syncwarp();
        if (lane == 0) {
          a.s.tiles[tile] = 0;
          red_release(pc + (tile < Q_TILES ? P3Q : P3K), 1u);
        }
      }
    }
  }
  for (int job = blockIdx.x; job < (Q_CHUNKS + KV_CHUNKS) * spans; job += gridDim.x) {
    const int chunk = job / spans, t0 = job % spans * span, kv = chunk >= Q_CHUNKS;
    wait(kv ? P3K : P3Q, kv ? QKV_TILES - Q_TILES : Q_TILES);
    if (kv) norm_tokens<true>(a, chunk - Q_CHUNKS, t0, span, tid);
    else norm_tokens<false>(a, chunk, t0, span, tid);
    if (!kv) signal(P4);
  }
  if (blockIdx.x < WQB_CTAS) {
    const int tile = blockIdx.x * GW + warp;
    Rows<QR, QR / 64, 1> rows;
    rows.load(a.wqb, a.wqb_s, tile, 0, lane);
    wait(P4, Q_CHUNKS * spans);
    rows.ready(lane);
    for (int t0 = 0; t0 < m; t0 += PT) {
      stage_mx(xq, QR + 16, xs, a.qr_q, QR, a.qr_s, 0, QR, t0, m, tid);
      __syncthreads();
      float acc[4] = {};
      rows.accumulate<PT>(xq, QR + 16, xs, QR / 32, 0, 0, lane, acc);
      store_q(a, tile, t0 + g, acc[0], acc[1], t);
      store_q(a, tile, t0 + g + 8, acc[2], acc[3], t);
      __syncthreads();
    }
  }
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    last = atomicAdd(pc + PFIN, 1u) == gridDim.x - 1;
  }
  __syncthreads();
  if (last && tid < PCOUNTERS) pc[tid] = 0;
}

struct Layer {
  Weights w;
  long image_lo;
  std::vector<torch::Tensor> keep;
};

struct Engine {
  Peers peers;
  Scratch s;
  int rank;
  int ctas;
  std::vector<Layer> layers;
};

std::vector<Engine> engines;

template <class T>
long row_words(const torch::Tensor& t) {
  return t.stride(0) * t.element_size() / long(sizeof(T));
}

}

int64_t flag_bytes() { return FLAG_BYTES; }

int64_t inbox_bytes() { return 2L * 2 * 2 * PUB_SLOT * sizeof(__nv_bfloat16); }

int64_t act_scale_bytes(int64_t m) { return QR / 128 * act_stride(int(m)); }

int64_t create(std::vector<int64_t> buffers, int64_t rank) {
  TORCH_CHECK(buffers.size() == RANKS);
  Engine e{};
  for (int r = 0; r < RANKS; ++r) {
    e.peers.flags[r] = reinterpret_cast<unsigned*>(buffers[r]);
    e.peers.pub[r] = reinterpret_cast<__nv_bfloat16*>(buffers[r] + FLAG_BYTES);
  }
  e.rank = int(rank);
  e.s.epoch = zeroed<unsigned>(1);
  e.s.counters = zeroed<unsigned>(COUNTERS);
  e.s.table = zeroed<Table>(1);
  e.s.zacc = zeroed<float>(long(MAXM) * OA);
  e.s.zq = zeroed<uint8_t>(long(MAXM) * OA);
  e.s.zse = zeroed<uint8_t>(long(MAXM) * 64);
  e.s.yacc = zeroed<float>(long(MAXM) * HID);
  e.s.yattn = zeroed<__nv_bfloat16>(long(MAXM) * HID);
  e.s.mixp = zeroed<float>(long(SLICES) * MAXM * MIXW);
  e.s.mixq = zeroed<float>(long(MAXM) * (MIX + 1));
  e.s.xc = zeroed<__nv_bfloat16>(long(MAXM) * HID);
  e.s.xsq = zeroed<float>(long(SLICES) * MAXM);
  e.s.xbf = zeroed<__nv_bfloat16>(long(MAXM) * HID);
  e.s.xq = zeroed<uint8_t>(long(MAXM) * HID);
  e.s.xse = zeroed<uint8_t>(long(MAXM) * SLICES);
  e.s.gacc = zeroed<float>(long(MAXM) * EXPERTS);
  e.s.topk_w = zeroed<float>(ROUTES);
  e.s.sacc = zeroed<float>(long(MAXM) * 2 * INTER);
  e.s.hsq = zeroed<uint8_t>(long(MAXM) * INTER);
  e.s.hse = zeroed<uint8_t>(long(MAXM) * HQ_LD);
  e.s.shacc = zeroed<float>(long(MAXM) * HID);
  e.s.hacc = zeroed<float>(long(ROUTES) * 2 * INTER);
  e.s.hq = zeroed<uint8_t>(long(ROUTES) * INTER);
  e.s.hqe = zeroed<uint8_t>(long(ROUTES) * HQ_LD);
  e.s.yracc = zeroed<float>(long(MAXM) * HID);
  e.s.aq = zeroed<uint8_t>(long(MAXM) * HID);
  e.s.as = zeroed<uint8_t>(long(SLICES / 4) * 16 * MAXM);
  e.s.qkp = zeroed<float>(long(KSPLIT) * MAXM * QKV);
  e.s.qkv = zeroed<float>(long(MAXM) * QKV);
  e.s.qsq = zeroed<float>(long(QKV_TILES) * MAXM);
  e.s.tiles = zeroed<unsigned>(QKV_TILES);
  e.s.pc = zeroed<unsigned>(PCOUNTERS);
  int sms = 0, per = 0;
  TORCH_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, c10::cuda::current_device()) == cudaSuccess);
  opt_in_smem<layer_kernel>(sizeof(Smem));
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, layer_kernel, THREADS, sizeof(Smem)) == cudaSuccess);
  TORCH_CHECK(per >= PER_SM, "layer kernel fits ", per, " CTAs per SM, needs ", PER_SM);
  e.ctas = sms * PER_SM;
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  engines.push_back(e);
  return long(engines.size()) - 1;
}

int64_t add_layer(int64_t handle, torch::Tensor woa, torch::Tensor woa_s, torch::Tensor wob, torch::Tensor wob_s, torch::Tensor gate, torch::Tensor gate_b,
                  std::optional<torch::Tensor> bias_vl, torch::Tensor sh13, torch::Tensor sh13_s, torch::Tensor sh2, torch::Tensor sh2_s, torch::Tensor w13,
                  torch::Tensor s13, torch::Tensor w2, torch::Tensor s2, torch::Tensor fn, torch::Tensor hc_scale, torch::Tensor hc_base, torch::Tensor norm,
                  int64_t image_lo) {
  TORCH_CHECK(woa.numel() == long(OA) * OAK && sh2_s.size(-1) == SH2_LD);
  for (const auto& t : {woa_s, wob_s, sh13_s, sh2_s}) TORCH_CHECK(t.scalar_type() == at::kByte, "dense scales must be e8m0 bytes");
  Layer l{};
  l.w.woa = ptr<uint8_t>(woa);
  l.w.woa_s = ptr<uint8_t>(woa_s);
  l.w.wob = ptr<uint8_t>(wob);
  l.w.wob_s = ptr<uint8_t>(wob_s);
  l.w.gate = ptr<uint8_t>(gate);
  l.w.gate_b = ptr<float>(gate_b);
  l.w.bias_vl = bias_vl ? ptr<float>(*bias_vl) : nullptr;
  l.w.sh13 = ptr<uint8_t>(sh13);
  l.w.sh13_s = ptr<uint8_t>(sh13_s);
  l.w.sh2 = ptr<uint8_t>(sh2);
  l.w.sh2_s = ptr<uint8_t>(sh2_s);
  l.w.w13 = ptr<uint4>(w13);
  l.w.s13 = ptr<unsigned>(s13);
  l.w.w2 = ptr<uint4>(w2);
  l.w.s2 = ptr<unsigned>(s2);
  l.w.w13_stride = row_words<uint4>(w13);
  l.w.s13_stride = row_words<unsigned>(s13);
  l.w.w2_stride = row_words<uint4>(w2);
  l.w.s2_stride = row_words<unsigned>(s2);
  l.w.fn = ptr<float>(fn);
  l.w.hc_scale = ptr<float>(hc_scale);
  l.w.hc_base = ptr<float>(hc_base);
  l.w.norm = ptr<__nv_bfloat16>(norm);
  l.image_lo = image_lo;
  l.keep = {woa, woa_s, wob, wob_s, gate, gate_b, sh13, sh13_s, sh2, sh2_s, w13, s13, w2, s2, fn, hc_scale, hc_base, norm};
  if (bias_vl) l.keep.push_back(*bias_vl);
  auto& layers = engines.at(handle).layers;
  layers.push_back(std::move(l));
  return long(layers.size()) - 1;
}

void run(int64_t handle, int64_t layer, torch::Tensor o, torch::Tensor positions, std::optional<torch::Tensor> input_ids, torch::Tensor cos_sin, torch::Tensor resid0,
         torch::Tensor attn_post, torch::Tensor attn_comb, torch::Tensor attn_pre, std::optional<torch::Tensor> attn_scale, std::optional<torch::Tensor> attn_base,
         torch::Tensor y_ffn, torch::Tensor resid1, torch::Tensor ffn_post, torch::Tensor ffn_comb, torch::Tensor ffn_pre) {
  const auto& e = engines.at(handle);
  const auto& l = e.layers.at(layer);
  const int m = int(resid0.size(0));
  TORCH_CHECK(m >= 1 && m <= MAXM && o.stride(-1) == 1 && o.size(-1) == HDIM && o.stride(1) == HDIM && positions.scalar_type() == at::kLong);
  const bool ids = l.w.bias_vl && input_ids && input_ids->numel() && input_ids->is_contiguous();
  TORCH_CHECK(!ids || input_ids->scalar_type() == at::kLong || input_ids->scalar_type() == at::kInt);
  TORCH_CHECK(attn_scale.has_value() == attn_base.has_value());
  const c10::cuda::CUDAGuard guard(resid0.device());
  Params p{};
  p.peers = e.peers;
  p.s = e.s;
  p.w = l.w;
  p.rank = e.rank;
  p.m = m;
  p.o = data_as<const __nv_bfloat16>(o);
  p.o_stride = o.stride(0);
  p.positions = ptr<long>(positions);
  p.input_ids = ids ? ptr<void>(*input_ids) : nullptr;
  p.ids64 = ids && input_ids->scalar_type() == at::kLong;
  p.image_lo = l.image_lo;
  p.cos_sin = cos_sin.numel() ? ptr<float>(cos_sin) : nullptr;
  p.resid0 = ptr<__nv_bfloat16>(resid0);
  p.attn_post = mut<float>(attn_post);
  p.attn_comb = mut<float>(attn_comb);
  p.attn_pre = mut<float>(attn_pre);
  p.attn_mix = attn_scale.has_value();
  p.attn_scale = p.attn_mix ? ptr<float>(*attn_scale) : nullptr;
  p.attn_base = p.attn_mix ? ptr<float>(*attn_base) : nullptr;
  p.y_ffn = mut<__nv_bfloat16>(y_ffn);
  p.resid1 = mut<__nv_bfloat16>(resid1);
  p.ffn_post = mut<float>(ffn_post);
  p.ffn_comb = mut<float>(ffn_comb);
  p.ffn_pre = mut<float>(ffn_pre);
  launch_pdl("layer kernel", layer_kernel, dim3(e.ctas), dim3(THREADS), sizeof(Smem), p);
}

int prologue_ctas() {
  static const int ctas = [] {
    opt_in_smem<prologue_kernel>(PRO_SMEM);
    int device = 0, sms = 0, per = 0;
    TORCH_CHECK(cudaGetDevice(&device) == cudaSuccess && cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess);
    TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, prologue_kernel, GT, PRO_SMEM) == cudaSuccess && per > 0);
    TORCH_CHECK(sms * per >= QKV_CTAS && sms * per >= WQB_CTAS, "prologue grid too small");
    return sms * per;
  }();
  return ctas;
}

void prologue(int64_t handle, torch::Tensor y, torch::Tensor resid, torch::Tensor post, torch::Tensor comb, torch::Tensor pre, torch::Tensor fn, torch::Tensor norm,
              torch::Tensor wqa, torch::Tensor wqa_s, torch::Tensor q_norm, torch::Tensor kv_norm, double eps, torch::Tensor wqb, torch::Tensor wqb_s,
              torch::Tensor positions, torch::Tensor cos_sin, torch::Tensor resid_out, torch::Tensor input_out, torch::Tensor qr, torch::Tensor kv,
              torch::Tensor qr_q, torch::Tensor qr_s, torch::Tensor q, c10::optional<torch::Tensor> swa, int64_t swa_page, c10::optional<torch::Tensor> swa_slots) {
  const auto& e = engines.at(handle);
  const int m = int(resid.size(0));
  TORCH_CHECK(m >= 1 && m <= MAXM && resid.dim() == 3 && resid.size(1) == HC && resid.size(2) == HID && y.size(0) == m && y.size(-1) == HID);
  TORCH_CHECK(fn.size(0) == MIX && fn.size(1) == HCD && resid_out.sizes() == resid.sizes() && input_out.size(0) == m && input_out.size(1) == HID);
  TORCH_CHECK(wqa.size(0) == QKV && wqa.size(1) == HID && wqb.size(0) == WQB_ROWS && wqb.size(1) == QR && positions.scalar_type() == at::kLong);
  TORCH_CHECK(cos_sin.scalar_type() == at::kFloat && cos_sin.size(-1) == ROPE && qr.size(1) == QR && kv.size(1) == KVW && q.numel() == long(m) * WQB_ROWS);
  TORCH_CHECK(qr_s.numel() >= long(act_scale(QR / 32 - 1, m - 1, m)) + 1);
  const c10::cuda::CUDAGuard guard(resid.device());
  Prologue a{};
  a.in = AttnIn{ptr<__nv_bfloat16>(y), ptr<__nv_bfloat16>(resid), ptr<float>(post), ptr<float>(comb), ptr<float>(pre), ptr<float>(fn), ptr<__nv_bfloat16>(norm),
                mut<__nv_bfloat16>(resid_out), mut<__nv_bfloat16>(input_out), m};
  a.wqa = ptr<uint8_t>(wqa);
  a.wqa_s = ptr<uint8_t>(wqa_s);
  a.q_norm = ptr<__nv_bfloat16>(q_norm);
  a.kv_norm = ptr<__nv_bfloat16>(kv_norm);
  a.eps = float(eps);
  a.qr = mut<__nv_bfloat16>(qr);
  a.kv = mut<__nv_bfloat16>(kv);
  a.qr_q = mut<uint8_t>(qr_q);
  a.qr_s = mut<uint8_t>(qr_s);
  a.wqb = ptr<uint8_t>(wqb);
  a.wqb_s = ptr<uint8_t>(wqb_s);
  a.positions = ptr<long>(positions);
  a.cos_sin = ptr<float>(cos_sin);
  a.q = mut<__nv_bfloat16>(q);
  if (swa.has_value()) {
    TORCH_CHECK(swa->is_cuda() && swa->scalar_type() == at::kByte && swa->dim() == 2 && swa->stride(1) == 1 && swa->size(1) >= swa_page * SWA_RECORD);
    TORCH_CHECK(swa_slots->scalar_type() == at::kLong && swa_slots->numel() >= m);
    a.swa = data_as<uint8_t>(*swa);
    a.swa_stride = swa->stride(0);
    a.swa_capacity = swa->size(0) * swa_page;
    a.swa_slots = ptr<long>(*swa_slots);
    a.swa_page = int(swa_page);
  }
  a.s = e.s;
  launch_pdl("prologue", prologue_kernel, dim3(prologue_ctas()), dim3(GT), PRO_SMEM, a);
}

int64_t create_prologue(torch::Tensor anchor) {
  TORCH_CHECK(anchor.is_cuda());
  const c10::cuda::CUDAGuard guard(anchor.device());
  Engine e{};
  e.s.xc = zeroed<__nv_bfloat16>(long(MAXM) * HID);
  e.s.mixp = zeroed<float>(long(SLICES) * MAXM * MIXW);
  e.s.mixq = zeroed<float>(long(MAXM) * (MIX + 1));
  e.s.xsq = zeroed<float>(long(SLICES) * MAXM);
  e.s.aq = zeroed<uint8_t>(long(MAXM) * HID);
  e.s.as = zeroed<uint8_t>(long(SLICES / 4) * 16 * MAXM);
  e.s.qkp = zeroed<float>(long(KSPLIT) * MAXM * QKV);
  e.s.qkv = zeroed<float>(long(MAXM) * QKV);
  e.s.qsq = zeroed<float>(long(QKV_TILES) * MAXM);
  e.s.tiles = zeroed<unsigned>(QKV_TILES);
  e.s.pc = zeroed<unsigned>(PCOUNTERS);
  prologue_ctas();
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  engines.push_back(e);
  return long(engines.size()) - 1;
}

// Flare keeps rotary transforms and cache mutation in its existing attention path.
void prologue_out(int64_t handle, torch::Tensor y, torch::Tensor resid, torch::Tensor post,
                  torch::Tensor comb, torch::Tensor pre, torch::Tensor fn, torch::Tensor norm,
                  torch::Tensor hc_scale, torch::Tensor hc_base, torch::Tensor wqa, torch::Tensor wqa_s,
                  torch::Tensor q_norm, torch::Tensor kv_norm, torch::Tensor wqb, torch::Tensor wqb_s,
                  torch::Tensor resid_out, torch::Tensor input_out, torch::Tensor qr, torch::Tensor kv,
                  torch::Tensor qr_q, torch::Tensor qr_s, torch::Tensor q, torch::Tensor post_out,
                  torch::Tensor comb_out, torch::Tensor pre_out, bool defer_mix) {
  const int m = int(resid.size(0));
  TORCH_CHECK(m >= 1 && m <= 8 && resid.dim() == 3 && resid.size(1) == HC && resid.size(2) == HID);
  auto bf16 = [&](const torch::Tensor& t, int64_t elements) {
    TORCH_CHECK(t.is_cuda() && t.device() == resid.device() && t.is_contiguous()
                && t.scalar_type() == at::kBFloat16 && t.numel() == elements);
  };
  auto fp32 = [&](const torch::Tensor& t, int64_t elements) {
    TORCH_CHECK(t.is_cuda() && t.device() == resid.device() && t.is_contiguous()
                && t.scalar_type() == at::kFloat && t.numel() == elements);
  };
  auto bytes = [&](const torch::Tensor& t, int64_t elements) {
    TORCH_CHECK(t.is_cuda() && t.device() == resid.device() && t.is_contiguous()
                && t.element_size() == 1 && t.numel() == elements);
  };
  bf16(y, long(m) * HID); bf16(resid, long(m) * HCD);
  bf16(norm, HID); bf16(q_norm, QR); bf16(kv_norm, KVW);
  bf16(resid_out, long(m) * HCD); bf16(input_out, long(m) * HID);
  bf16(qr, long(m) * QR); bf16(kv, long(m) * KVW); bf16(q, long(m) * WQB_ROWS);
  fp32(post, long(m) * HC); fp32(comb, long(m) * HC * HC); fp32(pre, long(m) * HC);
  fp32(fn, MIX * HCD); fp32(hc_scale, 3); fp32(hc_base, MIX);
  fp32(post_out, long(m) * HC); fp32(comb_out, long(m) * HC * HC); fp32(pre_out, long(m) * HC);
  bytes(wqa, QKV * HID); bytes(wqa_s, QKV * HID / 32);
  bytes(wqb, WQB_ROWS * QR); bytes(wqb_s, WQB_ROWS * QR / 32);
  bytes(qr_q, long(m) * QR); bytes(qr_s, act_scale_bytes(m));
  const c10::cuda::CUDAGuard guard(resid.device());
  Prologue a{};
  a.in = {ptr<__nv_bfloat16>(y), ptr<__nv_bfloat16>(resid), ptr<float>(post), ptr<float>(comb),
          ptr<float>(pre), ptr<float>(fn), ptr<__nv_bfloat16>(norm),
          mut<__nv_bfloat16>(resid_out), mut<__nv_bfloat16>(input_out), m};
  a.in.mix_scale = defer_mix ? nullptr : ptr<float>(hc_scale);
  a.in.mix_base = ptr<float>(hc_base);
  a.in.post_out = mut<float>(post_out);
  a.in.comb_out = mut<float>(comb_out);
  a.in.pre_out = mut<float>(pre_out);
  a.in.quant_floor = 1e-4f;
  a.wqa = ptr<uint8_t>(wqa); a.wqa_s = ptr<uint8_t>(wqa_s);
  a.q_norm = ptr<__nv_bfloat16>(q_norm); a.kv_norm = ptr<__nv_bfloat16>(kv_norm);
  a.eps = NORM_EPS;
  a.qr = mut<__nv_bfloat16>(qr); a.kv = mut<__nv_bfloat16>(kv);
  a.qr_q = mut<uint8_t>(qr_q); a.qr_s = mut<uint8_t>(qr_s);
  a.wqb = ptr<uint8_t>(wqb); a.wqb_s = ptr<uint8_t>(wqb_s);
  a.q = mut<__nv_bfloat16>(q);
  a.s = engines.at(handle).s;
  launch_pdl("prepared prologue", prologue_kernel, dim3(prologue_ctas()), dim3(GT), PRO_SMEM, a);
}

py::dict inspect_state(int64_t handle, torch::Tensor anchor, int64_t rows) {
  TORCH_CHECK(rows >= 1 && rows <= MAXM);
  const c10::cuda::CUDAGuard guard(anchor.device());
  const auto& s = engines.at(handle).s;
  py::dict result;
  result["attention"] = torch::from_blob(s.yattn, {rows, HID}, anchor.options().dtype(at::kBFloat16)).clone();
  result["mixed"] = torch::from_blob(s.xc, {rows, HID}, anchor.options().dtype(at::kBFloat16)).clone();
  result["routing_weights"] = torch::from_blob(s.topk_w, {rows, TOPK}, anchor.options().dtype(at::kFloat)).clone();
  result["entry_experts"] = torch::from_blob(s.table->expert, {ENTRIES}, anchor.options().dtype(at::kInt)).clone();
  return result;
}

void destroy_engine(int64_t handle, torch::Tensor anchor) {
  const c10::cuda::CUDAGuard guard(anchor.device());
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  auto& engine = engines.at(handle);
  const Scratch s = engine.s;
  engine.layers.clear();
  void* allocations[] = {s.epoch, s.counters, s.table, s.zacc, s.zq, s.zse, s.yacc, s.yattn,
    s.mixp, s.mixq, s.xc, s.xsq, s.xbf, s.xq, s.xse, s.gacc, s.topk_w, s.sacc, s.hsq, s.hse,
    s.shacc, s.hacc, s.hq, s.hqe, s.yracc, s.aq, s.as, s.qkp, s.qkv, s.qsq, s.tiles, s.pc};
  for (void* allocation : allocations)
    if (allocation) TORCH_CHECK(cudaFree(allocation) == cudaSuccess);
  engine = Engine{};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("inspect_state", &inspect_state);
  m.def("destroy_engine", &destroy_engine);
  m.attr("RANKS") = RANKS;
  m.attr("MAXM") = MAXM;
  m.attr("HID") = HID;
  m.attr("HC") = HC;
  m.attr("EXPERTS") = EXPERTS;
  m.attr("INTER") = INTER;
  m.attr("KVW") = KVW;
  m.def("flag_bytes", &flag_bytes);
  m.def("inbox_bytes", &inbox_bytes);
  m.def("act_scale_bytes", &act_scale_bytes);
  m.def("alloc_ipc", &alloc_ipc);
  m.def("open_ipc", &open_ipc);
  m.def("create", &create);
  m.def("add_layer", &add_layer, py::arg("handle"), py::arg("woa"), py::arg("woa_s"), py::arg("wob"), py::arg("wob_s"), py::arg("gate"), py::arg("gate_b"),
        py::arg("bias_vl"), py::arg("sh13"), py::arg("sh13_s"), py::arg("sh2"), py::arg("sh2_s"), py::arg("w13"), py::arg("s13"), py::arg("w2"), py::arg("s2"),
        py::arg("fn"), py::arg("hc_scale"), py::arg("hc_base"), py::arg("norm"), py::arg("image_lo"));
  m.def("run", &run);
  m.def("prologue", &prologue);
  m.def("create_prologue", &create_prologue);
  m.def("prologue_out", &prologue_out);
}
