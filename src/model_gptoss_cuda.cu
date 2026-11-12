// gpt-oss (openai/gpt-oss-20b) decoder on a CUDA GPU (-DQPACK_CUDA=ON, QPACK_DEVICE=cuda); same model as model_gptoss.cpp. Written for
// Pascal (sm_61, Tesla P4, 7.5 GB): everything except the embedding table lives on the device -- the embedding stays in the host mmap and
// only the looked-up row is sent per token. Weights: q8g dense matrices (attention, head), f32 router/norms/biases, experts in mxfp4 and/or
// q1 (+ optional second residual q1 plane per matrix), all uploaded once. KV cache fp16 on the device. One token = CUDA graph A (all
// layers: the router, top-4 and softmax run on the device and the expert kernels take their expert ids from device memory) + graph B/C
// (final norm, head, optionally argmax).
//
// Activations are int16 per group of 32 columns (scale = absmax / 32767) in two int8 planes (hi signed, lo unsigned). A group (a "unit",
// 32 columns) is stored permuted: two 16-byte halves h = col / 16, word j = col % 4, byte m = (col / 4) % 4 -- the order in which the
// weights are consumed by dp4a after their byte-permute expansion (see the unit functions). Weights are re-laid out at upload time to
// the same order (q8g bytes, mxfp4 nibbles; q1 bits are used as stored).
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "qpack/qpack.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); } while (0)

namespace qp {
namespace {

constexpr int UMAX = 128;      // units (32 columns) per activation vector: 4096 columns max
constexpr int NK = 4;          // experts per token
constexpr int ABYTES = 4 * UMAX * 16;   // bytes of one quantised activation vector (4 arrays of UMAX int4)
struct Ctl { int tok, pos; };
struct ExpD { const void *gq[2], *gs[2], *dq[2], *ds[2]; int kind, np; };   // kind 0 = mxfp4, 1 = q1; np planes (q1 only >1)
struct DQ { const int4 *w; const unsigned short *s; int rows; };           // q8g, unit = 2 int4

// ------------------------------------------------------------------ device helpers
__device__ __forceinline__ int dp4a_us(unsigned a, int b, int c) { int d; asm("dp4a.u32.s32 %0,%1,%2,%3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }
__device__ __forceinline__ int dp4a_uu(unsigned a, unsigned b, int c) { int d; asm("dp4a.u32.u32 %0,%1,%2,%3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }
__device__ __forceinline__ int dp4a_su(int a, unsigned b, int c) { int d; asm("dp4a.s32.u32 %0,%1,%2,%3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }
__device__ __forceinline__ unsigned prmt(unsigned a, unsigned b, unsigned s) { unsigned d; asm("prmt.b32 %0,%1,%2,%3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
__device__ __forceinline__ float h2f(unsigned short h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ float warp_sum(float v) { for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o); return v; }
__device__ __forceinline__ float mx4_scale(unsigned e) { return __uint_as_float((e > 1 ? e - 1 : 0) << 23); }   // 2^(e - 128)

// quantise one group of 32 values (one per lane) into the permuted two-plane layout
__device__ __forceinline__ void warp_actq(float v, int g, int l, int8_t *aq, float *xs, int *xsum) {
  float m = fabsf(v); for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
  const float s = m / 32767.f, inv = s > 0 ? 1.f / s : 0.f;
  int q = __float2int_rn(v * inv); q = max(-32767, min(32767, q));
  const int h = l >> 4, off = 16 * g + 4 * (l & 3) + ((l >> 2) & 3);
  aq[(size_t)(0 * 2 + h) * UMAX * 16 + off] = (int8_t)(q >> 8);
  aq[(size_t)(1 * 2 + h) * UMAX * 16 + off] = (int8_t)(q & 255);
  int sm = q; for (int o = 16; o; o >>= 1) sm += __shfl_xor_sync(~0u, sm, o);
  if (l == 0) { xs[g] = s; xsum[g] = sm; }
}

struct XU { int4 h0, h1, l0, l1; };
__device__ __forceinline__ XU ldx(const int4 *sx, int u) { XU x; x.h0 = sx[u]; x.h1 = sx[UMAX + u]; x.l0 = sx[2 * UMAX + u]; x.l1 = sx[3 * UMAX + u]; return x; }

// integer dot products of one unit (32 columns); all return sum_c w[c] * q[c] with q the int16 activation values
__device__ __forceinline__ int q8_unit(uint4 w0, uint4 w1, const XU &x) {
  int d = __dp4a((int)w0.x, x.h0.x, 0); d = __dp4a((int)w0.y, x.h0.y, d); d = __dp4a((int)w0.z, x.h0.z, d); d = __dp4a((int)w0.w, x.h0.w, d);
  d = __dp4a((int)w1.x, x.h1.x, d); d = __dp4a((int)w1.y, x.h1.y, d); d = __dp4a((int)w1.z, x.h1.z, d); d = __dp4a((int)w1.w, x.h1.w, d);
  int e = dp4a_su((int)w0.x, (unsigned)x.l0.x, 0); e = dp4a_su((int)w0.y, (unsigned)x.l0.y, e); e = dp4a_su((int)w0.z, (unsigned)x.l0.z, e); e = dp4a_su((int)w0.w, (unsigned)x.l0.w, e);
  e = dp4a_su((int)w1.x, (unsigned)x.l1.x, e); e = dp4a_su((int)w1.y, (unsigned)x.l1.y, e); e = dp4a_su((int)w1.z, (unsigned)x.l1.z, e); e = dp4a_su((int)w1.w, (unsigned)x.l1.w, e);
  return (d << 8) + e;
}
// mxfp4: word k of the unit holds, in nibbles 0..3 / 4..7, the codes of columns 4m + k / 16 + 4m + k (m = 0..3). A code is (sign, 3-bit index)
// into the 2 x E2M1 magnitudes {0,1,2,3,4,6,8,12}. Two byte-permutes build the unsigned bytes 12 + v: one with the code as selector (a set sign
// bit selects "replicate the sign of the table byte" = 0 because the table bytes are < 128), one with the sign flipped over the table 12 - |v|.
// So u = P[code] | N[code ^ 8] has 12 + v in every byte, and sum(w q) = sum(u q) - 12 sum(q).
__device__ __forceinline__ int mx4_unit(uint4 w, const XU &x, int xsum) {
  const unsigned P0 = 0x0F0E0D0Cu, P1 = 0x18141210u, N0 = 0x090A0B0Cu, N1 = 0x00040608u;
  int d = 0, e = 0;
#pragma unroll
  for (int k = 0; k < 4; k++) {
    const unsigned wk = k == 0 ? w.x : k == 1 ? w.y : k == 2 ? w.z : w.w, wn = wk ^ 0x88888888u;
    const unsigned u0 = prmt(P0, P1, wk) | prmt(N0, N1, wn), u1 = prmt(P0, P1, wk >> 16) | prmt(N0, N1, wn >> 16);
    const int xh0 = k == 0 ? x.h0.x : k == 1 ? x.h0.y : k == 2 ? x.h0.z : x.h0.w, xh1 = k == 0 ? x.h1.x : k == 1 ? x.h1.y : k == 2 ? x.h1.z : x.h1.w;
    const int xl0 = k == 0 ? x.l0.x : k == 1 ? x.l0.y : k == 2 ? x.l0.z : x.l0.w, xl1 = k == 0 ? x.l1.x : k == 1 ? x.l1.y : k == 2 ? x.l1.z : x.l1.w;
    d = dp4a_us(u0, xh0, d); d = dp4a_us(u1, xh1, d); e = dp4a_uu(u0, (unsigned)xl0, e); e = dp4a_uu(u1, (unsigned)xl1, e);
  }
  return (d << 8) + e - 12 * xsum;
}
// q1: bit b of the word = column b (1 => +1). For j = 0..3 the bits j, j+4, j+8, .. are moved to the low bit of nibbles by (w >> j) & 0x11111111
// and a byte-permute with the table {0xFF, 0x01} turns four of them into four +-1 bytes (columns 4m + j, resp. 16 + 4m + j).
__device__ __forceinline__ int q1_unit(unsigned w, const XU &x) {
  int d = 0, e = 0;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const unsigned s = (w >> j) & 0x11111111u, a = prmt(0x000001FFu, 0, s), b = prmt(0x000001FFu, 0, s >> 16);
    const int xh0 = j == 0 ? x.h0.x : j == 1 ? x.h0.y : j == 2 ? x.h0.z : x.h0.w, xh1 = j == 0 ? x.h1.x : j == 1 ? x.h1.y : j == 2 ? x.h1.z : x.h1.w;
    const int xl0 = j == 0 ? x.l0.x : j == 1 ? x.l0.y : j == 2 ? x.l0.z : x.l0.w, xl1 = j == 0 ? x.l1.x : j == 1 ? x.l1.y : j == 2 ? x.l1.z : x.l1.w;
    d = __dp4a((int)a, xh0, d); d = __dp4a((int)b, xh1, d); e = dp4a_us((unsigned)xl0, (int)a, e); e = dp4a_us((unsigned)xl1, (int)b, e);
  }
  return (d << 8) + e;
}

// ------------------------------------------------------------------ per-row register state (loaded early, consumed after the smem barrier)
constexpr int NSE = 3;   // units per lane for the 90-unit expert rows
template <int K> struct RR;
template <> struct RR<0> {   // mxfp4 row: 90 units of 16 bytes + 90 e8m0 scales
  uint4 w[NSE]; unsigned e[NSE];
  __device__ __forceinline__ void load(const void *q, const void *s, size_t row, int units, int lane) {
    const uint4 *p = (const uint4 *)q + row * units; const unsigned char *sp = (const unsigned char *)s + row * units;
#pragma unroll
    for (int i = 0; i < NSE; i++) { const int u = lane + 32 * i; if (u < units) { w[i] = __ldcs(p + u); e[i] = sp[u]; } else { w[i] = make_uint4(0, 0, 0, 0); e[i] = 0; } }
  }
  __device__ __forceinline__ float dot(const int4 *sx, const float *sxs, const int *sxsum, int units, int lane) const {
    float acc = 0;
#pragma unroll
    for (int i = 0; i < NSE; i++) { const int u = lane + 32 * i; if (u < units) acc += (float)mx4_unit(w[i], ldx(sx, u), sxsum[u]) * (sxs[u] * mx4_scale(e[i])); }
    return acc;
  }
};
template <> struct RR<1> {   // q1 row: ceil(cols / 128) groups of 16 bytes of sign bits + one fp16 scale per group
  unsigned w[NSE]; unsigned short s[NSE];
  __device__ __forceinline__ void load(const void *q, const void *sc, size_t row, int units, int lane) {
    const int G = (units * 32 + 127) / 128; const unsigned *p = (const unsigned *)q + row * G * 4; const unsigned short *sp = (const unsigned short *)sc + row * G;
#pragma unroll
    for (int i = 0; i < NSE; i++) { const int u = lane + 32 * i; if (u < units) { w[i] = __ldcs(p + u); s[i] = __ldg(sp + (u >> 2)); } else { w[i] = 0; s[i] = 0; } }
  }
  __device__ __forceinline__ float dot(const int4 *sx, const float *sxs, const int *, int units, int lane) const {
    float acc = 0;
#pragma unroll
    for (int i = 0; i < NSE; i++) { const int u = lane + 32 * i; if (u < units) acc += (float)q1_unit(w[i], ldx(sx, u)) * (sxs[u] * h2f(s[i])); }
    return acc;
  }
};

// ------------------------------------------------------------------ kernels
// x += sum of the previous layer's weighted expert outputs (moe, may be null); y = rmsnorm(x) * w quantised per 32 columns (one block of 1024
// threads, n <= 3072); optionally also stored as float; optionally the router (32 f32 logits + bias, top-4, softmax over the 4 -> sel / wt)
__global__ void __launch_bounds__(1024) k_norm(float *x, const float *w, const float *moe, int n, float eps, int8_t *aq, float *xs, int *xsum, float *yf,
                                               const float *rw, const float *rb, int E, int *sel, float *wt) {
  __shared__ float red[32]; __shared__ float sy[3072]; __shared__ float lg[64];
  const int t = threadIdx.x, wp = t >> 5, l = t & 31;
  float xv[3], ss = 0;
#pragma unroll
  for (int i = 0; i < 3; i++) {
    const int idx = t + 1024 * i; float v = 0;
    if (idx < n) { v = x[idx]; if (moe) { v += moe[idx] + moe[n + idx] + moe[2 * n + idx] + moe[3 * n + idx]; x[idx] = v; } }
    xv[i] = v; ss += v * v;
  }
  ss = warp_sum(ss); if (l == 0) red[wp] = ss; __syncthreads();
  float tot = 0; for (int i = 0; i < 32; i++) tot += red[i];
  const float inv = 1.f / sqrtf(tot / n + eps);
#pragma unroll
  for (int i = 0; i < 3; i++) { const int idx = t + 1024 * i; if (idx < n) { const float y = xv[i] * inv * w[idx]; sy[idx] = y; if (yf) yf[idx] = y; } }
  __syncthreads();
  for (int g = wp; g < (n >> 5); g += 32) warp_actq(sy[g * 32 + l], g, l, aq, xs, xsum);
  if (rw) {
    if (wp < E) {
      const float4 *r4 = (const float4 *)(rw + (size_t)wp * n), *s4 = (const float4 *)sy; float a = 0;
      for (int i = l; i < (n >> 2); i += 32) { const float4 r = r4[i], s = s4[i]; a += r.x * s.x + r.y * s.y + r.z * s.z + r.w * s.w; }
      a = warp_sum(a); if (l == 0) lg[wp] = a + rb[wp];
    }
    __syncthreads();
    if (t == 0) {
      int idx[NK]; float v[NK];
      for (int k = 0; k < NK; k++) {
        int b = -1; for (int e = 0; e < E; e++) { bool used = false; for (int j = 0; j < k; j++) used |= idx[j] == e; if (!used && (b < 0 || lg[e] > lg[b])) b = e; }
        idx[k] = b; v[k] = lg[b];
      }
      float mx = v[0], sum = 0; for (int k = 1; k < NK; k++) mx = fmaxf(mx, v[k]);
      float p[NK]; for (int k = 0; k < NK; k++) { p[k] = expf(v[k] - mx); sum += p[k]; }
      for (int k = 0; k < NK; k++) { sel[k] = idx[k]; wt[k] = p[k] / sum; }
    }
  }
}

struct DenseArgs {
  DQ m[3]; const float *bias[3]; float *out[3]; int nm, rowstart[4]; int units;
  const int4 *aq; const float *as; const int *asum; float *resid;   // non-null: resid += y
};
// q8g matvec, warp per row, up to three matrices sharing the input vector
template <int NS, int WPB> __global__ void __launch_bounds__(WPB * 32) k_dense(DenseArgs a) {
  __shared__ int4 sx[4 * UMAX]; __shared__ float sxs[UMAX]; __shared__ int sxsum[UMAX];
  const int units = a.units, tid = threadIdx.x, lane = tid & 31;
  int r = blockIdx.x * WPB + (tid >> 5); const bool valid = r < a.rowstart[a.nm]; int mi = 0;
  uint4 w0[NS], w1[NS]; unsigned short sc[NS];
  if (valid) {
    while (r >= a.rowstart[mi + 1]) mi++; r -= a.rowstart[mi];
    const uint4 *wp = (const uint4 *)a.m[mi].w + (size_t)r * units * 2; const unsigned short *sp = a.m[mi].s + (size_t)r * units;
#pragma unroll
    for (int i = 0; i < NS; i++) { const int u = lane + 32 * i; if (u < units) { w0[i] = __ldcs(wp + 2 * u); w1[i] = __ldcs(wp + 2 * u + 1); sc[i] = __ldg(sp + u); } else { w0[i] = w1[i] = make_uint4(0, 0, 0, 0); sc[i] = 0; } }
  }
  for (int i = tid; i < 4 * units; i += WPB * 32) { const int ph = i / units, u = i - ph * units; sx[ph * UMAX + u] = a.aq[ph * UMAX + u]; }
  for (int i = tid; i < units; i += WPB * 32) { sxs[i] = a.as[i]; sxsum[i] = a.asum[i]; }
  __syncthreads();
  if (!valid) return;
  float acc = 0;
#pragma unroll
  for (int i = 0; i < NS; i++) { const int u = lane + 32 * i; if (u < units) acc += (float)q8_unit(w0[i], w1[i], ldx(sx, u)) * (sxs[u] * h2f(sc[i])); }
  const float y = warp_sum(acc);
  if (lane == 0) {
    float v = y; if (a.bias[mi]) v += a.bias[mi][r];
    if (a.resid) a.resid[r] += v; else a.out[mi][r] = v;
  }
}

// ---- experts
template <int K, int NP> struct Pair { RR<K> a[NP], b[NP]; };
template <int K, int NP> __device__ __forceinline__ void load_pair(Pair<K, NP> &p, const ExpD &D, int n, int units, int lane) {
#pragma unroll
  for (int q = 0; q < NP; q++) { p.a[q].load(D.gq[q], D.gs[q], 2 * (size_t)n, units, lane); p.b[q].load(D.gq[q], D.gs[q], 2 * (size_t)n + 1, units, lane); }
}
// gate/up of the expert in device slot blockIdx.y for 32 neurons (rows 2n = gate, 2n + 1 = up, interleaved), swiglu-oai, quantised h
constexpr int GUT = 512, NPW = 32 * 32 / GUT / 32 * 0 + 32 / (GUT / 32);   // threads per gate/up block; neurons per warp
template <int K, int NP> __device__ __forceinline__ void gu_body(const ExpD &D, int eid, const float *bgu, float limit, float alpha, int units, int ff, const int4 *aq, const float *as, const int *asum,
                                                                 int8_t *hq, float *hs, int *hsum, int4 *sx, float *sxs, int *sxsum, float *sh) {
  const int tid = threadIdx.x, wp = tid >> 5, lane = tid & 31, nb = blockIdx.x * 32 + wp * NPW;
  Pair<K, NP> cur, nxt; load_pair(cur, D, nb, units, lane);
  for (int i = tid; i < 4 * units; i += GUT) { const int ph = i / units, u = i - ph * units; sx[ph * UMAX + u] = aq[ph * UMAX + u]; }
  for (int i = tid; i < units; i += GUT) { sxs[i] = as[i]; sxsum[i] = asum[i]; }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < NPW; i++) {
    if (i < NPW - 1) load_pair(nxt, D, nb + i + 1, units, lane);
    float g = 0, u = 0;
#pragma unroll
    for (int q = 0; q < NP; q++) { g += cur.a[q].dot(sx, sxs, sxsum, units, lane); u += cur.b[q].dot(sx, sxs, sxsum, units, lane); }
    g = warp_sum(g); u = warp_sum(u);
    if (lane == 0) {
      const int n = nb + i; g += bgu[(size_t)eid * 2 * ff + 2 * n]; u += bgu[(size_t)eid * 2 * ff + 2 * n + 1];
      const float gl = fminf(g, limit), ul = fmaxf(-limit, fminf(limit, u));
      sh[wp * NPW + i] = gl / (1.f + expf(-alpha * gl)) * (ul + 1.f);
    }
    if (i < NPW - 1) cur = nxt;
  }
  __syncthreads();
  if (wp == 0) warp_actq(sh[lane], blockIdx.x, lane, hq, hs, hsum);
}
// MODE 0: mxfp4 experts, 1: q1, 2: q1 with a second plane; blocks whose expert has another kind exit (separate kernels give each kind its own register budget)
template <int MODE> __global__ void __launch_bounds__(GUT, MODE == 0 ? 1 : 2) k_gateup(const ExpD *tab, const int *sel, const float *bgu, float limit, float alpha, int units, int ff, const int4 *aq, const float *as, const int *asum,
                                                int8_t *hq, float *hs, int *hsum) {
  const int k = blockIdx.y, eid = sel[k]; const ExpD D = tab[eid];
  if ((D.kind == 0 ? 0 : D.np == 1 ? 1 : 2) != MODE) return;
  __shared__ int4 sx[4 * UMAX]; __shared__ float sxs[UMAX]; __shared__ int sxsum[UMAX]; __shared__ float sh[32];
  int8_t *q = hq + (size_t)k * ABYTES; float *s = hs + k * UMAX; int *sm = hsum + k * UMAX;
  if (MODE == 0) gu_body<0, 1>(D, eid, bgu, limit, alpha, units, ff, aq, as, asum, q, s, sm, sx, sxs, sxsum, sh);
  else if (MODE == 1) gu_body<1, 1>(D, eid, bgu, limit, alpha, units, ff, aq, as, asum, q, s, sm, sx, sxs, sxsum, sh);
  else gu_body<1, 2>(D, eid, bgu, limit, alpha, units, ff, aq, as, asum, q, s, sm, sx, sxs, sxsum, sh);
}
// down projection of expert slot blockIdx.y, 8 rows per block: ye[k][r] = wt[k] * (W h + b)
template <int K, int NP> __device__ __forceinline__ void dn_body(const ExpD &D, int eid, float wk, const float *bd, int units, int d, const int4 *aq, const float *as, const int *asum, float *ye, int4 *sx, float *sxs, int *sxsum) {
  const int tid = threadIdx.x, wp = tid >> 5, lane = tid & 31, r = blockIdx.x * 8 + wp;
  RR<K> row[NP];
#pragma unroll
  for (int q = 0; q < NP; q++) row[q].load(D.dq[q], D.ds[q], r, units, lane);
  for (int i = tid; i < 4 * units; i += 256) { const int ph = i / units, u = i - ph * units; sx[ph * UMAX + u] = aq[ph * UMAX + u]; }
  for (int i = tid; i < units; i += 256) { sxs[i] = as[i]; sxsum[i] = asum[i]; }
  __syncthreads();
  float acc = 0;
#pragma unroll
  for (int q = 0; q < NP; q++) acc += row[q].dot(sx, sxs, sxsum, units, lane);
  acc = warp_sum(acc);
  if (lane == 0) ye[(size_t)blockIdx.y * d + r] = wk * (acc + bd[(size_t)eid * d + r]);
}
template <int MODE> __global__ void __launch_bounds__(256, MODE == 0 ? 5 : 8) k_down(const ExpD *tab, const int *sel, const float *wt, const float *bd, int units, int d, const int4 *aq, const float *as, const int *asum, float *ye) {
  const int k = blockIdx.y, eid = sel[k]; const ExpD D = tab[eid];
  if ((D.kind == 0 ? 0 : D.np == 1 ? 1 : 2) != MODE) return;
  __shared__ int4 sx[4 * UMAX]; __shared__ float sxs[UMAX]; __shared__ int sxsum[UMAX];
  const int4 *q = aq + (size_t)k * (ABYTES / 16); const float *s = as + k * UMAX; const int *sm = asum + k * UMAX;
  if (MODE == 0) dn_body<0, 1>(D, eid, wt[k], bd, units, d, q, s, sm, ye, sx, sxs, sxsum);
  else if (MODE == 1) dn_body<1, 1>(D, eid, wt[k], bd, units, d, q, s, sm, ye, sx, sxs, sxsum);
  else dn_body<1, 2>(D, eid, wt[k], bd, units, d, q, s, sm, ye, sx, sxs, sxsum);
}

// ---- attention (head_dim 64, 8 q heads per kv head): block = (kv group, chunk of CH positions), 256 threads; scores: CH positions x 16 lanes (4
// dims each), values: thread = (dim, head subset). rope of q on the fly (tables include the YaRN concentration), the current position takes
// its rotated k / v from smem and the block holding it appends them (fp16) to the cache. Positions [t_lo, T), t_lo = max(0, T - win) if win > 0.
constexpr int CH = 16;
constexpr int ANT = 256;
constexpr int HD = 64;
__global__ void __launch_bounds__(ANT) k_attn(const float *q, const float *kn, const float *vn, const float *cs, const float *sn, __half *kc, __half *vc, float *part, const Ctl *ctl,
                                              int nkv, int nch, float scl, int win) {
  constexpr int DPL = HD / 16, GR = ANT / HD, NA = 8 / GR, rep = 8;
  const int g = blockIdx.x, c = blockIdx.y, pos = ctl->pos, T = pos + 1, tlo = win > 0 ? max(0, T - win) : 0, t0 = tlo + c * CH; if (t0 >= T) return;
  const int n = min(CH, T - t0), tid = threadIdx.x, kvd = nkv * HD, half = HD / 2, pl = pos - t0;
  const bool last = pl >= 0 && pl < CH;
  __shared__ __align__(16) float sq[8 * HD]; __shared__ float sc[8][CH]; __shared__ float pm[8], ps[8]; __shared__ float sk[HD], sv[HD];
  const int p = tid / 16, sb = tid % 16, d = tid % HD, grp = tid / HD;
  __half2 kh[DPL / 2];
#pragma unroll
  for (int e = 0; e < DPL / 2; e++) kh[e] = __float2half2_rn(0.f);
  if (p < n && p != pl) { const __half2 *kr = (const __half2 *)(kc + (size_t)(t0 + p) * kvd + g * HD + sb * DPL);
#pragma unroll
    for (int e = 0; e < DPL / 2; e++) kh[e] = kr[e]; }
  float vv[CH];
#pragma unroll
  for (int t = 0; t < CH; t++) vv[t] = (t < n && t != pl) ? __half2float(vc[(size_t)(t0 + t) * kvd + g * HD + d]) : 0.f;
  constexpr int NR = (8 * HD / 2 + ANT - 1) / ANT;
  float qa[NR], qb[NR], qc[NR], qs[NR], ka = 0, kb = 0, kcs = 0, ksn = 0, vnew = 0;
#pragma unroll
  for (int r = 0; r < NR; r++) {
    const int idx = tid + r * ANT, rj = idx / half, ri = idx % half; qa[r] = qb[r] = qc[r] = qs[r] = 0;
    if (rj < rep) { const float *pq = q + (size_t)(g * rep + rj) * HD; qa[r] = pq[ri]; qb[r] = pq[ri + half]; qc[r] = cs[(size_t)pos * half + ri]; qs[r] = sn[(size_t)pos * half + ri]; }
  }
  if (last && tid < half) { ka = kn[g * HD + tid]; kb = kn[g * HD + tid + half]; kcs = cs[(size_t)pos * half + tid]; ksn = sn[(size_t)pos * half + tid]; }
  if (last && tid < HD) vnew = vn[g * HD + tid];
#pragma unroll
  for (int t = 0; t < CH; t++) asm volatile("" : "+f"(vv[t]));
#pragma unroll
  for (int r = 0; r < NR; r++) {
    const int idx = tid + r * ANT, rj = idx / half, ri = idx % half;
    if (rj < rep) { sq[rj * HD + ri] = qa[r] * qc[r] - qb[r] * qs[r]; sq[rj * HD + ri + half] = qb[r] * qc[r] + qa[r] * qs[r]; }
  }
  if (last) {
    if (tid < half) {
      const __half h0 = __float2half(ka * kcs - kb * ksn), h1 = __float2half(kb * kcs + ka * ksn);
      sk[tid] = __half2float(h0); sk[tid + half] = __half2float(h1);
      kc[(size_t)pos * kvd + g * HD + tid] = h0; kc[(size_t)pos * kvd + g * HD + tid + half] = h1;
    }
    if (tid < HD) { const __half hv = __float2half(vnew); sv[tid] = __half2float(hv); vc[(size_t)pos * kvd + g * HD + tid] = hv; }
  }
  __syncthreads();
  {
    float s[8];
#pragma unroll
    for (int j = 0; j < 8; j++) s[j] = 0;
    if (p < n) {
      float kf[DPL];
      if (p == pl) {
#pragma unroll
        for (int e = 0; e < DPL; e++) kf[e] = sk[sb * DPL + e];
      } else {
#pragma unroll
        for (int e = 0; e < DPL / 2; e++) { const float2 f = __half22float2(kh[e]); kf[2 * e] = f.x; kf[2 * e + 1] = f.y; }
      }
#pragma unroll
      for (int j = 0; j < 8; j++) {
#pragma unroll
        for (int e = 0; e < DPL; e++) s[j] = fmaf(sq[j * HD + sb * DPL + e], kf[e], s[j]);
      }
    }
#pragma unroll
    for (int j = 0; j < 8; j++) { float v = s[j]; for (int o = 1; o < 16; o <<= 1) v += __shfl_xor_sync(~0u, v, o); if (sb == 0 && p < n) sc[j][p] = v * scl; }
  }
  __syncthreads();
  if (tid / 16 < rep) {
    const int j = tid / 16, l16 = tid & 15; const float v = l16 < n ? sc[j][l16] : -1e30f; float mx = v;
    for (int o = 8; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o, 16));
    const float e = l16 < n ? expf(v - mx) : 0.f; float sum = e;
    for (int o = 8; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o, 16);
    sc[j][l16] = e; if (l16 == 0) { pm[j] = mx; ps[j] = sum; }
  }
  __syncthreads();
  float acc[NA];
#pragma unroll
  for (int i = 0; i < NA; i++) acc[i] = 0;
#pragma unroll
  for (int t = 0; t < CH; t++) {
    const float v = t == pl ? sv[d] : vv[t];
#pragma unroll
    for (int i = 0; i < NA; i++) { const int j = grp + i * GR; acc[i] = fmaf(t < n ? sc[j][t] : 0.f, v, acc[i]); }
  }
#pragma unroll
  for (int i = 0; i < NA; i++) { const int j = grp + i * GR; float *pp = part + ((size_t)(g * rep + j) * nch + c) * (HD + 2); pp[2 + d] = acc[i]; if (d == 0) { pp[0] = pm[j]; pp[1] = ps[j]; } }
}
// merge the chunk partials and the sink logit of one head, quantise (block = head, 64 threads)
__global__ void k_merge(const float *part, const float *sinks, const Ctl *ctl, int nch, int win, int8_t *aq, float *xs, int *xsum) {
  const int h = blockIdx.x, d = threadIdx.x, T = ctl->pos + 1, tlo = win > 0 ? max(0, T - win) : 0, nc = (T - tlo + CH - 1) / CH;
  const float *ph = part + (size_t)h * nch * (HD + 2); const float sink = sinks[h]; float M = sink;
  for (int c = 0; c < nc; c++) M = fmaxf(M, ph[(size_t)c * (HD + 2)]);
  float L = expf(sink - M), o = 0;
  for (int c = 0; c < nc; c++) { const float *pc = ph + (size_t)c * (HD + 2); const float f = expf(pc[0] - M); L += pc[1] * f; o += f * pc[2 + d]; }
  o /= L;
  warp_actq(o, h * (HD / 32) + (d >> 5), d & 31, aq, xs, xsum);
}

constexpr int AM_BLOCKS = 64;
__global__ void __launch_bounds__(1024) k_argmax1(const float *x, int n, float *pv, int *pi) {
  float bv = -3.4e38f; int bi = 0x7fffffff;
  for (int i = blockIdx.x * 1024 + threadIdx.x; i < n; i += AM_BLOCKS * 1024) { const float v = x[i]; if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; } }
  for (int o = 16; o; o >>= 1) { const float ov = __shfl_xor_sync(~0u, bv, o); const int oi = __shfl_xor_sync(~0u, bi, o); if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; } }
  __shared__ float sv[32]; __shared__ int si[32];
  if ((threadIdx.x & 31) == 0) { sv[threadIdx.x >> 5] = bv; si[threadIdx.x >> 5] = bi; }
  __syncthreads();
  if (threadIdx.x < 32) {
    bv = sv[threadIdx.x]; bi = si[threadIdx.x];
    for (int o = 16; o; o >>= 1) { const float ov = __shfl_xor_sync(~0u, bv, o); const int oi = __shfl_xor_sync(~0u, bi, o); if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; } }
    if (threadIdx.x == 0) { pv[blockIdx.x] = bv; pi[blockIdx.x] = bi; }
  }
}
__global__ void k_argmax2(const float *pv, const int *pi, int *out) {
  float bv = pv[threadIdx.x]; int bi = pi[threadIdx.x];
  for (int o = 16; o; o >>= 1) { const float ov = __shfl_xor_sync(~0u, bv, o); const int oi = __shfl_xor_sync(~0u, bi, o); if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; } }
  __shared__ float sv[2]; __shared__ int si[2];
  if ((threadIdx.x & 31) == 0) { sv[threadIdx.x >> 5] = bv; si[threadIdx.x >> 5] = bi; }
  __syncthreads();
  if (threadIdx.x == 0) { if (sv[1] > sv[0] || (sv[1] == sv[0] && si[1] < si[0])) { sv[0] = sv[1]; si[0] = si[1]; } *out = si[0]; }
}

// ------------------------------------------------------------------ host
struct Layer {
  DQ wq, wk, wv, wo; const float *an, *fn, *bq, *bk, *bv, *bo, *sinks, *rw, *rb, *bgu, *bd; const ExpD *tab; bool sliding;
};

class GptOssCuda : public Model {
 public:
  GptOssCuda(std::shared_ptr<Container> c, int max_ctx) {
    c_ = c; cfg_ = c->config();
    d_ = (int)cfg_["hidden_size"].as_int(); L_ = (int)cfg_["num_hidden_layers"].as_int(); nh_ = (int)cfg_["num_attention_heads"].as_int();
    nkv_ = (int)cfg_["num_key_value_heads"].as_int(); hd_ = (int)cfg_["head_dim"].as_int(64); ff_ = (int)cfg_["intermediate_size"].as_int();
    V_ = (int)cfg_["vocab_size"].as_int(); E_ = (int)cfg_["num_local_experts"].as_int(); K_ = (int)cfg_["num_experts_per_tok"].as_int(4);
    eps_ = (float)cfg_["rms_norm_eps"].as_num(1e-5); window_ = (int)cfg_["sliding_window"].as_int(128);
    limit_ = (float)cfg_["swiglu_limit"].as_num(7.0); alpha_ = (float)cfg_["swiglu_alpha"].as_num(1.702);
    if (const char *p = getenv("QPACK_LAYERS")) L_ = std::min(L_, atoi(p));
    if (hd_ != 64 || nh_ != 8 * nkv_ || K_ != NK || E_ > 32) throw std::runtime_error("cuda gptoss: head_dim 64, 8 q heads per kv head, 4 experts of <= 32 expected");
    if (d_ % 32 || d_ > 3072 || ff_ % 32 || ff_ > 3072 || (nh_ * hd_) > 4096 || (nh_ * hd_) % 32) throw std::runtime_error("cuda gptoss: unsupported dimensions");
    const int lim = (int)cfg_["max_position_embeddings"].as_int(1 << 20);
    ctx_ = max_ctx > 0 ? max_ctx : std::min(4096, lim);
    kvd_ = nkv_ * hd_; qd_ = nh_ * hd_; ud_ = d_ / 32; uff_ = ff_ / 32;
    CK(cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking));
    embed_ = &c->at("embed");
    {   // memory check: the device holds everything but the embedding
      size_t need = 0; for (auto &kv : c->tensors()) { const Tensor &t = kv.second; if (&t == embed_) continue; if (kv.first.rfind("layers.", 0) == 0 && atoi(kv.first.c_str() + 7) >= L_) continue; need += tensor_bytes(t); }
      size_t fr = 0, tot = 0; CK(cudaMemGetInfo(&fr, &tot));
      const size_t kv = 2ull * L_ * ctx_ * kvd_ * 2 + (96u << 20);
      if (need + kv > fr) throw std::runtime_error("cuda gptoss: model needs ~" + std::to_string((need + kv) >> 20) + " MiB of device memory (weights " + std::to_string(need >> 20) + " + KV/buffers " + std::to_string(kv >> 20) + "), only " + std::to_string(fr >> 20) + " MiB free");
    }
    head_ = up_q8g(c->at("lm_head")); fnorm_ = upf(c->at("final_norm"));
    if (head_.rows != V_) throw std::runtime_error("cuda gptoss: head rows");
    const Json &lt = cfg_["layer_types"];
    for (int l = 0; l < L_; l++) {
      auto g = [&](const char *n) { return &c->at("layers." + std::to_string(l) + "." + n); };
      Layer ly{};
      ly.wq = up_q8g(*g("wq")); ly.wk = up_q8g(*g("wk")); ly.wv = up_q8g(*g("wv")); ly.wo = up_q8g(*g("wo"));
      ly.an = upf(*g("attn_norm")); ly.fn = upf(*g("ffn_norm")); ly.bq = upf(*g("bq")); ly.bk = upf(*g("bk")); ly.bv = upf(*g("bv")); ly.bo = upf(*g("bo")); ly.sinks = upf(*g("sinks"));
      ly.rw = upf(*g("router_w")); ly.rb = upf(*g("router_b")); ly.bgu = upf(*g("b_gate_up")); ly.bd = upf(*g("b_down"));
      if (ly.wq.rows != qd_ || ly.wo.rows != d_) throw std::runtime_error("cuda gptoss: unexpected attention shapes");
      ly.sliding = lt.type == Json::Array && l < (int)lt.arr.size() ? lt.arr[l].as_str() == "sliding_attention" : (l % 2 == 0);
      build_experts(l, ly);
      layers_.push_back(ly);
    }
    const size_t kvn = (size_t)L_ * ctx_ * kvd_;
    CK(cudaMalloc(&kc_, kvn * 2)); CK(cudaMalloc(&vc_, kvn * 2)); CK(cudaMemset(kc_, 0, kvn * 2)); CK(cudaMemset(vc_, 0, kvn * 2));
    nch_ = (ctx_ + CH - 1) / CH;
    const size_t arena_n = ((size_t)ctx_ * (hd_ / 2) * 8 + (size_t)nh_ * nch_ * (hd_ + 2) * 4 + (size_t)V_ * 4 + (6 << 20) + (size_t)ctx_ * 4 + 4095) & ~(size_t)4095;
    CK(cudaMalloc(&arena_, arena_n)); size_t aoff = 0;
    auto ab = [&](size_t n) { void *r = (char *)arena_ + aoff; aoff += (n + 255) & ~(size_t)255; if (aoff > arena_n) throw std::runtime_error("arena overflow"); return r; };
    auto fl = [&](float **p, size_t n) { *p = (float *)ab(n * 4); };
    fl(&x_, d_); fl(&xn_, d_); fl(&q_, qd_); fl(&k_, kvd_); fl(&v_, kvd_); fl(&logits_, V_); fl(&part_, (size_t)nh_ * nch_ * (hd_ + 2)); fl(&ye_, NK * d_); fl(&wt_, NK);
    sel_ = (int *)ab(NK * 4);
    aA_q_ = (int8_t *)ab(ABYTES); fl(&aA_s_, UMAX); aA_sum_ = (int *)ab(UMAX * 4);
    aO_q_ = (int8_t *)ab(ABYTES); fl(&aO_s_, UMAX); aO_sum_ = (int *)ab(UMAX * 4);
    aH_q_ = (int8_t *)ab((size_t)NK * ABYTES); fl(&aH_s_, NK * UMAX); aH_sum_ = (int *)ab(NK * UMAX * 4);
    ctl_ = (Ctl *)ab(sizeof(Ctl)); amv_ = (float *)ab(AM_BLOCKS * 4); ami_ = (int *)ab(AM_BLOCKS * 4); tok_ = (int *)ab(4);
    CK(cudaMemset(arena_, 0, aoff));
    CK(cudaMallocHost(&hctl_, RING * sizeof(Ctl))); CK(cudaMallocHost(&hlog_, (size_t)V_ * 4)); CK(cudaMallocHost(&hxn_, (size_t)d_ * 4)); CK(cudaMallocHost(&hemb_, (size_t)RING * d_ * 4)); CK(cudaMallocHost(&htok_, 4));
    // YaRN rope tables with the concentration folded in (same maths as the CPU model)
    std::vector<double> inv(hd_ / 2); double conc = 1.0;
    {
      const double theta = cfg_["rope_theta"].as_num(150000), factor = cfg_["rope_scaling"]["factor"].as_num(1);
      const double orig = cfg_["rope_scaling"]["original_max_position_embeddings"].as_num(4096), beta_fast = cfg_["rope_scaling"]["beta_fast"].as_num(32), beta_slow = cfg_["rope_scaling"]["beta_slow"].as_num(1);
      const int half = hd_ / 2;
      for (int i = 0; i < half; i++) inv[i] = 1.0 / std::pow(theta, (2.0 * i) / hd_);
      if (factor > 1.0) {
        conc = 0.1 * std::log(factor) + 1.0;
        double low = half * std::log(orig / (beta_fast * 2 * M_PI)) / std::log(theta), high = half * std::log(orig / (beta_slow * 2 * M_PI)) / std::log(theta);
        if (cfg_["rope_scaling"]["truncate"].as_bool(false)) { low = std::floor(low); high = std::ceil(high); }
        for (int i = 0; i < half; i++) {
          const double ex = inv[i], in = inv[i] / factor, ramp = std::min(1.0, std::max(0.0, (i - low) / (high - low))), mask = 1 - ramp;
          inv[i] = in * (1 - mask) + ex * mask;
        }
      }
    }
    const int half = hd_ / 2; std::vector<float> cs((size_t)ctx_ * half), sn((size_t)ctx_ * half);
    for (int p = 0; p < ctx_; p++) for (int i = 0; i < half; i++) { const double a = p * inv[i]; cs[(size_t)p * half + i] = (float)(std::cos(a) * conc); sn[(size_t)p * half + i] = (float)(std::sin(a) * conc); }
    cos_ = (float *)dalloc(cs.size() * 4); sin_ = (float *)dalloc(sn.size() * 4);
    CK(cudaMemcpy((void *)cos_, cs.data(), cs.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy((void *)sin_, sn.data(), sn.size() * 4, cudaMemcpyHostToDevice));
    for (int i = 0; i < 3; i++) has_[i] = seen_[i];
    build_graphs();
    size_t fr = 0, tot = 0; CK(cudaMemGetInfo(&fr, &tot));
    if (getenv("QPACK_VERBOSE")) fprintf(stderr, "cuda gptoss: loaded, %zu MiB device memory free of %zu\n", fr >> 20, tot >> 20);
  }
  ~GptOssCuda() override {
    for (void *p : allocs_) cudaFree(p);
    cudaFree(kc_); cudaFree(vc_); cudaFree(arena_); cudaFreeHost(hctl_); cudaFreeHost(hlog_); cudaFreeHost(hxn_); cudaFreeHost(hemb_); cudaFreeHost(htok_);
    if (ga_) cudaGraphExecDestroy(ga_); if (gb_) cudaGraphExecDestroy(gb_); if (gc_) cudaGraphExecDestroy(gc_); cudaStreamDestroy(st_);
  }
  int vocab() const override { return V_; }
  int hidden() const override { return d_; }
  int max_ctx() const override { return ctx_; }
  int pos() const override { return pos_; }
  std::string arch() const override { return "gptoss@cuda"; }
  void reset() override { pos_ = 0; }
  std::vector<uint8_t> checkpoint() const override { std::vector<uint8_t> c(4); memcpy(c.data(), &pos_, 4); return c; }
  void rollback(const std::vector<uint8_t> &c) override { memcpy(&pos_, c.data(), 4); }
  bool rewind(int pos) override { if (pos < 0 || pos > pos_) return false; pos_ = pos; return true; }
  std::vector<uint8_t> snapshot() const override {   // same layout as the CPU model
    std::vector<uint8_t> s(4 + 2ull * L_ * pos_ * kvd_ * 2); memcpy(s.data(), &pos_, 4); uint8_t *p = s.data() + 4; const size_t n = (size_t)pos_ * kvd_ * 2;
    for (int l = 0; l < L_; l++) for (const __half *buf : {kc_, vc_}) { CK(cudaMemcpy(p, buf + (size_t)l * ctx_ * kvd_, n, cudaMemcpyDeviceToHost)); p += n; }
    return s;
  }
  void restore(const std::vector<uint8_t> &s) override {
    memcpy(&pos_, s.data(), 4); const uint8_t *p = s.data() + 4; const size_t n = (size_t)pos_ * kvd_ * 2;
    for (int l = 0; l < L_; l++) for (__half *buf : {kc_, vc_}) { CK(cudaMemcpy(buf + (size_t)l * ctx_ * kvd_, p, n, cudaMemcpyHostToDevice)); p += n; }
  }

  // greedy tokens: the embedding row of each next token comes from the host, so one small round trip per token
  int greedy_run(int tok, int n, int *out) override {
    n = std::min(n, ctx_ - pos_); if (n <= 0) return 0;
    if (tok < 0 || tok >= V_) throw std::runtime_error("token id out of range");
    for (int i = 0; i < n; i++) {
      launch_token(tok); CK(cudaGraphLaunch(gc_, st_)); CK(cudaMemcpyAsync(htok_, tok_, 4, cudaMemcpyDeviceToHost, st_)); CK(cudaStreamSynchronize(st_));
      pending_ = 0; out[i] = tok = *htok_;
    }
    return n;
  }
  const float *step(int tok, int want) override {
    if (pos_ >= ctx_) throw std::runtime_error("context window full (" + std::to_string(ctx_) + " tokens)");
    if (tok < 0 || tok >= V_) throw std::runtime_error("token id out of range");
    launch_token(tok);
    if (!want) { if (++pending_ >= RING / 2) { CK(cudaStreamSynchronize(st_)); pending_ = 0; } return nullptr; }
    CK(cudaGraphLaunch(gb_, st_));
    if (want == 2) { CK(cudaMemcpyAsync(hxn_, xn_, (size_t)d_ * 4, cudaMemcpyDeviceToHost, st_)); CK(cudaStreamSynchronize(st_)); pending_ = 0; return hxn_; }
    CK(cudaMemcpyAsync(hlog_, logits_, (size_t)V_ * 4, cudaMemcpyDeviceToHost, st_)); CK(cudaStreamSynchronize(st_)); pending_ = 0;
    return hlog_;
  }

 private:
  static constexpr int RING = 256;
  // ---- uploads
  void *dalloc(size_t n) { void *p; CK(cudaMalloc(&p, n)); allocs_.push_back(p); return p; }
  static size_t tensor_bytes(const Tensor &t) {
    const size_t R = t.rows(), C = t.cols();
    switch (t.kind) {
      case Kind::Q8G: case Kind::Q8: return R * C + R * C / 32 * 2;
      case Kind::Mxfp4: return R * C / 2 + R * C / 32;
      case Kind::Q1: return R * ((C + 127) / 128) * 18;
      case Kind::F32: return 4 * (size_t)t.numel();
      case Kind::F16: return t.shape.size() == 2 ? 0 : 4 * (size_t)t.numel();
      default: return 0;
    }
  }
  const float *upf(const Tensor &t) {
    void *p = dalloc(4 * (size_t)t.numel()); CK(cudaMemcpy(p, t.f32(), 4 * (size_t)t.numel(), cudaMemcpyHostToDevice));
    if (t.kind != Kind::F32 && !(t.kind == Kind::F16 && t.shape.size() != 2)) throw std::runtime_error("cuda gptoss: expected a float tensor: " + t.name);
    return (const float *)p;
  }
  void copy_chunked(void *dst, const void *src, size_t n) { for (size_t o = 0; o < n; o += (64u << 20)) CK(cudaMemcpy((char *)dst + o, (const char *)src + o, std::min<size_t>(64u << 20, n - o), cudaMemcpyHostToDevice)); }
  DQ up_q8g(const Tensor &t) {
    if (t.kind != Kind::Q8G && t.kind != Kind::Q8) throw std::runtime_error("cuda gptoss: expected a q8g (or per-row q8) tensor: " + t.name);
    const size_t R = t.rows(), C = t.cols(), U = C / 32; DQ m{};
    m.rows = (int)R; void *w = dalloc(R * C); void *s = dalloc(R * U * 2);
    if (t.kind == Kind::Q8G) copy_chunked(s, t.scale16, R * U * 2);
    else {   // per-row q8: the row scale (rounded to fp16) replicated per group
      std::vector<__half> hs(R * U); for (size_t r = 0; r < R; r++) for (size_t u = 0; u < U; u++) hs[r * U + u] = __float2half(t.scale[r]);
      copy_chunked(s, hs.data(), R * U * 2);
    }
    const size_t CR = std::max<size_t>(1, (16u << 20) / C); std::vector<int8_t> buf(CR * C); const int8_t *src = (const int8_t *)t.data;
    for (size_t r0 = 0; r0 < R; r0 += CR) {
      const size_t nr = std::min(CR, R - r0);
      for (size_t r = 0; r < nr; r++) for (size_t u = 0; u < U; u++) {   // unit permutation: new[16h + 4j + m] = old[16h + 4m + j]
        const int8_t *o = src + (r0 + r) * C + u * 32; int8_t *nw = buf.data() + r * C + u * 32;
        for (int h = 0; h < 2; h++) for (int j = 0; j < 4; j++) for (int m = 0; m < 4; m++) nw[16 * h + 4 * j + m] = o[16 * h + 4 * m + j];
      }
      CK(cudaMemcpy((char *)w + r0 * C, buf.data(), nr * C, cudaMemcpyHostToDevice));
    }
    m.w = (const int4 *)w; m.s = (const unsigned short *)s; return m;
  }
  struct DT { char *q = nullptr, *s = nullptr; int kind = -1; size_t R = 0, C = 0; };   // an uploaded stacked expert tensor [n * R, C]
  DT up_exp(const Tensor &t) {
    DT d; d.R = t.rows(); d.C = t.cols();
    if (t.kind == Kind::Mxfp4) {
      d.kind = 0; const size_t R = d.R, C = d.C; d.q = (char *)dalloc(R * C / 2); d.s = (char *)dalloc(R * C / 32); copy_chunked(d.s, t.scale8, R * C / 32);
      const size_t CR = std::max<size_t>(1, (16u << 20) / (C / 2)); std::vector<uint8_t> buf(CR * C / 2); const uint8_t *src = (const uint8_t *)t.data;
      for (size_t r0 = 0; r0 < R; r0 += CR) {
        const size_t nr = std::min(CR, R - r0);
        for (size_t r = 0; r < nr; r++) for (size_t u = 0; u < C / 32; u++) {   // word k: nibble m = low nibble of byte k + 4m, nibble 4 + m = its high nibble
          const uint8_t *o = src + (r0 + r) * C / 2 + u * 16; uint32_t *nw = (uint32_t *)(buf.data() + r * C / 2 + u * 16);
          for (int k = 0; k < 4; k++) { uint32_t w = 0; for (int m = 0; m < 4; m++) { const uint8_t b = o[k + 4 * m]; w |= (uint32_t)(b & 15) << (4 * m); w |= (uint32_t)(b >> 4) << (16 + 4 * m); } nw[k] = w; }
        }
        CK(cudaMemcpy(d.q + r0 * C / 2, buf.data(), nr * C / 2, cudaMemcpyHostToDevice));
      }
    } else if (t.kind == Kind::Q1) {
      d.kind = 1; const size_t G = (d.C + 127) / 128; d.q = (char *)dalloc(d.R * G * 16); d.s = (char *)dalloc(d.R * G * 2);
      copy_chunked(d.q, t.data, d.R * G * 16); copy_chunked(d.s, t.scale16, d.R * G * 2);
    } else throw std::runtime_error("cuda gptoss: expert tensors must be mxfp4 or q1 (got " + t.name + ")");
    return d;
  }
  // per-expert pointers of expert j of a stacked tensor with n experts
  void plane(const DT &d, int j, int n, const void **q, const void **s) const {
    const size_t R = d.R / n;
    if (d.kind == 0) { *q = d.q + (size_t)j * R * d.C / 2; *s = d.s + (size_t)j * R * d.C / 32; }
    else { const size_t G = (d.C + 127) / 128; *q = d.q + (size_t)j * R * G * 16; *s = d.s + (size_t)j * R * G * 2; }
  }
  void build_experts(int l, Layer &ly) {
    const std::string ls = std::to_string(l); const Json &mx = cfg_["moe_exact"];
    std::vector<int> exact; if (mx.has(ls)) for (auto &x : mx[ls].arr) exact.push_back((int)x.as_int());
    auto T = [&](const char *n) { return c_->get("layers." + ls + "." + n); };
    const Tensor *gu = T("w_gate_up"), *dn = T("w_down"), *gur = T("w_gate_up_r2"), *dnr = T("w_down_r2"), *gux = T("w_gate_up_x"), *dnx = T("w_down_x");
    DT dgu, ddn, dgur, ddnr, dgux, ddnx;
    std::vector<int> r2ids; const bool r2sel = cfg_["moe_r2"].has(ls); if (r2sel) for (auto &x : cfg_["moe_r2"][ls].arr) r2ids.push_back((int)x.as_int());
    const int nq = E_ - (int)exact.size(), nx = (int)exact.size();
    if (nq > 0) { dgu = up_exp(*gu); ddn = up_exp(*dn); if (gur) dgur = up_exp(*gur); if (dnr) ddnr = up_exp(*dnr); }
    if (nx > 0) { if (!gux || !dnx) throw std::runtime_error("cuda gptoss: exact experts listed without w_gate_up_x / w_down_x in layer " + ls); dgux = up_exp(*gux); ddnx = up_exp(*dnx); }
    std::vector<ExpD> tab(E_); int iq = 0;
    for (int e = 0; e < E_; e++) {
      ExpD D{}; auto it = std::find(exact.begin(), exact.end(), e);
      if (it != exact.end()) {
        const int j = (int)(it - exact.begin()); D.kind = dgux.kind; D.np = 1;
        plane(dgux, j, nx, &D.gq[0], &D.gs[0]); plane(ddnx, j, nx, &D.dq[0], &D.ds[0]);
      } else {
        D.kind = dgu.kind; D.np = 1; plane(dgu, iq, nq, &D.gq[0], &D.gs[0]); plane(ddn, iq, nq, &D.dq[0], &D.ds[0]);
        if (gur) {   // second plane for every non-exact expert, or (config "moe_r2") only for the listed ones, rows in that order
          if (dgur.kind != 1 || D.kind != 1) throw std::runtime_error("cuda gptoss: second planes must be q1");
          if (!dnr) throw std::runtime_error("cuda gptoss: *_r2 mismatch");
          int j = iq, nj = nq;
          if (r2sel) { auto r = std::find(r2ids.begin(), r2ids.end(), e); j = r == r2ids.end() ? -1 : (int)(r - r2ids.begin()); nj = (int)r2ids.size(); }
          if (j >= 0) { D.np = 2; plane(dgur, j, nj, &D.gq[1], &D.gs[1]); plane(ddnr, j, nj, &D.dq[1], &D.ds[1]); }
        }
        iq++;
      }
      if (D.kind == 0 && (c_->get("layers." + ls + ".w_gate_up_x") == nullptr && gu->kind != Kind::Mxfp4)) throw std::runtime_error("cuda gptoss: expert kind");
      tab[e] = D; seen_[D.kind == 0 ? 0 : D.np == 1 ? 1 : 2] = true;
    }
    ExpD *dt = (ExpD *)dalloc(sizeof(ExpD) * E_); CK(cudaMemcpy(dt, tab.data(), sizeof(ExpD) * E_, cudaMemcpyHostToDevice)); ly.tab = dt;
  }

  // ---- one token
  void launch_token(int tok) {
    Ctl &c = hctl_[slot_ % RING]; c.tok = tok; c.pos = pos_; float *e = hemb_ + (size_t)(slot_ % RING) * d_; slot_++;
    row_f32(*embed_, tok, e);
    CK(cudaMemcpyAsync(ctl_, &c, sizeof(Ctl), cudaMemcpyHostToDevice, st_)); CK(cudaMemcpyAsync(x_, e, (size_t)d_ * 4, cudaMemcpyHostToDevice, st_));
    CK(cudaGraphLaunch(ga_, st_)); pos_++;
  }
  void dense(cudaStream_t s, const DQ *m, const float *const *bias, float *const *out, int nm, int units, const int4 *aq, const float *as, const int *asum, float *resid) {
    DenseArgs a{}; a.nm = nm; a.units = units; a.aq = aq; a.as = as; a.asum = asum; a.resid = resid; int tot = 0;
    for (int i = 0; i < nm; i++) { a.m[i] = m[i]; a.bias[i] = bias ? bias[i] : nullptr; a.out[i] = out ? out[i] : nullptr; a.rowstart[i] = tot; tot += m[i].rows; }
    a.rowstart[nm] = tot;
    if (units <= 96) k_dense<3, 8><<<(tot + 7) / 8, 256, 0, s>>>(a); else k_dense<4, 8><<<(tot + 7) / 8, 256, 0, s>>>(a);
  }
  void launch_gateup(cudaStream_t s, const ExpD *tab, const float *bgu, const int4 *aA) {
    const dim3 g(ff_ / 32, NK);
    if (has_[0]) k_gateup<0><<<g, GUT, 0, s>>>(tab, sel_, bgu, limit_, alpha_, ud_, ff_, aA, aA_s_, aA_sum_, aH_q_, aH_s_, aH_sum_);
    if (has_[1]) k_gateup<1><<<g, GUT, 0, s>>>(tab, sel_, bgu, limit_, alpha_, ud_, ff_, aA, aA_s_, aA_sum_, aH_q_, aH_s_, aH_sum_);
    if (has_[2]) k_gateup<2><<<g, GUT, 0, s>>>(tab, sel_, bgu, limit_, alpha_, ud_, ff_, aA, aA_s_, aA_sum_, aH_q_, aH_s_, aH_sum_);
  }
  bool has_[3] = {false, false, false}, seen_[3] = {false, false, false};
  void build_graphs() {
    const float scl = 1.f / std::sqrt((float)hd_); cudaStream_t s = st_;
    const int4 *aA = (const int4 *)aA_q_, *aO = (const int4 *)aO_q_, *aH = (const int4 *)aH_q_;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int l = 0; l < L_; l++) {
      const Layer &ly = layers_[l]; __half *kl = kc_ + (size_t)l * ctx_ * kvd_, *vl = vc_ + (size_t)l * ctx_ * kvd_;
      k_norm<<<1, 1024, 0, s>>>(x_, ly.an, l ? ye_ : nullptr, d_, eps_, aA_q_, aA_s_, aA_sum_, nullptr, nullptr, nullptr, 0, nullptr, nullptr);
      { DQ m[3] = {ly.wq, ly.wk, ly.wv}; const float *b[3] = {ly.bq, ly.bk, ly.bv}; float *o[3] = {q_, k_, v_}; dense(s, m, b, o, 3, ud_, aA, aA_s_, aA_sum_, nullptr); }
      const int win = ly.sliding ? window_ : 0;
      k_attn<<<dim3(nkv_, ly.sliding ? (window_ + CH - 1) / CH : nch_), ANT, 0, s>>>(q_, k_, v_, cos_, sin_, kl, vl, part_, ctl_, nkv_, nch_, scl, win);
      k_merge<<<nh_, HD, 0, s>>>(part_, ly.sinks, ctl_, nch_, win, aO_q_, aO_s_, aO_sum_);
      { DQ m[1] = {ly.wo}; dense(s, m, &ly.bo, nullptr, 1, qd_ / 32, aO, aO_s_, aO_sum_, x_); }
      k_norm<<<1, 1024, 0, s>>>(x_, ly.fn, nullptr, d_, eps_, aA_q_, aA_s_, aA_sum_, nullptr, ly.rw, ly.rb, E_, sel_, wt_);
      launch_gateup(s, ly.tab, ly.bgu, aA);
      { const dim3 g(d_ / 8, NK);
        if (has_[0]) k_down<0><<<g, 256, 0, s>>>(ly.tab, sel_, wt_, ly.bd, uff_, d_, aH, aH_s_, aH_sum_, ye_);
        if (has_[1]) k_down<1><<<g, 256, 0, s>>>(ly.tab, sel_, wt_, ly.bd, uff_, d_, aH, aH_s_, aH_sum_, ye_);
        if (has_[2]) k_down<2><<<g, 256, 0, s>>>(ly.tab, sel_, wt_, ly.bd, uff_, d_, aH, aH_s_, aH_sum_, ye_); }
    }
    cudaGraph_t ga; CK(cudaStreamEndCapture(s, &ga)); CK(cudaGraphInstantiate(&ga_, ga, nullptr, nullptr, 0)); CK(cudaGraphDestroy(ga));
    for (int variant = 0; variant < 2; variant++) {
      CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
      k_norm<<<1, 1024, 0, s>>>(x_, fnorm_, L_ ? ye_ : nullptr, d_, eps_, aA_q_, aA_s_, aA_sum_, variant == 0 ? xn_ : nullptr, nullptr, nullptr, 0, nullptr, nullptr);
      { DQ m[1] = {head_}; float *o[1] = {logits_}; dense(s, m, nullptr, o, 1, ud_, aA, aA_s_, aA_sum_, nullptr); }
      if (variant == 1) { k_argmax1<<<AM_BLOCKS, 1024, 0, s>>>(logits_, V_, amv_, ami_); k_argmax2<<<1, 64, 0, s>>>(amv_, ami_, tok_); }
      cudaGraph_t g; CK(cudaStreamEndCapture(s, &g)); CK(cudaGraphInstantiate(variant == 0 ? &gb_ : &gc_, g, nullptr, nullptr, 0)); CK(cudaGraphDestroy(g));
    }
  }

  int d_, L_, nh_, nkv_, hd_, ff_, V_, E_, K_, ctx_, kvd_, qd_, ud_, uff_, nch_, window_, pos_ = 0, slot_ = 0, pending_ = 0; float eps_, limit_, alpha_;
  cudaStream_t st_ = nullptr; cudaGraphExec_t ga_ = nullptr, gb_ = nullptr, gc_ = nullptr;
  const Tensor *embed_ = nullptr; DQ head_{}; const float *fnorm_ = nullptr; std::vector<Layer> layers_; std::vector<void *> allocs_;
  __half *kc_ = nullptr, *vc_ = nullptr; void *arena_ = nullptr;
  float *x_, *xn_, *q_, *k_, *v_, *logits_, *part_, *ye_, *wt_, *amv_, *aA_s_, *aO_s_, *aH_s_; const float *cos_, *sin_; int *sel_, *ami_, *tok_, *aA_sum_, *aO_sum_, *aH_sum_;
  int8_t *aA_q_, *aO_q_, *aH_q_;
  Ctl *ctl_, *hctl_; float *hlog_, *hxn_, *hemb_; int *htok_;
};

}  // namespace

// expert matvec microbenchmark: 32 distinct synthetic experts of a layer (gate_up 5760x2880 + down 2880x2880), 4 per launch, rotating selections
void cuda_gptoss_bench(FILE *out) {
  const int E = 32, d = 2880, ff = 2880, units = 90; cudaStream_t st; CK(cudaStreamCreate(&st)); cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  int8_t *aq; float *as, *ye, *hs, *wt, *bgu, *bd; int *asum, *hsum, *sel; int8_t *hq;
  CK(cudaMalloc(&aq, ABYTES)); CK(cudaMalloc(&as, UMAX * 4)); CK(cudaMalloc(&asum, UMAX * 4)); CK(cudaMalloc(&hq, NK * ABYTES)); CK(cudaMalloc(&hs, NK * UMAX * 4)); CK(cudaMalloc(&hsum, NK * UMAX * 4));
  CK(cudaMalloc(&ye, NK * d * 4)); CK(cudaMalloc(&wt, 16)); CK(cudaMalloc(&bgu, (size_t)E * 2 * ff * 4)); CK(cudaMalloc(&bd, (size_t)E * d * 4)); CK(cudaMalloc(&sel, 8 * NK * 4));
  CK(cudaMemset(aq, 0x11, ABYTES)); CK(cudaMemset(bgu, 0, (size_t)E * 2 * ff * 4)); CK(cudaMemset(bd, 0, (size_t)E * d * 4)); CK(cudaMemset(hq, 0x11, NK * ABYTES));
  { std::vector<float> f(UMAX, 1e-4f); CK(cudaMemcpy(as, f.data(), UMAX * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(hs, std::vector<float>(NK * UMAX, 1e-4f).data(), NK * UMAX * 4, cudaMemcpyHostToDevice)); float w[4] = {.25f, .25f, .25f, .25f}; CK(cudaMemcpy(wt, w, 16, cudaMemcpyHostToDevice)); }
  CK(cudaMemset(asum, 0, UMAX * 4)); CK(cudaMemset(hsum, 0, NK * UMAX * 4));
  { int h[32]; for (int i = 0; i < 8; i++) for (int k = 0; k < 4; k++) h[i * 4 + k] = (i * 4 + k) % E; CK(cudaMemcpy(sel, h, sizeof h, cudaMemcpyHostToDevice)); }
  struct V { const char *n; int kind, np; } vs[] = {{"mxfp4", 0, 1}, {"q1", 1, 1}, {"q1+r2", 1, 2}};
  fprintf(out, "%-8s %-8s %10s %9s %8s %7s\n", "fmt", "matrix", "MB/launch", "us", "GB/s", "%150");
  for (auto &v : vs) for (int mt = 0; mt < 2; mt++) {
    const size_t R = mt == 0 ? 2 * ff : d, C = 2880, G = (C + 127) / 128;
    const size_t qb = v.kind == 0 ? R * C / 2 : R * G * 16, sb = v.kind == 0 ? R * C / 32 : R * G * 2;
    std::vector<ExpD> tab(E); std::vector<void *> mem;
    for (int e = 0; e < E; e++) {
      ExpD D{}; D.kind = v.kind; D.np = v.np;
      for (int p = 0; p < v.np; p++) {
        void *q, *sc; CK(cudaMalloc(&q, qb)); CK(cudaMalloc(&sc, sb)); CK(cudaMemset(q, 0x5a + e + p, qb)); CK(cudaMemset(sc, v.kind == 0 ? 0x7c : 0x04, sb)); mem.push_back(q); mem.push_back(sc);
        if (mt == 0) { D.gq[p] = q; D.gs[p] = sc; } else { D.dq[p] = q; D.ds[p] = sc; }
      }
      tab[e] = D;
    }
    ExpD *dt; CK(cudaMalloc(&dt, sizeof(ExpD) * E)); CK(cudaMemcpy(dt, tab.data(), sizeof(ExpD) * E, cudaMemcpyHostToDevice));
    auto run = [&](int i) {
      if (mt == 0) { const dim3 g(ff / 32, NK); const int *sl = sel + 4 * (i & 7); const int4 *a4 = (const int4 *)aq;
        if (v.kind == 0) k_gateup<0><<<g, GUT, 0, st>>>(dt, sl, bgu, 7.f, 1.702f, units, ff, a4, as, asum, hq, hs, hsum);
        else if (v.np == 1) k_gateup<1><<<g, GUT, 0, st>>>(dt, sl, bgu, 7.f, 1.702f, units, ff, a4, as, asum, hq, hs, hsum);
        else k_gateup<2><<<g, GUT, 0, st>>>(dt, sl, bgu, 7.f, 1.702f, units, ff, a4, as, asum, hq, hs, hsum); }
      else { const dim3 g(d / 8, NK); const int *sl = sel + 4 * (i & 7); const int4 *h4 = (const int4 *)hq;
        if (v.kind == 0) k_down<0><<<g, 256, 0, st>>>(dt, sl, wt, bd, units, d, h4, hs, hsum, ye);
        else if (v.np == 1) k_down<1><<<g, 256, 0, st>>>(dt, sl, wt, bd, units, d, h4, hs, hsum, ye);
        else k_down<2><<<g, 256, 0, st>>>(dt, sl, wt, bd, units, d, h4, hs, hsum, ye); }
    };
    for (int i = 0; i < 300; i++) run(i);
    CK(cudaStreamSynchronize(st)); float best = 1e9;
    for (int rep = 0; rep < 5; rep++) { CK(cudaEventRecord(e0, st)); for (int i = 0; i < 16; i++) run(i); CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); best = std::min(best, ms / 16); }
    const double bytes = (double)NK * v.np * (qb + sb), us = best * 1e3, gbs = bytes / (us * 1e-6) / 1e9;
    fprintf(out, "%-8s %-8s %10.2f %9.1f %8.1f %6.1f%%\n", v.n, mt == 0 ? "gate_up" : "down", bytes / 1048576.0, us, gbs, 100 * gbs / 150);
    for (void *m : mem) cudaFree(m);
    cudaFree(dt);
  }
}

std::unique_ptr<Model> make_gptoss_cuda(std::shared_ptr<Container> c, int max_ctx) { return std::unique_ptr<Model>(new GptOssCuda(c, max_ctx)); }

}  // namespace qp
