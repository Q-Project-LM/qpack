// Llama-family decoder on a CUDA GPU (built with -DQPACK_CUDA=ON, selected with QPACK_DEVICE=cuda). Same model as
// model_llama.cpp: the quantised weights (q8 / q4 / q1) are uploaded once, the KV cache (fp16) and all activations stay
// on the device, and one token is a single CUDA graph launch. Written for Pascal (sm_61, Tesla P4): no fp16 math (1/64
// rate), all arithmetic is fp32 / int8 dp4a.
//
// Matvec: one warp per output row; activations are quantised to int8 with a scale per 32-column group, weights are used
// as stored (q8 per-row scale, q4 groups of 32 with fp16 scale, q1 sign bits groups of 128 with fp16 scale).
// Per layer: norm+quant | qkv (one launch) | rope+kv write | attention (split over KV chunks) | merge+quant | o + residual |
// norm+quant | gate+up+SiLU+quant (one launch) | down + residual.
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

// activations: int16 per 32-column group (scale = absmax / 32767), stored as two int8 planes (hi signed, lo unsigned: q = 256 hi + lo)
// at xq[0, cols) and xq[PLANE, PLANE + cols); a dot product costs two dp4a per 4 columns. (One int8 plane has ~1% relative error
// per matrix, which adds up to ~0.1 logit units over 24 layers.)
constexpr int PLANE = 8192;
constexpr int UN = 1;   // independent weight loads in flight per lane
enum { K_Q8 = 0, K_Q4 = 1, K_Q1 = 2, K_F16 = 3, K_F32 = 4 };
struct DMat { const void *q; const void *s; int rows, cols, kind; };

// ------------------------------------------------------------------ device helpers
__device__ __forceinline__ int dp4a_us(unsigned a, int b, int c) { int d; asm("dp4a.u32.s32 %0,%1,%2,%3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }
__device__ __forceinline__ int dp4a_uu(unsigned a, unsigned b, int c) { int d; asm("dp4a.u32.u32 %0,%1,%2,%3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }
__device__ __forceinline__ int dp4a_su(int a, unsigned b, int c) { int d; asm("dp4a.s32.u32 %0,%1,%2,%3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }
__device__ __forceinline__ float h2f(unsigned short h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ float warp_sum(float v) { for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o); return v; }
__device__ __forceinline__ unsigned nib_mask(unsigned n) { return ((n * 0x00204081u) & 0x01010101u) * 255u; }   // 4 bits -> 4 bytes 0x00 / 0xFF

// quantise one group of 32 values (one per lane) to int8
__device__ __forceinline__ void warp_actq(float v, int g, int l, int8_t *xq, float *xs, int *xsum, int np) {
  float m = fabsf(v); for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
  const int qm = np == 2 ? 32767 : 127;
  const float s = m / (float)qm, inv = s > 0 ? 1.f / s : 0.f;
  int q = __float2int_rn(v * inv); q = max(-qm, min(qm, q));
  if (np == 2) { xq[g * 32 + l] = (int8_t)(q >> 8); xq[PLANE + g * 32 + l] = (int8_t)(q & 255); } else xq[g * 32 + l] = (int8_t)q;
  int sm = q; for (int o = 16; o; o >>= 1) sm += __shfl_xor_sync(~0u, sm, o);
  if (l == 0) { xs[g] = s; xsum[g] = sm; }
}

// dot product of weight row r with the quantised activations in shared memory; result in all lanes
__device__ __forceinline__ float row_dot(const DMat &m, int r, int l, const int4 *sx, const int4 *sl, const float *sxs, const int *sxsum, int np) {
  const int cols = m.cols; float acc = 0;
  if (m.kind == K_Q4) {
    const int G = cols >> 5;
    const uint4 *w = (const uint4 *)((const uint8_t *)m.q + (size_t)r * (cols >> 1)); const unsigned short *s = (const unsigned short *)m.s + (size_t)r * G;
    for (int gb = 0; gb < G; gb += 32 * UN) {
      uint4 vv[UN];
#pragma unroll
      for (int u = 0; u < UN; u++) { const int g = gb + l + 32 * u; if (g < G) vv[u] = __ldcs(w + g); }
#pragma unroll
      for (int u = 0; u < UN; u++) {
        const int g = gb + l + 32 * u; if (g >= G) break;
        const uint4 v = vv[u]; const int4 xl = sx[2 * g], xh = sx[2 * g + 1], ll = np == 2 ? sl[2 * g] : xl, lh = np == 2 ? sl[2 * g + 1] : xh; const unsigned M = 0x0F0F0F0Fu;
        const unsigned w0 = v.x & M, w1 = v.y & M, w2 = v.z & M, w3 = v.w & M, w4 = (v.x >> 4) & M, w5 = (v.y >> 4) & M, w6 = (v.z >> 4) & M, w7 = (v.w >> 4) & M;
        int d = dp4a_us(w0, xl.x, 0); d = dp4a_us(w1, xl.y, d); d = dp4a_us(w2, xl.z, d); d = dp4a_us(w3, xl.w, d);
        d = dp4a_us(w4, xh.x, d); d = dp4a_us(w5, xh.y, d); d = dp4a_us(w6, xh.z, d); d = dp4a_us(w7, xh.w, d);
        if (np == 2) {
          int e = dp4a_uu(w0, (unsigned)ll.x, 0); e = dp4a_uu(w1, (unsigned)ll.y, e); e = dp4a_uu(w2, (unsigned)ll.z, e); e = dp4a_uu(w3, (unsigned)ll.w, e);
          e = dp4a_uu(w4, (unsigned)lh.x, e); e = dp4a_uu(w5, (unsigned)lh.y, e); e = dp4a_uu(w6, (unsigned)lh.z, e); e = dp4a_uu(w7, (unsigned)lh.w, e);
          d = (d << 8) + e;
        }
        d -= 8 * sxsum[g];
        acc += (float)d * (sxs[g] * h2f(__ldg(s + g)));
      }
    }
  } else if (m.kind == K_Q8) {
    const int nch = cols >> 4; const uint4 *w = (const uint4 *)((const int8_t *)m.q + (size_t)r * cols);
    for (int cb = 0; cb < nch; cb += 32 * UN) {
      uint4 vv[UN];
#pragma unroll
      for (int u = 0; u < UN; u++) { const int c = cb + l + 32 * u; if (c < nch) vv[u] = __ldcs(w + c); }
#pragma unroll
      for (int u = 0; u < UN; u++) {
        const int c = cb + l + 32 * u; if (c >= nch) break;
        const uint4 v = vv[u]; const int4 x = sx[c], y = np == 2 ? sl[c] : x;
        int d = __dp4a((int)v.x, x.x, 0); d = __dp4a((int)v.y, x.y, d); d = __dp4a((int)v.z, x.z, d); d = __dp4a((int)v.w, x.w, d);
        if (np == 2) {
          int e = dp4a_su((int)v.x, (unsigned)y.x, 0); e = dp4a_su((int)v.y, (unsigned)y.y, e); e = dp4a_su((int)v.z, (unsigned)y.z, e); e = dp4a_su((int)v.w, (unsigned)y.w, e);
          d = (d << 8) + e;
        }
        acc += (float)d * sxs[c >> 1];
      }
    }
    acc *= ((const float *)m.s)[r];
  } else {   // q1: lane takes 32 columns (4 bytes of sign bits); sum(x * +-1) = 2 * sum(x[bit=1]) - sum(x)
    const int C = cols >> 5; const unsigned *w = (const unsigned *)((const uint8_t *)m.q + (size_t)r * (cols >> 3));
    const unsigned short *s = (const unsigned short *)m.s + (size_t)r * (cols >> 7); const int *xi = (const int *)sx, *xj = (const int *)sl;
    for (int c = l; c < C; c += 32) {
      const unsigned bits = __ldg(w + c); int s1 = 0, s2 = 0;
#pragma unroll
      for (int i = 0; i < 8; i++) { const unsigned mk = nib_mask((bits >> (4 * i)) & 15u); s1 = __dp4a((int)(xi[8 * c + i] & mk), 0x01010101, s1); if (np == 2) s2 = dp4a_uu((unsigned)xj[8 * c + i] & mk, 0x01010101u, s2); }
      acc += (float)(2 * (np == 2 ? (s1 << 8) + s2 : s1) - sxsum[c]) * (sxs[c] * h2f(__ldg(s + (c >> 2))));
    }
  }
  return warp_sum(acc);
}

// ------------------------------------------------------------------ kernels
struct Ctl { int tok, pos; };

__global__ void k_embed(DMat e, const Ctl *ctl, float *x) {
  const int r = ctl->tok, n = e.cols;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float v;
    if (e.kind == K_Q8) v = (float)((const int8_t *)e.q)[(size_t)r * n + i] * ((const float *)e.s)[r];
    else if (e.kind == K_Q4) { const uint8_t b = ((const uint8_t *)e.q)[(size_t)r * (n >> 1) + (i >> 5) * 16 + (i & 15)]; const int nib = (i & 16) ? (b >> 4) : (b & 15); v = (float)(nib - 8) * h2f(((const unsigned short *)e.s)[(size_t)r * (n >> 5) + (i >> 5)]); }
    else if (e.kind == K_F16) v = h2f(((const unsigned short *)e.q)[(size_t)r * n + i]);
    else v = ((const float *)e.q)[(size_t)r * n + i];
    x[i] = v;
  }
}

// y = rmsnorm(x) * w, quantised per 32 columns (one block of 1024 threads, n <= 1024); optionally also stored as float
__global__ void __launch_bounds__(1024) k_norm_actq(const float *x, const float *w, int n, float eps, int8_t *xq, float *xs, int *xsum, float *yf, int np) {
  __shared__ float red[32]; __shared__ float sy[1024];
  const int t = threadIdx.x, wp = t >> 5, l = t & 31;
  const float xv = t < n ? x[t] : 0.f;
  float ss = warp_sum(xv * xv); if (l == 0) red[wp] = ss; __syncthreads();
  float tot = 0; for (int i = 0; i < 32; i++) tot += red[i];
  const float inv = 1.f / sqrtf(tot / n + eps);
  if (t < n) { const float y = xv * inv * w[t]; sy[t] = y; if (yf) yf[t] = y; }
  __syncthreads();
  if (wp < (n >> 5)) warp_actq(sy[t], wp, l, xq, xs, xsum, np);
}

struct MvArgs {
  DMat m[3]; const float *bias[3]; float *out[3]; int nm; int rowstart[4];
  const int8_t *xq; const float *xs; const int *xsum; int cols, np;
  float *resid;   // non-null: out = resid += y
  const Ctl *ctl; const char *pfk, *pfv; int kvd;   // optional: prefetch the K / V cache of this layer (positions 0..pos) into L2
};
// up to three matrices sharing one input vector; warp per row
template <int WPB> __global__ void __launch_bounds__(WPB * 32) k_mv(MvArgs a) {
  extern __shared__ int4 smem[];
  if (a.pfk) {
    const size_t bytes = (size_t)(a.ctl->pos + 1) * a.kvd * 2, nl = (bytes + 127) / 128;
    for (size_t i = (size_t)blockIdx.x * (WPB * 32) + threadIdx.x; i < nl; i += (size_t)gridDim.x * (WPB * 32)) {
      asm volatile("prefetch.global.L2 [%0];" :: "l"(a.pfk + i * 128)); asm volatile("prefetch.global.L2 [%0];" :: "l"(a.pfv + i * 128));
    }
  }
  const int cols = a.cols, G = cols >> 5;
  int4 *sx = smem, *sl = smem + (cols >> 4); float *sxs = (float *)(smem + 2 * (cols >> 4)); int *sxsum = (int *)(sxs + G);
  for (int i = threadIdx.x; i < (cols >> 4); i += WPB * 32) { sx[i] = ((const int4 *)a.xq)[i]; sl[i] = ((const int4 *)(a.xq + PLANE))[i]; }
  for (int i = threadIdx.x; i < G; i += WPB * 32) { sxs[i] = a.xs[i]; sxsum[i] = a.xsum[i]; }
  __syncthreads();
  int r = blockIdx.x * WPB + (threadIdx.x >> 5); const int l = threadIdx.x & 31;
  if (r >= a.rowstart[a.nm]) return;
  int i = 0; while (r >= a.rowstart[i + 1]) i++;
  r -= a.rowstart[i];
  float y = row_dot(a.m[i], r, l, sx, sl, sxs, sxsum, a.np);
  if (l == 0) {
    if (a.bias[i]) y += a.bias[i][r];
    if (a.resid) a.resid[r] += y; else a.out[i][r] = y;
  }
}

// gate and up rows for 32 consecutive output rows per block (NW warps, 32 / NW rows each), h = silu(gate) * up, quantised
template <int NW> __global__ void __launch_bounds__(NW * 32) k_gateup(DMat g, DMat u, const int8_t *xq, const float *xs, const int *xsum, int8_t *hq, float *hs, int *hsum, int np, int npo) {
  extern __shared__ int4 smem[];
  const int cols = g.cols, G = cols >> 5;
  int4 *sx = smem, *sl = smem + (cols >> 4); float *sxs = (float *)(smem + 2 * (cols >> 4)); int *sxsum = (int *)(sxs + G); __shared__ float sh[32];
  for (int i = threadIdx.x; i < (cols >> 4); i += NW * 32) { sx[i] = ((const int4 *)xq)[i]; sl[i] = ((const int4 *)(xq + PLANE))[i]; }
  for (int i = threadIdx.x; i < G; i += NW * 32) { sxs[i] = xs[i]; sxsum[i] = xsum[i]; }
  __syncthreads();
  const int wp = threadIdx.x >> 5, l = threadIdx.x & 31;
  for (int i = 0; i < 32 / NW; i++) {
    const int rl = wp * (32 / NW) + i, r = blockIdx.x * 32 + rl;
    const float gg = row_dot(g, r, l, sx, sl, sxs, sxsum, np), uu = row_dot(u, r, l, sx, sl, sxs, sxsum, np);
    if (l == 0) sh[rl] = gg / (1.f + expf(-gg)) * uu;
  }
  __syncthreads();
  if (wp == 0) warp_actq(sh[l], blockIdx.x, l, hq, hs, hsum, npo);
}

__device__ long long g_clk[16];
#define CLK(i) do { if (dbg == 9 && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) g_clk[i] = clock64(); } while (0)
constexpr int CH = 16;    // positions per attention block
constexpr int ANT = 256;   // attention block size: a decode token has too little work per block to hide instruction latency with few threads
// block = (kv group, chunk of CH positions), 256 threads. Scores: CH positions x 16 lanes (HD/16 dims each); values: thread = (dim d,
// head subset). All global loads (q, k / v of the cache chunk, the new k / v, rope tables) are issued up front so their latencies
// overlap. q is rotated on the fly; the position being decoded takes its rotated k / v from shared memory (rounded to fp16 like the
// cache) and the block holding it appends them to the cache.
template <int HD> __global__ void __launch_bounds__(ANT) k_attn(const float *q, const float *kn, const float *vn, const float *cs, const float *sn, const __half *kcc, const __half *vcc,
                                                                float *part, const Ctl *ctl, int rep, int nkv, int nch, float scl, int dbg) {
  __half *kc = (__half *)kcc, *vc = (__half *)vcc;
  constexpr int DPL = HD / 16, GR = ANT / HD, NA = 8 / GR;   // dims per score lane; head groups in the value phase; heads per group
  const int g = blockIdx.x, c = blockIdx.y, pos = ctl->pos, T = pos + 1, t0 = c * CH; if (t0 >= T) return;
  CLK(0);
  const int n = min(CH, T - t0), tid = threadIdx.x, kvd = nkv * HD, half = HD / 2, pl = pos - t0;   // pl in [0, CH) only in the last chunk
  const bool last = pl >= 0 && pl < CH;
  __shared__ __align__(16) float sq[8 * HD]; __shared__ float sc[8][CH]; __shared__ float pm[8], ps[8]; __shared__ float sk[HD], sv[HD];
  const int p = tid / 16, sb = tid % 16, d = tid % HD, grp = tid / HD;
  // ---- loads
  __half2 kh[DPL / 2];
#pragma unroll
  for (int e = 0; e < DPL / 2; e++) kh[e] = __float2half2_rn(0.f);
  if (p < n && p != pl) { const __half2 *kr = (const __half2 *)(kc + (size_t)(t0 + p) * kvd + g * HD + sb * DPL);
#pragma unroll
    for (int e = 0; e < DPL / 2; e++) kh[e] = kr[e]; }
  float vv[CH];
#pragma unroll
  for (int t = 0; t < CH; t++) vv[t] = (t < n && t != pl) ? __half2float(vc[(size_t)(t0 + t) * kvd + g * HD + d]) : 0.f;
  constexpr int NR = (8 * HD / 2 + ANT - 1) / ANT;   // rope work items (head, pair) per thread
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
  // ---- rope, current k / v
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
  CLK(1);
  // ---- scores
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
      for (int j = 0; j < 8; j++) if (j < rep) {
#pragma unroll
        for (int e = 0; e < DPL; e++) s[j] = fmaf(sq[j * HD + sb * DPL + e], kf[e], s[j]);
      }
    }
#pragma unroll
    for (int j = 0; j < 8; j++) { float v = s[j]; for (int o = 1; o < 16; o <<= 1) v += __shfl_xor_sync(~0u, v, o); if (j < rep && sb == 0 && p < n) sc[j][p] = v * scl; }
  }
  __syncthreads();
  CLK(2);
  // ---- softmax: 16 lanes per head
  if (tid / 16 < rep) {
    const int j = tid / 16, l16 = tid & 15; const float v = l16 < n ? sc[j][l16] : -1e30f; float mx = v;
    for (int o = 8; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o, 16));
    const float e = l16 < n ? expf(v - mx) : 0.f; float sum = e;
    for (int o = 8; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o, 16);
    sc[j][l16] = e; if (l16 == 0) { pm[j] = mx; ps[j] = sum; }
  }
  __syncthreads();
  CLK(3);
  // ---- values: thread (d, grp) accumulates heads grp, grp + GR, ...
  float acc[NA];
#pragma unroll
  for (int i = 0; i < NA; i++) acc[i] = 0;
#pragma unroll
  for (int t = 0; t < CH; t++) {
    const float v = t == pl ? sv[d] : vv[t];
#pragma unroll
    for (int i = 0; i < NA; i++) { const int j = grp + i * GR; if (j < rep) acc[i] = fmaf(t < n ? sc[j][t] : 0.f, v, acc[i]); }
  }
#pragma unroll
  for (int i = 0; i < NA; i++) { const int j = grp + i * GR; if (j < rep) { float *pp = part + ((size_t)(g * rep + j) * nch + c) * (HD + 2); pp[2 + d] = acc[i]; if (d == 0) { pp[0] = pm[j]; pp[1] = ps[j]; } } }
  CLK(5);
}

// merge the chunk partials of one head and quantise the result (block = one head, HD threads, one warp per 32 dims)
template <int HD> __global__ void k_merge(const float *part, const Ctl *ctl, int nch, int8_t *xq, float *xs, int *xsum, int np) {
  const int h = blockIdx.x, d = threadIdx.x, T = ctl->pos + 1, nc = (T + CH - 1) / CH;
  const float *ph = part + (size_t)h * nch * (HD + 2); float M = -1e30f;
  for (int c = 0; c < nc; c++) M = fmaxf(M, ph[(size_t)c * (HD + 2)]);
  float L = 0, o = 0;
  for (int c = 0; c < nc; c++) { const float *pc = ph + (size_t)c * (HD + 2); const float f = expf(pc[0] - M); L += pc[1] * f; o += f * pc[2 + d]; }
  o /= L;
  warp_actq(o, h * (HD / 32) + (d >> 5), d & 31, xq, xs, xsum, np);
}

// greedy sampling on the device: per-block argmax (ties -> lowest index, like the host scan), then one block merges the partials,
// stores the token as out[pos], and advances the control block (next token, next position) so tokens chain without the host
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
__global__ void k_argmax2(const float *pv, const int *pi, Ctl *ctl, int *out) {
  float bv = pv[threadIdx.x]; int bi = pi[threadIdx.x];   // 64 threads = 2 warps
  for (int o = 16; o; o >>= 1) { const float ov = __shfl_xor_sync(~0u, bv, o); const int oi = __shfl_xor_sync(~0u, bi, o); if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; } }
  __shared__ float sv[2]; __shared__ int si[2];
  if ((threadIdx.x & 31) == 0) { sv[threadIdx.x >> 5] = bv; si[threadIdx.x >> 5] = bi; }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (sv[1] > sv[0] || (sv[1] == sv[0] && si[1] < si[0])) { sv[0] = sv[1]; si[0] = si[1]; }
    out[ctl->pos] = si[0]; ctl->tok = si[0]; ctl->pos++;
  }
}

// ------------------------------------------------------------------ host
struct Layer { DMat wq, wk, wv, wo, wg, wu, wd; const float *an, *fn, *bq, *bk, *bv; };

class LlamaCuda : public Model {
 public:
  LlamaCuda(std::shared_ptr<Container> c, int max_ctx) {
    c_ = c; cfg_ = c->config();
    d_ = (int)cfg_["hidden_size"].as_int(); L_ = (int)cfg_["num_hidden_layers"].as_int(); nh_ = (int)cfg_["num_attention_heads"].as_int();
    nkv_ = (int)cfg_["num_key_value_heads"].as_int(); hd_ = (int)cfg_["head_dim"].as_int(d_ / nh_); ff_ = (int)cfg_["intermediate_size"].as_int();
    V_ = (int)cfg_["vocab_size"].as_int(); eps_ = (float)cfg_["rms_norm_eps"].as_num(1e-6); const double theta = cfg_["rope_theta"].as_num(10000);
    const bool tie = cfg_["tie_embeddings"].as_bool(false);
    const int limit = (int)cfg_["max_position_embeddings"].as_int(1 << 20);
    ctx_ = max_ctx > 0 ? max_ctx : std::min(4096, limit);
    if (hd_ != 64 && hd_ != 128) throw std::runtime_error("cuda: head_dim must be 64 or 128");
    if (nh_ % nkv_ || nh_ / nkv_ > 8) throw std::runtime_error("cuda: unsupported GQA ratio");
    if (d_ % 32 || ff_ % 32 || d_ > 1024 || (nh_ * hd_) % 32 || (nh_ * hd_) > 4864 || d_ > 4864 || ff_ > PLANE) throw std::runtime_error("cuda: unsupported dimensions");
    kvd_ = nkv_ * hd_; qd_ = nh_ * hd_;
    int dev = 0; CK(cudaGetDevice(&dev)); CK(cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking));
    embed_ = up(c->at("embed"), true); head_ = tie ? embed_ : up(c->at("lm_head"), false);
    if (head_.kind > K_Q1) throw std::runtime_error("cuda: the output head must be q8 / q4 / q1");
    fnorm_ = upf(c->at("final_norm"));
    for (int l = 0; l < L_; l++) {
      auto g = [&](const char *n) { return c->get("layers." + std::to_string(l) + "." + n); };
      auto m = [&](const char *n) { auto *t = g(n); if (!t) throw std::runtime_error("layer " + std::to_string(l) + ": missing " + n); return up(*t, false); };
      Layer ly{m("wq"), m("wk"), m("wv"), m("wo"), m("w_gate"), m("w_up"), m("w_down"), upf(*g("attn_norm")), upf(*g("ffn_norm")), nullptr, nullptr, nullptr};
      if (g("bq")) { ly.bq = upf(*g("bq")); ly.bk = upf(*g("bk")); ly.bv = upf(*g("bv")); }
      if (ly.wq.cols != d_ || ly.wd.cols != ff_ || ly.wo.cols != qd_) throw std::runtime_error("cuda: unexpected tensor shapes");
      layers_.push_back(ly);
    }
    const size_t kvn = (size_t)L_ * ctx_ * kvd_;
    CK(cudaMalloc(&kc_, kvn * 2)); CK(cudaMalloc(&vc_, kvn * 2)); CK(cudaMemset(kc_, 0, kvn * 2)); CK(cudaMemset(vc_, 0, kvn * 2));
    nch_ = (ctx_ + CH - 1) / CH;
    // all small per-token state in one arena (few pages: the TLB entries of scattered small buffers are cold after 350 MB of streamed weights)
    const size_t arena_n = ((size_t)ctx_ * (hd_ / 2) * 8 + (size_t)nh_ * nch_ * (hd_ + 2) * 4 + (size_t)V_ * 4 + (1 << 20) + (size_t)ctx_ * 4 + 4095) & ~(size_t)4095;
    CK(cudaMalloc(&arena_, arena_n)); size_t aoff = 0;
    auto ab = [&](size_t n) { void *r = (char *)arena_ + aoff; aoff += (n + 255) & ~(size_t)255; if (aoff > arena_n) throw std::runtime_error("arena overflow"); return r; };
    auto fl = [&](float **p, size_t n) { *p = (float *)ab(n * 4); };
    fl(&x_, d_); fl(&q_, qd_); fl(&k_, kvd_); fl(&v_, kvd_); fl(&xn_, d_); fl(&logits_, V_); fl(&part_, (size_t)nh_ * nch_ * (hd_ + 2));
    const int maxA = std::max(std::max(d_, qd_), ff_);
    for (int i = 0; i < 2; i++) { aq_[i] = (int8_t *)ab(2 * PLANE); fl(&as_[i], maxA / 32); asum_[i] = (int *)ab(maxA / 32 * 4); }
    ctl_ = (Ctl *)ab(sizeof(Ctl)); amv_ = (float *)ab(AM_BLOCKS * 4); ami_ = (int *)ab(AM_BLOCKS * 4); out_ = (int *)ab((size_t)ctx_ * 4);
    CK(cudaMallocHost(&hctl_, RING * sizeof(Ctl))); CK(cudaMallocHost(&hlog_, (size_t)V_ * 4)); CK(cudaMallocHost(&hxn_, (size_t)d_ * 4)); CK(cudaMallocHost(&hout_, (size_t)ctx_ * 4));
    const int half = hd_ / 2; std::vector<float> cs((size_t)ctx_ * half), sn((size_t)ctx_ * half);
    for (int p = 0; p < ctx_; p++) for (int i = 0; i < half; i++) { const double inv = 1.0 / std::pow(theta, (2.0 * i) / hd_), a = p * inv; cs[(size_t)p * half + i] = (float)std::cos(a); sn[(size_t)p * half + i] = (float)std::sin(a); }
    cos_ = (float *)ab(cs.size() * 4); sin_ = (float *)ab(sn.size() * 4);
    CK(cudaMemcpy(cos_, cs.data(), cs.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(sin_, sn.data(), sn.size() * 4, cudaMemcpyHostToDevice));
    build_graphs();
  }
  ~LlamaCuda() override {
    for (void *p : allocs_) cudaFree(p);
    cudaFree(kc_); cudaFree(vc_); cudaFree(arena_);
        cudaFreeHost(hctl_); cudaFreeHost(hlog_); cudaFreeHost(hxn_); cudaFreeHost(hout_);
    if (ga_) cudaGraphExecDestroy(ga_); if (gb_) cudaGraphExecDestroy(gb_); if (gc_) cudaGraphExecDestroy(gc_); cudaStreamDestroy(st_);
  }
  int vocab() const override { return V_; }
  int hidden() const override { return d_; }
  int max_ctx() const override { return ctx_; }
  int pos() const override { return pos_; }
  std::string arch() const override { return "llama:" + cfg_["model_type"].as_str() + "@cuda"; }
  void reset() override { pos_ = 0; }
  std::vector<uint8_t> checkpoint() const override { std::vector<uint8_t> c(4); memcpy(c.data(), &pos_, 4); return c; }
  void rollback(const std::vector<uint8_t> &c) override { memcpy(&pos_, c.data(), 4); }
  bool rewind(int pos) override { if (pos < 0 || pos > pos_) return false; pos_ = pos; return true; }
  std::vector<uint8_t> snapshot() const override {   // same layout as the CPU model: position + used prefix of K and V per layer
    std::vector<uint8_t> s(4 + 2ull * L_ * pos_ * kvd_ * 2); memcpy(s.data(), &pos_, 4); uint8_t *p = s.data() + 4; const size_t n = (size_t)pos_ * kvd_ * 2;
    for (int l = 0; l < L_; l++) for (const __half *buf : {kc_, vc_}) { CK(cudaMemcpy(p, buf + (size_t)l * ctx_ * kvd_, n, cudaMemcpyDeviceToHost)); p += n; }
    return s;
  }
  void restore(const std::vector<uint8_t> &s) override {
    memcpy(&pos_, s.data(), 4); const uint8_t *p = s.data() + 4; const size_t n = (size_t)pos_ * kvd_ * 2;
    for (int l = 0; l < L_; l++) for (__half *buf : {kc_, vc_}) { CK(cudaMemcpy(buf + (size_t)l * ctx_ * kvd_, p, n, cudaMemcpyHostToDevice)); p += n; }
  }

  // n greedy tokens without any host round trip per token: the device argmax feeds the next step
  int greedy_run(int tok, int n, int *out) override {
    n = std::min(n, ctx_ - pos_); if (n <= 0) return 0;
    if (tok < 0 || tok >= V_) throw std::runtime_error("token id out of range");
    Ctl &c = hctl_[slot_ % RING]; c.tok = tok; c.pos = pos_; slot_++;
    const int p0 = pos_;
    CK(cudaMemcpyAsync(ctl_, &c, sizeof(Ctl), cudaMemcpyHostToDevice, st_));
    for (int i = 0; i < n; i++) { CK(cudaGraphLaunch(ga_, st_)); CK(cudaGraphLaunch(gc_, st_)); }
    CK(cudaMemcpyAsync(hout_, out_ + p0, (size_t)n * 4, cudaMemcpyDeviceToHost, st_)); CK(cudaStreamSynchronize(st_));
    memcpy(out, hout_, (size_t)n * 4); pos_ += n; pending_ = 0; return n;
  }

  const float *step(int tok, int want) override {
    if (pos_ >= ctx_) throw std::runtime_error("context window full (" + std::to_string(ctx_) + " tokens)");
    if (tok < 0 || tok >= V_) throw std::runtime_error("token id out of range");
    Ctl &c = hctl_[slot_ % RING]; c.tok = tok; c.pos = pos_; slot_++;
    CK(cudaMemcpyAsync(ctl_, &c, sizeof(Ctl), cudaMemcpyHostToDevice, st_));
    CK(cudaGraphLaunch(ga_, st_)); pos_++;
    if (!want) { if (++pending_ >= RING / 2) { CK(cudaStreamSynchronize(st_)); pending_ = 0; } return nullptr; }
    if (dbg_ == 9 && pos_ == 70) { CK(cudaStreamSynchronize(st_)); long long h[16]; CK(cudaMemcpyFromSymbol(h, g_clk, sizeof h)); fprintf(stderr, "attn clk: loads+rope %lld scores %lld softmax %lld pv+write %lld\n", h[1]-h[0], h[2]-h[1], h[3]-h[2], h[5]-h[3]); }
    CK(cudaGraphLaunch(gb_, st_));
    if (want == 2) { CK(cudaMemcpyAsync(hxn_, xn_, (size_t)d_ * 4, cudaMemcpyDeviceToHost, st_)); CK(cudaStreamSynchronize(st_)); pending_ = 0; return hxn_; }
    CK(cudaMemcpyAsync(hlog_, logits_, (size_t)V_ * 4, cudaMemcpyDeviceToHost, st_)); CK(cudaStreamSynchronize(st_)); pending_ = 0;
    return hlog_;
  }

 private:
  static constexpr int RING = 256;
  void *dalloc(size_t n) { void *p; CK(cudaMalloc(&p, n)); allocs_.push_back(p); return p; }
  void *dcopy(const void *src, size_t n) { void *p = dalloc(n); CK(cudaMemcpy(p, src, n, cudaMemcpyHostToDevice)); return p; }
  const float *upf(const Tensor &t) {
    if (t.kind != Kind::F32) throw std::runtime_error("cuda: expected an f32 tensor: " + t.name);
    return (const float *)dcopy(t.data, 4 * (size_t)t.numel());
  }
  DMat up(const Tensor &t, bool allow_float) {
    DMat m{}; m.rows = (int)t.rows(); m.cols = (int)t.cols(); const size_t R = m.rows, C = m.cols;
    switch (t.kind) {
      case Kind::Q8: m.kind = K_Q8; m.q = dcopy(t.data, R * C); m.s = dcopy(t.scale, R * 4); if (C % 16) throw std::runtime_error("cuda: q8 cols % 16"); break;
      case Kind::Q4: m.kind = K_Q4; m.q = dcopy(t.data, R * C / 2); m.s = dcopy(t.scale16, R * C / 32 * 2); break;
      case Kind::Q1: if (C % 128) throw std::runtime_error("cuda: q1 cols % 128"); m.kind = K_Q1; m.q = dcopy(t.data, R * C / 8); m.s = dcopy(t.scale16, R * C / 128 * 2); break;
      case Kind::F16: if (!allow_float) throw std::runtime_error("cuda: f16 matrix"); m.kind = K_F16; m.q = dcopy(t.data, R * C * 2); break;
      case Kind::F32: if (!allow_float) throw std::runtime_error("cuda: f32 matrix"); m.kind = K_F32; m.q = dcopy(t.data, R * C * 4); break;
      default: throw std::runtime_error("cuda: unsupported tensor kind for " + t.name);
    }
    if (m.kind <= K_Q1 && C % 32) throw std::runtime_error("cuda: cols % 32");
    return m;
  }
  size_t smem_for(int cols) const { return 2 * (size_t)cols + (size_t)cols / 32 * 8; }

  void mv(cudaStream_t s, const DMat *m, const float *const *bias, float *const *out, int nm, int ai, int np, float *resid, const __half *pfk = nullptr, const __half *pfv = nullptr) {
    MvArgs a{}; if (pfk && !getenv("QPACK_CUDA_NOPF")) { a.ctl = ctl_; a.pfk = (const char *)pfk; a.pfv = (const char *)pfv; a.kvd = kvd_; } a.nm = nm; a.cols = m[0].cols; a.np = np; a.xq = aq_[ai]; a.xs = as_[ai]; a.xsum = asum_[ai]; a.resid = resid; int tot = 0;
    for (int i = 0; i < nm; i++) { a.m[i] = m[i]; a.bias[i] = bias ? bias[i] : nullptr; a.out[i] = out ? out[i] : nullptr; a.rowstart[i] = tot; tot += m[i].rows; }
    a.rowstart[nm] = tot;
    switch (wpb_) {
      case 2: k_mv<2><<<(tot + 1) / 2, 64, smem_for(a.cols), s>>>(a); break;
      case 4: k_mv<4><<<(tot + 3) / 4, 128, smem_for(a.cols), s>>>(a); break;
      case 16: k_mv<16><<<(tot + 15) / 16, 512, smem_for(a.cols), s>>>(a); break;
      default: k_mv<8><<<(tot + 7) / 8, 256, smem_for(a.cols), s>>>(a);
    }
  }
  void norm(cudaStream_t s, const float *w, int ai, float *yf, int np) { k_norm_actq<<<1, 1024, 0, s>>>(x_, w, d_, eps_, aq_[ai], as_[ai], asum_[ai], yf, np); }

  int P(int bit) const { return (p1mask_ >> bit) & 1 ? 1 : 2; }   // activation planes (1 or 2) of the vector produced at stage `bit`
  void build_graphs() {
    const float scl = 1.f / std::sqrt((float)hd_); const int rep = nh_ / nkv_;
    cudaStream_t s = st_;
    // opt-in to the dynamic shared memory sizes used (all well under 48 KB)
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    k_embed<<<1, 256, 0, s>>>(embed_, ctl_, x_);
    for (int l = 0; l < L_; l++) {
      const Layer &ly = layers_[l]; __half *kl = kc_ + (size_t)l * ctx_ * kvd_, *vl = vc_ + (size_t)l * ctx_ * kvd_;
      norm(s, ly.an, 0, nullptr, P(0));
      { DMat m[3] = {ly.wq, ly.wk, ly.wv}; const float *b[3] = {ly.bq, ly.bk, ly.bv}; float *o[3] = {q_, k_, v_}; mv(s, m, b, o, 3, 0, P(0), nullptr, kl, vl); }
      for (int rpt = 0; rpt < (dbg_ == 8 ? 2 : 1); rpt++)
      if (hd_ == 64) k_attn<64><<<dim3(nkv_, nch_), ANT, 0, s>>>(q_, k_, v_, cos_, sin_, kl, vl, part_, ctl_, rep, nkv_, nch_, scl, dbg_);
      else k_attn<128><<<dim3(nkv_, nch_), ANT, 0, s>>>(q_, k_, v_, cos_, sin_, kl, vl, part_, ctl_, rep, nkv_, nch_, scl, dbg_);
      if (hd_ == 64) k_merge<64><<<nh_, 64, 0, s>>>(part_, ctl_, nch_, aq_[0], as_[0], asum_[0], P(2));
      else k_merge<128><<<nh_, 128, 0, s>>>(part_, ctl_, nch_, aq_[0], as_[0], asum_[0], P(2));
      { DMat m[1] = {ly.wo}; mv(s, m, nullptr, nullptr, 1, 0, P(2), x_); }
      norm(s, ly.fn, 0, nullptr, P(1));
      switch (gu_nw_) {
        case 8: k_gateup<8><<<ff_ / 32, 8 * 32, smem_for(d_), s>>>(ly.wg, ly.wu, aq_[0], as_[0], asum_[0], aq_[1], as_[1], asum_[1], P(1), P(3)); break;
        case 16: k_gateup<16><<<ff_ / 32, 16 * 32, smem_for(d_), s>>>(ly.wg, ly.wu, aq_[0], as_[0], asum_[0], aq_[1], as_[1], asum_[1], P(1), P(3)); break;
        default: k_gateup<32><<<ff_ / 32, 32 * 32, smem_for(d_), s>>>(ly.wg, ly.wu, aq_[0], as_[0], asum_[0], aq_[1], as_[1], asum_[1], P(1), P(3));
      }
      { DMat m[1] = {ly.wd}; mv(s, m, nullptr, nullptr, 1, 1, P(3), x_); }
    }
    cudaGraph_t ga; CK(cudaStreamEndCapture(s, &ga)); CK(cudaGraphInstantiate(&ga_, ga, nullptr, nullptr, 0)); CK(cudaGraphDestroy(ga));
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    norm(s, fnorm_, 0, xn_, P(4));
    { DMat m[1] = {head_}; float *o[1] = {logits_}; mv(s, m, nullptr, o, 1, 0, P(4), nullptr); }
    cudaGraph_t gb; CK(cudaStreamEndCapture(s, &gb)); CK(cudaGraphInstantiate(&gb_, gb, nullptr, nullptr, 0)); CK(cudaGraphDestroy(gb));
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));   // greedy tail: final norm, head, device argmax (advances ctl)
    norm(s, fnorm_, 0, nullptr, P(4));
    { DMat m[1] = {head_}; float *o[1] = {logits_}; mv(s, m, nullptr, o, 1, 0, P(4), nullptr); }
    k_argmax1<<<AM_BLOCKS, 1024, 0, s>>>(logits_, V_, amv_, ami_); k_argmax2<<<1, 64, 0, s>>>(amv_, ami_, ctl_, out_);
    cudaGraph_t gc; CK(cudaStreamEndCapture(s, &gc)); CK(cudaGraphInstantiate(&gc_, gc, nullptr, nullptr, 0)); CK(cudaGraphDestroy(gc));
  }

  const int wpb_ = getenv("QPACK_CUDA_WPB") ? atoi(getenv("QPACK_CUDA_WPB")) : 8;
  const int p1mask_ = getenv("QPACK_CUDA_P1") ? atoi(getenv("QPACK_CUDA_P1")) : 0;
  const int dbg_ = getenv("QPACK_CUDA_DBG") ? atoi(getenv("QPACK_CUDA_DBG")) : 0;
  const int gu_nw_ = getenv("QPACK_CUDA_GU") ? atoi(getenv("QPACK_CUDA_GU")) : 32;
  int d_, L_, nh_, nkv_, hd_, ff_, V_, ctx_, kvd_, qd_, nch_, pos_ = 0, slot_ = 0, pending_ = 0; float eps_;
  cudaStream_t st_ = nullptr; cudaGraphExec_t ga_ = nullptr, gb_ = nullptr, gc_ = nullptr;
  DMat embed_{}, head_{}; const float *fnorm_ = nullptr; std::vector<Layer> layers_; std::vector<void *> allocs_;
  __half *kc_ = nullptr, *vc_ = nullptr; void *arena_ = nullptr;
  float *amv_; int *ami_, *out_;
  float *x_, *q_, *k_, *v_, *xn_, *logits_, *part_, *cos_, *sin_, *as_[2]; int8_t *aq_[2]; int *asum_[2];
  Ctl *ctl_, *hctl_; float *hlog_, *hxn_; int *hout_;
};

}  // namespace

std::unique_ptr<Model> make_llama_cuda(std::shared_ptr<Container> c, int max_ctx) { return std::unique_ptr<Model>(new LlamaCuda(c, max_ctx)); }

}  // namespace qp
