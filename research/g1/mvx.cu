// standalone q4 matvec experiments (shape 4864x896 / 896x4864 / 80000x896), distinct weight copies, CUDA graph
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)
constexpr int PLANE = 8192;
__device__ __forceinline__ int dp4a_us(unsigned a, int b, int c) { int d; asm("dp4a.u32.s32 %0,%1,%2,%3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }
__device__ __forceinline__ int dp4a_uu(unsigned a, unsigned b, int c) { int d; asm("dp4a.u32.u32 %0,%1,%2,%3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }
__device__ __forceinline__ float h2f(unsigned short h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ float warp_sum(float v) { for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o); return v; }

struct A { const uint8_t *q; const unsigned short *s; int rows, cols; const int8_t *xq; const float *xs; const int *xsum; float *y; };

// V0: current structure. MODE 0 = full, 1 = np1, 2 = loads only (xor)
template <int WPB, int MODE> __global__ void __launch_bounds__(WPB * 32) k0(A a) {
  extern __shared__ int4 smem[];
  const int cols = a.cols, G = cols >> 5;
  int4 *sx = smem, *sl = smem + (cols >> 4); float *sxs = (float *)(smem + 2 * (cols >> 4)); int *sxsum = (int *)(sxs + G);
  const int r = blockIdx.x * WPB + (threadIdx.x >> 5), l = threadIdx.x & 31;
  uint4 pv = make_uint4(0, 0, 0, 0); unsigned short ps = 0;
  const uint4 *w = (const uint4 *)(a.q + (size_t)r * (cols >> 1)); const unsigned short *s = a.s + (size_t)r * G;
  if (l < G) { pv = __ldcs(w + l); ps = __ldg(s + l); }
  for (int i = threadIdx.x; i < (cols >> 4); i += WPB * 32) { sx[i] = ((const int4 *)a.xq)[i]; sl[i] = ((const int4 *)(a.xq + PLANE))[i]; }
  for (int i = threadIdx.x; i < G; i += WPB * 32) { sxs[i] = a.xs[i]; sxsum[i] = a.xsum[i]; }
  __syncthreads();
  float acc = 0;
  for (int gb = 0; gb < G; gb += 32) {
    const int g = gb + l; uint4 v = pv; unsigned short sh = ps;
    if (gb) { if (g < G) { v = __ldcs(w + g); sh = __ldg(s + g); } }
    if (g >= G) break;
    if (MODE == 2) { acc += (float)(v.x ^ v.y ^ v.z ^ v.w) * h2f(sh); continue; }
    const int4 xl = sx[2 * g], xh = sx[2 * g + 1]; const unsigned M = 0x0F0F0F0Fu;
    const unsigned w0 = v.x & M, w1 = v.y & M, w2 = v.z & M, w3 = v.w & M, w4 = (v.x >> 4) & M, w5 = (v.y >> 4) & M, w6 = (v.z >> 4) & M, w7 = (v.w >> 4) & M;
    int d = dp4a_us(w0, xl.x, 0); d = dp4a_us(w1, xl.y, d); d = dp4a_us(w2, xl.z, d); d = dp4a_us(w3, xl.w, d);
    d = dp4a_us(w4, xh.x, d); d = dp4a_us(w5, xh.y, d); d = dp4a_us(w6, xh.z, d); d = dp4a_us(w7, xh.w, d);
    if (MODE != 1) {
      const int4 ll = sl[2 * g], lh = sl[2 * g + 1];
      int e = dp4a_uu(w0, (unsigned)ll.x, 0); e = dp4a_uu(w1, (unsigned)ll.y, e); e = dp4a_uu(w2, (unsigned)ll.z, e); e = dp4a_uu(w3, (unsigned)ll.w, e);
      e = dp4a_uu(w4, (unsigned)lh.x, e); e = dp4a_uu(w5, (unsigned)lh.y, e); e = dp4a_uu(w6, (unsigned)lh.z, e); e = dp4a_uu(w7, (unsigned)lh.w, e);
      d = (d << 8) + e;
    }
    d -= 8 * sxsum[g];
    acc += (float)d * (sxs[g] * h2f(sh));
  }
  acc = warp_sum(acc);
  if (l == 0) a.y[r] = acc;
}

// V1: R rows per warp, all weight loads issued up front (R*uint4 per lane), rows contiguous per warp (r = (blk*WPB+w)*R + i)
template <int WPB, int R> __global__ void __launch_bounds__(WPB * 32) k1(A a) {
  extern __shared__ int4 smem[];
  const int cols = a.cols, G = cols >> 5;
  int4 *sx = smem, *sl = smem + (cols >> 4); float *sxs = (float *)(smem + 2 * (cols >> 4)); int *sxsum = (int *)(sxs + G);
  const int r0 = (blockIdx.x * WPB + (threadIdx.x >> 5)) * R, l = threadIdx.x & 31;
  uint4 pv[R]; unsigned short ps[R];
#pragma unroll
  for (int i = 0; i < R; i++) { pv[i] = make_uint4(0, 0, 0, 0); ps[i] = 0;
    if (l < G) { pv[i] = __ldcs((const uint4 *)(a.q + (size_t)(r0 + i) * (cols >> 1)) + l); ps[i] = __ldg(a.s + (size_t)(r0 + i) * G + l); } }
  for (int i = threadIdx.x; i < (cols >> 4); i += WPB * 32) { sx[i] = ((const int4 *)a.xq)[i]; sl[i] = ((const int4 *)(a.xq + PLANE))[i]; }
  for (int i = threadIdx.x; i < G; i += WPB * 32) { sxs[i] = a.xs[i]; sxsum[i] = a.xsum[i]; }
  __syncthreads();
  if (l >= G) { for (int i = 0; i < R; i++) { float z = warp_sum(0.f); if (l == 0) a.y[r0 + i] = z; } return; }
  const int4 xl = sx[2 * l], xh = sx[2 * l + 1], ll = sl[2 * l], lh = sl[2 * l + 1]; const float xsc = sxs[l]; const int xsm = 8 * sxsum[l];
#pragma unroll
  for (int i = 0; i < R; i++) {
    const uint4 v = pv[i]; const unsigned M = 0x0F0F0F0Fu;
    const unsigned w0 = v.x & M, w1 = v.y & M, w2 = v.z & M, w3 = v.w & M, w4 = (v.x >> 4) & M, w5 = (v.y >> 4) & M, w6 = (v.z >> 4) & M, w7 = (v.w >> 4) & M;
    int d = dp4a_us(w0, xl.x, 0); d = dp4a_us(w1, xl.y, d); d = dp4a_us(w2, xl.z, d); d = dp4a_us(w3, xl.w, d);
    d = dp4a_us(w4, xh.x, d); d = dp4a_us(w5, xh.y, d); d = dp4a_us(w6, xh.z, d); d = dp4a_us(w7, xh.w, d);
    int e = dp4a_uu(w0, (unsigned)ll.x, 0); e = dp4a_uu(w1, (unsigned)ll.y, e); e = dp4a_uu(w2, (unsigned)ll.z, e); e = dp4a_uu(w3, (unsigned)ll.w, e);
    e = dp4a_uu(w4, (unsigned)lh.x, e); e = dp4a_uu(w5, (unsigned)lh.y, e); e = dp4a_uu(w6, (unsigned)lh.z, e); e = dp4a_uu(w7, (unsigned)lh.w, e);
    d = (d << 8) + e; d -= xsm;
    const float acc = warp_sum((float)d * (xsc * h2f(ps[i])));
    if (l == 0) a.y[r0 + i] = acc;
  }
}

int main(int argc, char **argv) {
  struct Sh { const char *n; int r, c; } shapes[] = {{"gate/up 4864x896", 4864, 896}, {"down 896x4864", 896, 4864}, {"head 80000x896", 80000, 896}};
  cudaStream_t st; CK(cudaStreamCreate(&st)); cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  int8_t *xq; float *xs; int *xsum, *dummy; float *y; CK(cudaMalloc(&xq, 2 * PLANE)); CK(cudaMalloc(&xs, 4 * PLANE / 32)); CK(cudaMalloc(&xsum, 4 * PLANE / 32)); CK(cudaMalloc(&y, 160000 * 4));
  { std::vector<int8_t> h(2 * PLANE); for (auto &v : h) v = rand(); CK(cudaMemcpy(xq, h.data(), h.size(), cudaMemcpyHostToDevice)); std::vector<float> f(PLANE / 32, 0.001f); CK(cudaMemcpy(xs, f.data(), f.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemset(xsum, 0, 4 * PLANE / 32)); }
  for (auto &sh : shapes) {
    const size_t R = sh.r, C = sh.c, qb = R * C / 2, sb = R * C / 32 * 2, bytes = qb + sb;
    const int copies = (int)std::max<size_t>(1, std::min<size_t>(64, (96u << 20) / bytes + 1));
    std::vector<uint8_t *> qs; std::vector<unsigned short *> ss;
    for (int i = 0; i < copies; i++) { void *q, *s; CK(cudaMalloc(&q, qb)); CK(cudaMalloc(&s, sb)); CK(cudaMemset(q, 0x5a + i, qb)); CK(cudaMemset(s, 0x3c, sb)); qs.push_back((uint8_t *)q); ss.push_back((unsigned short *)s); }
    const size_t smem = 2 * C + C / 32 * 8;
    auto run = [&](const char *name, auto launch) {
      cudaGraph_t g; cudaGraphExec_t ge; CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
      for (int i = 0; i < copies; i++) { A a{qs[i], ss[i], sh.r, sh.c, xq, xs, xsum, y}; launch(a); }
      CK(cudaStreamEndCapture(st, &g)); CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
      CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st)); float best = 1e9;
      for (int rep = 0; rep < 9; rep++) { CK(cudaEventRecord(e0, st)); for (int k = 0; k < 5; k++) CK(cudaGraphLaunch(ge, st)); CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); best = std::min(best, ms / (5 * copies)); }
      const double us = best * 1e3; printf("%-18s %-14s %8.1f us %7.1f GB/s\n", sh.n, name, us, bytes / (us * 1e-6) / 1e9);
      CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
    };
    run("V0 full w8", [&](A a) { k0<8, 0><<<(sh.r + 7) / 8, 256, smem, st>>>(a); });
    run("V0 np1 w8", [&](A a) { k0<8, 1><<<(sh.r + 7) / 8, 256, smem, st>>>(a); });
    run("V0 loadonly w8", [&](A a) { k0<8, 2><<<(sh.r + 7) / 8, 256, smem, st>>>(a); });
    run("V0 full w4", [&](A a) { k0<4, 0><<<(sh.r + 3) / 4, 128, smem, st>>>(a); });
    run("V0 full w16", [&](A a) { k0<16, 0><<<(sh.r + 15) / 16, 512, smem, st>>>(a); });
    if (sh.c == 896 && sh.r % 32 == 0) {
      run("V1 R2 w4", [&](A a) { k1<4, 2><<<sh.r / 8, 128, smem, st>>>(a); });
      run("V1 R4 w4", [&](A a) { k1<4, 4><<<sh.r / 16, 128, smem, st>>>(a); });
      run("V1 R4 w8", [&](A a) { k1<8, 4><<<sh.r / 32, 256, smem, st>>>(a); });
      run("V1 R8 w4", [&](A a) { k1<4, 8><<<sh.r / 32, 128, smem, st>>>(a); });
    }
    for (auto p : qs) cudaFree(p); for (auto p : ss) cudaFree(p);
  }
}
