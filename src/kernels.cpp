// CPU kernels, architecture-independent part: activation quantisation, work splitting over the thread pool and dispatch
// to the AVX2 (kernels_x86.cpp) or NEON (kernels_neon.cpp) row kernels. Weights: f32 / f16 / q8 / q4.
// q8 / q4: the activation vector is quantised once per call to int16 with a scale per 128-column block.
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <vector>

#include "kernels_impl.h"
#include "qpack/qpack.h"

namespace qp {
using namespace impl;

// int16 quantisation of B activation rows (one scale per QB block), rows of ldq entries
static void quant_rows(const float *X, int B, int ldx, int cols, int ldq, int nb, int16_t *xq, float *bs) {
  auto quant = [&](int i0, int i1) {
    for (int i = i0; i < i1; i++) {
      const float *x = X + (int64_t)i * ldx; int16_t *q = xq + (int64_t)i * ldq; float *s = bs + (int64_t)i * nb;
      for (int k = 0; k < nb; k++) {
        const int c0 = k * QB, c1 = std::min(cols, c0 + QB); float mx = 0;
        for (int c = c0; c < c1; c++) mx = std::max(mx, std::fabs(x[c]));
        const float isx = mx > 0 ? kActMax / mx : 0.f; s[k] = mx / kActMax;
        int16_t tmp[QB]; for (int c = c0; c < c1; c++) tmp[c - c0] = (int16_t)lrintf(x[c] * isx);
        store_act_block(tmp, c1 - c0, q, c0, cols);
      }
    }
  };
  if ((int64_t)B * cols >= (1 << 15)) pool().parallel_for(B, quant, 1); else quant(0, B);
}

std::function<void(const Tensor &, const float *, int, int, int)> g_act_hook;
// matrix-vector products below this many multiply-adds run on the calling thread (a pool round trip costs ~2 us)
thread_local bool g_exact_batches = false;
static const int64_t kParMin = [] { const char *e = getenv("QPACK_PAR_MIN"); return e ? atoll(e) : (int64_t)1 << 16; }();

// rows [a, b) of W x; xp / bp: the int16 activations and block scales (q8 / q4 only)
static void rows_any(const Tensor &W, const float *x, const int16_t *xp, const float *bp, float *y, int a, int b, int cols) {
  switch (W.kind) {
    case Kind::F32: rows_f32(W.f32(), x, y, a, b, cols); break;
    case Kind::F16: rows_f16((const uint16_t *)W.data, x, y, a, b, cols); break;
    case Kind::Q8: rows_q8((const int8_t *)W.data, W.scale, xp, bp, y, a, b, cols); break;
    case Kind::Q4: rows_q4((const uint8_t *)W.data, W.scale16, xp, bp, y, a, b, cols); break;
    default: throw std::runtime_error("matvec: unsupported tensor kind for " + W.name);
  }
}

void matvec_n(const Tensor *const *W, int nw, const float *x, float *const *y, int cols) {
  if (nw == 1) { matvec(*W[0], x, y[0], (int)W[0]->rows(), cols); return; }
  int total = 0, start[8]; bool quant = false;
  for (int i = 0; i < nw; i++) {
    if (g_act_hook) g_act_hook(*W[i], x, 1, cols, cols);
    start[i] = total; total += (int)W[i]->rows(); quant |= W[i]->kind == Kind::Q8 || W[i]->kind == Kind::Q4;
  }
  start[nw] = total;
  const int nb = (cols + QB - 1) / QB;
  thread_local std::vector<int16_t> xq; thread_local std::vector<float> bs;
  if ((int)xq.size() < cols + 32) xq.resize(cols + 32);
  if ((int)bs.size() < nb) bs.resize(nb);
  if (quant) quant_rows(x, 1, cols, cols, cols, nb, xq.data(), bs.data());
  const int16_t *xp = xq.data(); const float *bp = bs.data();
  auto f = [&](int a, int b) {   // global row range -> pieces of each matrix
    for (int i = 0; i < nw; i++) { const int lo = std::max(a, start[i]), hi = std::min(b, start[i + 1]); if (lo < hi) rows_any(*W[i], x, xp, bp, y[i], lo - start[i], hi - start[i], cols); }
  };
  if ((int64_t)total * cols >= kParMin) pool().parallel_for(total, f, 8); else f(0, total);
}

void matvec(const Tensor &W, const float *x, float *y, int rows, int cols) {
  if (g_act_hook) g_act_hook(W, x, 1, cols, cols);
  const int64_t work = (int64_t)rows * cols;
  const bool par = work >= kParMin;
  auto run = [&](const std::function<void(int, int)> &f) { if (par) pool().parallel_for(rows, f, 8); else f(0, rows); };
  switch (W.kind) {
    case Kind::F32: run([&](int a, int b) { rows_f32(W.f32(), x, y, a, b, cols); }); break;
    case Kind::F16: run([&](int a, int b) { rows_f16((const uint16_t *)W.data, x, y, a, b, cols); }); break;
    case Kind::Q8: case Kind::Q4: {
      const int nb = (cols + QB - 1) / QB;
      thread_local std::vector<int16_t> xq; thread_local std::vector<float> bs;
      if ((int)xq.size() < cols + 32) xq.resize(cols + 32);
      if ((int)bs.size() < nb) bs.resize(nb);
      quant_rows(x, 1, cols, cols, cols, nb, xq.data(), bs.data());
      const int16_t *xp = xq.data(); const float *bp = bs.data();
      if (W.kind == Kind::Q8) run([&](int a, int b) { rows_q8((const int8_t *)W.data, W.scale, xp, bp, y, a, b, cols); });
      else run([&](int a, int b) { rows_q4((const uint8_t *)W.data, W.scale16, xp, bp, y, a, b, cols); });
      break;
    }
    default: throw std::runtime_error("matvec: unsupported tensor kind for " + W.name);
  }
}
void matmul(const Tensor &W, const float *X, int B, int ldx, float *Y, int ldy, int rows, int cols) {
  if (B == 1) { matvec(W, X, Y, rows, cols); return; }
  if (g_act_hook) g_act_hook(W, X, B, ldx, cols);
  if (B <= 16 && g_exact_batches) {   // small batches (speculative verification): per vector exactly the arithmetic of matvec, each weight row
                   // still read once for all B vectors, so the results are bit-identical to B matvec() calls
    const int nb = (cols + QB - 1) / QB, ldq = (cols + 31) & ~31;
    thread_local std::vector<int16_t> xq; thread_local std::vector<float> bs;
    if (xq.size() < (size_t)B * ldq + 32) xq.resize((size_t)B * ldq + 32);
    if (bs.size() < (size_t)B * nb) bs.resize((size_t)B * nb);
    const bool qz = W.kind == Kind::Q8 || W.kind == Kind::Q4;
    if (qz) for (int b = 0; b < B; b++) quant_rows(X + (size_t)b * ldx, 1, cols, cols, cols, nb, xq.data() + (size_t)b * ldq, bs.data() + (size_t)b * nb);
    const int16_t *xp = xq.data(); const float *bp = bs.data();
    auto f = [&](int a, int b2) {
      for (int r = a; r < b2; r++)
        for (int b = 0; b < B; b += 4) {
          const int nt = std::min(4, B - b);
          if (W.kind == Kind::Q4) row_q4_multi((const uint8_t *)W.data, W.scale16, xp + (size_t)b * ldq, ldq, bp + (size_t)b * nb, nb, Y + (size_t)b * ldy, ldy, r, nt, cols);
          else if (W.kind == Kind::Q8) row_q8_multi((const int8_t *)W.data, W.scale, xp + (size_t)b * ldq, ldq, bp + (size_t)b * nb, nb, Y + (size_t)b * ldy, ldy, r, nt, cols);
          else for (int j = b; j < b + nt; j++) rows_any(W, X + (size_t)j * ldx, xp + (size_t)j * ldq, bp + (size_t)j * nb, Y + (size_t)j * ldy, r, r + 1, cols);
        }
    };
    if ((int64_t)rows * cols >= (1 << 16)) pool().parallel_for(rows, f, 8); else f(0, rows);
    return;
  }
  auto run = [&](const std::function<void(int, int)> &f) { if ((int64_t)rows * cols * B >= (1 << 16)) pool().parallel_for(rows, f, 4); else f(0, rows); };
  switch (W.kind) {
    case Kind::F32: run([&](int a, int b) { mm_f32(W.f32(), X, ldx, B, Y, ldy, a, b, cols); }); break;
    case Kind::F16: run([&](int a, int b) { mm_f16((const uint16_t *)W.data, X, ldx, B, Y, ldy, a, b, cols); }); break;
    case Kind::Q8: case Kind::Q4: {
      const int nb = (cols + QB - 1) / QB, ldq = (cols + 31) & ~31;
      thread_local std::vector<int16_t> xq; thread_local std::vector<float> bs;
      if (xq.size() < (size_t)B * ldq) xq.resize((size_t)B * ldq);
      if (bs.size() < (size_t)B * nb) bs.resize((size_t)B * nb);
      quant_rows(X, B, ldx, cols, ldq, nb, xq.data(), bs.data());
      const int16_t *xp = xq.data(); const float *bp = bs.data();
      if (W.kind == Kind::Q8) run([&](int a, int b) { mm_q8((const int8_t *)W.data, W.scale, xp, bp, ldq, nb, B, Y, ldy, a, b, cols); });
      else run([&](int a, int b) { mm_q4((const uint8_t *)W.data, W.scale16, xp, bp, ldq, nb, B, Y, ldy, a, b, cols); });
      break;
    }
    default: for (int i = 0; i < B; i++) matvec(W, X + (int64_t)i * ldx, Y + (int64_t)i * ldy, rows, cols);
  }
}

void row_f32(const Tensor &W, int64_t r, float *out) {
  const int64_t n = W.cols();
  switch (W.kind) {
    case Kind::F32: memcpy(out, W.f32() + r * n, 4 * n); break;
    case Kind::F16: { const uint16_t *h = (const uint16_t *)W.data + r * n; for (int64_t i = 0; i < n; i++) out[i] = f16_to_f32(h[i]); break; }
    case Kind::Q8: { const int8_t *q = (const int8_t *)W.data + r * n; const float s = W.scale[r]; for (int64_t i = 0; i < n; i++) out[i] = q[i] * s; break; }
    case Kind::Q4: {
      const uint8_t *q = (const uint8_t *)W.data + r * (n / 2); const uint16_t *s = W.scale16 + r * (n / 32);
      for (int64_t k = 0; k < n / 32; k++) {
        const float sc = f16_to_f32(s[k]);
        for (int j = 0; j < 16; j++) { const uint8_t b = q[16 * k + j]; out[32 * k + j] = ((int)(b & 15) - 8) * sc; out[32 * k + j + 16] = ((int)(b >> 4) - 8) * sc; }
      }
      break;
    }
    default: throw std::runtime_error("row_f32: unsupported tensor kind for " + W.name);
  }
}

void rmsnorm(const float *x, const float *w, int n, float eps, float *out, bool zc) {
  float ss = 0; for (int i = 0; i < n; i++) ss += x[i] * x[i];
  const float inv = 1.f / std::sqrt(ss / n + eps);
  if (zc) for (int i = 0; i < n; i++) out[i] = x[i] * inv * (1.f + w[i]);
  else for (int i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

}  // namespace qp
