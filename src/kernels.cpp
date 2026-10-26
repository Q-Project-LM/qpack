// CPU kernels (AVX2 + FMA + F16C, with scalar fallbacks): matrix-vector products for f32 / f16 / q8 weights.
// q8: the activation vector is quantised once per call to int16 with a scale per 128-column block, then
// _mm256_madd_epi16 multiplies the int8 weights (sign-extended to int16) by it 16 lanes at a time.
#include <immintrin.h>

#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <vector>

#include "qpack/qpack.h"

namespace qp {

static inline float hsum256(__m256 v) {
  __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  a = _mm_add_ps(a, _mm_movehl_ps(a, a)); a = _mm_add_ss(a, _mm_shuffle_ps(a, a, 1)); return _mm_cvtss_f32(a);
}
static inline int hsum256i(__m256i v) {
  __m128i a = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
  a = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0x4e)); a = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0xb1)); return _mm_cvtsi128_si32(a);
}

static void rows_f32(const float *W, const float *x, float *y, int r0, int r1, int cols) {
  for (int r = r0; r < r1; r++) {
    const float *w = W + (int64_t)r * cols; __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0; int c = 0;
    for (; c + 32 <= cols; c += 32) {
      a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + c), _mm256_loadu_ps(x + c), a0);
      a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + c + 8), _mm256_loadu_ps(x + c + 8), a1);
      a2 = _mm256_fmadd_ps(_mm256_loadu_ps(w + c + 16), _mm256_loadu_ps(x + c + 16), a2);
      a3 = _mm256_fmadd_ps(_mm256_loadu_ps(w + c + 24), _mm256_loadu_ps(x + c + 24), a3);
    }
    float s = hsum256(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    for (; c < cols; c++) s += w[c] * x[c];
    y[r] = s;
  }
}
static void rows_f16(const uint16_t *W, const float *x, float *y, int r0, int r1, int cols) {
  for (int r = r0; r < r1; r++) {
    const uint16_t *w = W + (int64_t)r * cols; __m256 a0 = _mm256_setzero_ps(), a1 = a0; int c = 0;
    for (; c + 16 <= cols; c += 16) {
      a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + c))), _mm256_loadu_ps(x + c), a0);
      a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + c + 8))), _mm256_loadu_ps(x + c + 8), a1);
    }
    float s = hsum256(_mm256_add_ps(a0, a1));
    for (; c < cols; c++) s += _cvtsh_ss(w[c]) * x[c];
    y[r] = s;
  }
}
// int16 activations with one scale per block of QB columns (outlier channels — Qwen has hidden values in the hundreds —
// only coarsen their own block), int8 weights: per block, an int32 dot product via _mm256_madd_epi16, then one float FMA
static constexpr int QB = 128;
static void rows_q8(const int8_t *Q, const float *scale, const int16_t *xq, const float *bs, float *y, int r0, int r1, int cols) {
  const int nb = cols / QB;
  for (int r = r0; r < r1; r++) {
    const int8_t *q = Q + (int64_t)r * cols; float acc = 0;
    for (int b = 0; b < nb; b++) {
      const int8_t *qb = q + b * QB; const int16_t *xb = xq + b * QB; __m256i a0 = _mm256_setzero_si256(), a1 = a0;
      for (int c = 0; c < QB; c += 32) {
        a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(qb + c))), _mm256_loadu_si256((const __m256i *)(xb + c))));
        a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(qb + c + 16))), _mm256_loadu_si256((const __m256i *)(xb + c + 16))));
      }
      acc += bs[b] * (float)hsum256i(_mm256_add_epi32(a0, a1));
    }
    float tail = 0; for (int c = nb * QB; c < cols; c++) tail += (float)q[c] * (float)xq[c];
    if (nb * QB < cols) acc += bs[nb] * tail;
    y[r] = acc * scale[r];
  }
}

void matvec(const Tensor &W, const float *x, float *y, int rows, int cols) {
  const int64_t work = (int64_t)rows * cols;
  const bool par = work >= (1 << 16);
  auto run = [&](const std::function<void(int, int)> &f) { if (par) pool().parallel_for(rows, f, 8); else f(0, rows); };
  switch (W.kind) {
    case Kind::F32: run([&](int a, int b) { rows_f32(W.f32(), x, y, a, b, cols); }); break;
    case Kind::F16: run([&](int a, int b) { rows_f16((const uint16_t *)W.data, x, y, a, b, cols); }); break;
    case Kind::Q8: {
      const int nb = (cols + QB - 1) / QB;
      thread_local std::vector<int16_t> xq; thread_local std::vector<float> bs;
      if ((int)xq.size() < cols + 32) xq.resize(cols + 32);
      if ((int)bs.size() < nb) bs.resize(nb);
      for (int b = 0; b < nb; b++) {
        const int c0 = b * QB, c1 = std::min(cols, c0 + QB); float mx = 0;
        for (int c = c0; c < c1; c++) mx = std::max(mx, std::fabs(x[c]));
        const float isx = mx > 0 ? 32767.f / mx : 0.f; bs[b] = mx / 32767.f;
        for (int c = c0; c < c1; c++) xq[c] = (int16_t)lrintf(x[c] * isx);
      }
      const int16_t *xp = xq.data(); const float *bp = bs.data();
      run([&](int a, int b) { rows_q8((const int8_t *)W.data, W.scale, xp, bp, y, a, b, cols); });
      break;
    }
    default: throw std::runtime_error("matvec: unsupported tensor kind for " + W.name);
  }
}

void row_f32(const Tensor &W, int64_t r, float *out) {
  const int64_t n = W.cols();
  switch (W.kind) {
    case Kind::F32: memcpy(out, W.f32() + r * n, 4 * n); break;
    case Kind::F16: { const uint16_t *h = (const uint16_t *)W.data + r * n; for (int64_t i = 0; i < n; i++) out[i] = _cvtsh_ss(h[i]); break; }
    case Kind::Q8: { const int8_t *q = (const int8_t *)W.data + r * n; const float s = W.scale[r]; for (int64_t i = 0; i < n; i++) out[i] = q[i] * s; break; }
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
