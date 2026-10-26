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

// q4: 32-column groups, 16 bytes each (byte j = column j in the low nibble, column j + 16 in the high one, stored as
// q + 8), one fp16 scale per group. Unpacked to int16 and multiplied like q8; the group scale times the activation block
// scale goes into one float FMA per group.
static inline void unpack_q4(const uint8_t *p, __m256i &lo, __m256i &hi) {
  const __m128i b = _mm_loadu_si128((const __m128i *)p), m = _mm_set1_epi8(0x0F), e = _mm_set1_epi8(8);
  lo = _mm256_cvtepi8_epi16(_mm_sub_epi8(_mm_and_si128(b, m), e));
  hi = _mm256_cvtepi8_epi16(_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(b, 4), m), e));
}
static void rows_q4(const uint8_t *Q, const uint16_t *S, const int16_t *xq, const float *bs, float *y, int r0, int r1, int cols) {
  const int G = cols / 32;
  for (int r = r0; r < r1; r++) {
    const uint8_t *q = Q + (int64_t)r * (cols / 2); const uint16_t *s = S + (int64_t)r * G;
    __m256 a0 = _mm256_setzero_ps(), a1 = a0; int k = 0;
    for (; k + 2 <= G; k += 2) {
      __m256i l0, h0, l1, h1; unpack_q4(q + 16 * k, l0, h0); unpack_q4(q + 16 * k + 16, l1, h1);
      const int16_t *x = xq + 32 * k;
      const __m256i d0 = _mm256_add_epi32(_mm256_madd_epi16(l0, _mm256_loadu_si256((const __m256i *)x)), _mm256_madd_epi16(h0, _mm256_loadu_si256((const __m256i *)(x + 16))));
      const __m256i d1 = _mm256_add_epi32(_mm256_madd_epi16(l1, _mm256_loadu_si256((const __m256i *)(x + 32))), _mm256_madd_epi16(h1, _mm256_loadu_si256((const __m256i *)(x + 48))));
      a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d0), _mm256_set1_ps(_cvtsh_ss(s[k]) * bs[k >> 2]), a0);
      a1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d1), _mm256_set1_ps(_cvtsh_ss(s[k + 1]) * bs[(k + 1) >> 2]), a1);
    }
    for (; k < G; k++) {
      __m256i l0, h0; unpack_q4(q + 16 * k, l0, h0); const int16_t *x = xq + 32 * k;
      const __m256i d0 = _mm256_add_epi32(_mm256_madd_epi16(l0, _mm256_loadu_si256((const __m256i *)x)), _mm256_madd_epi16(h0, _mm256_loadu_si256((const __m256i *)(x + 16))));
      a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d0), _mm256_set1_ps(_cvtsh_ss(s[k]) * bs[k >> 2]), a0);
    }
    y[r] = hsum256(_mm256_add_ps(a0, a1));
  }
}

// int16 quantisation of B activation rows (one scale per QB block), rows of ldq entries
static void quant_rows(const float *X, int B, int ldx, int cols, int ldq, int nb, int16_t *xq, float *bs) {
  auto quant = [&](int i0, int i1) {
    for (int i = i0; i < i1; i++) {
      const float *x = X + (int64_t)i * ldx; int16_t *q = xq + (int64_t)i * ldq; float *s = bs + (int64_t)i * nb;
      for (int k = 0; k < nb; k++) {
        const int c0 = k * QB, c1 = std::min(cols, c0 + QB); float mx = 0;
        for (int c = c0; c < c1; c++) mx = std::max(mx, std::fabs(x[c]));
        const float isx = mx > 0 ? 32767.f / mx : 0.f; s[k] = mx / 32767.f;
        for (int c = c0; c < c1; c++) q[c] = (int16_t)lrintf(x[c] * isx);
      }
    }
  };
  if ((int64_t)B * cols >= (1 << 15)) pool().parallel_for(B, quant, 1); else quant(0, B);
}

void matvec(const Tensor &W, const float *x, float *y, int rows, int cols) {
  const int64_t work = (int64_t)rows * cols;
  const bool par = work >= (1 << 16);
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

// ---- batched products (prompt): rows are split over threads, every thread runs its rows against all B vectors, four
// vectors per pass so each loaded weight vector feeds four accumulators
static void mm_q8(const int8_t *Q, const float *scale, const int16_t *XQ, const float *BS, int ldq, int nb, int B, float *Y, int ldy, int r0, int r1, int cols) {
  const int full = cols / QB;
  int r = r0;
  for (; r + 2 <= r1 && full * QB == cols; r += 2) {   // 2 rows x 4 vectors per pass: every load feeds two or four madds
    const int8_t *qa = Q + (int64_t)r * cols, *qb = qa + cols;
    int b = 0;
    for (; b + 4 <= B; b += 4) {
      const int16_t *x0 = XQ + (int64_t)b * ldq, *x1 = x0 + ldq, *x2 = x1 + ldq, *x3 = x2 + ldq;
      const float *s0 = BS + (int64_t)b * nb, *s1 = s0 + nb, *s2 = s1 + nb, *s3 = s2 + nb;
      __m256 f[8]; for (int j = 0; j < 8; j++) f[j] = _mm256_setzero_ps();
      for (int k = 0; k < full; k++) {
        __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0, c0 = a0, c1 = a0, c2 = a0, c3 = a0; const int cb = k * QB;
        for (int c = cb; c < cb + QB; c += 16) {
          const __m256i wa = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(qa + c))), wb = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(qb + c)));
          __m256i x = _mm256_loadu_si256((const __m256i *)(x0 + c)); a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(wa, x)); c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(wb, x));
          x = _mm256_loadu_si256((const __m256i *)(x1 + c)); a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(wa, x)); c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(wb, x));
          x = _mm256_loadu_si256((const __m256i *)(x2 + c)); a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(wa, x)); c2 = _mm256_add_epi32(c2, _mm256_madd_epi16(wb, x));
          x = _mm256_loadu_si256((const __m256i *)(x3 + c)); a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(wa, x)); c3 = _mm256_add_epi32(c3, _mm256_madd_epi16(wb, x));
        }
        const __m256 g0 = _mm256_set1_ps(s0[k]), g1 = _mm256_set1_ps(s1[k]), g2 = _mm256_set1_ps(s2[k]), g3 = _mm256_set1_ps(s3[k]);
        f[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(a0), g0, f[0]); f[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(a1), g1, f[1]);
        f[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(a2), g2, f[2]); f[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(a3), g3, f[3]);
        f[4] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c0), g0, f[4]); f[5] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c1), g1, f[5]);
        f[6] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c2), g2, f[6]); f[7] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c3), g3, f[7]);
      }
      for (int j = 0; j < 4; j++) { Y[(int64_t)(b + j) * ldy + r] = hsum256(f[j]) * scale[r]; Y[(int64_t)(b + j) * ldy + r + 1] = hsum256(f[4 + j]) * scale[r + 1]; }
    }
    for (; b < B; b++) rows_q8(Q, scale, XQ + (int64_t)b * ldq, BS + (int64_t)b * nb, Y + (int64_t)b * ldy, r, r + 2, cols);
  }
  for (; r < r1; r++) {
    const int8_t *q = Q + (int64_t)r * cols;
    int b = 0;
    for (; b + 4 <= B; b += 4) {
      const int16_t *x0 = XQ + (int64_t)b * ldq, *x1 = x0 + ldq, *x2 = x1 + ldq, *x3 = x2 + ldq;
      const float *s0 = BS + (int64_t)b * nb, *s1 = s0 + nb, *s2 = s1 + nb, *s3 = s2 + nb;
      __m256 f0 = _mm256_setzero_ps(), f1 = f0, f2 = f0, f3 = f0;
      for (int k = 0; k < full; k++) {
        __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0; const int c0 = k * QB;
        for (int c = c0; c < c0 + QB; c += 16) {
          const __m256i w = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(q + c)));
          a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(w, _mm256_loadu_si256((const __m256i *)(x0 + c))));
          a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(w, _mm256_loadu_si256((const __m256i *)(x1 + c))));
          a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(w, _mm256_loadu_si256((const __m256i *)(x2 + c))));
          a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(w, _mm256_loadu_si256((const __m256i *)(x3 + c))));
        }
        f0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(a0), _mm256_set1_ps(s0[k]), f0); f1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(a1), _mm256_set1_ps(s1[k]), f1);
        f2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(a2), _mm256_set1_ps(s2[k]), f2); f3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(a3), _mm256_set1_ps(s3[k]), f3);
      }
      float t[4] = {hsum256(f0), hsum256(f1), hsum256(f2), hsum256(f3)};
      if (full * QB < cols) {
        const int16_t *xs[4] = {x0, x1, x2, x3}; const float *ss[4] = {s0, s1, s2, s3};
        for (int j = 0; j < 4; j++) { float tail = 0; for (int c = full * QB; c < cols; c++) tail += (float)q[c] * (float)xs[j][c]; t[j] += ss[j][full] * tail; }
      }
      for (int j = 0; j < 4; j++) Y[(int64_t)(b + j) * ldy + r] = t[j] * scale[r];
    }
    for (; b < B; b++) rows_q8(Q, scale, XQ + (int64_t)b * ldq, BS + (int64_t)b * nb, Y + (int64_t)b * ldy, r, r + 1, cols);
  }
}
static void mm_q4(const uint8_t *Q, const uint16_t *S, const int16_t *XQ, const float *BS, int ldq, int nb, int B, float *Y, int ldy, int r0, int r1, int cols) {
  const int G = cols / 32;
  for (int r = r0; r < r1; r++) {
    const uint8_t *q = Q + (int64_t)r * (cols / 2); const uint16_t *s = S + (int64_t)r * G;
    int b = 0;
    for (; b + 4 <= B; b += 4) {
      const int16_t *x0 = XQ + (int64_t)b * ldq, *x1 = x0 + ldq, *x2 = x1 + ldq, *x3 = x2 + ldq;
      const float *s0 = BS + (int64_t)b * nb, *s1 = s0 + nb, *s2 = s1 + nb, *s3 = s2 + nb;
      __m256 f0 = _mm256_setzero_ps(), f1 = f0, f2 = f0, f3 = f0;
      for (int k = 0; k < G; k++) {
        __m256i lo, hi; unpack_q4(q + 16 * k, lo, hi); const int c = 32 * k, blk = k >> 2; const float sw = _cvtsh_ss(s[k]);
#define QP_Q4DOT(x) _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_madd_epi16(lo, _mm256_loadu_si256((const __m256i *)((x) + c))), _mm256_madd_epi16(hi, _mm256_loadu_si256((const __m256i *)((x) + c + 16)))))
        f0 = _mm256_fmadd_ps(QP_Q4DOT(x0), _mm256_set1_ps(sw * s0[blk]), f0); f1 = _mm256_fmadd_ps(QP_Q4DOT(x1), _mm256_set1_ps(sw * s1[blk]), f1);
        f2 = _mm256_fmadd_ps(QP_Q4DOT(x2), _mm256_set1_ps(sw * s2[blk]), f2); f3 = _mm256_fmadd_ps(QP_Q4DOT(x3), _mm256_set1_ps(sw * s3[blk]), f3);
#undef QP_Q4DOT
      }
      Y[(int64_t)b * ldy + r] = hsum256(f0); Y[(int64_t)(b + 1) * ldy + r] = hsum256(f1); Y[(int64_t)(b + 2) * ldy + r] = hsum256(f2); Y[(int64_t)(b + 3) * ldy + r] = hsum256(f3);
    }
    for (; b < B; b++) rows_q4(Q, S, XQ + (int64_t)b * ldq, BS + (int64_t)b * nb, Y + (int64_t)b * ldy, r, r + 1, cols);
  }
}

template <typename LoadW>
static void mm_float(LoadW loadw, const float *X, int ldx, int B, float *Y, int ldy, int r0, int r1, int cols, const std::function<float(int, int)> &tail) {
  for (int r = r0; r < r1; r++) {
    int b = 0;
    for (; b + 4 <= B; b += 4) {
      const float *x0 = X + (int64_t)b * ldx, *x1 = x0 + ldx, *x2 = x1 + ldx, *x3 = x2 + ldx;
      __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0; int c = 0;
      for (; c + 8 <= cols; c += 8) {
        const __m256 w = loadw(r, c);
        a0 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x0 + c), a0); a1 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x1 + c), a1);
        a2 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x2 + c), a2); a3 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x3 + c), a3);
      }
      float t[4] = {hsum256(a0), hsum256(a1), hsum256(a2), hsum256(a3)};
      for (int j = 0; j < 4; j++) { const float *xj = X + (int64_t)(b + j) * ldx; for (int cc = c; cc < cols; cc++) t[j] += tail(r, cc) * xj[cc]; Y[(int64_t)(b + j) * ldy + r] = t[j]; }
    }
    for (; b < B; b++) {
      const float *x = X + (int64_t)b * ldx; __m256 a = _mm256_setzero_ps(); int c = 0;
      for (; c + 8 <= cols; c += 8) a = _mm256_fmadd_ps(loadw(r, c), _mm256_loadu_ps(x + c), a);
      float t = hsum256(a); for (; c < cols; c++) t += tail(r, c) * x[c];
      Y[(int64_t)b * ldy + r] = t;
    }
  }
}

void matmul(const Tensor &W, const float *X, int B, int ldx, float *Y, int ldy, int rows, int cols) {
  if (B == 1) { matvec(W, X, Y, rows, cols); return; }
  auto run = [&](const std::function<void(int, int)> &f) { if ((int64_t)rows * cols * B >= (1 << 16)) pool().parallel_for(rows, f, 4); else f(0, rows); };
  switch (W.kind) {
    case Kind::F32: {
      const float *w = W.f32();
      run([&](int a, int b) { mm_float([&](int r, int c) { return _mm256_loadu_ps(w + (int64_t)r * cols + c); }, X, ldx, B, Y, ldy, a, b, cols,
                                       [&](int r, int c) { return w[(int64_t)r * cols + c]; }); });
      break;
    }
    case Kind::F16: {
      const uint16_t *w = (const uint16_t *)W.data;
      run([&](int a, int b) { mm_float([&](int r, int c) { return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + (int64_t)r * cols + c))); }, X, ldx, B, Y, ldy, a, b, cols,
                                       [&](int r, int c) { return _cvtsh_ss(w[(int64_t)r * cols + c]); }); });
      break;
    }
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
    case Kind::F16: { const uint16_t *h = (const uint16_t *)W.data + r * n; for (int64_t i = 0; i < n; i++) out[i] = _cvtsh_ss(h[i]); break; }
    case Kind::Q8: { const int8_t *q = (const int8_t *)W.data + r * n; const float s = W.scale[r]; for (int64_t i = 0; i < n; i++) out[i] = q[i] * s; break; }
    case Kind::Q4: {
      const uint8_t *q = (const uint8_t *)W.data + r * (n / 2); const uint16_t *s = W.scale16 + r * (n / 32);
      for (int64_t k = 0; k < n / 32; k++) {
        const float sc = _cvtsh_ss(s[k]);
        for (int j = 0; j < 16; j++) { const uint8_t b = q[16 * k + j]; out[32 * k + j] = ((int)(b & 15) - 8) * sc; out[32 * k + j + 16] = ((int)(b >> 4) - 8) * sc; }
      }
      break;
    }
    default: throw std::runtime_error("row_f32: unsupported tensor kind for " + W.name);
  }
}

float dot_f32(const float *a, const float *b, int n) {
  __m256 s0 = _mm256_setzero_ps(), s1 = s0; int i = 0;
  for (; i + 16 <= n; i += 16) { s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0); s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1); }
  for (; i + 8 <= n; i += 8) s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
  float s = hsum256(_mm256_add_ps(s0, s1)); for (; i < n; i++) s += a[i] * b[i];
  return s;
}
void axpy_f32(float a, const float *x, float *y, int n) {
  const __m256 va = _mm256_set1_ps(a); int i = 0;
  for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
  for (; i < n; i++) y[i] += a * x[i];
}

void to_f16(const float *x, uint16_t *y, int n) {
  int i = 0; for (; i + 8 <= n; i += 8) _mm_storeu_si128((__m128i *)(y + i), _mm256_cvtps_ph(_mm256_loadu_ps(x + i), 0));
  for (; i < n; i++) y[i] = _cvtss_sh(x[i], 0);
}
float dot_f16(const float *a, const uint16_t *b, int n) {
  __m256 s0 = _mm256_setzero_ps(), s1 = s0; int i = 0;
  for (; i + 16 <= n; i += 16) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(b + i))), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(b + i + 8))), s1);
  }
  for (; i + 8 <= n; i += 8) s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(b + i))), s0);
  float s = hsum256(_mm256_add_ps(s0, s1)); for (; i < n; i++) s += a[i] * _cvtsh_ss(b[i]);
  return s;
}
void axpy_f16(float a, const uint16_t *x, float *y, int n) {
  const __m256 va = _mm256_set1_ps(a); int i = 0;
  for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x + i))), _mm256_loadu_ps(y + i)));
  for (; i < n; i++) y[i] += a * _cvtsh_ss(x[i]);
}

void rmsnorm(const float *x, const float *w, int n, float eps, float *out, bool zc) {
  float ss = 0; for (int i = 0; i < n; i++) ss += x[i] * x[i];
  const float inv = 1.f / std::sqrt(ss / n + eps);
  if (zc) for (int i = 0; i < n; i++) out[i] = x[i] * inv * (1.f + w[i]);
  else for (int i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

}  // namespace qp
