// AArch64 kernels (NEON): the same algorithms and the same arithmetic as kernels_x86.cpp. int8 / int4 weights are widened
// to int16 and multiplied by the int16 activations with vmlal_s16 (exact int32 sums), then one float FMA per block / group.
#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>

#include <cmath>
#include <cstring>
#include <functional>
#include <vector>

#include "kernels_impl.h"
#include "qpack/qpack.h"

namespace qp {
namespace {
using impl::QB;

inline float32x4_t cvt_lo(uint16x8_t h) { return vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(h))); }
inline float32x4_t cvt_hi(uint16x8_t h) { return vcvt_high_f32_f16(vreinterpretq_f16_u16(h)); }
inline float32x4_t ld_f16(const uint16_t *p) { return vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(p))); }

// Activations arrive as two int8 planes per row (see store_act_block): x = 256 * hi + lo with lo in [-128, 127]; the lo plane
// starts `cols` bytes after the hi plane. A dot product with int8 weights is 256 * sdot(w, hi) + sdot(w, lo) -- exactly the
// int32 sum of the int16 products, with two instructions per 16 columns instead of six.
inline int32x4_t sdot(int32x4_t acc, int8x16_t a, int8x16_t b) {
#ifdef __ARM_FEATURE_DOTPROD
  return vdotq_s32(acc, a, b);
#else
  int16x8_t t = vmull_s8(vget_low_s8(a), vget_low_s8(b)); t = vmlal_high_s8(t, a, b); return vpadalq_s16(acc, t);
#endif
}
inline const int8_t *hi_plane(const int16_t *x) { return (const int8_t *)x; }
inline int xval(const int16_t *x, int cols, int c) { return 256 * ((const int8_t *)x)[c] + ((const int8_t *)x)[cols + c]; }
inline int comb(int32x4_t h, int32x4_t l) { return (vaddvq_s32(h) << 8) + vaddvq_s32(l); }

// q4 group: 16 bytes, byte j = column j (low nibble) and column j + 16 (high nibble), both stored as q + 8
inline void unpack_q4(const uint8_t *p, int8x16_t &lo, int8x16_t &hi) {
  const uint8x16_t b = vld1q_u8(p); const int8x16_t e = vdupq_n_s8(8);
  lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(b, vdupq_n_u8(15))), e); hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(b, 4)), e);
}
// one q4 group (32 columns) against the activations at xh / xl (plane pointers at the group's first column): 4 int32 partial sums
inline int32x4_t q4_group(int8x16_t lo, int8x16_t hi, const int8_t *xh, const int8_t *xl) {
  int32x4_t h = sdot(sdot(vdupq_n_s32(0), lo, vld1q_s8(xh)), hi, vld1q_s8(xh + 16));
  int32x4_t l = sdot(sdot(vdupq_n_s32(0), lo, vld1q_s8(xl)), hi, vld1q_s8(xl + 16));
  return vaddq_s32(vshlq_n_s32(h, 8), l);
}
// integer dot product of 4 q8 rows against one block (QB = 128 columns); the x loads are shared by the rows
inline void block_q8_x4(const int8_t *q0, int64_t ld, const int8_t *xh, const int8_t *xl, int out[4]) {
  int32x4_t h[4], l[4]; for (int i = 0; i < 4; i++) h[i] = l[i] = vdupq_n_s32(0);
  for (int c = 0; c < QB; c += 16) {
    const int8x16_t vh = vld1q_s8(xh + c), vl = vld1q_s8(xl + c);
    for (int i = 0; i < 4; i++) { const int8x16_t w = vld1q_s8(q0 + i * ld + c); h[i] = sdot(h[i], w, vh); l[i] = sdot(l[i], w, vl); }
  }
  for (int i = 0; i < 4; i++) out[i] = comb(h[i], l[i]);
}
inline int block_q8(const int8_t *qb, const int8_t *xh, const int8_t *xl) {
  int32x4_t h0 = vdupq_n_s32(0), h1 = h0, l0 = h0, l1 = h0;
  for (int c = 0; c < QB; c += 32) {
    const int8x16_t w0 = vld1q_s8(qb + c), w1 = vld1q_s8(qb + c + 16);
    h0 = sdot(h0, w0, vld1q_s8(xh + c)); l0 = sdot(l0, w0, vld1q_s8(xl + c)); h1 = sdot(h1, w1, vld1q_s8(xh + c + 16)); l1 = sdot(l1, w1, vld1q_s8(xl + c + 16));
  }
  return comb(vaddq_s32(h0, h1), vaddq_s32(l0, l1));
}

}  // namespace

namespace impl {

void store_act_block(const int16_t *v, int n, int16_t *row, int c0, int cols) {
  int8_t *hi = (int8_t *)row + c0, *lo = (int8_t *)row + cols + c0; int i = 0;
  for (; i + 8 <= n; i += 8) { const int16x8_t x = vld1q_s16(v + i); vst1_s8(hi + i, vrshrn_n_s16(x, 8)); vst1_s8(lo + i, vmovn_s16(x)); }
  for (; i < n; i++) { const int h = (v[i] + 128) >> 8; hi[i] = (int8_t)h; lo[i] = (int8_t)(v[i] - 256 * h); }
}

void rows_f32(const float *W, const float *x, float *y, int r0, int r1, int cols) {
  for (int r = r0; r < r1; r++) {
    const float *w = W + (int64_t)r * cols; float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0; int c = 0;
    for (; c + 16 <= cols; c += 16) {
      a0 = vfmaq_f32(a0, vld1q_f32(w + c), vld1q_f32(x + c)); a1 = vfmaq_f32(a1, vld1q_f32(w + c + 4), vld1q_f32(x + c + 4));
      a2 = vfmaq_f32(a2, vld1q_f32(w + c + 8), vld1q_f32(x + c + 8)); a3 = vfmaq_f32(a3, vld1q_f32(w + c + 12), vld1q_f32(x + c + 12));
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; c < cols; c++) s += w[c] * x[c];
    y[r] = s;
  }
}
void rows_f16(const uint16_t *W, const float *x, float *y, int r0, int r1, int cols) {
  for (int r = r0; r < r1; r++) {
    const uint16_t *w = W + (int64_t)r * cols; float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0; int c = 0;
    for (; c + 16 <= cols; c += 16) {
      const uint16x8_t h0 = vld1q_u16(w + c), h1 = vld1q_u16(w + c + 8);
      a0 = vfmaq_f32(a0, cvt_lo(h0), vld1q_f32(x + c)); a1 = vfmaq_f32(a1, cvt_hi(h0), vld1q_f32(x + c + 4));
      a2 = vfmaq_f32(a2, cvt_lo(h1), vld1q_f32(x + c + 8)); a3 = vfmaq_f32(a3, cvt_hi(h1), vld1q_f32(x + c + 12));
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; c < cols; c++) s += f16_to_f32(w[c]) * x[c];
    y[r] = s;
  }
}

void rows_q8(const int8_t *Q, const float *scale, const int16_t *xq, const float *bs, float *y, int r0, int r1, int cols) {
  const int nb = cols / QB; const int8_t *xh = hi_plane(xq), *xl = xh + cols;
  int r = r0;
  for (; r + 4 <= r1; r += 4) {
    const int8_t *q = Q + (int64_t)r * cols; float acc[4] = {0, 0, 0, 0};
    for (int b = 0; b < nb; b++) { int d[4]; block_q8_x4(q + b * QB, cols, xh + b * QB, xl + b * QB, d); for (int i = 0; i < 4; i++) acc[i] += bs[b] * (float)d[i]; }
    for (int i = 0; i < 4; i++) {
      float tail = 0; for (int c = nb * QB; c < cols; c++) tail += (float)q[(int64_t)i * cols + c] * (float)xval(xq, cols, c);
      if (nb * QB < cols) acc[i] += bs[nb] * tail;
      y[r + i] = acc[i] * scale[r + i];
    }
  }
  for (; r < r1; r++) {
    const int8_t *q = Q + (int64_t)r * cols; float acc = 0;
    for (int b = 0; b < nb; b++) acc += bs[b] * (float)block_q8(q + b * QB, xh + b * QB, xl + b * QB);
    float tail = 0; for (int c = nb * QB; c < cols; c++) tail += (float)q[c] * (float)xval(xq, cols, c);
    if (nb * QB < cols) acc += bs[nb] * tail;
    y[r] = acc * scale[r];
  }
}
void rows_q4(const uint8_t *Q, const uint16_t *S, const int16_t *xq, const float *bs, float *y, int r0, int r1, int cols, int gs) {
  const int G = cols / 32; const int8_t *xh = hi_plane(xq), *xl = xh + cols;
  float bsx[1024], cmb[1024], sf[1024];   // activation block scale per group, and group scale x block scale (cols <= 32768)
  for (int k = 0; k < G; k++) bsx[k] = bs[k >> 2];
  for (int r = r0; r < r1; r++) {
    const uint8_t *q = Q + (int64_t)r * (cols / 2); const uint16_t *s = S + (int64_t)r * (G >> gs);
    for (int k = 0; k < G; k++) sf[k] = f16_to_f32(s[k >> gs]);
    int k = 0;
    for (; k + 4 <= G; k += 4) vst1q_f32(cmb + k, vmulq_f32(vld1q_f32(sf + k), vld1q_f32(bsx + k)));
    for (; k < G; k++) cmb[k] = sf[k] * bsx[k];
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0; k = 0;
    for (; k + 2 <= G; k += 2) {
      int8x16_t l0, h0, l1, h1; unpack_q4(q + 16 * k, l0, h0); unpack_q4(q + 16 * k + 16, l1, h1);
      a0 = vfmaq_n_f32(a0, vcvtq_f32_s32(q4_group(l0, h0, xh + 32 * k, xl + 32 * k)), cmb[k]);
      a1 = vfmaq_n_f32(a1, vcvtq_f32_s32(q4_group(l1, h1, xh + 32 * k + 32, xl + 32 * k + 32)), cmb[k + 1]);
    }
    for (; k < G; k++) { int8x16_t l0, h0; unpack_q4(q + 16 * k, l0, h0); a0 = vfmaq_n_f32(a0, vcvtq_f32_s32(q4_group(l0, h0, xh + 32 * k, xl + 32 * k)), cmb[k]); }
    y[r] = vaddvq_f32(vaddq_f32(a0, a1));
  }
}

void row_q8_multi(const int8_t *Q, const float *scale, const int16_t *XQ, int ldq, const float *BS, int nbs, float *Y, int ldy, int r, int nt, int cols) {
  const int nb = cols / QB; const int8_t *q = Q + (int64_t)r * cols; float acc[4] = {0, 0, 0, 0};
  for (int b = 0; b < nb; b++)
    for (int j = 0; j < nt; j++) { const int8_t *xh = hi_plane(XQ + (int64_t)j * ldq); acc[j] += BS[(int64_t)j * nbs + b] * (float)block_q8(q + b * QB, xh + b * QB, xh + cols + b * QB); }
  for (int j = 0; j < nt; j++) {
    const int16_t *xq = XQ + (int64_t)j * ldq; const float *bs = BS + (int64_t)j * nbs;
    float tail = 0; for (int c = nb * QB; c < cols; c++) tail += (float)q[c] * (float)xval(xq, cols, c);
    if (nb * QB < cols) acc[j] += bs[nb] * tail;
    Y[(int64_t)j * ldy + r] = acc[j] * scale[r];
  }
}
void row_q4_multi(const uint8_t *Q, const uint16_t *S, const int16_t *XQ, int ldq, const float *BS, int nbs, float *Y, int ldy, int r, int nt, int cols, int gs) {
  const int G = cols / 32; const uint8_t *q = Q + (int64_t)r * (cols / 2); const uint16_t *s = S + (int64_t)r * (G >> gs);
  float cmb[4][1024], sf[1024];
  for (int k = 0; k < G; k++) sf[k] = f16_to_f32(s[k >> gs]);
  for (int j = 0; j < nt; j++) {
    const float *bs = BS + (int64_t)j * nbs; float bsx[1024]; for (int k = 0; k < G; k++) bsx[k] = bs[k >> 2];
    int k = 0;
    for (; k + 4 <= G; k += 4) vst1q_f32(cmb[j] + k, vmulq_f32(vld1q_f32(sf + k), vld1q_f32(bsx + k)));
    for (; k < G; k++) cmb[j][k] = sf[k] * bsx[k];
  }
  float32x4_t a0[4], a1[4]; for (int j = 0; j < nt; j++) a0[j] = a1[j] = vdupq_n_f32(0);
  int k = 0;
  for (; k + 2 <= G; k += 2) {
    int8x16_t l0, h0, l1, h1; unpack_q4(q + 16 * k, l0, h0); unpack_q4(q + 16 * k + 16, l1, h1);
    for (int j = 0; j < nt; j++) {
      const int8_t *xh = hi_plane(XQ + (int64_t)j * ldq) + 32 * k, *xl = xh + cols;
      a0[j] = vfmaq_n_f32(a0[j], vcvtq_f32_s32(q4_group(l0, h0, xh, xl)), cmb[j][k]);
      a1[j] = vfmaq_n_f32(a1[j], vcvtq_f32_s32(q4_group(l1, h1, xh + 32, xl + 32)), cmb[j][k + 1]);
    }
  }
  for (; k < G; k++) {
    int8x16_t l0, h0; unpack_q4(q + 16 * k, l0, h0);
    for (int j = 0; j < nt; j++) { const int8_t *xh = hi_plane(XQ + (int64_t)j * ldq) + 32 * k; a0[j] = vfmaq_n_f32(a0[j], vcvtq_f32_s32(q4_group(l0, h0, xh, xh + cols)), cmb[j][k]); }
  }
  for (int j = 0; j < nt; j++) Y[(int64_t)j * ldy + r] = vaddvq_f32(vaddq_f32(a0[j], a1[j]));
}

// ---- batched products (prompt): 4 vectors per pass so every weight load feeds four vectors; q8 also takes 2 rows per pass
void mm_q8(const int8_t *Q, const float *scale, const int16_t *XQ, const float *BS, int ldq, int nb, int B, float *Y, int ldy, int r0, int r1, int cols) {
  const int full = cols / QB;
  int r = r0;
  for (; r + 2 <= r1 && full * QB == cols; r += 2) {
    const int8_t *qa = Q + (int64_t)r * cols, *qb = qa + cols;
    int b = 0;
    for (; b + 4 <= B; b += 4) {
      const int8_t *xh[4]; const float *s[4];
      for (int j = 0; j < 4; j++) { xh[j] = hi_plane(XQ + (int64_t)(b + j) * ldq); s[j] = BS + (int64_t)(b + j) * nb; }
      float32x4_t f[8]; for (int j = 0; j < 8; j++) f[j] = vdupq_n_f32(0);
      for (int k = 0; k < full; k++) {
        int32x4_t h[8], l[8]; for (int j = 0; j < 8; j++) h[j] = l[j] = vdupq_n_s32(0);
        for (int c = k * QB; c < (k + 1) * QB; c += 16) {
          const int8x16_t wa = vld1q_s8(qa + c), wb = vld1q_s8(qb + c);
          for (int j = 0; j < 4; j++) {
            const int8x16_t vh = vld1q_s8(xh[j] + c), vl = vld1q_s8(xh[j] + cols + c);
            h[j] = sdot(h[j], wa, vh); l[j] = sdot(l[j], wa, vl); h[4 + j] = sdot(h[4 + j], wb, vh); l[4 + j] = sdot(l[4 + j], wb, vl);
          }
        }
        for (int j = 0; j < 8; j++) f[j] = vfmaq_n_f32(f[j], vcvtq_f32_s32(vaddq_s32(vshlq_n_s32(h[j], 8), l[j])), s[j & 3][k]);
      }
      for (int j = 0; j < 4; j++) { Y[(int64_t)(b + j) * ldy + r] = vaddvq_f32(f[j]) * scale[r]; Y[(int64_t)(b + j) * ldy + r + 1] = vaddvq_f32(f[4 + j]) * scale[r + 1]; }
    }
    for (; b < B; b++) rows_q8(Q, scale, XQ + (int64_t)b * ldq, BS + (int64_t)b * nb, Y + (int64_t)b * ldy, r, r + 2, cols);
  }
  for (; r < r1; r++) {
    const int8_t *q = Q + (int64_t)r * cols;
    int b = 0;
    for (; b + 4 <= B; b += 4) {
      const int16_t *x[4]; const float *s[4];
      for (int j = 0; j < 4; j++) { x[j] = XQ + (int64_t)(b + j) * ldq; s[j] = BS + (int64_t)(b + j) * nb; }
      float32x4_t f[4]; for (int j = 0; j < 4; j++) f[j] = vdupq_n_f32(0);
      for (int k = 0; k < full; k++) {
        int32x4_t h[4], l[4]; for (int j = 0; j < 4; j++) h[j] = l[j] = vdupq_n_s32(0);
        for (int c = k * QB; c < (k + 1) * QB; c += 16) {
          const int8x16_t w = vld1q_s8(q + c);
          for (int j = 0; j < 4; j++) { h[j] = sdot(h[j], w, vld1q_s8(hi_plane(x[j]) + c)); l[j] = sdot(l[j], w, vld1q_s8(hi_plane(x[j]) + cols + c)); }
        }
        for (int j = 0; j < 4; j++) f[j] = vfmaq_n_f32(f[j], vcvtq_f32_s32(vaddq_s32(vshlq_n_s32(h[j], 8), l[j])), s[j][k]);
      }
      float t[4]; for (int j = 0; j < 4; j++) t[j] = vaddvq_f32(f[j]);
      if (full * QB < cols)
        for (int j = 0; j < 4; j++) { float tail = 0; for (int c = full * QB; c < cols; c++) tail += (float)q[c] * (float)xval(x[j], cols, c); t[j] += s[j][full] * tail; }
      for (int j = 0; j < 4; j++) Y[(int64_t)(b + j) * ldy + r] = t[j] * scale[r];
    }
    for (; b < B; b++) rows_q8(Q, scale, XQ + (int64_t)b * ldq, BS + (int64_t)b * nb, Y + (int64_t)b * ldy, r, r + 1, cols);
  }
}
void mm_q4(const uint8_t *Q, const uint16_t *S, const int16_t *XQ, const float *BS, int ldq, int nb, int B, float *Y, int ldy, int r0, int r1, int cols, int gs) {
  const int G = cols / 32;
  std::vector<float> bsx((size_t)B * G), cmb(4 * (size_t)G), sw(G);   // per vector: activation block scale per group
  for (int b = 0; b < B; b++) for (int k = 0; k < G; k++) bsx[(size_t)b * G + k] = BS[(size_t)b * nb + (k >> 2)];
  for (int r = r0; r < r1; r++) {
    const uint8_t *q = Q + (int64_t)r * (cols / 2); const uint16_t *s = S + (int64_t)r * (G >> gs);
    for (int k = 0; k < G; k++) sw[k] = f16_to_f32(s[k >> gs]);
    int b = 0;
    for (; b + 4 <= B; b += 4) {
      const int8_t *xh[4]; for (int j = 0; j < 4; j++) xh[j] = hi_plane(XQ + (int64_t)(b + j) * ldq);
      for (int j = 0; j < 4; j++) { const float *bj = bsx.data() + (size_t)(b + j) * G; float *cj = cmb.data() + (size_t)j * G; for (int kk = 0; kk < G; kk++) cj[kk] = sw[kk] * bj[kk]; }
      const float *c[4] = {cmb.data(), cmb.data() + G, cmb.data() + 2 * G, cmb.data() + 3 * G};
      float32x4_t f[4]; for (int j = 0; j < 4; j++) f[j] = vdupq_n_f32(0);
      for (int k = 0; k < G; k++) {
        int8x16_t lo, hi; unpack_q4(q + 16 * k, lo, hi);
        for (int j = 0; j < 4; j++) f[j] = vfmaq_n_f32(f[j], vcvtq_f32_s32(q4_group(lo, hi, xh[j] + 32 * k, xh[j] + cols + 32 * k)), c[j][k]);
      }
      for (int j = 0; j < 4; j++) Y[(int64_t)(b + j) * ldy + r] = vaddvq_f32(f[j]);
    }
    for (; b < B; b++) rows_q4(Q, S, XQ + (int64_t)b * ldq, BS + (int64_t)b * nb, Y + (int64_t)b * ldy, r, r + 1, cols, gs);
  }
}

// float weights: 4 vectors per pass, each weight vector loaded (and converted) once
template <typename LoadW>
static void mm_float(LoadW loadw, const float *X, int ldx, int B, float *Y, int ldy, int r0, int r1, int cols, const std::function<float(int, int)> &tail) {
  for (int r = r0; r < r1; r++) {
    int b = 0;
    for (; b + 4 <= B; b += 4) {
      const float *x[4] = {X + (int64_t)b * ldx, X + (int64_t)(b + 1) * ldx, X + (int64_t)(b + 2) * ldx, X + (int64_t)(b + 3) * ldx};
      float32x4_t a[4]; for (int j = 0; j < 4; j++) a[j] = vdupq_n_f32(0);
      int c = 0;
      for (; c + 4 <= cols; c += 4) { const float32x4_t w = loadw(r, c); for (int j = 0; j < 4; j++) a[j] = vfmaq_f32(a[j], w, vld1q_f32(x[j] + c)); }
      for (int j = 0; j < 4; j++) { float t = vaddvq_f32(a[j]); for (int cc = c; cc < cols; cc++) t += tail(r, cc) * x[j][cc]; Y[(int64_t)(b + j) * ldy + r] = t; }
    }
    for (; b < B; b++) {
      const float *x = X + (int64_t)b * ldx; float32x4_t a = vdupq_n_f32(0); int c = 0;
      for (; c + 4 <= cols; c += 4) a = vfmaq_f32(a, loadw(r, c), vld1q_f32(x + c));
      float t = vaddvq_f32(a); for (; c < cols; c++) t += tail(r, c) * x[c];
      Y[(int64_t)b * ldy + r] = t;
    }
  }
}
void mm_f32(const float *w, const float *X, int ldx, int B, float *Y, int ldy, int r0, int r1, int cols) {
  mm_float([&](int r, int c) { return vld1q_f32(w + (int64_t)r * cols + c); }, X, ldx, B, Y, ldy, r0, r1, cols, [&](int r, int c) { return w[(int64_t)r * cols + c]; });
}
void mm_f16(const uint16_t *w, const float *X, int ldx, int B, float *Y, int ldy, int r0, int r1, int cols) {
  mm_float([&](int r, int c) { return ld_f16(w + (int64_t)r * cols + c); }, X, ldx, B, Y, ldy, r0, r1, cols, [&](int r, int c) { return f16_to_f32(w[(int64_t)r * cols + c]); });
}

}  // namespace impl

// ---- elementwise
namespace {
// exp(x) for 4 floats: 2^k * p(r), |r| <= ln2/2, degree-6 polynomial (relative error ~2e-7), clamped to [-87, 88]
inline float32x4_t exp4(float32x4_t x) {
  x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.f)), vdupq_n_f32(88.f));
  const float32x4_t k = vrndnq_f32(vmulq_n_f32(x, 1.44269504088896341f));
  const float32x4_t r = vfmsq_f32(vfmsq_f32(x, k, vdupq_n_f32(-2.12194440e-4f)), k, vdupq_n_f32(0.693359375f));
  float32x4_t p = vdupq_n_f32(1.9875691500e-4f);
  p = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), p, r); p = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), p, r);
  p = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), p, r); p = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), p, r);
  p = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), p, r);
  p = vfmaq_f32(vaddq_f32(r, vdupq_n_f32(1.f)), vmulq_f32(p, r), r);
  const int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(k), vdupq_n_s32(127)), 23);
  return vmulq_f32(p, vreinterpretq_f32_s32(e));
}
}  // namespace

void exp_v(const float *x, float *y, int n) {
  int i = 0; for (; i + 4 <= n; i += 4) vst1q_f32(y + i, exp4(vld1q_f32(x + i)));
  for (; i < n; i++) y[i] = std::exp(x[i]);
}
void sigmoid_v(const float *x, float *y, int n) {
  const float32x4_t one = vdupq_n_f32(1.f); int i = 0;
  for (; i + 4 <= n; i += 4) vst1q_f32(y + i, vdivq_f32(one, vaddq_f32(one, exp4(vnegq_f32(vld1q_f32(x + i))))));
  for (; i < n; i++) y[i] = 1.f / (1.f + std::exp(-x[i]));
}
void silu_mul(float *g, const float *u, int n) {   // g = silu(g) * u
  const float32x4_t one = vdupq_n_f32(1.f); int i = 0;
  for (; i + 4 <= n; i += 4) {
    const float32x4_t v = vld1q_f32(g + i);
    vst1q_f32(g + i, vmulq_f32(vdivq_f32(v, vaddq_f32(one, exp4(vnegq_f32(v)))), vld1q_f32(u + i)));
  }
  for (; i < n; i++) g[i] = g[i] / (1.f + std::exp(-g[i])) * u[i];
}

float dot_f32(const float *a, const float *b, int n) {
  float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0; int i = 0;
  for (; i + 16 <= n; i += 16) {
    s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i)); s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8)); s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
  }
  for (; i + 4 <= n; i += 4) s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
  float s = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3))); for (; i < n; i++) s += a[i] * b[i];
  return s;
}
void axpy_f32(float a, const float *x, float *y, int n) {
  int i = 0;
  for (; i + 4 <= n; i += 4) vst1q_f32(y + i, vfmaq_n_f32(vld1q_f32(y + i), vld1q_f32(x + i), a));
  for (; i < n; i++) y[i] += a * x[i];
}

void to_f16(const float *x, uint16_t *y, int n) {
  int i = 0; for (; i + 4 <= n; i += 4) vst1_u16(y + i, vreinterpret_u16_f16(vcvt_f16_f32(vld1q_f32(x + i))));
  for (; i < n; i++) y[i] = f32_to_f16(x[i]);
}
void f16_to_f32_v(const uint16_t *h, float *f, int n) {
  int i = 0; for (; i + 4 <= n; i += 4) vst1q_f32(f + i, ld_f16(h + i));
  for (; i < n; i++) f[i] = f16_to_f32(h[i]);
}
float dot_f16(const float *a, const uint16_t *b, int n) {
  float32x4_t s0 = vdupq_n_f32(0), s1 = s0; int i = 0;
  for (; i + 8 <= n; i += 8) {
    const uint16x8_t h = vld1q_u16(b + i);
    s0 = vfmaq_f32(s0, vld1q_f32(a + i), cvt_lo(h)); s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), cvt_hi(h));
  }
  float s = vaddvq_f32(vaddq_f32(s0, s1)); for (; i < n; i++) s += a[i] * f16_to_f32(b[i]);
  return s;
}
void axpy_f16(float a, const uint16_t *x, float *y, int n) {
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    const uint16x8_t h = vld1q_u16(x + i);
    vst1q_f32(y + i, vfmaq_n_f32(vld1q_f32(y + i), cvt_lo(h), a)); vst1q_f32(y + i + 4, vfmaq_n_f32(vld1q_f32(y + i + 4), cvt_hi(h), a));
  }
  for (; i < n; i++) y[i] += a * f16_to_f32(x[i]);
}

// ---- decode attention: the K / V row is converted to float once and shared by the rep query heads of its kv group
bool attn_scores(const uint16_t *kl, size_t kvd, const float *q, int g, int rep, int hd, int t0, int t1, int C, float scl, float *sc) {
  if (hd % 8 || hd > 256) return false;
  float kf[256];
  for (int t = t0; t < t1; t++) {
    f16_to_f32_v(kl + (size_t)t * kvd + (size_t)g * hd, kf, hd);
    for (int j = 0; j < rep; j++) sc[(size_t)j * C + (t - t0)] = dot_f32(q + (size_t)(g * rep + j) * hd, kf, hd) * scl;
  }
  return true;
}
bool attn_values(const uint16_t *vl, size_t kvd, int g, int rep, int hd, int t0, int t1, int C, const float *sc, float *out, size_t pstride) {
  if (hd % 8 || hd > 256) return false;
  float vf[256];
  for (int t = t0; t < t1; t++) {
    f16_to_f32_v(vl + (size_t)t * kvd + (size_t)g * hd, vf, hd);
    for (int j = 0; j < rep; j++) axpy_f32(sc[(size_t)j * C + (t - t0)], vf, out + j * pstride, hd);
  }
  return true;
}

void scan_gt(const float *x, int from, int to, float thr, std::vector<std::pair<int, float>> &out) {
  const float32x4_t t4 = vdupq_n_f32(thr); int v = from;
  for (; v + 16 <= to; v += 16) {
    const uint32x4_t m = vorrq_u32(vorrq_u32(vcgtq_f32(vld1q_f32(x + v), t4), vcgtq_f32(vld1q_f32(x + v + 4), t4)),
                                   vorrq_u32(vcgtq_f32(vld1q_f32(x + v + 8), t4), vcgtq_f32(vld1q_f32(x + v + 12), t4)));
    if (vmaxvq_u32(m)) for (int i = v; i < v + 16; i++) if (x[i] > thr) out.push_back({i, x[i]});
  }
  for (; v < to; v++) if (x[v] > thr) out.push_back({v, x[v]});
}

}  // namespace qp
#endif
