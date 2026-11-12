// Row kernels implemented per architecture (kernels_x86.cpp / kernels_neon.cpp), called by the dispatch in kernels.cpp.
// Weights are int8 (q8: one float scale per row) or 4-bit groups of 32 (q4: one fp16 scale per group); activations are
// int16 with one float scale per QB columns. y = W[r0..r1) x for the *rows_* functions; the *_multi functions compute one
// row against nt <= 4 vectors with, per vector, exactly the arithmetic of the matching rows_* function; mm_* are the tiled
// prompt kernels (Y[b * ldy + r], B vectors).
#pragma once
#include <cstdint>

namespace qp {
namespace impl {

constexpr int QB = 128;
// largest quantised activation magnitude: the NEON kernels split an int16 into two int8 planes with a rounding shift, which
// needs |x| <= 32511 to keep the high part within int8
#if defined(__aarch64__) || defined(_M_ARM64)
constexpr float kActMax = 32511.f;
#else
constexpr float kActMax = 32767.f;
#endif
// stores n quantised activations v[0..n) as columns c0.. of an activation row of `cols` columns (x86: plain int16;
// NEON: hi plane at byte c0, lo plane at byte cols + c0). The row kernels below read that layout.
void store_act_block(const int16_t *v, int n, int16_t *row, int c0, int cols);

void rows_f32(const float *W, const float *x, float *y, int r0, int r1, int cols);
void rows_f16(const uint16_t *W, const float *x, float *y, int r0, int r1, int cols);
void rows_q8(const int8_t *Q, const float *scale, const int16_t *xq, const float *bs, float *y, int r0, int r1, int cols);
void rows_q4(const uint8_t *Q, const uint16_t *S, const int16_t *xq, const float *bs, float *y, int r0, int r1, int cols, int gs);
void row_q8_multi(const int8_t *Q, const float *scale, const int16_t *XQ, int ldq, const float *BS, int nbs, float *Y, int ldy, int r, int nt, int cols);
void row_q4_multi(const uint8_t *Q, const uint16_t *S, const int16_t *XQ, int ldq, const float *BS, int nbs, float *Y, int ldy, int r, int nt, int cols, int gs);
void mm_q8(const int8_t *Q, const float *scale, const int16_t *XQ, const float *BS, int ldq, int nb, int B, float *Y, int ldy, int r0, int r1, int cols);
void mm_q4(const uint8_t *Q, const uint16_t *S, const int16_t *XQ, const float *BS, int ldq, int nb, int B, float *Y, int ldy, int r0, int r1, int cols, int gs);
void mm_f32(const float *w, const float *X, int ldx, int B, float *Y, int ldy, int r0, int r1, int cols);
void mm_f16(const uint16_t *w, const float *X, int ldx, int B, float *Y, int ldy, int r0, int r1, int cols);

}  // namespace impl
}  // namespace qp
