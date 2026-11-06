// q1 kernel unit check (not built by CMake):
//   g++ -O2 -march=native -std=c++17 -Iinclude bench/q1_check.cpp build/libqpack.a -lpthread -o q1_check
// 1. matvec results of a random q1 matrix vs a double-precision reference built from the bit layout in QPACK2.md
//    (tolerance: int16 activation rounding), including cols that are not a multiple of 128 (2880 = 22.5 * 128)
// 2. bit-identity of matvec, matvec_n, exact-batch matmul (B = 9, i.e. row_q1_multi) and the tiled matmul (mm_q1)
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "qpack/qpack.h"
using namespace qp;

int main() {
  std::mt19937 rng(7); std::normal_distribution<float> nd(0, 1); int bad = 0;
  for (int cols : {1, 5, 100, 128, 129, 256, 896, 2880, 4864}) for (int rows : {1, 7, 64}) {
    const int G = (cols + 127) / 128;
    std::vector<uint8_t> bits((size_t)rows * G * 16); std::vector<uint16_t> sc((size_t)rows * G);
    for (auto &b : bits) b = (uint8_t)rng();
    for (int r = 0; r < rows; r++) {   // padding bits of the last group are 0 as the quantiser writes them
      const int n = cols - 128 * (G - 1);
      for (int c = n; c < 128; c++) bits[((size_t)r * G + G - 1) * 16 + (c >> 3)] &= ~(1 << (c & 7));
      for (int g = 0; g < G; g++) sc[(size_t)r * G + g] = f32_to_f16(0.01f + 0.05f * std::fabs(nd(rng)));
    }
    Tensor W; W.name = "t"; W.shape = {rows, cols}; W.kind = Kind::Q1; W.data = bits.data(); W.scale16 = sc.data(); W.group = 128;
    const int B = 9; std::vector<float> X((size_t)B * cols);
    for (int b = 0; b < B; b++) for (int c = 0; c < cols; c++) X[(size_t)b * cols + c] = nd(rng) * (c % 97 == 0 ? 30.f : 1.f);
    auto bit_at = [&](int r, int c) { const int g = c / 128, k = c % 128; return (bits[((size_t)r * G + g) * 16 + (k >> 3)] >> (k & 7)) & 1; };
    auto ref = [&](int b, int r) {
      double s = 0, sa = 0;
      for (int c = 0; c < cols; c++) {
        const double w = (2 * bit_at(r, c) - 1) * (double)f16_to_f32(sc[(size_t)r * G + c / 128]);
        s += w * X[(size_t)b * cols + c]; sa += std::fabs(w * X[(size_t)b * cols + c]);
      }
      return std::make_pair(s, sa);
    };
    std::vector<float> row(cols), Yv((size_t)B * rows), Ym((size_t)B * rows), Ye((size_t)B * rows), Yn((size_t)rows), Y2((size_t)rows);
    int d_row = 0;
    for (int r = 0; r < rows; r++) {
      row_f32(W, r, row.data());
      for (int c = 0; c < cols; c++) d_row += row[c] != (2 * bit_at(r, c) - 1) * f16_to_f32(sc[(size_t)r * G + c / 128]);
    }
    for (int b = 0; b < B; b++) matvec(W, X.data() + (size_t)b * cols, Yv.data() + (size_t)b * rows, rows, cols);
    double worst = 0;
    for (int b = 0; b < B; b++) for (int r = 0; r < rows; r++) {
      auto sp = ref(b, r); worst = std::max(worst, std::fabs(Yv[(size_t)b * rows + r] - sp.first) / (sp.second * 1e-3 + 1e-9));
    }
    matmul(W, X.data(), B, cols, Ym.data(), rows, rows, cols);   // tiled path
    { ExactBatches e; matmul(W, X.data(), B, cols, Ye.data(), rows, rows, cols); }
    const Tensor *w2[2] = {&W, &W}; float *y2[2] = {Yn.data(), Y2.data()}; matvec_n(w2, 2, X.data(), y2, cols);
    int d_mm = 0, d_ex = 0, d_n = 0;
    for (size_t i = 0; i < Yv.size(); i++) { d_mm += memcmp(&Yv[i], &Ym[i], 4) != 0; d_ex += memcmp(&Yv[i], &Ye[i], 4) != 0; }
    for (int r = 0; r < rows; r++) d_n += (memcmp(&Yv[r], &Yn[r], 4) != 0) + (memcmp(&Yv[r], &Y2[r], 4) != 0);
    const bool ok = worst < 1.0 && !d_mm && !d_ex && !d_n && !d_row;
    printf("cols %5d rows %3d: max err %.3g (x1e-3 of sum|w x|), row_f32 bad %d, mm!=mv %d, exact!=mv %d, matvec_n!=mv %d  %s\n", cols, rows, worst, d_row, d_mm, d_ex, d_n, ok ? "ok" : "FAIL");
    bad += !ok;
  }
  puts(bad ? "FAILED" : "ALL OK"); return bad != 0;
}
