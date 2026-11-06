// mxfp4 kernel check: matvec / matmul (all batch shapes) against the float product with row_f32 (dequantised) weights, and
// exact equality between matvec, g_exact_batches matmul and the tiled matmul.
// build: g++ -O2 -march=native -std=c++17 -Iinclude bench/test_mx4.cpp build/libqpack.a -lpthread -o /tmp/test_mx4
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include "qpack/qpack.h"
using namespace qp;
int main() {
  std::mt19937 rng(1); const int R = 37, C = 2880;
  std::vector<uint8_t> q((size_t)R * C / 2), s((size_t)R * C / 32);
  for (auto &b : q) b = rng();
  for (auto &b : s) b = 110 + rng() % 12;
  Tensor W; W.name = "t"; W.shape = {R, C}; W.kind = Kind::Mxfp4; W.data = q.data(); W.scale8 = s.data(); W.group = 32;
  std::normal_distribution<float> nd; int B = 7; std::vector<float> X((size_t)B * C); for (auto &x : X) x = nd(rng);
  std::vector<float> Wf((size_t)R * C); for (int r = 0; r < R; r++) row_f32(W, r, Wf.data() + (size_t)r * C);
  double maxrel = 0; std::vector<float> Y1((size_t)B * R), Y2((size_t)B * R), Y3((size_t)B * R);
  for (int b = 0; b < B; b++) matvec(W, X.data() + (size_t)b * C, Y1.data() + (size_t)b * R, R, C);
  { ExactBatches eb; matmul(W, X.data(), B, C, Y2.data(), R, R, C); }
  matmul(W, X.data(), B, C, Y3.data(), R, R, C);
  for (int b = 0; b < B; b++) for (int r = 0; r < R; r++) {
    double ref = 0, nrm = 0;
    for (int c = 0; c < C; c++) { ref += (double)Wf[(size_t)r * C + c] * X[(size_t)b * C + c]; nrm += std::fabs((double)Wf[(size_t)r * C + c] * X[(size_t)b * C + c]); }
    maxrel = std::max(maxrel, std::fabs(Y1[(size_t)b * R + r] - ref) / nrm);
  }
  printf("max err relative to sum|w x|: %.2e; exact-batch identical: %d; tiled identical: %d\n", maxrel, !memcmp(Y1.data(), Y2.data(), 4 * Y1.size()), !memcmp(Y1.data(), Y3.data(), 4 * Y1.size()));
  return maxrel < 1e-3 ? 0 : 1;
}
