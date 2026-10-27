// prefill_all logits == step logits; checkpoint/rollback restores the exact continuation
#include <cmath>
#include <cstdio>
#include "qpack/qpack.h"
int main(int argc, char **argv) {
  auto a = qp::load_model(argv[1], 512), b = qp::load_model(argv[1], 512); const int V = a->vocab();
  std::vector<int> ids; for (int i = 0; i < 40; i++) ids.push_back(100 + (i * 7919) % 5000);
  for (int i = 0; i < 30; i++) { a->step(ids[i], 0); b->step(ids[i], 0); }
  auto ck = b->checkpoint();
  std::vector<float> L(10 * (size_t)V); { qp::ExactBatches ex; b->prefill_all(ids.data() + 30, 10, L.data()); }
  double mx = 0; for (int i = 0; i < 10; i++) { const float *l = a->step(ids[30 + i], 1); for (int v = 0; v < V; v++) mx = std::max(mx, (double)std::fabs(l[v] - L[(size_t)i * V + v])); }
  b->rollback(ck); for (int i = 30; i < 33; i++) b->step(ids[i], 0);   // replay 3 of the 10, continue
  auto c = qp::load_model(argv[1], 512); for (int i = 0; i < 33; i++) c->step(ids[i], 0);
  const float *x = b->step(7, 1); std::vector<float> X(x, x + V); const float *y = c->step(7, 1);
  double mx2 = 0; for (int v = 0; v < V; v++) mx2 = std::max(mx2, (double)std::fabs(X[v] - y[v]));
  printf("prefill_all vs step: %.3g; after rollback + replay: %.3g\n", mx, mx2);
}
