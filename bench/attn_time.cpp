// step time at growing context, and the share of attention (measured by timing a step with attention against the cost
// of the same step at position ~0)
#include <chrono>
#include <cstdio>
#include "qpack/qpack.h"
namespace qp { extern double g_attn_s; }
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char **argv) {
  qp::set_threads(atoi(argv[2])); auto m = qp::load_model(argv[1], 2100);
  std::vector<int> ids(2048); for (int i = 0; i < 2048; i++) ids[i] = 100 + (i * 37) % 5000;
  int done = 0;
  for (int target : {16, 128, 512, 1024, 2000}) {
    if (target > done) { m->prefill(ids.data() + done, target - done, 0); done = target; }
    qp::g_attn_s = 0; double a = now(); const int R = 24; for (int r = 0; r < R; r++) m->step(7, 1); const double d = (now() - a) / R;
    m->rewind(done);
    printf("ctx %5d: %.2f ms/token (%.1f tok/s), attention %.2f ms\n", target, d * 1e3, 1 / d, qp::g_attn_s / R * 1e3);
  }
}
