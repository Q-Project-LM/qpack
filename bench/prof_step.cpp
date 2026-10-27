// Where does one decode step go? Times step() and each weight matrix product alone (same shapes, same pool).
// usage: prof_step model.qpack [threads]
#include <chrono>
#include <cstdio>
#include <map>
#include <string>
#include "qpack/qpack.h"
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char **argv) {
  if (argc > 2) qp::set_threads(atoi(argv[2]));
  auto m = qp::load_model(argv[1], 1024);
  for (int i = 0; i < 64; i++) m->step(100 + i, 1);
  double t0 = now(); const int N = 64; for (int i = 0; i < N; i++) m->step(200 + i, 1); const double step = (now() - t0) / N;
  std::map<std::string, double> cat; double total = 0; std::vector<float> x(1 << 16, 0.01f), y(1 << 18);
  for (auto &kv : m->container().tensors()) {
    const qp::Tensor &t = kv.second; if (t.shape.size() != 2 || t.numel() < 4096 || t.kind == qp::Kind::I64) continue;
    if (kv.first.find("ple.table") != std::string::npos || kv.first == "embed.weight" || (kv.first == "embed" && !m->config()["tie_embeddings"].as_bool())) continue;
    for (int r = 0; r < 3; r++) qp::matvec(t, x.data(), y.data(), (int)t.rows(), (int)t.cols());
    double a = now(); const int R = 20; for (int r = 0; r < R; r++) qp::matvec(t, x.data(), y.data(), (int)t.rows(), (int)t.cols()); double d = (now() - a) / R;
    std::string k = kv.first; size_t p = k.rfind('.'); std::string c = k.compare(0, 7, "layers.") == 0 ? k.substr(k.find('.', 7) + 1) : k; (void)p;
    cat[c] += d; total += d;
  }
  printf("%s, %d threads: step %.3f ms (%.1f tok/s); matrices alone %.3f ms (%.0f%%)\n", argv[1], qp::pool().size(), step * 1e3, 1 / step, total * 1e3, 100 * total / step);
  for (auto &kv : cat) printf("  %-28s %.3f ms\n", kv.first.c_str(), kv.second * 1e3);
}
