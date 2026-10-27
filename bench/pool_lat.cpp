#include <atomic>
#include <chrono>
#include <cstdio>
#include "qpack/qpack.h"
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char **argv) {
  qp::set_threads(atoi(argv[1])); std::atomic<long> s{0};
  for (int r = 0; r < 100; r++) qp::pool().parallel_for(1024, [&](int a, int b) { s += b - a; }, 1);
  double t = now(); const int R = 20000; for (int r = 0; r < R; r++) qp::pool().parallel_for(1024, [&](int a, int b) { s += b - a; }, 1);
  printf("%d threads: %.2f us per empty parallel_for\n", qp::pool().size(), (now() - t) / R * 1e6);
}
