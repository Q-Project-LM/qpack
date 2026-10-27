#include <chrono>
#include <cstdio>
#include "qpack/qpack.h"
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char **argv) {
  qp::set_threads(atoi(argv[3]));
  qp::Container c(argv[1]); const qp::Tensor &t = c.at(argv[2]);
  std::vector<float> x(t.cols(), 0.01f), y(t.rows());
  for (int r = 0; r < 10; r++) qp::matvec(t, x.data(), y.data(), (int)t.rows(), (int)t.cols());
  double a = now(); const int R = 2000; for (int r = 0; r < R; r++) qp::matvec(t, x.data(), y.data(), (int)t.rows(), (int)t.cols());
  printf("%s kind %d [%ld x %ld]: %.2f us\n", argv[2], (int)t.kind, (long)t.rows(), (long)t.cols(), (now() - a) / R * 1e6);
}
