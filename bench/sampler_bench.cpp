#include <chrono>
#include <cstdio>
#include "qpack/engine.h"
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main() {
  const int V = 151936; std::vector<float> lg(V); for (int i = 0; i < V; i++) lg[i] = (float)((i * 7919) % 1000) / 100.f;
  std::vector<int> hist(300); for (int i = 0; i < 300; i++) hist[i] = (i * 31) % V;
  for (float temp : {0.f, 0.7f}) for (float rp : {1.f, 1.1f}) {
    qp::SamplerParams p; p.temperature = temp; p.repeat_penalty = rp; qp::Sampler s(p);
    double a = now(); int x = 0; for (int r = 0; r < 200; r++) x += s.sample(lg.data(), V, hist);
    printf("temp %.1f rep %.1f: %.1f us/token (%d)\n", temp, rp, (now() - a) / 200 * 1e6, x % 7);
  }
}
