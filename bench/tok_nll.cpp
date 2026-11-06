// Per-token NLL of the first window of a text (same windows as qpack-perplexity), one value per line: "pos token nll".
// build: g++ -O2 -march=native -std=c++17 -Iinclude bench/tok_nll.cpp build/libqpack.a -lpthread -o /tmp/tok_nll
// usage: tok_nll model.qpack text.txt out.txt [ctx=256] [chunk=0]
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include "qpack/engine.h"
int main(int argc, char **argv) {
  if (argc < 4) return 2;
  const int ctx = argc > 4 ? atoi(argv[4]) : 256, chunk = argc > 5 ? atoi(argv[5]) : 0;
  qp::set_threads(8);
  qp::Session s(qp::load_model(argv[1], ctx));
  std::ifstream in(argv[2], std::ios::binary); std::stringstream ss; ss << in.rdbuf();
  std::vector<int> ids = s.tok().encode(ss.str()); std::vector<int> w(ids.begin() + (size_t)chunk * ctx, ids.begin() + (size_t)(chunk + 1) * ctx);
  const int V = s.model().vocab(); std::vector<float> lg((size_t)64 * V); s.model().reset(); FILE *f = fopen(argv[3], "w");
  for (int i0 = 0; i0 + 1 < (int)w.size(); i0 += 64) {
    const int nb = std::min(64, (int)w.size() - 1 - i0); s.model().prefill_all(w.data() + i0, nb, lg.data());
    for (int b = 0; b < nb; b++) {
      const float *l = lg.data() + (size_t)b * V; float mx = l[0]; for (int v = 1; v < V; v++) mx = std::max(mx, l[v]);
      double z = 0; for (int v = 0; v < V; v++) z += std::exp((double)l[v] - mx);
      fprintf(f, "%d %d %.5f %s\n", i0 + b + 1, w[i0 + b + 1], mx + std::log(z) - l[w[i0 + b + 1]], s.tok().piece(w[i0 + b + 1]).c_str());
    }
  }
  fclose(f); return 0;
}
