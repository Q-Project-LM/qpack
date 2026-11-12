// cuda_dump: bit-exactness harness for the CUDA backend. usage: cuda_dump -m model.qpack -n 64 text.txt out.bin
// writes the full logits of each step for the first n tokens of the text, then 40 device-greedy tokens; compare two builds with cmp.
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <sstream>

#include "qpack/engine.h"
int main(int argc, char **argv) {
  std::string path, txt, out; int n = 64, tps = 0;
  for (int i = 1; i < argc; i++) { std::string a = argv[i]; if (a == "-m") path = argv[++i]; else if (a == "-n") n = std::stoi(argv[++i]); else if (a == "--tps") tps = std::stoi(argv[++i]); else if (txt.empty()) txt = a; else out = a; }
  setenv("QPACK_DEVICE", "cuda", 1);
  qp::Session s(qp::load_model(path, n + 64));
  std::ifstream in(txt, std::ios::binary); std::stringstream ss; ss << in.rdbuf();
  auto ids = s.tok().encode(ss.str()); if ((int)ids.size() > n) ids.resize(n);
  if (tps) {   // decode speed: warm-up, then tps timed runs of 64 greedy tokens (one tok/s figure per run)
    auto &mm = s.model(); int g[64];
    for (int w = 0; w < 3; w++) { mm.reset(); mm.greedy_run(ids[0], 64, g); }
    for (int r = 0; r < tps; r++) {
      mm.reset(); auto t0 = std::chrono::steady_clock::now(); mm.greedy_run(ids[0], 64, g);
      printf("%.2f ", 64 / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    printf("\n"); return 0;
  }
  FILE *f = fopen(out.c_str(), "wb"); auto &m = s.model(); m.reset();
  for (int t : ids) { const float *l = m.step(t, 1); fwrite(l, 4, m.vocab(), f); }
  int g[40]; int k = m.greedy_run(ids.back(), 40, g); fwrite(g, 4, k, f); fclose(f);
  for (int i = 0; i < k; i++) printf("%d ", g[i]);
  printf("\n");
}
