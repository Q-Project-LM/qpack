// qpack-cudacheck: logits of the CUDA backend vs the CPU engine on the same token sequences (first N tokens of each text file).
// usage: qpack-cudacheck -m model.qpack -n 64 file1.txt [file2.txt ...]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "qpack/engine.h"

int main(int argc, char **argv) {
  std::string path; int n = 64; std::vector<std::string> files;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "-m" && i + 1 < argc) path = argv[++i];
    else if (a == "-n" && i + 1 < argc) n = std::stoi(argv[++i]);
    else if (a == "-t" && i + 1 < argc) qp::set_threads(std::stoi(argv[++i]));
    else files.push_back(a);
  }
  try {
    setenv("QPACK_DEVICE", "cpu", 1); qp::Session cpu(qp::load_model(path, n + 8));
    setenv("QPACK_DEVICE", "cuda", 1); auto gpu = qp::load_model(path, n + 8);
    int cases = 0, top1 = 0, pos_ok = 0, pos_all = 0; double maxd = 0, sumd = 0; long cnt = 0;
    for (auto &f : files) {
      std::ifstream in(f, std::ios::binary); std::stringstream ss; ss << in.rdbuf();
      auto ids = cpu.tok().encode(ss.str()); if ((int)ids.size() > n) ids.resize(n);
      cpu.model().reset(); gpu->reset(); double cmax = 0; int last_ok = 0;
      for (size_t i = 0; i < ids.size(); i++) {
        const float *a = cpu.model().step(ids[i], 1); std::vector<float> la(a, a + cpu.model().vocab());
        const float *b = gpu->step(ids[i], 1);
        int ba = 0, bb = 0; for (int v = 1; v < (int)la.size(); v++) { if (la[v] > la[ba]) ba = v; if (b[v] > b[bb]) bb = v; }
        for (int v = 0; v < (int)la.size(); v++) { double d = std::fabs((double)la[v] - b[v]); cmax = std::max(cmax, d); sumd += d; cnt++; }
        pos_all++; pos_ok += ba == bb; last_ok = ba == bb;
      }
      cases++; top1 += last_ok; maxd = std::max(maxd, cmax);
      printf("%s: %zu tokens, final-position top-1 %s, max |dlogit| %.4f\n", f.c_str(), ids.size(), last_ok ? "equal" : "DIFFERENT", cmax);
    }
    if (!files.empty()) {   // greedy continuation: CPU engine vs device-side argmax chain
      std::ifstream in(files[0], std::ios::binary); std::stringstream ss; ss << in.rdbuf();
      auto ids = cpu.tok().encode(ss.str()); if (ids.size() > 32) ids.resize(32);
      cpu.model().reset(); gpu->reset(); for (size_t i = 0; i + 1 < ids.size(); i++) { cpu.model().step(ids[i], 0); gpu->step(ids[i], 0); }
      std::vector<int> a(40), b(40); cpu.model().greedy_run(ids.back(), 40, a.data()); gpu->greedy_run(ids.back(), 40, b.data());
      int same = 0; while (same < 40 && a[same] == b[same]) same++;
      printf("greedy_run: first %d of 40 tokens identical (CPU vs CUDA device argmax)\n", same);
    }
    printf("cases %d/%d top-1 equal at last position; all positions %d/%d top-1 equal; max |dlogit| %.4f, mean |dlogit| %.5f\n", top1, cases, pos_ok, pos_all, maxd, sumd / cnt);
  } catch (const std::exception &e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
}
