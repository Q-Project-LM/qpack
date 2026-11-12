// qpack-cudadump: bit-exactness probe for the CUDA backend: FNV-1a hash of the raw logits of every position of the first N tokens of each text file
// (step path) plus a 40-token greedy_run, so two builds can be compared without a reference engine. usage: qpack-cudadump -m model.qpack -n 96 file...
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "qpack/engine.h"

int main(int argc, char **argv) {
  std::string path; int n = 96; std::vector<std::string> files;
  for (int i = 1; i < argc; i++) { std::string a = argv[i]; if (a == "-m" && i + 1 < argc) path = argv[++i]; else if (a == "-n" && i + 1 < argc) n = std::stoi(argv[++i]); else files.push_back(a); }
  try {
    setenv("QPACK_DEVICE", "cuda", 1); qp::Session s(qp::load_model(path, n + 64));
    uint64_t all = 1469598103934665603ull;
    for (auto &f : files) {
      std::ifstream in(f, std::ios::binary); std::stringstream ss; ss << in.rdbuf();
      auto ids = s.tok().encode(ss.str()); if ((int)ids.size() > n) ids.resize(n);
      s.model().reset(); uint64_t h = 1469598103934665603ull;
      for (size_t i = 0; i < ids.size(); i++) {
        const float *l = s.model().step(ids[i], 1); const uint8_t *p = (const uint8_t *)l;
        for (size_t b = 0; b < (size_t)s.model().vocab() * 4; b++) { h ^= p[b]; h *= 1099511628211ull; }
      }
      int g[40]; s.model().greedy_run(ids.back(), 40, g); for (int i = 0; i < 40; i++) { h ^= (uint32_t)g[i]; h *= 1099511628211ull; }
      printf("%s: %zu tokens hash %016llx\n", f.c_str(), ids.size(), (unsigned long long)h); all ^= h; all *= 1099511628211ull;
    }
    printf("ALL %016llx\n", (unsigned long long)all);
  } catch (const std::exception &e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
}
