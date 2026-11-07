// Logits of the last position for given token ids (comma separated), via prefill (batched) or token-by-token steps.
// build: g++ -O2 -march=native -std=c++17 -Iinclude bench/dump_logits.cpp build/libqpack.a -lpthread -o /tmp/dump_logits
// usage: dump_logits model.qpack ids [out.f32] [step|prefill] [threads]      (QPACK_DUMP=file also writes the last token's hidden state per layer)
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include "qpack/qpack.h"
using namespace qp;
int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: dump_logits model.qpack ids [out.f32] [step|prefill] [threads]\n"); return 2; }
  if (argc > 5) set_threads(atoi(argv[5]));
  std::vector<int> ids; { std::string s = argv[2]; size_t p = 0; while (p < s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); ids.push_back(atoi(s.substr(p, q - p).c_str())); p = q + 1; } }
  auto m = load_model(argv[1], getenv("QPACK_CTX") ? atoi(getenv("QPACK_CTX")) : 0);
  const bool pre = argc > 4 && std::string(argv[4]) == "prefill";
  auto t0 = std::chrono::steady_clock::now(); const float *lg = nullptr;
  if (pre) lg = m->prefill(ids.data(), (int)ids.size(), 1);
  else for (size_t i = 0; i < ids.size(); i++) lg = m->step(ids[i], i + 1 == ids.size() ? 1 : 0);
  double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::vector<int> ord(m->vocab()); std::iota(ord.begin(), ord.end(), 0);
  std::partial_sort(ord.begin(), ord.begin() + 5, ord.end(), [&](int a, int b) { return lg[a] > lg[b]; });
  printf("%s %zu tokens %.2f s; top5:", pre ? "prefill" : "step", ids.size(), s);
  for (int i = 0; i < 5; i++) printf(" %d(%.4f)", ord[i], lg[ord[i]]);
  printf("\n");
  if (argc > 3 && std::string(argv[3]) != "-") { FILE *f = fopen(argv[3], "wb"); fwrite(lg, 4, m->vocab(), f); fclose(f); }
  return 0;
}
