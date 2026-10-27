// Speculative decoding check and speed: the same prompt decoded plainly, with prompt lookup, and with a draft model.
// Greedy outputs must be identical. usage: spec_bench model.qpack prompt.txt [n_new] [lookup_k] [draft.qpack draft_k]
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include "qpack/engine.h"
int main(int argc, char **argv) {
  std::ifstream f(argv[2]); std::stringstream ss; ss << f.rdbuf();
  const int n = argc > 3 ? atoi(argv[3]) : 256, k = argc > 4 ? atoi(argv[4]) : 8;
  qp::SamplerParams sp; sp.temperature = 0; sp.repeat_penalty = 1.0f;
  std::string base_text; double base_tps = 0;
  for (int mode = 0; mode < (argc > 6 ? 3 : 2); mode++) {
    qp::Session s(qp::load_model(argv[1], 4096));
    if (mode == 1) s.set_lookup(k);
    if (mode == 2) s.set_draft(qp::load_model(argv[5], 4096), atoi(argv[6]));
    std::vector<qp::Message> msgs{{"user", ss.str()}};
    qp::Sampler sm(sp);
    auto r = s.chat_turn(msgs, n, sm);
    const double tps = r.ids.size() / r.decode_s;
    if (mode == 0) { base_text = r.text; base_tps = tps; }
    printf("%-8s %4zu tok  %7.1f tok/s  x%.2f  drafted %d accepted %d (%.0f%%) passes %d  identical %s\n", mode == 0 ? "plain" : mode == 1 ? "lookup" : "draft",
           r.ids.size(), tps, tps / base_tps, r.drafted, r.accepted, r.drafted ? 100.0 * r.accepted / r.drafted : 0.0, r.verify_passes, r.text == base_text ? "yes" : "NO");
    if (mode == 0 && getenv("SHOW")) printf("%s\n", r.text.c_str());
  }
}
