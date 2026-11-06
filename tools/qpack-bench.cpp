// qpack-bench: prompt (prefill) and generation speed of a .qpack model for several thread counts (markdown table).
// usage: qpack-bench -m model.qpack [-p 128] [-n 128] [-t 1,4,8,16] [-r 2] [--spec prompt.txt]
// The thread pool is process-wide, so each thread count runs in a child process of this binary.
#include <chrono>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include <fstream>
#include <sstream>

#include "qpack/engine.h"
#include "qpack/qpack.h"

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char **argv) {
  std::string path, threads = "1,4,8,16", spec; int np = 128, ng = 128, reps = 2; bool row = false, greedy = false;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if ((a == "-m" || a == "--model") && i + 1 < argc) path = argv[++i];
    else if (a == "-p" && i + 1 < argc) np = std::stoi(argv[++i]);
    else if (a == "-n" && i + 1 < argc) ng = std::stoi(argv[++i]);
    else if (a == "-t" && i + 1 < argc) threads = argv[++i];
    else if (a == "--device" && i + 1 < argc) setenv("QPACK_DEVICE", argv[++i], 1);
    else if (a == "-r" && i + 1 < argc) reps = std::stoi(argv[++i]);
    else if (a == "--row") row = true;
    else if (a == "--greedy") greedy = true;
    else if (a == "--spec" && i + 1 < argc) spec = argv[++i];
    else if (path.empty() && a[0] != '-') path = a;
    else { fprintf(stderr, "usage: qpack-bench -m model.qpack [-p 128] [-n 128] [-t 1,4,8,16] [-r 2] [--spec prompt.txt]\n"); return 2; }
  }
  if (path.empty()) { fprintf(stderr, "usage: qpack-bench -m model.qpack [-p 128] [-n 128] [-t 1,4,8,16] [-r 2] [--spec prompt.txt]\n"); return 2; }
  if (row) {   // one measurement: best of `reps`
    qp::set_threads(std::stoi(threads));
    auto m = qp::load_model(path, np + ng + 8);
    double pp = 0, tg = 0;
    for (int r = 0; r < reps; r++) {
      m->reset(); double t0 = now();
      std::vector<int> toks(np); for (int i = 0; i < np; i++) toks[i] = 100 + (i * 37) % 5000;
      m->prefill(toks.data(), np, 1);
      double t1 = now();
      if (greedy) { std::vector<int> o(ng); m->greedy_run(100, ng, o.data()); }
      else for (int i = 0; i < ng; i++) m->step(100 + (i * 53) % 5000, 1);
      double t2 = now();
      pp = std::max(pp, np / (t1 - t0)); tg = std::max(tg, ng / (t2 - t1));
    }
    std::string name = path.substr(path.find_last_of('/') + 1);
    printf("| %s | %.1f | %s | %d | %.1f | %.1f |", name.c_str(), m->container().file_size() / 1048576.0, m->arch().c_str(), qp::pool().size(), pp, tg);
    if (!spec.empty()) {   // greedy chat reply to the prompt in the file: plain vs prompt-lookup speculative decoding
      std::ifstream f(spec); std::stringstream ss; ss << f.rdbuf(); double tps[2] = {0, 0}; std::string txt[2];
      for (int mode = 0; mode < 2; mode++) {
        qp::Session s(qp::load_model(path, 4096)); if (mode) s.set_lookup(15);
        qp::SamplerParams sp; sp.temperature = 0; sp.repeat_penalty = 1.f; qp::Sampler sm(sp);
        auto r = s.chat_turn({{"user", ss.str()}}, ng, sm); tps[mode] = r.ids.size() / r.decode_s; txt[mode] = r.text;
      }
      printf(" %.1f | %.1f%s |", tps[0], tps[1], txt[0] == txt[1] ? "" : " (MISMATCH)");
    }
    printf("\n");
    return 0;
  }
  if (spec.empty()) printf("| model | MiB | arch | threads | pp%d tok/s | tg%d tok/s |\n|---|---|---|---|---|---|\n", np, ng);
  else printf("| model | MiB | arch | threads | pp%d tok/s | tg%d tok/s | chat tok/s | chat + lookup tok/s |\n|---|---|---|---|---|---|---|---|\n", np, ng);
  fflush(stdout);
  std::stringstream ss(threads); std::string t;
  while (std::getline(ss, t, ',')) {
    std::string cmd = std::string("'") + argv[0] + "' --row -m '" + path + "' -p " + std::to_string(np) + " -n " + std::to_string(ng) + " -r " + std::to_string(reps) + " -t " + t + (spec.empty() ? "" : " --spec '" + spec + "'");
    FILE *f = popen(cmd.c_str(), "r"); if (!f) return 1;
    char buf[512]; while (fgets(buf, sizeof buf, f)) fputs(buf, stdout);
    fflush(stdout); pclose(f);
  }
  return 0;
}
