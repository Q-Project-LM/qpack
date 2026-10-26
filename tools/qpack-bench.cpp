// qpack-bench: prompt (prefill) and generation speed of a .qpack model for several thread counts (markdown table).
// usage: qpack-bench -m model.qpack [-p 128] [-n 128] [-t 1,4,8,16] [-r 2]
// The thread pool is process-wide, so each thread count runs in a child process of this binary.
#include <chrono>
#include <cstdio>
#include <sstream>
#include <string>

#include "qpack/qpack.h"

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char **argv) {
  std::string path, threads = "1,4,8,16"; int np = 128, ng = 128, reps = 2; bool row = false;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if ((a == "-m" || a == "--model") && i + 1 < argc) path = argv[++i];
    else if (a == "-p" && i + 1 < argc) np = std::stoi(argv[++i]);
    else if (a == "-n" && i + 1 < argc) ng = std::stoi(argv[++i]);
    else if (a == "-t" && i + 1 < argc) threads = argv[++i];
    else if (a == "-r" && i + 1 < argc) reps = std::stoi(argv[++i]);
    else if (a == "--row") row = true;
    else if (path.empty() && a[0] != '-') path = a;
    else { fprintf(stderr, "usage: qpack-bench -m model.qpack [-p 128] [-n 128] [-t 1,4,8,16] [-r 2]\n"); return 2; }
  }
  if (path.empty()) { fprintf(stderr, "usage: qpack-bench -m model.qpack [-p 128] [-n 128] [-t 1,4,8,16] [-r 2]\n"); return 2; }
  if (row) {   // one measurement: best of `reps`
    qp::set_threads(std::stoi(threads));
    auto m = qp::load_model(path, np + ng + 8);
    double pp = 0, tg = 0;
    for (int r = 0; r < reps; r++) {
      m->reset(); double t0 = now();
      for (int i = 0; i < np; i++) m->step(100 + (i * 37) % 5000, i + 1 == np ? 1 : 0);
      double t1 = now();
      for (int i = 0; i < ng; i++) m->step(100 + (i * 53) % 5000, 1);
      double t2 = now();
      pp = std::max(pp, np / (t1 - t0)); tg = std::max(tg, ng / (t2 - t1));
    }
    std::string name = path.substr(path.find_last_of('/') + 1);
    printf("| %s | %.1f | %s | %d | %.1f | %.1f |\n", name.c_str(), m->container().file_size() / 1048576.0, m->arch().c_str(), qp::pool().size(), pp, tg);
    return 0;
  }
  printf("| model | MiB | arch | threads | pp%d tok/s | tg%d tok/s |\n|---|---|---|---|---|---|\n", np, ng); fflush(stdout);
  std::stringstream ss(threads); std::string t;
  while (std::getline(ss, t, ',')) {
    std::string cmd = std::string("'") + argv[0] + "' --row -m '" + path + "' -p " + std::to_string(np) + " -n " + std::to_string(ng) + " -r " + std::to_string(reps) + " -t " + t;
    FILE *f = popen(cmd.c_str(), "r"); if (!f) return 1;
    char buf[512]; while (fgets(buf, sizeof buf, f)) fputs(buf, stdout);
    fflush(stdout); pclose(f);
  }
  return 0;
}
