// qpack-perplexity: perplexity of a .qpack model on a UTF-8 text file, in windows of --ctx tokens (non-overlapping).
// usage: qpack-perplexity -m model.qpack -f text.txt [-c 512] [--chunks N] [-t threads]
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "qpack/engine.h"

int main(int argc, char **argv) {
  std::string path, file; int ctx = 512, chunks = 0;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if ((a == "-m" || a == "--model") && i + 1 < argc) path = argv[++i];
    else if (a == "-f" && i + 1 < argc) file = argv[++i];
    else if ((a == "-c" || a == "--ctx") && i + 1 < argc) ctx = std::stoi(argv[++i]);
    else if (a == "--chunks" && i + 1 < argc) chunks = std::stoi(argv[++i]);
    else if ((a == "-t" || a == "--threads") && i + 1 < argc) qp::set_threads(std::stoi(argv[++i]));
    else if (a == "--device" && i + 1 < argc) setenv("QPACK_DEVICE", argv[++i], 1);
    else { fprintf(stderr, "usage: qpack-perplexity -m model.qpack -f text.txt [-c 512] [--chunks N] [-t threads]\n"); return 2; }
  }
  if (path.empty() || file.empty()) { fprintf(stderr, "usage: qpack-perplexity -m model.qpack -f text.txt [-c 512] [--chunks N] [-t threads]\n"); return 2; }
  try {
    qp::Session s(qp::load_model(path, ctx));
    std::ifstream in(file, std::ios::binary); std::stringstream ss; ss << in.rdbuf();
    std::vector<int> ids = s.tok().encode(ss.str());
    const int bos = s.chat().name() == "nq" ? s.tok().token_id("<bos>") : -1;
    const int win = ctx - (bos >= 0 ? 1 : 0);
    int n_chunks = (int)ids.size() / win; if (chunks > 0 && chunks < n_chunks) n_chunks = chunks;
    if (n_chunks == 0) { fprintf(stderr, "text too short: %zu tokens < %d\n", ids.size(), win); return 1; }
    double tot = 0; long cnt = 0;
    for (int c = 0; c < n_chunks; c++) {
      std::vector<int> w(ids.begin() + (size_t)c * win, ids.begin() + (size_t)(c + 1) * win);
      if (bos >= 0) w.insert(w.begin(), bos);
      int n = 0; double nll = s.nll(w, &n); tot += nll * n; cnt += n;
      fprintf(stderr, "\r[%d/%d] ppl %.4f", c + 1, n_chunks, std::exp(tot / cnt));
    }
    fprintf(stderr, "\n");
    printf("%s: %ld tokens, mean nll %.5f, perplexity %.4f\n", path.c_str(), cnt, tot / cnt, std::exp(tot / cnt));
  } catch (const std::exception &e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
  return 0;
}
