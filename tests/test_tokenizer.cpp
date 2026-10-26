// Compares qp::Tokenizer with reference ids from the Python `tokenizers` library (tests/make_tok_ref.py).
// usage: test_tokenizer <tokenizer.json | model.qpack> <ref.json>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "qpack/qpack.h"
#include "qpack/tokenizer.h"

static std::string slurp(const char *p) { std::ifstream f(p, std::ios::binary); std::stringstream s; s << f.rdbuf(); return s.str(); }

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s <tokenizer.json|model.qpack> <ref.json>\n", argv[0]); return 2; }
  std::string src = argv[1];
  qp::Json tj = src.size() > 6 && src.substr(src.size() - 6) == ".qpack" ? qp::Container(src).json("tokenizer.json") : qp::Json::parse(slurp(argv[1]));
  qp::Tokenizer tok(tj);
  qp::Json ref = qp::Json::parse(slurp(argv[2]));
  int ok = 0, n = 0, rt = 0; size_t ntok = 0;
  auto t0 = std::chrono::steady_clock::now();
  for (auto &c : ref.arr) {
    const std::string &text = c["text"].as_str(); auto ids = tok.encode(text); n++; ntok += ids.size();
    bool same = ids.size() == c["ids"].size();
    for (size_t i = 0; same && i < ids.size(); i++) same = ids[i] == c["ids"][i].as_int();
    if (same) ok++;
    else if (n - ok <= 5) {
      fprintf(stderr, "MISMATCH %s\n  got:", text.substr(0, 120).c_str()); for (int x : ids) fprintf(stderr, " %d", x);
      fprintf(stderr, "\n  ref:"); for (auto &x : c["ids"].arr) fprintf(stderr, " %lld", x.as_int()); fprintf(stderr, "\n");
    }
    if (tok.decode(ids, false) == text) rt++;
  }
  double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  printf("%d/%d match, %d/%d round-trip, %zu tokens in %.3f s\n", ok, n, rt, n, ntok, s);
  return ok == n ? 0 : 1;
}
