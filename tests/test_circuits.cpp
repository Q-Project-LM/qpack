// Compares qp::circuit_eval with the Python evaluate() (reference made by n-q/web_tests/make_circ_ref.py).
// usage: test_circuits circ_ref.json
#include <cstdio>
#include <fstream>
#include <sstream>

#include "qpack/circuits.h"

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s circ_ref.json\n", argv[0]); return 2; }
  std::ifstream f(argv[1]); std::stringstream ss; ss << f.rdbuf();
  qp::Json ref = qp::Json::parse(ss.str());
  int ok = 0, n = 0;
  for (auto &c : ref.arr) {
    std::string got; bool has = qp::circuit_eval(c["e"].as_str(), &got); n++;
    bool same = c["r"].is_null() ? !has : (has && got == c["r"].as_str());
    if (same) ok++;
    else if (n - ok <= 15) fprintf(stderr, "MISMATCH %s -> got %s, ref %s\n", c["e"].as_str().c_str(), has ? got.c_str() : "None", c["r"].is_null() ? "None" : c["r"].as_str().c_str());
  }
  printf("%d/%d match\n", ok, n);
  return ok == n ? 0 : 1;
}
