// Exact circuits for N-Q (<CALC>expr<EQ>result<ECALC>): a port of nq/model/circuits.py with Python semantics where
// they matter — exact integers (128-bit; overflow fails the call), floor division / modulo, round-half-even,
// str(float) formatting. Checked against the Python evaluate() on 40k corpus spans (tests/test_circuits.cpp).
#pragma once
#include <deque>
#include <string>
#include <vector>

#include "qpack/tokenizer.h"

namespace qp {

bool circuit_eval(const std::string &expr, std::string *result);   // false = Python evaluate() returns None

// decode-time hook: feed the whole token sequence before each sampling step; returns the token that must come next
// (forced result / <ECALC>) or -1 to sample freely
class CircuitExecutor {
 public:
  CircuitExecutor(const Tokenizer &tok);
  int next(const std::vector<int> &ids);
  void clear() { queue_.clear(); }
  int calls = 0, failed = 0;
  int calc_id, eq_id, ecalc_id;
 private:
  const Tokenizer &tok_; std::deque<int> queue_;
};

}  // namespace qp
