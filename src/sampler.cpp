#include <algorithm>
#include <cmath>
#include <unordered_set>

#include "qpack/engine.h"

namespace qp {

Sampler::Sampler(const SamplerParams &p) : p_(p), rng_(p.seed ? p.seed : std::random_device{}()) {}

int Sampler::sample(const float *logits, int V, const std::vector<int> &history) {
  // repetition penalty (llama.cpp style: divide positive logits, multiply negative ones) over the recent window
  std::unordered_set<int> seen;
  if (p_.repeat_penalty != 1.f)
    for (size_t i = history.size() > (size_t)p_.repeat_last_n ? history.size() - p_.repeat_last_n : 0; i < history.size(); i++) seen.insert(history[i]);
  auto adj = [&](int v) { float l = logits[v]; if (seen.count(v)) l = l > 0 ? l / p_.repeat_penalty : l * p_.repeat_penalty; return l; };
  if (p_.temperature <= 0) { int best = 0; float bl = adj(0); for (int v = 1; v < V; v++) { float l = adj(v); if (l > bl) { bl = l; best = v; } } return best; }
  const int K = p_.top_k > 0 ? std::min(p_.top_k, V) : V;
  cand_.resize(V); for (int v = 0; v < V; v++) cand_[v] = {adj(v), v};
  std::partial_sort(cand_.begin(), cand_.begin() + K, cand_.end(), [](const std::pair<float, int> &a, const std::pair<float, int> &b) { return a.first > b.first; });
  cand_.resize(K);
  const float mx = cand_[0].first; double sum = 0;
  for (auto &c : cand_) { c.first = std::exp((c.first - mx) / p_.temperature); sum += c.first; }
  size_t keep = cand_.size(); double cum = 0;
  for (size_t i = 0; i < cand_.size(); i++) {
    if (p_.min_p > 0 && cand_[i].first < p_.min_p * cand_[0].first) { keep = std::max<size_t>(i, 1); break; }
    cum += cand_[i].first / sum; if (cum >= p_.top_p) { keep = i + 1; break; }
  }
  double tot = 0; for (size_t i = 0; i < keep; i++) tot += cand_[i].first;
  double r = std::uniform_real_distribution<double>(0, tot)(rng_);
  for (size_t i = 0; i < keep; i++) { r -= cand_[i].first; if (r <= 0) return cand_[i].second; }
  return cand_[keep - 1].second;
}

}  // namespace qp
