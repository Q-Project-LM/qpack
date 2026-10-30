#include <algorithm>
#include <cmath>

#include "qpack/engine.h"

namespace qp {

Sampler::Sampler(const SamplerParams &p) : p_(p), rng_(p.seed ? p.seed : std::random_device{}()) {}

// the M largest logits (index, value), unordered: SIMD scan against a running threshold, so only a few hundred of the
// ~150k entries are ever compared one by one
static void top_m(const float *x, int V, int M, std::vector<std::pair<int, float>> &out) {
  out.clear();
  if (M >= V / 4) { out.resize(V); for (int v = 0; v < V; v++) out[v] = {v, x[v]}; return; }
  auto shrink = [&]() {
    std::nth_element(out.begin(), out.begin() + (M - 1), out.end(), [](auto &a, auto &b) { return a.second > b.second || (a.second == b.second && a.first < b.first); });
    out.resize(M); float t = INFINITY; for (auto &c : out) t = std::min(t, c.second); return t;
  };
  const int init = std::min(V, 4 * M + 8);
  for (int v = 0; v < init; v++) out.push_back({v, x[v]});
  float thr = shrink();
  for (int v = init; v < V; v += 4096) {
    scan_gt(x, v, std::min(V, v + 4096), thr, out);
    if ((int)out.size() > 4 * M) thr = shrink();
  }
  if ((int)out.size() > M) shrink();
}

void Sampler::dist(const float *logits, int V, const std::vector<int> &history, std::vector<std::pair<int, float>> &out, size_t hist_len) {
  // repetition penalty (llama.cpp style: divide positive logits, multiply negative ones) on the recent window. A penalty
  // only lowers a logit, so the top K after penalties lie within the top K + (number of penalised tokens) before them.
  const size_t hn = std::min(hist_len, history.size());
  pen_.clear();
  if (p_.repeat_penalty != 1.f)
    for (size_t i = hn > (size_t)p_.repeat_last_n ? hn - p_.repeat_last_n : 0; i < hn; i++) {
      const int t = history[i]; if (t < 0 || t >= V) continue;
      if (std::find(pen_.begin(), pen_.end(), t) == pen_.end()) pen_.push_back(t);
    }
  const bool greedy = p_.temperature <= 0;
  const int K = greedy ? 1 : p_.top_k > 0 ? std::min(p_.top_k, V) : V;
  top_m(logits, V, std::min(V, K + (int)pen_.size()), out);
  for (auto &c : out) if (std::find(pen_.begin(), pen_.end(), c.first) != pen_.end()) c.second = c.second > 0 ? c.second / p_.repeat_penalty : c.second * p_.repeat_penalty;
  std::sort(out.begin(), out.end(), [](auto &a, auto &b) { return a.second > b.second || (a.second == b.second && a.first < b.first); });
  if ((int)out.size() > K) out.resize(K);
  if (greedy) { out[0].second = 1.f; return; }
  const float mx = out[0].second; double sum = 0;
  for (auto &c : out) { c.second = std::exp((c.second - mx) / p_.temperature); sum += c.second; }
  size_t keep = out.size(); double cum = 0;
  for (size_t i = 0; i < out.size(); i++) {
    if (p_.min_p > 0 && out[i].second < p_.min_p * out[0].second) { keep = std::max<size_t>(i, 1); break; }
    cum += out[i].second / sum; if (cum >= p_.top_p) { keep = i + 1; break; }
  }
  out.resize(keep);
  double tot = 0; for (auto &c : out) tot += c.second;
  for (auto &c : out) c.second = (float)(c.second / tot);
}

int Sampler::draw(const std::vector<std::pair<int, float>> &d) {
  if (d.size() == 1) return d[0].first;
  double r = std::uniform_real_distribution<double>(0, 1)(rng_);
  for (auto &c : d) { r -= c.second; if (r <= 0) return c.first; }
  return d.back().first;
}

bool Sampler::accept(const std::vector<std::pair<int, float>> &d, int token, int *repl) {
  float p = 0; for (auto &c : d) if (c.first == token) { p = c.second; break; }
  if (p >= 1.f) return true;
  if (p > 0 && std::uniform_real_distribution<double>(0, 1)(rng_) < p) return true;
  std::vector<std::pair<int, float>> rest; rest.reserve(d.size());
  for (auto &c : d) if (c.first != token) rest.push_back({c.first, c.second / (1.f - p)});
  *repl = rest.empty() ? token : draw(rest);
  return false;
}

}  // namespace qp
