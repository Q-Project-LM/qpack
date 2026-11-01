// qpack-quantize: re-quantise a .qpack (f32 / f16 / q8 source) to q4 (4-bit groups of 32 + fp16 scale, ~4.5 bits/weight)
// or q8. Embedding and output head stay q8 (they hurt most at 4 bits), as do matrices whose width is not a multiple
// of 32. The q4 scale of every group is picked by minimum squared error over a grid of candidates plus a least-squares
// refit. The self-test is recomputed on the dequantised model, so it checks the q4 kernels, not the rounding.
// usage: qpack-quantize in.qpack out.qpack [--type q4|q4m|q8] [--keep-q8 name,name,...] [-t threads]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "qpack/engine.h"
#include "qpack/qpack.h"

using qp::Json;

namespace {

uint16_t to_f16(float x) { return qp::f32_to_f16(x); }
float from_f16(uint16_t h) { return qp::f16_to_f32(h); }

// one group of 32 weights -> 16 bytes + fp16 scale, minimum squared error weighted by imp (importance of each input
// column: mean squared activation from calibration text, or 1)
void q4_group(const float *w, const float *imp, uint8_t *out, uint16_t *scale) {
  float amax = 0, m = 0;
  for (int i = 0; i < 32; i++) if (std::fabs(w[i]) > amax) { amax = std::fabs(w[i]); m = w[i]; }
  int q[32], bq[32];
  if (amax == 0) { std::fill(bq, bq + 32, 0); *scale = 0; }
  else {
    float best = INFINITY; uint16_t bs = 0;
    auto try_scale = [&](float s) {
      const uint16_t h = to_f16(s); const float sh = from_f16(h); if (sh == 0 || !std::isfinite(sh)) return;
      float err = 0, wq = 0, qq = 0;
      for (int i = 0; i < 32; i++) { q[i] = std::max(-8, std::min(7, (int)std::lrintf(w[i] / sh))); const float d = w[i] - q[i] * sh; err += imp[i] * d * d; wq += imp[i] * w[i] * q[i]; qq += imp[i] * q[i] * q[i]; }
      if (err < best) { best = err; bs = h; std::copy(q, q + 32, bq); }
      if (qq > 0) {   // least-squares refit of the scale for these levels
        const uint16_t h2 = to_f16(wq / qq); const float s2 = from_f16(h2); if (s2 == 0 || !std::isfinite(s2)) return;
        float e2 = 0; int q2[32];
        for (int i = 0; i < 32; i++) { q2[i] = std::max(-8, std::min(7, (int)std::lrintf(w[i] / s2))); const float d = w[i] - q2[i] * s2; e2 += imp[i] * d * d; }
        if (e2 < best) { best = e2; bs = h2; std::copy(q2, q2 + 32, bq); }
      }
    };
    for (int k = 0; k <= 12; k++) try_scale(m / (7.f + 0.125f * k));      // extreme value on the +7 side
    for (int k = 0; k <= 12; k++) try_scale(-m / (7.5f + 0.125f * k));   // extreme value on the -8 side
    *scale = bs;
  }
  for (int j = 0; j < 16; j++) out[j] = (uint8_t)((bq[j] + 8) | ((bq[j + 16] + 8) << 4));
}

std::unique_ptr<qp::Model> make(std::shared_ptr<qp::Container> c) {
  if (c->config()["arch"].as_str() == "llama") return qp::make_llama(c, 0);
  if (c->config()["model_type"].as_str() == "nq4") return qp::make_nq(c, 0);
  throw std::runtime_error("unsupported model for the self-test");
}
// q4m: like llama.cpp's Q4_K_M, the value projections and the FFN down projections of the first and last eighth of the
// layers and of every third layer in between stay q8 (these matrices carry most of the 4-bit error)
bool more_bits(const std::string &name, int n_layers, bool last_half) {
  if (name.compare(0, 7, "layers.") != 0 || n_layers <= 0) return false;
  const int i = atoi(name.c_str() + 7); const std::string tail = name.substr(name.find('.', 7) + 1);
  if (tail == "wv" || tail == "mixer.v.weight") return true;
  if (tail != "w_down" && tail != "ffn.down.weight") return false;
  if (last_half) return i >= n_layers / 2;
  return i < n_layers / 8 || i >= 7 * n_layers / 8 || (i - n_layers / 8) % 3 == 2;
}
const char *kUsage = "usage: qpack-quantize in.qpack out.qpack [--type q4|q4m|q8] [--keep-q8 name,...] [--q4-head] [--q4m-last-half] [--calib text.txt [--gptq [--act-order]]] [--calib-tokens 4096] [-t threads]\n";

// statistics of every matrix input from running the source model over calibration text: mean squared activation per
// column (importance), and with gptq the second moment H = sum x x^T (shared by matrices reading the same input)
struct Calib {
  std::map<std::string, std::vector<float>> imp;
  std::map<std::string, std::shared_ptr<std::vector<float>>> H;   // cols x cols, full
  std::map<std::string, long> count;
};
Calib calibrate(const std::string &model_path, const std::string &text_path, int max_tokens, bool hessian) {
  qp::Session s(qp::load_model(model_path, 512));
  std::ifstream f(text_path, std::ios::binary); std::stringstream ss; ss << f.rdbuf();
  std::vector<int> all = s.tok().encode(ss.str()), ids;
  const int win = 511;
  {   // windows spread evenly over the whole text (it may mix languages / sources one after another)
    const size_t have = all.size() / win, want = std::max<size_t>(1, max_tokens / win);
    for (size_t k = 0; k < std::min(have, want); k++) { size_t w = have <= want ? k : k * have / want; ids.insert(ids.end(), all.begin() + w * win, all.begin() + (w + 1) * win); }
    if (have == 0) ids = all;
  }
  Calib cal; std::map<std::string, std::vector<double>> sum;
  const float *last_x = nullptr; int last_b = -1; std::string last_name;
  qp::g_act_hook = [&](const qp::Tensor &W, const float *X, int B, int ldx, int cols) {
    auto &v = sum[W.name]; if (v.empty()) v.assign(cols, 0.0);
    for (int b = 0; b < B; b++) for (int c = 0; c < cols; c++) { const double x = X[(size_t)b * ldx + c]; v[c] += x * x; }
    cal.count[W.name] += B;
    if (hessian) {
      // q/k/v (and gate/up) are called back to back on the same input: they share one H
      if (X == last_x && B == last_b && cal.H.count(last_name) && (int)std::sqrt((double)cal.H[last_name]->size()) == cols) { if (!cal.H.count(W.name)) cal.H[W.name] = cal.H[last_name]; }
      else {
        auto &h = cal.H[W.name]; if (!h) h = std::make_shared<std::vector<float>>((size_t)cols * cols, 0.f);
        float *H = h->data();
        qp::pool().parallel_for(cols, [&](int i0, int i1) {   // upper triangle, rank-B update
          for (int i = i0; i < i1; i++) for (int b = 0; b < B; b++) { const float *x = X + (size_t)b * ldx; if (x[i] != 0) qp::axpy_f32(x[i], x + i, H + (size_t)i * cols + i, cols - i); }
        }, 8);
      }
      last_x = X; last_b = B; last_name = W.name;
    }
  };
  const int bos = s.chat().name() == "nq" ? s.tok().token_id("<bos>") : -1;
  for (size_t p = 0; p < ids.size(); p += win) {
    std::vector<int> w(ids.begin() + p, ids.begin() + std::min(ids.size(), p + win)); if (bos >= 0) w.insert(w.begin(), bos);
    s.model().reset(); s.model().prefill(w.data(), (int)w.size(), 0); last_x = nullptr;
  }
  qp::g_act_hook = nullptr;
  for (auto &kv : sum) {
    std::vector<float> v(kv.second.size()); double mean = 0;
    for (size_t c = 0; c < v.size(); c++) { v[c] = (float)(kv.second[c] / cal.count[kv.first]); mean += v[c]; }
    mean /= v.size(); for (auto &x : v) x += (float)(0.01 * mean) + 1e-12f;   // keep every column in play
    cal.imp[kv.first] = std::move(v);
  }
  std::set<const void *> seen;
  for (auto &kv : cal.H) if (seen.insert(kv.second.get()).second) {   // mirror the upper triangle
    const int n = (int)std::sqrt((double)kv.second->size()); float *H = kv.second->data();
    for (int i = 0; i < n; i++) for (int j = 0; j < i; j++) H[(size_t)i * n + j] = H[(size_t)j * n + i];
  }
  fprintf(stderr, "calibration: %zu tokens, %zu matrices observed, %zu Hessians\n", ids.size(), cal.imp.size(), seen.size());
  return cal;
}

// GPTQ (Frantar et al. 2022) for q4 groups: columns are quantised left to right and the rounding error of each one is
// pushed onto the not yet quantised columns through the inverse Hessian, U = upper Cholesky factor of H^-1. A group's
// scale is chosen when the quantiser reaches its first column, from the already corrected weights.
// act_order: columns are visited by decreasing H_ii (most sensitive first); a group's scale is still fixed when its first
// column comes up, from the corrected weights of the whole (original) group, so the file layout does not change.
void gptq_q4(const qp::Tensor &t, const std::vector<float> &Hf, uint8_t *q, uint16_t *sc, bool act_order) {
  const int rows = (int)t.rows(), n = (int)t.cols(), G = n / 32;
  std::vector<int> perm(n), pos(n); for (int i = 0; i < n; i++) perm[i] = i;
  if (act_order) std::stable_sort(perm.begin(), perm.end(), [&](int a, int b) { return Hf[(size_t)a * n + a] > Hf[(size_t)b * n + b]; });
  for (int p = 0; p < n; p++) pos[perm[p]] = p;
  std::vector<double> H((size_t)n * n);   // permuted
  for (int a = 0; a < n; a++) for (int b = 0; b < n; b++) H[(size_t)a * n + b] = Hf[(size_t)perm[a] * n + perm[b]];
  std::vector<char> dead(n, 0); double mean = 0;
  for (int i = 0; i < n; i++) { if (H[(size_t)i * n + i] <= 0) { dead[i] = 1; H[(size_t)i * n + i] = 1; } mean += H[(size_t)i * n + i]; }
  const double damp = 0.01 * mean / n; for (int i = 0; i < n; i++) H[(size_t)i * n + i] += damp;
  // H = R R^T with R upper ("reverse" Cholesky, last column first); then U = R^-1 is the upper factor of H^-1 = U^T U
  std::vector<double> R((size_t)n * n, 0.0);
  for (int j = n - 1; j >= 0; j--) {
    const double *rj = R.data() + (size_t)j * n; double d = H[(size_t)j * n + j];
    for (int k = j + 1; k < n; k++) d -= rj[k] * rj[k];
    const double rjj = std::sqrt(std::max(d, 1e-12)); R[(size_t)j * n + j] = rjj;
    qp::pool().parallel_for(j, [&](int i0, int i1) {
      for (int i = i0; i < i1; i++) { const double *ri = R.data() + (size_t)i * n; double v = H[(size_t)i * n + j]; for (int k = j + 1; k < n; k++) v -= ri[k] * rj[k]; R[(size_t)i * n + j] = v / rjj; }
    }, 32);
  }
  std::vector<double> Lt((size_t)n * n, 0.0);   // L = R^T, rows contiguous
  for (int i = 0; i < n; i++) for (int k = i; k < n; k++) Lt[(size_t)k * n + i] = R[(size_t)i * n + k];
  R.clear(); R.shrink_to_fit();
  std::vector<float> U((size_t)n * n, 0.f);     // row j of U = column j of L^-1 (forward substitution, columns independent)
  qp::pool().parallel_for(n, [&](int j0, int j1) {
    std::vector<double> x(n);
    for (int j = j0; j < j1; j++) {
      for (int i = j; i < n; i++) { const double *li = Lt.data() + (size_t)i * n; double v = i == j ? 1.0 : 0.0; for (int k = j; k < i; k++) v -= li[k] * x[k]; x[i] = v / li[i]; }
      for (int i = j; i < n; i++) U[(size_t)j * n + i] = (float)x[i];
    }
  }, 4);
  Lt.clear(); Lt.shrink_to_fit();
  std::vector<float> imp(n); for (int i = 0; i < n; i++) imp[i] = (float)Hf[(size_t)i * n + i] + (float)damp;   // original order
  const int RB = 16;   // rows per block: each row of U is reused by all of them
  qp::pool().parallel_for((rows + RB - 1) / RB, [&](int b0, int b1) {
    std::vector<float> w((size_t)RB * n), row(n), sg((size_t)RB * G, 0.f); std::vector<char> have(G);
    for (int blk = b0; blk < b1; blk++) {
      const int r0 = blk * RB, nr = std::min(RB, rows - r0);
      for (int r = 0; r < nr; r++) { qp::row_f32(t, r0 + r, row.data()); for (int p = 0; p < n; p++) w[(size_t)r * n + p] = dead[perm[p]] ? 0.f : row[perm[p]]; }
      std::fill(have.begin(), have.end(), 0);
      for (int p = 0; p < n; p++) {
        const int i = perm[p], g = i / 32, j = i % 32;
        if (!have[g]) {   // the group's scale from its corrected (not yet quantised) weights
          have[g] = 1;
          for (int r = 0; r < nr; r++) {
            float gw[32]; for (int c = 0; c < 32; c++) gw[c] = w[(size_t)r * n + pos[32 * g + c]];
            uint8_t tmp[16]; uint16_t h; q4_group(gw, imp.data() + 32 * g, tmp, &h);
            sc[(size_t)(r0 + r) * G + g] = h; sg[(size_t)r * G + g] = qp::f16_to_f32(h);
          }
        }
        const float *u = U.data() + (size_t)p * n; const float upp = u[p];
        for (int r = 0; r < nr; r++) {
          float *wr = w.data() + (size_t)r * n; const float s = sg[(size_t)r * G + g];
          const int lv = s == 0 ? 0 : std::max(-8, std::min(7, (int)std::lrintf(wr[p] / s)));
          uint8_t &byte = q[(size_t)(r0 + r) * (n / 2) + 16 * g + (j & 15)];
          if (j < 16) byte = (uint8_t)((byte & 0xF0) | (lv + 8)); else byte = (uint8_t)((byte & 0x0F) | ((lv + 8) << 4));
          const float err = (wr[p] - lv * s) / upp;
          if (p + 1 < n) qp::axpy_f32(-err, u + p + 1, wr + p + 1, n - p - 1);
        }
      }
    }
  }, 1);
}

}  // namespace

int main(int argc, char **argv) {
  std::string in, out, type = "q4", calib; int calib_tokens = 4096; bool gptq = false, act_order = false, q4_head = false, q4m_last_half = false; std::set<std::string> keep;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--type" && i + 1 < argc) type = argv[++i];
    else if (a == "--keep-q8" && i + 1 < argc) {
      std::string l = argv[++i]; size_t p = 0;
      while (p <= l.size()) { size_t c = l.find(',', p); if (c == std::string::npos) c = l.size(); if (c > p) keep.insert(l.substr(p, c - p)); p = c + 1; }
    } else if ((a == "-t" || a == "--threads") && i + 1 < argc) qp::set_threads(std::stoi(argv[++i]));
    else if (a == "--calib" && i + 1 < argc) calib = argv[++i];
    else if (a == "--calib-tokens" && i + 1 < argc) calib_tokens = std::stoi(argv[++i]);
    else if (a == "--gptq") gptq = true;
    else if (a == "--q4-head") q4_head = true;
    else if (a == "--q4m-last-half") q4m_last_half = true;
    else if (a == "--act-order") act_order = true;
    else if (in.empty() && a[0] != '-') in = a;
    else if (out.empty() && a[0] != '-') out = a;
    else { fputs(kUsage, stderr); return 2; }
  }
  if (in.empty() || out.empty() || (type != "q4" && type != "q4m" && type != "q8")) { fputs(kUsage, stderr); return 2; }
  if (q4m_last_half && type != "q4m") { fprintf(stderr, "--q4m-last-half requires --type q4m\n"); return 2; }
  if (!q4_head) for (const char *n : {"embed", "lm_head", "embed.weight", "lm_head.weight"}) keep.insert(n);
  try {
    auto t0 = std::chrono::steady_clock::now();
    if (gptq && calib.empty()) { fprintf(stderr, "--gptq needs --calib text.txt\n"); return 2; }
    Calib cal; if (!calib.empty()) cal = calibrate(in, calib, calib_tokens, gptq);
    auto &imp = cal.imp;
    qp::Container src(in);
    std::vector<std::pair<std::string, std::string>> blobs;
    Json man = src.manifest(), tensors = Json::array();
    std::set<std::string> tensor_blobs;
    for (auto &e : man["tensors"].arr)
      for (const char *f : {"blob", "scale", "alpha"}) if (e.has(f)) tensor_blobs.insert(e[f].as_str());
    int64_t n_q4 = 0, n_q8 = 0, n_other = 0; int k = 0; const int n_layers = (int)src.config()["num_hidden_layers"].as_int(0);
    auto copy_blob = [&](const std::string &from, const std::string &bid) {   // keeps the extension (.f16, .i64, .scale, ...)
      size_t len; const uint8_t *p = src.blob(from, &len); const std::string to = bid + from.substr(from.find('.'));
      blobs.push_back({to, std::string((const char *)p, len)}); return to;
    };
    for (auto &e : man["tensors"].arr) {
      const std::string name = e["name"].as_str(), kind = e["kind"].as_str();
      const qp::Tensor *t = src.get(name);
      Json ent = Json::object(); ent.set("name", name); ent.set("shape", e["shape"]); ent.set("dtype", e["dtype"]);
      char b[16]; snprintf(b, sizeof b, "w%04d", k++); const std::string bid = b;
      const bool matrix = t && t->shape.size() == 2 && t->numel() >= 4096 &&
                          (t->kind == qp::Kind::F32 || t->kind == qp::Kind::F16 || t->kind == qp::Kind::Q8 || t->kind == qp::Kind::Q4);
      if (!matrix) {   // copied as stored (f16 tensors that are not matrices were widened to f32 at load: written as f32)
        if (t && t->kind == qp::Kind::F32 && kind == "f16") {
          blobs.push_back({bid + ".f32", std::string((const char *)t->data, 4 * t->numel())}); ent.set("kind", "f32"); ent.set("blob", bid + ".f32");
        } else {
          ent.set("kind", kind); ent.set("blob", copy_blob(e["blob"].as_str(), bid));
          for (const char *x : {"scale", "alpha"}) if (e.has(x)) ent.set(x, copy_blob(e[x].as_str(), bid));
          if (e.has("group")) ent.set("group", e["group"]);
        }
        n_other += t ? t->numel() : 0; tensors.push(ent); continue;
      }
      const int rows = (int)t->rows(), cols = (int)t->cols();
      const bool to_q4 = (type == "q4" || (type == "q4m" && !more_bits(name, n_layers, q4m_last_half))) && cols % 32 == 0 && !keep.count(name);
      if (!to_q4 && t->kind == qp::Kind::Q8) {   // already q8: copied, no second rounding
        ent.set("kind", "q8"); ent.set("blob", copy_blob(e["blob"].as_str(), bid)); ent.set("scale", copy_blob(e["scale"].as_str(), bid));
        n_q8 += t->numel(); tensors.push(ent); continue;
      }
      if (to_q4) {
        const int G = cols / 32; std::string q((size_t)rows * cols / 2, '\0'), sc((size_t)rows * G * 2, '\0');
        auto hit = cal.H.find(name);
        if (gptq && hit != cal.H.end()) {
          gptq_q4(*t, *hit->second, (uint8_t *)&q[0], (uint16_t *)&sc[0], act_order);
          blobs.push_back({bid + ".q4", std::move(q)}); blobs.push_back({bid + ".scale", std::move(sc)});
          ent.set("kind", "q4"); ent.set("blob", bid + ".q4"); ent.set("scale", bid + ".scale"); ent.set("group", 32); n_q4 += t->numel(); tensors.push(ent); continue;
        }
        auto it = imp.find(name); std::vector<float> ones(cols, 1.f);
        const float *im = it != imp.end() && (int)it->second.size() == cols ? it->second.data() : ones.data();
        qp::pool().parallel_for(rows, [&](int a, int b2) {
          std::vector<float> row(cols);
          for (int r = a; r < b2; r++) {
            qp::row_f32(*t, r, row.data());
            for (int g = 0; g < G; g++) q4_group(row.data() + 32 * g, im + 32 * g, (uint8_t *)&q[(size_t)r * cols / 2 + 16 * g], (uint16_t *)&sc[((size_t)r * G + g) * 2]);
          }
        }, 4);
        blobs.push_back({bid + ".q4", std::move(q)}); blobs.push_back({bid + ".scale", std::move(sc)});
        ent.set("kind", "q4"); ent.set("blob", bid + ".q4"); ent.set("scale", bid + ".scale"); ent.set("group", 32); n_q4 += t->numel();
      } else {   // q8 from float (same rule as container.quantize_q8)
        std::string q((size_t)rows * cols, '\0'), sc((size_t)rows * 4, '\0');
        qp::pool().parallel_for(rows, [&](int a, int b2) {
          std::vector<float> row(cols);
          for (int r = a; r < b2; r++) {
            qp::row_f32(*t, r, row.data()); float mx = 0; for (float v : row) mx = std::max(mx, std::fabs(v));
            float s = mx / 127.f; if (s == 0) s = 1.f;
            for (int c = 0; c < cols; c++) q[(size_t)r * cols + c] = (char)(int8_t)std::max(-127, std::min(127, (int)std::nearbyint(row[c] / s)));
            memcpy(&sc[(size_t)r * 4], &s, 4);
          }
        }, 4);
        blobs.push_back({bid + ".q8", std::move(q)}); blobs.push_back({bid + ".scale", std::move(sc)});
        ent.set("kind", "q8"); ent.set("blob", bid + ".q8"); ent.set("scale", bid + ".scale"); n_q8 += t->numel();
      }
      tensors.push(ent);
    }
    man.set("format", "QPACK2"); man.set("quant", type); if (q4m_last_half) man.set("q4m_last_half", true); if (!calib.empty()) { man.set("calibration_tokens", calib_tokens); man.set("gptq", gptq); } man.set("tensors", tensors);
    std::vector<std::pair<std::string, std::string>> all;
    all.push_back({"manifest.json", man.dump()});
    for (auto &n : src.blob_names()) if (!tensor_blobs.count(n) && n != "manifest.json" && n != "selftest.json") all.push_back({n, src.text(n)});
    const bool has_st = src.has("selftest.json");
    if (has_st) all.push_back({"selftest.json", src.text("selftest.json")});
    for (auto &bl : blobs) all.push_back(std::move(bl));
    qp::write_container(out, all);

    if (has_st) {   // reference logits from the dequantised copy of the new file, then the self-test is rewritten
      auto ref = std::make_shared<qp::Container>(out);
      for (auto &kv : ref->tensors()) {
        qp::Tensor &t = kv.second;
        if (t.shape.size() != 2 || (t.kind != qp::Kind::Q4 && t.kind != qp::Kind::Q8 && t.kind != qp::Kind::F16)) continue;
        std::vector<uint8_t> buf(4 * (size_t)t.numel());
        qp::pool().parallel_for((int)t.rows(), [&](int a, int b2) { for (int r = a; r < b2; r++) qp::row_f32(t, r, (float *)buf.data() + (size_t)r * t.cols()); }, 64);
        t.owned.swap(buf); t.data = t.owned.data(); t.kind = qp::Kind::F32; t.scale = nullptr; t.scale16 = nullptr;
      }
      auto m = make(ref); Json st = ref->json("selftest.json"), cases = Json::array();
      for (auto &c : st["cases"].arr) {
        m->reset(); std::vector<int> ids; for (auto &x : c["ids"].arr) ids.push_back((int)x.as_int());
        const float *lg = nullptr; for (size_t i = 0; i < ids.size(); i++) lg = m->step(ids[i], i + 1 == ids.size() ? 1 : 0);
        std::vector<int> idx(m->vocab()); for (int v = 0; v < m->vocab(); v++) idx[v] = v;
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int b2) { return lg[a] > lg[b2]; });
        Json cc = Json::object(), top = Json::array(), tl = Json::array(); cc.set("ids", c["ids"]);
        for (int j = 0; j < 5; j++) { top.push(idx[j]); tl.push((double)lg[idx[j]]); }
        cc.set("top5", top); cc.set("top5_logits", tl); cases.push(cc);
      }
      m.reset(); ref.reset();
      st.set("cases", cases); st.set("reference", "dequantised " + type + " weights (qpack-quantize)");
      for (auto &bl : all) if (bl.first == "selftest.json") bl.second = st.dump();
      qp::write_container(out, all);
    }
    qp::Container chk(out);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("%s -> %s: %.1f MiB -> %.1f MiB, q4 %.1fM / q8 %.1fM / other %.1fM params, %.1f s\n", in.c_str(), out.c_str(), src.file_size() / 1048576.0,
           chk.file_size() / 1048576.0, n_q4 / 1e6, n_q8 / 1e6, n_other / 1e6, s);
  } catch (const std::exception &e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
  return 0;
}
