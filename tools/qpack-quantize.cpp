// qpack-quantize: re-quantise a .qpack (f32 / f16 / q8 source) to q4 (4-bit groups of 32 + fp16 scale, ~4.5 bits/weight)
// or q8. Embedding and output head stay q8 (they hurt most at 4 bits), as do matrices whose width is not a multiple
// of 32. The q4 scale of every group is picked by minimum squared error over a grid of candidates plus a least-squares
// refit. The self-test is recomputed on the dequantised model, so it checks the q4 kernels, not the rounding.
// usage: qpack-quantize in.qpack out.qpack [--type q4|q8] [--keep-q8 name,name,...] [-t threads]
#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "qpack/qpack.h"

using qp::Json;

namespace {

uint16_t to_f16(float x) { return _cvtss_sh(x, 0); }
float from_f16(uint16_t h) { return _cvtsh_ss(h); }

// one group of 32 weights -> 16 bytes + fp16 scale, minimum squared error
void q4_group(const float *w, uint8_t *out, uint16_t *scale) {
  float amax = 0, m = 0;
  for (int i = 0; i < 32; i++) if (std::fabs(w[i]) > amax) { amax = std::fabs(w[i]); m = w[i]; }
  int q[32], bq[32];
  if (amax == 0) { std::fill(bq, bq + 32, 0); *scale = 0; }
  else {
    float best = INFINITY; uint16_t bs = 0;
    auto try_scale = [&](float s) {
      const uint16_t h = to_f16(s); const float sh = from_f16(h); if (sh == 0 || !std::isfinite(sh)) return;
      float err = 0, wq = 0, qq = 0;
      for (int i = 0; i < 32; i++) { q[i] = std::max(-8, std::min(7, (int)std::lrintf(w[i] / sh))); const float d = w[i] - q[i] * sh; err += d * d; wq += w[i] * q[i]; qq += (float)q[i] * q[i]; }
      if (err < best) { best = err; bs = h; std::copy(q, q + 32, bq); }
      if (qq > 0) {   // least-squares refit of the scale for these levels
        const uint16_t h2 = to_f16(wq / qq); const float s2 = from_f16(h2); if (s2 == 0 || !std::isfinite(s2)) return;
        float e2 = 0; int q2[32];
        for (int i = 0; i < 32; i++) { q2[i] = std::max(-8, std::min(7, (int)std::lrintf(w[i] / s2))); const float d = w[i] - q2[i] * s2; e2 += d * d; }
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
const char *kUsage = "usage: qpack-quantize in.qpack out.qpack [--type q4|q8] [--keep-q8 name,...] [-t threads]\n";

}  // namespace

int main(int argc, char **argv) {
  std::string in, out, type = "q4"; std::set<std::string> keep;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--type" && i + 1 < argc) type = argv[++i];
    else if (a == "--keep-q8" && i + 1 < argc) {
      std::string l = argv[++i]; size_t p = 0;
      while (p <= l.size()) { size_t c = l.find(',', p); if (c == std::string::npos) c = l.size(); if (c > p) keep.insert(l.substr(p, c - p)); p = c + 1; }
    } else if ((a == "-t" || a == "--threads") && i + 1 < argc) qp::set_threads(std::stoi(argv[++i]));
    else if (in.empty() && a[0] != '-') in = a;
    else if (out.empty() && a[0] != '-') out = a;
    else { fputs(kUsage, stderr); return 2; }
  }
  if (in.empty() || out.empty() || (type != "q4" && type != "q8")) { fputs(kUsage, stderr); return 2; }
  for (const char *n : {"embed", "lm_head", "embed.weight", "lm_head.weight"}) keep.insert(n);
  try {
    auto t0 = std::chrono::steady_clock::now();
    qp::Container src(in);
    std::vector<std::pair<std::string, std::string>> blobs;
    Json man = src.manifest(), tensors = Json::array();
    std::set<std::string> tensor_blobs;
    for (auto &e : man["tensors"].arr)
      for (const char *f : {"blob", "scale", "alpha"}) if (e.has(f)) tensor_blobs.insert(e[f].as_str());
    int64_t n_q4 = 0, n_q8 = 0, n_other = 0; int k = 0;
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
      const bool to_q4 = type == "q4" && cols % 32 == 0 && !keep.count(name);
      if (!to_q4 && t->kind == qp::Kind::Q8) {   // already q8: copied, no second rounding
        ent.set("kind", "q8"); ent.set("blob", copy_blob(e["blob"].as_str(), bid)); ent.set("scale", copy_blob(e["scale"].as_str(), bid));
        n_q8 += t->numel(); tensors.push(ent); continue;
      }
      if (to_q4) {
        const int G = cols / 32; std::string q((size_t)rows * cols / 2, '\0'), sc((size_t)rows * G * 2, '\0');
        qp::pool().parallel_for(rows, [&](int a, int b2) {
          std::vector<float> row(cols);
          for (int r = a; r < b2; r++) {
            qp::row_f32(*t, r, row.data());
            for (int g = 0; g < G; g++) q4_group(row.data() + 32 * g, (uint8_t *)&q[(size_t)r * cols / 2 + 16 * g], (uint16_t *)&sc[((size_t)r * G + g) * 2]);
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
    man.set("format", "QPACK2"); man.set("quant", type); man.set("tensors", tensors);
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
