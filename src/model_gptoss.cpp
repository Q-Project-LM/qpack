// gpt-oss (openai/gpt-oss-20b): RMSNorm, GQA attention with q/k/v/o biases, YaRN RoPE (concentration folded into cos/sin), learned
// per-head attention sinks, alternating sliding-window (128) / full layers, top-4 MoE with softmax-of-the-4 router weights and the
// interleaved clamped swiglu, expert biases, untied head. Tensor names are the ones written by python/qpack/convert_gptoss.py;
// the expert matrices are [E, R, C] tensors of any kind (typically mxfp4) viewed per expert.
//   decode: the 4 selected experts' gate_up views share one quantised activation (matvec_n);
//   prefill: tokens are processed in batches, grouped by expert so each expert's weights are streamed once per batch.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "qpack/qpack.h"

namespace qp {
namespace {

// view of expert e of a [E * R, C] tensor as an independent [R, C] tensor
Tensor expert_view(const Tensor &W, int e, int E) {
  Tensor v; const int64_t R = W.shape[0] / E, C = W.shape[1];
  v.name = W.name + "#" + std::to_string(e); v.shape = {R, C}; v.kind = W.kind; v.group = W.group;
  switch (W.kind) {
    case Kind::Q8: v.data = (const int8_t *)W.data + e * R * C; v.scale = W.scale + e * R; break;
    case Kind::Q4: v.data = (const uint8_t *)W.data + e * R * C / 2; v.scale16 = W.scale16 + e * R * (C / 32); break;
    case Kind::Q1: { const int64_t G = (C + 127) / 128; v.data = (const uint8_t *)W.data + e * R * G * 16; v.scale16 = W.scale16 + e * R * G; break; }
    case Kind::Mxfp4: v.data = (const uint8_t *)W.data + e * R * C / 2; v.scale8 = W.scale8 + e * R * (C / 32); break;
    case Kind::F32: v.data = W.f32() + e * R * C; break;
    case Kind::F16: v.data = (const uint16_t *)W.data + e * R * C; break;
    default: throw std::runtime_error("gptoss: unsupported expert tensor kind in " + W.name);
  }
  return v;
}

struct Layer {
  const Tensor *an, *fn, *wq, *wk, *wv, *wo, *bq, *bk, *bv, *bo, *sinks, *rw, *rb, *bgu, *bd;
  std::vector<std::vector<Tensor>> gu, dn;   // per-expert views (1, or 2 residual planes whose products are added)
  bool sliding;
};

class GptOss : public Model {
 public:
  GptOss(std::shared_ptr<Container> c, int max_ctx) {
    c_ = c; cfg_ = c->config();
    d_ = (int)cfg_["hidden_size"].as_int(); L_ = (int)cfg_["num_hidden_layers"].as_int(); nh_ = (int)cfg_["num_attention_heads"].as_int();
    nkv_ = (int)cfg_["num_key_value_heads"].as_int(); hd_ = (int)cfg_["head_dim"].as_int(64); ff_ = (int)cfg_["intermediate_size"].as_int();
    V_ = (int)cfg_["vocab_size"].as_int(); E_ = (int)cfg_["num_local_experts"].as_int(); K_ = (int)cfg_["num_experts_per_tok"].as_int(4);
    if (const char *p = getenv("QPACK_LAYERS")) L_ = std::min(L_, atoi(p));   // debug: truncated model
    eps_ = (float)cfg_["rms_norm_eps"].as_num(1e-5); window_ = (int)cfg_["sliding_window"].as_int(128);
    limit_ = (float)cfg_["swiglu_limit"].as_num(7.0); alpha_ = (float)cfg_["swiglu_alpha"].as_num(1.702);
    if (hd_ != 64 || K_ > 8) throw std::runtime_error("gptoss: head_dim 64 and <= 8 experts per token expected");
    const int lim = (int)cfg_["max_position_embeddings"].as_int(1 << 20);
    ctx_ = max_ctx > 0 ? max_ctx : std::min(4096, lim);
    embed_ = &c->at("embed"); head_ = &c->at("lm_head"); fnorm_ = &c->at("final_norm");
    const Json &lt = cfg_["layer_types"];
    for (int l = 0; l < L_; l++) {
      auto g = [&](const char *n) { return &c->at("layers." + std::to_string(l) + "." + n); };
      Layer ly;
      ly.an = g("attn_norm"); ly.fn = g("ffn_norm"); ly.wq = g("wq"); ly.wk = g("wk"); ly.wv = g("wv"); ly.wo = g("wo");
      ly.bq = g("bq"); ly.bk = g("bk"); ly.bv = g("bv"); ly.bo = g("bo"); ly.sinks = g("sinks"); ly.rw = g("router_w"); ly.rb = g("router_b");
      ly.bgu = g("b_gate_up"); ly.bd = g("b_down");
      // experts: all in w_gate_up / w_down (any kind), or -- config "moe_exact": {"<layer>": [ids]} -- the listed ones in the extra tensors
      // w_gate_up_x / w_down_x (typically mxfp4, in listed order) and the others, in ascending order, in w_gate_up / w_down (typically q1),
      // optionally with a second residual q1 plane in w_gate_up_r2 / w_down_r2 whose product is added
      const std::string ls = std::to_string(l); const Json &mx = cfg_["moe_exact"];
      std::vector<int> exact; if (mx.has(ls)) for (auto &x : mx[ls].arr) exact.push_back((int)x.as_int());
      const Tensor *gu = g("w_gate_up"), *dn = g("w_down"), *gur = c->get("layers." + ls + ".w_gate_up_r2"), *dnr = c->get("layers." + ls + ".w_down_r2");
      const Tensor *gux = c->get("layers." + ls + ".w_gate_up_x"), *dnx = c->get("layers." + ls + ".w_down_x");
      const int nq = E_ - (int)exact.size(); int iq = 0;
      for (int e = 0; e < E_; e++) {
        auto it = std::find(exact.begin(), exact.end(), e);
        if (it != exact.end()) {
          if (!gux || !dnx) throw std::runtime_error("gptoss: exact experts listed without w_gate_up_x / w_down_x in layer " + ls);
          const int j = (int)(it - exact.begin());
          ly.gu.push_back({expert_view(*gux, j, (int)exact.size())}); ly.dn.push_back({expert_view(*dnx, j, (int)exact.size())});
        } else {
          ly.gu.push_back({expert_view(*gu, iq, nq)}); ly.dn.push_back({expert_view(*dn, iq, nq)});
          if (gur) ly.gu.back().push_back(expert_view(*gur, iq, nq));
          if (dnr) ly.dn.back().push_back(expert_view(*dnr, iq, nq));
          iq++;
        }
      }
      ly.sliding = lt.type == Json::Array && l < (int)lt.arr.size() ? lt.arr[l].as_str() == "sliding_attention" : (l % 2 == 0);
      layers_.push_back(std::move(ly));
    }
    kvd_ = nkv_ * hd_;
    kc_.assign((size_t)L_ * ctx_ * kvd_, 0); vc_.assign((size_t)L_ * ctx_ * kvd_, 0);   // fp16 K/V cache
    x_.resize(d_); xn_.resize(d_); q_.resize(nh_ * hd_); k_.resize(kvd_); v_.resize(kvd_); att_.resize(nh_ * hd_); o_.resize(d_); logits_.resize(V_);
    rl_.resize(E_); gub_.resize((size_t)K_ * 2 * ff_); gub2_.resize((size_t)K_ * 2 * ff_); eo2_.resize(d_); hb_.resize((size_t)K_ * ff_); eo_.resize(d_);
    // YaRN (see the reference torch model): inv_freq blends interpolation and extrapolation by a ramp over the dimension index
    const double theta = cfg_["rope_theta"].as_num(150000), factor = cfg_["rope_scaling"]["factor"].as_num(1);
    const double orig = cfg_["rope_scaling"]["original_max_position_embeddings"].as_num(4096), beta_fast = cfg_["rope_scaling"]["beta_fast"].as_num(32), beta_slow = cfg_["rope_scaling"]["beta_slow"].as_num(1);
    const int half = hd_ / 2; inv_.resize(half); conc_ = 1.0;
    for (int i = 0; i < half; i++) inv_[i] = 1.0 / std::pow(theta, (2.0 * i) / hd_);
    if (factor > 1.0) {
      conc_ = 0.1 * std::log(factor) + 1.0;
      double low = half * std::log(orig / (beta_fast * 2 * M_PI)) / std::log(theta), high = half * std::log(orig / (beta_slow * 2 * M_PI)) / std::log(theta);
      if (cfg_["rope_scaling"]["truncate"].as_bool(false)) { low = std::floor(low); high = std::ceil(high); }
      for (int i = 0; i < half; i++) {
        const double ex = inv_[i], in = inv_[i] / factor, ramp = std::min(1.0, std::max(0.0, (i - low) / (high - low))), mask = 1 - ramp;
        inv_[i] = in * (1 - mask) + ex * mask;
      }
    }
    if (const char *p = getenv("QPACK_DUMP_EXPERT_IN")) { cap_dir_ = p; cap_n_ = getenv("QPACK_DUMP_EXPERT_N") ? atoi(getenv("QPACK_DUMP_EXPERT_N")) : 2048; }
    if (const char *p = getenv("QPACK_DUMP")) dump_ = fopen(p, "wb");
  }
  ~GptOss() override { if (dump_) fclose(dump_); for (FILE *f : cap_f_) if (f) fclose(f); }
  int vocab() const override { return V_; }
  int hidden() const override { return d_; }
  int max_ctx() const override { return ctx_; }
  int pos() const override { return pos_; }
  std::string arch() const override { return "gptoss"; }
  void reset() override { pos_ = 0; }
  std::vector<uint8_t> checkpoint() const override { std::vector<uint8_t> c(4); memcpy(c.data(), &pos_, 4); return c; }
  void rollback(const std::vector<uint8_t> &c) override { memcpy(&pos_, c.data(), 4); }
  bool rewind(int pos) override { if (pos < 0 || pos > pos_) return false; pos_ = pos; return true; }
  std::vector<uint8_t> snapshot() const override {
    std::vector<uint8_t> s(4 + 2ull * L_ * pos_ * kvd_ * 2); memcpy(s.data(), &pos_, 4); uint8_t *p = s.data() + 4;
    for (int l = 0; l < L_; l++) for (const auto *buf : {&kc_, &vc_}) { size_t n = (size_t)pos_ * kvd_ * 2; memcpy(p, buf->data() + (size_t)l * ctx_ * kvd_, n); p += n; }
    return s;
  }
  void restore(const std::vector<uint8_t> &s) override {
    memcpy(&pos_, s.data(), 4); const uint8_t *p = s.data() + 4;
    for (int l = 0; l < L_; l++) for (auto *buf : {&kc_, &vc_}) { size_t n = (size_t)pos_ * kvd_ * 2; memcpy(buf->data() + (size_t)l * ctx_ * kvd_, p, n); p += n; }
  }

  // ---------------------------------------------------------------- shared pieces
  void rope_tables(int pos, float *cs, float *sn) const {
    for (int j = 0; j < hd_ / 2; j++) { const double a = pos * inv_[j]; cs[j] = (float)(std::cos(a) * conc_); sn[j] = (float)(std::sin(a) * conc_); }
  }
  void rope(float *vec, const float *cs, const float *sn) const {
    const int half = hd_ / 2;
    for (int j = 0; j < half; j++) { const float a = vec[j], b = vec[j + half]; vec[j] = a * cs[j] - b * sn[j]; vec[j + half] = b * cs[j] + a * sn[j]; }
  }
  // router: top-K of the logits (descending), softmax over the selected ones
  void route(const float *logits, int *idx, float *w) const {
    int sel[8]; float v[8];
    for (int k = 0; k < K_; k++) {
      int b = -1; for (int e = 0; e < E_; e++) { bool used = false; for (int j = 0; j < k; j++) used |= sel[j] == e; if (!used && (b < 0 || logits[e] > logits[b])) b = e; }
      sel[k] = b; v[k] = logits[b];
    }
    float mx = v[0], sum = 0; for (int k = 1; k < K_; k++) mx = std::max(mx, v[k]);
    for (int k = 0; k < K_; k++) { w[k] = std::exp(v[k] - mx); sum += w[k]; }
    for (int k = 0; k < K_; k++) { w[k] /= sum; idx[k] = sel[k]; }
  }
  // h[j] = glu(y[2j]) * (clamp(y[2j+1]) + 1) for n pairs
  void swiglu(const float *y, float *h, int n) const {
    for (int j = 0; j < n; j++) {
      const float g = std::min(y[2 * j], limit_), u = std::max(-limit_, std::min(limit_, y[2 * j + 1]));
      h[j] = g / (1.f + std::exp(-alpha_ * g)) * (u + 1.f);
    }
  }

  // attention of one token over cache positions [t_lo, T) plus the sink: work items = (kv group, chunk of 64 positions); the 8 query
  // heads of a group share each converted K / V row; chunks are merged in a fixed order (same result for any thread count)
  void attend(const uint16_t *kl, const uint16_t *vl, const float *q, int t_lo, int T, const float *sinks, float *out) {
    const int C = 64, nc = (T - t_lo + C - 1) / C, rep = nh_ / nkv_, hd = hd_; const float scl = 1.f / std::sqrt((float)hd);
    if (part_.size() < (size_t)nh_ * nc * (hd + 2)) part_.resize((size_t)nh_ * nc * (hd + 2));
    auto item = [&](int w0, int w1) {
      std::vector<float> sc((size_t)rep * C), kv(hd);
      for (int w = w0; w < w1; w++) {
        const int g = w / nc, c = w % nc, a0 = t_lo + c * C, a1 = std::min(T, a0 + C), n = a1 - a0;
        if (!attn_scores(kl, kvd_, q, g, rep, hd, a0, a1, C, scl, sc.data())) for (int t = a0; t < a1; t++) {
          const uint16_t *kk = kl + (size_t)t * kvd_ + g * hd; f16_to_f32_v(kk, kv.data(), hd);
          for (int j = 0; j < rep; j++) sc[(size_t)j * C + (t - a0)] = dot_f32(q + (g * rep + j) * hd, kv.data(), hd) * scl;
        }
        for (int j = 0; j < rep; j++) {
          float *sj = sc.data() + (size_t)j * C; float mx = -1e30f; for (int t = 0; t < n; t++) mx = std::max(mx, sj[t]);
          for (int t = 0; t < n; t++) sj[t] -= mx;
          exp_v(sj, sj, n); float sum = 0; for (int t = 0; t < n; t++) sum += sj[t];
          float *pp = part_.data() + ((size_t)(g * rep + j) * nc + c) * (hd + 2); pp[0] = mx; pp[1] = sum; std::fill(pp + 2, pp + 2 + hd, 0.f);
        }
        float *pg = part_.data() + ((size_t)(g * rep) * nc + c) * (hd + 2) + 2; const size_t pstride = (size_t)nc * (hd + 2);
        if (!attn_values(vl, kvd_, g, rep, hd, a0, a1, C, sc.data(), pg, pstride)) for (int t = a0; t < a1; t++) {
          const uint16_t *vv = vl + (size_t)t * kvd_ + g * hd; f16_to_f32_v(vv, kv.data(), hd);
          for (int j = 0; j < rep; j++) axpy_f32(sc[(size_t)j * C + (t - a0)], kv.data(), pg + j * pstride, hd);
        }
      }
    };
    const int items = nkv_ * nc;
    if (items > 1 && (int64_t)(T - t_lo) * nh_ * hd > attn_par_min_) pool().parallel_for(items, item, 1); else item(0, items);
    for (int h = 0; h < nh_; h++) {   // merge the chunks and the sink logit: softmax over all positions + the sink
      const float *ph = part_.data() + (size_t)h * nc * (hd + 2); float M = sinks[h]; for (int c = 0; c < nc; c++) M = std::max(M, ph[(size_t)c * (hd + 2)]);
      float *o = out + h * hd; std::fill(o, o + hd, 0.f); float L = std::exp(sinks[h] - M);
      for (int c = 0; c < nc; c++) { const float *pc = ph + (size_t)c * (hd + 2); const float f = std::exp(pc[0] - M); L += pc[1] * f; axpy_f32(f, pc + 2, o, hd); }
      const float il = 1.f / L; for (int i = 0; i < hd; i++) o[i] *= il;
    }
  }

  // calibration capture (QPACK_DUMP_EXPERT_IN=dir [QPACK_DUMP_EXPERT_N=2048]): per layer l, files L{l}.in.f16 (the normalised MoE input = gate_up input,
  // fp16 [token, d]), L{l}.sel.u8 (4 selected expert ids per token, router order) and L{l}.w.f16 (their 4 softmax weights); first N positions only
  void capture(int l, int pos, const float *xn, const int *idx, const float *rw) {
    if (cap_dir_.empty() || pos >= cap_n_) return;
    if (cap_f_.empty()) cap_f_.assign(3 * L_, nullptr);
    if (!cap_f_[3 * l]) {
      const char *nm[3] = {"in.f16", "sel.u8", "w.f16"};
      for (int k = 0; k < 3; k++) { std::string f = cap_dir_ + "/L" + std::to_string(l) + "." + nm[k]; cap_f_[3 * l + k] = fopen(f.c_str(), "wb"); if (!cap_f_[3 * l + k]) throw std::runtime_error("cannot write " + f); }
    }
    std::vector<uint16_t> h(d_); to_f16(xn, h.data(), d_); fwrite(h.data(), 2, d_, cap_f_[3 * l]);
    uint8_t sel[8]; uint16_t w16[8]; for (int k = 0; k < K_; k++) sel[k] = (uint8_t)idx[k];
    to_f16(rw, w16, K_ < 4 ? 4 : K_); fwrite(sel, 1, K_, cap_f_[3 * l + 1]); fwrite(w16, 2, K_, cap_f_[3 * l + 2]);
  }
  void dump(const float *x) { if (dump_) { fwrite(x, 4, d_, dump_); fflush(dump_); } }

  // ---------------------------------------------------------------- one token
  const float *step(int tok, int want) override {
    if (pos_ >= ctx_) throw std::runtime_error("context window full (" + std::to_string(ctx_) + " tokens)");
    if (tok < 0 || tok >= V_) throw std::runtime_error("token id out of range");
    row_f32(*embed_, tok, x_.data()); dump(x_.data());
    const int T = pos_ + 1, qd = nh_ * hd_; float cs[32], sn[32]; rope_tables(pos_, cs, sn);
    for (int l = 0; l < L_; l++) {
      const Layer &ly = layers_[l];
      rmsnorm(x_.data(), ly.an->f32(), d_, eps_, xn_.data());
      { const Tensor *w[3] = {ly.wq, ly.wk, ly.wv}; float *o[3] = {q_.data(), k_.data(), v_.data()}; matvec_n(w, 3, xn_.data(), o, d_); }
      for (int i = 0; i < qd; i++) q_[i] += ly.bq->f32()[i];
      for (int i = 0; i < kvd_; i++) { k_[i] += ly.bk->f32()[i]; v_[i] += ly.bv->f32()[i]; }
      for (int h = 0; h < nh_; h++) rope(q_.data() + h * hd_, cs, sn);
      for (int g = 0; g < nkv_; g++) rope(k_.data() + g * hd_, cs, sn);
      uint16_t *kl = kc_.data() + (size_t)l * ctx_ * kvd_, *vl = vc_.data() + (size_t)l * ctx_ * kvd_;
      to_f16(k_.data(), kl + (size_t)pos_ * kvd_, kvd_); to_f16(v_.data(), vl + (size_t)pos_ * kvd_, kvd_);
      attend(kl, vl, q_.data(), ly.sliding ? std::max(0, T - window_) : 0, T, ly.sinks->f32(), att_.data());
      matvec(*ly.wo, att_.data(), o_.data(), d_, qd);
      for (int i = 0; i < d_; i++) x_[i] += o_[i] + ly.bo->f32()[i];
      // MoE
      rmsnorm(x_.data(), ly.fn->f32(), d_, eps_, xn_.data());
      matvec(*ly.rw, xn_.data(), rl_.data(), E_, d_); for (int e = 0; e < E_; e++) rl_[e] += ly.rb->f32()[e];
      int idx[8]; float rw[8]; route(rl_.data(), idx, rw); capture(l, pos_, xn_.data(), idx, rw);
      {
        const Tensor *w[8]; float *o[8]; int nw = 0, slot[8], np[8];   // one output slot per (expert, plane); plane 2 is added to plane 1
        for (int k = 0; k < K_; k++) {
          np[k] = (int)ly.gu[idx[k]].size(); slot[k] = nw;
          for (int p = 0; p < np[k]; p++) { w[nw] = &ly.gu[idx[k]][p]; o[nw] = p == 0 ? gub_.data() + (size_t)k * 2 * ff_ : gub2_.data() + (size_t)k * 2 * ff_; nw++; }
        }
        matvec_n(w, nw, xn_.data(), o, d_);
        for (int k = 0; k < K_; k++) if (np[k] > 1) { float *y = gub_.data() + (size_t)k * 2 * ff_; const float *y2 = gub2_.data() + (size_t)k * 2 * ff_; for (int i = 0; i < 2 * ff_; i++) y[i] += y2[i]; }
      }
      std::fill(eo_.begin(), eo_.end(), 0.f);
      for (int k = 0; k < K_; k++) {
        float *y = gub_.data() + (size_t)k * 2 * ff_; const float *b = ly.bgu->f32() + (size_t)idx[k] * 2 * ff_;
        for (int i = 0; i < 2 * ff_; i++) y[i] += b[i];
        float *h = hb_.data() + (size_t)k * ff_; swiglu(y, h, ff_);
        matvec(ly.dn[idx[k]][0], h, o_.data(), d_, ff_);
        if (ly.dn[idx[k]].size() > 1) { matvec(ly.dn[idx[k]][1], h, eo2_.data(), d_, ff_); for (int i = 0; i < d_; i++) o_[i] += eo2_[i]; }
        const float *bd = ly.bd->f32() + (size_t)idx[k] * d_;
        for (int i = 0; i < d_; i++) eo_[i] += rw[k] * (o_[i] + bd[i]);
      }
      for (int i = 0; i < d_; i++) x_[i] += eo_[i];
      dump(x_.data());
      if (debug_) fprintf(stderr, "pos %d layer %d: %.5f %.5f %.5f %.5f experts %d %d %d %d\n", pos_, l, x_[0], x_[1], x_[2], x_[3], idx[0], idx[1], idx[2], idx[3]);
    }
    pos_++;
    if (!want) return nullptr;
    rmsnorm(x_.data(), fnorm_->f32(), d_, eps_, xn_.data());
    if (want == 2) return xn_.data();
    matvec(*head_, xn_.data(), logits_.data(), V_, d_);
    return logits_.data();
  }

  void prefill_all(const int *toks, int n, float *logits) override {
    if (n == 1) { const float *l = step(toks[0], 1); memcpy(logits, l, 4ull * V_); return; }
    all_logits_ = logits; try { prefill(toks, n, 0); } catch (...) { all_logits_ = nullptr; throw; } all_logits_ = nullptr;
  }
  // ---------------------------------------------------------------- prompt batches
  const float *prefill(const int *toks, int n, int want) override {
    if (n == 1) return step(toks[0], want);
    if (pos_ + n > ctx_) throw std::runtime_error("context window full (" + std::to_string(ctx_) + " tokens)");
    const int PB = 128, rep = nh_ / nkv_, qd = nh_ * hd_; const float scl = 1.f / std::sqrt((float)hd_);
    bX_.resize((size_t)PB * d_); bXN_.resize((size_t)PB * d_); bQ_.resize((size_t)PB * qd); bK_.resize((size_t)PB * kvd_); bV_.resize((size_t)PB * kvd_);
    bA_.resize((size_t)PB * qd); bO_.resize((size_t)PB * d_); bR_.resize((size_t)PB * E_);
    for (int s0 = 0; s0 < n; s0 += PB) {
      const int B = std::min(PB, n - s0), p0 = pos_; const bool last = s0 + B == n;
      for (int i = 0; i < B; i++) {
        if (toks[s0 + i] < 0 || toks[s0 + i] >= V_) throw std::runtime_error("token id out of range");
        row_f32(*embed_, toks[s0 + i], bX_.data() + (size_t)i * d_);
      }
      if (last) dump(bX_.data() + (size_t)(B - 1) * d_);
      for (int l = 0; l < L_; l++) {
        const Layer &ly = layers_[l];
        pool().parallel_for(B, [&](int a, int b) { for (int i = a; i < b; i++) rmsnorm(bX_.data() + (size_t)i * d_, ly.an->f32(), d_, eps_, bXN_.data() + (size_t)i * d_); }, 4);
        matmul(*ly.wq, bXN_.data(), B, d_, bQ_.data(), qd, qd, d_); matmul(*ly.wk, bXN_.data(), B, d_, bK_.data(), kvd_, kvd_, d_); matmul(*ly.wv, bXN_.data(), B, d_, bV_.data(), kvd_, kvd_, d_);
        uint16_t *kl = kc_.data() + (size_t)l * ctx_ * kvd_, *vl = vc_.data() + (size_t)l * ctx_ * kvd_;
        for (int i = 0; i < B; i++) {
          float *q = bQ_.data() + (size_t)i * qd, *k = bK_.data() + (size_t)i * kvd_, *v = bV_.data() + (size_t)i * kvd_;
          for (int j = 0; j < qd; j++) q[j] += ly.bq->f32()[j];
          for (int j = 0; j < kvd_; j++) { k[j] += ly.bk->f32()[j]; v[j] += ly.bv->f32()[j]; }
          float cs[32], sn[32]; rope_tables(p0 + i, cs, sn);
          for (int h = 0; h < nh_; h++) rope(q + h * hd_, cs, sn);
          for (int g = 0; g < nkv_; g++) rope(k + g * hd_, cs, sn);
          to_f16(k, kl + (size_t)(p0 + i) * kvd_, kvd_); to_f16(v, vl + (size_t)(p0 + i) * kvd_, kvd_);
        }
        const float *sinks = ly.sinks->f32();
        pool().parallel_for(B * nh_, [&](int a, int b) {
          std::vector<float> sc(p0 + B + 1);
          for (int w = a; w < b; w++) {
            const int i = w / nh_, h = w % nh_, T = p0 + i + 1, g = h / rep, t_lo = ly.sliding ? std::max(0, T - window_) : 0;
            const float *qh = bQ_.data() + (size_t)i * qd + h * hd_; float mx = sinks[h];
            for (int t = t_lo; t < T; t++) { const uint16_t *kk = kl + (size_t)t * kvd_ + g * hd_; const float s = dot_f16(qh, kk, hd_) * scl; sc[t - t_lo] = s; mx = std::max(mx, s); }
            const int m = T - t_lo; for (int t = 0; t < m; t++) sc[t] -= mx;
            exp_v(sc.data(), sc.data(), m); float sum = std::exp(sinks[h] - mx); for (int t = 0; t < m; t++) sum += sc[t];
            float *out = bA_.data() + (size_t)i * qd + h * hd_; std::fill(out, out + hd_, 0.f); const float inv = 1.f / sum;
            for (int t = 0; t < m; t++) { const uint16_t *vv = vl + (size_t)(t_lo + t) * kvd_ + g * hd_; axpy_f16(sc[t] * inv, vv, out, hd_); }
          }
        }, 1);
        matmul(*ly.wo, bA_.data(), B, qd, bO_.data(), d_, d_, qd);
        for (int i = 0; i < B; i++) for (int j = 0; j < d_; j++) bX_[(size_t)i * d_ + j] += bO_[(size_t)i * d_ + j] + ly.bo->f32()[j];
        // MoE: route every token, then run each expert once over the tokens that chose it
        pool().parallel_for(B, [&](int a, int b) { for (int i = a; i < b; i++) rmsnorm(bX_.data() + (size_t)i * d_, ly.fn->f32(), d_, eps_, bXN_.data() + (size_t)i * d_); }, 4);
        matmul(*ly.rw, bXN_.data(), B, d_, bR_.data(), E_, E_, d_);
        std::vector<std::vector<std::pair<int, float>>> asg(E_);
        for (int i = 0; i < B; i++) {
          float *r = bR_.data() + (size_t)i * E_; for (int e = 0; e < E_; e++) r[e] += ly.rb->f32()[e];
          int idx[8]; float rw[8]; route(r, idx, rw); capture(l, p0 + i, bXN_.data() + (size_t)i * d_, idx, rw); for (int k = 0; k < K_; k++) asg[idx[k]].push_back({i, rw[k]});
        }
        std::fill(bO_.begin(), bO_.begin() + (size_t)B * d_, 0.f);   // accumulates the weighted expert outputs
        for (int e = 0; e < E_; e++) {
          const int ne = (int)asg[e].size(); if (!ne) continue;
          eX_.resize((size_t)ne * d_); eG_.resize((size_t)ne * 2 * ff_); eH_.resize((size_t)ne * ff_); eY_.resize((size_t)ne * d_);
          for (int j = 0; j < ne; j++) memcpy(eX_.data() + (size_t)j * d_, bXN_.data() + (size_t)asg[e][j].first * d_, 4 * d_);
          matmul(ly.gu[e][0], eX_.data(), ne, d_, eG_.data(), 2 * ff_, 2 * ff_, d_);
          if (ly.gu[e].size() > 1) {
            eG2_.resize((size_t)ne * 2 * ff_); matmul(ly.gu[e][1], eX_.data(), ne, d_, eG2_.data(), 2 * ff_, 2 * ff_, d_);
            for (size_t i = 0; i < (size_t)ne * 2 * ff_; i++) eG_[i] += eG2_[i];
          }
          const float *bg = ly.bgu->f32() + (size_t)e * 2 * ff_;
          pool().parallel_for(ne, [&](int a, int b) { for (int j = a; j < b; j++) { float *y = eG_.data() + (size_t)j * 2 * ff_; for (int i = 0; i < 2 * ff_; i++) y[i] += bg[i]; swiglu(y, eH_.data() + (size_t)j * ff_, ff_); } }, 1);
          matmul(ly.dn[e][0], eH_.data(), ne, ff_, eY_.data(), d_, d_, ff_);
          if (ly.dn[e].size() > 1) {
            eY2_.resize((size_t)ne * d_); matmul(ly.dn[e][1], eH_.data(), ne, ff_, eY2_.data(), d_, d_, ff_);
            for (size_t i = 0; i < (size_t)ne * d_; i++) eY_[i] += eY2_[i];
          }
          const float *bd = ly.bd->f32() + (size_t)e * d_;
          for (int j = 0; j < ne; j++) { float *o = bO_.data() + (size_t)asg[e][j].first * d_; const float w = asg[e][j].second, *y = eY_.data() + (size_t)j * d_; for (int i = 0; i < d_; i++) o[i] += w * (y[i] + bd[i]); }
        }
        for (size_t j = 0; j < (size_t)B * d_; j++) bX_[j] += bO_[j];
        if (last) dump(bX_.data() + (size_t)(B - 1) * d_);
      }
      if (all_logits_) {   // logits of every position (perplexity)
        pool().parallel_for(B, [&](int a, int b) { for (int i = a; i < b; i++) rmsnorm(bX_.data() + (size_t)i * d_, fnorm_->f32(), d_, eps_, bXN_.data() + (size_t)i * d_); }, 4);
        matmul(*head_, bXN_.data(), B, d_, all_logits_ + (size_t)s0 * V_, V_, V_, d_);
      }
      pos_ += B;
    }
    memcpy(x_.data(), bX_.data() + (size_t)((n - 1) % PB) * d_, 4 * d_);
    if (!want) return nullptr;
    rmsnorm(x_.data(), fnorm_->f32(), d_, eps_, xn_.data());
    if (want == 2) return xn_.data();
    matvec(*head_, xn_.data(), logits_.data(), V_, d_);
    return logits_.data();
  }

 private:
  const bool debug_ = getenv("QPACK_DEBUG") != nullptr;
  std::string cap_dir_; int cap_n_ = 2048; std::vector<FILE *> cap_f_;
  FILE *dump_ = nullptr; float *all_logits_ = nullptr;
  std::vector<float> part_;
  const int64_t attn_par_min_ = getenv("QPACK_ATTN_PAR") ? atoll(getenv("QPACK_ATTN_PAR")) : (1 << 20);
  int d_, L_, nh_, nkv_, hd_, ff_, V_, E_, K_, ctx_, kvd_, window_, pos_ = 0; float eps_, limit_, alpha_; double conc_;
  const Tensor *embed_, *head_, *fnorm_; std::vector<Layer> layers_;
  std::vector<uint16_t> kc_, vc_; std::vector<double> inv_;
  std::vector<float> x_, xn_, q_, k_, v_, att_, o_, logits_, rl_, gub_, gub2_, hb_, eo_, eo2_, eG2_, eY2_;
  std::vector<float> bX_, bXN_, bQ_, bK_, bV_, bA_, bO_, bR_, eX_, eG_, eH_, eY_;
};

}  // namespace

std::unique_ptr<Model> make_gptoss(std::shared_ptr<Container> c, int max_ctx) { return std::unique_ptr<Model>(new GptOss(c, max_ctx)); }

}  // namespace qp
