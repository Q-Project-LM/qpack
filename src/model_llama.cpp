// Llama-family decoder (Llama, Qwen2, Mistral, SmolLM): RMSNorm, RoPE (rotate-half), GQA, optional q/k/v bias, SwiGLU,
// tied or untied output head. Tensor names are the canonical ones written by python/qpack/convert_llama.py.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "qpack/qpack.h"

namespace qp {
double g_attn_s = 0;   // profiling: seconds spent in decode attention
namespace {

struct Layer { const Tensor *an, *fn, *wq, *wk, *wv, *wo, *bq, *bk, *bv, *wg, *wu, *wd; };

class Llama : public Model {
 public:
  Llama(std::shared_ptr<Container> c, int max_ctx) {
    c_ = c; cfg_ = c->config();
    d_ = (int)cfg_["hidden_size"].as_int(); L_ = (int)cfg_["num_hidden_layers"].as_int(); nh_ = (int)cfg_["num_attention_heads"].as_int();
    nkv_ = (int)cfg_["num_key_value_heads"].as_int(); hd_ = (int)cfg_["head_dim"].as_int(d_ / nh_); ff_ = (int)cfg_["intermediate_size"].as_int();
    V_ = (int)cfg_["vocab_size"].as_int(); eps_ = (float)cfg_["rms_norm_eps"].as_num(1e-6); theta_ = cfg_["rope_theta"].as_num(10000);
    tie_ = cfg_["tie_embeddings"].as_bool(false);
    int limit = (int)cfg_["max_position_embeddings"].as_int(1 << 20);
    ctx_ = max_ctx > 0 ? max_ctx : std::min(4096, limit);
    embed_ = &c->at("embed"); head_ = tie_ ? embed_ : &c->at("lm_head"); fnorm_ = &c->at("final_norm");
    for (int l = 0; l < L_; l++) {
      auto g = [&](const char *n) { return c->get("layers." + std::to_string(l) + "." + n); };
      Layer ly{g("attn_norm"), g("ffn_norm"), g("wq"), g("wk"), g("wv"), g("wo"), g("bq"), g("bk"), g("bv"), g("w_gate"), g("w_up"), g("w_down")};
      if (!ly.an || !ly.wq || !ly.wg) throw std::runtime_error("layer " + std::to_string(l) + ": missing tensors");
      layers_.push_back(ly);
    }
    kvd_ = nkv_ * hd_;
    kc_.assign((size_t)L_ * ctx_ * kvd_, 0); vc_.assign((size_t)L_ * ctx_ * kvd_, 0);   // fp16 K/V cache
    x_.resize(d_); xn_.resize(d_); q_.resize(nh_ * hd_); k_.resize(kvd_); v_.resize(kvd_); att_.resize(nh_ * hd_); o_.resize(d_);
    g_.resize(ff_); u_.resize(ff_); logits_.resize(V_);
    inv_.resize(hd_ / 2); for (int i = 0; i < hd_ / 2; i++) inv_[i] = 1.0 / std::pow(theta_, (2.0 * i) / hd_);
  }
  int vocab() const override { return V_; }
  int hidden() const override { return d_; }
  int max_ctx() const override { return ctx_; }
  int pos() const override { return pos_; }
  std::string arch() const override { return "llama:" + cfg_["model_type"].as_str(); }
  void reset() override { pos_ = 0; }
  std::vector<uint8_t> checkpoint() const override { std::vector<uint8_t> c(4); memcpy(c.data(), &pos_, 4); return c; }
  void rollback(const std::vector<uint8_t> &c) override { memcpy(&pos_, c.data(), 4); }
  void prefill_all(const int *toks, int n, float *logits) override {
    if (n == 1) { const float *l = step(toks[0], 1); memcpy(logits, l, 4ull * V_); return; }
    all_logits_ = logits; prefill(toks, n, 0); all_logits_ = nullptr;
  }
  bool rewind(int pos) override { if (pos < 0 || pos > pos_) return false; pos_ = pos; return true; }

  std::vector<uint8_t> snapshot() const override {   // position + the used prefix of the KV cache
    std::vector<uint8_t> s(4 + 2ull * L_ * pos_ * kvd_ * 2); memcpy(s.data(), &pos_, 4); uint8_t *p = s.data() + 4;
    for (int l = 0; l < L_; l++) for (const auto *buf : {&kc_, &vc_}) { size_t n = (size_t)pos_ * kvd_ * 2; memcpy(p, buf->data() + (size_t)l * ctx_ * kvd_, n); p += n; }
    return s;
  }
  void restore(const std::vector<uint8_t> &s) override {
    memcpy(&pos_, s.data(), 4); const uint8_t *p = s.data() + 4;
    for (int l = 0; l < L_; l++) for (auto *buf : {&kc_, &vc_}) { size_t n = (size_t)pos_ * kvd_ * 2; memcpy(buf->data() + (size_t)l * ctx_ * kvd_, p, n); p += n; }
  }

  // the prompt in batches of up to PB tokens: every projection is a matrix x matrix product, attention per (token, head)
  const float *prefill(const int *toks, int n, int want) override {
    if (n == 1) return step(toks[0], want);
    if (pos_ + n > ctx_) throw std::runtime_error("context window full (" + std::to_string(ctx_) + " tokens)");
    const int PB = 64, half = hd_ / 2, rep = nh_ / nkv_, qd = nh_ * hd_; const float scl = 1.f / std::sqrt((float)hd_);
    bX_.resize((size_t)PB * d_); bXN_.resize((size_t)PB * d_); bQ_.resize((size_t)PB * qd); bK_.resize((size_t)PB * kvd_); bV_.resize((size_t)PB * kvd_);
    bA_.resize((size_t)PB * qd); bO_.resize((size_t)PB * d_); bG_.resize((size_t)PB * ff_); bU_.resize((size_t)PB * ff_);
    for (int s0 = 0; s0 < n; s0 += PB) {
      const int B = std::min(PB, n - s0), p0 = pos_;
      for (int i = 0; i < B; i++) {
        if (toks[s0 + i] < 0 || toks[s0 + i] >= V_) throw std::runtime_error("token id out of range");
        row_f32(*embed_, toks[s0 + i], bX_.data() + (size_t)i * d_);
      }
      for (int l = 0; l < L_; l++) {
        const Layer &ly = layers_[l];
        pool().parallel_for(B, [&](int a, int b) { for (int i = a; i < b; i++) rmsnorm(bX_.data() + (size_t)i * d_, ly.an->f32(), d_, eps_, bXN_.data() + (size_t)i * d_); }, 4);
        matmul(*ly.wq, bXN_.data(), B, d_, bQ_.data(), qd, qd, d_); matmul(*ly.wk, bXN_.data(), B, d_, bK_.data(), kvd_, kvd_, d_); matmul(*ly.wv, bXN_.data(), B, d_, bV_.data(), kvd_, kvd_, d_);
        uint16_t *kl = kc_.data() + (size_t)l * ctx_ * kvd_, *vl = vc_.data() + (size_t)l * ctx_ * kvd_;
        for (int i = 0; i < B; i++) {
          float *q = bQ_.data() + (size_t)i * qd, *k = bK_.data() + (size_t)i * kvd_, *v = bV_.data() + (size_t)i * kvd_;
          if (ly.bq) { for (int j = 0; j < qd; j++) q[j] += ly.bq->f32()[j]; for (int j = 0; j < kvd_; j++) { k[j] += ly.bk->f32()[j]; v[j] += ly.bv->f32()[j]; } }
          float cs[256], sn[256];
          for (int j = 0; j < half; j++) { double a = (p0 + i) * inv_[j]; cs[j] = (float)std::cos(a); sn[j] = (float)std::sin(a); }
          auto rope = [&](float *vec) { for (int j = 0; j < half; j++) { float a = vec[j], b = vec[j + half]; vec[j] = a * cs[j] - b * sn[j]; vec[j + half] = b * cs[j] + a * sn[j]; } };
          for (int h = 0; h < nh_; h++) rope(q + h * hd_);
          for (int g = 0; g < nkv_; g++) rope(k + g * hd_);
          to_f16(k, kl + (size_t)(p0 + i) * kvd_, kvd_); to_f16(v, vl + (size_t)(p0 + i) * kvd_, kvd_);
        }
        if (g_exact_batches) {   // speculative verification: the same arithmetic as step()
          for (int i = 0; i < B; i++) attend(kl, vl, bQ_.data() + (size_t)i * qd, p0 + i + 1, bA_.data() + (size_t)i * qd);
        } else pool().parallel_for(B * nh_, [&](int a, int b) {
          std::vector<float> sc(p0 + B);
          for (int w = a; w < b; w++) {
            const int i = w / nh_, h = w % nh_, T = p0 + i + 1, g = h / rep; const float *qh = bQ_.data() + (size_t)i * qd + h * hd_; float mx = -1e30f;
            for (int t = 0; t < T; t++) { const uint16_t *kk = kl + (size_t)t * kvd_ + g * hd_; float s = dot_f16(qh, kk, hd_); s *= scl; sc[t] = s; mx = std::max(mx, s); }
            float sum = 0; for (int t = 0; t < T; t++) sc[t] -= mx;
            exp_v(sc.data(), sc.data(), T); for (int t = 0; t < T; t++) sum += sc[t];
            float *out = bA_.data() + (size_t)i * qd + h * hd_; std::fill(out, out + hd_, 0.f); const float inv = 1.f / sum;
            for (int t = 0; t < T; t++) { const uint16_t *vv = vl + (size_t)t * kvd_ + g * hd_; axpy_f16(sc[t] * inv, vv, out, hd_); }
          }
        }, 1);
        matmul(*ly.wo, bA_.data(), B, qd, bO_.data(), d_, d_, qd);
        for (size_t j = 0; j < (size_t)B * d_; j++) bX_[j] += bO_[j];
        pool().parallel_for(B, [&](int a, int b) { for (int i = a; i < b; i++) rmsnorm(bX_.data() + (size_t)i * d_, ly.fn->f32(), d_, eps_, bXN_.data() + (size_t)i * d_); }, 4);
        matmul(*ly.wg, bXN_.data(), B, d_, bG_.data(), ff_, ff_, d_); matmul(*ly.wu, bXN_.data(), B, d_, bU_.data(), ff_, ff_, d_);
        pool().parallel_for(B, [&](int a, int b) { silu_mul(bG_.data() + (size_t)a * ff_, bU_.data() + (size_t)a * ff_, (b - a) * ff_); }, 1);
        matmul(*ly.wd, bG_.data(), B, ff_, bO_.data(), d_, d_, ff_);
        for (size_t j = 0; j < (size_t)B * d_; j++) bX_[j] += bO_[j];
      }
      if (all_logits_) {   // speculative verification: logits of every position
        pool().parallel_for(B, [&](int a, int b) { for (int i = a; i < b; i++) rmsnorm(bX_.data() + (size_t)i * d_, fnorm_->f32(), d_, eps_, bXN_.data() + (size_t)i * d_); }, 4);
        matmul(*head_, bXN_.data(), B, d_, all_logits_ + (size_t)s0 * V_, V_, V_, d_);
      }
      if (g_act_hook) {   // calibration (qpack-quantize): the output head sees every token's final hidden state
        for (int i = 0; i < B; i++) rmsnorm(bX_.data() + (size_t)i * d_, fnorm_->f32(), d_, eps_, bXN_.data() + (size_t)i * d_);
        g_act_hook(*head_, bXN_.data(), B, d_, d_);
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

  // attention of one token over cache positions [0, T): work items = (kv group, chunk of 64 positions); the query heads
  // of a group share each converted K / V row; chunks are merged in a fixed order (same result for any thread count)
  void attend(const uint16_t *kl, const uint16_t *vl, const float *q, int T, float *out) {
    const int C = 64, nc = (T + C - 1) / C, rep = nh_ / nkv_, hd = hd_; const float scl = 1.f / std::sqrt((float)hd);
    if (part_.size() < (size_t)nh_ * nc * (hd + 2)) part_.resize((size_t)nh_ * nc * (hd + 2));
    auto item = [&](int w0, int w1) {
      std::vector<float> sc((size_t)rep * C), kv(hd);
      for (int w = w0; w < w1; w++) {
        const int g = w / nc, c = w % nc, t0 = c * C, t1 = std::min(T, t0 + C), n = t1 - t0;
        if (!attn_scores(kl, kvd_, q, g, rep, hd, t0, t1, C, scl, sc.data())) for (int t = t0; t < t1; t++) {
          const uint16_t *kk = kl + (size_t)t * kvd_ + g * hd; f16_to_f32_v(kk, kv.data(), hd);
          for (int j = 0; j < rep; j++) sc[(size_t)j * C + (t - t0)] = dot_f32(q + (g * rep + j) * hd, kv.data(), hd) * scl;
        }
        for (int j = 0; j < rep; j++) {
          float *sj = sc.data() + (size_t)j * C; float mx = -1e30f; for (int t = 0; t < n; t++) mx = std::max(mx, sj[t]);
          for (int t = 0; t < n; t++) sj[t] -= mx;
          exp_v(sj, sj, n); float sum = 0; for (int t = 0; t < n; t++) sum += sj[t];
          float *pp = part_.data() + ((size_t)(g * rep + j) * nc + c) * (hd + 2); pp[0] = mx; pp[1] = sum; std::fill(pp + 2, pp + 2 + hd, 0.f);
        }
        float *pg = part_.data() + ((size_t)(g * rep) * nc + c) * (hd + 2) + 2; const size_t pstride = (size_t)nc * (hd + 2);
        if (!attn_values(vl, kvd_, g, rep, hd, t0, t1, C, sc.data(), pg, pstride)) for (int t = t0; t < t1; t++) {
          const uint16_t *vv = vl + (size_t)t * kvd_ + g * hd; f16_to_f32_v(vv, kv.data(), hd);
          for (int j = 0; j < rep; j++) axpy_f32(sc[(size_t)j * C + (t - t0)], kv.data(), pg + j * pstride, hd);
        }
      }
    };
    const int items = nkv_ * nc;
    if (items > 1 && (int64_t)T * nh_ * hd > attn_par_min_) pool().parallel_for(items, item, 1); else item(0, items);
    for (int h = 0; h < nh_; h++) {   // merge the chunks: softmax over all positions
      const float *ph = part_.data() + (size_t)h * nc * (hd + 2); float M = -1e30f; for (int c = 0; c < nc; c++) M = std::max(M, ph[(size_t)c * (hd + 2)]);
      float *o = out + h * hd; std::fill(o, o + hd, 0.f); float L = 0;
      for (int c = 0; c < nc; c++) { const float *pc = ph + (size_t)c * (hd + 2); const float f = std::exp(pc[0] - M); L += pc[1] * f; axpy_f32(f, pc + 2, o, hd); }
      const float il = 1.f / L; for (int i = 0; i < hd; i++) o[i] *= il;
    }
  }

  const float *step(int tok, int want) override {
    if (pos_ >= ctx_) throw std::runtime_error("context window full (" + std::to_string(ctx_) + " tokens)");
    if (tok < 0 || tok >= V_) throw std::runtime_error("token id out of range");
    row_f32(*embed_, tok, x_.data());
    const int half = hd_ / 2, T = pos_ + 1;
    std::vector<float> cs(half), sn(half);
    for (int i = 0; i < half; i++) { double a = pos_ * inv_[i]; cs[i] = (float)std::cos(a); sn[i] = (float)std::sin(a); }
    auto rope = [&](float *vec) { for (int i = 0; i < half; i++) { float a = vec[i], b = vec[i + half]; vec[i] = a * cs[i] - b * sn[i]; vec[i + half] = b * cs[i] + a * sn[i]; } };
    for (int l = 0; l < L_; l++) {
      const Layer &ly = layers_[l];
      rmsnorm(x_.data(), ly.an->f32(), d_, eps_, xn_.data());
      { const Tensor *w[3] = {ly.wq, ly.wk, ly.wv}; float *o[3] = {q_.data(), k_.data(), v_.data()}; matvec_n(w, 3, xn_.data(), o, d_); }
      if (ly.bq) { for (int i = 0; i < nh_ * hd_; i++) q_[i] += ly.bq->f32()[i]; for (int i = 0; i < kvd_; i++) { k_[i] += ly.bk->f32()[i]; v_[i] += ly.bv->f32()[i]; } }
      for (int h = 0; h < nh_; h++) rope(q_.data() + h * hd_);
      for (int g = 0; g < nkv_; g++) rope(k_.data() + g * hd_);
      uint16_t *kl = kc_.data() + (size_t)l * ctx_ * kvd_, *vl = vc_.data() + (size_t)l * ctx_ * kvd_;
      to_f16(k_.data(), kl + (size_t)pos_ * kvd_, kvd_); to_f16(v_.data(), vl + (size_t)pos_ * kvd_, kvd_);
      { const auto ta = std::chrono::steady_clock::now(); attend(kl, vl, q_.data(), T, att_.data()); g_attn_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - ta).count(); }
      matvec(*ly.wo, att_.data(), o_.data(), d_, nh_ * hd_);
      for (int i = 0; i < d_; i++) x_[i] += o_[i];
      rmsnorm(x_.data(), ly.fn->f32(), d_, eps_, xn_.data());
      { const Tensor *w[2] = {ly.wg, ly.wu}; float *o[2] = {g_.data(), u_.data()}; matvec_n(w, 2, xn_.data(), o, d_); }
      silu_mul(g_.data(), u_.data(), ff_);
      matvec(*ly.wd, g_.data(), o_.data(), d_, ff_);
      for (int i = 0; i < d_; i++) x_[i] += o_[i];
      if (debug_) fprintf(stderr, "pos %d layer %d: %.5f %.5f %.5f %.5f\n", pos_, l, x_[0], x_[1], x_[2], x_[3]);
    }
    pos_++;
    if (!want) return nullptr;
    rmsnorm(x_.data(), fnorm_->f32(), d_, eps_, xn_.data());
    if (want == 2) return xn_.data();
    matvec(*head_, xn_.data(), logits_.data(), V_, d_);
    return logits_.data();
  }

 private:
  const bool debug_ = getenv("QPACK_DEBUG") != nullptr;
  float *all_logits_ = nullptr;
  std::vector<float> part_;   // attention partials: per head and chunk [max, sum, out(hd)]
  // decode attention runs on one thread below this many q x k products (~1170 positions for 14 heads x 64): measured
  // faster here than the pool round trip; QPACK_ATTN_PAR overrides
  const int64_t attn_par_min_ = getenv("QPACK_ATTN_PAR") ? atoll(getenv("QPACK_ATTN_PAR")) : (1 << 20);
  int d_, L_, nh_, nkv_, hd_, ff_, V_, ctx_, kvd_, pos_ = 0; float eps_; double theta_; bool tie_;
  const Tensor *embed_, *head_, *fnorm_; std::vector<Layer> layers_;
  std::vector<uint16_t> kc_, vc_;
  std::vector<float> x_, xn_, q_, k_, v_, att_, o_, g_, u_, logits_; std::vector<double> inv_;
  std::vector<float> bX_, bXN_, bQ_, bK_, bV_, bA_, bO_, bG_, bU_;   // prompt batch buffers
};

}  // namespace

std::unique_ptr<Model> make_llama(std::shared_ptr<Container> c, int max_ctx) { return std::unique_ptr<Model>(new Llama(c, max_ctx)); }

std::unique_ptr<Model> load_model(const std::string &path, int max_ctx) {
  auto c = std::make_shared<Container>(path);
  const std::string arch = c->config()["arch"].as_str(), mt = c->config()["model_type"].as_str();
  const char *dev = getenv("QPACK_DEVICE"); const bool cuda = dev && !strcmp(dev, "cuda");
  if (cuda && arch != "llama") throw std::runtime_error("QPACK_DEVICE=cuda: only Llama-family models are supported (arch='" + arch + "')");
  if (arch == "llama") {
    if (cuda) {
#ifdef QPACK_HAVE_CUDA
      return make_llama_cuda(c, max_ctx);
#else
      throw std::runtime_error("QPACK_DEVICE=cuda: this build has no CUDA support (configure with -DQPACK_CUDA=ON)");
#endif
    }
    return make_llama(c, max_ctx);
  }
  if (arch == "lfm3") return make_lfm3(c, max_ctx);
  if (mt == "nq4") return make_nq(c, max_ctx);
  throw std::runtime_error("unsupported model in " + path + " (arch='" + arch + "', model_type='" + mt + "')");
}

SelfTest run_selftest(Model &m, double tol) {
  SelfTest r; auto &c = m.container();
  if (!c.has("selftest.json")) return r;
  const Json st = c.json("selftest.json");   // keep it alive: ranging over a member of a temporary would dangle
  for (auto &cs : st["cases"].arr) {
    m.reset(); const float *lg = nullptr; auto &ids = cs["ids"].arr;
    for (size_t i = 0; i < ids.size(); i++) lg = m.step((int)ids[i].as_int(), i + 1 == ids.size() ? 1 : 0);
    int best = 0; for (int v = 1; v < m.vocab(); v++) if (lg[v] > lg[best]) best = v;
    double err = 0; for (size_t k = 0; k < cs["top5"].arr.size(); k++) err = std::max(err, std::fabs(lg[cs["top5"][k].as_int()] - cs["top5_logits"][k].as_num()));
    r.n++; r.max_err = std::max(r.max_err, err); if (best == cs["top5"][0].as_int() && err < tol) r.ok++;
  }
  m.reset(); return r;
}

}  // namespace qp
