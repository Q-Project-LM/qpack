// Liquid LFM3 decoder: a hybrid of gated short-convolution layers (depthwise causal conv of width conv_L_cache over
// B * x, gated by C) and attention layers (GQA with QK RMSNorm, a sigmoid gate per head, partial "zero-tail" RoPE with its
// own theta for full / sliding-window layers). Every block is sandwich-normed (norm before and after the operator and
// the SwiGLU MLP); the final norm is `embedding_norm`; the head is tied to the embedding. Tensor names: python/qpack/convert_np.py.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "qpack/qpack.h"

namespace qp {
namespace {

struct Layer {
  bool attn = false;
  const Tensor *opn = nullptr, *ffn = nullptr, *pon = nullptr, *pfn = nullptr, *wg = nullptr, *wu = nullptr, *wd = nullptr;
  const Tensor *cin = nullptr, *cw = nullptr, *cout = nullptr;                                          // conv layers
  const Tensor *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr, *qn = nullptr, *kn = nullptr, *gate = nullptr;   // attention layers
  bool sliding = false; int slot = 0;   // attention: KV cache index; conv: state index
};

class Lfm3 : public Model {
 public:
  Lfm3(std::shared_ptr<Container> c, int max_ctx) {
    c_ = c; cfg_ = c->config();
    d_ = (int)cfg_["hidden_size"].as_int(); L_ = (int)cfg_["num_hidden_layers"].as_int(); nh_ = (int)cfg_["num_attention_heads"].as_int();
    nkv_ = (int)cfg_["num_key_value_heads"].as_int(); hd_ = (int)cfg_["head_dim"].as_int(d_ / nh_); ff_ = (int)cfg_["intermediate_size"].as_int();
    V_ = (int)cfg_["vocab_size"].as_int(); K_ = (int)cfg_["conv_L_cache"].as_int(7); win_ = (int)cfg_["sliding_window"].as_int(1 << 30);
    eps_ = (float)cfg_["norm_eps"].as_num(1e-5); kvd_ = nkv_ * hd_;
    const int limit = (int)cfg_["max_position_embeddings"].as_int(1 << 20);
    ctx_ = max_ctx > 0 ? max_ctx : std::min(4096, limit);
    embed_ = &c->at("embed"); fnorm_ = &c->at("final_norm");
    for (int l = 0; l < L_; l++) {
      auto g = [&](const char *n) { return c->get("layers." + std::to_string(l) + "." + n); };
      Layer ly; const std::string t = cfg_["layer_types"].arr.at(l).as_str(); ly.attn = t != "conv"; ly.sliding = t == "sliding_attention";
      ly.opn = g("operator_norm"); ly.ffn = g("ffn_norm"); ly.pon = g("post_operator_norm"); ly.pfn = g("post_ffn_norm"); ly.wg = g("w_gate"); ly.wu = g("w_up"); ly.wd = g("w_down");
      if (ly.attn) { ly.wq = g("wq"); ly.wk = g("wk"); ly.wv = g("wv"); ly.wo = g("wo"); ly.qn = g("q_norm"); ly.kn = g("k_norm"); ly.gate = g("gate"); ly.slot = n_attn_++; }
      else { ly.cin = g("conv_in"); ly.cw = g("conv_w"); ly.cout = g("conv_out"); ly.slot = n_conv_++; }
      if (!ly.opn || !ly.ffn || !ly.pon || !ly.pfn || !ly.wg || !ly.wu || !ly.wd || (ly.attn ? !(ly.wq && ly.wk && ly.wv && ly.wo && ly.qn && ly.kn && ly.gate) : !(ly.cin && ly.cw && ly.cout)))
        throw std::runtime_error("layer " + std::to_string(l) + ": missing tensors");
      layers_.push_back(ly);
    }
    // RoPE: the first `partial * hd / 2` frequencies follow the progression of the full head dimension, the rest are zero
    // (those dimensions are not rotated)
    for (int s = 0; s < 2; s++) {
      const Json &r = cfg_["rope"][s ? "sliding_attention" : "full_attention"];
      const double theta = r["theta"].as_num(10000), part = r["partial"].as_num(1.0); angles_[s] = (int)(part * hd_) / 2;
      inv_[s].resize(angles_[s]); for (int i = 0; i < angles_[s]; i++) inv_[s][i] = 1.0 / std::pow(theta, (2.0 * i) / hd_);
    }
    kc_.assign((size_t)n_attn_ * ctx_ * kvd_, 0); vc_.assign((size_t)n_attn_ * ctx_ * kvd_, 0);
    conv_.assign((size_t)n_conv_ * (K_ - 1) * d_, 0.f);
    x_.resize(d_); xn_.resize(d_); o_.resize(d_); o2_.resize(d_); bcx_.resize(3 * d_); h_.resize(d_); y_.resize(d_);
    q_.resize(nh_ * hd_); k_.resize(kvd_); v_.resize(kvd_); gt_.resize(nh_); att_.resize(nh_ * hd_); g_.resize(ff_); u_.resize(ff_); logits_.resize(V_);
  }
  int vocab() const override { return V_; }
  int hidden() const override { return d_; }
  int max_ctx() const override { return ctx_; }
  int pos() const override { return pos_; }
  std::string arch() const override { return "lfm3"; }
  bool recurrent() const override { return true; }   // the conv state cannot be rewound: prefix reuse goes through snapshots
  void reset() override { pos_ = 0; std::fill(conv_.begin(), conv_.end(), 0.f); }
  bool rewind(int pos) override { return pos == pos_; }

  std::vector<uint8_t> snapshot() const override {   // position, conv state, used prefix of the KV caches
    const size_t kvn = (size_t)pos_ * kvd_ * 2, cn = conv_.size() * 4;
    std::vector<uint8_t> s(4 + cn + 2 * n_attn_ * kvn); memcpy(s.data(), &pos_, 4); uint8_t *p = s.data() + 4;
    memcpy(p, conv_.data(), cn); p += cn;
    for (int a = 0; a < n_attn_; a++) for (const auto *buf : {&kc_, &vc_}) { memcpy(p, buf->data() + (size_t)a * ctx_ * kvd_, kvn); p += kvn; }
    return s;
  }
  void restore(const std::vector<uint8_t> &s) override {
    memcpy(&pos_, s.data(), 4); const uint8_t *p = s.data() + 4; const size_t kvn = (size_t)pos_ * kvd_ * 2, cn = conv_.size() * 4;
    memcpy(conv_.data(), p, cn); p += cn;
    for (int a = 0; a < n_attn_; a++) for (auto *buf : {&kc_, &vc_}) { memcpy(buf->data() + (size_t)a * ctx_ * kvd_, p, kvn); p += kvn; }
  }

  const float *step(int tok, int want) override {
    if (pos_ >= ctx_) throw std::runtime_error("context window full (" + std::to_string(ctx_) + " tokens)");
    if (tok < 0 || tok >= V_) throw std::runtime_error("token id out of range");
    row_f32(*embed_, tok, x_.data());
    const int T = pos_ + 1, rep = nh_ / nkv_, half = hd_ / 2; const float scl = 1.f / std::sqrt((float)hd_);
    float cs[2][256], sn[2][256];   // RoPE angles of this position for full / sliding layers (angles <= hd / 2 <= 256)
    for (int s = 0; s < 2; s++) for (int i = 0; i < angles_[s]; i++) { const double a = pos_ * inv_[s][i]; cs[s][i] = (float)std::cos(a); sn[s][i] = (float)std::sin(a); }
    for (int l = 0; l < L_; l++) {
      const Layer &ly = layers_[l];
      rmsnorm(x_.data(), ly.opn->f32(), d_, eps_, xn_.data());
      if (!ly.attn) {
        matvec(*ly.cin, xn_.data(), bcx_.data(), 3 * d_, d_);
        const float *B = bcx_.data(), *C = B + d_, *X = C + d_, *w = ly.cw->f32();
        float *st = conv_.data() + (size_t)ly.slot * (K_ - 1) * d_; const int cur = pos_ % (K_ - 1);
        for (int i = 0; i < d_; i++) {
          const float hv = B[i] * X[i]; float acc = w[(size_t)i * K_ + K_ - 1] * hv;
          for (int k = 0; k < K_ - 1; k++) acc += w[(size_t)i * K_ + k] * st[(size_t)((pos_ + k) % (K_ - 1)) * d_ + i];   // input of time pos_ - (K_ - 1 - k)
          y_[i] = C[i] * acc; st[(size_t)cur * d_ + i] = hv;
        }
        matvec(*ly.cout, y_.data(), o_.data(), d_, d_);
      } else {
        { const Tensor *w[4] = {ly.wq, ly.wk, ly.wv, ly.gate}; float *o[4] = {q_.data(), k_.data(), v_.data(), gt_.data()}; matvec_n(w, 4, xn_.data(), o, d_); }
        for (int h = 0; h < nh_; h++) rmsnorm(q_.data() + h * hd_, ly.qn->f32(), hd_, eps_, q_.data() + h * hd_);
        for (int g = 0; g < nkv_; g++) rmsnorm(k_.data() + g * hd_, ly.kn->f32(), hd_, eps_, k_.data() + g * hd_);
        const int s = ly.sliding ? 1 : 0, na = angles_[s];
        auto rope = [&](float *vec) { for (int j = 0; j < na; j++) { const float a = vec[j], b = vec[j + half]; vec[j] = a * cs[s][j] - b * sn[s][j]; vec[j + half] = b * cs[s][j] + a * sn[s][j]; } };
        for (int h = 0; h < nh_; h++) rope(q_.data() + h * hd_);
        for (int g = 0; g < nkv_; g++) rope(k_.data() + g * hd_);
        uint16_t *kl = kc_.data() + (size_t)ly.slot * ctx_ * kvd_, *vl = vc_.data() + (size_t)ly.slot * ctx_ * kvd_;
        to_f16(k_.data(), kl + (size_t)pos_ * kvd_, kvd_); to_f16(v_.data(), vl + (size_t)pos_ * kvd_, kvd_);
        const int t0 = ly.sliding ? std::max(0, T - win_) : 0, n = T - t0;
        auto heads = [&](int a, int b) {
          thread_local std::vector<float> sc; if ((int)sc.size() < n) sc.resize(n);
          for (int h = a; h < b; h++) {
            const int g = h / rep; const float *qh = q_.data() + h * hd_; float mx = -1e30f;
            for (int t = 0; t < n; t++) { const float v = dot_f16(qh, kl + (size_t)(t0 + t) * kvd_ + g * hd_, hd_) * scl; sc[t] = v; mx = std::max(mx, v); }
            for (int t = 0; t < n; t++) sc[t] -= mx;
            exp_v(sc.data(), sc.data(), n); float sum = 0; for (int t = 0; t < n; t++) sum += sc[t];
            float *out = att_.data() + h * hd_; std::fill(out, out + hd_, 0.f);
            for (int t = 0; t < n; t++) axpy_f16(sc[t], vl + (size_t)(t0 + t) * kvd_ + g * hd_, out, hd_);
            const float f = 1.f / (sum * (1.f + std::exp(-gt_[h])));   // softmax normalisation x sigmoid gate of the head
            for (int i = 0; i < hd_; i++) out[i] *= f;
          }
        };
        if ((int64_t)n * nh_ * hd_ > (1 << 20)) pool().parallel_for(nh_, heads, 1); else heads(0, nh_);
        matvec(*ly.wo, att_.data(), o_.data(), d_, nh_ * hd_);
      }
      rmsnorm(o_.data(), ly.pon->f32(), d_, eps_, o_.data());
      for (int i = 0; i < d_; i++) x_[i] += o_[i];
      rmsnorm(x_.data(), ly.ffn->f32(), d_, eps_, xn_.data());
      { const Tensor *w[2] = {ly.wg, ly.wu}; float *o[2] = {g_.data(), u_.data()}; matvec_n(w, 2, xn_.data(), o, d_); }
      silu_mul(g_.data(), u_.data(), ff_);
      matvec(*ly.wd, g_.data(), o_.data(), d_, ff_);
      rmsnorm(o_.data(), ly.pfn->f32(), d_, eps_, o_.data());
      for (int i = 0; i < d_; i++) x_[i] += o_[i];
    }
    pos_++;
    if (!want) return nullptr;
    rmsnorm(x_.data(), fnorm_->f32(), d_, eps_, xn_.data());
    if (want == 2) return xn_.data();
    matvec(*embed_, xn_.data(), logits_.data(), V_, d_);
    return logits_.data();
  }

 private:
  int d_, L_, nh_, nkv_, hd_, ff_, V_, K_, win_, ctx_, kvd_, pos_ = 0, n_attn_ = 0, n_conv_ = 0; float eps_;
  int angles_[2] = {0, 0}; std::vector<double> inv_[2];
  const Tensor *embed_, *fnorm_; std::vector<Layer> layers_;
  std::vector<uint16_t> kc_, vc_; std::vector<float> conv_;   // conv state per conv layer: the last K - 1 inputs, a ring indexed by position
  std::vector<float> x_, xn_, o_, o2_, bcx_, h_, y_, q_, k_, v_, gt_, att_, g_, u_, logits_;
};

}  // namespace

std::unique_ptr<Model> make_lfm3(std::shared_ptr<Container> c, int max_ctx) { return std::unique_ptr<Model>(new Lfm3(c, max_ctx)); }

}  // namespace qp
