// N-Q (NQ4ForCausalLM, dense checkpoints): a port of site/nq-runtime/nq_core.c, itself checked against PyTorch.
//   embed -> hc residual streams -> L x [ (ple layer: + hashed n-gram PLE) -> Gated Residual -> Gated DeltaNet | gated
//   attention -> Gated Residual -> SwiGLU ] -> Gated Residual collapse -> lm_head
// Decode state: per DeltaNet layer a short-conv window + recurrent matrix S per value head, per attention layer a K/V
// cache, for PLE the last two tokens + a dilated conv window. All of it lives in one float block (snapshot = copy).
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "qpack/qpack.h"

namespace qp {
namespace {

static inline float sigm(float v) { return 1.f / (1.f + std::exp(-v)); }
static inline float silu(float v) { return v / (1.f + std::exp(-v)); }
static inline float softplus(float v) { return v > 20.f ? v : std::log1p(std::exp(v)); }
// zero-centred RMSNorm per group of g elements; w has n entries (whole vector) or g (shared per group)
static void rmsn(const float *x, const float *w, int n, int g, bool wfull, float eps, float *out) {
  for (int g0 = 0; g0 < n; g0 += g) {
    float ss = 0; for (int i = g0; i < g0 + g; i++) ss += x[i] * x[i];
    const float inv = 1.f / std::sqrt(ss / g + eps);
    for (int i = g0; i < g0 + g; i++) out[i] = x[i] * inv * (1.f + w[wfull ? i : i - g0]);
  }
}

struct GR { const Tensor *norm, *down, *up, *inj; };
struct Layer {
  bool attn; GR mix, ffn; const Tensor *gu, *down;
  const Tensor *qkvz, *ba, *conv, *alog, *dtb, *gnorm, *out;      // Gated DeltaNet
  const Tensor *q, *k, *v, *o, *qn, *kn;                          // gated attention
  size_t off_a = 0, off_b = 0;                                    // state offsets: conv window + S, or K + V cache
};

class NQ : public Model {
 public:
  NQ(std::shared_ptr<Container> c, int max_ctx) {
    c_ = c; cfg_ = c->config();
    if (cfg_["ternary"].as_bool() || cfg_["fingerprint"].as_bool() || cfg_["num_experts"].as_int() > 0) throw std::runtime_error("N-Q: only dense checkpoints are supported");
    d_ = (int)cfg_["hidden_size"].as_int(); hc_ = (int)cfg_["hc_count"].as_int(); L_ = (int)cfg_["num_hidden_layers"].as_int();
    rank_ = (int)cfg_["hc_lowrank"].as_int(); ff_ = (int)cfg_["intermediate_size"].as_int(); V_ = (int)cfg_["vocab_size"].as_int();
    eps_ = (float)cfg_["rms_norm_eps"].as_num(1e-6);
    nk_ = (int)cfg_["linear_num_key_heads"].as_int(); nv_ = (int)cfg_["linear_num_value_heads"].as_int();
    dk_ = (int)cfg_["linear_key_head_dim"].as_int(); dv_ = (int)cfg_["linear_value_head_dim"].as_int(); K_ = (int)cfg_["linear_conv_kernel"].as_int();
    nh_ = (int)cfg_["num_attention_heads"].as_int(); nkv_ = (int)cfg_["num_key_value_heads"].as_int(); hd_ = (int)cfg_["head_dim"].as_int();
    rot_ = (int)(hd_ * cfg_["partial_rotary_factor"].as_num(0.25));
    const int limit = (int)cfg_["max_position_embeddings"].as_int(4096);
    ctx_ = max_ctx > 0 ? max_ctx : limit;
    const Json &eos = cfg_["eos_token_id"]; eos_ = (int)(eos.type == Json::Array ? eos.arr.back().as_int() : eos.as_int());
    const std::string pat = cfg_["layer_pattern"].as_str();
    const double theta = cfg_["rope_theta"].as_num(1e6);
    for (int i = 0; i < rot_ / 2; i++) inv_.push_back(1.0 / std::pow(theta, (2.0 * i) / rot_));
    embed_ = &c->at("embed.weight"); head_ = &c->at("lm_head.weight");
    fin_ = {&c->at("mixer.norm.weight"), &c->at("mixer.down.weight"), &c->at("mixer.up.weight"), nullptr};
    decide_w_ = c->get("decide_head.weight"); decide_b_ = c->get("decide_head.bias");
    const int HD = hc_ * d_, Cc = 2 * nk_ * dk_ + nv_ * dv_;
    ple_layer_ = (int)cfg_["ple_layer"].as_int(0) - 1;
    size_t st = 0;
    if (ple_layer_ >= 0) {
      const std::string P = "layers." + std::to_string(ple_layer_) + ".ple.";
      const Tensor &sz = c->at(P + "sizes"), &of = c->at(P + "offsets"), &mu = c->at(P + "mult");
      for (int i = 0; i < sz.numel(); i++) { p_sizes_.push_back(((const int64_t *)sz.data)[i]); p_offs_.push_back(((const int64_t *)of.data)[i]); }
      for (int i = 0; i < mu.numel(); i++) p_mult_.push_back((uint64_t)((const int64_t *)mu.data)[i]);
      p_tab_ = &c->at(P + "table.weight"); p_key_ = &c->at(P + "key.weight"); p_val_ = &c->at(P + "value.weight");
      p_nk_ = &c->at(P + "norm_k.weight"); p_nq_ = &c->at(P + "norm_q.weight"); p_nc_ = &c->at(P + "norm_c.weight"); p_conv_ = &c->at(P + "conv.weight");
      p_hpn_ = (int)cfg_["heads_per_ngram"].as_int(); p_hdim_ = (int)p_tab_->shape[1]; p_dil_ = (int)cfg_["ngram_size"].as_int(); p_K_ = (int)cfg_["ple_conv_kernel"].as_int();
      p_span_ = (p_K_ - 1) * p_dil_; ple_win_ = st; st += (size_t)(p_span_ + 1) * HD;
    }
    for (int l = 0; l < L_; l++) {
      const std::string P = "layers." + std::to_string(l) + ".";
      auto g = [&](const std::string &n) { return c->get(P + n); };
      Layer ly{};
      ly.attn = pat[l % pat.size()] == 'A';
      ly.mix = {g("gr_mix.norm.weight"), g("gr_mix.down.weight"), g("gr_mix.up.weight"), g("gr_mix.inject.weight")};
      ly.ffn = {g("gr_ffn.norm.weight"), g("gr_ffn.down.weight"), g("gr_ffn.up.weight"), g("gr_ffn.inject.weight")};
      ly.gu = g("ffn.gate_up.weight"); ly.down = g("ffn.down.weight");
      if (ly.attn) {
        ly.q = g("mixer.q.weight"); ly.k = g("mixer.k.weight"); ly.v = g("mixer.v.weight"); ly.o = g("mixer.o.weight"); ly.qn = g("mixer.q_norm.weight"); ly.kn = g("mixer.k_norm.weight");
        if (!ly.q || !ly.o || !ly.qn) throw std::runtime_error("layer " + std::to_string(l) + ": missing attention tensors");
        ly.off_a = st; st += (size_t)ctx_ * nkv_ * hd_; ly.off_b = st; st += (size_t)ctx_ * nkv_ * hd_;
      } else {
        ly.qkvz = g("mixer.in_qkvz.weight"); ly.ba = g("mixer.in_ba.weight"); ly.conv = g("mixer.conv.weight"); ly.alog = g("mixer.A_log");
        ly.dtb = g("mixer.dt_bias"); ly.gnorm = g("mixer.norm.weight"); ly.out = g("mixer.out.weight");
        if (!ly.qkvz || !ly.out || !ly.conv) throw std::runtime_error("layer " + std::to_string(l) + ": missing DeltaNet tensors");
        ly.off_a = st; st += (size_t)K_ * Cc; ly.off_b = st; st += (size_t)nv_ * dk_ * dv_;
      }
      if (!ly.mix.norm || !ly.gu) throw std::runtime_error("layer " + std::to_string(l) + ": missing tensors");
      layers_.push_back(ly);
    }
    state_.assign(st, 0.f);
    h_.resize(HD); hn_.resize(HD); low_.resize(rank_); gate_.resize(HD); mixed_.resize(d_); inj_.resize(hc_); y_.resize(d_); gu_.resize(2 * ff_); act_.resize(ff_);
    qkvz_.resize(Cc + nv_ * dv_); ba_.resize(2 * nv_); xc_.resize(Cc); qh_.resize(nk_ * dk_); kh_.resize(nk_ * dk_); o_.resize(nv_ * dv_);
    qa_.resize(2 * nh_ * hd_); ka_.resize(nkv_ * hd_); va_.resize(nkv_ * hd_); ao_.resize(nh_ * hd_);
    e_.resize(p_sizes_.size() * p_hdim_ + 16); pk_.resize(HD); pv_.resize(d_); pq_.resize(HD); gv_.resize(HD); tmp_.resize(HD); fin_v_.resize(d_); logits_.resize(V_);
    reset();
  }
  int vocab() const override { return V_; }
  int hidden() const override { return d_; }
  int max_ctx() const override { return ctx_; }
  int pos() const override { return pos_; }
  std::string arch() const override { return "nq4"; }
  bool recurrent() const override { return true; }
  void reset() override { pos_ = 0; p1_ = p2_ = eos_; ple_n_ = 0; std::fill(state_.begin(), state_.end(), 0.f); }

  std::vector<uint8_t> snapshot() const override {   // header + the whole state block (K/V caches only up to pos)
    std::vector<uint8_t> s(16 + state_.size() * 4);
    int hdr[4] = {pos_, p1_, p2_, ple_n_}; memcpy(s.data(), hdr, 16); memcpy(s.data() + 16, state_.data(), state_.size() * 4);
    return s;
  }
  void restore(const std::vector<uint8_t> &s) override {
    if (s.size() != 16 + state_.size() * 4) throw std::runtime_error("N-Q: snapshot size mismatch");
    int hdr[4]; memcpy(hdr, s.data(), 16); pos_ = hdr[0]; p1_ = hdr[1]; p2_ = hdr[2]; ple_n_ = hdr[3];
    memcpy(state_.data(), s.data() + 16, state_.size() * 4);
  }
  bool decision(const float *hidden, float *score) const override {
    if (!decide_w_) return false;
    float s = decide_b_ ? decide_b_->f32()[0] : 0.f; const float *w = decide_w_->f32();
    for (int i = 0; i < d_; i++) s += w[i] * hidden[i];
    *score = std::max(-30.f, std::min(30.f, s)); return true;
  }

  // the prompt in batches: every projection is a matrix x matrix product over the batch, the recurrent parts (conv
  // windows, delta rule, PLE n-gram history) and the attention run token by token inside each layer
  const float *prefill(const int *toks, int n, int want) override {
    if (n == 1) return step(toks[0], want);
    if (pos_ + n > ctx_) throw std::runtime_error("context window full (" + std::to_string(ctx_) + " tokens)");
    const int PB = 64, D = d_, HC = hc_, HD = HC * D, Cc = 2 * nk_ * dk_ + nv_ * dv_, QK = Cc + nv_ * dv_, ne = (int)p_sizes_.size() * p_hdim_;
    const int qd2 = 2 * nh_ * hd_, kvd = nkv_ * hd_;
    auto rs = [&](std::vector<float> &v, size_t m) { if (v.size() < (size_t)PB * m) v.resize((size_t)PB * m); };
    rs(bH_, HD); rs(bHN_, HD); rs(bLOW_, rank_); rs(bGATE_, HD); rs(bMIX_, D); rs(bINJ_, HC); rs(bY_, D); rs(bE_, ne + 16); rs(bTK_, HD); rs(bPV_, D);
    rs(bQA_, qd2); rs(bKA_, kvd); rs(bVA_, kvd); rs(bAO_, nh_ * hd_); rs(bQKVZ_, QK); rs(bBA_, 2 * nv_); rs(bO_, nv_ * dv_); rs(bGU_, 2 * ff_); rs(bACT_, ff_);
    for (int s0 = 0; s0 < n; s0 += PB) {
      const int B = std::min(PB, n - s0), p0 = pos_;
      for (int i = 0; i < B; i++) {
        if (toks[s0 + i] < 0 || toks[s0 + i] >= V_) throw std::runtime_error("token id out of range");
        row_f32(*embed_, toks[s0 + i], tmp_.data());
        for (int c = 0; c < HC; c++) memcpy(bH_.data() + (size_t)i * HD + c * D, tmp_.data(), 4 * D);
      }
      for (int l = 0; l < L_; l++) {
        const Layer &ly = layers_[l];
        if (l == ple_layer_) {
          for (int i = 0; i < B; i++) ple_gather(toks[s0 + i], bE_.data() + (size_t)i * ne);
          matmul(*p_key_, bE_.data(), B, ne, bTK_.data(), HD, HD, ne); matmul(*p_val_, bE_.data(), B, ne, bPV_.data(), D, D, ne);
          for (int i = 0; i < B; i++) ple_mix(bTK_.data() + (size_t)i * HD, bPV_.data() + (size_t)i * D, bH_.data() + (size_t)i * HD);
        }
        gres_batch(ly.mix, true, B);
        if (ly.attn) {
          matmul(*ly.q, bMIX_.data(), B, D, bQA_.data(), qd2, qd2, D); matmul(*ly.k, bMIX_.data(), B, D, bKA_.data(), kvd, kvd, D); matmul(*ly.v, bMIX_.data(), B, D, bVA_.data(), kvd, kvd, D);
          for (int i = 0; i < B; i++) attn_kv(ly, bKA_.data() + (size_t)i * kvd, bVA_.data() + (size_t)i * kvd, p0 + i);
          pool().parallel_for(B * nh_, [&](int a, int b) { for (int w = a; w < b; w++) { const int i = w / nh_, h = w % nh_; attn_heads(ly, bQA_.data() + (size_t)i * qd2, p0 + i, bAO_.data() + (size_t)i * nh_ * hd_, h, h + 1); } }, 1);
          matmul(*ly.o, bAO_.data(), B, nh_ * hd_, bY_.data(), D, D, nh_ * hd_);
        } else {
          matmul(*ly.qkvz, bMIX_.data(), B, D, bQKVZ_.data(), QK, QK, D); matmul(*ly.ba, bMIX_.data(), B, D, bBA_.data(), 2 * nv_, 2 * nv_, D);
          for (int i = 0; i < B; i++) gdn_core(ly, bQKVZ_.data() + (size_t)i * QK, bBA_.data() + (size_t)i * 2 * nv_, bO_.data() + (size_t)i * nv_ * dv_);
          matmul(*ly.out, bO_.data(), B, nv_ * dv_, bY_.data(), D, D, nv_ * dv_);
        }
        write_batch(B);
        gres_batch(ly.ffn, true, B);
        matmul(*ly.gu, bMIX_.data(), B, D, bGU_.data(), 2 * ff_, 2 * ff_, D);
        for (int i = 0; i < B; i++) { float *g = bGU_.data() + (size_t)i * 2 * ff_; silu_mul(g, g + ff_, ff_); memcpy(bACT_.data() + (size_t)i * ff_, g, 4 * ff_); }
        matmul(*ly.down, bACT_.data(), B, ff_, bY_.data(), D, D, ff_);
        write_batch(B);
      }
      if (ple_layer_ < 0) for (int i = 0; i < B; i++) { p2_ = p1_; p1_ = toks[s0 + i]; }
      pos_ += B;
      if (s0 + B == n) memcpy(h_.data(), bH_.data() + (size_t)(B - 1) * HD, 4 * HD);
    }
    if (!want) return nullptr;
    gres(fin_, false); memcpy(fin_v_.data(), mixed_.data(), 4 * d_);
    if (want == 2) return fin_v_.data();
    matvec(*head_, fin_v_.data(), logits_.data(), V_, d_);
    return logits_.data();
  }

  const float *step(int tok, int want) override {
    if (pos_ >= ctx_) throw std::runtime_error("context window full (" + std::to_string(ctx_) + " tokens)");
    if (tok < 0 || tok >= V_) throw std::runtime_error("token id out of range");
    row_f32(*embed_, tok, tmp_.data());
    for (int c = 0; c < hc_; c++) memcpy(h_.data() + c * d_, tmp_.data(), 4 * d_);
    for (int l = 0; l < L_; l++) {
      const Layer &ly = layers_[l];
      if (l == ple_layer_) ple(tok);
      gres(ly.mix, true);
      if (ly.attn) attn(ly, mixed_.data()); else gdn(ly, mixed_.data());
      write(y_.data());
      gres(ly.ffn, true);
      matvec(*ly.gu, mixed_.data(), gu_.data(), 2 * ff_, d_);
      silu_mul(gu_.data(), gu_.data() + ff_, ff_);
      matvec(*ly.down, gu_.data(), y_.data(), d_, ff_);
      write(y_.data());
    }
    if (ple_layer_ < 0) { p2_ = p1_; p1_ = tok; }
    pos_++;
    if (!want) return nullptr;
    gres(fin_, false); memcpy(fin_v_.data(), mixed_.data(), 4 * d_);
    if (want == 2) return fin_v_.data();
    matvec(*head_, fin_v_.data(), logits_.data(), V_, d_);
    return logits_.data();
  }

 private:
  void gres(const GR &g, bool inject) {   // Gated Residual read: h -> mixed (d) [, inj (hc)]
    const int D = d_, HC = hc_, HD = HC * D;
    rmsn(h_.data(), g.norm->f32(), HD, D, true, eps_, hn_.data());
    matvec(*g.down, hn_.data(), low_.data(), rank_, HD);
    for (int i = 0; i < rank_; i++) low_[i] = silu(low_[i] / HC);
    matvec(*g.up, low_.data(), gate_.data(), HD, rank_);
    sigmoid_v(gate_.data(), gate_.data(), HD);
    for (int j = 0; j < D; j++) { float s = 0; for (int c = 0; c < HC; c++) s += gate_[c * D + j] * hn_[c * D + j]; mixed_[j] = s / HC; }
    if (inject) { matvec(*g.inj, hn_.data(), inj_.data(), HC, HD); for (int c = 0; c < HC; c++) inj_[c] = 2.f * sigm(inj_[c] / HC); }
  }
  void write(const float *y) { for (int c = 0; c < hc_; c++) for (int j = 0; j < d_; j++) h_[c * d_ + j] += y[j] * inj_[c]; }

  void gdn(const Layer &ly, const float *x) {
    const int D = d_;
    { const Tensor *w[2] = {ly.qkvz, ly.ba}; float *o[2] = {qkvz_.data(), ba_.data()}; matvec_n(w, 2, x, o, D); }
    gdn_core(ly, qkvz_.data(), ba_.data(), o_.data());
    matvec(*ly.out, o_.data(), y_.data(), D, nv_ * dv_);
  }
  // the sequential part of a Gated DeltaNet step: conv window, q/k normalisation, delta-rule recurrence, gated norm
  void gdn_core(const Layer &ly, const float *qkvz, const float *ba, float *o) {
    const int nk = nk_, nv = nv_, dk = dk_, dv = dv_, K = K_, kd = nk * dk, Cc = 2 * kd + nv * dv, rep = nv / nk;
    float *win = state_.data() + ly.off_a;
    memmove(win, win + Cc, 4ull * (K - 1) * Cc); memcpy(win + (size_t)(K - 1) * Cc, qkvz, 4ull * Cc);
    const float *cw = ly.conv->f32();
    for (int ch = 0; ch < Cc; ch++) { float s = 0; for (int j = 0; j < K; j++) s += cw[ch * K + j] * win[(size_t)j * Cc + ch]; xc_[ch] = s; }
    sigmoid_v(xc_.data(), gv_.data(), Cc); for (int ch = 0; ch < Cc; ch++) xc_[ch] *= gv_[ch];   // silu
    const float scale = 1.f / std::sqrt((float)dk);
    for (int h = 0; h < nk; h++) {   // l2-normalised q (scaled) and k per key head
      float sq = 0, sk = 0; for (int i = 0; i < dk; i++) { sq += xc_[h * dk + i] * xc_[h * dk + i]; sk += xc_[kd + h * dk + i] * xc_[kd + h * dk + i]; }
      const float nq = std::max(std::sqrt(sq), 1e-6f), nkk = std::max(std::sqrt(sk), 1e-6f);
      for (int i = 0; i < dk; i++) { qh_[h * dk + i] = xc_[h * dk + i] / nq * scale; kh_[h * dk + i] = xc_[kd + h * dk + i] / nkk; }
    }
    const float *Al = ly.alog->f32(), *dtb = ly.dtb->f32(), *nw = ly.gnorm->f32();
    float *Sb = state_.data() + ly.off_b;
    for (int vh = 0; vh < nv; vh++) {   // delta rule: S *= g; d = (v - k S) beta; S += k d^T; o = q S
      const float *q = qh_.data() + (vh / rep) * dk, *k = kh_.data() + (vh / rep) * dk, *v = xc_.data() + 2 * kd + vh * dv;
      float *S = Sb + (size_t)vh * dk * dv;
      const float beta = sigm(ba[vh]), g = std::exp(-std::exp(Al[vh]) * softplus(ba[nv + vh] + dtb[vh]));
      float kS[512] = {}, ov[512] = {};
      for (int i = 0; i < dk; i++) { float *Sr = S + i * dv; const float ki = k[i]; for (int j = 0; j < dv; j++) { Sr[j] *= g; kS[j] += ki * Sr[j]; } }
      for (int j = 0; j < dv; j++) kS[j] = (v[j] - kS[j]) * beta;
      for (int i = 0; i < dk; i++) { float *Sr = S + i * dv; const float ki = k[i], qi = q[i]; for (int j = 0; j < dv; j++) { Sr[j] += ki * kS[j]; ov[j] += qi * Sr[j]; } }
      float ss = 0; for (int j = 0; j < dv; j++) ss += ov[j] * ov[j];
      const float inv = 1.f / std::sqrt(ss / dv + eps_);
      float gz[512]; sigmoid_v(qkvz + Cc + vh * dv, gz, dv);
      for (int j = 0; j < dv; j++) o[vh * dv + j] = ov[j] * inv * (1.f + nw[j]) * gz[j];
    }
  }

  void attn(const Layer &ly, const float *x) {
    const int D = d_, nh = nh_, hd = hd_;
    { const Tensor *w[3] = {ly.q, ly.k, ly.v}; float *o[3] = {qa_.data(), ka_.data(), va_.data()}; matvec_n(w, 3, x, o, D); }
    attn_kv(ly, ka_.data(), va_.data(), pos_);
    auto heads = [&](int h0, int h1) { attn_heads(ly, qa_.data(), pos_, ao_.data(), h0, h1); };
    if ((int64_t)(pos_ + 1) * nh * hd > (1 << 15)) pool().parallel_for(nh, heads, 1); else heads(0, nh);
    matvec(*ly.o, ao_.data(), y_.data(), D, nh * hd);
  }
  // zero-centred RMSNorm over one head, then partial RoPE at position pos
  void rope_norm(float *out, const float *src, const float *w, int pos) const {
    const int hd = hd_, half = rot_ / 2; float cs[128], sn[128];
    for (int i = 0; i < half; i++) { const float a = (float)pos * (float)inv_[i]; cs[i] = std::cos(a); sn[i] = std::sin(a); }
    float ss = 0; for (int i = 0; i < hd; i++) ss += src[i] * src[i];
    const float inv = 1.f / std::sqrt(ss / hd + eps_);
    for (int i = 0; i < hd; i++) out[i] = src[i] * inv * (1.f + w[i]);
    for (int i = 0; i < half; i++) { float a = out[i], b = out[i + half]; out[i] = a * cs[i] - b * sn[i]; out[i + half] = b * cs[i] + a * sn[i]; }
  }
  void attn_kv(const Layer &ly, const float *k, const float *v, int pos) {   // K/V of one token into the cache
    float *kc = state_.data() + ly.off_a, *vc = state_.data() + ly.off_b; const float *kn = ly.kn->f32();
    for (int g = 0; g < nkv_; g++) { rope_norm(kc + ((size_t)pos * nkv_ + g) * hd_, k + g * hd_, kn, pos); memcpy(vc + ((size_t)pos * nkv_ + g) * hd_, v + g * hd_, 4 * hd_); }
  }
  // heads [h0, h1) of the token at pos: q2 = [query (hd) | gate (hd)] per head, attends to cache positions 0..pos
  void attn_heads(const Layer &ly, const float *q2, int pos, float *ao, int h0, int h1) const {
    const int hd = hd_, nkv = nkv_, rep = nh_ / nkv_, T = pos + 1; const float scl = 1.f / std::sqrt((float)hd);
    const float *kc = state_.data() + ly.off_a, *vc = state_.data() + ly.off_b, *qn = ly.qn->f32();
    std::vector<float> sc(T), qv(hd);
    for (int h = h0; h < h1; h++) {
      rope_norm(qv.data(), q2 + h * 2 * hd, qn, pos); const int g = h / rep; float mx = -1e30f;
      for (int t = 0; t < T; t++) { const float *kk = kc + ((size_t)t * nkv + g) * hd; float s = dot_f32(qv.data(), kk, hd); s *= scl; sc[t] = s; mx = std::max(mx, s); }
      float sum = 0; for (int t = 0; t < T; t++) { sc[t] = std::exp(sc[t] - mx); sum += sc[t]; }
      float *out = ao + h * hd; std::fill(out, out + hd, 0.f); const float isum = 1.f / sum;
      for (int t = 0; t < T; t++) { const float *vv = vc + ((size_t)t * nkv + g) * hd; axpy_f32(sc[t] * isum, vv, out, hd); }
      for (int i = 0; i < hd; i++) out[i] *= sigm(q2[h * 2 * hd + hd + i]);
    }
  }

  void ple(int tok) {   // hashed 2-/3-gram embeddings -> gated, normalised, dilated causal conv -> added to every stream
    const int ne = (int)p_sizes_.size() * p_hdim_;
    ple_gather(tok, e_.data());
    { const Tensor *w[2] = {p_key_, p_val_}; float *o[2] = {tmp_.data(), pv_.data()}; matvec_n(w, 2, e_.data(), o, ne); }
    ple_mix(tmp_.data(), pv_.data(), h_.data());
  }
  void ple_gather(int tok, float *e) {   // table rows of the n-grams ending at tok (advances the token history)
    const uint64_t sh[3] = {(uint64_t)tok, (uint64_t)p1_, (uint64_t)(p1_ == eos_ ? eos_ : p2_)};   // never reach across <eos>
    int j = 0;
    for (int ng = 2; ng <= 3; ng++) {
      uint64_t hh = sh[0] * p_mult_[0];
      for (int s = 1; s < ng; s++) hh ^= sh[s] * p_mult_[s];
      for (int k = (ng - 2) * p_hpn_; k < (ng - 1) * p_hpn_; k++, j++) row_f32(*p_tab_, (int64_t)(hh % (uint64_t)p_sizes_[k]) + p_offs_[k], e + j * p_hdim_);
    }
    p2_ = p1_; p1_ = tok;
  }
  void ple_mix(const float *key, const float *pv, float *h) {   // key: raw key projection (HD), pv: value (D); h += PLE term
    const int D = d_, HC = hc_, HD = HC * D;
    rmsn(key, p_nk_->f32(), HD, D, true, eps_, pk_.data());
    rmsn(h, p_nq_->f32(), HD, D, true, eps_, pq_.data());
    for (int c = 0; c < HC; c++) {
      float s = 0; for (int i = 0; i < D; i++) s += pk_[c * D + i] * pq_[c * D + i];
      s /= std::sqrt((float)D); s = std::sqrt(std::max(std::fabs(s), 1e-6f)) * (s > 0 ? 1.f : s < 0 ? -1.f : 0.f);
      const float gg = sigm(s); for (int i = 0; i < D; i++) gv_[c * D + i] = gg * pv[i];
    }
    float *win = state_.data() + ple_win_;   // (span+1) x HD, newest last; zeros before the start = conv zero padding
    memmove(win, win + HD, 4ull * p_span_ * HD);
    rmsn(gv_.data(), p_nc_->f32(), HD, D, true, eps_, win + (size_t)p_span_ * HD);
    const float *cw = p_conv_->f32(); const int K = p_K_;
    for (int ch = 0; ch < HD; ch++) { float s = 0; for (int jj = 0; jj < K; jj++) s += cw[ch * K + jj] * win[(size_t)(p_span_ - (K - 1 - jj) * p_dil_) * HD + ch]; pq_[ch] = s; }
    sigmoid_v(pq_.data(), pk_.data(), HD);
    for (int ch = 0; ch < HD; ch++) h[ch] += gv_[ch] + pq_[ch] * pk_[ch];   // silu
  }

  // Gated Residual for B tokens: H [B, HD] -> MIX [B, D] (+ INJ [B, HC])
  void gres_batch(const GR &g, bool inject, int B) {
    const int D = d_, HC = hc_, HD = HC * D;
    pool().parallel_for(B, [&](int a, int b) { for (int i = a; i < b; i++) rmsn(bH_.data() + (size_t)i * HD, g.norm->f32(), HD, D, true, eps_, bHN_.data() + (size_t)i * HD); }, 4);
    matmul(*g.down, bHN_.data(), B, HD, bLOW_.data(), rank_, rank_, HD);
    for (size_t i = 0; i < (size_t)B * rank_; i++) bLOW_[i] = silu(bLOW_[i] / HC);
    matmul(*g.up, bLOW_.data(), B, rank_, bGATE_.data(), HD, HD, rank_);
    pool().parallel_for(B, [&](int a, int b) {
      for (int i = a; i < b; i++) {
        float *gt = bGATE_.data() + (size_t)i * HD; const float *hn = bHN_.data() + (size_t)i * HD; float *mx = bMIX_.data() + (size_t)i * D;
        sigmoid_v(gt, gt, HD);
        for (int j = 0; j < D; j++) { float s = 0; for (int c = 0; c < HC; c++) s += gt[c * D + j] * hn[c * D + j]; mx[j] = s / HC; }
      }
    }, 4);
    if (inject) { matmul(*g.inj, bHN_.data(), B, HD, bINJ_.data(), HC, HC, HD); for (size_t i = 0; i < (size_t)B * HC; i++) bINJ_[i] = 2.f * sigm(bINJ_[i] / HC); }
  }
  void write_batch(int B) {   // H_i += Y_i * INJ_i per stream
    const int D = d_, HC = hc_;
    for (int i = 0; i < B; i++) { float *h = bH_.data() + (size_t)i * HC * D; const float *y = bY_.data() + (size_t)i * D, *inj = bINJ_.data() + (size_t)i * HC;
      for (int c = 0; c < HC; c++) for (int j = 0; j < D; j++) h[c * D + j] += y[j] * inj[c]; }
  }

  int d_, hc_, L_, rank_, ff_, V_, nk_, nv_, dk_, dv_, K_, nh_, nkv_, hd_, rot_, ctx_, eos_, ple_layer_ = -1; float eps_;
  int pos_ = 0, p1_ = 0, p2_ = 0, ple_n_ = 0;
  std::vector<double> inv_;
  const Tensor *embed_, *head_, *decide_w_ = nullptr, *decide_b_ = nullptr; GR fin_;
  std::vector<Layer> layers_;
  std::vector<int64_t> p_sizes_, p_offs_; std::vector<uint64_t> p_mult_;
  const Tensor *p_tab_ = nullptr, *p_key_, *p_val_, *p_nk_, *p_nq_, *p_nc_, *p_conv_; int p_hpn_ = 0, p_hdim_ = 0, p_dil_ = 1, p_K_ = 1, p_span_ = 0; size_t ple_win_ = 0;
  std::vector<float> state_;
  std::vector<float> bH_, bHN_, bLOW_, bGATE_, bMIX_, bINJ_, bY_, bE_, bTK_, bPV_, bQA_, bKA_, bVA_, bAO_, bQKVZ_, bBA_, bO_, bGU_, bACT_;   // prompt batch
  std::vector<float> h_, hn_, low_, gate_, mixed_, inj_, y_, gu_, act_, qkvz_, ba_, xc_, qh_, kh_, o_, qa_, ka_, va_, ao_, e_, pk_, pv_, pq_, gv_, tmp_, fin_v_, logits_;
};

}  // namespace

std::unique_ptr<Model> make_nq(std::shared_ptr<Container> c, int max_ctx) { return std::unique_ptr<Model>(new NQ(c, max_ctx)); }

}  // namespace qp
