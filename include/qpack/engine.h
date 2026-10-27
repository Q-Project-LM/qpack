// Text generation on top of a Model: sampler, chat formats, and a Session that keeps the token history so a new turn
// only evaluates the tokens after the longest common prefix (KV cache reuse; snapshots for recurrent models).
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "qpack/circuits.h"
#include "qpack/qpack.h"
#include "qpack/tokenizer.h"

namespace qp {

struct SamplerParams {
  float temperature = 0.7f;   // <= 0: greedy
  int top_k = 40;             // 0: off
  float top_p = 0.9f;         // 1: off
  float min_p = 0.05f;        // 0: off (relative to the top probability)
  float repeat_penalty = 1.1f;
  int repeat_last_n = 64;
  uint64_t seed = 0;          // 0: random
};

class Sampler {
 public:
  explicit Sampler(const SamplerParams &p);
  int sample(const float *logits, int vocab, const std::vector<int> &history) { dist(logits, vocab, history, tmp_); return draw(tmp_); }
  // the distribution sample() draws from (after repetition penalty, temperature, top-k / top-p / min-p), as
  // (token, probability) pairs summing to 1, most likely first; greedy = one token with probability 1. `hist_len` limits
  // the history to its first entries (speculative verification looks at earlier positions).
  void dist(const float *logits, int vocab, const std::vector<int> &history, std::vector<std::pair<int, float>> &out, size_t hist_len = (size_t)-1);
  int draw(const std::vector<std::pair<int, float>> &d);
  // speculative step for a proposed token: accept with probability p(token); on rejection *repl is drawn from p with
  // the token removed — the result has exactly the distribution of a plain draw (standard rule for a one-hot proposal)
  bool accept(const std::vector<std::pair<int, float>> &d, int token, int *repl);
  const SamplerParams &params() const { return p_; }
 private:
  SamplerParams p_; std::mt19937_64 rng_;
  std::vector<int> pen_; std::vector<std::pair<int, float>> tmp_;
};

struct Message { std::string role, content; };

// Renders conversations for the model's chat format: "chatml" (Qwen, SmolLM, ...) or "nq" (N-Q control tokens).
class ChatFormat {
 public:
  ChatFormat(const Json &cfg, const Tokenizer &tok);
  std::string name() const { return name_; }
  std::string render(const std::vector<Message> &msgs, bool add_generation_prompt, bool think = false) const;
  const std::vector<int> &stop_ids() const { return stop_; }
  bool is_stop(int id) const;
  // text shown for a generated token: N-Q control tokens become "<think>", "</think>", " = " (circuit result), ""
  std::string display(int id, const Tokenizer &tok) const;
 private:
  std::string name_, default_system_; std::vector<int> stop_;
  int think_ = -1, ethink_ = -1, eq_ = -1;
};

struct GenResult {
  std::vector<int> ids; std::string finish = "stop";
  std::string text, raw;   // text: for display; raw: with control tokens (to put back into a chat history)
  int prompt_tokens = 0, reused = 0, circuit_calls = 0, circuit_failed = 0; double prefill_s = 0, decode_s = 0;
  int drafted = 0, accepted = 0, verify_passes = 0;   // speculative decoding
};

class Session {
 public:
  Session(std::unique_ptr<Model> m, int max_ctx_hint = 0);
  Model &model() { return *m_; }
  const Tokenizer &tok() const { return *tok_; }
  const ChatFormat &chat() const { return *chat_; }
  // evaluate `prompt` (reusing the cached prefix), then sample up to max_new tokens; on_text gets UTF-8-safe deltas and
  // may return false to stop early
  GenResult generate(const std::vector<int> &prompt, int max_new, Sampler &s, const std::function<bool(const std::string &)> &on_text = nullptr,
                     const std::vector<std::string> &stop_strings = {});
  GenResult chat_turn(const std::vector<Message> &msgs, int max_new, Sampler &s, const std::function<bool(const std::string &)> &on_text = nullptr,
                      bool think = false);
  // speculative decoding: up to k tokens proposed per step by prompt lookup (the continuation of the latest earlier
  // occurrence of the last n-gram in the context) and/or a draft model with the same tokenizer; all of them are checked
  // in one batched pass of the model. Greedy output is identical to plain decoding, sampled output has the same
  // distribution. k = 0 turns it off.
  void set_lookup(int k, int ngram_max = 3, int ngram_min = 2) { lookup_k_ = k; ngram_max_ = ngram_max; ngram_min_ = ngram_min; }
  void set_draft(std::unique_ptr<Model> draft, int k);
  // mean negative log-likelihood of ids[1..] given the prefix (perplexity), evaluated from an empty state
  double nll(const std::vector<int> &ids, int *count = nullptr);
 private:
  const float *eval(const std::vector<int> &ids, int *reused);
  std::unique_ptr<Model> m_; std::unique_ptr<Tokenizer> tok_; std::unique_ptr<ChatFormat> chat_; std::unique_ptr<CircuitExecutor> circ_;
  std::vector<int> hist_;   // tokens currently in the model state
  std::vector<int> snap_ids_; std::vector<uint8_t> snap_;   // recurrent models: state after a previous prompt minus its last token
  std::vector<int> propose(const std::vector<int> &seq, int k);
  std::vector<int> propose_draft(const std::vector<int> &seq, int k);
  int lookup_k_ = 0, ngram_max_ = 3, ngram_min_ = 2, draft_k_ = 0;
  std::unique_ptr<Model> draft_; std::vector<int> dhist_; std::vector<uint8_t> dck_; size_t dck_len_ = 0;
  std::vector<float> vlog_;   // verification logits
};

}  // namespace qp
