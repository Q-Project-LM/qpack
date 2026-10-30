// Chat formats and the generation Session.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include "qpack/engine.h"

namespace qp {

// ------------------------------------------------------------------ chat formats
ChatFormat::ChatFormat(const Json &cfg, const Tokenizer &tok) {
  const std::string tmpl = cfg["chat_template"].as_str();
  if (cfg["model_type"].as_str() == "nq4" || tok.token_id("<MODEL>") >= 0) name_ = "nq";
  else if (cfg["chat_format"].as_str() == "chatml" || tok.token_id("<|im_start|>") >= 0) name_ = "chatml";
  else name_ = "raw";
  // templates that open with {{ bos_token }} (MiniCPM, Llama 3, ...): the model expects the BOS token in front of the conversation
  if (name_ == "chatml" && tmpl.compare(0, 2, "{{") == 0 && tmpl.substr(0, 24).find("bos_token") != std::string::npos) {
    const Json &b = cfg["bos_token_id"]; const int id = b.is_null() ? -1 : (int)b.as_int(-1);
    if (id >= 0 && tok.is_special(id)) bos_ = tok.piece(id);
  }
  // hybrid-reasoning templates (MiniCPM5, LFM3): what the template appends after "assistant\n" with enable_thinking off / on
  if (name_ == "chatml" && tok.token_id("<think>") >= 0 && tmpl.find("enable_thinking") != std::string::npos) {
    if (tmpl.find("<think>\\n\\n</think>\\n\\n") != std::string::npos) { think_off_ = "<think>\n\n</think>\n\n"; think_on_ = "<think>\n"; }
    else if (tmpl.find("<think></think>") != std::string::npos) { think_off_ = "<think></think>"; think_on_ = "<think>"; }
  }
  const std::string qwen = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
  if (tmpl.find(qwen) != std::string::npos) default_system_ = qwen;
  auto add = [&](int id) { if (id >= 0 && !is_stop(id)) stop_.push_back(id); };
  const Json &eos = cfg["eos_token_id"];
  if (eos.type == Json::Array) for (auto &e : eos.arr) add((int)e.as_int()); else if (!eos.is_null()) add((int)eos.as_int());
  for (const char *t : {"<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>", "<EOT>", "<eos>", "</s>"}) add(tok.token_id(t));
  if (name_ == "nq") { think_ = tok.token_id("<THINK>"); ethink_ = tok.token_id("<ETHINK>"); eq_ = tok.token_id("<EQ>"); }
}
std::string ChatFormat::display(int id, const Tokenizer &tok) const {
  if (!tok.is_special(id)) return tok.piece(id);
  if (id == think_) return "<think>";
  if (id == ethink_) return "</think>";
  if (id == eq_) return " = ";
  return "";
}
bool ChatFormat::is_stop(int id) const { for (int s : stop_) if (s == id) return true; return false; }

std::string ChatFormat::render(const std::vector<Message> &msgs, bool gen, bool think) const {
  std::string o;
  if (name_ == "nq") {   // <bos>[<SYSTEM>..<EOT>]<USER>..<EOT><MODEL>..<EOT>  (nq/dataset/chat.py)
    o = "<bos>";
    for (auto &m : msgs) o += (m.role == "system" ? "<SYSTEM>" : m.role == "assistant" ? "<MODEL>" : "<USER>") + m.content + "<EOT>";
    if (gen) { o += "<MODEL>"; if (think) o += "<THINK>"; }
    return o;
  }
  if (name_ == "chatml") {
    o = bos_;
    if (!default_system_.empty() && (msgs.empty() || msgs[0].role != "system")) o += "<|im_start|>system\n" + default_system_ + "<|im_end|>\n";
    for (auto &m : msgs) o += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
    if (gen) { o += "<|im_start|>assistant\n"; o += think ? think_on_ : think_off_; }
    return o;
  }
  for (auto &m : msgs) o += m.content + "\n";
  return o;
}

// ------------------------------------------------------------------ session
Session::Session(std::unique_ptr<Model> m, int) : m_(std::move(m)) {
  tok_.reset(new Tokenizer(m_->container().json("tokenizer.json")));
  chat_.reset(new ChatFormat(m_->config(), *tok_));
  if (chat_->name() == "nq") circ_.reset(new CircuitExecutor(*tok_));
}

const float *Session::eval(const std::vector<int> &ids, int *reused) {
  if (ids.empty()) throw std::runtime_error("empty prompt");
  if ((int)ids.size() > m_->max_ctx()) throw std::runtime_error("prompt (" + std::to_string(ids.size()) + " tokens) exceeds the context window (" + std::to_string(m_->max_ctx()) + ")");
  size_t common = 0; while (common < hist_.size() && common < ids.size() && hist_[common] == ids[common]) common++;
  if (common == ids.size()) common--;   // the last prompt token must be evaluated to get logits
  if (common < hist_.size() && !m_->rewind((int)common)) {
    // a recurrent state cannot be cut back: restart from the saved state of an earlier prompt if it is a prefix
    if (!snap_.empty() && snap_ids_.size() < ids.size() && std::equal(snap_ids_.begin(), snap_ids_.end(), ids.begin())) { m_->restore(snap_); common = snap_ids_.size(); }
    else { m_->reset(); common = 0; }
  }
  if (common == 0 && m_->pos() != 0) m_->reset();
  hist_.assign(ids.begin(), ids.begin() + common); if (reused) *reused = (int)common;
  if (common + 1 < ids.size()) { m_->prefill(ids.data() + common, (int)(ids.size() - 1 - common), 0); hist_.insert(hist_.end(), ids.begin() + common, ids.end() - 1); }
  if (m_->recurrent() && !hist_.empty() && hist_ != snap_ids_) { snap_ = m_->snapshot(); snap_ids_ = hist_; }
  const float *lg = m_->step(ids.back(), 1); hist_.push_back(ids.back());
  return lg;
}

void Session::set_draft(std::unique_ptr<Model> draft, int k) {
  if (draft && draft->vocab() != m_->vocab()) throw std::runtime_error("draft model: vocabulary differs from the main model");
  draft_ = std::move(draft); draft_k_ = draft_ ? k : 0; dhist_.clear(); dck_.clear(); dck_len_ = 0;
}

// prompt lookup: the tokens that followed the most recent earlier occurrence of the last n-gram (n = max .. min)
std::vector<int> Session::propose(const std::vector<int> &seq, int k) {
  std::vector<int> out;
  if (lookup_k_ > 0) {
    const int L = (int)seq.size();
    for (int n = ngram_max_; n >= ngram_min_ && out.empty(); n--) {
      if (L <= n) continue;
      for (int j = L - n - 1; j >= 0; j--) {   // occurrence seq[j .. j+n) == last n tokens, with something after it
        bool eq = true; for (int i = 0; i < n && eq; i++) eq = seq[j + i] == seq[L - n + i];
        if (!eq) continue;
        for (int i = j + n; i < L && (int)out.size() < std::min(k, lookup_k_); i++) out.push_back(seq[i]);
        break;
      }
    }
  }
  if (out.empty() && draft_) out = propose_draft(seq, std::min(k, draft_k_));
  return out;
}

// draft model: bring its state to `seq` (roll back its own guesses from the previous round), then k greedy tokens
std::vector<int> Session::propose_draft(const std::vector<int> &seq, int k) {
  Model &d = *draft_;
  size_t common = 0; while (common < dhist_.size() && common < seq.size() && dhist_[common] == seq[common]) common++;
  if (common < dhist_.size()) {
    if (!d.rewind((int)common)) {
      if (!dck_.empty() && dck_len_ <= common) { d.rollback(dck_); dhist_.resize(dck_len_); common = dck_len_; }
      else { d.reset(); dhist_.clear(); common = 0; }
    } else dhist_.resize(common);
  }
  if (common == seq.size()) {   // nothing new: re-feed the last token to get logits
    if (!d.rewind((int)common - 1)) { d.reset(); dhist_.clear(); common = 0; } else { dhist_.pop_back(); common--; }
  }
  if (common + 1 < seq.size()) d.prefill(seq.data() + common, (int)(seq.size() - 1 - common), 0);
  if (seq.size() > (size_t)d.max_ctx() - k - 1) return {};
  dck_ = d.checkpoint(); dck_len_ = seq.size() - 1;
  const float *lg = d.step(seq.back(), 1); dhist_ = seq;
  std::vector<int> out;
  for (int i = 0; i < k; i++) {
    const int t = (int)(std::max_element(lg, lg + d.vocab()) - lg);
    out.push_back(t);
    if (i + 1 < k) { lg = d.step(t, 1); dhist_.push_back(t); }
  }
  return out;
}

GenResult Session::generate(const std::vector<int> &prompt, int max_new, Sampler &s, const std::function<bool(const std::string &)> &on_text,
                            const std::vector<std::string> &stop_strings) {
  GenResult r; r.prompt_tokens = (int)prompt.size();
  auto t0 = std::chrono::steady_clock::now();
  const float *lg = eval(prompt, &r.reused);
  auto t1 = std::chrono::steady_clock::now(); r.prefill_s = std::chrono::duration<double>(t1 - t0).count();
  std::string bytes; size_t sent = 0;   // decoded bytes so far; `sent` = prefix already handed to on_text
  r.finish = "length";
  if (circ_) { circ_->clear(); circ_->calls = circ_->failed = 0; }
  const int V = m_->vocab();
  // one generated token: false = stop (stop token, stop string, on_text abort, max_new or context reached)
  auto emit = [&](int t, int pos_before) {   // pos_before: model position before t is fed
    if (chat_->is_stop(t)) { r.finish = "stop"; return false; }
    r.ids.push_back(t);
    bytes += chat_->display(t, *tok_);
    bool stop_hit = false;
    for (auto &ss : stop_strings) { size_t f = bytes.find(ss, sent > ss.size() ? sent - ss.size() : 0); if (!ss.empty() && f != std::string::npos) { bytes.resize(f); stop_hit = true; break; } }
    // hand out only complete UTF-8 characters, holding back a possible stop-string prefix
    size_t end = bytes.size();
    for (size_t k = 1; k <= 3 && k <= end - sent; k++) {
      unsigned char c = bytes[end - k];
      if ((c & 0xC0) == 0x80) continue;                                     // continuation byte: look further back
      size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
      if (need > k) end -= k;                                                // incomplete trailing character
      break;
    }
    if (!stop_hit) for (auto &ss : stop_strings) for (size_t k = std::min(ss.size(), end - sent + 1) - 1; k > 0 && !ss.empty(); k--) if (bytes.compare(end - k, k, ss, 0, k) == 0) { end -= k; break; }
    if (end > sent && on_text) { if (!on_text(bytes.substr(sent, end - sent))) { sent = end; r.finish = "stop"; return false; } }
    if (end > sent) sent = end;
    if (stop_hit) { r.finish = "stop"; return false; }
    if ((int)r.ids.size() >= max_new) { r.finish = "length"; return false; }
    if (pos_before + 1 > m_->max_ctx()) { r.finish = "length"; return false; }
    return true;
  };
  std::vector<std::pair<int, float>> dist;
  auto next_token = [&](const float *l, size_t hist_len) {   // circuit result / <ECALC> is forced, not sampled
    const int f = circ_ ? circ_->next(hist_) : -1;
    if (f >= 0) return f;
    s.dist(l, V, hist_, dist, hist_len); return s.draw(dist);
  };
  const bool spec = lookup_k_ > 0 || draft_;
  const int eq = tok_->token_id("<EQ>"), calc = tok_->token_id("<CALC>"), ecalc = tok_->token_id("<ECALC>");
  int t = max_new > 0 ? next_token(lg, hist_.size()) : -1;
  // adaptive draft length: doubled after a fully accepted pass, cut to (accepted + 1) after a rejection
  const int max_k = std::max(lookup_k_, draft_k_); int cur_k = std::min(4, max_k);
  while (t >= 0 && emit(t, m_->pos())) {
    std::vector<int> drafts;
    if (spec && !(circ_ && (circ_->busy() || t == eq))) {
      std::vector<int> seq = hist_; seq.push_back(t);
      const int room = std::min(max_new - (int)r.ids.size(), m_->max_ctx() - m_->pos() - 1);
      drafts = propose(seq, std::min(cur_k, std::min(room, 15)));
      for (size_t i = 0; i < drafts.size(); i++)   // circuit spans are left to the executor
        if (drafts[i] == eq || drafts[i] == calc || drafts[i] == ecalc || chat_->is_stop(drafts[i])) { drafts.resize(i); break; }
    }
    if (drafts.empty()) { lg = m_->step(t, 1); hist_.push_back(t); t = next_token(lg, hist_.size()); continue; }
    // verify t + drafts in one batched pass; logits row i = distribution after seq[i]
    const int k = (int)drafts.size(); const size_t base = hist_.size();
    std::vector<int> seq{t}; seq.insert(seq.end(), drafts.begin(), drafts.end());
    if (vlog_.size() < (size_t)(k + 1) * V) vlog_.resize((size_t)(k + 1) * V);
    const auto ck = m_->checkpoint();
    { ExactBatches ex; m_->prefill_all(seq.data(), k + 1, vlog_.data()); }
    hist_.insert(hist_.end(), seq.begin(), seq.end());
    r.drafted += k; r.verify_passes++;
    int acc = 0, next = -1; bool stopped = false;
    for (int i = 0; i < k; i++) {
      s.dist(vlog_.data() + (size_t)i * V, V, hist_, dist, base + 1 + i);
      int repl = -1;
      if (!s.accept(dist, drafts[i], &repl)) { next = repl; break; }
      acc++;
      if (!emit(drafts[i], (int)base + 1 + i)) { stopped = true; break; }
    }
    r.accepted += acc;
    cur_k = acc == k ? std::min(max_k, 2 * cur_k) : std::max(2, std::min(max_k, acc + 1));
    const int keep = stopped ? acc : 1 + acc;   // a token that ended the reply is not fed (as in plain decoding)
    if (keep < k + 1) {   // drop the rejected tail from the model state
      if (!m_->rewind((int)base + keep)) { ExactBatches ex; m_->rollback(ck); m_->prefill(seq.data(), keep, 0); }
      hist_.resize(base + keep);
    }
    if (stopped) break;
    if (next < 0) next = next_token(vlog_.data() + (size_t)k * V, hist_.size());   // all accepted: the token after the last draft
    t = next;
  }
  if (sent < bytes.size() && on_text) on_text(bytes.substr(sent));
  if (circ_) { r.circuit_calls = circ_->calls; r.circuit_failed = circ_->failed; }
  r.raw = tok_->decode(r.ids, false);
  r.text = bytes; r.decode_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
  return r;
}

GenResult Session::chat_turn(const std::vector<Message> &msgs_in, int max_new, Sampler &s, const std::function<bool(const std::string &)> &on_text,
                             bool think) {
  std::vector<Message> msgs = msgs_in; std::vector<int> ids;
  for (;;) {   // drop the oldest turns (keeping a system message) until prompt + reply fit the window
    ids = tok_->encode(chat_->render(msgs, true, think));
    if ((int)ids.size() + std::min(max_new, m_->max_ctx() / 4) <= m_->max_ctx()) break;
    size_t first = !msgs.empty() && msgs[0].role == "system" ? 1 : 0;
    if (msgs.size() <= first + 1) break;
    msgs.erase(msgs.begin() + first);
  }
  return generate(ids, std::min(max_new, m_->max_ctx() - (int)ids.size()), s, on_text);
}

double Session::nll(const std::vector<int> &ids, int *count) {
  m_->reset(); hist_.clear(); double tot = 0; int n = 0; const float *lg = nullptr; const int V = m_->vocab();
  for (size_t i = 0; i + 1 < ids.size() && (int)i < m_->max_ctx(); i++) {
    lg = m_->step(ids[i], 1); hist_.push_back(ids[i]);
    float mx = lg[0]; for (int v = 1; v < V; v++) mx = std::max(mx, lg[v]);
    double z = 0; for (int v = 0; v < V; v++) z += std::exp((double)lg[v] - mx);
    tot += mx + std::log(z) - lg[ids[i + 1]]; n++;
  }
  if (count) *count = n;
  return n ? tot / n : 0;
}

}  // namespace qp
