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
    if (!default_system_.empty() && (msgs.empty() || msgs[0].role != "system")) o += "<|im_start|>system\n" + default_system_ + "<|im_end|>\n";
    for (auto &m : msgs) o += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
    if (gen) o += "<|im_start|>assistant\n";
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

GenResult Session::generate(const std::vector<int> &prompt, int max_new, Sampler &s, const std::function<bool(const std::string &)> &on_text,
                            const std::vector<std::string> &stop_strings) {
  GenResult r; r.prompt_tokens = (int)prompt.size();
  auto t0 = std::chrono::steady_clock::now();
  const float *lg = eval(prompt, &r.reused);
  auto t1 = std::chrono::steady_clock::now(); r.prefill_s = std::chrono::duration<double>(t1 - t0).count();
  std::string bytes; size_t sent = 0;   // decoded bytes so far; `sent` = prefix already handed to on_text
  r.finish = "length";
  if (circ_) { circ_->clear(); circ_->calls = circ_->failed = 0; }
  for (int n = 0; n < max_new; n++) {
    int t = circ_ ? circ_->next(hist_) : -1;   // a circuit result / <ECALC> is forced, not sampled
    if (t < 0) t = s.sample(lg, m_->vocab(), hist_);
    if (chat_->is_stop(t)) { r.finish = "stop"; break; }
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
    if (end > sent && on_text) { if (!on_text(bytes.substr(sent, end - sent))) { sent = end; r.finish = "stop"; break; } }
    if (end > sent) sent = end;
    if (stop_hit) { r.finish = "stop"; break; }
    if (m_->pos() >= m_->max_ctx()) { r.finish = "length"; break; }
    lg = m_->step(t, 1); hist_.push_back(t);
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
