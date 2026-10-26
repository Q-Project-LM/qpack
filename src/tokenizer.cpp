#include "qpack/tokenizer.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <stdexcept>

#include "unicode_nfc.inc"

namespace qp {

// ------------------------------------------------------------------ UTF-8
std::vector<uint32_t> utf8_decode(const std::string &s) {
  std::vector<uint32_t> o; o.reserve(s.size());
  for (size_t i = 0; i < s.size();) {
    unsigned char c = s[i]; uint32_t cp; int n;
    if (c < 0x80) { cp = c; n = 1; } else if ((c >> 5) == 6) { cp = c & 31; n = 2; } else if ((c >> 4) == 14) { cp = c & 15; n = 3; } else if ((c >> 3) == 30) { cp = c & 7; n = 4; } else { cp = 0xFFFD; n = 1; }
    if (i + n > s.size()) { o.push_back(0xFFFD); break; }
    for (int k = 1; k < n; k++) cp = (cp << 6) | (s[i + k] & 63);
    o.push_back(cp); i += n;
  }
  return o;
}
std::string utf8_encode(const std::vector<uint32_t> &cps) {
  std::string o;
  for (uint32_t cp : cps) {
    if (cp < 0x80) o += (char)cp;
    else if (cp < 0x800) { o += (char)(0xC0 | (cp >> 6)); o += (char)(0x80 | (cp & 63)); }
    else if (cp < 0x10000) { o += (char)(0xE0 | (cp >> 12)); o += (char)(0x80 | ((cp >> 6) & 63)); o += (char)(0x80 | (cp & 63)); }
    else { o += (char)(0xF0 | (cp >> 18)); o += (char)(0x80 | ((cp >> 12) & 63)); o += (char)(0x80 | ((cp >> 6) & 63)); o += (char)(0x80 | (cp & 63)); }
  }
  return o;
}

// ------------------------------------------------------------------ NFC (canonical decomposition, reordering, composition)
static int ccc(uint32_t cp) {
  int lo = 0, hi = kCcc_N - 1;
  while (lo <= hi) { int m = (lo + hi) / 2; if (kCcc[m].cp == cp) return kCcc[m].c; if (kCcc[m].cp < cp) lo = m + 1; else hi = m - 1; }
  return 0;
}
static const UDecomp *find_decomp(uint32_t cp) {
  int lo = 0, hi = kDecomp_N - 1;
  while (lo <= hi) { int m = (lo + hi) / 2; if (kDecomp[m].cp == cp) return &kDecomp[m]; if (kDecomp[m].cp < cp) lo = m + 1; else hi = m - 1; }
  return nullptr;
}
static uint32_t compose(uint32_t a, uint32_t b) {
  const uint32_t SB = 0xAC00, LB = 0x1100, VB = 0x1161, TB = 0x11A7, LC = 19, VC = 21, TC = 28, NC = VC * TC;
  if (a >= LB && a < LB + LC && b >= VB && b < VB + VC) return SB + ((a - LB) * VC + (b - VB)) * TC;   // Hangul LV
  if (a >= SB && a < SB + LC * NC && (a - SB) % TC == 0 && b > TB && b < TB + TC) return a + (b - TB);   // Hangul LVT
  int lo = 0, hi = kComp_N - 1;
  while (lo <= hi) { int m = (lo + hi) / 2; const UComp &c = kComp[m]; if (c.a == a && c.b == b) return c.cp; if (c.a < a || (c.a == a && c.b < b)) lo = m + 1; else hi = m - 1; }
  return 0;
}
static void decompose(uint32_t cp, std::vector<uint32_t> &out) {
  const uint32_t SB = 0xAC00, LB = 0x1100, VB = 0x1161, TB = 0x11A7, VC = 21, TC = 28, NC = VC * TC, SC = 19 * NC;
  if (cp >= SB && cp < SB + SC) { uint32_t s = cp - SB; out.push_back(LB + s / NC); out.push_back(VB + (s % NC) / TC); if (s % TC) out.push_back(TB + s % TC); return; }
  const UDecomp *d = find_decomp(cp);
  if (!d) { out.push_back(cp); return; }
  decompose(d->a, out); if (d->b) decompose(d->b, out);
}
std::string nfc(const std::string &s) {
  bool ascii = true; for (unsigned char c : s) if (c >= 0x80) { ascii = false; break; }
  if (ascii) return s;
  std::vector<uint32_t> d; for (uint32_t cp : utf8_decode(s)) decompose(cp, d);
  for (size_t i = 1; i < d.size(); i++) {   // canonical ordering: stable sort of runs of non-starters by class
    int c = ccc(d[i]); if (!c) continue;
    size_t j = i; while (j > 0 && ccc(d[j - 1]) > c) { std::swap(d[j], d[j - 1]); j--; }
  }
  std::vector<uint32_t> o; int starter = -1, last_cc = 0;
  for (uint32_t cp : d) {
    int c = ccc(cp);
    if (starter >= 0) {   // not blocked: adjacent to the starter, or the previous mark has a lower non-zero class
      bool adj = (int)o.size() - 1 == starter;
      if (adj || (last_cc != 0 && last_cc < c)) { uint32_t comp = compose(o[starter], cp); if (comp) { o[starter] = comp; continue; } }
    }
    if (c == 0) starter = (int)o.size();
    last_cc = c; o.push_back(cp);
  }
  return utf8_encode(o);
}

// ------------------------------------------------------------------ regex
namespace {
struct Node {
  enum T { Char, Class, Any, Seq, Alt, Rep, NegLook } t = Seq;
  uint32_t ch = 0; bool icase = false;
  struct Item { int kind; uint32_t a, b; };   // kind: 0 range, 1 \p{L}, 2 \p{N}, 3 \s, 4 \S, 5 \d, 6 \w, 7 \P{L}, 8 \P{N}
  std::vector<Item> items; bool neg = false;
  std::vector<int> kids; int mn = 0, mx = 0; bool greedy = true;
};
using Cont = std::function<bool(size_t)>;
}  // namespace

struct Regex::Impl {
  std::vector<Node> n; int root = -1;
  std::vector<uint32_t> p; size_t i = 0;
  int add(Node x) { n.push_back(std::move(x)); return (int)n.size() - 1; }
  [[noreturn]] void fail(const std::string &m) { throw std::runtime_error("regex: " + m); }
  bool eof() const { return i >= p.size(); }
  uint32_t at(size_t k) const { return k < p.size() ? p[k] : 0; }
  static uint32_t esc_char(uint32_t c) { switch (c) { case 'n': return '\n'; case 'r': return '\r'; case 't': return '\t'; case 'f': return '\f'; case 'v': return '\v'; default: return c; } }
  Node::Item escape_item() {   // after '\'
    uint32_t c = p[i++]; Node::Item it{0, 0, 0};
    if (c == 'p' || c == 'P') {
      std::string name; if (at(i) == '{') { i++; while (!eof() && p[i] != '}') name += (char)p[i++]; i++; } else name = (char)p[i++];
      if (name == "L") it.kind = c == 'P' ? 7 : 1; else if (name == "N") it.kind = c == 'P' ? 8 : 2; else fail("unsupported \\p{" + name + "}");
      return it;
    }
    if (c == 's') it.kind = 3; else if (c == 'S') it.kind = 4; else if (c == 'd') it.kind = 5; else if (c == 'w') it.kind = 6;
    else it.a = it.b = esc_char(c);
    return it;
  }
  int atom(bool icase) {
    uint32_t c = p[i];
    if (c == '(') {
      i++; bool ic = icase, look = false;
      if (at(i) == '?') {
        i++;
        if (at(i) == ':') i++;
        else if (at(i) == 'i' && at(i + 1) == ':') { ic = true; i += 2; }
        else if (at(i) == '!') { look = true; i++; }
        else fail("unsupported group");
      }
      int x = alt(ic); if (at(i) != ')') fail("missing )"); i++;
      if (look) { Node nl; nl.t = Node::NegLook; nl.kids = {x}; return add(nl); }
      return x;
    }
    if (c == '[') {
      i++; Node cl; cl.t = Node::Class; cl.icase = icase;
      if (at(i) == '^') { cl.neg = true; i++; }
      bool first = true;
      while (!eof() && (p[i] != ']' || first)) {
        first = false; Node::Item it{0, 0, 0};
        if (p[i] == '\\') { i++; it = escape_item(); } else it.a = it.b = p[i++];
        if (it.kind == 0 && at(i) == '-' && at(i + 1) != ']' && i + 1 < p.size()) {
          i++; uint32_t hi; if (p[i] == '\\') { i++; hi = esc_char(p[i++]); } else hi = p[i++];
          it.b = hi;
        }
        cl.items.push_back(it);
      }
      if (eof()) fail("missing ]"); i++;
      return add(cl);
    }
    if (c == '.') { i++; Node a; a.t = Node::Any; return add(a); }
    if (c == '\\') {
      i++; Node::Item it = escape_item(); Node x;
      if (it.kind == 0) { x.t = Node::Char; x.ch = it.a; x.icase = icase; } else { x.t = Node::Class; x.items = {it}; }
      return add(x);
    }
    i++; Node x; x.t = Node::Char; x.ch = c; x.icase = icase; return add(x);
  }
  int repeat(bool icase) {
    int a = atom(icase);
    while (!eof()) {
      uint32_t c = p[i]; int mn, mx;
      if (c == '?') { mn = 0; mx = 1; } else if (c == '*') { mn = 0; mx = -1; } else if (c == '+') { mn = 1; mx = -1; }
      else if (c == '{') {
        size_t j = i + 1; std::string s; while (j < p.size() && p[j] != '}') s += (char)p[j++];
        auto comma = s.find(',');
        mn = std::stoi(s.substr(0, comma)); mx = comma == std::string::npos ? mn : (comma + 1 < s.size() ? std::stoi(s.substr(comma + 1)) : -1);
        i = j;
      } else break;
      i++; Node r; r.t = Node::Rep; r.kids = {a}; r.mn = mn; r.mx = mx;
      if (at(i) == '?') { r.greedy = false; i++; } else if (at(i) == '+') i++;   // possessive treated as greedy
      a = add(r);
    }
    return a;
  }
  int seq(bool icase) { Node s; s.t = Node::Seq; while (!eof() && p[i] != '|' && p[i] != ')') s.kids.push_back(repeat(icase)); return s.kids.size() == 1 ? s.kids[0] : add(s); }
  int alt(bool icase) { Node a; a.t = Node::Alt; a.kids.push_back(seq(icase)); while (at(i) == '|') { i++; a.kids.push_back(seq(icase)); } return a.kids.size() == 1 ? a.kids[0] : add(a); }

  static uint32_t lower(uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }
  static bool item_has(const Node::Item &it, uint32_t c) {
    switch (it.kind) {
      case 0: return c >= it.a && c <= it.b;
      case 1: return uni_is_letter(c);
      case 2: case 5: return uni_is_number(c);
      case 3: return uni_is_space(c);
      case 4: return !uni_is_space(c);
      case 6: return uni_is_letter(c) || uni_is_number(c) || c == '_';
      case 7: return !uni_is_letter(c);
      case 8: return !uni_is_number(c);
    }
    return false;
  }
  static bool cls(const Node &x, uint32_t c) {
    bool hit = false;
    for (auto &it : x.items) {
      if (item_has(it, c)) { hit = true; break; }
      if (x.icase && c < 128) { uint32_t l = lower(c), u = (l >= 'a' && l <= 'z') ? l - 32 : l; if (item_has(it, l) || item_has(it, u)) { hit = true; break; } }
    }
    return hit != x.neg;
  }
  static bool one(const Node &c, uint32_t ch) {
    switch (c.t) {
      case Node::Class: return cls(c, ch);
      case Node::Any: return ch != '\n';
      case Node::Char: return ch == c.ch || (c.icase && lower(ch) == lower(c.ch));
      default: return false;
    }
  }
  // continuation-passing backtracking matcher: k(pos) is "the rest of the pattern matches from pos"
  bool m(int id, const std::vector<uint32_t> &s, size_t pos, const Cont &k) const {
    const Node &x = n[id];
    switch (x.t) {
      case Node::Char: case Node::Class: case Node::Any: return pos < s.size() && one(x, s[pos]) && k(pos + 1);
      case Node::Seq: return seqm(x, 0, s, pos, k);
      case Node::Alt: for (int c : x.kids) if (m(c, s, pos, k)) return true; return false;
      case Node::NegLook: return !m(x.kids[0], s, pos, [](size_t) { return true; }) && k(pos);
      case Node::Rep: return rep(x, 0, s, pos, k);
    }
    return false;
  }
  bool seqm(const Node &x, size_t idx, const std::vector<uint32_t> &s, size_t pos, const Cont &k) const {
    if (idx == x.kids.size()) return k(pos);
    if (idx + 1 == x.kids.size()) return m(x.kids[idx], s, pos, k);
    return m(x.kids[idx], s, pos, [&](size_t p2) { return seqm(x, idx + 1, s, p2, k); });
  }
  bool rep(const Node &x, int cnt, const std::vector<uint32_t> &s, size_t pos, const Cont &k) const {
    const Node &c = n[x.kids[0]];
    if (c.t == Node::Class || c.t == Node::Char || c.t == Node::Any) {   // single-character repetition: scan the run, then try lengths
      size_t e = pos; int got = 0;
      while (e < s.size() && (x.mx < 0 || got < x.mx) && one(c, s[e])) { e++; got++; }
      if (got < x.mn) return false;
      if (x.greedy) { for (int g = got; g >= x.mn; g--) if (k(pos + g)) return true; }
      else { for (int g = x.mn; g <= got; g++) if (k(pos + g)) return true; }
      return false;
    }
    if (cnt < x.mn) return m(x.kids[0], s, pos, [&](size_t p2) { return rep(x, cnt + 1, s, p2, k); });
    auto more = [&]() { return (x.mx < 0 || cnt < x.mx) && m(x.kids[0], s, pos, [&](size_t p2) { return p2 != pos && rep(x, cnt + 1, s, p2, k); }); };
    if (x.greedy) return more() || k(pos);
    return k(pos) || more();
  }
};

Regex::Regex(const std::string &pattern) : impl_(new Impl) {
  impl_->p = utf8_decode(pattern); impl_->root = impl_->alt(false);
  if (!impl_->eof()) impl_->fail("unexpected ) in " + pattern);
}
Regex::~Regex() = default;
bool Regex::find(const std::vector<uint32_t> &s, size_t from, size_t *b, size_t *e) const {
  for (size_t st = from; st <= s.size(); st++) {
    size_t end = 0;
    if (impl_->m(impl_->root, s, st, [&](size_t p) { end = p; return true; })) { *b = st; *e = end; return true; }
  }
  return false;
}

// ------------------------------------------------------------------ tokenizer
static const char *kGPT2 = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

Tokenizer::Tokenizer(const Json &tj) {
  const Json &model = tj["model"];
  if (model["type"].as_str() != "BPE") throw std::runtime_error("tokenizer model '" + model["type"].as_str() + "' is not supported (byte-level BPE only)");
  ignore_merges_ = model["ignore_merges"].as_bool(false);
  // byte <-> printable char map of GPT-2 byte-level BPE
  std::vector<int> bs, cs; bool used[256] = {};
  for (int b = 0; b < 256; b++) if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) { bs.push_back(b); cs.push_back(b); used[b] = true; }
  int extra = 0; for (int b = 0; b < 256; b++) if (!used[b]) { bs.push_back(b); cs.push_back(256 + extra++); }
  for (size_t i = 0; i < bs.size(); i++) { b2c_[bs[i]] = utf8_encode({(uint32_t)cs[i]}); c2b_[(uint32_t)cs[i]] = (uint8_t)bs[i]; }
  for (auto &kv : model["vocab"].obj) {
    int id = (int)kv.second.as_int(); vocab_[kv.first] = id;
    if ((int)id_to_tok_.size() <= id) id_to_tok_.resize(id + 1);
    id_to_tok_[id] = kv.first;
  }
  int r = 0;
  for (auto &mg : model["merges"].arr) ranks_.emplace(mg.type == Json::Array ? mg[0].as_str() + " " + mg[1].as_str() : mg.as_str(), r++);
  for (auto &a : tj["added_tokens"].arr) {
    int id = (int)a["id"].as_int(); const std::string &t = a["content"].as_str(); added_[t] = id; added_sorted_.push_back(t);
    if ((int)id_to_tok_.size() <= id) id_to_tok_.resize(id + 1);
    id_to_tok_[id] = t;
    if (a["special"].as_bool()) special_[id] = true;
  }
  std::sort(added_sorted_.begin(), added_sorted_.end(), [](const std::string &a, const std::string &b) { return a.size() > b.size(); });
  std::function<void(const Json &)> norm = [&](const Json &x) {
    if (x.is_null()) return;
    const std::string &t = x["type"].as_str();
    if (t == "Sequence") { for (auto &y : x["normalizers"].arr) norm(y); }
    else if (t == "NFC") nfc_ = true;
    else throw std::runtime_error("normalizer '" + t + "' is not supported");
  };
  norm(tj["normalizer"]);
  std::function<void(const Json &)> pre = [&](const Json &x) {
    if (x.is_null()) return;
    const std::string &t = x["type"].as_str();
    if (t == "Sequence") { for (auto &y : x["pretokenizers"].arr) pre(y); }
    else if (t == "Split") {
      if (x["behavior"].as_str() != "Isolated" || x["invert"].as_bool()) throw std::runtime_error("Split behavior '" + x["behavior"].as_str() + "' is not supported");
      std::string pat;
      if (x["pattern"].has("Regex")) pat = x["pattern"]["Regex"].as_str();
      else for (char c : x["pattern"]["String"].as_str()) { if (strchr(".^$|()[]{}*+?\\", c)) pat += '\\'; pat += c; }
      pre_.push_back({Pre::Split, std::make_shared<Regex>(pat)});
    } else if (t == "Digits") pre_.push_back({Pre::Digits, nullptr, x["individual_digits"].as_bool()});
    else if (t == "ByteLevel") {
      if (x["add_prefix_space"].as_bool()) throw std::runtime_error("ByteLevel add_prefix_space is not supported");
      if (x["use_regex"].as_bool(true)) pre_.push_back({Pre::ByteLevelRegex, std::make_shared<Regex>(kGPT2)});
    } else throw std::runtime_error("pre-tokenizer '" + t + "' is not supported");
  };
  pre(tj["pre_tokenizer"]);
}

std::vector<std::string> Tokenizer::pretokenize(const std::string &s) const {
  std::vector<std::string> parts{s};
  for (auto &p : pre_) {
    std::vector<std::string> next;
    for (auto &part : parts) {
      std::vector<uint32_t> cps = utf8_decode(part);
      auto emit = [&](size_t a, size_t b) { if (b > a) next.push_back(utf8_encode(std::vector<uint32_t>(cps.begin() + a, cps.begin() + b))); };
      if (p.t == Pre::Digits) {   // runs of digits (or single digits) become their own pieces
        size_t st = 0;
        for (size_t i = 0; i < cps.size(); i++) {
          bool num = uni_is_number(cps[i]), prev = i > 0 && uni_is_number(cps[i - 1]);
          if (i > st && (num != prev || (num && p.individual))) { emit(st, i); st = i; }
        }
        emit(st, cps.size());
        continue;
      }
      // Split(Isolated) keeps the gaps between matches; ByteLevel's regex covers everything, so gaps never occur there
      size_t gap = 0, from = 0, b, e;
      while (from < cps.size() && p.re->find(cps, from, &b, &e)) {
        if (e == b) { from = b + 1; continue; }   // empty matches split nothing
        emit(gap, b); emit(b, e); gap = from = e;
      }
      emit(gap, cps.size());
    }
    parts.swap(next);
  }
  return parts;
}

void Tokenizer::bpe(const std::string &word, std::vector<int> &out) const {
  auto hit = cache_.find(word);
  if (hit != cache_.end()) { out.insert(out.end(), hit->second.begin(), hit->second.end()); return; }
  std::vector<int> ids;
  auto v = vocab_.find(word);
  if (ignore_merges_ && v != vocab_.end()) ids.push_back(v->second);
  else {
    std::vector<std::string> w; for (uint32_t cp : utf8_decode(word)) w.push_back(utf8_encode({cp}));
    while (w.size() > 1) {
      int best = -1, br = 1 << 30;
      for (size_t i = 0; i + 1 < w.size(); i++) { auto it = ranks_.find(w[i] + " " + w[i + 1]); if (it != ranks_.end() && it->second < br) { br = it->second; best = (int)i; } }
      if (best < 0) break;
      const std::string a = w[best], b = w[best + 1]; std::vector<std::string> nw;
      for (size_t i = 0; i < w.size(); i++) { if (i + 1 < w.size() && w[i] == a && w[i + 1] == b) { nw.push_back(a + b); i++; } else nw.push_back(w[i]); }
      w.swap(nw);
    }
    for (auto &t : w) { auto it = vocab_.find(t); if (it == vocab_.end()) throw std::runtime_error("BPE produced a token missing from the vocab"); ids.push_back(it->second); }
  }
  if (cache_.size() < 200000) cache_.emplace(word, ids);
  out.insert(out.end(), ids.begin(), ids.end());
}

std::vector<int> Tokenizer::encode(const std::string &text) const {
  std::vector<int> out;
  auto plain = [&](const std::string &chunk) {
    if (chunk.empty()) return;
    for (auto &piece : pretokenize(nfc_ ? nfc(chunk) : chunk)) { std::string mapped; for (unsigned char c : piece) mapped += b2c_[c]; bpe(mapped, out); }
  };
  size_t pos = 0;
  while (pos < text.size()) {   // added tokens are matched in the raw text first: earliest position, longest at a tie
    size_t best = std::string::npos; const std::string *bt = nullptr;
    for (auto &t : added_sorted_) { if (t.empty()) continue; size_t f = text.find(t, pos); if (f < best) { best = f; bt = &t; } }
    if (!bt) { plain(text.substr(pos)); break; }
    plain(text.substr(pos, best - pos)); out.push_back(added_.at(*bt)); pos = best + bt->size();
  }
  return out;
}

std::string Tokenizer::piece(int id) const {
  if (id < 0 || id >= (int)id_to_tok_.size()) return "";
  const std::string &t = id_to_tok_[id];
  if (added_.count(t) && !vocab_.count(t)) return t;
  std::string o; for (uint32_t cp : utf8_decode(t)) { auto it = c2b_.find(cp); if (it != c2b_.end()) o += (char)it->second; }
  return o;
}
std::string Tokenizer::decode(const std::vector<int> &ids, bool skip_special) const {
  std::string o; for (int id : ids) { if (skip_special && is_special(id)) continue; o += piece(id); }
  return o;
}
int Tokenizer::token_id(const std::string &t) const {
  auto a = added_.find(t); if (a != added_.end()) return a->second;
  auto v = vocab_.find(t); return v == vocab_.end() ? -1 : v->second;
}

}  // namespace qp
