// Byte-level BPE tokenizer read from a Hugging Face tokenizer.json (Qwen2, Llama 3, SmolLM, GPT-2, N-Q, ...):
// NFC normalizer, Split(regex) / Digits / ByteLevel pre-tokenizers, BPE merges (ignore_merges), added tokens.
// The regex engine is a small backtracking matcher over code points with Unicode \p{L}, \p{N}, \s classes.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "qpack/json.h"

namespace qp {

bool uni_is_letter(uint32_t cp);
bool uni_is_number(uint32_t cp);
bool uni_is_space(uint32_t cp);
std::vector<uint32_t> utf8_decode(const std::string &s);
std::string utf8_encode(const std::vector<uint32_t> &cps);
std::string nfc(const std::string &s);

class Regex {   // Rust-regex subset used by tokenizers: | (?:) (?i:) [...] \p{L} \p{N} \s \S \r \n ? * + {m,n} (?!...)
 public:
  explicit Regex(const std::string &pattern);
  ~Regex();
  // leftmost-first match starting the search at `from`; returns false if none. [*b, *e) in code-point indices.
  bool find(const std::vector<uint32_t> &s, size_t from, size_t *b, size_t *e) const;
 private:
  struct Impl; std::unique_ptr<Impl> impl_;
};

class Tokenizer {
 public:
  explicit Tokenizer(const Json &tokenizer_json);
  std::vector<int> encode(const std::string &text) const;
  std::string decode(const std::vector<int> &ids, bool skip_special = true) const;
  std::string piece(int id) const;               // decoded bytes of one token (may be a partial UTF-8 sequence)
  int token_id(const std::string &tok) const;     // -1 if unknown
  bool is_special(int id) const { return special_.count(id) > 0; }
  int vocab_size() const { return (int)id_to_tok_.size(); }
 private:
  struct Pre { enum T { Split, Digits, ByteLevelRegex } t; std::shared_ptr<Regex> re; bool individual = false; };
  std::vector<std::string> pretokenize(const std::string &s) const;
  void bpe(const std::string &word, std::vector<int> &out) const;
  std::unordered_map<std::string, int> vocab_, ranks_, added_;
  std::vector<std::string> id_to_tok_;
  std::unordered_map<int, bool> special_;
  std::vector<std::string> added_sorted_;          // longest first
  std::vector<Pre> pre_;
  bool nfc_ = false, ignore_merges_ = false;
  std::string b2c_[256]; std::unordered_map<uint32_t, uint8_t> c2b_;
  mutable std::unordered_map<std::string, std::vector<int>> cache_;
};

}  // namespace qp
