// Minimal JSON value + parser/serializer (no dependencies). Enough for QPACK manifests, configs, tokenizer.json and the
// OpenAI-style HTTP API of qpack-server.
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace qp {

struct Json {
  enum Type { Null, Bool, Number, String, Array, Object } type = Null;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<Json> arr;
  std::vector<std::pair<std::string, Json>> obj;   // insertion order kept (tokenizer.json vocab is an object)

  Json() = default;
  Json(std::nullptr_t) {}
  Json(bool v) : type(Bool), b(v) {}
  Json(int v) : type(Number), num(v) {}
  Json(long v) : type(Number), num((double)v) {}
  Json(long long v) : type(Number), num((double)v) {}
  Json(double v) : type(Number), num(v) {}
  Json(const char *s) : type(String), str(s) {}
  Json(std::string s) : type(String), str(std::move(s)) {}
  static Json array() { Json j; j.type = Array; return j; }
  static Json object() { Json j; j.type = Object; return j; }

  bool is_null() const { return type == Null; }
  bool has(const std::string &k) const;
  const Json &operator[](const std::string &k) const;   // Null if missing
  Json &set(const std::string &k, Json v);
  const Json &operator[](size_t i) const { return arr[i]; }
  size_t size() const { return type == Array ? arr.size() : obj.size(); }
  void push(Json v) { arr.push_back(std::move(v)); }

  double as_num(double def = 0) const { return type == Number ? num : def; }
  long long as_int(long long def = 0) const { return type == Number ? (long long)num : def; }
  bool as_bool(bool def = false) const { return type == Bool ? b : def; }
  const std::string &as_str() const;

  static Json parse(const char *p, size_t n);
  static Json parse(const std::string &s) { return parse(s.data(), s.size()); }
  std::string dump() const;
};

}  // namespace qp
