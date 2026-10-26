#include "qpack/json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace qp {

static const Json kNull;
static const std::string kEmpty;

bool Json::has(const std::string &k) const {
  if (type != Object) return false;
  for (auto &kv : obj) if (kv.first == k) return true;
  return false;
}
const Json &Json::operator[](const std::string &k) const {
  if (type == Object) for (auto &kv : obj) if (kv.first == k) return kv.second;
  return kNull;
}
Json &Json::set(const std::string &k, Json v) {
  if (type != Object) { type = Object; obj.clear(); }
  for (auto &kv : obj) if (kv.first == k) { kv.second = std::move(v); return kv.second; }
  obj.emplace_back(k, std::move(v)); return obj.back().second;
}
const std::string &Json::as_str() const { return type == String ? str : kEmpty; }

namespace {
struct P {
  const char *p, *e;
  void ws() { while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++; }
  [[noreturn]] void fail(const char *m) { throw std::runtime_error(std::string("json: ") + m); }
  static void utf8(std::string &o, unsigned cp) {
    if (cp < 0x80) o += (char)cp;
    else if (cp < 0x800) { o += (char)(0xC0 | (cp >> 6)); o += (char)(0x80 | (cp & 63)); }
    else if (cp < 0x10000) { o += (char)(0xE0 | (cp >> 12)); o += (char)(0x80 | ((cp >> 6) & 63)); o += (char)(0x80 | (cp & 63)); }
    else { o += (char)(0xF0 | (cp >> 18)); o += (char)(0x80 | ((cp >> 12) & 63)); o += (char)(0x80 | ((cp >> 6) & 63)); o += (char)(0x80 | (cp & 63)); }
  }
  unsigned hex4() {
    if (e - p < 4) fail("bad \\u escape");
    unsigned v = 0;
    for (int i = 0; i < 4; i++) { char c = *p++; v <<= 4; if (c >= '0' && c <= '9') v |= c - '0'; else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10; else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10; else fail("bad hex"); }
    return v;
  }
  std::string string() {
    std::string o; p++;
    while (p < e && *p != '"') {
      if (*p != '\\') { o += *p++; continue; }
      p++; if (p >= e) fail("eof in string");
      char c = *p++;
      switch (c) {
        case '"': o += '"'; break; case '\\': o += '\\'; break; case '/': o += '/'; break;
        case 'b': o += '\b'; break; case 'f': o += '\f'; break; case 'n': o += '\n'; break; case 'r': o += '\r'; break; case 't': o += '\t'; break;
        case 'u': {
          unsigned cp = hex4();
          if (cp >= 0xD800 && cp < 0xDC00 && p + 1 < e && p[0] == '\\' && p[1] == 'u') { p += 2; unsigned lo = hex4(); cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); }
          utf8(o, cp); break;
        }
        default: fail("bad escape");
      }
    }
    if (p >= e) fail("unterminated string");
    p++; return o;
  }
  Json value() {
    ws(); if (p >= e) fail("unexpected end");
    char c = *p;
    if (c == '{') {
      Json j = Json::object(); p++; ws();
      if (p < e && *p == '}') { p++; return j; }
      for (;;) {
        ws(); if (p >= e || *p != '"') fail("expected key");
        std::string k = string(); ws(); if (p >= e || *p != ':') fail("expected :"); p++;
        j.obj.emplace_back(std::move(k), value()); ws();
        if (p < e && *p == ',') { p++; continue; }
        if (p < e && *p == '}') { p++; return j; }
        fail("expected , or }");
      }
    }
    if (c == '[') {
      Json j = Json::array(); p++; ws();
      if (p < e && *p == ']') { p++; return j; }
      for (;;) {
        j.arr.push_back(value()); ws();
        if (p < e && *p == ',') { p++; continue; }
        if (p < e && *p == ']') { p++; return j; }
        fail("expected , or ]");
      }
    }
    if (c == '"') return Json(string());
    if (c == 't' && e - p >= 4) { p += 4; return Json(true); }
    if (c == 'f' && e - p >= 5) { p += 5; return Json(false); }
    if (c == 'n' && e - p >= 4) { p += 4; return Json(); }
    char *end; double v = strtod(p, &end); if (end == p) fail("bad value"); p = end; return Json(v);
  }
};

void dump_str(const std::string &s, std::string &o) {
  o += '"';
  for (unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break; case '\\': o += "\\\\"; break; case '\n': o += "\\n"; break; case '\r': o += "\\r"; break; case '\t': o += "\\t"; break;
      default: if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; } else o += (char)c;
    }
  }
  o += '"';
}
void dump(const Json &j, std::string &o) {
  switch (j.type) {
    case Json::Null: o += "null"; break;
    case Json::Bool: o += j.b ? "true" : "false"; break;
    case Json::Number: {
      if (std::floor(j.num) == j.num && std::fabs(j.num) < 1e15) { char b[32]; snprintf(b, sizeof b, "%lld", (long long)j.num); o += b; }
      else { char b[32]; snprintf(b, sizeof b, "%.9g", j.num); o += b; }
      break;
    }
    case Json::String: dump_str(j.str, o); break;
    case Json::Array: o += '['; for (size_t i = 0; i < j.arr.size(); i++) { if (i) o += ','; dump(j.arr[i], o); } o += ']'; break;
    case Json::Object: o += '{'; for (size_t i = 0; i < j.obj.size(); i++) { if (i) o += ','; dump_str(j.obj[i].first, o); o += ':'; dump(j.obj[i].second, o); } o += '}'; break;
  }
}
}  // namespace

Json Json::parse(const char *p, size_t n) { P ps{p, p + n}; Json j = ps.value(); return j; }
std::string Json::dump() const { std::string o; qp::dump(*this, o); return o; }

}  // namespace qp
