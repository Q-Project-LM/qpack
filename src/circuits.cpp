#include "qpack/circuits.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace qp {
namespace {

typedef __int128 i128;
struct Fail {};

// ------------------------------------------------------------------ Python number model
struct Num { bool is_int; i128 i; double f; };
Num I(i128 v) { return {true, v, 0}; }
Num F(double v) { return {false, 0, v}; }
double num(const Num &x) { return x.is_int ? (double)x.i : x.f; }

std::string i128_str(i128 v) {
  if (v == 0) return "0";
  bool neg = v < 0; unsigned __int128 u = neg ? -(unsigned __int128)v : (unsigned __int128)v; std::string s;
  while (u) { s += char('0' + (int)(u % 10)); u /= 10; }
  if (neg) s += '-';
  std::reverse(s.begin(), s.end()); return s;
}
std::string strip(const std::string &s) {   // Python str.strip(): Unicode whitespace (and the \x1c-\x1f separators)
  auto ws = [](uint32_t c) { return uni_is_space(c) || (c >= 0x1C && c <= 0x1F); };
  auto cps = utf8_decode(s); size_t a = 0, b = cps.size();
  while (a < b && ws(cps[a])) a++;
  while (b > a && ws(cps[b - 1])) b--;
  return utf8_encode(std::vector<uint32_t>(cps.begin() + a, cps.begin() + b));
}
bool starts(const std::string &s, const char *p) { return s.compare(0, strlen(p), p) == 0; }
std::vector<std::string> split(const std::string &s, const std::string &sep) {
  std::vector<std::string> o; size_t a = 0;
  for (;;) { size_t f = s.find(sep, a); if (f == std::string::npos) { o.push_back(s.substr(a)); return o; } o.push_back(s.substr(a, f - a)); a = f + sep.size(); }
}
std::vector<std::string> split2(const std::string &s, const std::string &sep) { auto v = split(s, sep); if (v.size() != 2) throw Fail(); return v; }

double py_float(const std::string &s0) {   // float(s): surrounding whitespace allowed, otherwise strict
  std::string s = strip(s0); if (s.empty()) throw Fail();
  std::string low; for (char c : s) low += (char)tolower((unsigned char)c);
  std::string body = low[0] == '+' || low[0] == '-' ? low.substr(1) : low;
  if (body == "inf" || body == "infinity") return low[0] == '-' ? -INFINITY : INFINITY;
  if (body == "nan") return NAN;
  size_t i = 0, n = body.size(); bool digits = false;   // \d+\.?\d*(e[+-]?\d+)? | \.\d+(e[+-]?\d+)?
  while (i < n && isdigit((unsigned char)body[i])) { i++; digits = true; }
  if (i < n && body[i] == '.') { i++; bool frac = false; while (i < n && isdigit((unsigned char)body[i])) { i++; frac = true; } if (!digits && !frac) throw Fail(); }
  else if (!digits) throw Fail();
  if (i < n && body[i] == 'e') { i++; if (i < n && (body[i] == '+' || body[i] == '-')) i++; size_t e0 = i; while (i < n && isdigit((unsigned char)body[i])) i++; if (i == e0) throw Fail(); }
  if (i != n) throw Fail();
  return strtod(s.c_str(), nullptr);
}
i128 py_int(const std::string &s0) {
  std::string s = strip(s0); size_t i = 0; bool neg = false;
  if (i < s.size() && (s[i] == '+' || s[i] == '-')) { neg = s[i] == '-'; i++; }
  if (i == s.size()) throw Fail();
  i128 v = 0;
  for (; i < s.size(); i++) { if (!isdigit((unsigned char)s[i])) throw Fail(); if (__builtin_mul_overflow(v, (i128)10, &v) || __builtin_add_overflow(v, (i128)(s[i] - '0'), &v)) throw Fail(); }
  return neg ? -v : v;
}

double py_round(double x, int n) {   // round(float, n): the exact binary value rounded half-even (glibc printf does exactly this)
  if (!std::isfinite(x)) return x;
  char buf[512]; snprintf(buf, sizeof buf, "%.*f", n, x); return strtod(buf, nullptr);
}
std::string py_str_float(double x) {   // repr(float): shortest round-trip digits, fixed for 1e-4 <= |x| < 1e16
  if (std::isnan(x)) return "nan";
  if (std::isinf(x)) return x > 0 ? "inf" : "-inf";
  if (x == 0) return std::signbit(x) ? "-0.0" : "0.0";
  char buf[64]; int p = 1;
  for (; p <= 17; p++) { snprintf(buf, sizeof buf, "%.*e", p - 1, x); if (strtod(buf, nullptr) == x) break; }
  std::string s = buf; size_t e = s.find('e'); int ex = atoi(s.c_str() + e + 1);
  std::string mant = s.substr(0, e); bool neg = mant[0] == '-'; if (neg) mant = mant.substr(1);
  std::string dig; for (char c : mant) if (c != '.') dig += c;
  while (dig.size() > 1 && dig.back() == '0') dig.pop_back();
  std::string o;
  if (ex < -4 || ex >= 16) {
    o = dig.substr(0, 1); if (dig.size() > 1) o += "." + dig.substr(1);
    char eb[16]; snprintf(eb, sizeof eb, "e%c%02d", ex < 0 ? '-' : '+', std::abs(ex)); o += eb;
  } else if (ex < 0) o = "0." + std::string(-ex - 1, '0') + dig;
  else if ((int)dig.size() <= ex + 1) o = dig + std::string(ex + 1 - dig.size(), '0') + ".0";
  else o = dig.substr(0, ex + 1) + "." + dig.substr(ex + 1);
  return neg ? "-" + o : o;
}
std::string fmt_num(const Num &x) {   // fmt_num: floats rounded to 6 places, integral values without ".0"
  if (x.is_int) return i128_str(x.i);
  double v = py_round(x.f, 6);
  if (std::isfinite(v) && v == std::floor(v)) { if (v == 0) return "0"; char buf[400]; snprintf(buf, sizeof buf, "%.0f", v); return buf; }
  return py_str_float(v);
}
std::string fmt_f(double v) { return fmt_num(F(v)); }

// ------------------------------------------------------------------ Python expression subset
Num binop(const std::string &op, const Num &a, const Num &b) {
  const bool ints = a.is_int && b.is_int;
  if (op == "**") {
    if (std::fabs(num(b)) > 12) throw Fail();
    if (ints && b.i >= 0) { i128 r = 1; for (i128 k = 0; k < b.i; k++) if (__builtin_mul_overflow(r, a.i, &r)) throw Fail(); return I(r); }
    if (ints && a.i == 0) throw Fail();
    double x = num(a), y = num(b);
    if (x == 0 && y < 0) throw Fail();
    if (x < 0 && y != std::floor(y)) throw Fail();   // Python returns a complex number; evaluate() then fails
    return F(std::pow(x, y));
  }
  if (op == "/") { if (num(b) == 0) throw Fail(); if (ints) return F((double)((long double)a.i / (long double)b.i)); return F(num(a) / num(b)); }
  if (op == "//" || op == "%") {
    if (num(b) == 0) throw Fail();
    if (ints) { i128 q = a.i / b.i; if (a.i % b.i != 0 && ((a.i < 0) != (b.i < 0))) q -= 1; return op == "//" ? I(q) : I(a.i - q * b.i); }
    double vx = num(a), wx = num(b), mod = std::fmod(vx, wx), div = (vx - mod) / wx;   // CPython float_divmod
    if (mod) {
      if ((wx < 0) != (mod < 0)) { mod += wx; div -= 1.0; }
    } else mod = std::copysign(0.0, wx);
    double fd; if (div) { fd = std::floor(div); if (div - fd > 0.5) fd += 1.0; } else fd = std::copysign(0.0, vx / wx);
    return op == "//" ? F(fd) : F(mod);
  }
  if (ints) {
    i128 r;
    bool ov = op == "+" ? __builtin_add_overflow(a.i, b.i, &r) : op == "-" ? __builtin_sub_overflow(a.i, b.i, &r) : __builtin_mul_overflow(a.i, b.i, &r);
    if (ov) throw Fail();
    return I(r);
  }
  double x = num(a), y = num(b); return F(op == "+" ? x + y : op == "-" ? x - y : x * y);
}

struct Parser {
  std::vector<std::string> t; size_t p = 0;
  explicit Parser(const std::string &s) {
    size_t i = 0, n = s.size();
    auto digs = [&](size_t &j) {   // \d(?:_?\d)*
      if (j >= n || !isdigit((unsigned char)s[j])) return false;
      j++; while (j < n) { if (isdigit((unsigned char)s[j])) j++; else if (s[j] == '_' && j + 1 < n && isdigit((unsigned char)s[j + 1])) j += 2; else break; }
      return true;
    };
    while (i < n) {
      while (i < n && strchr(" \t\n\r\f\v", s[i])) i++;
      if (i >= n) break;
      size_t j = i;
      if (isdigit((unsigned char)s[i]) || (s[i] == '.' && i + 1 < n && isdigit((unsigned char)s[i + 1]))) {
        if (s[i] == '.') { j++; digs(j); }
        else { digs(j); if (j < n && s[j] == '.') { j++; digs(j); } }
        if (j < n && (s[j] == 'e' || s[j] == 'E')) { size_t k = j + 1; if (k < n && (s[k] == '+' || s[k] == '-')) k++; if (digs(k)) j = k; }
        std::string tok; for (size_t k = i; k < j; k++) if (s[k] != '_') tok += s[k];
        t.push_back(tok); i = j; continue;
      }
      if (s.compare(i, 2, "**") == 0 || s.compare(i, 2, "//") == 0) { t.push_back(s.substr(i, 2)); i += 2; continue; }
      if (strchr("+-*/%()", s[i])) { t.push_back(std::string(1, s[i])); i++; continue; }
      throw Fail();
    }
  }
  const std::string *peek() const { return p < t.size() ? &t[p] : nullptr; }
  bool is(const char *x) const { return p < t.size() && t[p] == x; }
  Num atom() {
    if (p >= t.size()) throw Fail();
    const std::string tok = t[p++];
    if (tok == "(") { Num v = sum(); if (!is(")")) throw Fail(); p++; return v; }
    if (isdigit((unsigned char)tok[0]) || tok[0] == '.') {
      bool all_digits = std::all_of(tok.begin(), tok.end(), [](char c) { return isdigit((unsigned char)c); });
      if (all_digits) { if (tok.size() > 1 && tok[0] == '0' && tok.find_first_of("123456789") != std::string::npos) throw Fail(); return I(py_int(tok)); }
      return F(strtod(tok.c_str(), nullptr));
    }
    throw Fail();
  }
  Num power() { Num a = atom(); if (is("**")) { p++; Num b = unary(); return binop("**", a, b); } return a; }
  Num unary() {   // only unary minus: the Python evaluator has no UAdd
    if (is("+")) throw Fail();
    if (is("-")) { p++; Num v = unary(); if (v.is_int) { if (v.i == ((i128)1 << 126) * -2) throw Fail(); return I(-v.i); } return F(-v.f); }
    return power();
  }
  Num term() { Num a = unary(); while (is("*") || is("/") || is("//") || is("%")) { std::string op = t[p++]; a = binop(op, a, unary()); } return a; }
  Num sum() { Num a = term(); while (is("+") || is("-")) { std::string op = t[p++]; a = binop(op, a, term()); } return a; }
};

std::string replace_all(std::string s, const std::string &a, const std::string &b) {
  for (size_t f = 0; (f = s.find(a, f)) != std::string::npos; f += b.size()) s.replace(f, a.size(), b);
  return s;
}

// dates: proleptic Gregorian day numbers (Howard Hinnant's days_from_civil), range 0001-01-01 .. 9999-12-31 like datetime.date
int64_t days_from_civil(int64_t y, unsigned m, unsigned d) {
  y -= m <= 2; const int64_t era = (y >= 0 ? y : y - 399) / 400; const unsigned yoe = (unsigned)(y - era * 400);
  const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1, doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + (int64_t)doe - 719468;
}
void civil_from_days(int64_t z, int *y, int *m, int *d) {
  z += 719468; const int64_t era = (z >= 0 ? z : z - 146096) / 146097; const unsigned doe = (unsigned)(z - era * 146097);
  const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const unsigned mp = (5 * doy + 2) / 153; *d = (int)(doy - (153 * mp + 2) / 5 + 1); *m = (int)(mp < 10 ? mp + 3 : mp - 9); *y = (int)(yoe + era * 400 + (*m <= 2));
}
int64_t iso_date(const std::string &s) {
  if (s.size() != 10 || s[4] != '-' || s[7] != '-') throw Fail();
  for (int i : {0, 1, 2, 3, 5, 6, 8, 9}) if (!isdigit((unsigned char)s[i])) throw Fail();
  int y = atoi(s.substr(0, 4).c_str()), m = atoi(s.substr(5, 2).c_str()), d = atoi(s.substr(8, 2).c_str());
  static const int md[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  bool leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
  if (y < 1 || m < 1 || m > 12 || d < 1 || d > md[m - 1] + (m == 2 && leap)) throw Fail();
  return days_from_civil(y, m, d);
}
std::string iso(int64_t z) {
  int y, m, d; civil_from_days(z, &y, &m, &d);
  if (y < 1 || y > 9999) throw Fail();
  char b[48]; snprintf(b, sizeof b, "%04d-%02d-%02d", y, m, d); return b;
}
int weekday(int64_t z) { return (int)(((z % 7) + 7 + 3) % 7); }   // 1970-01-01 was a Thursday; Monday = 0

std::string eval(std::string expr) {
  static const char *WD_EN[] = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"};
  static const char *WD_RU[] = {"понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"};
  expr = strip(expr);
  if (starts(expr, "date:")) { auto v = split(expr.substr(5), "+"); if (v.size() != 2) throw Fail(); i128 n = py_int(v[1]); if (n > 4000000 || n < -4000000) throw Fail(); return iso(iso_date(v[0]) + (int64_t)n); }
  if (starts(expr, "wd:")) return WD_EN[weekday(iso_date(expr.substr(3)))];
  if (starts(expr, "wdru:")) return WD_RU[weekday(iso_date(expr.substr(5)))];
  if (starts(expr, "dur:")) { auto v = split2(expr.substr(4), ","); return std::to_string(std::llabs(iso_date(v[1]) - iso_date(v[0]))); }
  if (starts(expr, "conv:")) {
    std::string r = expr.substr(5); size_t sp = r.find(' '); if (sp == std::string::npos) throw Fail();
    double v = py_float(r.substr(0, sp)); auto ab = split2(r.substr(sp + 1), ">");
    static const struct { const char *a, *b; double k; } U[] = {{"km", "mi", 0.621371}, {"mi", "km", 1.609344}, {"kg", "lb", 2.204623}, {"lb", "kg", 0.453592}, {"m", "ft", 3.28084},
                                                               {"ft", "m", 0.3048}, {"l", "gal", 0.264172}, {"gal", "l", 3.78541}, {"cm", "in", 0.393701}, {"in", "cm", 2.54}};
    for (auto &u : U) if (ab[0] == u.a && ab[1] == u.b) return fmt_f(py_round(v * u.k, 2));
    if (ab[0] == "c" && ab[1] == "f") return fmt_f(py_round(v * 9 / 5 + 32, 2));
    if (ab[0] == "f" && ab[1] == "c") return fmt_f(py_round((v - 32) * 5 / 9, 2));
    throw Fail();
  }
  if (starts(expr, "cmp:")) { auto v = split2(expr.substr(4), ","); double a = py_float(v[0]), b = py_float(v[1]); return a > b ? "first" : a < b ? "second" : "equal"; }
  if (starts(expr, "sort:")) {
    auto xs = split(expr.substr(5), ","); std::vector<std::pair<double, size_t>> k;
    for (size_t i = 0; i < xs.size(); i++) k.push_back({py_float(xs[i]), i});
    std::stable_sort(k.begin(), k.end(), [](auto &a, auto &b) { return a.first < b.first; });
    std::string o; for (size_t i = 0; i < k.size(); i++) { if (i) o += ","; o += xs[k[i].second]; } return o;
  }
  if (starts(expr, "filter:")) {
    std::string r = expr.substr(7); size_t c = r.find(':'); if (c == std::string::npos) throw Fail();
    std::string cond = r.substr(0, c), rest = r.substr(c + 1);
    for (const char *sym : {">=", "<=", "==", ">", "<"}) {
      if (!starts(cond, sym)) continue;
      double thr = py_float(cond.substr(strlen(sym))); std::string o; bool any = false;
      for (auto &x : split(rest, ",")) {
        double v = py_float(x); std::string s = sym;
        bool ok = s == ">=" ? v >= thr : s == "<=" ? v <= thr : s == "==" ? v == thr : s == ">" ? v > thr : v < thr;
        if (ok) { if (any) o += ","; o += x; any = true; }
      }
      return any ? o : "none";
    }
    throw Fail();
  }
  if (starts(expr, "count:")) return std::to_string(utf8_decode(expr.substr(6)).size());
  for (const char *k : {"mean:", "max:", "min:", "sum:"}) {
    if (!starts(expr, k)) continue;
    std::vector<double> v; for (auto &x : split(expr.substr(strlen(k)), ",")) v.push_back(py_float(x));
    double r; std::string kk = k;
    if (kk == "max:") r = *std::max_element(v.begin(), v.end());   // first maximal element, like max()
    else if (kk == "min:") r = *std::min_element(v.begin(), v.end());
    else { r = 0; for (double x : v) r += x; if (kk == "mean:") r /= v.size(); }
    return fmt_f(py_round(r, 4));
  }
  auto gcd = [](i128 a, i128 b) { a = a < 0 ? -a : a; b = b < 0 ? -b : b; while (b) { i128 t = a % b; a = b; b = t; } return a; };
  if (starts(expr, "gcd:")) { auto v = split2(expr.substr(4), ","); return i128_str(gcd(py_int(v[0]), py_int(v[1]))); }
  if (starts(expr, "lcm:")) {
    auto v = split2(expr.substr(4), ","); i128 a = py_int(v[0]), b = py_int(v[1]), g = gcd(a, b), l = 0;
    if (g != 0 && __builtin_mul_overflow(a / g, b, &l)) throw Fail();
    return i128_str(l < 0 ? -l : l);
  }
  if (starts(expr, "prime:")) {
    i128 n = py_int(expr.substr(6)); if (n <= 1) return "no"; if (n > ((i128)1 << 62)) throw Fail();
    for (i128 d = 2; d * d <= n; d++) if (n % d == 0) return "no";
    return "yes";
  }
  if (starts(expr, "rev:")) { auto cps = utf8_decode(expr.substr(4)); std::reverse(cps.begin(), cps.end()); return utf8_encode(cps); }
  if (starts(expr, "fact:")) { i128 n = py_int(expr.substr(5)); if (n < 0 || n > 20) throw Fail(); i128 f = 1; for (i128 i = 2; i <= n; i++) f *= i; return i128_str(f); }
  if (starts(expr, "sqrt:")) { double v = py_float(expr.substr(5)); if (v < 0) throw Fail(); return fmt_f(py_round(std::sqrt(v), 3)); }
  if (starts(expr, "pchg:")) { auto v = split2(expr.substr(5), ","); double a = py_float(v[0]), b = py_float(v[1]); if (a == 0) throw Fail(); return fmt_f(py_round((b - a) / a * 100, 2)); }
  if (starts(expr, "words:")) {
    auto cps = utf8_decode(expr.substr(6)); int n = 0; bool in = false;
    for (uint32_t c : cps) { bool sp = uni_is_space(c) || (c >= 0x1C && c <= 0x1F); if (!sp && !in) n++; in = !sp; }
    return std::to_string(n);
  }
  Parser ps(replace_all(replace_all(replace_all(expr, "×", "*"), "÷", "/"), ",", "."));
  if (ps.t.empty()) throw Fail();
  Num r = ps.sum(); if (ps.p != ps.t.size()) throw Fail();
  if (!r.is_int && !std::isfinite(r.f)) { if (std::isinf(r.f)) throw Fail(); }
  return r.is_int ? i128_str(r.i) : fmt_num(F(py_round(r.f, 6)));
}

}  // namespace

bool circuit_eval(const std::string &expr, std::string *result) {
  try { *result = eval(expr); return true; } catch (const Fail &) { return false; } catch (const std::exception &) { return false; }
}

CircuitExecutor::CircuitExecutor(const Tokenizer &tok) : tok_(tok) {
  calc_id = tok.token_id("<CALC>"); eq_id = tok.token_id("<EQ>"); ecalc_id = tok.token_id("<ECALC>");
}
int CircuitExecutor::next(const std::vector<int> &ids) {
  if (queue_.empty() && !ids.empty() && ids.back() == eq_id && eq_id >= 0) {
    int s = -1; for (int i = (int)ids.size() - 1; i >= 0; i--) if (ids[i] == calc_id) { s = i; break; }
    std::string res; bool ok = s >= 0 && circuit_eval(tok_.decode(std::vector<int>(ids.begin() + s + 1, ids.end() - 1), true), &res);
    calls++; if (!ok) { failed++; res = "?"; }
    for (int t : tok_.encode(res)) queue_.push_back(t);
    queue_.push_back(ecalc_id);
  }
  if (queue_.empty()) return -1;
  int t = queue_.front(); queue_.pop_front(); return t;
}

}  // namespace qp
