// Static (per-blob) byte-wise rANS used by QPACK3 containers: lossless entropy coding of q4 nibbles, q8 bytes and fp16
// scale planes. Decoded once at load time into RAM, so no kernel sees compressed data. See QPACK2.md ("QPACK3 codec").
//
// Blob layout (little endian):  u8 codec (1 = bytes, 2 = nibbles) | u8 planes (1 or 2) | u32 raw_len |
//   per plane: u32 comp_len | u16 freq[nsym] (sum 4096, 0 = absent) | u32 state[4] | renormalisation bytes
// codec 1: raw bytes split into `planes` interleaved byte planes (plane k = raw[k], raw[k+planes], ...), coded as 256-ary.
// codec 2: planes == 1, each raw byte = two symbols (low nibble first, then high nibble), coded as 16-ary.
// Four interleaved rANS states (symbol i uses state i & 3), 12-bit probabilities, 32-bit state, L = 2^23.
#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace qp {
namespace rans {

constexpr uint32_t kProbBits = 12, kProbScale = 1u << kProbBits, kL = 1u << 23;

inline void put32(std::vector<uint8_t> &o, uint32_t v) { for (int i = 0; i < 4; i++) o.push_back((uint8_t)(v >> (8 * i))); }
inline uint32_t get32(const uint8_t *p) { return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24); }

// symbols (values < nsym) -> one coded plane
inline std::vector<uint8_t> encode_plane(const std::vector<uint8_t> &sym, int nsym) {
  std::vector<uint64_t> h(nsym, 0); for (uint8_t s : sym) h[s]++;
  uint64_t tot = sym.size(); std::vector<uint32_t> f(nsym, 0), c(nsym + 1, 0);
  if (tot == 0) { f[0] = kProbScale; }
  else {
    int64_t sum = 0; int big = 0;
    for (int i = 0; i < nsym; i++) {
      if (!h[i]) continue;
      uint32_t v = (uint32_t)((h[i] * kProbScale) / tot); if (v == 0) v = 1; f[i] = v; sum += v; if (h[i] > h[big]) big = i;
    }
    int64_t nb = (int64_t)f[big] + (int64_t)kProbScale - sum;
    if (nb >= 1) f[big] = (uint32_t)nb;
    else {   // many near-uniform symbols (high-entropy data): trim the largest frequencies one by one until the total is 4096
      for (int64_t d = sum - kProbScale; d > 0; d--) { int m = 0; for (int i = 1; i < nsym; i++) if (f[i] > f[m]) m = i; if (f[m] <= 1) throw std::runtime_error("rans: cannot normalise frequencies"); f[m]--; }
    }
  }
  for (int i = 0; i < nsym; i++) c[i + 1] = c[i] + f[i];
  std::vector<uint8_t> rev;   // bytes in reverse order
  uint32_t x[4] = {kL, kL, kL, kL};
  for (int64_t i = (int64_t)sym.size() - 1; i >= 0; i--) {
    uint32_t &st = x[i & 3]; uint32_t s = sym[i], fr = f[s], xmax = ((kL >> kProbBits) << 8) * fr;
    while (st >= xmax) { rev.push_back((uint8_t)st); st >>= 8; }
    st = ((st / fr) << kProbBits) + (st % fr) + c[s];
  }
  std::vector<uint8_t> o; o.reserve(rev.size() + 2 * nsym + 24); put32(o, 0);   // comp_len patched below
  for (int i = 0; i < nsym; i++) { o.push_back((uint8_t)f[i]); o.push_back((uint8_t)(f[i] >> 8)); }
  for (int i = 0; i < 4; i++) put32(o, x[i]);
  o.insert(o.end(), rev.rbegin(), rev.rend());
  uint32_t cl = (uint32_t)(o.size() - 4); memcpy(o.data(), &cl, 4);   // little-endian host assumed (x86/ARM)
  return o;
}

// decodes one plane starting at p (points at comp_len); writes n symbols through put(i, sym); returns bytes consumed
template <class Put>
inline size_t decode_plane(const uint8_t *p, size_t avail, int nsym, uint64_t n, Put put) {
  if (avail < 4) throw std::runtime_error("rans: truncated");
  uint32_t cl = get32(p); if ((size_t)cl + 4 > avail || cl < 2u * nsym + 16) throw std::runtime_error("rans: truncated plane");
  const uint8_t *q = p + 4, *end = p + 4 + cl;
  uint32_t f[256], c[257]; c[0] = 0;
  for (int i = 0; i < nsym; i++) { f[i] = q[2 * i] | (q[2 * i + 1] << 8); c[i + 1] = c[i] + f[i]; }
  if (c[nsym] != kProbScale) throw std::runtime_error("rans: bad frequency table");
  q += 2 * nsym;
  uint8_t slot[kProbScale]; for (int s = 0; s < nsym; s++) for (uint32_t k = c[s]; k < c[s + 1]; k++) slot[k] = (uint8_t)s;
  uint32_t x[4]; for (int i = 0; i < 4; i++) { x[i] = get32(q); q += 4; }
  for (uint64_t i = 0; i < n; i++) {
    uint32_t &st = x[i & 3]; uint32_t m = st & (kProbScale - 1); uint32_t s = slot[m];
    st = f[s] * (st >> kProbBits) + m - c[s];
    while (st < kL) { if (q >= end) throw std::runtime_error("rans: stream overrun"); st = (st << 8) | *q++; }
    put(i, (uint8_t)s);
  }
  return (size_t)cl + 4;
}

inline std::vector<uint8_t> encode(const uint8_t *raw, size_t n, int codec, int planes) {
  std::vector<uint8_t> o; o.push_back((uint8_t)codec); o.push_back((uint8_t)planes); put32(o, (uint32_t)n);
  if (codec == 2) {
    std::vector<uint8_t> s(2 * n); for (size_t i = 0; i < n; i++) { s[2 * i] = raw[i] & 15; s[2 * i + 1] = raw[i] >> 4; }
    auto pl = encode_plane(s, 16); o.insert(o.end(), pl.begin(), pl.end());
  } else {
    for (int k = 0; k < planes; k++) {
      std::vector<uint8_t> s; for (size_t i = k; i < n; i += planes) s.push_back(raw[i]);
      auto pl = encode_plane(s, 256); o.insert(o.end(), pl.begin(), pl.end());
    }
  }
  return o;
}

inline size_t raw_size(const uint8_t *p, size_t len) { if (len < 6) throw std::runtime_error("rans: short blob"); return get32(p + 2); }

// Chunked blob (codec 3): u8 3 | u8 0 | u32 raw_len | u32 chunk_raw | u32 nchunks | u32 comp_len[nchunks] | chunks, each a plain
// blob (codec 1/2) of chunk_raw raw bytes (last one shorter). chunk_raw is a multiple of 4. Lets big blobs decode in parallel.
inline std::vector<uint8_t> assemble_chunked(size_t n, size_t chunk_raw, const std::vector<std::vector<uint8_t>> &chunks) {
  std::vector<uint8_t> o; o.push_back(3); o.push_back(0); put32(o, (uint32_t)n); put32(o, (uint32_t)chunk_raw); put32(o, (uint32_t)chunks.size());
  for (auto &c : chunks) put32(o, (uint32_t)c.size());
  for (auto &c : chunks) o.insert(o.end(), c.begin(), c.end());
  return o;
}
struct ChunkRef { const uint8_t *p; size_t len; size_t off; };   // coded chunk and its offset in the raw output
inline std::vector<ChunkRef> chunk_refs(const uint8_t *p, size_t len) {
  if (len < 14) throw std::runtime_error("rans: short chunked blob");
  size_t cr = get32(p + 6), nc = get32(p + 10), pos = 14 + 4 * nc; std::vector<ChunkRef> r;
  if (pos > len) throw std::runtime_error("rans: truncated chunk table");
  for (size_t i = 0; i < nc; i++) {
    size_t cl = get32(p + 14 + 4 * i); if (pos + cl > len) throw std::runtime_error("rans: truncated chunk");
    r.push_back({p + pos, cl, i * cr}); pos += cl;
  }
  return r;
}

// out must hold raw_size bytes
inline void decode(const uint8_t *p, size_t len, uint8_t *out) {
  if (len >= 1 && p[0] == 3) { for (auto &c : chunk_refs(p, len)) decode(c.p, c.len, out + c.off); return; }
  size_t n = raw_size(p, len); int codec = p[0], planes = p[1]; size_t pos = 6;
  if (codec == 2) {
    if (n) memset(out, 0, n);
    pos += decode_plane(p + pos, len - pos, 16, 2 * n, [&](uint64_t i, uint8_t s) { out[i >> 1] |= (uint8_t)(s << (4 * (i & 1))); });
  } else if (codec == 1 && planes >= 1 && planes <= 4) {
    for (int k = 0; k < planes; k++) {
      uint64_t cnt = (n + planes - 1 - k) / planes;
      pos += decode_plane(p + pos, len - pos, 256, cnt, [&](uint64_t i, uint8_t s) { out[i * planes + k] = s; });
    }
  } else throw std::runtime_error("rans: unknown codec");
}

}  // namespace rans
}  // namespace qp
