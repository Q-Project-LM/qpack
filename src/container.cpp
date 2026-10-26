#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "qpack/qpack.h"

namespace qp {

static uint16_t rd16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }

Container::Container(const std::string &path) {
  fd_ = open(path.c_str(), O_RDONLY);
  if (fd_ < 0) throw std::runtime_error("cannot open " + path);
  struct stat st; fstat(fd_, &st); size_ = (size_t)st.st_size;
  map_ = (uint8_t *)mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (map_ == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);
  if (size_ < 16) throw std::runtime_error("not a QPACK file: " + path);
  if (!memcmp(map_, "QPACK1\0\0", 8)) version_ = 1;
  else if (!memcmp(map_, "QPACK2\0\0", 8)) version_ = 2;
  else throw std::runtime_error("not a QPACK1/QPACK2 file: " + path);
  uint32_t n; memcpy(&n, map_ + 8, 4);
  for (uint32_t i = 0; i < n; i++) {
    const uint8_t *b = map_ + 16 + 72 * (size_t)i;
    std::string name((const char *)b, strnlen((const char *)b, 56));
    uint64_t off, len; memcpy(&off, b + 56, 8); memcpy(&len, b + 64, 8);
    if (off + len > size_) throw std::runtime_error("corrupt table of contents: " + name);
    toc_[name] = {off, len};
  }
  manifest_ = json("manifest.json"); config_ = json("config.json");
  for (auto &e : manifest_["tensors"].arr) {
    Tensor t; t.name = e["name"].as_str();
    for (auto &s : e["shape"].arr) t.shape.push_back(s.as_int());
    const std::string &kind = e["kind"].as_str(); size_t len = 0;
    const uint8_t *p = blob(e["blob"].as_str(), &len);
    if (kind == "f32") { t.kind = Kind::F32; t.data = p; }
    else if (kind == "f16") { t.kind = Kind::F16; t.data = p; }
    else if (kind == "i64") { t.kind = Kind::I64; t.data = p; }
    else if (kind == "q8") { t.kind = Kind::Q8; t.data = p; size_t sl; t.scale = (const float *)blob(e["scale"].as_str(), &sl); }
    else if (kind == "q4") {
      t.kind = Kind::Q4; t.data = p; size_t sl; t.scale16 = (const uint16_t *)blob(e["scale"].as_str(), &sl); t.group = (int)e["group"].as_int(32);
      if (t.group != 32 || t.shape.size() != 2 || t.shape[1] % 32 || t.shape[1] > 32768) throw std::runtime_error("q4 tensor " + t.name + ": only group 32 with cols % 32 == 0 and cols <= 32768 is supported");
    }
    else if (kind == "tern5" || kind == "bits") throw std::runtime_error("tensor kind " + kind + " (Q-architecture checkpoints) is not supported by libqpack yet");
    else throw std::runtime_error("unknown tensor kind " + kind);
    // small f16 tensors (norms, biases, conv) are converted once so the model code can read them as float
    if (t.kind == Kind::F16 && t.shape.size() != 2) {
      t.owned.resize(4 * t.numel()); float *o = (float *)t.owned.data();
      for (int64_t i = 0; i < t.numel(); i++) {
        uint16_t h = rd16(p + 2 * i); uint32_t s = (h & 0x8000u) << 16, ex = (h >> 10) & 31, m = h & 1023, f;
        if (ex == 0) { if (!m) f = s; else { int e2 = -1; do { e2++; m <<= 1; } while (!(m & 1024)); m &= 1023; f = s | ((uint32_t)(127 - 15 - e2) << 23) | (m << 13); } }
        else if (ex == 31) f = s | 0x7f800000u | (m << 13); else f = s | ((ex + 112) << 23) | (m << 13);
        memcpy(o + i, &f, 4);
      }
      t.data = t.owned.data(); t.kind = Kind::F32;
    }
    tensors_[t.name] = std::move(t);
  }
}

Container::~Container() { if (map_ && map_ != MAP_FAILED) munmap(map_, size_); if (fd_ >= 0) close(fd_); }

const uint8_t *Container::blob(const std::string &name, size_t *len) const {
  auto it = toc_.find(name); if (it == toc_.end()) throw std::runtime_error("missing blob " + name);
  if (len) *len = it->second.second;
  return map_ + it->second.first;
}
std::vector<std::string> Container::blob_names() const {
  std::vector<std::pair<uint64_t, std::string>> v; for (auto &kv : toc_) v.push_back({kv.second.first, kv.first});
  std::sort(v.begin(), v.end()); std::vector<std::string> o; for (auto &x : v) o.push_back(x.second); return o;
}

void write_container(const std::string &path, const std::vector<std::pair<std::string, std::string>> &blobs, bool qpack2) {
  const uint64_t A = 64; uint64_t off = (16 + 72 * blobs.size() + A - 1) / A * A;
  std::string head(qpack2 ? "QPACK2\0\0" : "QPACK1\0\0", 8); uint32_t n = (uint32_t)blobs.size(), pad = 0;
  head.append((const char *)&n, 4); head.append((const char *)&pad, 4);
  std::vector<uint64_t> offs;
  for (auto &b : blobs) {
    if (b.first.size() > 55) throw std::runtime_error("blob name too long: " + b.first);
    std::string name = b.first; name.resize(56, '\0'); uint64_t len = b.second.size();
    head += name; head.append((const char *)&off, 8); head.append((const char *)&len, 8);
    offs.push_back(off); off = (off + len + A - 1) / A * A;
  }
  FILE *f = fopen(path.c_str(), "wb"); if (!f) throw std::runtime_error("cannot write " + path);
  fwrite(head.data(), 1, head.size(), f); uint64_t pos = head.size(); static const char zeros[64] = {};
  for (size_t i = 0; i < blobs.size(); i++) {
    fwrite(zeros, 1, offs[i] - pos, f); fwrite(blobs[i].second.data(), 1, blobs[i].second.size(), f); pos = offs[i] + blobs[i].second.size();
  }
  if (fclose(f) != 0) throw std::runtime_error("write failed: " + path);
}

std::string Container::text(const std::string &name) const { size_t n; const uint8_t *p = blob(name, &n); return std::string((const char *)p, n); }
const Tensor &Container::at(const std::string &name) const { auto *t = get(name); if (!t) throw std::runtime_error("missing tensor " + name); return *t; }

}  // namespace qp
