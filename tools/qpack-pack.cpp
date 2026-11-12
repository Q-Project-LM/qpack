// qpack-pack: lossless entropy coding of a QPACK2 file's blobs (static rANS per blob / per 16 MiB chunk, QPACK3 container).
// usage: qpack-pack in.qpack out.qpack [--only classes] [--unpack] [-t N]
//   classes (comma list, default all): q4 q8 scales(fp16/e8m0 scale blobs + small f32) embed(large f16) mx4 q8g q1
// The output is decoded back to plain QPACK2 data at load time (Container), so the kernels are untouched. Blobs left out of
// --only stay raw and mmap-lazy.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <sstream>
#include <thread>

#include "qpack/qpack.h"
#include "qpack/rans.h"

using qp::Json;

struct Plan { int codec, planes; std::string cls; };

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: qpack-pack in.qpack out.qpack [--only classes] [--unpack] [-t N]\n"); return 2; }
  bool unpack = false; std::set<std::string> only; unsigned nt = std::max(1u, std::min(64u, std::thread::hardware_concurrency()));
  for (int i = 3; i < argc; i++) {
    if (!strcmp(argv[i], "--unpack")) unpack = true;
    else if (!strcmp(argv[i], "--only") && i + 1 < argc) { std::stringstream ss(argv[++i]); std::string c; while (std::getline(ss, c, ',')) only.insert(c); }
    else if (!strcmp(argv[i], "-t") && i + 1 < argc) nt = (unsigned)atoi(argv[++i]);
    else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  try {
    qp::Container src(argv[1]);
    Json man = src.manifest();
    std::vector<std::pair<std::string, std::string>> all;
    if (unpack) {
      Json m2 = Json::object(); for (auto &kv : man.obj) if (kv.first != "encoded") m2.set(kv.first, kv.first == "format" ? Json("QPACK2") : kv.second);
      for (auto &n : src.blob_names()) all.push_back({n, n == "manifest.json" ? m2.dump() : src.text(n)});
      qp::write_container(argv[2], all, true); printf("unpacked -> %s\n", argv[2]); return 0;
    }
    auto blen = [&](const std::string &b) { size_t l; src.blob(b, &l); return l; };
    auto want = [&](const std::string &c) { return only.empty() || only.count(c); };
    std::map<std::string, Plan> plan;   // blob -> codec, planes, class
    auto add = [&](const std::string &blob, int codec, int planes, const char *cls) { if (want(cls)) plan[blob] = {codec, planes, cls}; };
    for (auto &e : man["tensors"].arr) {
      const std::string &k = e["kind"].as_str(); const std::string &b = e["blob"].as_str();
      if (k == "q4") { add(b, 2, 1, "q4"); add(e["scale"].as_str(), 1, 2, "scales"); }
      else if (k == "q8") add(b, 1, 1, "q8");
      else if (k == "mxfp4") { add(b, 1, 1, "mx4"); add(e["scale"].as_str(), 1, 1, "scales"); }
      else if (k == "q1") { add(b, 1, 1, "q1"); add(e["scale"].as_str(), 1, 2, "scales"); }
      else if (k == "q8g") { add(b, 1, 1, "q8g"); add(e["scale"].as_str(), 1, 2, "scales"); }
      else if (k == "f16") add(b, 1, 2, e["shape"].arr.size() == 2 && blen(b) > (1u << 20) ? "embed" : "scales");
      else if (k == "f32") add(b, 1, 4, "scales");
    }
    const size_t kThresh = 32u << 20, kChunk = 16u << 20;
    auto t0 = std::chrono::steady_clock::now();
    for (auto &n : src.blob_names()) all.push_back({n, n == "manifest.json" ? "" : src.text(n)});
    struct Job { size_t blob; size_t off, len; std::vector<uint8_t> out; };
    std::vector<Job> jobs; std::vector<size_t> first(all.size(), 0), count(all.size(), 0); std::vector<char> chunked(all.size(), 0);
    for (size_t i = 0; i < all.size(); i++) {
      auto it = plan.find(all[i].first); const std::string &d = all[i].second;
      if (it == plan.end() || d.empty()) continue;
      first[i] = jobs.size();
      if (d.size() > kThresh) { chunked[i] = 1; for (size_t o = 0; o < d.size(); o += kChunk) jobs.push_back({i, o, std::min(kChunk, d.size() - o), {}}); }
      else jobs.push_back({i, 0, d.size(), {}});
      count[i] = jobs.size() - first[i];
    }
    std::atomic<size_t> next{0}; std::vector<std::thread> th;
    for (unsigned t = 0; t < nt; t++) th.emplace_back([&] {
      for (size_t j; (j = next++) < jobs.size();) {
        Job &jb = jobs[j]; const Plan &pl = plan[all[jb.blob].first];
        jb.out = qp::rans::encode((const uint8_t *)all[jb.blob].second.data() + jb.off, jb.len, pl.codec, pl.planes);
      }
    });
    for (auto &t : th) t.join();
    Json enc = Json::array(); std::map<std::string, std::pair<size_t, size_t>> byClass;
    for (size_t i = 0; i < all.size(); i++) {
      if (!count[i]) continue;
      const Plan &pl = plan[all[i].first]; std::string &d = all[i].second; std::vector<uint8_t> c;
      if (chunked[i]) { std::vector<std::vector<uint8_t>> ch; for (size_t j = 0; j < count[i]; j++) ch.push_back(std::move(jobs[first[i] + j].out)); c = qp::rans::assemble_chunked(d.size(), kChunk, ch); }
      else c = std::move(jobs[first[i]].out);
      auto &bc = byClass[pl.cls]; bc.first += d.size();
      if (c.size() < d.size()) { bc.second += c.size(); d.assign((const char *)c.data(), c.size()); enc.push(Json(all[i].first)); }
      else bc.second += d.size();
    }
    man.set("encoded", enc); man.set("format", "QPACK3");
    for (auto &b : all) if (b.first == "manifest.json") b.second = man.dump();
    qp::write_container(argv[2], all, true, "QPACK3\0\0");
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    size_t raw = 0, packed = 0;
    for (auto &kv : byClass) { printf("  %-8s %10.2f -> %10.2f MiB (%.1f%%)\n", kv.first.c_str(), kv.second.first / 1048576.0, kv.second.second / 1048576.0, 100.0 * (1 - (double)kv.second.second / kv.second.first)); raw += kv.second.first; packed += kv.second.second; }
    printf("coded blobs %zu: %.2f MiB -> %.2f MiB (%.1f%%), file %.2f -> %.2f MiB, %.1f s\n", enc.arr.size(), raw / 1048576.0, packed / 1048576.0,
           100.0 * (1 - (double)packed / std::max<size_t>(raw, 1)), src.file_size() / 1048576.0, qp::Container(argv[2]).file_size() / 1048576.0, s);
  } catch (std::exception &e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
}
