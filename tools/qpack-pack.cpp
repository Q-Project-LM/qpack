// qpack-pack: lossless entropy coding of a QPACK2 file's q4 / q8 / fp16-scale blobs (static rANS per blob, QPACK3 container).
// usage: qpack-pack in.qpack out.qpack [--unpack]
// The output is decoded back to plain QPACK2 data at load time (Container), so the kernels are untouched.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>

#include "qpack/qpack.h"
#include "qpack/rans.h"

using qp::Json;

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: qpack-pack in.qpack out.qpack [--unpack]\n"); return 2; }
  bool unpack = argc > 3 && !strcmp(argv[3], "--unpack");
  try {
    qp::Container src(argv[1]);
    Json man = src.manifest();
    std::vector<std::pair<std::string, std::string>> all;
    if (unpack) {
      Json m2 = Json::object(); for (auto &kv : man.obj) if (kv.first != "encoded") m2.set(kv.first, kv.first == "format" ? Json("QPACK2") : kv.second);
      for (auto &n : src.blob_names()) all.push_back({n, n == "manifest.json" ? m2.dump() : src.text(n)});
      qp::write_container(argv[2], all, true); printf("unpacked -> %s\n", argv[2]); return 0;
    }
    std::map<std::string, std::pair<int, int>> plan;   // blob -> codec, planes
    for (auto &e : man["tensors"].arr) {
      const std::string &k = e["kind"].as_str();
      if (k == "q4") { plan[e["blob"].as_str()] = {2, 1}; plan[e["scale"].as_str()] = {1, 2}; }
      else if (k == "q8") plan[e["blob"].as_str()] = {1, 1};
    }
    Json enc = Json::array(); size_t raw = 0, packed = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (auto &n : src.blob_names()) {
      std::string data = n == "manifest.json" ? "" : src.text(n);
      auto it = plan.find(n);
      if (it != plan.end() && !data.empty()) {
        auto c = qp::rans::encode((const uint8_t *)data.data(), data.size(), it->second.first, it->second.second);
        raw += data.size();
        if (c.size() < data.size()) { packed += c.size(); data.assign((const char *)c.data(), c.size()); enc.push(Json(n)); }
        else packed += data.size();
      }
      all.push_back({n, std::move(data)});
    }
    man.set("encoded", enc); man.set("format", "QPACK3");
    for (auto &b : all) if (b.first == "manifest.json") b.second = man.dump();
    qp::write_container(argv[2], all, true, "QPACK3\0\0");
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("coded blobs %zu: %.2f MiB -> %.2f MiB (%.1f%%), file %.2f -> %.2f MiB, %.1f s\n", enc.arr.size(), raw / 1048576.0, packed / 1048576.0,
           100.0 * (1 - (double)packed / raw), src.file_size() / 1048576.0, qp::Container(argv[2]).file_size() / 1048576.0, s);
  } catch (std::exception &e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
}
