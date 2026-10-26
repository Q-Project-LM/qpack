// libqpack: load QPACK1/QPACK2 single-file models and run them on the CPU (AVX2/FMA, threads).
//   Container  — mmap of a .qpack file: table of contents, manifest, tensors viewed in place (no copies).
//   Tensor     — f32 / f16 / q8 (int8 rows + float32 per-row scale) / i64, with matvec() kernels.
//   Model      — one decode step at a time: Llama family (Llama, Qwen2, Mistral, SmolLM) and N-Q (Gated DeltaNet).
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "qpack/json.h"

namespace qp {

// ------------------------------------------------------------------ threads
// A persistent pool: parallel_for(n, f) runs f(begin, end) on contiguous slices of [0, n).
class ThreadPool {
 public:
  explicit ThreadPool(int n);
  ~ThreadPool();
  int size() const { return n_; }
  void parallel_for(int n, const std::function<void(int, int)> &f, int min_chunk = 16);
 private:
  struct Impl; std::unique_ptr<Impl> impl_; int n_;
};
ThreadPool &pool();             // process-wide pool (QPACK_THREADS env or hardware threads, capped)
void set_threads(int n);         // call before the first pool() use

// ------------------------------------------------------------------ container
enum class Kind { F32, F16, Q8, Q4, I64, Tern5, Bits };
struct Tensor {
  std::string name;
  std::vector<int64_t> shape;
  Kind kind = Kind::F32;
  const void *data = nullptr;          // points into the mmap (or into `owned`)
  const float *scale = nullptr;        // q8: one per row; q4: one per group of `group` columns (as float)
  int group = 0;                       // q4 group size
  std::vector<uint8_t> owned;          // storage for tensors converted at load time
  int64_t rows() const { return shape.empty() ? 0 : shape[0]; }
  int64_t cols() const { int64_t c = 1; for (size_t i = 1; i < shape.size(); i++) c *= shape[i]; return shape.size() == 1 ? shape[0] : c; }
  int64_t numel() const { int64_t n = 1; for (auto s : shape) n *= s; return n; }
  const float *f32() const { return (const float *)data; }
};

class Container {
 public:
  explicit Container(const std::string &path);
  ~Container();
  int version() const { return version_; }                 // 1 or 2
  bool has(const std::string &blob) const { return toc_.count(blob) > 0; }
  std::string text(const std::string &blob) const;
  Json json(const std::string &blob) const { return Json::parse(text(blob)); }
  const uint8_t *blob(const std::string &name, size_t *len) const;
  const Json &manifest() const { return manifest_; }
  const Json &config() const { return config_; }
  std::unordered_map<std::string, Tensor> &tensors() { return tensors_; }
  const Tensor *get(const std::string &name) const { auto it = tensors_.find(name); return it == tensors_.end() ? nullptr : &it->second; }
  const Tensor &at(const std::string &name) const;
  size_t file_size() const { return size_; }
 private:
  int fd_ = -1; uint8_t *map_ = nullptr; size_t size_ = 0; int version_ = 1;
  std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> toc_;
  Json manifest_, config_;
  std::unordered_map<std::string, Tensor> tensors_;
};

// ------------------------------------------------------------------ kernels
// y[0..rows) = W x for a 2-D tensor (rows x cols); threaded over rows when the matrix is large enough.
void matvec(const Tensor &W, const float *x, float *y, int rows, int cols);
void row_f32(const Tensor &W, int64_t r, float *out);       // one row as float (embedding / table lookups)
void rmsnorm(const float *x, const float *w, int n, float eps, float *out, bool zero_centred = false);

// ------------------------------------------------------------------ models
class Model {
 public:
  virtual ~Model() = default;
  virtual int vocab() const = 0;
  virtual int hidden() const = 0;
  virtual int max_ctx() const = 0;
  virtual int pos() const = 0;
  // one token; want: 0 = update the state only, 1 = return logits (vocab floats), 2 = return the final hidden state
  virtual const float *step(int token, int want = 1) = 0;
  virtual void reset() = 0;
  // drop the state after the first `pos` tokens; false if the model cannot (recurrent state without a snapshot)
  virtual bool rewind(int pos) { return pos == this->pos(); }
  virtual bool recurrent() const { return false; }   // state is not a per-token cache (prefix reuse needs snapshots)
  virtual std::vector<uint8_t> snapshot() const = 0;
  virtual void restore(const std::vector<uint8_t> &s) = 0;
  virtual std::string arch() const = 0;
  // decision head (N-Q JEV models) on a final hidden state from step(.., 2); false if the model has none
  virtual bool decision(const float *hidden, float *score) const { return false; }
  const Json &config() const { return cfg_; }
  Container &container() { return *c_; }
 protected:
  std::shared_ptr<Container> c_; Json cfg_;
};
// opens any supported .qpack; max_ctx = 0 picks a default (4096, or the model's limit if smaller)
std::unique_ptr<Model> load_model(const std::string &path, int max_ctx = 0);
std::unique_ptr<Model> make_llama(std::shared_ptr<Container> c, int max_ctx);
std::unique_ptr<Model> make_nq(std::shared_ptr<Container> c, int max_ctx);

// self-test stored in the container: top-1 must match and the top-5 logits must agree within tol
struct SelfTest { int ok = 0, n = 0; double max_err = 0; };
SelfTest run_selftest(Model &m, double tol = 5e-2);

}  // namespace qp
