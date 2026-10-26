// Persistent thread pool tuned for many short jobs (one decode step is ~100 matrix-vector products): workers spin on a
// generation counter for a while before sleeping, and the calling thread takes part in every job.
#include <immintrin.h>

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>

#include "qpack/qpack.h"

namespace qp {

struct ThreadPool::Impl {
  std::vector<std::thread> threads;
  std::atomic<uint64_t> gen{0};
  std::atomic<int> next{0}, pending{0};
  std::atomic<bool> stop{false};
  const std::function<void(int, int)> *job = nullptr;
  int n = 0, chunk = 1;
  std::mutex m; std::condition_variable cv;

  void work() { for (;;) { int b = next.fetch_add(chunk); if (b >= n) break; (*job)(b, b + chunk < n ? b + chunk : n); } }
  void loop() {
    uint64_t seen = 0;
    while (!stop.load(std::memory_order_acquire)) {
      int spins = 0;
      while (gen.load(std::memory_order_acquire) == seen && !stop.load(std::memory_order_relaxed)) {
        if (++spins < 20000) _mm_pause();
        else { std::unique_lock<std::mutex> lk(m); cv.wait_for(lk, std::chrono::milliseconds(2)); spins = 0; }
      }
      if (stop.load()) break;
      seen = gen.load(std::memory_order_acquire);
      work();
      pending.fetch_sub(1, std::memory_order_acq_rel);
    }
  }
};

ThreadPool::ThreadPool(int n) : impl_(new Impl), n_(n < 1 ? 1 : n) {
  for (int i = 1; i < n_; i++) impl_->threads.emplace_back([this] { impl_->loop(); });
}
ThreadPool::~ThreadPool() {
  impl_->stop = true; impl_->cv.notify_all();
  for (auto &t : impl_->threads) t.join();
}
void ThreadPool::parallel_for(int n, const std::function<void(int, int)> &f, int min_chunk) {
  if (n_ == 1 || n <= min_chunk) { f(0, n); return; }
  Impl &I = *impl_;
  int chunk = n / (n_ * 4); if (chunk < min_chunk) chunk = min_chunk;
  I.job = &f; I.n = n; I.chunk = chunk; I.next = 0; I.pending = n_ - 1;
  I.gen.fetch_add(1, std::memory_order_acq_rel); I.cv.notify_all();
  I.work();
  while (I.pending.load(std::memory_order_acquire) > 0) _mm_pause();
}

static int g_threads = 0;
void set_threads(int n) { g_threads = n; }
ThreadPool &pool() {
  static ThreadPool *p = [] {
    int n = g_threads;
    if (n <= 0) { const char *e = getenv("QPACK_THREADS"); n = e ? atoi(e) : (int)std::thread::hardware_concurrency(); if (n > 16) n = 16; }
    return new ThreadPool(n);
  }();
  return *p;
}

}  // namespace qp
