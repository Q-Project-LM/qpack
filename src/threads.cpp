// Persistent thread pool tuned for many short jobs (one decode step is ~170 matrix-vector products): every job is split
// into ~4 chunks per thread taken from a shared counter (a slow core on a busy machine gets help), workers spin on a generation counter and report
// completion through their own cache line, and the condition variable is only touched when a worker went to sleep.
// The calling thread runs slice 0. QPACK_PIN=1 pins thread i to CPU i.
#include <pthread.h>
#if defined(__linux__) && !defined(__ANDROID__)
#include <sched.h>
#endif
#ifdef __APPLE__
#include <pthread/qos.h>
#endif

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>

#include "qpack/qpack.h"
#include "qpack/simd.h"

namespace qp {

struct alignas(64) Done { std::atomic<uint64_t> gen{0}; char pad[56]; };

struct ThreadPool::Impl {
  std::vector<std::thread> threads;
  alignas(64) std::atomic<uint64_t> gen{0};
  alignas(64) std::atomic<int> sleepers{0};
  std::atomic<bool> stop{false};
  const std::function<void(int, int)> *job = nullptr;
  int n = 0, parts = 1, T = 1, chunk = 1;
  alignas(64) std::atomic<int> next{0};
  std::unique_ptr<Done[]> done;
  std::mutex m; std::condition_variable cv;

  void slice(int) { for (;;) { const int a = next.fetch_add(chunk, std::memory_order_relaxed); if (a >= n) return; (*job)(a, a + chunk < n ? a + chunk : n); } }
  void loop(int id) {
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);   // keep workers on the performance cores
#endif
    uint64_t seen = 0;
    for (;;) {
      int spins = 0; uint64_t g;
      while ((g = gen.load(std::memory_order_acquire)) == seen) {
        if (stop.load(std::memory_order_relaxed)) return;
        if (++spins < 200000) { cpu_relax(); continue; }
        std::unique_lock<std::mutex> lk(m); sleepers.fetch_add(1);
        cv.wait(lk, [&] { return gen.load(std::memory_order_acquire) != seen || stop.load(); });
        sleepers.fetch_sub(1); spins = 0;
      }
      if (stop.load()) return;
      seen = g; slice(id);
      done[id].gen.store(g, std::memory_order_release);
    }
  }
};

ThreadPool::ThreadPool(int n) : impl_(new Impl), n_(n < 1 ? 1 : n) {
  impl_->T = n_; impl_->done.reset(new Done[n_]);
  const char *pin = getenv("QPACK_PIN"); const bool do_pin = pin && atoi(pin) > 0;
#if defined(__linux__) && !defined(__ANDROID__)
  auto pin_to = [](std::thread::native_handle_type h, int cpu) { cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s); pthread_setaffinity_np(h, sizeof s, &s); };
#else
  auto pin_to = [](std::thread::native_handle_type, int) {};   // no pthread affinity API (macOS, bionic)
#endif
  for (int i = 1; i < n_; i++) { impl_->threads.emplace_back([this, i] { impl_->loop(i); }); if (do_pin) pin_to(impl_->threads.back().native_handle(), i); }
  if (do_pin) pin_to(pthread_self(), 0);
#ifdef __APPLE__
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}
ThreadPool::~ThreadPool() {
  { std::lock_guard<std::mutex> lk(impl_->m); impl_->stop = true; }
  impl_->cv.notify_all();
  for (auto &t : impl_->threads) t.join();
}
void ThreadPool::parallel_for(int n, const std::function<void(int, int)> &f, int min_chunk) {
  if (n_ == 1 || n <= min_chunk) { f(0, n); return; }
  Impl &I = *impl_;
  int parts = (n + min_chunk - 1) / min_chunk; if (parts > n_) parts = n_;
  int chunk = (n + parts * 4 - 1) / (parts * 4); if (chunk < min_chunk) chunk = min_chunk;   // ~4 chunks per thread: stragglers get help
  I.job = &f; I.n = n; I.parts = parts; I.chunk = chunk; I.next.store(0, std::memory_order_relaxed);
  const uint64_t g = I.gen.fetch_add(1, std::memory_order_acq_rel) + 1;
  if (I.sleepers.load(std::memory_order_acquire) > 0) { std::lock_guard<std::mutex> lk(I.m); I.cv.notify_all(); }
  I.slice(0);
  for (int i = 1; i < n_; i++) while (I.done[i].gen.load(std::memory_order_acquire) != g) cpu_relax();
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
