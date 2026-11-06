// qpack-cudamv: CUDA matvec microbenchmark (q8 / q4 / q1, Qwen2.5-0.5B shapes); needs a -DQPACK_CUDA=ON build
#include <cstdio>
#include "qpack/qpack.h"
int main() { try { qp::cuda_matvec_bench(stdout); } catch (const std::exception &e) { fprintf(stderr, "error: %s\n", e.what()); return 1; } }
