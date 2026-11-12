// qpack-cudamv: CUDA matvec microbenchmark (q8 / q4 / q1, Qwen2.5-0.5B shapes); needs a -DQPACK_CUDA=ON build
#include <cstdio>
#include <string>
#include "qpack/qpack.h"
int main(int argc, char **argv) { try { if (argc > 1 && std::string(argv[1]) == "--gptoss") qp::cuda_gptoss_bench(stdout); else qp::cuda_matvec_bench(stdout); } catch (const std::exception &e) { fprintf(stderr, "error: %s\n", e.what()); return 1; } }
