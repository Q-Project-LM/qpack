// Per-architecture basics shared by the kernels: which SIMD flavour is compiled in, fp16 <-> fp32 scalars, spin-wait hint.
#pragma once
#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#define QP_X86 1
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#define QP_NEON 1
#include <arm_neon.h>
#else
#error "qpack needs x86-64 (AVX2 + FMA + F16C) or AArch64 (NEON)"
#endif

namespace qp {

#if QP_X86
inline float f16_to_f32(uint16_t h) { return _cvtsh_ss(h); }
inline uint16_t f32_to_f16(float f) { return _cvtss_sh(f, 0); }
inline void cpu_relax() { _mm_pause(); }
#else
inline float f16_to_f32(uint16_t h) { __fp16 x; memcpy(&x, &h, 2); return (float)x; }
inline uint16_t f32_to_f16(float f) { __fp16 x = (__fp16)f; uint16_t h; memcpy(&h, &x, 2); return h; }
inline void cpu_relax() { asm volatile("yield" ::: "memory"); }
#endif

}  // namespace qp
