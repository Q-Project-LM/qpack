p = '/workspace/qpack/.claude/worktrees/agent-a7197741fb8c38627/src/model_llama_cuda.cu'
s = open(p).read()

# kernel signature
s = s.replace("float *part, const Ctl *ctl, int rep, int nkv, int nch, float scl, int dbg) {",
              "float *part, const Ctl *ctl, int rep, int nkv, int nch, float scl, int dbg, int *cnt, int8_t *xq, float *xs, int *xsum, int np) {", 1)

# tail: last block per kv group merges
old_tail = "  CLK(5);\n}\n"
new_tail = """  CLK(5);
  if (!cnt) return;
  // the last chunk block of this kv group to finish merges the partials of the group's heads and quantises the result (same arithmetic as k_merge)
  __threadfence(); __syncthreads();
  __shared__ int s_last;
  if (tid == 0) { const int old = atomicAdd(cnt + g, 1); s_last = old == ((T + CH - 1) / CH) - 1; if (s_last) cnt[g] = 0; }
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  const int nc = (T + CH - 1) / CH;
#pragma unroll
  for (int i = 0; i < NA; i++) {
    const int j = grp + i * GR; if (j >= rep) break;
    const int h = g * rep + j; const float *ph = part + (size_t)h * nch * (HD + 2); float M = -1e30f;
    for (int cc = 0; cc < nc; cc++) M = fmaxf(M, __ldcg(ph + (size_t)cc * (HD + 2)));
    float L = 0, o = 0;
    for (int cc = 0; cc < nc; cc++) { const float *pc = ph + (size_t)cc * (HD + 2); const float f = expf(__ldcg(pc) - M); L += __ldcg(pc + 1) * f; o += f * __ldcg(pc + 2 + d); }
    o /= L;
    warp_actq(o, h * (HD / 32) + (d >> 5), d & 31, xq, xs, xsum, np);
  }
}
"""
# only the k_attn tail: first occurrence of CLK(5);
i = s.index(old_tail)
s = s[:i] + new_tail + s[i + len(old_tail):]

# host: counter
s = s.replace("fl(&x_, d_);", "fl(&x_, d_); cnt_ = (int *)ab(64);", 1)
s = s.replace("    CK(cudaMalloc(&arena_, arena_n)); size_t aoff = 0;", "    CK(cudaMalloc(&arena_, arena_n)); CK(cudaMemset(arena_, 0, 4096)); size_t aoff = 0;", 1)
s = s.replace("float *x_, *q_,", "int *cnt_; float *x_, *q_,", 1)
s = s.replace("part_, ctl_, rep, nkv_, nch_, scl, dbg_);\n      else k_attn<128>", "part_, ctl_, rep, nkv_, nch_, scl, dbg_, fmerge_ ? cnt_ : nullptr, aq_[0], as_[0], asum_[0], P(2));\n      else k_attn<128>", 1)
s = s.replace("kl, vl, part_, ctl_, rep, nkv_, nch_, scl, dbg_);\n      if (hd_ == 64) k_merge", "kl, vl, part_, ctl_, rep, nkv_, nch_, scl, dbg_, fmerge_ ? cnt_ : nullptr, aq_[0], as_[0], asum_[0], P(2));\n      if (fmerge_) {} else if (hd_ == 64) k_merge", 1)
s = s.replace("  const int gu_nw_ =", "  const int fmerge_ = getenv(\"QPACK_CUDA_FMERGE\") ? atoi(getenv(\"QPACK_CUDA_FMERGE\")) : 1;   // merge of the attention chunks in the last block of each kv group\n  const int gu_nw_ =", 1)
open(p, 'w').write(s)
