"""Compare C++ engine output (bench/dump_logits: logits .f32, QPACK_DUMP hidden-state file) with ref_gptoss.py's npz.
    python -m qpack.cmp_gptoss ref.npz cpp_logits.f32 [cpp_hidden.bin]
The hidden file holds the last token's hidden state after the embedding and after every layer (float32 [d] each); only the last
(layers + 1) vectors are used (a token-by-token run dumps every token)."""
import sys
import numpy as np

ref = np.load(sys.argv[1]); lg = np.fromfile(sys.argv[2], dtype=np.float32)
if len(sys.argv) > 3:
    H = ref["hidden"]; n = H.shape[0]; d = H.shape[1]
    h = np.fromfile(sys.argv[3], dtype=np.float32).reshape(-1, d)[-n:]
    for i in range(n):
        e = np.abs(h[i] - H[i]).max(); rel = np.linalg.norm(h[i] - H[i]) / np.linalg.norm(H[i])
        print(f"hidden[{i}] ({'embed' if i == 0 else 'after layer %d' % (i - 1)}): max abs err {e:.4g}, rel L2 {rel:.3e}, ref |max| {np.abs(H[i]).max():.3g}")
rl = ref["logits"]
if np.abs(rl).sum() > 0 and "layers" not in sys.argv:
    t_ref = np.argsort(-rl)[:5]; t_cpp = np.argsort(-lg)[:5]
    print("ref top5", t_ref.tolist(), np.round(rl[t_ref], 3).tolist()); print("cpp top5", t_cpp.tolist(), np.round(lg[t_cpp], 3).tolist())
    print("top-5 ids equal:", t_ref.tolist() == t_cpp.tolist(), " top-1 equal:", t_ref[0] == t_cpp[0])
    print("max abs logit err on the ref top-5: %.4g; over all vocab: max %.4g, rel L2 %.3e" % (np.abs(rl[t_ref] - lg[t_ref]).max(), np.abs(rl - lg).max(), np.linalg.norm(rl - lg) / np.linalg.norm(rl)))
