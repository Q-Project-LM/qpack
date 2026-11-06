"""Layer-streaming float32 reference for gpt-oss-20b (numpy only; follows openai/gpt-oss torch/model.py semantics).
Weights are dequantised one layer (and one expert) at a time from the safetensors mmap, so the 13.8 GB model never has to fit in RAM.

    python -m qpack.ref_gptoss /workspace/qpack-models/gpt-oss-20b --ids 200006,1428,... --out ref.npz [--layers N]

Writes: logits (last position, float32 [vocab]), hidden (float32 [layers + 1, d]: last-position hidden state after the embedding and
after every layer, the same thing QPACK_DUMP produces in the C++ engine), ids, and a selftest.json next to the npz when --selftest is given.
"""
import argparse, glob, json, math, os, sys, time

import numpy as np

from .convert_gptoss import SafeTensors, dq_mx4_rows


def rms(x, w, eps=1e-5):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def yarn(head_dim, theta, factor, orig, beta_fast, beta_slow):
    half = head_dim // 2
    freq = theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    conc = 0.1 * math.log(factor) + 1.0
    low = half * math.log(orig / (beta_fast * 2 * math.pi)) / math.log(theta)
    high = half * math.log(orig / (beta_slow * 2 * math.pi)) / math.log(theta)
    ramp = np.clip((np.arange(half) - low) / (high - low), 0, 1); mask = 1 - ramp
    return (1.0 / (factor * freq)) * (1 - mask) + (1.0 / freq) * mask, conc


def rotate(x, cos, sin):   # x [T, heads, hd]; cos/sin [T, hd/2]
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    c, s = cos[:, None, :], sin[:, None, :]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)


def run(src, ids, nlayers=0, log=True):
    st = SafeTensors(sorted(glob.glob(os.path.join(src, "model-*.safetensors"))))
    cfg = json.load(open(os.path.join(src, "config.json"))); L = nlayers or cfg["num_hidden_layers"]
    nh, nkv, hd, d = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["hidden_size"]; T = len(ids)
    rs = cfg["rope_scaling"]; inv, conc = yarn(hd, cfg["rope_theta"], rs["factor"], rs["original_max_position_embeddings"], rs["beta_fast"], rs["beta_slow"])
    ang = np.arange(T)[:, None] * inv[None, :]; cos = (np.cos(ang) * conc).astype(np.float32); sin = (np.sin(ang) * conc).astype(np.float32)
    emb = np.stack([st.get("model.embed_tokens.weight", int(i), int(i) + 1)[0] for i in ids]).astype(np.float32)
    x = emb; hidden = [x[-1].copy()]; t0 = time.time()
    g = lambda k: st.get(k).astype(np.float32)
    for l in range(L):
        p = f"model.layers.{l}."
        # attention
        t = rms(x, g(p + "input_layernorm.weight"))
        q = t @ g(p + "self_attn.q_proj.weight").T + g(p + "self_attn.q_proj.bias")
        k = t @ g(p + "self_attn.k_proj.weight").T + g(p + "self_attn.k_proj.bias")
        v = t @ g(p + "self_attn.v_proj.weight").T + g(p + "self_attn.v_proj.bias")
        q = rotate(q.reshape(T, nh, hd), cos, sin); k = rotate(k.reshape(T, nkv, hd), cos, sin); v = v.reshape(T, nkv, hd)
        sinks = g(p + "self_attn.sinks")
        mask = np.triu(np.full((T, T), -np.inf, np.float32), 1)
        if l % 2 == 0: mask = mask + np.tril(np.full((T, T), -np.inf, np.float32), -cfg["sliding_window"])
        out = np.zeros((T, nh, hd), np.float32)
        for h in range(nh):
            kv = h // (nh // nkv)
            s = (q[:, h] @ k[:, kv].T) * 0.125 + mask
            s = np.concatenate([s, np.broadcast_to(sinks[h], (T, 1))], 1)
            w = np.exp(s - s.max(1, keepdims=True)); w = (w / w.sum(1, keepdims=True))[:, :-1]
            out[:, h] = w @ v[:, kv]
        x = x + out.reshape(T, nh * hd) @ g(p + "self_attn.o_proj.weight").T + g(p + "self_attn.o_proj.bias")
        # MoE
        t = rms(x, g(p + "post_attention_layernorm.weight"))
        lg = t @ g(p + "mlp.router.weight").T + g(p + "mlp.router.bias")
        top = np.argsort(-lg, 1, kind="stable")[:, :cfg["num_experts_per_tok"]]
        tv = np.take_along_axis(lg, top, 1); ew = np.exp(tv - tv.max(1, keepdims=True)); ew /= ew.sum(1, keepdims=True)
        gub, dnb = g(p + "mlp.experts.gate_up_proj_bias"), g(p + "mlp.experts.down_proj_bias")
        moe = np.zeros_like(x)
        for e in sorted(set(top.reshape(-1).tolist())):
            tok, slot = np.nonzero(top == e)
            wgu = dq_mx4_rows(st.get(p + "mlp.experts.gate_up_proj_blocks", e, e + 1)[0], st.get(p + "mlp.experts.gate_up_proj_scales", e, e + 1)[0])
            y = t[tok] @ wgu.T + gub[e]; glu, lin = np.minimum(y[:, ::2], 7.0), np.clip(y[:, 1::2], -7.0, 7.0)
            h = glu / (1 + np.exp(-1.702 * glu)) * (lin + 1)
            wd = dq_mx4_rows(st.get(p + "mlp.experts.down_proj_blocks", e, e + 1)[0], st.get(p + "mlp.experts.down_proj_scales", e, e + 1)[0])
            o = h @ wd.T + dnb[e]
            np.add.at(moe, tok, o * ew[tok, slot][:, None])
        x = x + moe; hidden.append(x[-1].copy())
        if log: print(f"layer {l} done ({time.time() - t0:.0f}s) experts {top[-1].tolist()}", file=sys.stderr, flush=True)
    xn = rms(x[-1], g("model.norm.weight"))
    V = cfg["vocab_size"]; logits = np.zeros(V, np.float32)
    for r0 in range(0, V, 16384):
        logits[r0:r0 + 16384] = st.get("lm_head.weight", r0, min(V, r0 + 16384)) @ xn
    return logits, np.stack(hidden), xn


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("--ids", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=0); ap.add_argument("--selftest", default=None, help="write a QPACK selftest.json case list")
    a = ap.parse_args(); cases = [list(map(int, c.split(","))) for c in a.ids.split(";")]; st_out = []
    for n, ids in enumerate(cases):
        logits, hidden, xn = run(a.src, ids, a.layers)
        np.savez(a.out if len(cases) == 1 else a.out.replace(".npz", f".{n}.npz"), logits=logits, hidden=hidden, ids=np.array(ids))
        top = np.argsort(-logits)[:5]; print("top5", top.tolist(), logits[top].tolist(), file=sys.stderr)
        st_out.append(dict(ids=ids, top5=top.tolist(), top5_logits=logits[top].tolist()))
    if a.selftest: json.dump(dict(cases=st_out), open(a.selftest, "w"))


if __name__ == "__main__":
    main()
