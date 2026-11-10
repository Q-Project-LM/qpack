"""C8: per-expert allocation (exact MXFP4 / two-plane r2gl / one-plane q1g) under a GPU-memory budget, quantised with batched GPTQ.

    python -m qpack.convert_gptoss_c8 prepass            -> $OUT/contrib.json (routed output energy of every expert, from the calibration capture)
    python -m qpack.convert_gptoss_c8 alloc              -> alloc.json (greedy marginal-benefit allocation under the budget)
    python -m qpack.convert_gptoss_c8 convert SRC OUT --alloc alloc.json --cache DIR

Allocation model: expert e of layer l has weight c[l,e] = (its routed output energy) / (layer total) * v[l], v[l] = squared relative error of plain q1g in that layer
(measured 0.7^2 in layers 0-9, falling to ~0.24^2 at layer 23); relative squared error q1g 1.0, r2gl 0.28, exact 0 (layer-12 proxy: 0.565 vs 0.298 + exact).
Greedy by gain/byte from "all q1g" until the budget is spent. Layout (read by model_gptoss.cpp): config moe_exact / moe_r2 lists per layer;
w_gate_up / w_down = plane 1 for all non-exact experts (ascending), *_r2 = plane 2 for the moe_r2 experts (ascending), *_x = exact mxfp4."""
import argparse, glob, heapq, json, os, pickle, sys, time
import numpy as np
import torch

from . import convert_gptoss as CG
from . import c7_quant as Q
from .c7_lloyd import gptq_lloyd
from .c7_proxy import load_calib
from .container import StreamWriter, Container
from .convert_gptoss_c7 import pack_q1, group_scale_from_levels

OUT = "/workspace/qpack-models/alloc"
CONTRIB = os.environ.get("QPACK_CONTRIB", "contrib.json")
SRC = "/workspace/qpack-models/gpt-oss-20b"
COST = {"q1g": 3576960, "r2gl": 7153920, "exact": 13219200}
DENSE_GIB = 0.573 + 0.665   # head q8g + attention/router/norm/bias (measured)


def prepass(stride=8):
    st = CG.SafeTensors(sorted(glob.glob(f"{SRC}/model-*.safetensors"))); dv = Q.dev(); out = {}; t0 = time.time()
    for l in range(24):
        X, sel, w = load_calib(l); X, sel, w = X[::stride], sel[::stride], w[::stride]
        p = f"model.layers.{l}.mlp.experts."
        gub = torch.from_numpy(st.get(p + "gate_up_proj_bias").astype(np.float32)).to(dv); dnb = torch.from_numpy(st.get(p + "down_proj_bias").astype(np.float32)).to(dv)
        EA = torch.zeros(len(X), 4, device=dv)
        for e in range(32):
            tt, kk = torch.nonzero(sel == e, as_tuple=True)
            if len(tt) == 0: continue
            W1 = Q.dq_mx4(st.get(p + "gate_up_proj_blocks", e, e + 1)[0], st.get(p + "gate_up_proj_scales", e, e + 1)[0])
            W2 = Q.dq_mx4(st.get(p + "down_proj_blocks", e, e + 1)[0], st.get(p + "down_proj_scales", e, e + 1)[0])
            o = Q.swiglu(X[tt] @ W1.T + gub[e]) @ W2.T + dnb[e]; EA[tt, kk] = (o ** 2).sum(1) * w[tt, kk] ** 2
        frac = EA / EA.sum(1, keepdim=True).clamp_min(1e-30)   # per-token share of each routed expert: massive-activation tokens do not dominate
        c = [float(frac[sel == e].sum()) for e in range(32)]
        out[l] = c; print(f"  layer {l} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(f"{OUT}/contrib_tok.json", "w"))


def alloc(budget_gib=6.8, margin=0.01e9):
    con = json.load(open(f"{OUT}/contrib.json"))
    v = [0.7 ** 2] * 10 + [float((0.55 + (0.235 - 0.55) * (l - 10) / 13) ** 2) for l in range(10, 24)]
    budget = (budget_gib - DENSE_GIB) * 2 ** 30 - margin
    level = {(l, e): 0 for l in range(24) for e in range(32)}; used = 24 * 32 * COST["q1g"]
    c = {(l, e): con[str(l)][e] / max(sum(con[str(l)]), 1e-30) * v[l] for l in range(24) for e in range(32)}
    eps = [1.0, 0.28, 0.0]; names = ["q1g", "r2gl", "exact"]; heap = []
    push = lambda k: heapq.heappush(heap, (-(c[k] * (eps[level[k]] - eps[level[k] + 1])) / (COST[names[level[k] + 1]] - COST[names[level[k]]]), k)) if level[k] < 2 else None
    for k in level: push(k)
    while heap:
        r, k = heapq.heappop(heap); dc = COST[names[level[k] + 1]] - COST[names[level[k]]]
        if used + dc > budget: continue
        level[k] += 1; used += dc; push(k)
    res = {str(l): dict(exact=[e for e in range(32) if level[(l, e)] == 2], r2=[e for e in range(32) if level[(l, e)] == 1]) for l in range(24)}
    json.dump(res, open(f"{OUT}/alloc.json", "w"))
    print("experts bytes %.3f GiB (budget %.3f); exact %d, r2gl %d, q1g %d" % (used / 2 ** 30, budget / 2 ** 30, sum(len(r["exact"]) for r in res.values()), sum(len(r["r2"]) for r in res.values()),
          sum(32 - len(r["exact"]) - len(r["r2"]) for r in res.values())))
    for l in range(24): print(l, "exact", len(res[str(l)]["exact"]), "r2", len(res[str(l)]["r2"]), "q1g", 32 - len(res[str(l)]["exact"]) - len(res[str(l)]["r2"]))


def alloc_convex(budget_gib=6.8, margin=0.01e9, out="alloc_b.json"):
    """like alloc() but minimising sum_l E_l^2 (E_l = layer relative squared error under the model): equalises the layers instead of
    spending where one expert's share is largest (the linear objective left layer 6 all-q1g and gave worse PPL than the hand-made per-layer profile)"""
    con = json.load(open(f"{OUT}/" + CONTRIB))
    v = [0.7 ** 2] * 10 + [float((0.55 + (0.235 - 0.55) * (l - 10) / 13) ** 2) for l in range(10, 24)]
    budget = (budget_gib - DENSE_GIB) * 2 ** 30 - margin; used = 24 * 32 * COST["q1g"]
    eps = [1.0, 0.28, 0.0]; names = ["q1g", "r2gl", "exact"]
    share = {l: [x / max(sum(con[str(l)]), 1e-30) for x in con[str(l)]] for l in range(24)}
    lvl = {l: [0] * 32 for l in range(24)}
    E = {l: v[l] * sum(share[l][e] * eps[0] for e in range(32)) for l in range(24)}
    def best(l):
        b = None
        for e in range(32):
            k = lvl[l][e]
            if k >= 2: continue
            d = share[l][e] * v[l] * (eps[k] - eps[k + 1]); cost = COST[names[k + 1]] - COST[names[k]]
            g = (E[l] ** 2 - (E[l] - d) ** 2) / cost
            if b is None or g > b[0]: b = (g, e, cost, d)
        return b
    cand = {l: best(l) for l in range(24)}
    while True:
        ok = [(c[0], l) for l, c in cand.items() if c and used + c[2] <= budget]
        if not ok: break
        _, l = max(ok); g, e, cost, d = cand[l]
        lvl[l][e] += 1; used += cost; E[l] -= d; cand[l] = best(l)
    res = {str(l): dict(exact=[e for e in range(32) if lvl[l][e] == 2], r2=[e for e in range(32) if lvl[l][e] == 1]) for l in range(24)}
    json.dump(res, open(f"{OUT}/{out}", "w"))
    print("experts bytes %.3f GiB (budget %.3f); exact %d, r2gl %d, q1g %d" % (used / 2 ** 30, budget / 2 ** 30, sum(len(r["exact"]) for r in res.values()), sum(len(r["r2"]) for r in res.values()),
          sum(32 - len(r["exact"]) - len(r["r2"]) for r in res.values())))
    for l in range(24): print(l, "exact", len(res[str(l)]["exact"]), "r2", len(res[str(l)]["r2"]), "q1g", 32 - len(res[str(l)]["exact"]) - len(res[str(l)]["r2"]), "E_model %.3f" % E[l])


def quant_batch(Ws, Hinv, h, method):
    """one GPTQ run over the concatenated rows of several experts' matrices (they share Hinv) -> per expert list of planes"""
    W = torch.cat(Ws, 0); n = [w.shape[0] for w in Ws]
    if method == "q1g":
        q = Q.gptq_plane(W, Hinv, h); planes = [(q >= 0, group_scale_from_levels(q))]
    else:
        q, b1, b2, s1, s2 = gptq_lloyd(W, Hinv, h, planes=True); planes = [(b1, s1), (b2, s2)]
    out = []; o = 0
    for k in n: out.append([(a[o:o + k], b[o:o + k]) for a, b in planes]); o += k
    return out


def quantize_layer(st, l, al, ntok, kb=4, old=None):
    dv = Q.dev(); X, sel, w = load_calib(l); X, sel, w = X[:ntok], sel[:ntok], w[:ntok]
    p = f"model.layers.{l}.mlp.experts."; ex, r2 = set(al["exact"]), set(al["r2"])
    h1 = (X ** 2).mean(0); Hinv1 = Q.prep_hinv(X.T @ X / len(X), 0.01)
    gub = torch.from_numpy(st.get(p + "gate_up_proj_bias").astype(np.float32)).to(dv); dnb = torch.from_numpy(st.get(p + "down_proj_bias").astype(np.float32)).to(dv)
    res = {e: dict(gux=(np.array(st.get(p + "gate_up_proj_blocks", e, e + 1)[0]), np.array(st.get(p + "gate_up_proj_scales", e, e + 1)[0])),
                   dnx=(np.array(st.get(p + "down_proj_blocks", e, e + 1)[0]), np.array(st.get(p + "down_proj_scales", e, e + 1)[0]))) for e in range(32)}
    need = lambda e, n: not (old and e in old and len(old[e].get("gu", [])) == n)   # reuse a previous cache entry with the same plane count
    for e in range(32):
        if e not in ex and not need(e, 2 if e in r2 else 1): res[e]["gu"], res[e]["dn"] = old[e]["gu"], old[e]["dn"]
    for method, ids in (("r2gl", sorted(e for e in r2 if need(e, 2))), ("q1g", [e for e in range(32) if e not in ex and e not in r2 and need(e, 1)])):
        for i0 in range(0, len(ids), kb):
            grp = ids[i0:i0 + kb]; W1s = []
            for e in grp: W1s.append(Q.dq_mx4(*res[e]["gux"]))
            gu = quant_batch(W1s, Hinv1, h1, method)
            for j, e in enumerate(grp):
                res[e]["gu"] = [pack_q1(*pp) for pp in gu[j]]
                tt, kk = torch.nonzero(sel == e, as_tuple=True)
                W2 = Q.dq_mx4(*res[e]["dnx"])
                if len(tt) > 0:
                    hid = Q.swiglu(X[tt] @ W1s[j].T + gub[e]); h2 = (hid ** 2).mean(0); h2 = 0.7 * h2 + 0.3 * h2.mean()
                    H2 = hid.T @ hid / len(hid); H2 = 0.7 * H2 + 0.3 * torch.diag(torch.diag(H2).mean() * torch.ones(2880, device=dv)); Hinv2 = Q.prep_hinv(H2, 0.01)
                else: h2, Hinv2 = None, torch.eye(2880, device=dv)
                res[e]["dn"] = [pack_q1(*pp) for pp in quant_batch([W2], Hinv2, h2, method)[0]]
            del W1s
    return res


def convert(a):
    alc = json.load(open(a.alloc)); L = 24
    st0 = CG.SafeTensors(sorted(glob.glob(f"{SRC}/model-*.safetensors")))

    def hook(specs, st, L_):
        out = []
        for s in specs:
            n = s["name"]
            if n.endswith(".w_gate_up") or n.endswith(".w_down"):
                l = int(n.split(".")[1]); al = alc[str(l)]; N = len(al["exact"]); n2 = len(al["r2"]); E, R, C = s["shape"]; what = "gu" if n.endswith("gate_up") else "dn"
                out.append(dict(name=n, kind="q1", shape=[E - N, R, C], src=("c8", l, what, 0)))
                if n2: out.append(dict(name=n + "_r2", kind="q1", shape=[n2, R, C], src=("c8", l, what, 1)))
                if N: out.append(dict(name=n + "_x", kind="mxfp4", shape=[N, R, C], src=("c8", l, "gux" if what == "gu" else "dnx", 0)))
            else: out.append(s)
        return out
    class A: pass
    b = A(); b.src = SRC; b.layers = 0; b.experts = "mxfp4"; b.attn = a.attn; b.embed = a.embed; b.head = a.head; b.router = "f32"
    st, cfg, L, specs, plan, man = CG.build(b, hook)
    tok_cfg = json.load(open(f"{SRC}/tokenizer_config.json"))
    rc = CG.runtime_config(cfg, tok_cfg, b, L); rc["quant"] = dict(experts="c8-alloc", attn=a.attn, embed=a.embed, head=a.head)
    rc["moe_exact"] = {k: v["exact"] for k, v in alc.items() if v["exact"]}; rc["moe_r2"] = {k: v["r2"] for k, v in alc.items() if v["r2"]}
    extra = {"manifest.json": json.dumps(dict(format="QPACK2", tensors=man, quant="c8")).encode(), "config.json": json.dumps(rc).encode(),
             "tokenizer.json": open(f"{SRC}/tokenizer.json", "rb").read(), "generation_config.json": open(f"{SRC}/generation_config.json", "rb").read(),
             "selftest.json": b'{"cases":[]}'.ljust(32768, b" ")}
    plan = [(n, len(x)) for n, x in extra.items()] + plan
    w = StreamWriter(a.out, plan)
    for n, x in extra.items(): w.write(n, x)
    os.makedirs(a.cache, exist_ok=True); t0 = time.time(); cur = None; res = None; gpu = 0.0
    for i, s in enumerate(specs):
        src = s["src"]
        if src[0] == "c8":
            _, l, what, pi = src; al = alc[str(l)]
            if cur != l:
                tg = time.time(); f = f"{a.cache}/L{l}.pkl"
                if os.path.exists(f): res = pickle.load(open(f, "rb"))
                else:
                    oldf = f"{a.reuse}/L{l}.pkl" if a.reuse else None
                    res = quantize_layer(st, l, al, a.ntok, old=pickle.load(open(oldf, "rb")) if oldf and os.path.exists(oldf) else None); pickle.dump(res, open(f, "wb"), protocol=4)
                gpu += time.time() - tg; cur = l
                print(f"  layer {l} done ({time.time() - t0:.0f}s, GPU-phase {gpu:.0f}s) exact {len(al['exact'])} r2gl {len(al['r2'])}", file=sys.stderr, flush=True)
            bid = s["bid"]; ex, r2 = al["exact"], al["r2"]
            if what in ("gu", "dn"):
                ids = [e for e in range(32) if e not in ex] if pi == 0 else r2
                for e in ids: w.write(bid + ".q1", res[e][what][pi][0])
                for e in ids: w.write(bid + ".scale", res[e][what][pi][1])
            else:
                for e in ex: bl, sc = res[e][what]; d, _ = CG.mx4_rows(bl, sc); w.write(bid + ".mx4", d)
                for e in ex: bl, sc = res[e][what]; _, s_ = CG.mx4_rows(bl, sc); w.write(bid + ".scale", s_)
        else: CG.emit(w, st, s, b)
    size = w.close()
    print(f"{a.out}: {size / 2**20:.1f} MiB, wall {time.time() - t0:.0f}s, GPU-phase {gpu:.0f}s", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("out", nargs="?"); ap.add_argument("--alloc", default=f"{OUT}/alloc.json"); ap.add_argument("--cache", default="/workspace/qpack-models/qcache_c8")
    ap.add_argument("--attn", default="q8g"); ap.add_argument("--embed", default="f16"); ap.add_argument("--head", default="q8g"); ap.add_argument("--ntok", type=int, default=16384)
    ap.add_argument("--budget", type=float, default=6.8); ap.add_argument("--reuse", default=None); a = ap.parse_args()
    if a.cmd == "prepass": prepass()
    elif a.cmd == "alloc": alloc(a.budget)
    elif a.cmd == "alloc2": alloc_convex(a.budget)
    else: convert(a)


if __name__ == "__main__":
    main()
