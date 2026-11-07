"""C7 converter: gpt-oss-20b with 1-2 bit experts (GPTQ-style, importance aware) and a few exact MXFP4 experts per layer.

    python -m qpack.convert_gptoss_c7 SRC OUT --method q1g|r2gl|"r2gl:0-9,q1g:10-23" --exact N|N0,N1,... [--cache DIR] [--from-cache DIR [--heal]]

Per layer (GPU, fp32, streamed): every expert is quantised with the layer's method (Hessians from the captured calibration inputs in
$QPACK_CALIB, default /workspace/qpack-models/calib), the N experts with the largest output-error energy are kept as exact MXFP4.
--cache DIR stores each layer's quantised result (pickle); --from-cache DIR reuses it (no GPTQ), --heal then tunes group scales + expert
biases (c7_heal.py) before writing.
Layout read by src/model_gptoss.cpp (config "moe_exact"): w_gate_up / w_down = q1 plane(s) for the non-exact experts (ascending ids),
w_gate_up_r2 / w_down_r2 = second q1 plane (r2gl layers), w_gate_up_x / w_down_x = mxfp4 for the exact experts (listed order)."""
import argparse, glob, json, os, pickle, sys, time
import numpy as np
import torch

from . import convert_gptoss as CG
from . import c7_quant as Q
from . import c7_heal as H
from .c7_lloyd import gptq_lloyd
from .c7_proxy import load_calib
from .container import StreamWriter, Container

G = 128


def pack_q1(sign_pos, scales):
    R, C = sign_pos.shape; ng = (C + G - 1) // G
    pad = np.zeros((R, ng * G), np.uint8); pad[:, :C] = sign_pos.cpu().numpy()
    return np.packbits(pad, axis=1, bitorder="little").tobytes(), scales.cpu().numpy().astype(np.float16).tobytes()


def group_scale_from_levels(Qm):
    P, _ = Q.group_view(Qm.abs()); return P.amax(2)


def quant_matrix(W, Hinv, h, method):
    if method == "q1g":
        q = Q.gptq_plane(W, Hinv, h)
        return [(q >= 0, group_scale_from_levels(q))], q
    q, b1, b2, s1, s2 = gptq_lloyd(W, Hinv, h, planes=True)
    return [(b1, s1), (b2, s2)], q


def load_x(l, ntok):
    X, sel, w = load_calib(l); return X[:ntok], sel[:ntok], w[:ntok]


def quantize_layer(st, l, method, nexact, ntok):
    dv = Q.dev(); X, sel, w = load_x(l, ntok)
    p = f"model.layers.{l}.mlp.experts."
    h1 = (X ** 2).mean(0); Hinv1 = Q.prep_hinv(X.T @ X / len(X), 0.01)
    gub = torch.from_numpy(st.get(p + "gate_up_proj_bias").astype(np.float32)).to(dv); dnb = torch.from_numpy(st.get(p + "down_proj_bias").astype(np.float32)).to(dv)
    res = {}; energy = torch.zeros(32, device=dv)
    for e in range(32):
        W1 = Q.dq_mx4(st.get(p + "gate_up_proj_blocks", e, e + 1)[0], st.get(p + "gate_up_proj_scales", e, e + 1)[0])
        W2 = Q.dq_mx4(st.get(p + "down_proj_blocks", e, e + 1)[0], st.get(p + "down_proj_scales", e, e + 1)[0])
        tt, kk = torch.nonzero(sel == e, as_tuple=True)
        Xe = X[tt]; hid = Q.swiglu(Xe @ W1.T + gub[e]); ref = hid @ W2.T + dnb[e]
        if len(tt) > 0:
            h2 = (hid ** 2).mean(0); h2 = 0.7 * h2 + 0.3 * h2.mean()
            H2 = hid.T @ hid / len(hid); H2 = 0.7 * H2 + 0.3 * torch.diag(torch.diag(H2).mean() * torch.ones(2880, device=dv)); Hinv2 = Q.prep_hinv(H2, 0.01)
        else: h2, Hinv2 = None, torch.eye(2880, device=dv)
        pl1, q1 = quant_matrix(W1, Hinv1, h1, method); pl2, q2 = quant_matrix(W2, Hinv2, h2, method)
        if len(tt) > 0:
            oq = Q.swiglu(Xe @ q1.T + gub[e]) @ q2.T + dnb[e]
            energy[e] = ((oq - ref) ** 2).sum(1).mul(w[tt, kk] ** 2).sum()
        res[e] = dict(gu=[pack_q1(*pp) for pp in pl1], dn=[pack_q1(*pp) for pp in pl2],
                      gux=(np.array(st.get(p + "gate_up_proj_blocks", e, e + 1)[0]), np.array(st.get(p + "gate_up_proj_scales", e, e + 1)[0])),
                      dnx=(np.array(st.get(p + "down_proj_blocks", e, e + 1)[0]), np.array(st.get(p + "down_proj_scales", e, e + 1)[0])))
    exact = sorted(torch.argsort(-energy)[:nexact].tolist()) if nexact else []
    return res, exact, (float(energy[exact].sum() / energy.sum()) if nexact else 0.0)


_HB = int(os.environ.get('QPACK_HELD_BLOCK', '64')); _HM = int(os.environ.get('QPACK_HELD_MOD', '8'))
HELD = lambda T: ((torch.arange(T) // _HB) % _HM == _HM - 1)   # held-out blocks of _HB positions (default 64 of 8; v2 uses 512 of 6 = whole documents)


def heal_layer(st, l, res, exact, ntok, steps, lr_s, lr_b, cap=200):
    """tune scales + biases of all non-exact experts of layer l; returns new res, new biases (numpy [32,5760], [32,2880]) and metrics"""
    dv = Q.dev(); X, sel, w = load_x(l, ntok); T = X.shape[0]; held = HELD(T).to(dv)
    p = f"model.layers.{l}.mlp.experts."
    gub = torch.from_numpy(st.get(p + "gate_up_proj_bias").astype(np.float32)).to(dv); dnb = torch.from_numpy(st.get(p + "down_proj_bias").astype(np.float32)).to(dv)
    REF = torch.zeros(T, 4, 2880, device=dv, dtype=torch.float16); DB = torch.zeros_like(REF); DA = torch.zeros_like(REF)
    newg, newd = gub.clone(), dnb.clone(); out = {}; nrev = 0
    for e in range(32):
        tt, kk = torch.nonzero(sel == e, as_tuple=True)
        W1 = Q.dq_mx4(st.get(p + "gate_up_proj_blocks", e, e + 1)[0], st.get(p + "gate_up_proj_scales", e, e + 1)[0])
        W2 = Q.dq_mx4(st.get(p + "down_proj_blocks", e, e + 1)[0], st.get(p + "down_proj_scales", e, e + 1)[0])
        Xe = X[tt]; ref = Q.swiglu(Xe @ W1.T + gub[e]) @ W2.T + dnb[e]; we = w[tt, kk]
        if len(tt): REF[tt, kk] = (ref * we[:, None]).half()
        r = res[e]
        if e in exact: out[e] = r; continue
        pg = [H.unpack_plane(b, s, 5760, 2880) for b, s in r["gu"]]; pd = [H.unpack_plane(b, s, 2880, 2880) for b, s in r["dn"]]
        hm = held[tt]; fm = ~hm
        fi = torch.nonzero(fm)[:, 0]
        if len(fi) > cap: fm = torch.zeros_like(fm); fm[fi[torch.randperm(len(fi), generator=torch.Generator().manual_seed(l * 100 + e))[:cap].to(dv)]] = True
        if len(tt) == 0: out[e] = r; continue
        # error before (for the layer-level metric)
        def outq(Sg, Sd, bg, bd):
            return Q.swiglu(Xe @ H.build([B for B, _ in pg], Sg).T + bg) @ H.build([B for B, _ in pd], Sd).T + bd
        with torch.no_grad(): DB[tt, kk] = ((outq([S for _, S in pg], [S for _, S in pd], gub[e], dnb[e]) - ref) * we[:, None]).half()
        Sg, Sd, bg, bd, eb, ea, ok = H.heal_expert(pg, pd, gub[e], dnb[e], Xe[fm], we[fm], ref[fm], Xe[hm], we[hm], ref[hm], steps, lr_s, lr_b)
        if not ok:
            nrev += 1; DA[tt, kk] = DB[tt, kk]; out[e] = r; continue
        with torch.no_grad(): DA[tt, kk] = ((outq(Sg, Sd, bg, bd) - ref) * we[:, None]).half()
        newg[e], newd[e] = bg, bd
        out[e] = dict(r, gu=[(b, H.pack_scales(S)) for (b, _), S in zip(r["gu"], Sg)], dn=[(b, H.pack_scales(S)) for (b, _), S in zip(r["dn"], Sd)])
    ref_moe = REF.float().sum(1); hm_t = held; fm_t = ~held
    rel = lambda D, m: float(D.float().sum(1)[m].norm() / ref_moe[m].norm())
    met = dict(held_before=rel(DB, hm_t), held_after=rel(DA, hm_t), fit_before=rel(DB, fm_t), fit_after=rel(DA, fm_t), reverted=nrev)
    return out, newg.cpu().numpy(), newd.cpu().numpy(), met


def parse_methods(s, L):
    if ":" not in s: return [s] * L
    m = [None] * L
    for part in s.split(","):
        name, rng = part.split(":"); a, b = (rng.split("-") + [rng])[:2]
        for l in range(int(a), int(b) + 1): m[l] = name
    return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--method", default="q1g"); ap.add_argument("--exact", default="0")
    ap.add_argument("--attn", default="q8g"); ap.add_argument("--embed", default="f16"); ap.add_argument("--head", default="q8g"); ap.add_argument("--router", default="f32")
    ap.add_argument("--layers", type=int, default=0); ap.add_argument("--ntok", type=int, default=8192)
    ap.add_argument("--cache", default=None); ap.add_argument("--from-cache", default=None); ap.add_argument("--heal", action="store_true")
    ap.add_argument("--heal-steps", type=int, default=40); ap.add_argument("--heal-layers", type=int, default=99); ap.add_argument("--heal-tokens", type=int, default=200); ap.add_argument("--lr-s", type=float, default=2e-3); ap.add_argument("--lr-b", type=float, default=5e-3)
    a = ap.parse_args(); a.experts = "mxfp4"
    ex = [int(x) for x in a.exact.split(",")]; nex = lambda l: ex[l] if len(ex) > 1 else ex[0]
    st0 = CG.SafeTensors(sorted(glob.glob(os.path.join(a.src, "model-*.safetensors"))))
    Lm = parse_methods(a.method, json.load(open(os.path.join(a.src, "config.json")))["num_hidden_layers"])
    if a.from_cache:   # exact sets come from the cache
        exact_of = {l: pickle.load(open(f"{a.from_cache}/L{l}.pkl", "rb"))["exact"] for l in range(a.layers or 24)}
        nex = lambda l: len(exact_of[l])

    def hook(specs, st, L):
        out = []
        for s in specs:
            n = s["name"]
            if n.endswith(".w_gate_up") or n.endswith(".w_down"):
                l = int(n.split(".")[1]); N = nex(l); E, R, C = s["shape"]; what = "gu" if n.endswith("gate_up") else "dn"
                out.append(dict(name=n, kind="q1", shape=[E - N, R, C], src=("c7", l, what, 0)))
                if Lm[l] == "r2gl": out.append(dict(name=n + "_r2", kind="q1", shape=[E - N, R, C], src=("c7", l, what, 1)))
                if N: out.append(dict(name=n + "_x", kind="mxfp4", shape=[N, R, C], src=("c7", l, "gux" if what == "gu" else "dnx", 0)))
            elif n.endswith(".b_gate_up") or n.endswith(".b_down"):
                s = dict(s); s["src"] = ("c7bias", int(n.split(".")[1]), "g" if n.endswith("gate_up") else "d"); out.append(s)
            else: out.append(s)
        return out
    st, cfg, L, specs, plan, man = CG.build(a, hook)
    tok_cfg = json.load(open(os.path.join(a.src, "tokenizer_config.json")))
    rc = CG.runtime_config(cfg, tok_cfg, a, L); rc["quant"] = dict(experts=a.method, exact=ex, attn=a.attn, embed=a.embed, head=a.head, heal=a.heal)
    rc["moe_exact"] = {}
    cfgblob = json.dumps(rc).encode().ljust(len(json.dumps(rc)) + 8192, b" ")
    extra = {"manifest.json": json.dumps(dict(format="QPACK2", tensors=man, quant=a.method)).encode(), "config.json": cfgblob,
             "tokenizer.json": open(os.path.join(a.src, "tokenizer.json"), "rb").read(),
             "generation_config.json": open(os.path.join(a.src, "generation_config.json"), "rb").read(), "selftest.json": b'{"cases":[]}'.ljust(32768, b" ")}
    plan = [(n, len(b)) for n, b in extra.items()] + plan
    w = StreamWriter(a.out, plan)
    for n, b in extra.items(): w.write(n, b)
    t0 = time.time(); cache = {}; chosen = {}; gpu_s = 0.0; metrics = {}
    if a.cache: os.makedirs(a.cache, exist_ok=True)
    for i, s in enumerate(specs):
        src = s["src"]
        if src[0] in ("c7", "c7bias"):
            l = src[1]
            if l not in cache:
                cache.clear(); tg = time.time(); newb = None
                if a.from_cache:
                    d = pickle.load(open(f"{a.from_cache}/L{l}.pkl", "rb")); res, exact = d["res"], d["exact"]
                    if a.heal and l < a.heal_layers:
                        res, ng, nd, met = heal_layer(st, l, res, exact, a.ntok, a.heal_steps, a.lr_s, a.lr_b, a.heal_tokens); newb = (ng, nd); metrics[l] = met
                        print(f"  layer {l} healed ({time.time() - t0:.0f}s): held {met['held_before']:.3f} -> {met['held_after']:.3f}, fit {met['fit_before']:.3f} -> {met['fit_after']:.3f}, reverted {met['reverted']}", file=sys.stderr, flush=True)
                elif a.cache and os.path.exists(f"{a.cache}/L{l}.pkl"):   # resume
                    d = pickle.load(open(f"{a.cache}/L{l}.pkl", "rb")); res, exact = d["res"], d["exact"]; print(f"  layer {l} from cache", file=sys.stderr, flush=True)
                else:
                    res, exact, ee = quantize_layer(st, l, Lm[l], nex(l), a.ntok)
                    if a.cache: pickle.dump(dict(res=res, exact=exact), open(f"{a.cache}/L{l}.pkl", "wb"), protocol=4)
                    print(f"  layer {l} quantised ({time.time() - t0:.0f}s, {Lm[l]}), exact {exact}, error energy of exact experts {ee:.2f}", file=sys.stderr, flush=True)
                gpu_s += time.time() - tg; cache[l] = (res, exact, newb); chosen[str(l)] = exact
            res, exact, newb = cache[l]; bid = s["bid"]
            if src[0] == "c7bias":
                if newb is None: CG.emit(w, st, dict(s, src=("dense", f"model.layers.{l}.mlp.experts.{'gate_up' if src[2] == 'g' else 'down'}_proj_bias")), a)
                else:
                    arr = np.ascontiguousarray(newb[0 if src[2] == "g" else 1], np.float32); w.write(bid + ".f32", arr)
                continue
            what, plane_i = src[2], src[3]
            if what in ("gu", "dn"):
                ids = [e for e in range(32) if e not in exact]
                for e in ids: w.write(bid + ".q1", res[e][what][plane_i][0])
                for e in ids: w.write(bid + ".scale", res[e][what][plane_i][1])
            else:
                for e in exact:
                    b, sc = res[e][what]; d, s_ = CG.mx4_rows(b, sc); w.write(bid + ".mx4", d)
                for e in exact:
                    b, sc = res[e][what]; d, s_ = CG.mx4_rows(b, sc); w.write(bid + ".scale", s_)
        else: CG.emit(w, st, s, a)
        if i % 80 == 0: print(f"  {i}/{len(specs)} {s['name']} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    size = w.close()
    rc["moe_exact"] = {k: v for k, v in chosen.items() if v}
    blob = json.dumps(rc).encode()
    c = Container(a.out); off, ln = c.toc["config.json"]; c.close()
    assert len(blob) <= ln, "config does not fit"
    with open(a.out, "r+b") as f: f.seek(off); f.write(blob.ljust(ln, b" "))
    if metrics: json.dump(metrics, open(a.out + ".heal.json", "w"))
    print(f"{a.out}: {size / 2**20:.1f} MiB ({a.method}, exact {ex}), wall {time.time() - t0:.0f}s, GPU-phase {gpu_s:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
