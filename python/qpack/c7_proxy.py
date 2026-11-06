"""C7 proxy: MoE-layer output error of 1-2 bit expert quantisers on captured calibration inputs.
    python -m qpack.c7_proxy LAYER[,LAYER...] [--methods q1,q1i,...] [--out proxy.json]
Fit on tokens < 1536, evaluate on tokens >= 1536 of /workspace/qpack-models/calib (QPACK_DUMP_EXPERT_IN data)."""
import argparse, glob, json, os, sys, time
import numpy as np
import torch

from .convert_gptoss import SafeTensors
from . import c7_quant as Q
from .c7_lloyd import r2_lloyd, gptq_lloyd

SRC = "/workspace/qpack-models/gpt-oss-20b"; CAL = os.environ.get("QPACK_CALIB", "/workspace/qpack-models/calib"); NFIT = 1536


def load_calib(l, d=2880):
    X = np.fromfile(f"{CAL}/L{l}.in.f16", dtype=np.float16).reshape(-1, d)
    sel = np.fromfile(f"{CAL}/L{l}.sel.u8", dtype=np.uint8).reshape(-1, 4)
    w = np.fromfile(f"{CAL}/L{l}.w.f16", dtype=np.float16).reshape(-1, 4)
    dv = Q.dev()
    return torch.from_numpy(X.astype(np.float32)).to(dv), torch.from_numpy(sel.astype(np.int64)).to(dv), torch.from_numpy(w.astype(np.float32)).to(dv)


BPW = {"q1": 1.125, "q1i": 1.125, "q1g": 1.125, "r2": 2.25, "r2ls": 2.25, "r2g": 2.25, "r2it": 2.25, "r2gl": 2.25}


def run_layer(st, l, methods, down_lambda=float(os.environ.get("QPACK_DOWN_LAMBDA", "0.3")), log=True):
    dv = Q.dev(); X, sel, w = load_calib(l); T = X.shape[0]
    p = f"model.layers.{l}.mlp.experts."
    Xf = X[:NFIT]; h1 = (Xf ** 2).mean(0); H1 = Xf.T @ Xf / NFIT
    Hinv1 = Q.prep_hinv(H1, 0.01)
    gub = torch.from_numpy(st.get(p + "gate_up_proj_bias").astype(np.float32)).to(dv)
    dnb = torch.from_numpy(st.get(p + "down_proj_bias").astype(np.float32)).to(dv)
    D = {m: torch.zeros(T, 4, 2880, device=dv, dtype=torch.float16) for m in methods}
    REF = torch.zeros(T, 4, 2880, device=dv, dtype=torch.float16)
    t0 = time.time()
    for e in range(32):
        tt, kk = torch.nonzero(sel == e, as_tuple=True)
        if len(tt) == 0: continue
        W1 = Q.dq_mx4(st.get(p + "gate_up_proj_blocks", e, e + 1)[0], st.get(p + "gate_up_proj_scales", e, e + 1)[0])
        W2 = Q.dq_mx4(st.get(p + "down_proj_blocks", e, e + 1)[0], st.get(p + "down_proj_scales", e, e + 1)[0])
        Xe = X[tt]; fitm = tt < NFIT
        hid = Q.swiglu(Xe @ W1.T + gub[e]); ref = hid @ W2.T + dnb[e]
        REF[tt, kk] = (ref * w[tt, kk][:, None]).half()
        hf = hid[fitm]
        if len(hf) > 0:
            h2 = (hf ** 2).mean(0); h2 = (1 - down_lambda) * h2 + down_lambda * h2.mean()
            H2 = hf.T @ hf / len(hf); H2 = (1 - down_lambda) * H2 + down_lambda * torch.diag(torch.diag(H2).mean() * torch.ones(2880, device=dv))
            Hinv2 = Q.prep_hinv(H2, 0.01)
        else:
            h2 = None; Hinv2 = None
        for m in methods:
            if m == "q1": a, b = Q.plane(W1), Q.plane(W2)
            elif m == "q1i": a, b = Q.plane(W1, h1), Q.plane(W2, h2)
            elif m == "q1g": a, b = Q.gptq_plane(W1, Hinv1, h1), (Q.gptq_plane(W2, Hinv2, h2) if Hinv2 is not None else Q.plane(W2))
            elif m == "r2": a, b = (lambda W: Q.plane(W) + Q.plane(W - Q.plane(W)))(W1), (lambda W: Q.plane(W) + Q.plane(W - Q.plane(W)))(W2)
            elif m == "r2ls": a, b = Q.r2_ls(W1, h1), Q.r2_ls(W2, h2)
            elif m == "r2it": a, b = r2_lloyd(W1, h1), r2_lloyd(W2, h2)
            elif m == "r2gl": a, b = gptq_lloyd(W1, Hinv1, h1), (gptq_lloyd(W2, Hinv2, h2) if Hinv2 is not None else r2_lloyd(W2))
            elif m == "r2g": a, b = Q.gptq_planes(W1, Hinv1, 2, h1), (Q.gptq_planes(W2, Hinv2, 2, h2) if Hinv2 is not None else Q.r2_ls(W2))
            hq = Q.swiglu(Xe @ a.T + gub[e]); oq = hq @ b.T + dnb[e]
            D[m][tt, kk] = ((oq - ref) * w[tt, kk][:, None]).half()
        if log and e % 8 == 7: print(f"  layer {l} expert {e} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    ref_moe = REF.float().sum(1)
    res = {}
    for m in methods:
        d = D[m].float()
        # per-expert fit error energy -> ranking for exact experts
        en = torch.zeros(32, device=dv)
        for e in range(32):
            msk = (sel[:NFIT] == e)
            en[e] = (d[:NFIT][msk] ** 2).sum()
        con = torch.zeros(32, device=dv)
        for e in range(32): con[e] = (REF.float()[:NFIT][(sel[:NFIT] == e)] ** 2).sum()
        res[m] = dict(energy=en.tolist(), contrib=con.tolist())
        for key, score in (('by_N', en), ('by_N_c', con)):
            order = torch.argsort(-score).tolist(); out = {}
            for N in (0, 4, 8, 12, 16):
                ex = torch.zeros(32, dtype=torch.bool, device=dv); ex[order[:N]] = True
                keep = (~ex[sel]).float()[..., None]       # [T,4,1] non-exact assignments
                err = (d * keep).sum(1)
                rel = lambda a, b: float(err[a:b].norm() / ref_moe[a:b].norm())
                out[N] = dict(held=rel(NFIT, T), fit=rel(0, NFIT), bpw=((32 - N) * BPW[m] + N * 4.25) / 32)
            res[m][key] = out
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("layers"); ap.add_argument("--methods", default="q1,q1i,q1g,r2,r2ls,r2g"); ap.add_argument("--out", default="/tmp/claude-0/proxy.json")
    a = ap.parse_args(); st = SafeTensors(sorted(glob.glob(f"{SRC}/model-*.safetensors"))); ms = a.methods.split(",")
    allres = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for l in map(int, a.layers.split(",")):
        r = run_layer(st, l, ms); allres.setdefault(str(l), {}).update(r); json.dump(allres, open(a.out, "w"))
        print(f"layer {l}: " + "  ".join(f"{m}: held {r[m]['by_N'][0]['held']:.3f} (N=8: {r[m]['by_N'][8]['held']:.3f})" for m in ms), flush=True)


if __name__ == "__main__":
    main()
