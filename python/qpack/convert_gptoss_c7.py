"""C7 converter: gpt-oss-20b with 1-2 bit experts (GPTQ-style, importance aware) and a few exact MXFP4 experts per layer.

    python -m qpack.convert_gptoss_c7 SRC OUT --method q1g|r2gl --exact N|N0,N1,... [--attn q8g --embed f16 --head q8g]

Per layer (GPU, fp32, streamed): every expert is quantised with the chosen method (Hessians from the captured calibration inputs in
/workspace/qpack-models/calib), the N experts with the largest output-error energy are kept as exact MXFP4 instead.
Layout read by src/model_gptoss.cpp (config "moe_exact"): w_gate_up / w_down = q1 plane(s) for the non-exact experts (ascending ids),
w_gate_up_r2 / w_down_r2 = second q1 plane (r2gl), w_gate_up_x / w_down_x = mxfp4 for the exact experts (listed order)."""
import argparse, glob, json, os, sys, time
import numpy as np
import torch

from . import convert_gptoss as CG
from . import c7_quant as Q
from .c7_lloyd import gptq_lloyd
from .c7_proxy import load_calib
from .container import StreamWriter

G = 128


def pack_q1(sign_pos, scales):
    """sign_pos bool [R, C] (True => +1), scales fp16-able [R, ng] -> (bits bytes, scale bytes)"""
    R, C = sign_pos.shape; ng = (C + G - 1) // G
    pad = np.zeros((R, ng * G), np.uint8); pad[:, :C] = sign_pos.cpu().numpy()
    return np.packbits(pad, axis=1, bitorder="little").tobytes(), scales.cpu().numpy().astype(np.float16).tobytes()


def group_scale_from_levels(Qm):
    R, C = Qm.shape; P, _ = Q.group_view(Qm.abs()); return P.amax(2)


def quant_matrix(W, Hinv, h, method):
    """-> list of planes [(sign_pos, scales)] and the dequantised matrix"""
    if method == "q1g":
        q = Q.gptq_plane(W, Hinv, h) if Hinv is not None else Q.plane(W, h)
        return [(q >= 0, group_scale_from_levels(q))], q
    if method == "r2gl":
        if Hinv is None:
            from .c7_lloyd import r2_lloyd
            raise RuntimeError("expert without calibration tokens")
        q, b1, b2, s1, s2 = gptq_lloyd(W, Hinv, h, planes=True)
        return [(b1, s1), (b2, s2)], q
    raise ValueError(method)


def quantize_layer(st, l, method, nexact, nfit):
    dv = Q.dev(); X, sel, w = load_calib(l); X = X[:nfit]; sel = sel[:nfit]; w = w[:nfit]
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
                      gux=(st.get(p + "gate_up_proj_blocks", e, e + 1)[0], st.get(p + "gate_up_proj_scales", e, e + 1)[0]),
                      dnx=(st.get(p + "down_proj_blocks", e, e + 1)[0], st.get(p + "down_proj_scales", e, e + 1)[0]))
    exact = sorted(torch.argsort(-energy)[:nexact].tolist()) if nexact else []
    exact_energy = float(energy[exact].sum() / energy.sum()) if nexact else 0.0
    return res, exact, exact_energy


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--method", default="q1g", choices=["q1g", "r2gl"]); ap.add_argument("--exact", default="0")
    ap.add_argument("--attn", default="q8g"); ap.add_argument("--embed", default="f16"); ap.add_argument("--head", default="q8g"); ap.add_argument("--router", default="f32")
    ap.add_argument("--layers", type=int, default=0); ap.add_argument("--nfit", type=int, default=2048)
    a = ap.parse_args(); a.experts = "mxfp4"
    ex = [int(x) for x in a.exact.split(",")]; nplane = 2 if a.method == "r2gl" else 1
    nex = lambda l: ex[l] if len(ex) > 1 else ex[0]

    def hook(specs, st, L):
        out = []
        for s in specs:
            n = s["name"]
            if n.endswith(".w_gate_up") or n.endswith(".w_down"):
                l = int(n.split(".")[1]); N = nex(l); E, R, C = s["shape"]; base = n
                out.append(dict(name=base, kind="q1", shape=[E - N, R, C], src=("c7", l, "gu" if n.endswith("gate_up") else "dn", 0)))
                if nplane > 1: out.append(dict(name=base + "_r2", kind="q1", shape=[E - N, R, C], src=("c7", l, "gu" if n.endswith("gate_up") else "dn", 1)))
                if N: out.append(dict(name=base + "_x", kind="mxfp4", shape=[N, R, C], src=("c7", l, "gux" if n.endswith("gate_up") else "dnx", 0)))
            else: out.append(s)
        return out
    st, cfg, L, specs, plan, man = CG.build(a, hook)
    tok_cfg = json.load(open(os.path.join(a.src, "tokenizer_config.json")))
    rc = CG.runtime_config(cfg, tok_cfg, a, L); rc["quant"] = dict(experts=a.method, exact=ex, attn=a.attn, embed=a.embed, head=a.head)
    rc["moe_exact"] = {}
    cfgblob = json.dumps(rc).encode().ljust(len(json.dumps(rc)) + 8192, b" ")   # patched with the chosen expert ids at the end
    extra = {"manifest.json": json.dumps(dict(format="QPACK2", tensors=man, quant=a.method)).encode(), "config.json": cfgblob,
             "tokenizer.json": open(os.path.join(a.src, "tokenizer.json"), "rb").read(),
             "generation_config.json": open(os.path.join(a.src, "generation_config.json"), "rb").read(), "selftest.json": b'{"cases":[]}'.ljust(32768, b" ")}
    plan = [(n, len(b)) for n, b in extra.items()] + plan
    w = StreamWriter(a.out, plan)
    for n, b in extra.items(): w.write(n, b)
    t0 = time.time(); cache = {}; chosen = {}; stats = []
    for i, s in enumerate(specs):
        src = s["src"]
        if src[0] == "c7":
            _, l, what, plane_i = src
            if l not in cache:
                cache.clear(); res, exact, ee = quantize_layer(st, l, a.method, nex(l), a.nfit); cache[l] = (res, exact); chosen[str(l)] = exact; stats.append((l, ee))
                print(f"  layer {l} quantised ({time.time() - t0:.0f}s), exact {exact}, error energy of exact experts {ee:.2f}", file=sys.stderr, flush=True)
            res, exact = cache[l]; bid = s["bid"]
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
        if i % 40 == 0: print(f"  {i}/{len(specs)} {s['name']} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    size = w.close()
    rc["moe_exact"] = {k: v for k, v in chosen.items() if v}
    blob = json.dumps(rc).encode()
    from .container import Container
    c = Container(a.out); off, ln = c.toc["config.json"]; c.close()
    assert len(blob) <= ln, "config does not fit"
    with open(a.out, "r+b") as f: f.seek(off); f.write(blob.ljust(ln, b" "))
    print(f"{a.out}: {size / 2**20:.1f} MiB ({a.method}, exact {ex}), {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
