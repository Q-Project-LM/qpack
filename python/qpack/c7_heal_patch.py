"""Healing from a finished model + the quantisation cache (no safetensors needed): tunes the group scales of the q1 planes and patches
them in place into a copy of the model file (scale blobs keep their size). The exact reference experts come from the cache
(res[e]['gux'/'dnx'] = original MXFP4 blocks), the original expert biases from the model file itself (never changed here: --lr-b 0).
    python -m qpack.c7_heal_patch BASE.qpack OUT.qpack CACHE_DIR [--layers 16 --steps 20 --lr-s 5e-4 --tokens 200]"""
import argparse, os, pickle, shutil, sys, time
import numpy as np

from . import convert_gptoss_c7 as C7
from .container import Container


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("base"); ap.add_argument("out"); ap.add_argument("cache")
    ap.add_argument("--layers", type=int, default=16); ap.add_argument("--steps", type=int, default=20); ap.add_argument("--lr-s", type=float, default=5e-4)
    ap.add_argument("--lr-b", type=float, default=0.0); ap.add_argument("--tokens", type=int, default=200); ap.add_argument("--ntok", type=int, default=8192)
    a = ap.parse_args()
    shutil.copyfile(a.base, a.out)
    c = Container(a.base); man = {e["name"]: e for e in c.json("manifest.json")["tensors"]}
    t0 = time.time()
    for l in range(a.layers):
        d = pickle.load(open(f"{a.cache}/L{l}.pkl", "rb")); res, exact = d["res"], d["exact"]; ids = [e for e in range(32) if e not in exact]
        # biases of the base model
        bg = np.frombuffer(c.blob(man[f"layers.{l}.b_gate_up"]["blob"]), np.float32).reshape(32, 5760); bd = np.frombuffer(c.blob(man[f"layers.{l}.b_down"]["blob"]), np.float32).reshape(32, 2880)

        class S:
            def get(self, key, e0=None, e1=None):
                name = key.split(".")[-1]
                if name == "gate_up_proj_bias": return bg
                if name == "down_proj_bias": return bd
                which = "gux" if name.startswith("gate") else "dnx"; k = 0 if name.endswith("blocks") else 1
                return np.stack([res[e][which][k] for e in range(e0, e1)])
        new, _, _, met = C7.heal_layer(S(), l, res, exact, a.ntok, a.steps, a.lr_s, a.lr_b, a.tokens)
        with open(a.out, "r+b") as f:
            for what, nm in (("gu", "w_gate_up"), ("dn", "w_down")):
                for p in range(len(res[ids[0]][what])):
                    ent = man[f"layers.{l}.{nm}" + ("_r2" if p else "")]; off, ln = c.toc[ent["scale"]]; pos = off
                    for e in ids:
                        b = new[e][what][p][1]; f.seek(pos); f.write(b); pos += len(b)
                    assert pos - off == ln, (pos - off, ln)
        print(f"  layer {l} healed ({time.time() - t0:.0f}s): held {met['held_before']:.3f} -> {met['held_after']:.3f}, fit {met['fit_before']:.3f} -> {met['fit_after']:.3f}, reverted {met['reverted']}", file=sys.stderr, flush=True)
    print("done", time.time() - t0, file=sys.stderr)


if __name__ == "__main__":
    main()
