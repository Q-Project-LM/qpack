"""Healing step 1: block-wise tuning of the group scales (all q1 planes) and the expert biases of one MoE layer, sign bits frozen.

Objective per expert (independent): sum_t w_t^2 || o_q(x_t) - o_ref(x_t) ||^2 over the calibration tokens routed to the expert (fit split),
o = down(swiglu(gate_up(x) + b_gu)) + b_dn; o_ref = exact MXFP4 expert. Adam on relative scale deltas (S = S0 (1 + d)) and absolute bias deltas;
an expert is reverted if its held-out error does not improve. Everything is fp32 on the GPU; scales are rounded to fp16 afterwards."""
import math
import numpy as np
import torch

from . import c7_quant as Q

G = 128


def unpack_plane(bits_bytes, scale_bytes, R, C):
    ng = (C + G - 1) // G
    b = np.unpackbits(np.frombuffer(bits_bytes, np.uint8).reshape(R, ng * 16), axis=1, bitorder="little")[:, :C]
    s = np.frombuffer(scale_bytes, np.float16).reshape(R, ng).astype(np.float32)
    dv = Q.dev()
    return (torch.from_numpy(b).to(dv).to(torch.float16) * 2 - 1), torch.from_numpy(s).to(dv)


def build(Bs, Ss):
    """sum_p expand(S_p) * B_p -> [R, C] fp32"""
    W = None
    for B, S in zip(Bs, Ss):
        t = S.repeat_interleave(G, dim=1)[:, :B.shape[1]] * B
        W = t if W is None else W + t
    return W


def pack_scales(S): return S.detach().half().cpu().numpy().tobytes()


def heal_expert(planes_gu, planes_dn, bg0, bd0, Xf, wf, ref_f, Xh, wh, ref_h, steps, lr_s, lr_b):
    """planes_*: list of (B, S0) torch tensors. Returns (S_gu list, S_dn list, bg, bd, err_before_held, err_after_held, used)"""
    dv = Q.dev()
    d_gu = [torch.zeros_like(S, requires_grad=True) for _, S in planes_gu]
    d_dn = [torch.zeros_like(S, requires_grad=True) for _, S in planes_dn]
    dbg = torch.zeros_like(bg0, requires_grad=True); dbd = torch.zeros_like(bd0, requires_grad=True)
    rms_o = ref_f.pow(2).mean().sqrt().clamp_min(1e-6)

    def fwd(X, d_gu, d_dn, dbg, dbd):
        W1 = build([B for B, _ in planes_gu], [S * (1 + d) for (_, S), d in zip(planes_gu, d_gu)])
        W2 = build([B for B, _ in planes_dn], [S * (1 + d) for (_, S), d in zip(planes_dn, d_dn)])
        return Q.swiglu(X @ W1.T + bg0 + dbg) @ W2.T + bd0 + dbd

    def err(X, w, ref, *p):
        with torch.no_grad(): return float((w[:, None] ** 2 * (fwd(X, *p) - ref) ** 2).sum())

    zero = ([torch.zeros_like(S) for _, S in planes_gu], [torch.zeros_like(S) for _, S in planes_dn], torch.zeros_like(bg0), torch.zeros_like(bd0))
    e_before = err(Xh, wh, ref_h, *zero) if len(Xh) else 0.0
    opt = torch.optim.Adam([{"params": d_gu + d_dn, "lr": lr_s}, {"params": [dbg, dbd], "lr": lr_b * float(rms_o)}])
    den = float((wf[:, None] ** 2 * ref_f ** 2).sum()) + 1e-12
    for it in range(steps):
        for g, base in zip(opt.param_groups, (lr_s, lr_b * float(rms_o))): g["lr"] = base * 0.5 * (1 + math.cos(math.pi * it / steps))
        opt.zero_grad(set_to_none=True)
        loss = (wf[:, None] ** 2 * (fwd(Xf, d_gu, d_dn, dbg, dbd) - ref_f) ** 2).sum() / den
        loss.backward(); opt.step()
    e_after = err(Xh, wh, ref_h, d_gu, d_dn, dbg, dbd) if len(Xh) else 0.0
    ok = len(Xh) == 0 or e_after < e_before
    if not ok: return None, None, None, None, e_before, e_before, False
    S_gu = [(S * (1 + d)).detach() for (_, S), d in zip(planes_gu, d_gu)]; S_dn = [(S * (1 + d)).detach() for (_, S), d in zip(planes_dn, d_dn)]
    return S_gu, S_dn, (bg0 + dbg).detach(), (bd0 + dbd).detach(), e_before, e_after, True
