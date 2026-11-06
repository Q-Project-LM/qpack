"""Two-plane (2.25 bpw) quantiser as a per-group 4-level quantiser {+-s1 +-s2}, fitted by Lloyd-style alternation."""
import torch
from .c7_quant import group_view, G


def gptq_lloyd(W, Hinv, h=None, planes=False):
    """GPTQ error feedback with the 4-level per-group quantiser: group levels fitted (Lloyd) on the error-compensated weights at the start
    of each 128-block, every column then rounded to the nearest of {+-s1 +-s2}"""
    R, C = W.shape; W = W.clone(); Q = torch.zeros_like(W)
    B1 = torch.zeros(R, C, dtype=torch.bool, device=W.device); B2 = B1.clone(); S1 = []; S2 = []
    for i1 in range(0, C, G):
        i2 = min(C, i1 + G); W1 = W[:, i1:i2].clone(); n = i2 - i1
        s1, s2 = r2_lloyd(W1, None if h is None else h[i1:i2], 4, scales=True)
        a = s1[:, None]; b = s2[:, None]
        lv = torch.cat([a + b, a - b, -a + b, -a - b], 1)                  # [R, 4]
        Hi1 = Hinv[i1:i2, i1:i2]; Err = torch.zeros_like(W1); Q1 = torch.zeros_like(W1)
        for j in range(n):
            w = W1[:, j]
            ix = (lv - w[:, None]).abs().argmin(1, keepdim=True); q = lv.gather(1, ix)[:, 0]
            B1[:, i1 + j] = ix[:, 0] < 2; B2[:, i1 + j] = (ix[:, 0] == 0) | (ix[:, 0] == 2)
            Q1[:, j] = q; err = (w - q) / Hi1[j, j]
            W1[:, j:] -= err[:, None] * Hi1[j, j:][None, :]; Err[:, j] = err
        Q[:, i1:i2] = Q1; S1.append(s1); S2.append(s2)
        if i2 < C: W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    if planes: return Q, B1, B2, torch.stack(S1, 1), torch.stack(S2, 1)
    return Q


def r2_lloyd(W, h=None, iters=6, scales=False):
    """alternate (a) nearest of the four levels per weight (b) importance weighted least squares for (s1, s2);
    bits b1, b2 per weight are the sign bits of the two q1 planes with scales s1, s2 (fp16)"""
    R, C = W.shape; P, m = group_view(W)
    hh = (torch.ones(C, device=W.device) if h is None else h)
    hp = torch.zeros(P.shape[1] * G, device=W.device); hp[:C] = hh; hp = hp.view(-1, G)
    s1 = (P.abs() * hp).sum(2) / hp.sum(1); s2 = 0.35 * s1
    for _ in range(iters):
        a = s1[..., None]; b = s2[..., None]
        lv = torch.stack([a + b, a - b, -a + b, -a - b], 0)                    # [4, R, ng, 1]
        idx = ((P[None] - lv) ** 2).argmin(0)
        b1 = torch.where(idx < 2, 1.0, -1.0) * m; b2 = torch.where((idx == 0) | (idx == 2), 1.0, -1.0) * m
        a11 = (hp * b1 * b1).sum(2); a12 = (hp * b1 * b2).sum(2); a22 = (hp * b2 * b2).sum(2)
        c1 = (hp * b1 * P).sum(2); c2 = (hp * b2 * P).sum(2)
        det = (a11 * a22 - a12 * a12).clamp_min(1e-12)
        s1 = (c1 * a22 - c2 * a12) / det; s2 = (a11 * c2 - a12 * c1) / det
    s1 = s1.half().float(); s2 = s2.half().float()
    if scales: return s1[:, 0], s2[:, 0]
    return (b1 * s1[..., None] + b2 * s2[..., None]).view(R, -1)[:, :C]
