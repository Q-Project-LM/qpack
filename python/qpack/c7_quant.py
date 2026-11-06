"""C7: 1-2 bit quantisers for gpt-oss experts (torch, fp32, GPU if available) and the layer-output proxy used to rank them.

A "plane" is the q1 format: per group of 128 columns, W ~ s * sign(W) (s fp16), tail group (2880 = 22*128 + 64) padded.
Quantisers return dequantised float tensors (what the engine would compute with) and, for packing, the sign bits + scales.
"""
import glob, os, sys, time, json, math
import numpy as np
import torch

from .convert_gptoss import SafeTensors
from .container import MX4_LUT

G = 128


def dev(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dq_mx4(blocks, scales):
    """HF mxfp4 rows (uint8 numpy [R,G,16], [R,G]) -> float32 torch [R, C] on dev"""
    b = torch.from_numpy(np.ascontiguousarray(blocks)).to(dev()).long()
    lut = torch.tensor(MX4_LUT, device=dev())
    v = torch.stack([lut[b & 15], lut[b >> 4]], -1).reshape(b.shape[0], b.shape[1], 32)
    s = torch.ldexp(torch.ones((), device=dev()), torch.from_numpy(scales.astype(np.int32)).to(dev()) - 127)
    return (v * s[..., None]).reshape(b.shape[0], -1)


def group_view(W):
    """[R, C] -> [R, ng, G] zero padded, mask [ng, G] of valid columns"""
    R, C = W.shape; ng = (C + G - 1) // G
    P = torch.zeros(R, ng * G, device=W.device, dtype=W.dtype); P[:, :C] = W
    m = torch.zeros(ng * G, device=W.device, dtype=W.dtype); m[:C] = 1
    return P.view(R, ng, G), m.view(ng, G)


def plane(W, h=None):
    """one q1 plane of W [R, C] with scale = sum h|w| / sum h per group (h None: plain mean |w|); scales rounded to fp16; returns dequantised [R, C]"""
    R, C = W.shape; P, m = group_view(W)
    hh = (torch.ones(C, device=W.device) if h is None else h)
    hp = torch.zeros(P.shape[1] * G, device=W.device); hp[:C] = hh; hp = hp.view(-1, G)
    s = (P.abs() * hp).sum(2) / hp.sum(1)
    s = s.half().float()
    sg = torch.where(P >= 0, 1.0, -1.0)
    return (sg * s[..., None]).view(R, -1)[:, :C]


def r2_ls(W, h=None):
    """two planes, scales jointly least-squares refit per group (signs: sign(W), sign(W - s1 sign W))"""
    R, C = W.shape; P, m = group_view(W)
    hh = (torch.ones(C, device=W.device) if h is None else h)
    hp = torch.zeros(P.shape[1] * G, device=W.device); hp[:C] = hh; hp = hp.view(-1, G)
    b1 = torch.where(P >= 0, 1.0, -1.0) * m
    s1 = ((P.abs() * hp).sum(2) / hp.sum(1)).half().float()
    Rr = P - b1 * s1[..., None]
    b2 = torch.where(Rr >= 0, 1.0, -1.0) * m
    # normal equations per group with weights hp
    a11 = (hp * b1 * b1).sum(2); a12 = (hp * b1 * b2).sum(2); a22 = (hp * b2 * b2).sum(2)
    c1 = (hp * b1 * P).sum(2); c2 = (hp * b2 * P).sum(2)
    det = (a11 * a22 - a12 * a12).clamp_min(1e-12)
    t1 = ((c1 * a22 - c2 * a12) / det).half().float(); t2 = ((a11 * c2 - a12 * c1) / det).half().float()
    return (b1 * t1[..., None] + b2 * t2[..., None]).view(R, -1)[:, :C]


def prep_hinv(H, damp):
    H = H.clone(); d = torch.arange(H.shape[0], device=H.device)
    H[d, d] += damp * H[d, d].mean().clamp_min(1e-8)
    L = torch.linalg.cholesky(H)
    Hi = torch.cholesky_inverse(L)
    return torch.linalg.cholesky(Hi, upper=True)


def gptq_plane(W, Hinv, h=None):
    """GPTQ with a q1 quantiser (group scale from the error-compensated weights at the start of each 128-block)"""
    R, C = W.shape; W = W.clone(); Q = torch.zeros_like(W)
    for i1 in range(0, C, G):
        i2 = min(C, i1 + G); W1 = W[:, i1:i2].clone(); n = i2 - i1
        hh = None if h is None else h[i1:i2]
        s = ((W1.abs() * (1 if hh is None else hh)).sum(1) / (n if hh is None else hh.sum())).half().float()
        Hi1 = Hinv[i1:i2, i1:i2]; Err = torch.zeros_like(W1); Q1 = torch.zeros_like(W1)
        for j in range(n):
            w = W1[:, j]; q = torch.where(w >= 0, 1.0, -1.0) * s
            Q1[:, j] = q; err = (w - q) / Hi1[j, j]
            W1[:, j:] -= err[:, None] * Hi1[j, j:][None, :]; Err[:, j] = err
        Q[:, i1:i2] = Q1
        if i2 < C: W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return Q


def gptq_planes(W, Hinv, nplanes, h=None):
    Q = torch.zeros_like(W)
    for _ in range(nplanes): Q = Q + gptq_plane(W - Q, Hinv, h)
    return Q


def swiglu(y, limit=7.0, alpha=1.702):
    g = y[:, ::2].clamp(max=limit); u = y[:, 1::2].clamp(-limit, limit)
    return g * torch.sigmoid(alpha * g) * (u + 1)
