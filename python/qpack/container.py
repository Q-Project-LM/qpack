"""QPACK1: single-file packed container for N-Q models (same on-disk format as Q-164M / Q-Omni: header, table of contents,
64-byte aligned blobs, mmap-friendly).

  magic  "QPACK1\\0\\0" (8 bytes) | count u32 | pad u32 | count x [name[56] NUL-padded, offset u64, length u64] | blobs, each aligned to 64 bytes

Contents: manifest.json, config.json, generation_config.json, tokenizer.json, selftest.json,
  ternary projections  -> <id>.tern5  (5 trits per byte = 1.6 bits/weight) + <id>.alpha (fp32 per-row scale)
  frozen fingerprints  -> codes.bits   (+-1 codes, 1 bit each)
  integer buffers      -> <id>.i64    (n-gram hash sizes / offsets / multipliers: must be exact)
  QPACK2 only: int8    -> <id>.q8 + <id>.scale (int8 rows, float32 scale per row; pack_model(quant='q8'))
  1-D float tensors    -> <id>.f32    (norms, decay params, unigram prior: tiny, kept exact)
  everything else      -> <id>.f16 (default) or .f32 (--dense f32)
Which projections are ternary is read from the model itself (Linear modules with .ternary=True), so the same packer covers
the dense qwen4 ladder, the ternary hybrid and the Q-164M-style arch. Ternary weights are stored losslessly.

    python qpack.py --selftest model.qpack      # loads the container into torch and checks the stored top-5 logits
"""
import json, mmap, os, struct, sys
import numpy as np

MAGIC, MAGIC2, ALIGN = b"QPACK1\0\0", b"QPACK2\0\0", 64   # QPACK2 = same layout + int8 'q8' tensors
_LUT = None


def _lut():
    global _LUT
    if _LUT is None:
        v = np.arange(243, dtype=np.int32); _LUT = np.stack([(v // 3 ** i) % 3 for i in range(5)], 1).astype(np.int8)
    return _LUT


def pack_tern5(t):
    """t: int array in {-1,0,1}, any shape -> bytes (row-major, 5 trits per byte)"""
    d = (np.asarray(t, dtype=np.int8).reshape(-1) + 1).astype(np.uint8); pad = (-len(d)) % 5
    if pad: d = np.concatenate([d, np.ones(pad, np.uint8)])
    d = d.reshape(-1, 5).astype(np.uint16); return (d[:, 0] + 3 * d[:, 1] + 9 * d[:, 2] + 27 * d[:, 3] + 81 * d[:, 4]).astype(np.uint8).tobytes()


def unpack_tern5(buf, n):
    return (_lut()[np.frombuffer(buf, dtype=np.uint8)].reshape(-1)[:n] - 1).astype(np.int8)


def write_container(path, blobs, magic=None):
    names = list(blobs); head = 16 + 72 * len(names); off = (head + ALIGN - 1) // ALIGN * ALIGN; ent = []
    for n in names:
        if len(n.encode()) > 55: raise ValueError("blob name too long: " + n)
        ent.append((n, off, len(blobs[n]))); off = (off + len(blobs[n]) + ALIGN - 1) // ALIGN * ALIGN
    with open(path, "wb") as o:
        if magic is None:   # QPACK2 as soon as the container holds int8 tensors, QPACK1 otherwise
            magic = MAGIC2 if any(n.endswith(".q8") for n in names) else MAGIC
        o.write(magic + struct.pack("<II", len(names), 0))
        for n, of, ln in ent: o.write(n.encode().ljust(56, b"\0") + struct.pack("<QQ", of, ln))
        for n, of, ln in ent:
            o.write(b"\0" * (of - o.tell())); o.write(blobs[n])
    return os.path.getsize(path)


class Container:
    def __init__(self, path):
        self.f = open(path, "rb"); self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        if self.mm[:8] not in (MAGIC, MAGIC2): raise ValueError("not a QPACK1/QPACK2 container: " + str(path))
        self.version = 2 if self.mm[:8] == MAGIC2 else 1
        n, _ = struct.unpack("<II", self.mm[8:16]); self.toc = {}
        for i in range(n):
            b = 16 + 72 * i; name = self.mm[b:b + 56].rstrip(b"\0").decode(); off, ln = struct.unpack("<QQ", self.mm[b + 56:b + 72]); self.toc[name] = (off, ln)

    def blob(self, name): off, ln = self.toc[name]; return self.mm[off:off + ln]
    def json(self, name): return json.loads(bytes(self.blob(name)).decode())
    def close(self): self.mm.close(); self.f.close()


def ternary_weight_names(model):
    """state_dict names of the weights the model re-quantises to {-1,0,+1}*alpha on every forward"""
    return {f"{n}.weight" for n, m in model.named_modules() if getattr(m, "ternary", False) and hasattr(m, "weight")}


def quantize_q8(w):
    """per-row symmetric int8: w ~= q * scale[row], q in [-127, 127]. Returns (int8 array, float32 scales)."""
    a = w.detach().cpu().float().numpy().reshape(w.shape[0], -1)
    scale = np.abs(a).max(1) / 127.0; scale[scale == 0] = 1.0
    q = np.clip(np.rint(a / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def pack_model(model, tokenizer_json=None, gen_cfg=None, dense="f16", selftest=None, quant=None):
    """model: an N-Q model (NQ4ForCausalLM / NQForCausalLM). Returns dict of blobs for write_container.
    quant='q8' (QPACK2): every 2-D float matrix with >= 4096 entries is stored as int8 rows + a float32 scale per row."""
    import torch
    tern = ternary_weight_names(model); state = model.state_dict()
    blobs, man, k = {}, [], 0
    dt, dk = (np.float16, "f16") if dense == "f16" else (np.float32, "f32")
    for name, w in state.items():
        ent = dict(name=name, shape=list(w.shape), dtype=str(w.dtype).replace("torch.", "")); bid = f"w{k:04d}"; k += 1
        if not w.is_floating_point():
            blobs[bid + ".i64"] = w.detach().cpu().numpy().astype(np.int64).tobytes(); ent.update(kind="i64", blob=bid + ".i64")
        elif name in tern:   # mirror Linear.forward exactly (same op order), so the stored trits are the ones the QAT model computes with
            wf = w.detach().cpu().float(); al = wf.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
            t = torch.clamp(torch.round(wf / al), -1, 1).numpy().astype(np.int8)
            blobs[bid + ".tern5"] = pack_tern5(t); blobs[bid + ".alpha"] = al.numpy().astype(np.float32).reshape(-1).tobytes()
            ent.update(kind="tern5", blob=bid + ".tern5", alpha=bid + ".alpha")
        elif name.endswith("codes") and w.ndim == 2:
            a = w.detach().cpu().float().numpy(); blobs["codes.bits"] = np.packbits((a > 0).astype(np.uint8), axis=1).tobytes(); ent.update(kind="bits", blob="codes.bits")
        elif quant == "q8" and w.ndim == 2 and w.numel() >= 4096:
            q, sc = quantize_q8(w)
            blobs[bid + ".q8"] = q.tobytes(); blobs[bid + ".scale"] = sc.tobytes(); ent.update(kind="q8", blob=bid + ".q8", scale=bid + ".scale")
        elif w.ndim <= 1 or name.endswith("logit_bias"):
            blobs[bid + ".f32"] = w.detach().cpu().float().numpy().tobytes(); ent.update(kind="f32", blob=bid + ".f32")
        else:
            blobs[bid + "." + dk] = w.detach().cpu().float().numpy().astype(dt).tobytes(); ent.update(kind=dk, blob=bid + "." + dk)
        man.append(ent)
    cfg = model.config.to_dict(); cfg["model_type"] = model.config.model_type
    blobs["manifest.json"] = json.dumps(dict(format="QPACK2" if quant else "QPACK1", tensors=man, dense=dense, quant=quant)).encode(); blobs["config.json"] = json.dumps(cfg).encode()
    if tokenizer_json: blobs["tokenizer.json"] = tokenizer_json if isinstance(tokenizer_json, bytes) else tokenizer_json.encode()
    if gen_cfg: blobs["generation_config.json"] = json.dumps(gen_cfg).encode()
    if selftest: blobs["selftest.json"] = json.dumps(selftest).encode()
    return blobs


def make_selftest(model, cases):
    """cases: list of token-id lists. Stores the original model's last-position top-5 ids and logits (fp32)."""
    import torch
    out = []
    model.eval()
    with torch.no_grad():
        for ids in cases:
            lg = model(torch.tensor([ids]), return_logits=True).logits[0, -1].float()
            v, i = lg.topk(5); out.append(dict(ids=list(ids), top5=i.tolist(), top5_logits=v.tolist()))
    return dict(cases=out)


def load_state(path):
    """QPACK1 container -> (state_dict of torch tensors ready for load_state_dict, config dict). Ternary weights come back as latent
    weights W' = t * alpha / f  (f = fraction of non-zeros in the row), for which the QAT forward pass (absmean re-quantisation)
    reproduces exactly t * alpha."""
    import torch
    c = Container(path); man = c.json("manifest.json"); cfg = c.json("config.json"); state = {}
    for e in man["tensors"]:
        shp = e["shape"]; n = int(np.prod(shp)) if shp else 1
        if e["kind"] == "i64":
            state[e["name"]] = torch.from_numpy(np.frombuffer(c.blob(e["blob"]), dtype=np.int64).reshape(shp).copy()); continue
        if e["kind"] == "q8":
            q = np.frombuffer(c.blob(e["blob"]), dtype=np.int8).reshape(shp[0], -1).astype(np.float32)
            arr = (q * np.frombuffer(c.blob(e["scale"]), dtype=np.float32)[:, None]).reshape(shp)
            state[e["name"]] = torch.from_numpy(np.ascontiguousarray(arr)); continue
        if e["kind"] == "tern5":
            t = unpack_tern5(c.blob(e["blob"]), n).reshape(shp).astype(np.float32); alpha = np.frombuffer(c.blob(e["alpha"]), dtype=np.float32).reshape(-1, 1)
            f = np.maximum((t != 0).mean(1, keepdims=True), 1e-12); arr = np.where(t != 0, t * (alpha / f), 0.0).astype(np.float32)
        elif e["kind"] == "bits":
            arr = np.unpackbits(np.frombuffer(c.blob(e["blob"]), dtype=np.uint8).reshape(shp[0], -1), axis=1)[:, :shp[1]].astype(np.float32) * 2 - 1
        else:
            dtp = np.float16 if e["kind"] == "f16" else np.float32; arr = np.frombuffer(c.blob(e["blob"]), dtype=dtp).reshape(shp).astype(np.float32)
        tt = torch.from_numpy(np.ascontiguousarray(arr))
        state[e["name"]] = tt.half() if e.get("dtype") == "float16" else tt
    c.close(); return state, cfg


def selftest(path, model=None):
    """Load the container (unless a model is given) and compare against the stored reference: top-1 must match and the
    stored top-5 logits must agree within 5e-2 (dense f16 storage moves logits by ~1e-2)."""
    import torch
    c = Container(path); st = c.json("selftest.json"); c.close()
    if model is None:
        here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
        try:   # (in try/except so transformers' remote-code import check does not take it for a pip package)
            import modeling_nq4 as M
        except ImportError:
            import nq.model.modeling_nq4 as M
        c = Container(path); arch = (c.json("config.json").get("architectures") or ["NQ4ForCausalLM"])[0]; c.close()
        model = getattr(M, arch).from_qpack(path)   # NQ4ForCausalLM, or NQ4ForDecision for a JEV checkpoint
    ok = 0
    with torch.no_grad():
        for case in st["cases"]:
            lg = model(torch.tensor([case["ids"]]), return_logits=True).logits[0, -1].float()
            good = int(lg.argmax()) == case["top5"][0] and (lg[case["top5"]] - torch.tensor(case["top5_logits"])).abs().max().item() < 5e-2
            ok += good
    print(f"SELFTEST {ok}/{len(st['cases'])}"); return ok == len(st["cases"])


if __name__ == "__main__":
    if "--selftest" in sys.argv: sys.exit(0 if selftest(sys.argv[sys.argv.index("--selftest") + 1]) else 1)
    print(__doc__)
