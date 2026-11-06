"""Convert openai/gpt-oss-20b (HF safetensors, MXFP4 experts) to a QPACK2 container with bounded RAM (numpy only, no torch).

    python -m qpack.convert_gptoss /workspace/qpack-models/gpt-oss-20b out.qpack [--experts mxfp4|q8|q4|q1] [--attn q8|q4|f16]
                                   [--embed q8|q4|f16] [--router f32|q8|f16] [--selftest ref.json] [--layers N]

Everything is streamed: tensors are read from the safetensors mmap and written in row chunks (the table of contents is written first
from a plan computed from the safetensors headers, see StreamWriter), so peak RAM is a few hundred MB.

Experts (96% of the weights) default to `mxfp4`: the checkpoint's own nibbles and e8m0 scales, kept exactly (nibbles only permuted to
the kernel layout, byte j = column j low / column j + 16 high). q8 / q4 / q1 dequantise each expert chunk and re-quantise it (RTN).

Canonical tensor names (what src/model_gptoss.cpp reads):
  embed, lm_head [vocab, d]; final_norm [d]
  layers.{i}.attn_norm / ffn_norm [d] f32; wq [4096, d] wk [512, d] wv [512, d] wo [d, 4096]; bq bk bv bo f32; sinks [64] f32
  layers.{i}.router_w [32, d] router_b [32]; w_gate_up [32, 5760, d] (rows interleaved: 2j = gate j, 2j + 1 = up j); b_gate_up [32, 5760] f32;
  w_down [32, d, 2880]; b_down [32, d] f32
"""
import argparse, glob, json, mmap, os, struct, sys, time

import numpy as np

from .container import MX4_LUT, StreamWriter, dequant_mxfp4

CHUNK = 1 << 24   # values per processing chunk


class SafeTensors:
    """minimal safetensors reader over mmap (no copies until a slice is converted)"""

    def __init__(self, files):
        self.t = {}; self.maps = []
        for f in files:
            fh = open(f, "rb"); mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ); n = struct.unpack("<Q", mm[:8])[0]
            h = json.loads(mm[8:8 + n]); base = 8 + n; self.maps.append((fh, mm))
            for k, v in h.items():
                if k != "__metadata__": self.t[k] = (mm, base + v["data_offsets"][0], v["dtype"], v["shape"])

    def shape(self, k): return self.t[k][3]

    def get(self, k, r0=0, r1=None):
        """rows [r0, r1) of tensor k viewed over the first dimension (flattened [shape[0] * ..., last] for 3-D / 4-D uint8 blocks)"""
        mm, off, dt, shp = self.t[k]; esz = {"BF16": 2, "U8": 1, "F32": 4, "F16": 2}[dt]
        row = int(np.prod(shp[1:])) if len(shp) > 1 else 1
        r1 = shp[0] if r1 is None else r1
        raw = np.frombuffer(mm, dtype=np.uint16 if esz == 2 else np.uint8 if esz == 1 else np.uint32, count=(r1 - r0) * row, offset=off + r0 * row * esz)
        if dt == "BF16": return (raw.astype(np.uint32) << 16).view(np.float32).reshape((r1 - r0,) + tuple(shp[1:]))
        if dt == "F32": return raw.view(np.float32).reshape((r1 - r0,) + tuple(shp[1:]))
        return raw.reshape((r1 - r0,) + tuple(shp[1:]))


def f16(x): return np.asarray(x, np.float32).astype(np.float16)


def q8_rows(w):
    amax = np.abs(w).max(1); sc = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    return np.clip(np.rint(w / sc[:, None]), -127, 127).astype(np.int8).tobytes(), sc.tobytes()


def q8g_rows(w):
    """int8 with an fp16 scale per group of 32 columns (outlier rows only coarsen their own groups)"""
    R, C = w.shape; g = w.reshape(R, C // 32, 32)
    sc = f16(np.abs(g).max(2) / 127.0).astype(np.float32); sc1 = np.where(sc > 0, sc, 1.0)
    q = np.clip(np.rint(g / sc1[..., None]), -127, 127).astype(np.int8)
    return q.tobytes(), f16(sc).tobytes()


def q4_rows(w):
    """RTN-ish q4: groups of 32, scale from the extreme value (+7 / -8 side) refitted by least squares, fp16 scale"""
    R, C = w.shape; g = w.reshape(R, C // 32, 32)
    i = np.abs(g).argmax(2); m = np.take_along_axis(g, i[..., None], 2)[..., 0]
    s = f16(np.where(m > 0, m / 7.0, -m / 8.0)).astype(np.float32); s[s == 0] = 1.0
    q = np.clip(np.rint(g / s[..., None]), -8, 7)
    qq = (q * q).sum(2); s2 = f16(np.where(qq > 0, (g * q).sum(2) / np.maximum(qq, 1e-30), s)).astype(np.float32); s2[s2 == 0] = 1.0
    q2 = np.clip(np.rint(g / s2[..., None]), -8, 7)
    e1 = ((g - q * s[..., None]) ** 2).sum(2); e2 = ((g - q2 * s2[..., None]) ** 2).sum(2)
    use2 = (e2 < e1); q = np.where(use2[..., None], q2, q); s = np.where(use2, s2, s)
    allz = (np.abs(g).max(2) == 0); s = np.where(allz, 0.0, s)
    qi = (q.astype(np.int16) + 8).astype(np.uint8)
    packed = (qi[..., :16] | (qi[..., 16:] << 4)).astype(np.uint8)
    return packed.tobytes(), f16(s).tobytes()


def q1_rows(w):
    R, C = w.shape; G = (C + 127) // 128; pad = np.zeros((R, G * 128), np.float32); pad[:, :C] = w
    bits = np.packbits((pad >= 0).reshape(R, G * 128), axis=1, bitorder="little")
    n = np.minimum(128, C - 128 * np.arange(G)).astype(np.float32)
    sc = np.abs(pad).reshape(R, G, 128).sum(2) / n
    return bits.tobytes(), f16(sc).tobytes()


def mx4_rows(blocks, scales):
    """HF mxfp4 (byte j = values 2j low / 2j + 1 high) -> kernel layout (byte j = value j low / value j + 16 high); blocks [R, G, 16]"""
    nib = np.stack([blocks & 15, blocks >> 4], axis=-1).reshape(blocks.shape[0], blocks.shape[1], 32)
    return (nib[..., :16] | (nib[..., 16:] << 4)).astype(np.uint8).tobytes(), np.ascontiguousarray(scales).tobytes()


_PAIR = np.stack([MX4_LUT[np.arange(256) & 15], MX4_LUT[np.arange(256) >> 4]], axis=-1).astype(np.float32)   # [256, 2]


def dq_mx4_rows(blocks, scales):
    """HF mxfp4 rows -> float32 [R, C]"""
    v = _PAIR[blocks].reshape(blocks.shape[0], blocks.shape[1], 32)   # byte -> (low, high) value pair, in HF order
    return (v * np.ldexp(np.float32(1), scales.astype(np.int32) - 127)[..., None]).reshape(blocks.shape[0], -1)


def blob_sizes(kind, R, C):
    if kind == "f32": return {"": 4 * R * C}
    if kind == "f16": return {"": 2 * R * C}
    if kind == "q8": return {"": R * C, ".scale": 4 * R}
    if kind == "q8g": assert C % 32 == 0; return {"": R * C, ".scale": 2 * R * (C // 32)}
    if kind == "q4": assert C % 32 == 0; return {"": R * C // 2, ".scale": 2 * R * (C // 32)}
    if kind == "q1": G = (C + 127) // 128; return {"": R * G * 16, ".scale": 2 * R * G}
    if kind == "mxfp4": assert C % 32 == 0; return {"": R * C // 2, ".scale": R * (C // 32)}
    raise ValueError(kind)


EXT = {"f32": ".f32", "f16": ".f16", "q8": ".q8", "q8g": ".q8g", "q4": ".q4", "q1": ".q1", "mxfp4": ".mx4"}


def build(args, spec_hook=None):
    st = SafeTensors(sorted(glob.glob(os.path.join(args.src, "model-*.safetensors")) or glob.glob(os.path.join(args.src, "*.safetensors"))))
    cfg = json.load(open(os.path.join(args.src, "config.json"))); L = args.layers or cfg["num_hidden_layers"]
    specs = []   # (canonical name, kind, shape [E?, R, C], source producer id)

    def add(name, kind, shape, src): specs.append(dict(name=name, kind=kind, shape=list(shape), src=src))
    add("embed", args.embed, st.shape("model.embed_tokens.weight"), ("dense", "model.embed_tokens.weight"))
    for i in range(L):
        p = f"model.layers.{i}."; o = f"layers.{i}."
        add(o + "attn_norm", "f32", st.shape(p + "input_layernorm.weight"), ("dense", p + "input_layernorm.weight"))
        add(o + "ffn_norm", "f32", st.shape(p + "post_attention_layernorm.weight"), ("dense", p + "post_attention_layernorm.weight"))
        for n, h in (("wq", "q_proj"), ("wk", "k_proj"), ("wv", "v_proj"), ("wo", "o_proj")):
            add(o + n, args.attn, st.shape(p + f"self_attn.{h}.weight"), ("dense", p + f"self_attn.{h}.weight"))
            add(o + "b" + n[1], "f32", st.shape(p + f"self_attn.{h}.bias"), ("dense", p + f"self_attn.{h}.bias"))
        add(o + "sinks", "f32", st.shape(p + "self_attn.sinks"), ("dense", p + "self_attn.sinks"))
        add(o + "router_w", args.router, st.shape(p + "mlp.router.weight"), ("dense", p + "mlp.router.weight"))
        add(o + "router_b", "f32", st.shape(p + "mlp.router.bias"), ("dense", p + "mlp.router.bias"))
        for n, h in (("w_gate_up", "gate_up_proj"), ("w_down", "down_proj")):
            E, R, G, _ = st.shape(p + f"mlp.experts.{h}_blocks")
            add(o + n, args.experts, (E, R, G * 32), ("mx4", p + f"mlp.experts.{h}_blocks", p + f"mlp.experts.{h}_scales"))
            add(o + "b_" + n[2:], "f32", st.shape(p + f"mlp.experts.{h}_bias"), ("dense", p + f"mlp.experts.{h}_bias"))
    add("final_norm", "f32", st.shape("model.norm.weight"), ("dense", "model.norm.weight"))
    add("lm_head", args.head or args.embed, st.shape("lm_head.weight"), ("dense", "lm_head.weight"))
    if spec_hook: specs = spec_hook(specs, st, L)
    plan, man = [], []
    for k, s in enumerate(specs):
        shp = s["shape"]; C = shp[-1] if len(shp) > 1 else 1; R = int(np.prod(shp[:-1])) if len(shp) > 1 else 1
        if len(shp) == 1: R, C = 1, shp[0]
        bid = f"w{k:04d}"; sz = blob_sizes(s["kind"], R, C); s["bid"] = bid; s["RC"] = (R, C)
        ent = dict(name=s["name"], shape=shp, dtype="float32", kind=s["kind"], blob=bid + EXT[s["kind"]])
        for suf, n in sz.items(): plan.append((bid + (EXT[s["kind"]] if suf == "" else suf), n))
        if ".scale" in [x for x in sz]: ent["scale"] = bid + ".scale"
        if s["kind"] in ("q4", "mxfp4", "q8g"): ent["group"] = 32
        if s["kind"] == "q1": ent["group"] = 128
        man.append(ent)
    return st, cfg, L, specs, plan, man


def runtime_config(cfg, tok_cfg, args, L):
    rs = cfg["rope_scaling"]
    return {
        "arch": "gptoss", "model_type": "gpt_oss", "hidden_size": cfg["hidden_size"], "num_hidden_layers": L, "num_attention_heads": cfg["num_attention_heads"],
        "num_key_value_heads": cfg["num_key_value_heads"], "head_dim": cfg["head_dim"], "intermediate_size": cfg["intermediate_size"], "vocab_size": cfg["vocab_size"],
        "num_local_experts": cfg["num_local_experts"], "num_experts_per_tok": cfg["num_experts_per_tok"], "swiglu_limit": cfg["swiglu_limit"], "swiglu_alpha": 1.702,
        "sliding_window": cfg["sliding_window"], "layer_types": cfg["layer_types"][:L], "rms_norm_eps": cfg["rms_norm_eps"], "rope_theta": cfg["rope_theta"],
        "rope_scaling": {"type": "yarn", "factor": rs["factor"], "beta_fast": rs["beta_fast"], "beta_slow": rs["beta_slow"],
                         "original_max_position_embeddings": rs["original_max_position_embeddings"], "truncate": rs.get("truncate", False)},
        "max_position_embeddings": cfg["max_position_embeddings"], "tie_embeddings": False,
        "bos_token_id": 199998, "eos_token_id": cfg["eos_token_id"], "pad_token_id": cfg["pad_token_id"],
        "chat_format": "harmony", "chat_template": tok_cfg.get("chat_template", ""),
        "quant": {"experts": args.experts, "attn": args.attn, "embed": args.embed, "router": args.router},
    }


NPROC = int(os.environ.get("QPACK_CONVERT_PROCS", "4")); _POOL = None; _WST = None


def _expert_chunk(st, t):
    bk, sk, e, r0, r1, kind = t
    b = st.get(bk, e, e + 1)[0, r0:r1]; sc = st.get(sk, e, e + 1)[0, r0:r1]
    if kind == "mxfp4": return mx4_rows(b, sc)
    return {"q8": q8_rows, "q8g": q8g_rows, "q4": q4_rows, "q1": q1_rows}[kind](np.ascontiguousarray(dq_mx4_rows(b, sc), np.float32))


def _init_worker(src):
    global _WST
    _WST = SafeTensors(sorted(glob.glob(os.path.join(src, "model-*.safetensors"))))


def _expert_chunk_w(t): return _expert_chunk(_WST, t)


def _pool():
    global _POOL
    if _POOL is None:
        import multiprocessing as mp
        _POOL = mp.get_context("fork").Pool(NPROC, initializer=_init_worker, initargs=(ARGS_SRC,))
    return _POOL


ARGS_SRC = None


def emit(w, st, s, kind_args):
    """stream tensor spec s into the writer"""
    kind, (R, C) = s["kind"], s["RC"]; bid = s["bid"]; ext = EXT[kind]; scales = []
    src = s["src"]
    if src[0] == "dense":
        shp = st.shape(src[1])
        if len(shp) == 1: x = st.get(src[1]).reshape(1, -1); step = 1
        else: step = max(1, CHUNK // C)
        n = x.shape[0] if len(shp) == 1 else shp[0]
        for r0 in range(0, n, step):
            x = st.get(src[1], r0, min(n, r0 + step)).reshape(-1, C) if len(shp) > 1 else st.get(src[1]).reshape(1, -1)
            emit_rows(w, kind, bid, ext, x, scales)
    else:   # experts: HF blocks [E, R, G, 16] + scales [E, R, G]
        E, Rr, G, _ = st.t[src[1]][3]; step = max(1, CHUNK // (G * 32))
        tasks = [(src[1], src[2], e, r0, min(Rr, r0 + step), kind) for e in range(E) for r0 in range(0, Rr, step)]
        if kind == "mxfp4" or NPROC <= 1: results = map(lambda t: _expert_chunk(st, t), tasks)
        else: results = _pool().imap(_expert_chunk_w, tasks)   # requantisation is CPU-heavy: a few worker processes, results in order
        for d, s_ in results:
            w.write(bid + ext, d)
            if s_: scales.append(s_)
    if scales: w.write(bid + ".scale", b"".join(scales))


def emit_rows(w, kind, bid, ext, x, scales):
    if kind == "f32": w.write(bid + ext, np.ascontiguousarray(x, np.float32))
    elif kind == "f16": w.write(bid + ext, f16(x))
    else:
        d, s_ = {"q8": q8_rows, "q8g": q8g_rows, "q4": q4_rows, "q1": q1_rows}[kind](np.ascontiguousarray(x, np.float32)); w.write(bid + ext, d); scales.append(s_)


def patch_selftest(path, ref_json):
    """overwrite the (padded) selftest.json blob of an existing container with ref_json (must fit)"""
    from .container import Container
    c = Container(path); off, ln = c.toc["selftest.json"]; c.close()
    data = open(ref_json, "rb").read()
    if len(data) > ln: raise SystemExit(f"selftest ({len(data)} bytes) does not fit the reserved {ln}")
    with open(path, "r+b") as f: f.seek(off); f.write(data.ljust(ln, b" "))


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--patch-selftest": return patch_selftest(sys.argv[2], sys.argv[3])
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--experts", default="mxfp4", choices=["mxfp4", "q8", "q4", "q1"])
    ap.add_argument("--attn", default="q8", choices=["q8", "q8g", "q4", "f16"])
    ap.add_argument("--embed", default="q8", choices=["q8", "q8g", "q4", "f16"], help="embedding and lm_head")
    ap.add_argument("--head", default=None, choices=["q8", "q8g", "q4", "f16"], help="lm_head (default: same as --embed)")
    ap.add_argument("--router", default="f32", choices=["f32", "q8", "f16"])
    ap.add_argument("--layers", type=int, default=0, help="convert only the first N layers (debugging)")
    ap.add_argument("--selftest", default=None, help="selftest.json (reference top-5 logits) to embed")
    args = ap.parse_args()
    global ARGS_SRC; ARGS_SRC = args.src
    st, cfg, L, specs, plan, man = build(args)
    tok_cfg = json.load(open(os.path.join(args.src, "tokenizer_config.json")))
    extra = {"manifest.json": json.dumps(dict(format="QPACK2", tensors=man, quant=args.experts)).encode(),
             "config.json": json.dumps(runtime_config(cfg, tok_cfg, args, L)).encode(),
             "tokenizer.json": open(os.path.join(args.src, "tokenizer.json"), "rb").read()}
    gp = os.path.join(args.src, "generation_config.json")
    if os.path.exists(gp): extra["generation_config.json"] = open(gp, "rb").read()
    # selftest.json: embedded now, or a padded placeholder that `python -m qpack.convert_gptoss --patch-selftest file.qpack ref.json` overwrites in place
    extra["selftest.json"] = open(args.selftest, "rb").read() if args.selftest else b'{"cases":[]}'.ljust(32768, b" ")
    plan = [(n, len(b)) for n, b in extra.items()] + plan
    t0 = time.time(); w = StreamWriter(args.out, plan)
    for n, b in extra.items(): w.write(n, b)
    for i, s in enumerate(specs):
        emit(w, st, s, args)
        if i % 20 == 0: print(f"  {i}/{len(specs)} {s['name']} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    size = w.close()
    print(f"{args.out}: {size / 2**20:.1f} MiB ({args.experts} experts, {args.attn} attn, {args.embed} embed), {len(plan)} blobs, {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
