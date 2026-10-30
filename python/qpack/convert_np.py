"""Torch-free converter: Hugging Face safetensors (bf16 / f16 / f32, sharded or not) -> unquantised f32 QPACK container.

    python -m qpack.convert_np models/Qwen2.5-0.5B-Instruct out/qwen2.5-0.5b.f32.qpack      (Llama family: Llama, Qwen2, Mistral, MiniCPM5, ...)
    python -m qpack.convert_np models/LFM3-1.7B out/lfm3-1.7b.f32.qpack                      (Liquid LFM3: short-conv + attention hybrid)

Then compress with the C++ tool: qpack-quantize out.f32.qpack out.q8.qpack --type q8   (or q4 / q4m, see README).
Tensor names follow convert_llama.py for the Llama family; LFM3 uses
  embed [V, d] (tied head), final_norm [d]
  layers.{i}.operator_norm / ffn_norm / post_operator_norm / post_ffn_norm [d]
  conv layers:      conv_in [3d, d]  conv_w [d * K] (flat, kept f32)  conv_out [d, d]
  attention layers: wq wk wv wo, q_norm / k_norm [head_dim], gate [heads, d]
  every layer:      w_gate (w1) [ff, d]  w_up (w3) [ff, d]  w_down (w2) [d, ff]
No selftest.json is written (that needs the reference implementation)."""
import glob, json, os, struct, sys

import numpy as np

from .container import write_container


class Safetensors:
    def __init__(self, folder):
        self.files = []; self.index = {}
        for p in sorted(glob.glob(os.path.join(folder, "*.safetensors"))):
            raw = np.memmap(p, dtype=np.uint8, mode="r"); hl = struct.unpack("<Q", bytes(raw[:8]))[0]
            hdr = json.loads(bytes(raw[8:8 + hl])); base = 8 + hl
            for k, e in hdr.items():
                if k != "__metadata__": self.index[k] = (raw, base, e)
        if not self.index: raise SystemExit(f"no .safetensors files in {folder}")

    def __contains__(self, k): return k in self.index

    def get(self, k):
        raw, base, e = self.index[k]; a, b = e["data_offsets"]; buf = np.asarray(raw[base + a:base + b])
        if e["dtype"] == "BF16": return (buf.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(e["shape"])
        if e["dtype"] == "F16": return buf.view(np.float16).astype(np.float32).reshape(e["shape"])
        if e["dtype"] == "F32": return buf.view(np.float32).reshape(e["shape"])
        raise SystemExit(f"{k}: unsupported dtype {e['dtype']}")


def llama_state(st, cfg):
    sd = {"embed": st.get("model.embed_tokens.weight"), "final_norm": st.get("model.norm.weight")}
    if not cfg.get("tie_word_embeddings", False): sd["lm_head"] = st.get("lm_head.weight")
    m = {"attn_norm": "input_layernorm.weight", "ffn_norm": "post_attention_layernorm.weight", "wq": "self_attn.q_proj.weight", "wk": "self_attn.k_proj.weight",
         "wv": "self_attn.v_proj.weight", "wo": "self_attn.o_proj.weight", "bq": "self_attn.q_proj.bias", "bk": "self_attn.k_proj.bias", "bv": "self_attn.v_proj.bias",
         "w_gate": "mlp.gate_proj.weight", "w_up": "mlp.up_proj.weight", "w_down": "mlp.down_proj.weight"}
    for i in range(cfg["num_hidden_layers"]):
        for k, v in m.items():
            if f"model.layers.{i}.{v}" in st: sd[f"layers.{i}.{k}"] = st.get(f"model.layers.{i}.{v}")
    return sd


def lfm3_state(st, cfg):
    sd = {"embed": st.get("model.embed_tokens.weight"), "final_norm": st.get("model.embedding_norm.weight")}
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        for k, v in {"operator_norm": "operator_norm", "ffn_norm": "ffn_norm", "post_operator_norm": "post_operator_norm", "post_ffn_norm": "post_ffn_norm"}.items():
            if p + v + ".weight" in st: sd[f"layers.{i}.{k}"] = st.get(p + v + ".weight")
        for k, v in {"w_gate": "feed_forward.w1", "w_up": "feed_forward.w3", "w_down": "feed_forward.w2"}.items(): sd[f"layers.{i}.{k}"] = st.get(p + v + ".weight")
        if cfg["layer_types"][i] == "conv":
            sd[f"layers.{i}.conv_in"] = st.get(p + "conv.in_proj.weight"); sd[f"layers.{i}.conv_out"] = st.get(p + "conv.out_proj.weight")
            sd[f"layers.{i}.conv_w"] = st.get(p + "conv.conv.weight").reshape(-1)
            if p + "conv.in_proj.bias" in st: raise SystemExit("LFM3 with conv_bias is not supported")
        else:
            for k, v in {"wq": "q_proj", "wk": "k_proj", "wv": "v_proj", "wo": "o_proj", "gate": "gate_proj"}.items(): sd[f"layers.{i}.{k}"] = st.get(p + f"self_attn.{v}.weight")
            sd[f"layers.{i}.q_norm"] = st.get(p + "self_attn.q_norm.weight"); sd[f"layers.{i}.k_norm"] = st.get(p + "self_attn.k_norm.weight")
    return sd


def main():
    if len(sys.argv) != 3: raise SystemExit(__doc__)
    src, out = sys.argv[1:]
    cfg = json.load(open(f"{src}/config.json")); tc = json.load(open(f"{src}/tokenizer_config.json"))
    tmpl = tc.get("chat_template") or ""
    if not tmpl and os.path.exists(f"{src}/chat_template.jinja"): tmpl = open(f"{src}/chat_template.jinja").read()
    st = Safetensors(src); mt = cfg["model_type"]
    gen = json.load(open(f"{src}/generation_config.json")) if os.path.exists(f"{src}/generation_config.json") else {}
    tj = json.load(open(f"{src}/tokenizer.json"))
    common = {"chat_template": tmpl, "chat_format": "chatml" if "<|im_start|>" in tmpl else "raw", "model_type": mt,
              "bos_token_id": cfg.get("bos_token_id"), "eos_token_id": cfg.get("eos_token_id", gen.get("eos_token_id")),
              "special_tokens": {t["content"]: t["id"] for t in tj["added_tokens"] if t.get("special")}}
    hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    if mt == "lfm3":
        sd = lfm3_state(st, cfg); rp = cfg["rope_parameters"]
        rc = dict(common, arch="lfm3", hidden_size=cfg["hidden_size"], num_hidden_layers=cfg["num_hidden_layers"], layer_types=cfg["layer_types"],
                  num_attention_heads=cfg["num_attention_heads"], num_key_value_heads=cfg["num_key_value_heads"], head_dim=hd,
                  intermediate_size=cfg["intermediate_size"], vocab_size=cfg["vocab_size"], conv_L_cache=cfg["conv_L_cache"], norm_eps=float(cfg["norm_eps"]),
                  sliding_window=cfg["sliding_window"], max_position_embeddings=cfg["max_position_embeddings"], tie_embeddings=bool(cfg.get("tie_word_embeddings", True)),
                  rope={t: {"theta": float(v["rope_theta"]), "partial": float(v.get("partial_rotary_factor", 1.0))} for t, v in rp.items()})
    else:
        rp = cfg.get("rope_parameters") or cfg.get("rope_scaling") or {}
        if rp.get("rope_type", rp.get("type")) not in (None, "default"): raise SystemExit(f"RoPE {rp} is not supported yet")
        sd = llama_state(st, cfg)
        rc = dict(common, arch="llama", hidden_size=cfg["hidden_size"], num_hidden_layers=cfg["num_hidden_layers"], num_attention_heads=cfg["num_attention_heads"],
                  num_key_value_heads=cfg["num_key_value_heads"], head_dim=hd, intermediate_size=cfg["intermediate_size"], vocab_size=cfg["vocab_size"],
                  rope_theta=float(rp.get("rope_theta", cfg.get("rope_theta", 10000))), max_position_embeddings=cfg["max_position_embeddings"],
                  rms_norm_eps=float(cfg["rms_norm_eps"]), tie_embeddings=bool(cfg.get("tie_word_embeddings", False)), qkv_bias=mt == "qwen2")
    blobs, man = {}, []
    for k, (name, w) in enumerate(sd.items()):
        bid = f"w{k:04d}"; blobs[bid + ".f32"] = np.ascontiguousarray(w, dtype=np.float32).tobytes()
        man.append(dict(name=name, shape=list(w.shape), dtype="float32", kind="f32", blob=bid + ".f32"))
    blobs["manifest.json"] = json.dumps(dict(format="QPACK1", tensors=man, quant=None)).encode()
    blobs["config.json"] = json.dumps(rc).encode()
    blobs["tokenizer.json"] = open(f"{src}/tokenizer.json", "rb").read()
    blobs["generation_config.json"] = json.dumps(gen).encode()
    size = write_container(out, blobs)
    print(f"{out}: {size / 2**20:.1f} MiB, {sum(w.size for w in sd.values()) / 1e6:.1f}M params ({rc['arch']})", file=sys.stderr)


if __name__ == "__main__":
    main()
