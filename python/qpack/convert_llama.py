"""Convert a Hugging Face Llama-family checkpoint (Llama, Qwen2, Mistral, SmolLM, ...) to a QPACK2 container.

    python -m qpack.convert_llama models/Qwen2.5-0.5B-Instruct out/qwen2.5-0.5b-instruct.q8.qpack [--quant q8]

Canonical tensor names (what the runtime reads):
  embed                         [vocab, d]            (also the output head when tie_embeddings)
  lm_head                       [vocab, d]            (untied models only)
  final_norm                    [d]
  layers.{i}.attn_norm / ffn_norm                     [d]
  layers.{i}.wq [nh*hd, d]  wk [nkv*hd, d]  wv [nkv*hd, d]  wo [d, nh*hd]   (+ bq / bk / bv when the model has q/k/v biases)
  layers.{i}.w_gate [ff, d]  w_up [ff, d]  w_down [d, ff]
config.json: {"arch": "llama", dims, rope, eps, tie, qkv_bias, chat template, special token ids}. The selftest is computed
on the dequantised model, so it checks the runtime, not the rounding."""
import argparse, json, os, sys

import numpy as np
import torch

from .container import quantize_q8, write_container, MAGIC2

CHATML = "chatml"


def canonical_state(model):
    sd = model.state_dict(); cfg = model.config; out = {}
    out["embed"] = sd["model.embed_tokens.weight"]
    if not cfg.tie_word_embeddings:
        out["lm_head"] = sd["lm_head.weight"]
    out["final_norm"] = sd["model.norm.weight"]
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}."
        m = {"attn_norm": "input_layernorm.weight", "ffn_norm": "post_attention_layernorm.weight",
             "wq": "self_attn.q_proj.weight", "wk": "self_attn.k_proj.weight", "wv": "self_attn.v_proj.weight", "wo": "self_attn.o_proj.weight",
             "bq": "self_attn.q_proj.bias", "bk": "self_attn.k_proj.bias", "bv": "self_attn.v_proj.bias",
             "w_gate": "mlp.gate_proj.weight", "w_up": "mlp.up_proj.weight", "w_down": "mlp.down_proj.weight"}
        for k, v in m.items():
            if p + v in sd:
                out[f"layers.{i}.{k}"] = sd[p + v]
    return out


def runtime_config(cfg, tok_cfg):
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    # transformers >= 5 keeps RoPE settings in rope_parameters (rope_theta is no longer an attribute): read both layouts
    rp = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
    if rp.get("rope_type", rp.get("type")) not in (None, "default"):
        raise SystemExit(f"RoPE {rp} is not supported yet (only the default rotary embedding)")
    theta = rp.get("rope_theta", getattr(cfg, "rope_theta", None))
    if theta is None:
        raise SystemExit("rope_theta not found in the config")
    tmpl = (tok_cfg.get("chat_template") or "")
    return {
        "arch": "llama", "model_type": cfg.model_type, "hidden_size": cfg.hidden_size, "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads, "num_key_value_heads": cfg.num_key_value_heads, "head_dim": hd,
        "intermediate_size": cfg.intermediate_size, "vocab_size": cfg.vocab_size, "rope_theta": float(theta), "max_position_embeddings": int(cfg.max_position_embeddings),
        "rms_norm_eps": float(cfg.rms_norm_eps), "tie_embeddings": bool(cfg.tie_word_embeddings),
        "qkv_bias": bool(getattr(cfg, "attention_bias", False) or cfg.model_type == "qwen2"),
        "bos_token_id": cfg.bos_token_id, "eos_token_id": cfg.eos_token_id,
        "chat_format": CHATML if "<|im_start|>" in tmpl else "raw", "chat_template": tmpl,
    }


def pack(state, rcfg, tokenizer_json, gen_cfg, quant, selftest=None):
    blobs, man = {}, []
    for k, (name, w) in enumerate(state.items()):
        bid = f"w{k:04d}"; w = w.detach().float().cpu(); ent = dict(name=name, shape=list(w.shape), dtype="float32")
        if quant == "q8" and w.ndim == 2 and w.numel() >= 4096:
            q, sc = quantize_q8(w); blobs[bid + ".q8"] = q.tobytes(); blobs[bid + ".scale"] = sc.tobytes(); ent.update(kind="q8", blob=bid + ".q8", scale=bid + ".scale")
        else:
            blobs[bid + ".f32"] = w.numpy().astype(np.float32).tobytes(); ent.update(kind="f32", blob=bid + ".f32")
        man.append(ent)
    blobs["manifest.json"] = json.dumps(dict(format="QPACK2" if quant else "QPACK1", tensors=man, quant=quant)).encode()
    blobs["config.json"] = json.dumps(rcfg).encode()
    blobs["tokenizer.json"] = tokenizer_json
    blobs["generation_config.json"] = json.dumps(gen_cfg).encode()
    if selftest:
        blobs["selftest.json"] = json.dumps(selftest).encode()
    return blobs


def dequantize_inplace(model, quant):
    """replace the model's matrices by their quantised-then-dequantised values (what the container stores)"""
    if quant != "q8":
        return
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.ndim == 2 and p.numel() >= 4096:
                q, sc = quantize_q8(p); p.copy_(torch.from_numpy(q.astype(np.float32) * sc[:, None]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--quant", default="q8", choices=["q8", "none"])
    a = ap.parse_args(); quant = None if a.quant == "none" else a.quant
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(a.src, torch_dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(a.src)
    tok_cfg = json.load(open(os.path.join(a.src, "tokenizer_config.json")))
    rcfg = runtime_config(model.config, tok_cfg)
    rcfg["special_tokens"] = {t: tok.convert_tokens_to_ids(t) for t in tok.all_special_tokens}
    gen_path = os.path.join(a.src, "generation_config.json")
    gen_cfg = json.load(open(gen_path)) if os.path.exists(gen_path) else {}
    dequantize_inplace(model, quant)
    msgs = [[{"role": "user", "content": "Hello! Who are you?"}], [{"role": "user", "content": "Сколько будет 2+2?"}]]
    cases = [tok.encode(t) for t in ("The capital of France is", "Москва — столица", "def fibonacci(n):")]
    if rcfg["chat_format"] != "raw":
        cases += [tok.encode(tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False), add_special_tokens=False) for m in msgs]
    out = []
    with torch.no_grad():
        for ids in cases:
            lg = model(torch.tensor([ids])).logits[0, -1].float(); v, i = lg.topk(5)
            out.append(dict(ids=list(map(int, ids)), top5=i.tolist(), top5_logits=v.tolist()))
    blobs = pack(canonical_state(model), rcfg, open(os.path.join(a.src, "tokenizer.json"), "rb").read(), gen_cfg, quant, dict(cases=out))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    size = write_container(a.out, blobs)
    print(f"{a.out}: {size / 2**20:.1f} MiB, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, {len(blobs)} blobs, "
          f"format {'QPACK2 ' + quant if quant else 'QPACK1'}", file=sys.stderr)


if __name__ == "__main__":
    main()
