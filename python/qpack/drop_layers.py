"""Drop whole transformer layers from a QPACK container (any weight format), renumber the rest.

    python -m qpack.drop_layers in.qpack out.qpack --drop 12 17

Layer tensors are those named `model.layers.<i>.*` (manifest name); blobs of dropped tensors are not copied, the remaining
layers are renumbered 0..n-1, config.json num_hidden_layers (and layer_types / max_window_layers if present) is updated.
selftest.json is removed (reference logits belong to the full network)."""
import argparse, json, re
from .container import Container, write_container

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("--drop", type=int, nargs="+", required=True)
    a = ap.parse_args(); drop = set(a.drop)
    c = Container(a.inp); man = c.json("manifest.json"); cfg = c.json("config.json"); n = cfg["num_hidden_layers"]
    assert all(0 <= d < n for d in drop)
    remap = {}; j = 0
    for i in range(n):
        if i not in drop: remap[i] = j; j += 1
    pat = re.compile(r"^(.*?layers\.)(\d+)(\..*)$"); blobs = {}; tens = []; keep_blobs = set()
    for e in man["tensors"]:
        m = pat.match(e["name"])
        if m:
            i = int(m.group(2))
            if i in drop: continue
            e = dict(e); e["name"] = f"{m.group(1)}{remap[i]}{m.group(3)}"
        tens.append(e)
        for k in ("blob", "scale", "alpha"):
            if k in e: keep_blobs.add(e[k])
    for name in c.toc:
        if name in keep_blobs or name in ("config.json", "generation_config.json", "tokenizer.json"): blobs[name] = bytes(c.blob(name))
    man["tensors"] = tens; blobs["manifest.json"] = json.dumps(man).encode()
    cfg["num_hidden_layers"] = j
    if "layer_types" in cfg: cfg["layer_types"] = [t for i, t in enumerate(cfg["layer_types"]) if i not in drop]
    if "max_window_layers" in cfg: cfg["max_window_layers"] = min(cfg["max_window_layers"], j)
    blobs["config.json"] = json.dumps(cfg).encode()
    print("layers", n, "->", j, "bytes", write_container(a.out, blobs, magic=bytes(c.mm[:8])))
main()
