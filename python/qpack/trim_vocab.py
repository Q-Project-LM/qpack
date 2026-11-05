"""Vocabulary trimming of a byte-level-BPE model in a QPACK container (tied or untied embedding), no runtime change.

    python -m qpack.trim_vocab in.f32.qpack out.f32.qpack --texts a.txt b.txt ... [--keep 48000] [--stats stats.json]

Token selection: count token frequencies of --texts with the model's own tokenizer; keep the most frequent tokens, then
close the set under BPE merges (for every kept token the two parts of its lowest-rank merge are kept too), always keep the
256 single-byte tokens and all added/special tokens. If the texts do not cover --keep tokens, the remaining slots are
filled with the lowest original ids (BPE training order ~ general frequency) whose text is Latin/Cyrillic/ASCII only, all Cyrillic tokens first.
Then: ids are remapped (regular tokens keep their relative order, added tokens go last), embed (and lm_head if present)
rows are sliced, merges that mention a removed token are dropped, tokenizer.json / config.json / selftest.json are
rewritten. Input must be a float container (f32/f16 embed); quantize afterwards with qpack-quantize.
"""
import argparse, json, struct, sys, unicodedata
import numpy as np
from .container import Container, write_container


def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)); cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def tok_text(tok, u2b):
    return bytes(u2b[c] for c in tok).decode("utf-8", "replace")


def allowed_script(s):
    for ch in s:
        if ord(ch) < 0x250 or 0x300 <= ord(ch) <= 0x36F or 0x400 <= ord(ch) <= 0x52F or 0x2000 <= ord(ch) <= 0x206F: continue
        if unicodedata.category(ch)[0] in "PSZ" and ord(ch) < 0x3000: continue
        return False
    return True


def is_cyr(s):
    return any(0x400 <= ord(ch) <= 0x52F for ch in s)


def trim(inp, outp, texts, keep=48000, stats_path=None):
    from tokenizers import Tokenizer
    c = Container(inp); tj = c.json("tokenizer.json"); cfg = c.json("config.json"); man = c.json("manifest.json")
    model = tj["model"]; vocab = model["vocab"]; merges = [tuple(m) if isinstance(m, list) else tuple(m.split(" ")) for m in model["merges"]]
    added = {a["content"]: a["id"] for a in tj["added_tokens"]}
    tk = Tokenizer.from_str(json.dumps(tj)); b2u = bytes_to_unicode(); u2b = {v: k for k, v in b2u.items()}
    cnt = np.zeros(max(vocab.values()) + 1 + len(added) + 8, np.int64); total_chars = 0
    for t in texts:
        s = open(t, encoding="utf-8").read(); total_chars += len(s)
        for i in range(0, len(s), 20000):   # chunk at line boundaries is unnecessary for counting
            for i_ in tk.encode(s[i:i + 20000]).ids: cnt[i_] += 1
    id2tok = {i: t for t, i in vocab.items()}
    first_merge = {}
    for r, (a, b) in enumerate(merges):
        first_merge.setdefault(a + b, (a, b))
    forced = {b2u[b] for b in range(256)}
    for t in forced: assert t in vocab, "missing byte token"
    def closure(seed):
        keep_set = set(seed) | forced; stack = list(keep_set)
        while stack:
            t = stack.pop()
            if t in first_merge:
                for p in first_merge[t]:
                    if p not in keep_set: keep_set.add(p); stack.append(p)
        return keep_set
    n_reg = len(vocab); target = keep - len(added)
    seen = [i for i in np.argsort(-cnt[:n_reg], kind="stable") if cnt[i] > 0 and i in id2tok]
    n_seen = len(seen)
    # grow the frequency-ranked seed until the merge-closed set reaches the target size
    lo, hi = 0, n_seen
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(closure({id2tok[i] for i in seen[:mid]})) <= target: lo = mid
        else: hi = mid - 1
    kept = closure({id2tok[i] for i in seen[:lo]}); n_obs = len(kept)
    filler = 0
    if len(kept) < target:   # fill with lowest original ids of Latin/Cyrillic-only tokens (plus their merge ancestors)
        txt = {i: tok_text(t, u2b) for i, t in id2tok.items()}
        cand = [i for i in range(n_reg) if i in txt and "\ufffd" not in txt[i] and allowed_script(txt[i])]
        cand = [i for i in cand if is_cyr(txt[i])] + [i for i in cand if not is_cyr(txt[i])]   # Cyrillic first, then Latin/ASCII by id
        for i in cand:
            if len(kept) >= target: break
            t = id2tok[i]
            if t in kept: continue
            add, stack = {t}, [t]
            while stack:
                x = stack.pop()
                for q in first_merge.get(x, ()):
                    if q not in kept and q not in add: add.add(q); stack.append(q)
            if len(kept) + len(add) <= target: filler += len(add); kept |= add
    # new ids: regular tokens in original id order, then added tokens in original order
    reg = sorted(kept, key=lambda t: vocab[t]); new_vocab = {t: i for i, t in enumerate(reg)}
    old_of_new = [vocab[t] for t in reg]
    n_new_reg = len(reg); amap = {}
    new_added = []
    for a in sorted(tj["added_tokens"], key=lambda a: a["id"]):
        nid = n_new_reg + len(new_added); amap[a["id"]] = nid; new_added.append(dict(a, id=nid)); old_of_new.append(a["id"])
    new_merges = [[a, b] if isinstance(model["merges"][0], list) else a + " " + b for a, b in merges if a in kept and b in kept and a + b in kept]
    V_old, V_new = cfg["vocab_size"], n_new_reg + len(new_added)
    idmap = {o: n for n, o in enumerate(old_of_new)}
    tj2 = json.loads(json.dumps(tj)); tj2["model"]["vocab"] = new_vocab; tj2["model"]["merges"] = new_merges; tj2["added_tokens"] = new_added
    cfg2 = dict(cfg); cfg2["vocab_size"] = V_new
    for k in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if k in cfg2 and cfg2[k] is not None:
            cfg2[k] = [idmap[x] for x in cfg2[k]] if isinstance(cfg2[k], list) else idmap[cfg2[k]]
    if "special_tokens" in cfg2: cfg2["special_tokens"] = {k: idmap[v] for k, v in cfg2["special_tokens"].items()}
    sel = c.json("selftest.json") if "selftest.json" in c.toc else None
    if sel:
        cases = []
        for cs in sel["cases"]:
            if any(i not in idmap for i in cs["ids"]): continue
            pairs = [(idmap[i], l) for i, l in zip(cs["top5"], cs["top5_logits"]) if i in idmap]
            if not pairs or pairs[0][0] != idmap.get(cs["top5"][0], -1): continue
            cases.append(dict(ids=[idmap[i] for i in cs["ids"]], top5=[p[0] for p in pairs], top5_logits=[p[1] for p in pairs]))
        sel = dict(sel, cases=cases)
    sel_idx = np.array(old_of_new, np.int64)
    blobs = {n: bytes(c.blob(n)) for n in c.toc if n not in ("manifest.json", "config.json", "tokenizer.json", "selftest.json", "generation_config.json")}
    gen = json.loads(bytes(c.blob("generation_config.json")).decode()) if "generation_config.json" in c.toc else None
    if gen:
        for k in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if k in gen and gen[k] is not None:
                gen[k] = [idmap[x] for x in gen[k]] if isinstance(gen[k], list) else idmap[gen[k]]
    for t in man["tensors"]:
        if t["shape"] and t["shape"][0] == V_old and t["name"] in ("embed", "lm_head"):
            dt = {"f32": np.float32, "f16": np.float16}[t["kind"]]
            a = np.frombuffer(blobs[t["blob"]], dt).reshape(V_old, -1)[sel_idx]
            blobs[t["blob"]] = np.ascontiguousarray(a).tobytes(); t["shape"][0] = V_new
        elif t["shape"] and V_old in t["shape"]: raise SystemExit("unexpected tensor with vocab dim: " + t["name"])
    blobs["manifest.json"] = json.dumps(man).encode(); blobs["config.json"] = json.dumps(cfg2).encode()
    blobs["tokenizer.json"] = json.dumps(tj2, ensure_ascii=False).encode()
    if gen: blobs["generation_config.json"] = json.dumps(gen).encode()
    if sel: blobs["selftest.json"] = json.dumps(sel).encode()
    # keep the original container's blob order, metadata last
    order = [n for n in c.toc if n in blobs]
    size = write_container(outp, {n: blobs[n] for n in order}, magic=bytes(c.mm[:8]))
    # coverage stats on the counting texts
    tk2 = Tokenizer.from_str(json.dumps(tj2)); n_old = n_new = 0
    for t in texts:
        s = open(t, encoding="utf-8").read(); n_old += len(tk.encode(s).ids); n_new += len(tk2.encode(s).ids)
    st = dict(vocab_old=V_old, vocab_new=V_new, regular_kept=n_new_reg, added=len(new_added), observed_types=int(n_seen), seed_types=lo,
              after_closure=n_obs, filler_tokens=filler, merges_old=len(merges), merges_new=len(new_merges), count_texts=texts,
              count_chars=total_chars, count_tokens_old=n_old, count_tokens_new=n_new, size=size)
    if stats_path: json.dump(st, open(stats_path, "w"), indent=1)
    c.close(); return st


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input"); ap.add_argument("output"); ap.add_argument("--texts", nargs="+", required=True)
    ap.add_argument("--keep", type=int, default=48000); ap.add_argument("--stats")
    a = ap.parse_args(); print(json.dumps(trim(a.input, a.output, a.texts, a.keep, a.stats), indent=1, ensure_ascii=False))
