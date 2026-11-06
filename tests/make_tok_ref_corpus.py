"""reference ids from the Python `tokenizers` library on a self-contained varied corpus (no external data files needed beyond
the repo, the Python stdlib and unicodedata): prose in several scripts, code, special tokens, random Unicode, edge cases.
usage: python tests/make_tok_ref_corpus.py tokenizer.json ref.json [special tokens file-free]"""
import glob, json, os, random, sys, unicodedata
from tokenizers import Tokenizer

tj, out = sys.argv[1], sys.argv[2]
tok = Tokenizer.from_file(tj)
rnd = random.Random(1234)
texts = []
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
# 1) repo prose (English + Russian research notes) in chunks
for f in sorted(glob.glob(root + "/*.md") + glob.glob(root + "/research/*.md"))[:40]:
    s = open(f, encoding="utf-8", errors="replace").read()
    for i in range(0, min(len(s), 12000), 700): texts.append(s[i:i + 700])
# 2) code
import json as _j, os as _o, re as _r, argparse as _a
for m in (_j, _o, _r, _a):
    s = open(m.__file__, encoding="utf-8").read()
    for i in range(0, min(len(s), 20000), 900): texts.append(s[i:i + 900])
for f in sorted(glob.glob(root + "/src/*.cpp"))[:6]:
    s = open(f, encoding="utf-8", errors="replace").read()
    for i in range(0, min(len(s), 9000), 900): texts.append(s[i:i + 900])
# 3) random assigned code points of every category (drawn in runs, so casing / mark sequences occur)
cps = [c for c in range(0x110000) if not (0xD800 <= c < 0xE000) and unicodedata.category(chr(c)) not in ("Cn", "Cs", "Co", "Cc")]
for _ in range(1500):
    n = rnd.randint(1, 40); i0 = rnd.randrange(len(cps))
    texts.append("".join(chr(rnd.choice(cps)) if rnd.random() < 0.6 else chr(cps[min(i0 + k, len(cps) - 1)]) for k in range(n)))
# mixtures of letters in all cases / marks / digits / punctuation / spaces
pool = "aAbBzZ éÉ ǅ ʰ ª ǆ Ǉ ĳ ß ẞ ΐ ῼ дДёЁ 中文 日本語 한국어 ا ب 😀👍🏽 ‍ ́ ̈ ़ 123 ٣ ² ½ Ⅻ _-.,;:!?'\" \t\n\r    　 ( ) [ ] / \\ ' 's 'S 't 'RE 've 'M 'll 'D ſ ſ"
for _ in range(1500):
    texts.append("".join(rnd.choice(pool) for _ in range(rnd.randint(1, 30))))
# 4) special tokens and edge cases
sp = ["<|start|>", "<|end|>", "<|message|>", "<|channel|>", "<|return|>", "<|call|>", "<|constrain|>", "<|startoftext|>", "<|endoftext|>", "<|reserved_200000|>"]
texts += ["<|start|>user<|message|>Hello there<|end|><|start|>assistant<|channel|>final<|message|>Hi!<|return|>",
          "".join(rnd.choice(sp + ["x", " ", "\n", "я"]) for _ in range(200)), "<|start", "<|", "|>", "<|endoftext|>" * 30]
texts += ["347*86", "3.14 и 2,5", " 12  345 ", "x²+y³=z½", "١٢٣ ٤", "１２３", "Ⅻ век", "emoji 🙂👍🏽 ok", "tab\tnew\nline  \n\n", "don't You'D I'LL it's THEY'RE we've I'M",
          "ЁЖИК ёжик Привет, мир! Как дела?", "", "   ", "a" * 600, "ab" * 300, "中文 测试 123 日本語のテキスト 한국어 텍스트", "def f(x):\n    return x**2\n", "café naïve Ⅳ ﬁ ǅ ǆ",
          "\r\n\r\nx", "á ẹ̈ क़ क़", "HELLO World HeLLo wORLD camelCaseWord CONSTANT_NAME snake_case", "   \n   \n\n\n  x", "​‍﻿ zero-width", "I'mSTILL", "Σίσυφος ΣΊΣΥΦΟΣ", "ǈ ǋ ǲ DžDž"]
json.dump([{"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in texts], open(out, "w"), ensure_ascii=False)
print(len(texts), "strings")
