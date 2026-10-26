"""reference ids from the Python `tokenizers` library for tests/test_tokenizer.cjs"""
import json, sys
from tokenizers import Tokenizer
tj, out = sys.argv[1], sys.argv[2]
tok = Tokenizer.from_file(tj); texts = []
for i, line in enumerate(open("/workspace/qu/n-q/data/sft/sft_mix.jsonl")):
    if i % 200: continue
    for m in json.loads(line)["conversations"]: texts.append(m["content"][:500])
    if len(texts) >= 3000: break
texts += ["347*86", "3.14 и 2,5", " 12  345 ", "x²+y³=z½", "١٢٣ ٤", "１２３", "Ⅻ век", "emoji 🙂👍🏽 ok", "tab\tnew\nline  \n\n", "don't You'D I'LL",
          "ЁЖИК ёжик", "<|im_start|>user\nhi<|im_end|>\n", "", "   ", "a" * 60, "中文 测试 123", "def f(x):\n    return x**2\n", "café naïve Ⅳ ﬁ", "\r\n\r\nx"]
json.dump([{"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in texts], open(out, "w"), ensure_ascii=False)
print(len(texts), "strings")
