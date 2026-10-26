# qpack

Single-file model format and a small C++ inference engine in the spirit of llama.cpp: `qpack-cli`, `qpack-server`
(OpenAI-compatible), `qpack-bench`, `qpack-perplexity`. Plus a WebAssembly runtime for the browser.

* **Format.** QPACK1 / QPACK2: one mmap-able file with a table of contents, tensors (f32 / f16 / int8 `q8` rows with a
  float32 scale per row / i64), `config.json`, `tokenizer.json` and a built-in `selftest.json` (token ids → reference
  logits), see [QPACK2.md](QPACK2.md).
* **Models.** N-Q (Gated DeltaNet + gated attention, Gated Residual, n-gram PLE, exact `<CALC>` circuits) and the
  Llama family (Llama, Qwen2, Mistral, SmolLM: RMSNorm, RoPE, GQA, optional q/k/v bias, SwiGLU, tied or untied head).
* **Engine.** C++17, no dependencies: AVX2/FMA/F16C kernels (int8 / int4 weights × int16 activations via `madd`,
  batched matrix × matrix for prompts), fp16 KV cache, a spinning thread pool,
  byte-level BPE tokenizer read from `tokenizer.json` (regex pre-tokenizers, NFC, added tokens), sampler (temperature,
  top-k, top-p, min-p, repetition penalty), chat formats (ChatML, N-Q), KV-cache / state reuse across requests.

## Build

```bash
cmake -S . -B build && cmake --build build -j
```

`-DQPACK_NATIVE=OFF` builds without `-march=native` (the kernels still need AVX2 + FMA + F16C).

## Run

```bash
# chat in the terminal (empty line or /exit to quit, /reset to clear the history)
build/qpack-cli -m N-Q-26M-Chat/model.q8.qpack
# one question / raw completion
build/qpack-cli -m qwen2.5-0.5b-instruct.q8.qpack -q "Сколько будет 17*23?" --temp 0
build/qpack-cli -m qwen2.5-0.5b-instruct.q8.qpack -p "The capital of France is" -n 40
# check that the runtime reproduces the reference logits stored in the file
build/qpack-cli -m model.q8.qpack --selftest
# OpenAI-compatible server with a minimal chat page at http://127.0.0.1:8080/
build/qpack-server -m N-Q-26M-Chat/model.q8.qpack --port 8080
# speed and perplexity
build/qpack-bench -m model.q8.qpack -t 1,8,16
build/qpack-perplexity -m model.q8.qpack -f text.txt -c 512
```

Server endpoints: `POST /v1/chat/completions` (`messages`, `max_tokens`, `temperature`, `top_p`, `top_k`, `min_p`,
`repeat_penalty`, `seed`, `stop`, `stream`), `POST /v1/completions`, `GET /v1/models`, `POST /tokenize`,
`POST /detokenize`, `GET /health`. For N-Q, `"think": true` opens the answer with `<THINK>`; the reasoning comes back as
`reasoning_content`, circuit results (`<CALC>347*86<EQ>29842<ECALC>`) are shown as `347*86 = 29842`, and `usage`
reports `circuit_calls`.

## Convert a model

```bash
pip install torch transformers numpy
PYTHONPATH=python python -m qpack.convert_llama models/Qwen2.5-0.5B-Instruct qwen2.5-0.5b-instruct.q8.qpack --quant q8
```

N-Q checkpoints are packed by `nq/model/qpack.py` in the N-Q repository (`--qpack2` in `export_hf.py`).

## Compress (q4)

```bash
# 4.5 bits/weight: 4-bit groups of 32 + fp16 scale; embedding / output head stay q8
build/qpack-quantize model.f32.qpack model.q4.qpack
# better: GPTQ with a calibration text (any representative RU/EN text, a few thousand tokens)
build/qpack-quantize model.f32.qpack model.q4.qpack --calib calib.txt --calib-tokens 8192 --gptq
# q4m: as q4, plus q8 for the value projections and part of the FFN down projections (like llama.cpp Q4_K_M)
build/qpack-quantize model.f32.qpack model.q4m.qpack --type q4m --calib calib.txt --calib-tokens 8192 --gptq
```

The source can be f32, f16 or q8 (best: the unquantised file). Group scales are chosen by minimum (activation-weighted)
squared error; `--gptq` quantises columns left to right and pushes each rounding error onto the remaining columns
through the inverse of H = Σ x xᵀ collected on the calibration text (Frantar et al., 2022). The self-test of the output
file is recomputed on its dequantised weights, so `--selftest` checks the q4 kernels (5/5 on every file below).

Perplexity with `qpack-perplexity` (Russian Wikipedia / English educational web text, held out from the calibration
text; N-Q: 60 × 512 tokens, Qwen: 30 × 256 tokens):

| model | file | MiB | ppl RU | ppl EN | vs unquantised |
|---|---|---|---|---|---|
| N-Q-26M-Chat | fp16 (QPACK1) | 89.4 | 11.343 | 12.657 | — |
| N-Q-26M-Chat | q8 | 46.5 | 11.343 | 12.665 | +0.0% / +0.1% |
| N-Q-26M-Chat | q4, round-to-nearest | 29.5 | 11.655 | 12.947 | +2.8% / +2.3% |
| N-Q-26M-Chat | q4, GPTQ | 29.5 | 11.540 | 12.871 | +1.7% / +1.7% |
| Qwen2.5-0.5B-Instruct | f32 | 1891 | 9.824 | 19.258 | — |
| Qwen2.5-0.5B-Instruct | q8 | 480 | 9.881 | 19.226 | +0.6% / −0.2% |
| Qwen2.5-0.5B-Instruct | q4, round-to-nearest | 329 | 11.739 | 21.769 | +19.5% / +13.0% |
| Qwen2.5-0.5B-Instruct | q4, GPTQ | 329 | 10.506 | 20.901 | +6.9% / +8.5% |
| Qwen2.5-0.5B-Instruct | q4m, GPTQ | 352 | 10.202 | 20.296 | +3.8% / +5.4% |

For reference, llama.cpp on the same texts (its own `llama-perplexity` protocol, so only the relative change from f16
compares): Q8_0 +0.0% / +0.2%, Q4_0 (330 MiB) +14.4% / +14.3%, Q4_K_M (374 MiB) +2.5% / +3.1%.

## Speed

Xeon E5-2683 v4 (AVX2, no AVX-512); prompt = `pp` (64 tokens for Qwen, 128 for N-Q), generation = `tg`, tok/s at
1 / 8 / 16 threads (`qpack-bench`, `llama-bench`):

| model | file | MiB | prompt tok/s | generation tok/s |
|---|---|---|---|---|
| N-Q-26M-Chat | q8 | 46.5 | 177 / 823 / 1212 | 151 / 426 / 411 |
| N-Q-26M-Chat | q4 | 29.5 | 363 / 1192 / 1492 | 305 / 503 / 428 |
| Qwen2.5-0.5B | qpack q8 | 480 | 46 / 242 / 454 | 14.0 / 55.0 / 56.7 |
| Qwen2.5-0.5B | qpack q4 | 329 | 34 / 207 / 367 | 18.7 / 73.5 / 81.8 |
| Qwen2.5-0.5B | qpack q4m | 352 | 37 / 221 / 385 | 19.3 / 69.4 / 77.8 |
| Qwen2.5-0.5B | llama.cpp Q8_0 | 501 | 43 / 309 / 525 | 11.3 / 50.8 / 46.0 |
| Qwen2.5-0.5B | llama.cpp Q4_0 | 330 | 68 / 421 / 709 | 19.4 / 79.1 / 84.3 |
| Qwen2.5-0.5B | llama.cpp Q4_K_M | 374 | 39 / 263 / 437 | 14.4 / 63.1 / 67.9 |

Generation is on par with or faster than llama.cpp at equal bits; prompt processing still trails Q4_0 (llama.cpp
multiplies int8 activations, qpack keeps int16 activations for accuracy).

## Tests

```bash
build/test_tokenizer models/Qwen2.5-0.5B-Instruct/tokenizer.json tokref_qwen.json   # ids vs the Python `tokenizers` library
build/test_circuits circ_ref.json                                                    # circuits vs the Python evaluate()
build/qpack-cli -m model.q8.qpack --selftest                                         # logits vs PyTorch
```

`tests/make_tok_ref.py` makes the tokenizer references. Current results: tokenizer 3022/3022 strings identical for
Qwen2.5 and N-Q, circuits 40034/40034, self-test 5/5 for every N-Q release file and for Qwen2.5-0.5B (f32, q8, q4,
q4m); batched prompt evaluation equals token-by-token decoding (max |Δlogit| 3e-6 f16, ~1e-3 q8/q4).

## Browser

`runtime/` is the WebAssembly runtime used on the site: `nq_core.c` (the whole N-Q step in WASM SIMD, f32 / q8 / q4
weights; N-Q-26M q4: 138 tok/s in node, logits within 4e-3 of the C++ engine),
`llama_core.c`, `hf_tokenizer.js`, `qpack.js` (container reader), `nq_worker.js`.
