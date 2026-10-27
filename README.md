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

Xeon E5-2683 v4 (AVX2, no AVX-512, a shared 64-vCPU VM), `qpack-bench -p 64 -n 128` / `llama-bench -p 64 -n 128`,
tok/s at 1 / 8 / 16 threads:

| model | file | MiB | prompt tok/s | generation tok/s |
|---|---|---|---|---|
| N-Q-26M-Chat | q8 | 46.5 | 183 / 958 / 1513 | 151 / 511 / 551 |
| N-Q-26M-Chat | q4 (GPTQ) | 29.5 | 406 / 1563 / 2152 | 355 / 667 / 572 |
| Qwen2.5-0.5B | qpack q8 | 480 | 54 / 321 / 584 | 17.3 / 63.4 / 65.8 |
| Qwen2.5-0.5B | qpack q4 | 329 | 38 / 233 / 397 | 19.6 / 78.6 / 95.7 |
| Qwen2.5-0.5B | qpack q4m | 352 | 40 / 247 / 450 | 19.7 / 79.4 / 96.1 |
| Qwen2.5-0.5B | qpack q4 + q4 head (`--q4-head`) | 272 | 36 / 239 / 431 | 18.7 / 85.2 / 105.1 |
| Qwen2.5-0.5B | llama.cpp Q8_0 | 501 | 52 / 303 / 534 | 13.2 / 53.5 / 47.7 |
| Qwen2.5-0.5B | llama.cpp Q4_0 | 330 | 71 / 415 / 726 | 19.8 / 77.5 / 82.9 |
| Qwen2.5-0.5B | llama.cpp Q4_K_M | 374 | 43 / 251 / 455 | 15.9 / 65.5 / 72.9 |

Generation is faster than llama.cpp at equal size: q8 +23-38%, q4 vs Q4_0 +1-15%, q4m vs Q4_K_M +21-32%. What made
the decode step faster: per-thread completion flags in the pool instead of one contended counter, one pass for q/k/v
and for gate/up (one activation quantisation, one barrier), q4 group scales converted per row with F16C instead of per
group, AVX2 `exp` for SiLU / sigmoid / softmax, a top-k sampler that scans the 150k logits with SIMD compares instead
of sorting them, and decode attention whose query heads of a KV group share each converted K / V row. For a model as
small as N-Q-26M, 8 threads beat 16 (the step is ~1.5 ms, synchronisation dominates). Prompt processing still trails
Q4_0 (llama.cpp multiplies int8 activations; qpack keeps int16 activations for accuracy).

### Speculative decoding

```bash
build/qpack-cli -m model.qpack --lookup 15                          # prompt lookup: draft = continuation of the last n-gram seen earlier
build/qpack-cli -m model.qpack --draft small.qpack --draft-k 4      # draft model with the same tokenizer
build/qpack-server -m model.qpack --lookup 15                       # same flags; usage reports draft_tokens / draft_accepted
build/qpack-bench -m model.qpack -t 16 --spec prompt.txt            # adds plain vs lookup chat columns
```

Drafted tokens are checked in one batched pass of the model; the draft length adapts (doubled after a fully accepted
pass, cut after a rejection). Verification and the replay after a rejection run with kernels that are bit-identical to
the single-token path, so greedy output is exactly the plain output (checked: `identical yes` in `bench/spec_bench`),
and sampling uses the standard acceptance rule (accept with p(token), otherwise draw from p without it), which keeps
the distribution. Qwen2.5-0.5B q4, 16 threads, greedy: rewriting a function 84 → 156 tok/s (×1.87), a free-form
answer 88 → 97 tok/s (×1.09). It pays off for models that are bound by memory bandwidth (hundreds of MB and up);
N-Q-26M is so small that the extra work outweighs the saved weight reads, so it is off by default.

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
