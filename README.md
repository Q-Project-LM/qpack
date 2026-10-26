# qpack

Single-file model format and a small C++ inference engine in the spirit of llama.cpp: `qpack-cli`, `qpack-server`
(OpenAI-compatible), `qpack-bench`, `qpack-perplexity`. Plus a WebAssembly runtime for the browser.

* **Format.** QPACK1 / QPACK2: one mmap-able file with a table of contents, tensors (f32 / f16 / int8 `q8` rows with a
  float32 scale per row / i64), `config.json`, `tokenizer.json` and a built-in `selftest.json` (token ids → reference
  logits), see [QPACK2.md](QPACK2.md).
* **Models.** N-Q (Gated DeltaNet + gated attention, Gated Residual, n-gram PLE, exact `<CALC>` circuits) and the
  Llama family (Llama, Qwen2, Mistral, SmolLM: RMSNorm, RoPE, GQA, optional q/k/v bias, SwiGLU, tied or untied head).
* **Engine.** C++17, no dependencies: AVX2/FMA/F16C kernels (int8 × int16 `madd` for `q8`), a spinning thread pool,
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

## Speed

Xeon E5-2683 v4 (AVX2, no AVX-512), q8 weights, 128-token prompt, 128 generated tokens:

| model | file | threads | prompt tok/s | generation tok/s |
|---|---|---|---|---|
| N-Q-26M-Chat | 46.5 MiB | 1 | 160 | 144 |
| N-Q-26M-Chat | 46.5 MiB | 8 | 389 | 361 |
| Qwen2.5-0.5B-Instruct | 480 MiB | 1 | 21 | 15 |
| Qwen2.5-0.5B-Instruct | 480 MiB | 8 | 73 | 55 |

Same machine, llama.cpp (CPU, Q8_0, 501 MiB), `llama-bench -p 64 -n 64`: generation 11.3 / 50.8 / 46.0 tok/s at
1 / 8 / 16 threads (qpack: 15.2 / 55.3 / 58.6); prompt 43 / 309 / 525 tok/s — llama.cpp evaluates the prompt in
batches (matrix × matrix), qpack still token by token (21 / 73 / 73).

## Tests

```bash
build/test_tokenizer models/Qwen2.5-0.5B-Instruct/tokenizer.json tokref_qwen.json   # ids vs the Python `tokenizers` library
build/test_circuits circ_ref.json                                                    # circuits vs the Python evaluate()
build/qpack-cli -m model.q8.qpack --selftest                                         # logits vs PyTorch
```

`tests/make_tok_ref.py` makes the tokenizer references. Current results: tokenizer 3022/3022 strings identical for
Qwen2.5 and N-Q, circuits 40034/40034, self-test 5/5 for every N-Q release file and for Qwen2.5-0.5B (q8 and f32).

## Browser

`runtime/` is the WebAssembly runtime used on the site: `nq_core.c` (the whole N-Q step in WASM SIMD),
`llama_core.c`, `hf_tokenizer.js`, `qpack.js` (container reader), `nq_worker.js`.
