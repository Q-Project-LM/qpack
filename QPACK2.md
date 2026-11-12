# QPACK2 — int8 single-file container for N-Q models

QPACK2 keeps the QPACK1 container byte for byte (header, table of contents, blobs aligned to 64 bytes, mmap-friendly)
and adds one tensor kind, `q8`. A reader that understands QPACK1 needs only the new kind and the new magic.

```
magic   "QPACK2\0\0"   (QPACK1: "QPACK1\0\0")     8 bytes
count   uint32                                       number of entries
pad     uint32
entry   name[56] (NUL padded), off u64, len u64      72 bytes each
blobs   each aligned to 64 bytes
```

| kind | blobs | value |
|---|---|---|
| `q8` (new) | `wNNNN.q8` int8 `[rows, cols]` row-major + `wNNNN.scale` float32 `[rows]` | `W[r, c] = q[r, c] * scale[r]`, `q ∈ [-127, 127]`, `scale[r] = max_c |W[r, c]| / 127` |
| `q4` | `wNNNN.q4` uint8 `[rows, cols / 2]` + `wNNNN.scale` fp16 `[rows, cols / 32]`; manifest `"group": 32` | groups of 32 columns, 16 bytes each: byte `j` holds column `j` (low nibble) and column `j + 16` (high nibble) as `q + 8`, `q ∈ [-8, 7]`; `W[r, 32g + c] = q * scale[r, g]`. Requires `cols % 32 == 0`. Written by `qpack-quantize` |
| `f16` / `f32` | `wNNNN.f16` / `.f32` | dense tensors (1-D tensors — norms, decay parameters — are always f32) |
| `i64` | `wNNNN.i64` | integer buffers (n-gram hash table sizes, offsets, multipliers): must be exact |
| `tern5`, `bits` | as in QPACK1 | ternary weights, frozen fingerprints (Q-architecture checkpoints) |

`manifest.json` lists every tensor (`name`, `shape`, `dtype`, `kind`, blob names; `q8` entries also carry `scale`) and
`"format": "QPACK2", "quant": "q8"`. Which tensors are `q8`: every 2-D floating matrix with at least 4096 entries —
attention / Gated DeltaNet / FFN projections, Gated Residual matrices, the embedding, the n-gram table and the output
head. `config.json`, `generation_config.json`, `tokenizer.json` and `selftest.json` are stored as in QPACK1. The
selftest of a QPACK2 file is computed on the dequantised model, so `python qpack.py --selftest model.q8.qpack` checks
the runtime, not the rounding.

## QPACK3: entropy-coded blobs (lossless, decoded at load)

`qpack-pack in.qpack out.qpack` re-writes a QPACK2 file with magic `"QPACK3\0\0"` and `"format": "QPACK3"`; layout, TOC, tensor
kinds and blob names are unchanged. The manifest gains `"encoded": [blob names...]`: those blobs hold a static rANS stream
instead of the raw bytes, and the reader (`Container`, `src/container.cpp`, `include/qpack/rans.h`) decodes them once at
load into 64-byte aligned heap buffers, so `blob()`, the tensors and every kernel see exactly the QPACK2 bytes
(`qpack-pack --unpack` restores a byte-identical QPACK2 file). Coded: `.q4` (nibbles), `.q8` (bytes), fp16 `.scale` of q4
(two byte planes); blobs that do not shrink are left raw. Blob layout (little endian):

```
u8 codec (1 bytes, 2 nibbles) | u8 planes | u32 raw_len | per plane: u32 comp_len | u16 freq[nsym] | u32 state[4] | bytes
```

Codec 1 splits the raw bytes into `planes` interleaved planes (byte `i` -> plane `i % planes`) and codes each as a 256-ary
source; codec 2 codes each byte as two 16-ary symbols (low nibble first). rANS: 12-bit frequencies summing to 4096, 32-bit
states renormalised bytewise to `[2^23, 2^31)`, 4 interleaved states (symbol `i` uses state `i & 3`), encoder runs backwards
and the stream is stored in decode order; the 4 final states are the initial decoder states. Cost: decode of a 0.5B model is
~1 s on 8 cores and the weights then live in anonymous memory instead of the page cache (no lazy mmap). The Python reader
(`python/qpack/container.py`) refuses encoded blobs; unpack with `qpack-pack --unpack` first.

## Why

* **Half the size.** N-Q-26M: 89.4 MiB (QPACK1, fp16 matrices) → 46.5 MiB.
* **Faster.** A decode step is a chain of matrix–vector products and is memory-bound; int8 rows read 4× fewer bytes
  than float32. With the WebAssembly SIMD kernel (`nq-runtime/nq_kernel.c`: `matvec_q8`, `matvec_f32`, `row_q8`) the
  browser runtime goes from 16.5 tok/s (plain JS, QPACK1) to 51.6 (WASM, QPACK1) and 76.7 tok/s (WASM, QPACK2) in node;
  loading takes 126 ms instead of 0.8 s because nothing has to be converted from fp16.
* **Lazy tables.** Embedding and n-gram rows are dequantised on lookup (8 n-gram rows and 1 embedding row per token)
  instead of expanding 16.8M + 4.2M parameters to float32 at load.

## Quality (N-Q-26M-Chat, f32 vs q8)

| check | f32 | q8 |
|---|---|---|
| held-out loss EN / RU-edu / RU-wiki | 2.4865 / 3.0205 / 2.4697 | 2.4868 / 3.0206 / 2.4698 |
| tool-call probe | 15/16 | 15/16 |
| greedy 64-token continuations identical | — | 6/8 (the other two diverge after 40–56 tokens) |
| max \|Δlogit\| on a test prompt | — | 0.19–0.23 |

## Use

```python
from nq.model import qpack                       # or qpack.py next to modeling_nq4.py in a release folder
blobs = qpack.pack_model(model, tokenizer_json=..., quant="q8"); qpack.write_container("model.q8.qpack", blobs)
state, cfg = qpack.load_state("model.q8.qpack")  # dequantised float32 state_dict
model = NQ4ForCausalLM.from_qpack("model.q8.qpack")
```

`python -m nq.scripts.export_hf --ckpt runs/<run>/final.pt --out <folder> --qpack2 1` writes `model.q8.qpack` next to
`model.qpack`. In the browser: `nq-runtime/nq_model.js` (`loadNQ`, `NQ4JS.attachKernel`) reads both formats.
