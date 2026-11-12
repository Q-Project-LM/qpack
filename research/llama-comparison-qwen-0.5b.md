# J1: qpack vs llama.cpp on Qwen2.5-0.5B (CPU speed, perplexity/bpb, GPU)

Machine: Xeon E5-2683 v4 VM (64 vCPU, shared host, load avg 6-10 during runs), Tesla P4. llama.cpp 3f8af15 (build/ CPU, build-cuda/), qpack integration branch (build/, buildc/). Raw data: results/J1-data/ (speed.log, poll-test.log, ppl-*.log, gpu-*.log, *_agg.md, agg_*.py).

## Models
| model | engine | size MiB | how made |
|---|---|---|---|
| q4m (calib-4096) | qpack | 352.4 | GPTQ act-order, calibrated on 4088 tokens RU/EN text |
| trim80k+q4head (F2, rANS-packed) | qpack | 232.2 | vocab trim 80k, q4 head, GPTQ, lossless rANS (decoded to RAM at load) |
| q8 | qpack | 479.9 | q8 |
| Q4_K_M | llama.cpp | 379.4 | pre-existing gguf from f16, no imatrix known to have been used |
| Q4_0 | llama.cpp | 335.8 | llama-quantize from f16, no imatrix |
| Q8_0 | llama.cpp | 506.5 | llama-quantize from f16 |
| f16 | llama.cpp | 948.1 | reference |

## A. CPU speed (taskset cores 32+, interleaved, 5 rounds; llama-bench -r 3 -p 128 / -n 128 -d 128; qpack in-process qspeed 1 warm-up + 3 reps, pp128 and tg128 at depth 128)
Mean +- std over rounds (qpack n=15 = 5 rounds x 3 reps pooled, llama n=5 round means). Raw min/max in speed_agg.md.
Decode tok/s:
| model | 4 thr | 8 thr | 16 thr |
|---|---|---|---|
| qpack q4m 352 MiB | 45.6 +- 1.8 | 67.4 +- 7.1 | 77.4 +- 9.6 |
| qpack trim80k 232 MiB | 54.4 +- 3.2 | 83.9 +- 4.1 | 90.9 +- 2.4 |
| qpack q8 480 MiB | 36.0 +- 1.2 | 53.2 +- 1.3 | 57.1 +- 0.8 |
| llama Q4_K_M 379 MiB | 37.3 +- 2.2 | 57.0 +- 8.0 | 34.8 +- 2.9 |
| llama Q4_0 336 MiB | 46.6 +- 1.6 | 68.6 +- 3.4 | 76.6 +- 4.0 |
| llama Q8_0 507 MiB | 34.2 +- 1.1 | 47.9 +- 1.2 | 36.4 +- 0.3 |

Prompt processing pp128 tok/s:
| model | 4 thr | 8 thr | 16 thr |
|---|---|---|---|
| qpack q4m | 121.4 +- 10.0 | 227.3 +- 19.6 | 405.0 +- 18.3 |
| qpack trim80k | 120.4 +- 12.0 | 228.8 +- 15.2 | 396.8 +- 10.9 |
| qpack q8 | 160.3 +- 10.9 | 295.5 +- 20.1 | 529.7 +- 29.7 |
| llama Q4_K_M | 138.4 +- 10.0 | 251.7 +- 14.8 | 443.2 +- 19.0 |
| llama Q4_0 | 206.2 +- 18.0 | 403.6 +- 18.6 | 707.6 +- 22.2 |
| llama Q8_0 | 161.4 +- 14.5 | 297.2 +- 15.8 | 512.5 +- 11.9 |

Harness caveat: qpack numbers come from my own in-process harness (qspeed.cpp), llama from llama-bench; both measure decode at depth 128 and pp128 from an empty cache, but they are not the same code, so differences of a few percent are not meaningful. The host is a noisy shared VM (std of 5-10% for several rows).

## A2. llama.cpp 16-thread decode collapse (Q4_K_M, Q8_0; Q4_0 does not collapse)
Reproducible (5/5 rounds). Tests (poll-test.log, Q8_0 tg at 16 thr): --poll 0/50/100 -> 35.9/36.0/37.0; pinned to cores 48-63 or 0-15, or unpinned -> 36.3-36.5; 12 threads -> ~49.5 (8 thr 47.9). So it is NOT --poll and NOT core placement; I could not find a cause (not shown to be a thread-sync artefact either; Q4_0 at 16 threads is fine). A wider -t 12/14 x poll x pin sweep over all three models was started by me but cancelled because it ran concurrently with perplexity jobs (contaminated; poll-partial-contaminated.log, not used). For a fair comparison use llama.cpp at its best thread count per model: Q4_K_M 57.0 (8 thr), Q8_0 ~49.5 (12 thr; 47.9 at 8), Q4_0 76.6 (16 thr). The qpack-vs-llama numbers above at 4 and 8 threads are unaffected by the collapse.

## B. Quality: perplexity and bits-per-byte (llama-perplexity protocol, -c 512, non-overlapping chunks, scored positions 256..510 of each chunk)
Texts: wikiEN = matrix-20261102-003657/data/eval-en.txt (24 KB, 9 chunks), wikiRU = eval-ru.txt (15 chunks), extEN = extended-holdout/en.txt (21 chunks), extRU = ru.txt (31 chunks). 
Validation of my qpack ppl tool (qppl): qpack f32 reference on wikiEN gives PPL 11.0345 vs llama.cpp f16 11.0348, with the same tokens, chunk count and scored count (2295) -> protocol and tokenization agree. For q8 the chunk counts equal llama.cpp's on all four texts. The f32 model was too slow (one token at a time) to run on all texts. qppl is single-pass (first two-pass version replaced; same maths).
Cell = PPL / bits-per-byte; bpb = nll_sum / ln2 / scored bytes (scored bytes taken from each qpack tokenizer's own pieces; llama.cpp rows use the qpack-q8 byte counts, valid because tokenizations are identical).
| model | wikiEN | wikiRU | extEN | extRU | all-4 bpb | vs f16 |
|---|---|---|---|---|---|---|
| llama.cpp f16 | 11.0348 / 0.6611 | 8.2823 / 0.5531 | 10.5600 / 0.6359 | 8.9747 / 0.6076 | 0.61056 | +0.00% |
| llama.cpp Q8_0 | 11.0573 / 0.6616 | 8.3059 / 0.5539 | 10.5880 / 0.6366 | 9.0117 / 0.6087 | 0.61143 | +0.14% |
| llama.cpp Q4_K_M | 11.3566 / 0.6690 | 8.6571 / 0.5647 | 10.8316 / 0.6428 | 9.2994 / 0.6174 | 0.61970 | +1.50% |
| llama.cpp Q4_0 | 12.1562 / 0.6877 | 9.7273 / 0.5952 | 11.8422 / 0.6668 | 10.2825 / 0.6453 | 0.64596 | +5.80% |
| qpack q8 | 11.0404 / 0.6612 | 8.3140 / 0.5541 | 10.5676 / 0.6361 | 8.9929 / 0.6082 | 0.61106 | +0.08% |
| qpack q4m (calib) | 11.4594 / 0.6715 | 8.5477 / 0.5614 | 11.0026 / 0.6470 | 9.3202 / 0.6181 | 0.62073 | +1.67% |
| qpack trim80k+q4head | 11.5916 / 0.6759 | 8.6347 / 0.5640 | 11.2384 / 0.6531 | 9.3910 / 0.6212 | 0.62473 | +2.32% |

cells = PPL(llama protocol, ctx512) / bits-per-byte over scored bytes
scored bytes (untrimmed tokenization): {'wikiEN': 12026, 'wikiRU': 21092, 'extEN': 28636, 'extRU': 41188}
trim80k wikiEN tokens/bytes scored (2295, 9) 12003 vs (2295, 9) 12026
trim80k wikiRU tokens/bytes scored (3825, 15) 21092 vs (3825, 15) 21092
trim80k extEN tokens/bytes scored (5355, 21) 28618 vs (5355, 21) 28636
trim80k extRU tokens/bytes scored (7905, 31) 41120 vs (7905, 31) 41188

trim80k scored bytes differ slightly from untrimmed (12003 vs 12026, 28618 vs 28636, 41120 vs 41188, wikiRU equal) because rare tokens are split differently, which is why bpb and not PPL is the comparison.

Size vs quality (all-4 bpb vs f16):
| model | MiB | bpb vs f16 |
|---|---|---|
| llama Q8_0 | 506.5 | +0.14% |
| qpack q8 | 479.9 | +0.08% |
| llama Q4_K_M | 379.4 | +1.50% |
| qpack q4m | 352.4 | +1.67% |
| llama Q4_0 | 335.8 | +5.80% |
| qpack trim80k | 232.2 | +2.32% |

Caveats: texts are small (stderr of a single PPL is 0.25-0.65 PPL, ~3-4%), so differences under ~0.3% bpb (q8 vs Q8_0, q4m vs Q4_K_M) are within noise and no ranking is claimed there. qpack q4m is GPTQ-calibrated on 4088 tokens of RU/EN text; the same RU/EN text domain as the eval texts (calibration text itself is disjoint from the eval texts per earlier protocol). The llama.cpp quants use no imatrix (Q4_K_M: no imatrix known; Q4_0/Q8_0: none). An imatrix run might improve llama Q4_K_M/Q4_0 and was not tried; so the llama.cpp 4-bit rows are likely not their best quality.

## C. GPU (Tesla P4, tg ~64 tokens from an empty cache, all layers on GPU, greedy)
qpack: QPACK_DEVICE=cuda buildc/cuda_dump --tps 3 (device-argmax greedy_run, 64 tokens); llama.cpp: llama-bench -ngl 99 -p 0 -n 64 -r 3 (build-cuda), default and GGML_CUDA_FORCE_MMQ=1 (MMQ). Interleaved rounds (qpack n=6, llama n=5 rounds, each the mean of 3), host CPU pinned to cores 0-15 while a CPU perplexity job ran on 48-63. SM clock 1113 MHz in every run, GPU temp 57-79 C (the run was paused when >=76 C; no throttling to 455 MHz seen). No other process on the GPU (checked before every run, none appeared).
| model | MiB | tok/s mean +- std | range |
|---|---|---|---|
| qpack trim80k+q4head | 232.2 | 257.8 +- 0.1 | 257.5-257.9 |
| qpack q4m | 352.4 | 226.7 +- 0.3 | 226.4-227.0 |
| qpack q8 | 479.9 | 190.4 +- 0.2 | 190.2-190.6 |
| llama Q4_0 (default / MMQ) | 335.8 | 164.8 +- 3.1 / 165.4 +- 2.7 | 160.9-167.9 |
| llama Q4_K_M (default / MMQ) | 379.4 | 150.3 +- 1.3 / 150.0 +- 2.3 | 148.2-151.9 |
| llama Q8_0 (default / MMQ) | 506.5 | 145.9 +- 2.1 / 147.5 +- 0.8 | 143.6-148.5 |
| llama f16 | 948.1 | 97.1 +- 2.5 | 92.6-98.4 |
Decode only; prompt processing on GPU not measured. Context depth ~0-64 only. qpack's CUDA path was tuned specifically on this card, which is the same GPU used here.

## Where qpack is better / worse (Qwen2.5-0.5B only, one CPU, one GPU)
Better:
- Decode on CPU at equal or smaller file: qpack q8 beats llama Q8_0 at 4/8/16 thr (36.0/53.2/57.1 vs 34.2/47.9/36.4; the 16-thr gap is inflated by the llama collapse; at 8 thr +11%). qpack q4m beats Q4_K_M at 4/8/16 (45.6/67.4/77.4 vs 37.3/57.0/34.8; vs Q4_K_M's best 57.0 that is +18% at 8 thr, +36% at 16) and 352 vs 379 MiB. qpack q4m is on par with llama Q4_0 (77.4 vs 76.6 at 16 thr) with Q4_0 being 5.8% worse bpb.
- trim80k (232 MiB) gives the best decode (90.9 tok/s at 16 thr, +19% over llama's best 4-bit, Q4_0) and the smallest file, at +2.3% bpb, which is between Q4_K_M (+1.5%) and Q4_0 (+5.8%), at 61% of Q4_0's size.
- GPU decode: +51% (q4m 226.7 vs Q4_K_M 150.3), +37% over the fastest llama.cpp quant (Q4_0 164.8), +74% for trim80k over Q4_0 default; q8 190 vs Q8_0 146 (+30%).
Worse / not better:
- Prompt processing on CPU: llama.cpp is faster for the 4-bit formats (Q4_K_M 443 vs q4m 405 at 16 thr, -9% for qpack; Q4_0 708, qpack is 43% slower than Q4_0). qpack q8 is on par with Q8_0 (530 vs 513, within noise).
- Quality at ~4 bits: qpack q4m is not better than Q4_K_M (+1.67% vs +1.50% bpb, within noise), despite being GPTQ-calibrated with the eval-domain language mix and llama.cpp being uncalibrated; qpack's q4m advantage is speed and ~7% size, not quality. trim80k pays +2.3% bpb for its size.
- Not tested: other models, imatrix llama.cpp quants, larger contexts, other CPUs/GPUs, prompt speed on GPU. qpack single-file only supports few architectures.
- llama.cpp 16-thread collapse for K-quant/Q8_0 is unexplained and may be specific to this VM.
