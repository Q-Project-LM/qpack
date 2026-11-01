# Quantization matrix

Source SHA256: 1e22aafe3679970cdd136cbab1b99d60a3430f762d50a6ca69019391577d4c0c

18 configurations: f16, q8, and q4/q4m x plain/calib/GPTQ/GPTQ-act-order x head q8/q4. Embedding follows tool defaults. No algorithm changes.

Primary PPL: separate Wikipedia articles for calibration/evaluation, context256, up to8 chunks, threads8. Calibration cap8192, balanced RU/EN article order; actual counts in logs. Pilot PPL uses saved baseline prose, 2 chunks. Corpora are preliminary and narrow; not comparable with README scores.

Speed: synthetic tokens, prompt64/decode128, 1/8/16 threads, separate best-of3 per phase. Selftest checks recalculated quantized references, not original quality.

| Variant | MiB | PPL RU | PPL EN | Decode 1 | Decode 8 | Decode 16 | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| f16 | - | - | - | - | - | - | FAILED: Selftest did not pass 5/5: model.qpack: nq4, vocab 8256, ctx 4096, loaded in 0.01 s, self-test 5/5 (max |dlogit| 6.02e-03), threads 8  |
| q8 | - | - | - | - | - | - | FAILED: Selftest did not pass 5/5: research/matrix-20261102-003657/artifacts/q8.qpack: nq4, vocab 8256, ctx 4096, loaded in 0.01 s, self-test 5/5 (max |dlogit| 6.29e-04 |
| q4-plain | - | - | - | - | - | - | FAILED: Selftest did not pass 5/5: research/matrix-20261102-003657/artifacts/q4-plain.qpack: nq4, vocab 8256, ctx 4096, loaded in 0.01 s, self-test 5/5 (max |dlogit| 5. |
| q4-plain-head4 | - | - | - | - | - | - | FAILED: Selftest did not pass 5/5: research/matrix-20261102-003657/artifacts/q4-plain-head4.qpack: nq4, vocab 8256, ctx 4096, loaded in 0.01 s, self-test 5/5 (max |dlog |
| q4-calib | - | - | - | - | - | - | FAILED: Selftest did not pass 5/5: research/matrix-20261102-003657/artifacts/q4-calib.qpack: nq4, vocab 8256, ctx 4096, loaded in 0.01 s, self-test 5/5 (max |dlogit| 9. |
| q4-calib-head4 | - | - | - | - | - | - | FAILED: Selftest did not pass 5/5: research/matrix-20261102-003657/artifacts/q4-calib-head4.qpack: nq4, vocab 8256, ctx 4096, loaded in 0.01 s, self-test 5/5 (max |dlog |

See results.json for pilot PPL, all prompt speeds, token counts, sizes and model hashes; commands.json and logs/ preserve all commands and errors. sources.json records article URLs, revisions and attribution.