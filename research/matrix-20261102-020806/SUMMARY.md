# Quantization matrix

Source SHA256: 1e22aafe3679970cdd136cbab1b99d60a3430f762d50a6ca69019391577d4c0c

18 configurations: f16, q8, and q4/q4m x plain/calib/GPTQ/GPTQ-act-order x head q8/q4. Embedding follows tool defaults. No algorithm changes.

Primary PPL: separate Wikipedia articles for calibration/evaluation, context256, up to8 chunks, threads8. Calibration cap8192, balanced RU/EN article order; actual counts in logs. Pilot PPL uses saved baseline prose, 2 chunks. Corpora are preliminary and narrow; not comparable with README scores.

Speed: synthetic tokens, prompt64/decode128, 1/8/16 threads, separate best-of3 per phase. Selftest checks recalculated quantized references, not original quality.

| Variant | MiB | PPL RU | PPL EN | Decode 1 | Decode 8 | Decode 16 | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| f16 | 89.39 | 23.4519 | 11.8276 | 148.0 | 454.3 | 416.3 | 5/5 |
| q8 | 46.54 | 23.4483 | 11.8381 | 162.9 | 505.0 | 517.4 | 5/5 |
| q4-plain | 29.45 | 23.9525 | 12.2081 | 338.7 | 656.0 | 625.8 | 5/5 |
| q4-plain-head4 | 25.86 | 24.1651 | 12.3774 | 327.7 | 654.5 | 623.6 | 5/5 |
| q4-calib | 29.45 | 23.7861 | 12.1435 | 323.1 | 600.1 | 543.4 | 5/5 |
| q4-calib-head4 | 25.86 | 24.0423 | 12.3193 | 360.4 | 636.2 | 537.1 | 5/5 |
| q4-gptq | 29.45 | 23.4445 | 11.9639 | 328.6 | 628.4 | 547.2 | 5/5 |
| q4-gptq-head4 | 25.86 | 23.6976 | 12.1310 | 346.7 | 636.7 | 552.4 | 5/5 |
| q4-gptq-act | 29.45 | 23.5392 | 11.8974 | 285.4 | 615.6 | 548.0 | 5/5 |
| q4-gptq-act-head4 | 25.86 | 23.7738 | 12.0667 | 328.7 | 619.9 | 552.6 | 5/5 |
| q4m-plain | 30.12 | 23.7679 | 12.1341 | 319.3 | 644.1 | 633.4 | 5/5 |
| q4m-plain-head4 | 26.53 | 23.9790 | 12.3057 | 292.4 | 632.7 | 656.6 | 5/5 |
| q4m-calib | 30.12 | 23.6534 | 12.0704 | 322.4 | 625.5 | 548.7 | 5/5 |
| q4m-calib-head4 | 26.53 | 23.9012 | 12.2420 | 282.9 | 660.9 | 639.7 | 5/5 |
| q4m-gptq | 30.12 | 23.3751 | 11.9168 | 326.7 | 643.1 | 567.6 | 5/5 |
| q4m-gptq-head4 | 26.53 | 23.6242 | 12.0855 | 270.9 | 646.6 | 612.4 | 5/5 |
| q4m-gptq-act | 30.12 | 23.5224 | 11.8951 | 313.6 | 617.0 | 544.7 | 5/5 |
| q4m-gptq-act-head4 | 26.53 | 23.7571 | 12.0652 | 286.9 | 647.9 | 617.0 | 5/5 |

See results.json for pilot PPL, all prompt speeds, token counts, sizes and model hashes; commands.json and logs/ preserve all commands and errors. sources.json records article URLs, revisions and attribution.