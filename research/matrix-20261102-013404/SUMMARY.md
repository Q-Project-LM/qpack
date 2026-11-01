# Quantization matrix

Source SHA256: 1e22aafe3679970cdd136cbab1b99d60a3430f762d50a6ca69019391577d4c0c

18 configurations: f16, q8, and q4/q4m x plain/calib/GPTQ/GPTQ-act-order x head q8/q4. Embedding follows tool defaults. No algorithm changes.

Primary PPL: separate Wikipedia articles for calibration/evaluation, context256, up to8 chunks, threads8. Calibration cap8192, balanced RU/EN article order; actual counts in logs. Pilot PPL uses saved baseline prose, 2 chunks. Corpora are preliminary and narrow; not comparable with README scores.

Speed: synthetic tokens, prompt64/decode128, 1/8/16 threads, separate best-of3 per phase. Selftest checks recalculated quantized references, not original quality.

| Variant | MiB | PPL RU | PPL EN | Decode 1 | Decode 8 | Decode 16 | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| f16 | 89.39 | 23.4519 | 11.8276 | 130.0 | 473.3 | 410.8 | 5/5 |
| q8 | 46.54 | 23.4483 | 11.8381 | 132.6 | 495.7 | 529.4 | 5/5 |
| q4-plain | 29.45 | 23.9525 | 12.2081 | 342.8 | 655.3 | 641.9 | 5/5 |
| q4-plain-head4 | 25.86 | 24.1651 | 12.3774 | 286.3 | 631.7 | 619.9 | 5/5 |
| q4-calib | 29.45 | 23.7861 | 12.1435 | 294.9 | 613.1 | 538.8 | 5/5 |
| q4-calib-head4 | 25.86 | 24.0423 | 12.3193 | 333.9 | 623.5 | 531.4 | 5/5 |
| q4-gptq | 29.45 | 23.4445 | 11.9639 | 355.5 | 641.0 | 537.7 | 5/5 |
| q4-gptq-head4 | 25.86 | 23.6976 | 12.1310 | 348.9 | 613.9 | 534.6 | 5/5 |
| q4-gptq-act | 29.45 | 23.5392 | 11.8974 | 301.0 | 626.2 | 570.3 | 5/5 |
| q4-gptq-act-head4 | 25.86 | 23.7738 | 12.0667 | 335.9 | 628.7 | 562.8 | 5/5 |
| q4m-plain | 30.12 | 23.7679 | 12.1341 | 307.1 | 633.3 | 639.6 | 5/5 |
| q4m-plain-head4 | 26.53 | 23.9790 | 12.3057 | 290.0 | 627.1 | 629.2 | 5/5 |
| q4m-calib | 30.12 | 23.6534 | 12.0704 | 311.8 | 625.7 | 561.6 | 5/5 |
| q4m-calib-head4 | 26.53 | 23.9012 | 12.2420 | 284.4 | 652.8 | 622.8 | 5/5 |
| q4m-gptq | 30.12 | 23.3751 | 11.9168 | 303.5 | 603.2 | 541.4 | 5/5 |
| q4m-gptq-head4 | 26.53 | 23.6242 | 12.0855 | 267.0 | 655.3 | 610.5 | 5/5 |
| q4m-gptq-act | 30.12 | 23.5224 | 11.8951 | 320.5 | 623.5 | 561.3 | 5/5 |
| q4m-gptq-act-head4 | 26.53 | 23.7571 | 12.0652 | 282.6 | 645.0 | 629.7 | 5/5 |

See results.json for pilot PPL, all prompt speeds, token counts, sizes and model hashes; commands.json and logs/ preserve all commands and errors. sources.json records article URLs, revisions and attribution.