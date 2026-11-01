# Baseline f16 vs q4

Date (Moscow): 2026-11-02T00:16:40.883826+03:00
Branch: improve-quantization; commit: e92b52f26f90dd5f00fa6ac2e98c373fce1bfa1a

Preliminary smoke test on saved assistant-authored prose. No calibration or GPTQ. Do not treat this as general quality or compare directly with README corpus scores.

Perplexity: ctx 256, up to 2 chunks per language, 8 threads. Speed: prompt 64, generation 128, threads 1 and 8, 3 repetitions. See raw benchmark logs for aggregation.

| Model | MiB | PPL RU | PPL EN |
|---|---:|---:|---:|
| f16 | 89.39 | 16.1914 | 11.8907 |
| q4 | 29.45 | 16.4853 | 12.0433 |
RU: 510 tokens per model; q4 PPL change +1.82%.
EN: 510 tokens per model; q4 PPL change +1.28%.

## Speed f16
| model | MiB | arch | threads | pp64 tok/s | tg128 tok/s |
|---|---|---|---|---|---|
| model.qpack | 89.4 | nq4 | 1 | 348.9 | 145.6 |
| model.qpack | 89.4 | nq4 | 8 | 1578.6 | 447.8 |


## Speed q4
| model | MiB | arch | threads | pp64 tok/s | tg128 tok/s |
|---|---|---|---|---|---|
| model.q4.qpack | 29.5 | nq4 | 1 | 401.4 | 323.0 |
| model.q4.qpack | 29.5 | nq4 | 8 | 1570.5 | 653.3 |


commands.json records all commands and exit codes. logs/ holds complete output. Source model not overwritten.

Benchmark aggregation verified in source: separate best prompt/decode rates over 3 repetitions. Synthetic token workload, not a chat speed measurement. Quantized-file selftest references are recalculated by the quantizer: passing selftest verifies kernels, not preservation of original model quality.
