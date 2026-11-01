
## 2026-11-02T00:16:40.883826+03:00
Started baseline-20261102-001640. Baseline f16 versus existing q4 (no GPTQ). Source untouched.

Completed: [baseline-20261102-001640](baseline-20261102-001640/SUMMARY.md). Both selftests passed; see raw logs. No inference code changes.

Technical note: initial runner creation had a newline escaping SyntaxError before any model command ran. Output preserved in baseline-attempt1.log; fixed runner completed all commands successfully. Benchmark aggregation and both selftests subsequently inspected.

## 2026-11-02T00:37:08.605482+03:00
Started matrix-20261102-003657: 18 main configurations; separate RU/EN Wikipedia calibration and evaluation pages. Original f16 unchanged.

Matrix preparation initially hit HTTP429; snapshots cached and requests retried. Initial matrix runner then misparsed successful self-test output (expected selftest, actual self-test). Original results and logs retained; resume_matrix.py fixes parsing and reuses completed quantizations.

Matrix preparation initially hit HTTP429; snapshots cached and requests retried. Initial matrix runner then misparsed successful self-test output (expected selftest, actual self-test). Original results and logs retained; resume_matrix.py fixes parsing and reuses completed quantizations.

Finished matrix-20261102-003657: 18/18 variants completed successfully; see SUMMARY.md and failure logs.

Final audit passed: all 18 configurations, 18 selftests 5/5, equal 2040 primary and 510 pilot tokens per language, all weight hashes and source identity verified. See research/matrix-20261102-003657/REPORT.html.

Verified final numeric conclusions saved to research/matrix-20261102-003657/CONCLUSIONS.md. The definitive q4m GPTQ deltas are -0.33% RU / +0.75% EN; any differing preliminary chat readings are superseded by audited results.json. All calibration logs show 8176 tokens. Binary hashes unchanged.
