#!/bin/bash
# usage: dump.sh <cudadump binary>  (env passes through)
D=/workspace/qpack/research/matrix-20261102-003657/data
cd /workspace/qpack
QPACK_DEVICE=cuda $1 -m ~/autoresearch-runs/qpack-tps-size/results/E1b-trim80k-q4head.qpack -n 96 $D/eval-ru.txt $D/eval-en.txt research/extended-holdout/ru.txt research/extended-holdout/en.txt research/qwen-fiction-holdout/alice.txt
