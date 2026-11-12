#!/bin/bash
# usage: ab.sh reps binA binB ... ; interleaved, logs temp/clock
M=~/autoresearch-runs/qpack-tps-size/results/E1b-trim80k-q4head.qpack
reps=$1; shift
for r in $(seq $reps); do for b in "$@"; do
  t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
  others=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  out=$(QPACK_DEVICE=cuda $b -m $M --row -t 8 -p 64 -n 128 -r 2 --greedy 2>&1 | grep "@cuda" | tr '\n' ' ')
  c=$(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader)
  echo "$(basename $b) T0=$t others=$others after[clk,T]=$c :: $out"
done; done
