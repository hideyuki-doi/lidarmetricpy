#!/bin/bash
set -euo pipefail
PIDS=()
cd "$(dirname "$0")"
RSCRIPT=${RSCRIPT:-Rscript}      # R with lidR 4.3.3, lidRmetrics 0.1.5 (+ leafR, forestr, ForestGapR, canopyLazR)
start=$(date +%s.%N)
for i in 1 2 3 4; do
  for m in 09LC9423 09MC0523; do
    read f x0 y1 <<< $(python3 -c "import json;c={d['mesh']:d for d in json.load(open('data/full.json'))}['$m'];print(c['file'],c['x0'],c['y1'])")
    ( nice -n 10 $RSCRIPT ref_lidr.R $f $x0 $y1 40 30 par/${m}_$i > par/${m}_$i.log 2>&1 ) &
  PIDS+=($!)
  done
done
for p in "${PIDS[@]}"; do wait "$p"; done
end=$(date +%s.%N)
echo "lidR 8 jobs parallel(8): $(echo "$end - $start" | bc) s"
