#!/bin/bash
set -euo pipefail
PIDS=()
cd "$(dirname "$0")"
RSCRIPT=${RSCRIPT:-Rscript}      # R with lidR 4.3.3, lidRmetrics 0.1.5 (+ leafR, forestr, ForestGapR, canopyLazR)
mapfile -t JOBS < <(python3 -c "
import json
for c in json.load(open('data/full.json')):
    print(c['file'], c['x0'], c['y1'], c['ncol'], c['nrow'], 'ref_full/'+c['mesh'])
")
for j in "${JOBS[@]}"; do
  set -- $j
  ( /usr/bin/time -f "%e s %M KB" nice -n 10 $RSCRIPT ref_lidr.R $1 $2 $3 $4 $5 $6 > $6.log 2>&1 ) &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done
echo ALLDONE
