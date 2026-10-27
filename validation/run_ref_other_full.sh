#!/bin/bash
cd "$(dirname "$0")"
RSCRIPT=${RSCRIPT:-Rscript}      # R with lidR 4.3.3, lidRmetrics 0.1.5 (+ leafR, forestr, ForestGapR, canopyLazR)
mapfile -t JOBS < <(python3 -c "
import json
for c in json.load(open('data/full.json')): print(c['file'], c['x0'], c['y1'], c['ncol'], c['nrow'], 'ref_full/'+c['mesh'])
")
for j in "${JOBS[@]}"; do
  set -- $j
  nice -n 10 $RSCRIPT ref_other.R $1 $2 $3 $4 $5 $6 > $6.other.log 2>&1 &
done
wait
echo ALLDONE
