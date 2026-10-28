#!/bin/bash
# all references in the foreground of this script (no pipe subshell, so `wait` waits for them)
cd "$(dirname "$0")"
RSCRIPT=${RSCRIPT:-Rscript}      # R with lidR 4.3.3, lidRmetrics 0.1.5 (+ leafR, forestr, ForestGapR, canopyLazR)
PYTHON=${PYTHON:-python}         # Python with lidarmetricpy installed
mapfile -t JOBS < <($PYTHON -c "
import json
for c in json.load(open('data/clips.json')):
    if c['n']>0: print(c['file'], c['x0'], c['y1'], c['ncol'], c['nrow'], 'ref/'+c['mesh'])
for c in json.load(open('data/full.json')): print(c['file'], c['x0'], c['y1'], c['ncol'], c['nrow'], 'ref_full/'+c['mesh'])
")
for j in "${JOBS[@]}"; do
  set -- $j
  ( nice -n 10 $RSCRIPT ref_other.R $1 $2 $3 $4 $5 $6 > $6.other.log 2>&1
    nice -n 10 $PYTHON -I ref_pyforestscan.py $1 $2 $3 $4 $5 $6 > $6.pfs.log 2>&1 ) &
done
wait
echo ALLDONE
