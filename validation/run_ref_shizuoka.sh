#!/bin/bash
set -euo pipefail
PIDS=()
# References for the Shizuoka tiles (data/shizuoka.json, made by make_shizuoka.py) → ref_shizuoka/
cd "$(dirname "$0")"
RSCRIPT=${RSCRIPT:-Rscript}      # R with lidR 4.3.3, lidRmetrics 0.1.5 (+ leafR, forestr, ForestGapR)
PYTHON=${PYTHON:-python}         # Python with lidarmetricpy and pyforestscan 0.3.6
mkdir -p ref_shizuoka
JOBFILE=$(mktemp)
trap 'rm -f "$JOBFILE"' EXIT
$PYTHON -c "
import json
for c in json.load(open('data/shizuoka.json')): print(c['file'], c['x0'], c['y1'], c['ncol'], c['nrow'], 'ref_shizuoka/'+c['mesh'])
" > "$JOBFILE"                               # a failure here stops the script (set -e)
mapfile -t JOBS < "$JOBFILE"
[ ${#JOBS[@]} -gt 0 ] || { echo "no job in the manifest" >&2; exit 1; }
echo "${#JOBS[@]} job(s)"
for j in "${JOBS[@]}"; do
  set -- $j
  ( /usr/bin/time -f "%e s %M KB" nice -n 10 $RSCRIPT ref_lidr.R $1 $2 $3 $4 $5 $6 > $6.log 2>&1 ) &
  PIDS+=($!)
  ( nice -n 10 $RSCRIPT ref_other.R $1 $2 $3 $4 $5 $6 > $6.other.log 2>&1
    nice -n 10 $PYTHON -I ref_pyforestscan.py $1 $2 $3 $4 $5 $6 > $6.pfs.log 2>&1 ) &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done
echo ALLDONE
