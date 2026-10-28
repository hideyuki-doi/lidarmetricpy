#!/bin/bash
# Whole-process timings of lidarmetricpy (fresh process each: import, compute, save), as run_timing_lidr.sh.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
mkdir -p timing
for b in cupy numpy; do
  for m in 09LC9423 09MC0523; do
    /usr/bin/time -f "%e s %M KB" nice -n 10 "$PYTHON" -W ignore py_end_to_end.py $b $m timing/py_${b}_$m \
      > timing/py_${b}_$m.log 2>&1
    rm -f timing/py_${b}_${m}_cells.csv timing/py_${b}_${m}_trees.csv timing/py_${b}_${m}_chm.tif
  done
done
echo ALLDONE
