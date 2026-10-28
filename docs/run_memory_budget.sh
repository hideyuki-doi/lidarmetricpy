#!/bin/bash
# Memory budgets only (one dense tile per setting, fresh process each) → data/memory_budget.csv
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}         # Python with lidarmetricpy installed
PY="nice -n 10 $PYTHON -W ignore"
rm -f data/memory_budget.csv
for cfg in "cupy none none" "cupy 16 none" "cupy 8 none" "cupy 6 none" "cupy 5 none" "cupy 4 none" "cupy 2 none" "cupy none 8" "cupy 2 4" "numpy none none" "numpy none 16" "numpy none 8" "numpy none 4"; do
  set -- $cfg; $PY memory_budget.py $1 $2 $3 2>&1 | grep -v ca_bundle
done
echo ALLDONE
