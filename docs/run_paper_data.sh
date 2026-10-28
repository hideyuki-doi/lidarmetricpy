#!/bin/bash
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}         # Python with lidarmetricpy installed
PY="nice -n 10 $PYTHON -W ignore"
$PY make_paper_data.py 2>&1 | grep -v ca_bundle
./run_memory_budget.sh
