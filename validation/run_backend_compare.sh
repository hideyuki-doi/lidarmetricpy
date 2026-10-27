#!/bin/bash
# GPU / CPU comparison on a frozen copy of the package. Low priority: the machine is shared.
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}         # Python with pcmetrics installed
SNAP=$PWD/backend_compare/snapshot
rm -rf $SNAP && mkdir -p $SNAP && cp -r ../pcmetrics $SNAP/ && rm -rf $SNAP/pcmetrics/__pycache__
export PCMETRICS_PATH=$SNAP
mkdir -p backend_compare && rm -f backend_compare/single.csv backend_compare/throughput.csv
PY="nice -n 10 $PYTHON -W ignore"
for b in cupy torch-cuda torch-cuda32 numpy torch-cpu; do
  for m in 09LC9423 09MC0523; do $PY backend_compare.py single $b $m 2>&1 | grep -v ca_bundle; done
done
for b in cupy numpy; do $PY backend_compare.py single $b 09LC9423 other 2>&1 | grep -v ca_bundle; done
$PY backend_compare.py throughput cupy 1 6
$PY backend_compare.py throughput cupy 4 12
$PY backend_compare.py throughput cupy 12 12
$PY backend_compare.py throughput torch-cuda 12 12
$PY backend_compare.py throughput torch-cuda32 12 12
$PY backend_compare.py throughput numpy 1 2
$PY backend_compare.py throughput numpy 4 4 40
$PY backend_compare.py throughput numpy 8 8 24
$PY backend_compare.py throughput numpy 16 8 12
$PY backend_compare.py throughput torch-cpu 16 8 12
echo ALLDONE
