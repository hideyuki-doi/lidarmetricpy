#!/bin/bash
# Reference values for the synthetic tiles (run once; the outputs are committed in tests/data/ref).
#   RSCRIPT=<Rscript with lidR 4.3.3, lidRmetrics 0.1.5, leafR 0.3.5, forestr 2.0.2, ForestGapR 0.1.7, canopyLazR 2.0>
#   PYTHON=<python with pyforestscan 0.3.6>  ./make_references.sh
cd "$(dirname "$0")"
RSCRIPT=${RSCRIPT:-Rscript}
PYTHON=${PYTHON:-python}
V=../validation
G="data/synthetic.laz -62000 -57750 4 4 data/ref/synthetic"
nice -n 10 $RSCRIPT $V/ref_lidr.R $G > data/ref/synthetic.log 2>&1
nice -n 10 $RSCRIPT $V/ref_other.R $G > data/ref/synthetic.other.log 2>&1
nice -n 10 $PYTHON -I $V/ref_pyforestscan.py $G > data/ref/synthetic.pfs.log 2>&1
for xy in 10 5; do
  nice -n 10 $RSCRIPT $V/ref_canopylazr.R data/synthetic_raw.laz data/ref/synthetic_xy$xy $xy 1 5 > data/ref/synthetic_xy$xy.clz.log 2>&1
done
rm -f data/ref/synthetic_chm.tif
echo done
