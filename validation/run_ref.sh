#!/bin/bash
cd "$(dirname "$0")"
RSCRIPT=${RSCRIPT:-Rscript}      # R with lidR 4.3.3, lidRmetrics 0.1.5 (+ leafR, forestr, ForestGapR, canopyLazR)
python3 -c "
import json
for c in json.load(open('data/clips.json')):
    if c['n']>0: print(c['file'], c['x0'], c['y1'], c['ncol'], c['nrow'], 'ref/'+c['mesh'])
" | while read f x0 y1 nc nr out; do
  ( /usr/bin/time -f "%e s" nice -n 10 $RSCRIPT ref_lidr.R $f $x0 $y1 $nc $nr $out > $out.log 2>&1 ) &
done
wait
echo ALLDONE
