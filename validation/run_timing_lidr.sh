#!/bin/bash
# lidR timings with the same intervals as the Python ones (docs: speed): one tile at a time, one R process,
# /usr/bin/time for the whole process (start -> files saved), read_s / area_s / tree_s for the work inside.
set -euo pipefail
cd "$(dirname "$0")"
RSCRIPT=${RSCRIPT:-Rscript}
mkdir -p timing
JOBFILE=$(mktemp)
trap 'rm -f "$JOBFILE"' EXIT
python3 -c "
import json
for c in json.load(open('data/full.json')): print(c['file'], c['x0'], c['y1'], c['ncol'], c['nrow'], c['mesh'])
" > "$JOBFILE"
mapfile -t JOBS < "$JOBFILE"
[ ${#JOBS[@]} -gt 0 ] || { echo "no job in the manifest" >&2; exit 1; }
for v in ref_lidr_subset ref_lidr; do
  for j in "${JOBS[@]}"; do
    set -- $j
    /usr/bin/time -f "%e s %M KB" nice -n 10 "$RSCRIPT" $v.R $1 $2 $3 $4 $5 timing/${v}_$6 > timing/${v}_$6.log 2>&1
    rm -f timing/${v}_$6_cells.csv timing/${v}_$6_trees.csv timing/${v}_$6_chm.tif
  done
done
echo ALLDONE
