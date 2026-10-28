#!/bin/bash
# Reference values for the test data (run once; the outputs are committed in tests/data/ref).
#   RSCRIPT=<Rscript with lidR 4.3.3, lidRmetrics 0.1.5, leafR 0.3.5, forestr 2.0.2, ForestGapR 0.1.7, canopyLazR 2.0>
#   PYTHON=<python with lidarmetricpy and pyforestscan 0.3.6>  ./make_references.sh
# Everything is written to a temporary folder first; the committed files are replaced only when every
# step succeeded and every output has the expected number of rows.
set -euo pipefail
cd "$(dirname "$0")"
RSCRIPT=${RSCRIPT:-Rscript}
PYTHON=${PYTHON:-python}
V=../validation
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
G="data/synthetic.laz -62000 -57750 4 4 $TMP/synthetic"
nice -n 10 "$RSCRIPT" $V/ref_lidr.R $G > "$TMP/synthetic.log" 2>&1
nice -n 10 "$RSCRIPT" $V/ref_other.R $G > "$TMP/synthetic.other.log" 2>&1
nice -n 10 "$PYTHON" -I $V/ref_pyforestscan.py $G > "$TMP/synthetic.pfs.log" 2>&1
for xy in 10 5; do
  nice -n 10 "$RSCRIPT" $V/ref_canopylazr.R data/synthetic_raw.laz "$TMP/synthetic_xy$xy" $xy 1 5 \
    > "$TMP/synthetic_xy$xy.clz.log" 2>&1
done
"$PYTHON" oracle_cases.py > /dev/null
cp data/ref/oracle_input.txt "$TMP/"
nice -n 10 "$RSCRIPT" make_oracle.R "$TMP/oracle_input.txt" "$TMP/oracle_cases.csv" > "$TMP/oracle.log" 2>&1
# expected rows: 16 cells, 100 / 400 canopyLazR columns, one row per oracle case
"$PYTHON" - "$TMP" <<'EOF'
import sys, pandas as pd
t = sys.argv[1]
want = {"synthetic_cells.csv": 16, "synthetic_other.csv": 16, "synthetic_pfs.csv": 16,
        "synthetic_xy10_clz.csv": 100, "synthetic_xy5_clz.csv": 400,
        "oracle_cases.csv": sum(1 for _ in open(f"{t}/oracle_input.txt"))}
for f, n in want.items():
    got = len(pd.read_csv(f"{t}/{f}"))
    if got != n:
        sys.exit(f"{f}: {got} rows, expected {n}")
if len(pd.read_csv(f"{t}/synthetic_trees.csv")) == 0:
    sys.exit("synthetic_trees.csv is empty")
EOF
for f in synthetic_cells.csv synthetic_trees.csv synthetic_other.csv synthetic_pfs.csv synthetic_xy10_clz.csv \
         synthetic_xy5_clz.csv oracle_cases.csv; do
  cp "$TMP/$f" data/ref/
done
"$RSCRIPT" -e 'cat(R.version.string, "\n"); for (p in c("lidR", "lidRmetrics", "leafR", "forestr", "ForestGapR", "canopyLazR", "Lmoments")) cat(p, as.character(packageVersion(p)), "\n")' \
  > data/ref/versions.txt
"$PYTHON" -c 'import importlib.metadata as m; print("pyforestscan", m.version("pyforestscan"))' >> data/ref/versions.txt
echo done
