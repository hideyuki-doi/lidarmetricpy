"""Work time of compute_tile (reading, metrics, trees) after a warm-up call, as reported in the README.

    python time_work.py <backend> [repeats]   → timing/work_<backend>.csv (both full tiles, median of repeats)
lidRmetrics metrics_set3 + trees (Options.lidr(other_packages=False)), the work timed in R by ref_lidr.R.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lidarmetricpy import Options, compute_tile  # noqa: E402

backend = sys.argv[1]
rep = int(sys.argv[2]) if len(sys.argv) > 2 else 3
rows = []
for c in json.loads((HERE / "data/full.json").read_text()):
    args = (str(HERE / c["file"]), c["x0"], c["y1"], c["ncol"], c["nrow"])
    opt = Options.lidr(other_packages=False)
    compute_tile(*args, backend=backend, options=opt)                      # warm-up (compilation, pools)
    ts = []
    for _ in range(rep):
        t = time.time()
        df = compute_tile(*args, backend=backend, options=opt)
        ts.append(time.time() - t)
    rows.append({"mesh": c["mesh"], "points": c["n"], "backend": backend, "seconds_median": float(np.median(ts)),
                 "seconds_all": " ".join(f"{v:.2f}" for v in ts), "columns": df.shape[1],
                 "stages_last": json.dumps(df.attrs["stats"]["stage_seconds"])})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(HERE / f"timing/work_{backend}.csv", index=False)
