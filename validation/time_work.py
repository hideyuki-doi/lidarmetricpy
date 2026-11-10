"""Work time of compute_tile (reading, metrics, trees) after a warm-up call, as reported in the README.

    python time_work.py <backend> [repeats] [gpu budgets, e.g. 16,32,90]   → timing/work_<backend>.csv
    (both full tiles and every GPU budget, median of the repeats)
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
budgets = [float(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else [None]   # GPU budgets, GiB
rows = []
for gb, c in [(g, c) for g in budgets for c in json.loads((HERE / "data/full.json").read_text())]:
    args = (str(HERE / c["file"]), c["x0"], c["y1"], c["ncol"], c["nrow"])
    opt = Options.lidr(other_packages=False)
    kw = dict(backend=backend, options=opt, gpu_memory_gb=gb, cpu_memory_gb=64)
    compute_tile(*args, **kw)                                              # warm-up (compilation, pools)
    ts = []
    for _ in range(rep):
        t = time.time()
        df = compute_tile(*args, **kw)
        ts.append(time.time() - t)
    rows.append({"mesh": c["mesh"], "points": c["n"], "backend": backend, "gpu_memory_gb": gb, "seconds_median": float(np.median(ts)),
                 "seconds_all": " ".join(f"{v:.2f}" for v in ts), "columns": df.shape[1], "path": df.attrs["stats"].get("path", ""),
                 "stages_last": json.dumps(df.attrs["stats"]["stage_seconds"])})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(HERE / f"timing/work_{backend}.csv", index=False)
