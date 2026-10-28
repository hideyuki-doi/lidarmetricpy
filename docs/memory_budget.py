"""Memory budgets on the current code: one dense full tile (90.5 M points) per setting, fresh process each.

    python memory_budget.py <backend> <gpu_memory_gb|none> <cpu_memory_gb|none>
Appends to docs/data/memory_budget.csv: seconds, path, blocks, passes, peak GPU, peak RSS, max |diff| vs the
unlimited run of the same backend (docs/data/mb_<backend>_none_none.parquet).
"""
import json
import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
VAL = HERE.parent / "validation"
sys.path.insert(0, str(HERE.parent))
from lidarmetricpy import Options, compute_tile  # noqa: E402

backend, mem, host = sys.argv[1], sys.argv[2], sys.argv[3]
mg = None if mem == "none" else float(mem)
hg = None if host == "none" else float(host)
c = {d["mesh"]: d for d in json.loads((VAL / "data/full.json").read_text())}["09LC9423"]
opt = Options(spike_filter=False, zrange=(-np.inf, np.inf), other_packages=False)
peak, stop = [0], [False]


def poll():
    if backend != "cupy":
        return
    import cupy
    pool = cupy.get_default_memory_pool()
    while not stop[0]:
        peak[0] = max(peak[0], pool.total_bytes())
        time.sleep(0.01)


clip = [d for d in json.loads((VAL / "data/clips.json").read_text()) if d["mesh"] == "09MC0523"][0]
compute_tile(str(VAL / clip["file"]), clip["x0"], clip["y1"], 10, 10, backend=backend, options=opt)      # warm-up
th = threading.Thread(target=poll, daemon=True)
th.start()
t = time.time()
df = compute_tile(str(VAL / c["file"]), c["x0"], c["y1"], 40, 30, backend=backend, gpu_memory_gb=mg, cpu_memory_gb=hg,
                  options=opt)
sec = time.time() - t
stop[0] = True
th.join()
df = df.sort_values(["y", "x"]).reset_index(drop=True)
tag = f"mb_{backend}_{mem}_{host}"
(HERE / "data").mkdir(exist_ok=True)
df.to_parquet(HERE / "data" / f"{tag}.parquet")
base = HERE / "data" / f"mb_{backend}_none_none.parquet"
diff = np.nan
if base.exists():
    b = pd.read_parquet(base)
    cols = [k for k in df.columns if k not in ("x", "y")]
    diff = max(float(np.nanmax(np.abs(df[k].values - b[k].values))) for k in cols)
    nan_mis = sum(int((df[k].isna() != b[k].isna()).sum()) for k in cols)
    diff = diff if nan_mis == 0 else float("inf")
st = df.attrs["stats"]
row = {"backend": backend, "gpu_memory_gb": mg, "cpu_memory_gb": hg, "seconds": round(sec, 2), "path": st.get("path"),
       "blocks": st["blocks"], "passes": st["passes"], "peak_gpu_gb": round(peak[0] / 1024 ** 3, 2),
       "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2),
       "max_abs_diff_vs_unlimited": diff}
p = HERE / "data" / "memory_budget.csv"
pd.DataFrame([row]).to_csv(p, mode="a", header=not p.exists(), index=False)
print(row)
