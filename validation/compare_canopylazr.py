"""Compare pcmetrics.canopylazr with canopyLazR (ref/<mesh>_xy<res>_clz.csv) on the raw clips.
    python compare_canopylazr.py → compare_canopylazr.csv"""
import json
import sys
from pathlib import Path

import laspy
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from pcmetrics.canopylazr import canopylazr  # noqa: E402

rows = []
for c in json.loads((HERE / "data/rawclips.json").read_text()):
    las = laspy.read(str(HERE / c["file"]))
    for xy in (10, 5):
        p = HERE / "ref" / f"{c['mesh']}_xy{xy}_clz.csv"
        if not p.exists():
            continue
        r = pd.read_csv(p)
        o = pd.DataFrame(canopylazr(las.x, las.y, las.z, xy, 1, 5))
        m = r.merge(o, on=["x", "y"], how="outer", suffixes=("_r", "_p"), indicator=True)
        for k in [k for k in r.columns if k.startswith("clz")]:
            a, b = m[k + "_r"], m[k + "_p"]
            both = a.notna() & b.notna()
            d = (a[both] - b[both]).abs()
            rows.append({"mesh": c["mesh"], "xy_res": xy, "metric": k, "columns": len(m),
                         "unmatched_columns": int((m._merge != "both").sum()), "n": int(both.sum()),
                         "nan_mismatch": int((a.isna() != b.isna()).sum()), "max_abs_diff": float(d.max()),
                         "equal_share": float(((d <= 1e-6 * np.maximum(1, a[both].abs())) | (a[both] == b[both])).mean())})
res = pd.DataFrame(rows)
res.to_csv(HERE / "compare_canopylazr.csv", index=False)
print(res.groupby("metric")[["n", "nan_mismatch", "unmatched_columns", "max_abs_diff", "equal_share"]]
      .agg({"n": "sum", "nan_mismatch": "sum", "unmatched_columns": "max", "max_abs_diff": "max", "equal_share": "min"}))
