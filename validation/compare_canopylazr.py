"""Compare lidarmetricpy.canopylazr with canopyLazR (ref/<mesh>_xy<res>_clz.csv) on the raw clips.
    python compare_canopylazr.py → compare_canopylazr.csv"""
import json
import sys
from pathlib import Path

import laspy
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lidarmetricpy.canopylazr import canopylazr  # noqa: E402
from lidarmetricpy.validate import compare_tables  # noqa: E402

rows = []
clips = json.loads((HERE / "data/rawclips.json").read_text())
need = [HERE / "ref" / f"{c['mesh']}_xy{xy}_clz.csv" for c in clips for xy in (10, 5)]
missing = [str(p) for p in need if not p.exists()]
if missing or not clips:
    sys.exit("missing reference files: " + ", ".join(missing) if missing else "no raw clip in the manifest")
for c in clips:
    las = laspy.read(str(HERE / c["file"]))
    for xy in (10, 5):
        p = HERE / "ref" / f"{c['mesh']}_xy{xy}_clz.csv"
        r = pd.read_csv(p)
        o = pd.DataFrame(canopylazr(las.x, las.y, las.z, xy, 1, 5))
        res_c = compare_tables(r, o, [k for k in r.columns if k.startswith("clz")])
        res_c["mesh"], res_c["xy_res"] = c["mesh"], xy
        rows.append(res_c)
res = pd.concat(rows, ignore_index=True)
res.to_csv(HERE / "compare_canopylazr.csv", index=False)
print(res.groupby("metric")[["n_finite", "missing", "extra", "class_mismatch", "value_mismatch", "max_abs_diff", "equal"]]
      .agg({"n_finite": "sum", "missing": "sum", "extra": "sum", "class_mismatch": "sum", "value_mismatch": "sum",
            "max_abs_diff": "max", "equal": "all"}))
sys.exit(0 if res.equal.all() else 1)
