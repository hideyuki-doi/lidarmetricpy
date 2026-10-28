"""Compare lidarmetricpy with leafR / forestr / ForestGapR (ref/<mesh>_other.csv) and PyForestScan (ref/<mesh>_pfs.csv).

    python compare_other.py [backend ...]   → compare_other_<backend>.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lidarmetricpy import Options, compute_tile  # noqa: E402

OTHER = ["leafr_lai", "leafr_lahv", "leafr_lahv_laiw", "leafr_lahv_hw", "leafr_lahv_laiw_hw", "leafr_fhd",
         "leafr_fhd_even", "leafr_gs", "leafr_gs_even", "leafr_gc", "enl", "gap_frac", "gap_n", "gap_area_mean"]
PFS = ["pfs_pai", "pfs_cover", "pfs_fhd"]


def run(backend, full=False):
    src, refdir = ("data/full.json", "ref_full") if full else ("data/clips.json", "ref")
    clips = [c for c in json.loads((HERE / src).read_text()) if c["n"] > 0]
    rows = []
    for c in clips:
        p1, p2 = HERE / refdir / f"{c['mesh']}_other.csv", HERE / refdir / f"{c['mesh']}_pfs.csv"
        if not (p1.exists() and p2.exists()):
            print("no reference for", c["mesh"])
            continue
        df = compute_tile(str(HERE / c["file"]), c["x0"], c["y1"], c["ncol"], c["nrow"], backend=backend,
                          options=Options(spike_filter=False, zrange=(-np.inf, np.inf)))
        ref = pd.read_csv(p1).merge(pd.read_csv(p2), on=["x", "y"], how="outer")
        m = ref.merge(df, on=["x", "y"], how="left", suffixes=("_r", "_p"))
        m["mesh"] = c["mesh"]
        rows.append(m)
    m = pd.concat(rows, ignore_index=True)
    res = []
    for k in OTHER + PFS:
        a, b = m[k + "_r"], m[k + "_p"]
        both = a.notna() & b.notna()
        d = (a[both] - b[both]).abs()
        tol = 1e-6 * np.maximum(1, a[both].abs())
        res.append({"metric": k, "n": int(both.sum()), "nan_mismatch": int((a.isna() != b.isna()).sum()),
                    "max_abs_diff": float(d.max()) if len(d) else np.nan,
                    "equal_share": float(((d <= tol) | (a[both] == b[both])).mean()) if len(d) else np.nan})
    res = pd.DataFrame(res)
    res.to_csv(HERE / f"compare_other_{backend}{'_full' if full else ''}.csv", index=False)
    print(f"== {backend}{' (full tiles)' if full else ''}")
    print(res.to_string(index=False))
    return m


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--full"]
    for b in args or ["cupy"]:
        run(b, full="--full" in sys.argv)
