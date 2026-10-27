"""Compare pcmetrics with the lidR/lidRmetrics reference on the clips.

    python compare.py [backend ...]      (default: cupy numpy torch-cuda torch-cpu)
Writes validation/compare_<backend>.csv: per metric, cells compared, max |diff|,
share of cells equal within 1e-6 (relative), and Spearman rho; plus tree-top matching.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from pcmetrics import Options, compute_tile  # noqa: E402

METRICS = ["zq20", "zcv", "CRR", "zentropy", "zpcum7", "zpcum9", "pz_above_30", "L4", "Lskew", "Lkurt",
           "lad_max", "lad_mean", "lad_cv", "p_single", "p_last", "ratio_last_first", "OpenGapSpace",
           "ClosedGapSpace", "Euphotic", "Oligophotic", "vzsd", "vzcv", "kde_peaks_count", "rumple",
           "tree_n", "tree_h_mean", "tree_h_sd", "tree_h_cv", "tree_h_max", "tree_h_gini", "tree_h_shannon",
           "crown_m2_mean", "crown_m2_sd"]


def run(backend):
    clips = [c for c in json.loads((HERE / "data/clips.json").read_text()) if c["n"] > 0]
    rows, trees, times = [], [], []
    opt = Options(spike_filter=False, zrange=(-np.inf, np.inf))
    for c in clips:
        refp = HERE / "ref" / f"{c['mesh']}_cells.csv"
        if not refp.exists():
            print("no reference yet for", c["mesh"])
            continue
        t = time.time()
        df, (chm, L, (rr, cc, area)) = compute_tile(str(HERE / c["file"]), c["x0"], c["y1"], c["ncol"], c["nrow"],
                                                    backend=backend, options=opt, return_trees=True)
        times.append({"mesh": c["mesh"], "points": c["n"], "seconds": time.time() - t})
        ref = pd.read_csv(refp)
        m = ref.merge(df, on=["x", "y"], suffixes=("_r", "_p"))
        m["mesh"] = c["mesh"]
        rows.append(m)
        # tree tops
        rt = pd.read_csv(HERE / "ref" / f"{c['mesh']}_trees.csv")
        pt = pd.DataFrame({"x": L.xmin + (cc + 0.5) * L.res, "y": L.ymax - (rr + 0.5) * L.res,
                           "h": chm[rr, cc], "crown_m2": area})
        j = rt.merge(pt, on=["x", "y"], how="outer", suffixes=("_r", "_p"), indicator=True)
        trees.append({"mesh": c["mesh"], "ref": len(rt), "pc": len(pt), "both": int((j._merge == "both").sum()),
                      "crown_equal": int((np.abs(j.crown_m2_r - j.crown_m2_p) < 1e-9).sum()),
                      "h_equal": int((np.abs(j.h_r - j.h_p) < 1e-9).sum())})
    m = pd.concat(rows, ignore_index=True)
    res = []
    for k in METRICS:
        a, b = m[f"{k}_r"] if f"{k}_r" in m else m.get(k), m[f"{k}_p"] if f"{k}_p" in m else None
        if a is None or b is None:
            res.append({"metric": k, "n": 0})
            continue
        both = a.notna() & b.notna()
        nan_mismatch = int((a.isna() != b.isna()).sum())
        d = (a[both] - b[both]).abs()
        tol = 1e-6 * np.maximum(1, a[both].abs())
        rho = spearmanr(a[both], b[both]).statistic if both.sum() > 2 and a[both].nunique() > 1 else np.nan
        res.append({"metric": k, "n": int(both.sum()), "nan_mismatch": nan_mismatch, "max_abs_diff": d.max(),
                    "equal_share": float(((d <= tol) | (a[both] == b[both])).mean()), "spearman": rho})
    res = pd.DataFrame(res)
    res.to_csv(HERE / f"compare_{backend}.csv", index=False)
    pd.DataFrame(trees).to_csv(HERE / f"compare_trees_{backend}.csv", index=False)
    pd.DataFrame(times).to_csv(HERE / f"time_{backend}.csv", index=False)
    print(f"== {backend}")
    print(res.to_string(index=False))
    print(pd.DataFrame(trees).to_string(index=False))
    print(pd.DataFrame(times).to_string(index=False))


if __name__ == "__main__":
    for b in sys.argv[1:] or ["cupy", "numpy", "torch-cuda", "torch-cpu"]:
        run(b)
