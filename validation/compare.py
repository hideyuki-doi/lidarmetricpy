"""Compare lidarmetricpy with the lidR/lidRmetrics reference on the clips.

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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lidarmetricpy import Options, compute_tile  # noqa: E402
from refcheck import check  # noqa: E402

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
        res_c = check(ref, df, METRICS)
        res_c["mesh"] = c["mesh"]
        rows.append(res_c)
        # tree tops
        rt = pd.read_csv(HERE / "ref" / f"{c['mesh']}_trees.csv")
        pt = pd.DataFrame({"x": L.xmin + (cc + 0.5) * L.res, "y": L.ymax - (rr + 0.5) * L.res,
                           "h": chm[rr, cc], "crown_m2": area})
        j = rt.merge(pt, on=["x", "y"], how="outer", suffixes=("_r", "_p"), indicator=True)
        trees.append({"mesh": c["mesh"], "ref": len(rt), "pc": len(pt), "both": int((j._merge == "both").sum()),
                      "crown_equal": int((np.abs(j.crown_m2_r - j.crown_m2_p) < 1e-9).sum()),
                      "h_equal": int((np.abs(j.h_r - j.h_p) < 1e-9).sum())})
    allres = pd.concat(rows, ignore_index=True)
    res = allres.groupby("metric", sort=False).agg(
        missing=("missing", "sum"), extra=("extra", "sum"), class_mismatch=("class_mismatch", "sum"),
        value_mismatch=("value_mismatch", "sum"), n_finite=("n_finite", "sum"), max_abs_diff=("max_abs_diff", "max"),
        max_rel_diff=("max_rel_diff", "max"), equal=("equal", "all")).reset_index()
    res.to_csv(HERE / f"compare_{backend}.csv", index=False)
    pd.DataFrame(trees).to_csv(HERE / f"compare_trees_{backend}.csv", index=False)
    pd.DataFrame(times).to_csv(HERE / f"time_{backend}.csv", index=False)
    print(f"== {backend}")
    print(res.to_string(index=False))
    print(pd.DataFrame(trees).to_string(index=False))
    print(pd.DataFrame(times).to_string(index=False))
    t = pd.DataFrame(trees)
    trees_ok = bool((t.ref == t.pc).all() and (t.both == t.ref).all() and (t.crown_equal == t.ref).all()
                    and (t.h_equal == t.ref).all())
    return bool(res.equal.all()) and trees_ok


if __name__ == "__main__":
    ok = True
    for b in sys.argv[1:] or ["cupy", "numpy", "torch-cuda", "torch-cpu"]:
        ok &= run(b)
    sys.exit(0 if ok else 1)
