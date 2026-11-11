"""How much does each metric change when the point cloud is thinned?

    python density_sensitivity.py [backend]          (default: cupy)
Computes all 137 metrics on the four clips at full density (52-133 points per m2) and on the same clips thinned
by pulse to 2, 5 and 10 points per m2 (make_thin.py), and compares cell by cell (400 cells of 25 m).
Writes validation/density_sensitivity.csv: per metric and density, the median absolute change and the median
change, both divided by the median absolute value at full density, and Spearman's rho between thinned and full.
The thinned clouds come from the same flights, so this shows the effect of density alone, not of sensor,
scan geometry or season.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lidarmetricpy import Options, compute_tile  # noqa: E402


def run(c, backend):
    return compute_tile(str(HERE / c["file"]), c["x0"], c["y1"], c["ncol"], c["nrow"], backend=backend,
                        options=Options.lidr()).assign(source=c.get("source", c["mesh"]))


def main(backend="cupy"):
    clips = [c for c in json.loads((HERE / "data/clips.json").read_text()) if c["n"] > 0]
    thin = json.loads((HERE / "data/thin.json").read_text())
    full = pd.concat([run(c, backend) for c in clips], ignore_index=True)
    metrics = [c for c in full.columns if c not in ("x", "y", "source")]
    rows = []
    for d in (2, 5, 10):
        t = pd.concat([run(c, backend) for c in thin if c["mesh"].endswith(f"_d{d}")], ignore_index=True)
        j = full.merge(t, on=["source", "x", "y"], suffixes=("_f", "_t"))
        for m in metrics:
            a, b = j[m + "_f"].to_numpy(float), j[m + "_t"].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            scale = np.median(np.abs(a[ok])) if ok.any() else np.nan
            rho = pd.Series(a[ok]).corr(pd.Series(b[ok]), method="spearman") if ok.sum() > 2 else np.nan
            rows.append({"metric": m, "points_m2": d, "cells": int(ok.sum()),
                         "finite_full": int(np.isfinite(a).sum()), "finite_thin": int(np.isfinite(b).sum()),
                         "median_full": np.median(a[ok]) if ok.any() else np.nan,
                         "median_abs_change": np.median(np.abs(b[ok] - a[ok])) / scale if ok.any() and scale > 0 else np.nan,
                         "median_change": np.median(b[ok] - a[ok]) / scale if ok.any() and scale > 0 else np.nan,
                         "spearman": rho})
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "density_sensitivity.csv", index=False)
    w = out.pivot(index="metric", columns="points_m2", values="median_abs_change").loc[metrics]
    print(w.round(3).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:2])
