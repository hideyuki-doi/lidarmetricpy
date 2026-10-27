"""Per-cell pcmetrics vs reference values for the paper figures.

    python make_paper_data.py  → docs/data/cells_vs_reference.parquet, docs/data/canopylazr_vs_reference.parquet

Recomputes pcmetrics (cupy, float64; torch-cuda32, float32) on the four clips and the two full tiles and joins
the stored references (validation/ref*, written by ref_lidr.R, ref_other.R, ref_pyforestscan.py, ref_canopylazr.R).
"""
import json
import sys
from pathlib import Path

import laspy
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
VAL = HERE.parent / "validation"
sys.path.insert(0, str(HERE.parent))
from pcmetrics import Options, canopylazr, compute_tile  # noqa: E402

OUT = HERE / "data"
OUT.mkdir(exist_ok=True)
OPT = Options(spike_filter=False, zrange=(-np.inf, np.inf))


def refs(refdir, mesh):
    r = pd.read_csv(VAL / refdir / f"{mesh}_cells.csv")
    o = pd.read_csv(VAL / refdir / f"{mesh}_other.csv").merge(pd.read_csv(VAL / refdir / f"{mesh}_pfs.csv"),
                                                              on=["x", "y"], how="outer")
    return r.merge(o, on=["x", "y"], how="outer")


def main():
    rows = []
    sets = [("clip", "data/clips.json", "ref"), ("full", "data/full.json", "ref_full")]
    for kind, js, refdir in sets:
        for c in [c for c in json.loads((VAL / js).read_text()) if c["n"] > 0]:
            ref = refs(refdir, c["mesh"])
            for backend in ("cupy", "torch-cuda", "numpy", "torch-cpu", "torch-cuda32"):
                df = compute_tile(str(VAL / c["file"]), c["x0"], c["y1"], c["ncol"], c["nrow"], backend=backend, options=OPT)
                m = ref.merge(df, on=["x", "y"], how="left", suffixes=("_ref", "_pcm"))
                m["dataset"], m["mesh"], m["backend"] = kind, c["mesh"], backend
                rows.append(m)
                print(kind, c["mesh"], backend, len(m), flush=True)
    pd.concat(rows, ignore_index=True).to_parquet(OUT / "cells_vs_reference.parquet")
    cl = []
    for c in json.loads((VAL / "data/rawclips.json").read_text()):
        las = laspy.read(str(VAL / c["file"]))
        for xy in (10, 5):
            r = pd.read_csv(VAL / "ref" / f"{c['mesh']}_xy{xy}_clz.csv")
            o = pd.DataFrame(canopylazr(las.x, las.y, las.z, xy, 1, 5))
            m = r.merge(o, on=["x", "y"], how="outer", suffixes=("_ref", "_pcm"))
            m["mesh"], m["xy_res"] = c["mesh"], xy
            cl.append(m)
    pd.concat(cl, ignore_index=True).to_parquet(OUT / "canopylazr_vs_reference.parquet")
    print("done")


if __name__ == "__main__":
    main()
