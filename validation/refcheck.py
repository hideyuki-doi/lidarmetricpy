"""Shared rules for comparing lidarmetricpy with the stored R / Python references (strict: lidarmetricpy.validate).

- A reference cell whose lidRmetrics values are all NA had no point: lidR lists every raster cell,
  lidarmetricpy only cells with points. Such cells are left out (they must then be absent from the result).
- The R tree summary has no row for a cell without trees: tree_n is 0 there.
- Every other difference (missing or extra cells, NA / Inf pattern, values beyond 1e-6 relative) fails.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidarmetricpy.validate import compare_tables  # noqa: E402

LIDRMETRICS = ["zq20", "zcv", "CRR", "zentropy", "zpcum7", "zpcum9", "pz_above_30", "L4", "Lskew", "Lkurt",
               "lad_max", "lad_mean", "lad_cv", "p_single", "p_last", "ratio_last_first", "OpenGapSpace",
               "ClosedGapSpace", "Euphotic", "Oligophotic", "vzsd", "vzcv", "kde_peaks_count", "rumple"]


def prepare_reference(ref, cells_ref=None):
    """Drop empty cells (judged on the lidRmetrics reference cells_ref, default ref itself) and set tree_n."""
    base = ref if cells_ref is None else cells_ref
    cols = [c for c in LIDRMETRICS + ["n"] if c in base.columns]
    empty = base.loc[base[cols].isna().all(axis=1), ["x", "y"]]
    out = ref.merge(empty.assign(_empty=True), on=["x", "y"], how="left")
    out = out[out._empty.isna()].drop(columns="_empty")
    if "tree_n" in out:
        out.loc[out.tree_n.isna(), "tree_n"] = 0
    return out.reset_index(drop=True)


def check(ref, got, metrics, cells_ref=None):
    """compare_tables after prepare_reference; returns the per-metric table."""
    return compare_tables(prepare_reference(ref, cells_ref), got, metrics)
