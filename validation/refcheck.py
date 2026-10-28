"""Shared rules for comparing lidarmetricpy with the stored R / Python references (strict: lidarmetricpy.validate).

- The cells that must be compared are the cells holding points, counted from the LAZ file itself
  (lidarmetricpy.validate.occupied_cells), not guessed from NA values: lidR lists every raster cell, and a
  metric can be NA in a cell with points.
- The R tree summary has no row for a cell without trees: tree_n is 0 there.
- Every other difference (missing or extra cells, NA / Inf pattern, values beyond 1e-6 relative) fails.
"""
import sys
from pathlib import Path

import laspy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidarmetricpy.validate import compare_tables, occupied_cells  # noqa: E402

HERE = Path(__file__).resolve().parent


def occupancy(laz, x0, y1, ncol, nrow, cell=25.0):
    las = laspy.read(str(laz))
    return occupied_cells(np.asarray(las.x), np.asarray(las.y), x0, y1, ncol, nrow, cell)


def prepare_reference(ref, occupied):
    """The reference rows of the occupied cells, with tree_n = 0 where the R summary has no tree."""
    out = ref.merge(occupied.assign(_occ=True), on=["x", "y"], how="inner").drop(columns="_occ")
    if "tree_n" in out:
        out.loc[out.tree_n.isna(), "tree_n"] = 0
    return out.reset_index(drop=True)


def check(ref, got, metrics, occupied):
    """compare_tables on the occupied cells; returns the per-metric table."""
    return compare_tables(prepare_reference(ref, occupied), got, metrics, occupied=occupied)


def require(paths):
    """Stop before any computation when a reference file is missing."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        sys.exit("missing reference files:\n  " + "\n  ".join(missing))
