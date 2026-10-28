"""lidarmetricpy against the reference packages on the synthetic tile (tests/data, references in tests/data/ref).

Every float64 backend must reproduce every reference value (within 1e-6 relative, same NA pattern) and every
tree top. Backends that are not installed are skipped.
"""
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
import pytest

from lidarmetricpy import Options, canopylazr, compute_tile, get_backend
from lidarmetricpy.validate import compare_tables, occupied_cells

DATA = Path(__file__).resolve().parent / "data"
REF = DATA / "ref"
X0, Y1, NC, NR = -62000.0, -57750.0, 4, 4
OPT = Options(spike_filter=False, zrange=(-np.inf, np.inf))      # the references use every point

LIDRMETRICS = ["zq20", "zcv", "CRR", "zentropy", "zpcum7", "zpcum9", "pz_above_30", "L4", "Lskew", "Lkurt",
               "lad_max", "lad_mean", "lad_cv", "p_single", "p_last", "ratio_last_first", "OpenGapSpace",
               "ClosedGapSpace", "Euphotic", "Oligophotic", "vzsd", "vzcv", "kde_peaks_count", "rumple"]
TREES = ["tree_n", "tree_h_mean", "tree_h_sd", "tree_h_cv", "tree_h_max", "tree_h_gini", "tree_h_shannon",
         "crown_m2_mean", "crown_m2_sd"]
OTHER = ["leafr_lai", "leafr_lahv", "leafr_lahv_laiw", "leafr_lahv_hw", "leafr_lahv_laiw_hw", "leafr_fhd",
         "leafr_fhd_even", "leafr_gs", "leafr_gs_even", "leafr_gc", "enl", "gap_frac", "gap_n", "gap_area_mean",
         "pfs_pai", "pfs_cover", "pfs_fhd"]
CLZ = ["clz_chm", "clz_ground", "clz_vertical_sd_lad", "clz_rugosity", "clz_filled_ratio", "clz_porosity_ratio",
       "clz_max_lad", "clz_max_lad_ht", "clz_toc_rugosity"]


def _available(name):
    """True only if the backend can actually allocate on its device (CPU-only PyTorch has no 'cuda')."""
    try:
        if name in ("torch-cuda", "torch-cuda32"):
            import torch
            if not torch.cuda.is_available():
                return False
        if name == "mps":
            import torch
            if not torch.backends.mps.is_available():
                return False
        B = get_backend(name)
        B.zeros(1)
        return True
    except Exception:  # noqa: BLE001
        return False


BACKENDS = [pytest.param(b, marks=pytest.mark.skipif(not _available(b), reason=f"{b} not available"))
            for b in ("numpy", "torch-cpu", "cupy", "torch-cuda")]


def assert_same(ref, got, cols, what, occupied=None):
    res = compare_tables(ref, got, cols, occupied=occupied)
    bad = res[~res.equal]
    assert bad.empty, what + "\n" + bad.to_string()


@pytest.fixture(scope="module")
def reference():
    r = pd.read_csv(REF / "synthetic_cells.csv")
    o = pd.read_csv(REF / "synthetic_other.csv").merge(pd.read_csv(REF / "synthetic_pfs.csv"), on=["x", "y"])
    return r.merge(o, on=["x", "y"], how="outer")


@pytest.mark.parametrize("backend", BACKENDS)
def test_cells_match_reference(backend, reference):
    df, (chm, L, (rr, cc, area)) = compute_tile(str(DATA / "synthetic.laz"), X0, Y1, NC, NR, backend=backend,
                                                options=OPT, return_trees=True)
    las = laspy.read(DATA / "synthetic.laz")
    occ = occupied_cells(np.asarray(las.x), np.asarray(las.y), X0, Y1, NC, NR, 25.0)    # cells with points
    assert len(occ) == 15 and len(df) == len(occ)
    ref = reference.merge(occ, on=["x", "y"])
    ref.loc[ref.tree_n.isna(), "tree_n"] = 0                 # the R summary has no row for cells without trees
    assert_same(ref, df, ["n"] + LIDRMETRICS, "lidRmetrics", occ)
    assert_same(ref, df, TREES, "lidR trees", occ)
    assert_same(ref, df, OTHER, "leafR / forestr / ForestGapR / PyForestScan", occ)
    # tree tops: same positions, heights and crown areas
    rt = pd.read_csv(REF / "synthetic_trees.csv")
    pt = pd.DataFrame({"x": L.xmin + (cc + 0.5) * L.res, "y": L.ymax - (rr + 0.5) * L.res,
                       "h": chm[rr, cc], "crown_m2": area})
    j = rt.merge(pt, on=["x", "y"], how="outer", suffixes=("_r", "_p"), indicator=True)
    assert (j._merge == "both").all(), "tree tops differ"
    assert np.allclose(j.h_r, j.h_p, rtol=0, atol=1e-9)
    assert np.allclose(j.crown_m2_r, j.crown_m2_p, rtol=0, atol=1e-9)


def test_canopylazr_matches_reference():
    las = laspy.read(DATA / "synthetic_raw.laz")
    for xy in (10, 5):
        r = pd.read_csv(REF / f"synthetic_xy{xy}_clz.csv")
        o = pd.DataFrame(canopylazr(las.x, las.y, las.z, xy, 1, 5))
        assert len(r) == len(o)
        assert_same(r, o, CLZ, f"canopyLazR {xy} m")
