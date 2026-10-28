"""Single-cell edge cases against lidRmetrics::metrics_set3 run in R (tests/make_oracle.R → data/ref)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lidarmetricpy import Options, compute_tile
from lidarmetricpy.area import AREA_COLUMNS
from lidarmetricpy.validate import compare_tables
from oracle_cases import cases
from test_reference import BACKENDS

REF = Path(__file__).resolve().parent / "data" / "ref" / "oracle_cases.csv"
METRICS = ["n"] + [m for m in AREA_COLUMNS if m != "enl"] + ["rumple"]
# Documented deviation: when all heights are equal, L2 = 0 and the L-moment ratios of R's Lmoments package
# are rounding noise (NaN, +-Inf or any number); lidarmetricpy returns NA for Lskew and Lkurt.
EXCEPTIONS = {("constant", "Lskew"), ("constant", "Lkurt")}


@pytest.mark.parametrize("backend", BACKENDS)
def test_edge_cases_match_lidrmetrics(backend):
    ref = pd.read_csv(REF)
    rows = []
    for name, d in cases().items():
        opt = Options(zmin=d["zmin"], spike_filter=False, zrange=(-np.inf, np.inf), trees=False,
                      other_packages=False)
        src = dict(x=d["x"], y=d["y"], z=d["z"], return_number=d["rn"], number_of_returns=d["nr"])
        df = compute_tile(src, 0.0, 25.0, 1, 1, cell=25.0, backend=backend, options=opt)
        assert len(df) == 1 and df.n_points.iloc[0] == len(d["z"]), name
        rows.append(df.assign(case=name))
    got = pd.concat(rows, ignore_index=True)
    for case, m in EXCEPTIONS:
        assert np.isnan(got.loc[got.case == case, m].iloc[0])
        ref.loc[ref.case == case, m] = np.nan
    res = compare_tables(ref, got, METRICS, keys=["case"])
    bad = res[~res.equal]
    assert bad.empty, bad.to_string()
