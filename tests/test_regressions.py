"""Regression tests for the issues of the 2026-10-28 review (each was a reproduced failure of v0.2.0)."""
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from lidarmetricpy import Grid, Options, compute_tile, get_backend
from lidarmetricpy.validate import compare_tables
from test_reference import BACKENDS, _available

STRICT = dict(spike_filter=False, zrange=(-np.inf, np.inf))


def src(x, y, z, rn=None, nr=None):
    n = len(z)
    return dict(x=np.asarray(x, float), y=np.asarray(y, float), z=np.asarray(z, float),
                return_number=np.ones(n) if rn is None else np.asarray(rn), number_of_returns=np.ones(n) if nr is None
                else np.asarray(nr))


def same(a, b, cols):
    for k in cols:
        x, y = np.asarray(a[k], float), np.asarray(b[k], float)
        assert np.array_equal(np.isnan(x), np.isnan(y)) and np.array_equal(x[~np.isnan(x)], y[~np.isnan(y)]), k


# ---------------------------------------------------------------- cells must not influence each other
@pytest.mark.parametrize("backend", BACKENDS)
def test_rumple_fractional_grid_cells_independent(backend):
    x = [24.1, 25.1, 24.2, 25.05, 25.3, 26.3, 25.4, 26.2]
    y = [1.1, 1.1, 2.2, 2.1] * 2
    z = [10] * 4 + [5] * 4
    opt = Options(trees=False, other_packages=False, **STRICT)
    both = compute_tile(src(x, y, z), 0.25, 25.25, 2, 1, cell=25, backend=backend, options=opt)
    assert both.rumple.tolist() == [1.0, 1.0]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("x0, y1, cell", [(0.4, 25.0, 25.0), (0.25, 10.7, 2.5), (-61999.6, -57750.3, 7.3)])
def test_cells_same_alone_or_together(backend, x0, y1, cell):
    """Every cell gives the same values whether its neighbours are computed with it or not."""
    rng = np.random.default_rng(11)
    n = 3000
    x = x0 + rng.uniform(0, 3 * cell, n)
    y = y1 - rng.uniform(0, 2 * cell, n)
    z = np.round(rng.gamma(2.5, 5, n), 3)
    nr = rng.integers(1, 4, n)
    rn = np.minimum(rng.integers(1, 4, n), nr)
    opt = Options(trees=False, **STRICT)
    allc = compute_tile(src(x, y, z, rn, nr), x0, y1, 3, 2, cell=cell, backend=backend, options=opt)
    cols = [c for c in allc.columns if c not in ("x", "y")]
    for r in range(2):
        for c in range(3):
            one = compute_tile(src(x, y, z, rn, nr), x0 + c * cell, y1 - r * cell, 1, 1, cell=cell,
                               backend=backend, options=opt)
            assert len(allc) == 6 and len(one) == 1
            row = allc.iloc[[r * 3 + c]].reset_index(drop=True)          # rows: top row first, then x
            assert np.isclose(row.x.iloc[0], one.x.iloc[0]) and np.isclose(row.y.iloc[0], one.y.iloc[0])
            same(row, one, cols)


@pytest.mark.parametrize("backend", BACKENDS)
def test_high_canopy_does_not_leak(backend):
    opt = Options(zrange=(-1, 100), spike_filter=False, trees=False, other_packages=False)
    x, y = [1, 2, 3, 26, 27, 28], [1, 2, 3, 1, 2, 3]
    df = compute_tile(src(x, [24 - v for v in y], [1, 2, 61.2, 1, 2, 3]), 0, 25, 2, 1, cell=25,
                      backend=backend, options=opt)
    assert np.allclose(df.zentropy, [math.log(3) / math.log(62), math.log(2) / math.log(3)], rtol=1e-12)
    one = compute_tile(src([1, 2, 3], [23, 22, 21], [1, 2, 61.2]), 0, 25, 1, 1, cell=25, backend=backend,
                       options=opt)
    assert np.isclose(one.zentropy.iloc[0], math.log(3) / math.log(62), rtol=1e-12)


# ---------------------------------------------------------------- ground cells and a fixed table
@pytest.mark.parametrize("trees", [False, True])
def test_ground_cells_kept_and_columns_fixed(trees):
    rng = np.random.default_rng(3)
    xl, yl = rng.uniform(0, 25, 100), rng.uniform(0, 25, 100)
    xr, yr = rng.uniform(25, 50, 100), rng.uniform(0, 25, 100)
    opt = Options(trees=trees, **STRICT)
    mixed = compute_tile(src(np.r_[xl, xr], np.r_[yl, yr], np.r_[np.full(100, 0.2), rng.uniform(10, 15, 100)]),
                         0, 25, 2, 1, cell=25, backend="numpy", options=opt)
    assert len(mixed) == 2
    left = mixed.iloc[0]
    assert left.n_points == 100 and np.isnan(left.n)
    assert left.pfs_cover == 0 and left.pfs_pai == 0 and left.pfs_fhd == 0
    ground = compute_tile(src(xl, yl, np.full(100, 0.2)), 0, 25, 2, 1, cell=25, backend="numpy", options=opt)
    assert list(ground.columns) == list(mixed.columns)
    assert len(ground) == 1 and ground.n_points.iloc[0] == 100
    empty = compute_tile(src(xl + 1000, yl, np.full(100, 0.2)), 0, 25, 2, 1, cell=25, backend="numpy", options=opt)
    assert list(empty.columns) == list(mixed.columns) and len(empty) == 0


# ---------------------------------------------------------------- DTM
def _write_dtm(path, a, x0, y1, res, dtype, nodata=None):
    import rasterio
    from rasterio.transform import from_origin
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1], count=1, dtype=dtype,
                       transform=from_origin(x0, y1, res, res), nodata=nodata) as d:
        d.write(a.astype(dtype), 1)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("res", [1.0, 0.5, 2.0])
def test_dtm_on_its_own_pixel_grid(tmp_path, backend, res):
    import rasterio
    rng = np.random.default_rng(5)
    a = rng.uniform(0, 30, (40, 40))
    _write_dtm(tmp_path / "d.tif", a, 0.0, 30.0, res, "float64")
    x, y = rng.uniform(0.6, 9.4, 400), rng.uniform(20.6, 29.4, 400)
    zabs = rng.uniform(5, 60, 400)
    with rasterio.open(tmp_path / "d.tif") as s:
        g = np.array([v[0] for v in s.sample(zip(x, y))])
    opt = Options(trees=False, other_packages=False, **STRICT)
    got = compute_tile(src(x, y, zabs), 0.4, 29.6, 1, 1, cell=9.0, dtm=str(tmp_path / "d.tif"), backend=backend,
                       options=opt)
    want = compute_tile(src(x, y, zabs - g), 0.4, 29.6, 1, 1, cell=9.0, backend=backend, options=opt)
    same(got, want, [c for c in got.columns if c not in ("x", "y")])


def test_integer_dtm_without_nodata_outside_coverage(tmp_path):
    a = np.full((5, 5), 100, np.int16)
    _write_dtm(tmp_path / "d.tif", a, 0.0, 5.0, 1.0, "int16")
    opt = Options(trees=False, other_packages=False, **STRICT)
    # half of the points lie outside the DTM: they must be dropped, not normalised with a ground of 0
    x = np.r_[np.linspace(0.5, 4.5, 10), np.linspace(10.5, 14.5, 10)]
    y = np.full(20, 2.5)
    df = compute_tile(src(x, y, np.full(20, 110.0)), 0, 5, 3, 1, cell=5, dtm=str(tmp_path / "d.tif"),
                      backend="numpy", options=opt)
    assert df.n_points.sum() == 10 and np.allclose(df.zq20.dropna(), 10)


# ---------------------------------------------------------------- canopy hull and sparse clouds
def test_hull_vertices_exact():
    import shapely
    from lidarmetricpy.trees import _hull_vertices
    B = get_backend("numpy")
    rng = np.random.default_rng(2)
    t = 0.0889892693111186 + np.arange(80) * 2 * np.pi / 80
    circle = np.c_[0.25648522761476594 + 3 * np.cos(t), 0.12248230053439824 + 3 * np.sin(t)]
    for pts in (circle, rng.uniform(0, 100, (5000, 2)), rng.normal(0, 10, (20000, 2)),
                np.r_[circle, rng.uniform(-1, 1, (3000, 2))]):
        v = _hull_vertices(B, pts[:, 0].copy(), pts[:, 1].copy())
        h = shapely.MultiPoint(pts).convex_hull
        assert shapely.MultiPoint(v).convex_hull.equals(h)
        parts = [_hull_vertices(B, p[:, 0].copy(), p[:, 1].copy()) for p in np.array_split(pts, 7)]
        assert shapely.MultiPoint(np.concatenate(parts)).convex_hull.equals(h)


def test_hull_prefilter_drops_interior_points():
    """The hull of a 90 M point tile must not be computed from every point (0.3.0 development regression)."""
    from lidarmetricpy.trees import _hull_candidates
    rng = np.random.default_rng(4)
    for pts in (rng.uniform(0, 1000, (200_000, 2)), rng.normal(0, 50, (200_000, 2))):
        keep = _hull_candidates(get_backend("numpy"), pts[:, 0].copy(), pts[:, 1].copy())
        assert keep is not None and keep.mean() < 0.05


@pytest.mark.parametrize("n", [1, 2, 3])
def test_sparse_clouds_with_trees(n):
    x = np.linspace(1, 20, n)
    df = compute_tile(src(x, x, np.linspace(6, 12, n)), 0, 25, 1, 1, cell=25, backend="numpy",
                      options=Options(**STRICT))
    assert len(df) == 1


def test_spike_filter_same_on_host_and_device_paths():
    """Cells of 2.5 m: a 1 m column crosses two rows of cells; both paths must keep the same points."""
    from lidarmetricpy.api import _spike_filter, _spike_filter_dev
    grid = Grid(0.0, 5.0, 2, 2, 2.5)
    x = np.array([0.2, 0.2, 0.3, 1.2, 1.3, 1.25])
    y = np.array([2.4, 2.6, 2.55, 4.0, 4.1, 4.05])
    z = np.array([1.0, 20.0, 2.0, 1.0, 30.0, 1.5])
    host = np.zeros(len(z), bool)
    rows = np.minimum(np.floor((grid.y1 - y) / grid.cell).astype(int), grid.nrow - 1)
    for r in np.unique(rows):
        k = rows == r
        host[np.flatnonzero(k)] = _spike_filter(x[k], y[k], z[k], grid)
    B = get_backend("numpy")
    P = [x.copy(), y.copy(), z.copy(), np.ones(6, np.uint8), np.ones(6, np.uint8), np.arange(6, dtype=np.int32)]
    from lidarmetricpy.api import Options as O
    _spike_filter_dev(B, grid, O(), P)
    assert sorted(P[5].tolist()) == np.flatnonzero(host).tolist()


@pytest.mark.skipif(not _available("cupy"), reason="cupy not available")
def test_device_and_host_paths_fractional_cells():
    rng = np.random.default_rng(9)
    n = 20000
    x, y = rng.uniform(0, 10, n), rng.uniform(0, 10, n)
    z = np.where(rng.uniform(size=n) < 0.01, 45.0, rng.gamma(2, 4, n))
    s = src(x, y, z)
    a = compute_tile(s, 0, 10, 4, 4, cell=2.5, backend="cupy", options=Options(spike_filter=True))
    b = compute_tile(s, 0, 10, 4, 4, cell=2.5, backend="cupy", options=Options(spike_filter=True, device_prepare=False))
    assert a.attrs["stats"]["path"] == "device" and b.attrs["stats"]["path"] == "host"
    same(a, b, [c for c in a.columns])


# ---------------------------------------------------------------- strict comparison and budgets
def test_compare_tables_rejects_false_positives():
    ref = pd.DataFrame({"x": [0.5, 1.5, 2.5], "y": [0.5, 0.5, 0.5], "m": [1.0, np.inf, -np.inf]})
    ok = compare_tables(ref, ref.copy(), ["m"])
    assert ok.equal.all() and ok.n_finite.iloc[0] == 1
    for got in (ref.iloc[:1], ref.iloc[:0], ref.assign(m=[1.0, 0.0, -np.inf]), ref.assign(m=[1.0, -np.inf, -np.inf]),
                ref.assign(m=[1.0, np.inf, 100.0]), ref.assign(m=[np.nan, np.inf, -np.inf]),
                pd.concat([ref, pd.DataFrame({"x": [9.5], "y": [0.5], "m": [1.0]})])):
        assert not compare_tables(ref, got, ["m"]).equal.all()
    with pytest.raises(ValueError):
        compare_tables(pd.concat([ref, ref]), ref, ["m"])


def test_budget_below_minimum_is_refused():
    s = src([1, 2, 3], [1, 2, 3], [5, 6, 7])
    with pytest.raises(ValueError):
        compute_tile(s, 0, 25, 1, 1, backend="numpy", cpu_memory_gb=0.001)
