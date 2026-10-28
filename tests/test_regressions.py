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


@pytest.mark.parametrize("backend", BACKENDS)
def test_voxels_of_points_at_zmin_do_not_depend_on_neighbours(backend):
    """Points at z = 0.5: no point above zmin, but round(0.5) = 1 gives voxels (lidRmetrics create_voxels)."""
    rng = np.random.default_rng(1)
    xl, yl = rng.uniform(1, 24, 4), rng.uniform(1, 24, 4)
    xr, yr = rng.uniform(26, 49, 50), rng.uniform(1, 24, 50)
    opt = Options.lidr(trees=False)
    one = compute_tile(src(xl, yl, np.full(4, 0.5)), 0, 25, 1, 1, cell=25, backend=backend, options=opt)
    both = compute_tile(src(np.r_[xl, xr], np.r_[yl, yr], np.r_[np.full(4, 0.5), rng.uniform(5, 20, 50)]),
                        0, 25, 2, 1, cell=25, backend=backend, options=opt)
    assert np.isnan(one.n.iloc[0]) and one.Euphotic.iloc[0] == 100 and one.enl.iloc[0] == 1
    same(one, both.iloc[:1].reset_index(drop=True), [c for c in one.columns])


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


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("res", [0.1, 0.2, 0.25])
def test_dtm_pixel_on_boundaries_independent_of_grid_origin(tmp_path, backend, res):
    """Points on DTM pixel edges take the pixel floor((x - x_dtm) / res) of the DTM's own grid, whatever the
    origin of the metric grid (0.3.0 used the window origin and could take the neighbouring pixel)."""
    nr = nc = 40
    a = 100 + 0.1 * np.arange(nc)[None, :] + 0.01 * np.arange(nr)[:, None]
    _write_dtm(tmp_path / "d.tif", a, 0.0, nr * res, res, "float64")
    k = np.arange(4, 30)
    x = k * res                                        # every x on a column edge
    y = nr * res - (k[::-1] * res)                     # every y on a row edge
    zabs = 110.0 + np.arange(len(k))
    col = np.floor((x - 0.0) / res).astype(int)        # independent reference, as lidR
    row = np.floor((nr * res - y) / res).astype(int)
    want = compute_tile(src(x, y, zabs - a[row, col]), 0.0, nr * res, 1, 1, cell=nr * res, backend=backend,
                        options=Options.lidr(trees=False, other_packages=False))
    for x0 in (0.0, res, 0.05, -0.3):
        n = int(np.ceil((nr * res - x0) / (nr * res)))
        got = compute_tile(src(x, y, zabs), x0, nr * res, n + 1, 1, cell=nr * res, dtm=str(tmp_path / "d.tif"),
                           backend=backend, options=Options.lidr(trees=False, other_packages=False))
        assert got.n_points.sum() == len(k)
        g = got[got.n_points > 0]
        if len(g) == 1:
            same(g.reset_index(drop=True), want, ["n_points", "n", "zq20", "zcv", "CRR", "L4"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_dtm_right_and_bottom_edges_take_the_last_pixel(tmp_path, backend):
    """A point exactly on the DTM's right / bottom edge takes the last column / row (lidR raster_cell_from_xy);
    0.3.1 dropped it. Ground values from lidR 4.3.3 normalize_height on the same 0.5 x 1 m DTM (R 4.5.3)."""
    import rasterio
    from rasterio.transform import from_origin
    with rasterio.open(tmp_path / "d.tif", "w", driver="GTiff", width=3, height=2, count=1, dtype="float64",
                       transform=from_origin(0, 2, 0.5, 1)) as ds:
        ds.write(np.array([[100., 101, 102], [103, 104, 105]]), 1)
    x = np.array([1.5, 1.5, 0.25, 1.25, 1.5, 0.25, 0.75, 0, 1.0, 0.5])      # right, bottom, corner, top, left, inside
    y = np.array([0.5, 1.5, 0, 0, 0, 0.5, 2, 1, 1, 0])
    ground_r = np.array([105, 102, 103, 105, 105, 103, 101, 103, 105, 104.])
    xo, yo = np.array([1.501, 0.75, -0.001, 0.75]), np.array([1, -0.001, 1, 2.001])     # outside the DTM: dropped
    got = compute_tile(src(np.r_[x, xo], np.r_[y, yo], np.r_[ground_r + 10, np.full(4, 200.0)]), -1, 3, 1, 1,
                       cell=4, dtm=str(tmp_path / "d.tif"), backend=backend,
                       options=Options.lidr(trees=False, other_packages=False))
    assert got.n_points.sum() == len(x)
    assert got.zq20.iloc[0] == 10 and got.zcv.iloc[0] == 0             # every point took the R pixel
    # the review's case: 2 x 2 DTM of 1 m, points on the right and bottom edges
    with rasterio.open(tmp_path / "e.tif", "w", driver="GTiff", width=2, height=2, count=1, dtype="float64",
                       transform=from_origin(0, 2, 1, 1)) as ds:
        ds.write(np.full((2, 2), 100.), 1)
    got = compute_tile(src([2, 2, .5, 1.5], [.5, 1.5, 0, 0], [110, 111, 112, 113.]), 0, 3, 1, 1, cell=4,
                       dtm=str(tmp_path / "e.tif"), backend=backend,
                       options=Options.lidr(trees=False, other_packages=False))
    assert got.n_points.sum() == 4 and np.isclose(got.zq20.iloc[0], 10.6)


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


@pytest.mark.skipif(not _available("cupy"), reason="cupy not available")
def test_device_path_reads_laz_within_ram_budget(monkeypatch):
    """The LAZ decoding chunk follows cpu_memory_gb on the GPU path too."""
    import laspy
    from pathlib import Path
    seen = []
    orig = laspy.LasReader.chunk_iterator

    def spy(self, n):
        seen.append(n)
        return orig(self, n)

    monkeypatch.setattr(laspy.LasReader, "chunk_iterator", spy)
    laz = str(Path(__file__).resolve().parent / "data" / "synthetic.laz")
    df = compute_tile(laz, -62000.0, -57750.0, 4, 4, backend="cupy", gpu_memory_gb=8, cpu_memory_gb=0.008)
    assert df.attrs["stats"]["path"] == "device"
    assert seen and max(seen) <= int(0.25 * 0.008 * 1024 ** 3 / 64) + 1


@pytest.mark.parametrize("kind", ["laz", "dict"])
def test_device_path_transfers_within_gpu_chunk(kind):
    """On the device path every array sent to the device while reading is at most the GPU chunk
    (options.chunk_points), even when the RAM budget allows larger reading chunks; 0.3.1 sent whole reading
    chunks. Run with a NumPy-backed device so that it needs no GPU; results do not depend on the chunk."""
    import laspy
    from pathlib import Path
    from lidarmetricpy.api import Grid, _device_tile
    from lidarmetricpy.backend import NumpyLike
    laz = Path(__file__).resolve().parent / "data" / "synthetic.laz"
    source = str(laz)
    if kind == "dict":
        las = laspy.read(laz)
        source = dict(x=np.asarray(las.x), y=np.asarray(las.y), z=np.asarray(las.z),
                      return_number=np.asarray(las.return_number), number_of_returns=np.asarray(las.number_of_returns))
    grid = Grid(-62000.0, -57750.0, 4, 4, 25.0)

    def run(chunk):
        B = NumpyLike(np, "device-emulation")
        B.device = "cuda"
        sizes, reading = [], [True]
        asarray, free = B.asarray, B.free

        def spy(a, *args, **kw):
            if reading[0]:
                sizes.append(int(np.size(a)))
            return asarray(a, *args, **kw)

        def stop():
            reading[0] = False
            return free()

        B.asarray, B.free = spy, stop
        T = {"read": 0.0, "prepare": 0.0, "area": 0.0, "canopy": 0.0, "trees": 0.0}
        frames, _, n = _device_tile(source, grid, Options(trees=False, other_packages=False, chunk_points=chunk), B,
                                    None, T, 10 ** 7, read_points=4 * chunk)
        return sizes, pd.concat(frames).sort_values(["x", "y"]).reset_index(drop=True), n

    sizes, small, n = run(1000)
    assert n > 5000 and sizes and max(sizes) <= 1000
    _, big, n2 = run(10 ** 7)
    assert n2 == n
    same(small, big, [c for c in big.columns if c not in ("x", "y")])


# ---------------------------------------------------------------- strict comparison and budgets
def test_compare_tables_rejects_false_positives():
    ref = pd.DataFrame({"x": [0.5, 1.5, 2.5], "y": [0.5, 0.5, 0.5], "m": [1.0, np.inf, -np.inf]})
    ok = compare_tables(ref, ref.copy(), ["m"])
    assert ok.equal.all() and ok.n_finite.iloc[0] == 1 and ok.status.iloc[0] == "equal"
    for got in (ref.iloc[:1], ref.iloc[:0], ref.assign(m=[1.0, 0.0, -np.inf]), ref.assign(m=[1.0, -np.inf, -np.inf]),
                ref.assign(m=[1.0, np.inf, 100.0]), ref.assign(m=[np.nan, np.inf, -np.inf]),
                pd.concat([ref, pd.DataFrame({"x": [9.5], "y": [0.5], "m": [1.0]})])):
        assert not compare_tables(ref, got, ["m"]).equal.all()
    with pytest.raises(ValueError):
        compare_tables(pd.concat([ref, ref]), ref, ["m"])
    # an NA metric in a cell with points is not an empty cell: the missing row must fail
    na = pd.DataFrame({"x": [0.5], "y": [0.5], "lad_cv": [np.nan]})
    assert not compare_tables(na, na.iloc[:0], ["lad_cv"]).equal.all()
    occ = pd.DataFrame({"x": [0.5], "y": [0.5]})
    assert not compare_tables(na, na.iloc[:0], ["lad_cv"], occupied=occ).equal.all()
    assert compare_tables(na, na.copy(), ["lad_cv"], occupied=occ).equal.all()
    # a reference cell outside `occupied` (no point) may be absent; a result row there may not
    two = pd.DataFrame({"x": [0.5, 1.5], "y": [0.5, 0.5], "m": [1.0, np.nan]})
    assert compare_tables(two, two.iloc[:1], ["m"], occupied=occ).equal.all()
    assert not compare_tables(two, two, ["m"], occupied=occ).equal.all()
    # strings are errors, not NA; two empty tables are "empty", not equal
    with pytest.raises(ValueError):
        compare_tables(na, na.assign(lad_cv=["broken"]), ["lad_cv"])
    e = compare_tables(na.iloc[:0], na.iloc[:0], ["lad_cv"])
    assert e.status.iloc[0] == "empty" and not e.equal.iloc[0]


def test_budget_below_minimum_is_refused():
    s = src([1, 2, 3], [1, 2, 3], [5, 6, 7])
    with pytest.raises(ValueError):
        compute_tile(s, 0, 25, 1, 1, backend="numpy", cpu_memory_gb=0.001)
