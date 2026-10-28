"""Individual trees, reproducing the lidR 4.3 workflow

    chm    <- rasterize_canopy(las, 0.5, p2r(subcircle = 0.15, na.fill = knnidw()))
    ttops  <- locate_trees(chm, lmf(ws = function(h) pmin(pmax(2.5 + 0.15 * h, 3), 8), hmin = 5))
    crowns <- dalponte2016(chm, ttops, th_tree = 2)()

and summarising trees per grid cell (by tree-top position).

The canopy raster and the local-maximum search run on the backend; the Dalponte
region growing is inherently sequential and runs on the CPU (numba).
"""
from __future__ import annotations

import math

import numpy as np

try:
    import numba
    _njit = numba.njit(cache=True)
except ImportError:  # pragma: no cover
    numba = None

    def _njit(f):
        return f

RES = 0.5
TREE_COLUMNS = ("tree_n", "tree_h_mean", "tree_h_sd", "tree_h_cv", "tree_h_max", "tree_h_gini", "tree_h_shannon",
                "crown_m2_mean", "crown_m2_sd")
SUBCIRCLE = 0.15
EPS = 1e-8                                           # lidR Shapes.h EPSILON


def r_round(x, digits):
    """R's round(x, digits) (R >= 4.0, fround.c): the closer of the two decimal neighbours of the
    double value, ties to even. numpy.round(x * 10^d) / 10^d differs in about 6 % of cases."""
    x = np.asarray(x, dtype=np.float64)
    sgn = np.where(x < 0, -1.0, 1.0)
    a = np.abs(x)
    p10 = 10.0 ** digits
    x10 = a * p10
    i10 = np.floor(x10)
    xd = i10 / p10
    xu = np.ceil(x10) / p10
    du, dd = xu - a, a - xd
    with np.errstate(invalid="ignore"):
        r = np.where((du < dd) | ((du == dd) & (np.fmod(i10, 2) == 1)), xu, xd)
    return sgn * r


def _round_any(v, acc):
    """lidR:::round_any = roundc(x / acc) * acc, with C round() (halves away from zero)."""
    q = v / acc
    t = np.trunc(q)
    return (t + np.sign(q) if abs(q - t) == 0.5 else np.rint(q)) * acc


class CanopyLayout:
    """lidR::raster_layout(las, res): bbox snapped with st_adjust_bbox."""

    def __init__(self, xmin, xmax, ymin, ymax, res=RES):
        self.res = res
        self.xmin = _round_any(xmin - 0.5 * res, res)
        self.xmax = _round_any(xmax - 0.5 * res, res) + res
        self.ymin = _round_any(ymin - 0.5 * res, res)
        self.ymax = _round_any(ymax - 0.5 * res, res) + res
        self.ncol = int(round((self.xmax - self.xmin) / res))
        self.nrow = int(round((self.ymax - self.ymin) / res))


class CanopyAccumulator:
    """p2r(subcircle) maximum raster, accumulated block by block (max is associative)."""

    def __init__(self, B, layout, subcircle=SUBCIRCLE):
        self.B, self.L, self.sc = B, layout, subcircle
        self.chm = B.full(layout.nrow * layout.ncol, -B.inf)
        a = np.arange(8) * 2 * np.pi / 8
        self.cs, self.ss = np.cos(a), np.sin(a)
        # extremes for the convex hull used by na.fill
        self.ext = []

    def add(self, x, y, z):
        """x, y, z: float64 numpy arrays (map coordinates, heights)."""
        B, L = self.B, self.L
        xres = (L.xmax - L.xmin) / L.ncol
        yres = (L.ymax - L.ymin) / L.nrow
        f64 = "float32" not in str(B.f)
        xd = B.asarray(x) if f64 else None
        yd = B.asarray(y) if f64 else None
        zd = B.asarray(z)
        for c, s in zip(self.cs, self.ss):
            if f64:
                xx = xd + self.sc * float(c)
                yy = yd + self.sc * float(s)
                col = B.astype(B.floor(B.sdiv(xx - L.xmin, xres)), B.i64)
                row = B.astype(B.floor(B.sdiv(L.ymax - yy, yres)), B.i64)
                row = B.where(yy == L.ymin, L.nrow - 1, row)
                col = B.where(xx == L.xmax, L.ncol - 1, col)
                del xx, yy
            else:
                # float32 devices (Apple MPS): pixel indices in float64 on the host
                xx = x + self.sc * float(c)
                yy = y + self.sc * float(s)
                col = np.floor((xx - L.xmin) / xres).astype(np.int64)
                row = np.floor((L.ymax - yy) / yres).astype(np.int64)
                row[yy == L.ymin] = L.nrow - 1
                col[xx == L.xmax] = L.ncol - 1
                col, row = B.asarray(col), B.asarray(row)
            ok = (row >= 0) & (row < L.nrow) & (col >= 0) & (col < L.ncol)
            cell = row[ok] * L.ncol + col[ok]
            self.chm = B.maximum(self.chm, B.scatter_max(L.nrow * L.ncol, cell, zd[ok], -B.inf))
        if f64:
            self._extremes_dev(xd, yd, x, y)
        else:
            self._extremes(x, y)

    def add_device(self, xd, yd, zd, x=None, y=None):
        """Same as add() with x, y, z already on a float64 device (avoids a second transfer).
        Temporaries are processed per subcircle offset, so memory stays within ~6 arrays of the block."""
        B, L = self.B, self.L
        xres = (L.xmax - L.xmin) / L.ncol
        yres = (L.ymax - L.ymin) / L.nrow
        for c, s in zip(self.cs, self.ss):
            xx = xd + self.sc * float(c)
            yy = yd + self.sc * float(s)
            col = B.astype(B.floor(B.sdiv(xx - L.xmin, xres)), B.i64)
            row = B.astype(B.floor(B.sdiv(L.ymax - yy, yres)), B.i64)
            row = B.where(yy == L.ymin, L.nrow - 1, row)
            col = B.where(xx == L.xmax, L.ncol - 1, col)
            del xx, yy
            ok = (row >= 0) & (row < L.nrow) & (col >= 0) & (col < L.ncol)
            cell = row[ok] * L.ncol + col[ok]
            del row, col
            self.chm = B.maximum(self.chm, B.scatter_max(L.nrow * L.ncol, cell, zd[ok], -B.inf))
        self._extremes_dev(xd, yd)

    def _extremes(self, x, y):
        """Convex-hull vertices of a block given on the host (float32 devices)."""
        from .backend import get_backend
        self.ext.append(_hull_vertices(get_backend("numpy"), np.asarray(x, np.float64), np.asarray(y, np.float64)))

    def _extremes_dev(self, xd, yd, x=None, y=None):
        """Convex-hull vertices of a block on a float64 device; the hull of all blocks' vertices is the hull
        of the whole tile (lidR st_convex_hull uses every point)."""
        self.ext.append(_hull_vertices(self.B, xd, yd))

    def finish(self, fill=True, k=10, p=2.0, rmax=50.0):
        """Return the canopy raster (nrow, ncol) as numpy float64, NA = nan, rounded to 3 decimals."""
        L = self.L
        chm = self.B.to_numpy(self.chm).astype(np.float64)
        chm[np.isinf(chm)] = np.nan
        chm = chm.reshape(L.nrow, L.ncol)
        if fill and np.isnan(chm).any():
            chm = _knnidw_fill(chm, L, np.concatenate(self.ext), k, p, rmax)
        return r_round(chm, 3)


def _hull_candidates(B, x, y):
    """Mask of the points that can be convex-hull vertices (Akl-Toussaint).

    The extreme points in the directions x, y, x + y and x - y span a convex polygon inside the hull; points
    strictly inside it (by more than 1e-6 m, far above the rounding error of the test) cannot be hull
    vertices. Returns None when the polygon is degenerate (keep every point)."""
    import shapely
    idx = []
    for kv in (x, y, x + y, x - y):
        idx.append(int(B.item(B.argmin(kv))))
        idx.append(int(B.item(B.argmax(kv))))
    ii = B.asarray(np.array(idx))
    P = np.column_stack([B.to_numpy(x[ii]), B.to_numpy(y[ii])]).astype(np.float64)
    poly = shapely.MultiPoint(P).convex_hull
    if not isinstance(poly, shapely.Polygon) or poly.area <= 0:
        return None
    V = np.asarray(poly.exterior.coords)[:-1]
    area2 = float(np.sum(V[:, 0] * np.roll(V[:, 1], -1) - np.roll(V[:, 0], -1) * V[:, 1]))
    if area2 < 0:                                                      # make it counter-clockwise
        V = V[::-1]
    inside = None
    for i in range(len(V)):
        ax, ay = float(V[i, 0]), float(V[i, 1])
        ex, ey = float(V[(i + 1) % len(V), 0]) - ax, float(V[(i + 1) % len(V), 1]) - ay
        d = (ex * (y - ay) - ey * (x - ax)) / math.hypot(ex, ey)       # distance to the left of the edge
        m = d > 1e-6
        inside = m if inside is None else (inside & m)
    return ~inside


def _hull_vertices(B, x, y):
    """Exact convex-hull vertices of the points (x, y backend arrays) as an (m, 2) numpy array."""
    import shapely
    n = int(x.shape[0])
    if n == 0:
        return np.zeros((0, 2))
    if n > 64:
        keep = _hull_candidates(B, x, y)
        if keep is not None:
            x, y = x[keep], y[keep]
    pts = np.column_stack([B.to_numpy(x), B.to_numpy(y)]).astype(np.float64)
    hull = shapely.MultiPoint(pts).convex_hull
    if isinstance(hull, shapely.Polygon):
        return np.asarray(hull.exterior.coords)[:-1]
    return np.asarray(shapely.get_coordinates(hull))                  # point or segment (degenerate)


def _knnidw_fill(chm, L, ext, k, p, rmax):
    """Fill empty cells inside the convex hull of the points (buffered by one cell) with lidR knnidw."""
    import shapely
    from .knnidw import knnidw
    if len(ext) == 0:
        return chm
    hull = shapely.MultiPoint(ext).convex_hull                     # polygon, or a segment / point if degenerate
    poly = shapely.buffer(hull, L.res, quad_segs=30)                # sf::st_buffer default nQuadSegs = 30
    rr, cc = np.nonzero(np.isnan(chm))
    xc = L.xmin + (cc + 0.5) * L.res
    yc = L.ymax - (rr + 0.5) * L.res
    inside = shapely.intersects_xy(poly, xc, yc)
    rr, cc, xc, yc = rr[inside], cc[inside], xc[inside], yc[inside]
    if len(rr) == 0:
        return chm
    gr, gc = np.nonzero(~np.isnan(chm))                  # raster order, as raster_as_las()
    gx = L.xmin + (gc + 0.5) * L.res
    gy = L.ymax - (gr + 0.5) * L.res
    out = chm.copy()
    out[rr, cc] = knnidw(gx, gy, chm[gr, gc], xc, yc, k=k, p=p, rmax=rmax)
    return out


def ws_default(h):
    return np.minimum(np.maximum(2.5 + 0.15 * h, 3.0), 8.0)


def local_maxima(B, chm, res=RES, hmin=5.0, ws=ws_default):
    """lidR lmf() on the canopy raster with a height-dependent circular window.

    Returns (row, col) of tree tops in lidR order (raster order, top-left first).
    Tie rule as lidR (C++ LAS::filter_local_maxima): a cell whose window holds an equal
    height already tagged as a maximum is not a maximum; earlier cells (raster order) win.
    """
    if "float32" in str(B.f):
        # heights are compared for equality (ties); keep the raster in float64 on the CPU
        from .backend import get_backend
        B = get_backend("numpy")
    nr, nc = chm.shape
    Z = np.where(np.isnan(chm), -np.inf, chm)
    wsv = ws(np.where(np.isfinite(Z), Z, 0.0))
    r2 = (wsv / 2) ** 2 + EPS
    rmax = float(np.max(wsv[Z >= hmin]) / 2) if (Z >= hmin).any() else 0.0
    kmax = int(np.ceil(rmax / res))
    offs = [(dy, dx) for dy in range(-kmax, kmax + 1) for dx in range(-kmax, kmax + 1)
            if (dy or dx) and (dy * res) ** 2 + (dx * res) ** 2 <= rmax ** 2 + EPS]
    Zd = B.asarray(Z)
    r2d = B.asarray(r2)
    pad = kmax
    Zp = B.full((nr + 2 * pad, nc + 2 * pad), -B.inf)
    Zp[pad:pad + nr, pad:pad + nc] = Zd
    higher = B.zeros((nr, nc), dtype=B.bool)
    for dy, dx in offs:
        d2 = (dy * res) ** 2 + (dx * res) ** 2
        sh = Zp[pad + dy:pad + dy + nr, pad + dx:pad + dx + nc]
        higher = higher | ((d2 <= r2d) & (sh > Zd))
    cand = (~higher) & (Zd >= hmin)
    # equal heights among candidates within the window
    Cp = B.zeros((nr + 2 * pad, nc + 2 * pad), dtype=B.bool)
    Cp[pad:pad + nr, pad:pad + nc] = cand
    tie = B.zeros((nr, nc), dtype=B.bool)
    for dy, dx in offs:
        d2 = (dy * res) ** 2 + (dx * res) ** 2
        sh = Zp[pad + dy:pad + dy + nr, pad + dx:pad + dx + nc]
        tie = tie | ((d2 <= r2d) & Cp[pad + dy:pad + dy + nr, pad + dx:pad + dx + nc] & (sh == Zd))
    cand = B.to_numpy(cand)
    tie = B.to_numpy(tie) & cand
    if tie.any():
        tr, tc = np.nonzero(tie)                   # raster order
        cand[tr, tc] = _resolve_ties(tr, tc, Z, r2, res)
    rr, cc = np.nonzero(cand)
    return rr, cc


@_njit
def _resolve_ties(tr, tc, Z, r2, res):
    n = len(tr)
    keep = np.zeros(n, np.bool_)
    for i in range(n):
        ok = True
        for j in range(i):
            if keep[j] and Z[tr[j], tc[j]] == Z[tr[i], tc[i]]:
                dy = (tr[j] - tr[i]) * res
                dx = (tc[j] - tc[i]) * res
                if dy * dy + dx * dx <= r2[tr[i], tc[i]]:
                    ok = False
                    break
        keep[i] = ok
    return keep


@_njit
def _dalponte(Image, Seeds, sx, sy, th_seed, th_crown, th_tree, DIST):
    """Exact port of lidR C_dalponte2016 (matrix indexed [x][y from bottom])."""
    nrow, ncol = Image.shape
    Region = Seeds.copy()
    Regiontemp = Seeds.copy()
    nseed = len(sx) - 1                              # sx[0] is unused (ids start at 1)
    sum_h = np.zeros(nseed + 1)
    npix = np.zeros(nseed + 1, np.int64)
    for i in range(1, nseed + 1):
        sum_h[i] = Image[sx[i], sy[i]]
        npix[i] = 1
    di = np.array([-1, 0, 0, 1])
    dk = np.array([0, -1, 1, 0])
    grown = True
    while grown:
        grown = False
        for r in range(1, nrow - 1):
            for k in range(1, ncol - 1):
                id_ = Region[r, k]
                if id_ != 0:
                    hSeed = Image[sx[id_], sy[id_]]
                    mh = sum_h[id_] / npix[id_]
                    for q in range(4):
                        pr = r + di[q]
                        pk = k + dk[q]
                        z = Image[pr, pk]
                        if z > th_tree:
                            if (z > hSeed * th_seed and z > mh * th_crown and z <= hSeed + hSeed * 0.05
                                    and abs(sx[id_] - pr) < DIST and abs(sy[id_] - pk) < DIST
                                    and Region[pr, pk] == 0):
                                Regiontemp[pr, pk] = id_
                                npix[id_] += 1
                                sum_h[id_] += Image[pr, pk]
                                grown = True
        Region[:, :] = Regiontemp
    return Region


def dalponte2016(chm, rr, cc, th_tree=2.0, th_seed=0.45, th_cr=0.55, max_cr=10):
    """Crown raster (nrow, ncol) with tree ids 1..n (0 = none), ids in the order of (rr, cc)."""
    nr, nc = chm.shape
    img = np.where(np.isnan(chm), -np.inf, chm)
    # lidR raster_as_matrix: z[x, y] with y from the bottom
    Image = np.ascontiguousarray(img[::-1, :].T)
    Seeds = np.zeros((nc, nr), np.int32)
    sx = np.zeros(len(rr) + 1, np.int64)
    sy = np.zeros(len(rr) + 1, np.int64)
    sx[1:], sy[1:] = cc, nr - 1 - rr
    Seeds[sx[1:], sy[1:]] = np.arange(1, len(rr) + 1)
    Region = _dalponte(Image, Seeds, sx, sy, th_seed, th_cr, th_tree, float(max_cr))
    return Region.T[::-1, :]


def tree_metrics(chm, layout, x0, y1, cell, nrow, ncol, B=None, rr=None, cc=None, crowns=True):
    """Trees per grid cell (by top position): n, height mean/sd/cv/max/Gini/Shannon (2 m), crown area mean/sd."""
    if rr is None:
        rr, cc = local_maxima(B, chm)
    n_t = len(rr)
    names = ["tree_n", "tree_h_mean", "tree_h_sd", "tree_h_cv", "tree_h_max", "tree_h_gini", "tree_h_shannon",
             "crown_m2_mean", "crown_m2_sd"]
    out = {k: np.full(nrow * ncol, np.nan) for k in names}
    out["tree_n"][:] = 0
    if n_t == 0:
        return out, (rr, cc, np.zeros((0,)))
    h = chm[rr, cc]
    tx = layout.xmin + (cc + 0.5) * layout.res
    ty = layout.ymax - (rr + 0.5) * layout.res
    ci = np.floor((tx - x0) / cell).astype(np.int64)
    ri = np.floor((y1 - ty) / cell).astype(np.int64)
    ok = (ci >= 0) & (ci < ncol) & (ri >= 0) & (ri < nrow)
    area = np.full(n_t, np.nan)
    if crowns:
        reg = dalponte2016(chm, rr, cc)
        cnt = np.bincount(reg.ravel(), minlength=n_t + 1)[1:]
        area = np.where(cnt > 0, cnt * layout.res ** 2, np.nan)
    cid = (ri * ncol + ci)[ok]
    df_ = {"cid": cid, "h": h[ok], "a": area[ok]}
    import pandas as pd
    d = pd.DataFrame(df_)

    def gini(v):
        v = np.sort(v)
        k = len(v)
        if k < 2:
            return np.nan
        return np.sum((2 * np.arange(1, k + 1) - k - 1) * v) / (k * v.sum())

    def shan(v):
        p = np.bincount(np.floor(v / 2).astype(int))
        p = p[p > 0] / len(v)
        return -np.sum(p * np.log(p))

    g = d.groupby("cid")
    res = pd.DataFrame({
        "tree_n": g.size(),
        "tree_h_mean": g.h.mean(),
        "tree_h_sd": g.h.std(),
        "tree_h_max": g.h.max(),
        "tree_h_gini": g.h.apply(gini),
        "tree_h_shannon": g.h.apply(shan),
        "crown_m2_mean": g.a.mean(),
        "crown_m2_sd": g.a.std(),
    })
    res["tree_h_cv"] = res.tree_h_sd / res.tree_h_mean
    for k in names:
        out[k][res.index.values] = res[k].values
    return out, (rr, cc, area)
