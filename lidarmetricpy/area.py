"""Area-based metrics per grid cell, same definitions as lidRmetrics::metrics_set3 (v0.1.5).

All points of a block of cells are processed at once with sort / bincount on the
backend. Inputs are backend arrays prepared on the host (integer indices are
computed in float64 on the host so that cell membership is exact on every
backend):

    cell : int64, block-local cell index (0 .. ncell-1)
    z    : float, height above ground
    rn, nr : int64, ReturnNumber / NumberOfReturns
    vx, vy : int64, 1 m voxel column indices (round(x), round(y) as in lidRmetrics)
    px   : int64, 1 m pixel index (lidR::highest(1) grid) for rumple
    order: int64, point order in the file (ties are broken like lidR: first point wins)

Rumple needs a Delaunay triangulation per cell, which runs on the CPU (scipy/qhull,
the same library lidR uses through the 'geometry' package).
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ZMIN = 0.5
# median(diff(c(-Inf, 0, 0.5, 2, 5, 10, 20, 30, Inf))) of the zintervals used with lidRmetrics (validation/ref_lidr.R)
PZ_MEDIAN_DIFF = 7.5
AREA_COLUMNS = ("zq20", "zcv", "CRR", "zentropy", "zpcum7", "zpcum9", "pz_above_30", "L4", "Lskew", "Lkurt",
                "lad_max", "lad_mean", "lad_cv", "p_single", "p_last", "ratio_last_first", "OpenGapSpace",
                "ClosedGapSpace", "Euphotic", "Oligophotic", "vzsd", "vzcv", "kde_peaks_count", "enl")


def _rsum(B, a):
    """Row sums of a small (cells × height layers) matrix, added left to right on the host with numpy.
    Unlike a pairwise or parallel sum, the result depends neither on how many zero layers follow (the
    height range of the block) nor on the backend."""
    h = np.cumsum(B.to_numpy(a).astype(np.float64), 1)[:, -1] if a.shape[1] else np.zeros(a.shape[0])
    return B.asarray(h.astype(B.to_numpy(a).dtype, copy=False))


def _voxel_metrics(B, cell_all, z_all, vx_all, vy_all, ncell, zmin):
    """Voxel metrics (metrics_voxels) and forestr ENL from the rounded heights of every point.
    lidRmetrics::create_voxels rounds z first and then keeps round(z) > zmin, so a point at z = 0.5 is a voxel
    at 1 m even though z > zmin is false: these metrics do not depend on the points above zmin."""
    out = {}
    # voxels (1 m) — Lefsky light classes and vertical voxel distribution
    vz = B.astype(B.round_away(z_all), B.i64)
    vk = B.astype(vz, B.f) > zmin
    vox_names = ("OpenGapSpace", "ClosedGapSpace", "Euphotic", "Oligophotic", "vzsd", "vzcv")
    enl = np.full(ncell, np.nan)
    if not bool(B.item(B.sum(B.astype(vk, B.i64)) > 0)):       # no rounded voxel above zmin
        for name in vox_names:
            out[name] = B.full(ncell, B.nan)
    else:
        VZ0 = int(B.item(B.amin(vz[vk])))                        # key radix from the heights present
        VZ = int(B.item(B.amax(vz[vk]))) - VZ0 + 1
        VX = int(B.item(B.amax(vx_all))) + 2                      # voxel column offsets within a block
        VY = int(B.item(B.amax(vy_all))) + 2
        key = ((cell_all[vk] * VX + vx_all[vk]) * VY + vy_all[vk]) * VZ + (vz[vk] - VZ0)
        key = B.sort(key)
        keep = B.concatenate([B.full(1, True, dtype=B.bool), key[1:] != key[:-1]])
        key = key[keep]
        vzz = key % VZ + VZ0                                      # voxel height (m)
        colk = key // VZ
        vc = colk // VY // VX
        # forestr::calc_enl on the filled voxels (z > zmin): 1 / sum over 1 m layers of (n_layer / n_filled)^2
        venl = B.to_numpy(B.bincount(vc * VZ + (vzz - VZ0), minlength=ncell * VZ))
        cnt_enl = venl.reshape(ncell, VZ).astype(np.float64)
        tot_enl = np.cumsum(cnt_enl, 1)[:, -1]
        with np.errstate(divide="ignore", invalid="ignore"):
            enl = np.where(tot_enl > 0, 1 / np.cumsum((cnt_enl / tot_enl[:, None]) ** 2, 1)[:, -1], np.nan)
        big = 2 ** 40
        vzmin = B.scatter_min(ncell, vc, vzz, big)
        vzmax = B.scatter_max(ncell, vc, vzz, -big)
        ucol, inv = B.unique_inverse(colk)
        ncol_ = int(ucol.shape[0])
        col_cell = ucol // VY // VX
        col_zmax = B.scatter_max(ncol_, inv, vzz, -big)
        col_nfill = B.bincount(inv, minlength=ncol_)
        eu = B.bincount(inv[B.astype(vzz, B.f) >= 0.65 * B.astype(col_zmax[inv], B.f)], minlength=ncol_)
        open_ = B.bincount(col_cell, B.astype(vzmax[col_cell] - col_zmax, B.f), ncell)
        closed = B.bincount(col_cell, B.astype((col_zmax - vzmin[col_cell] + 1) - col_nfill, B.f), ncell)
        euph = B.bincount(col_cell, B.astype(eu, B.f), ncell)
        olig = B.bincount(col_cell, B.astype(col_nfill - eu, B.f), ncell)
        tot = open_ + closed + euph + olig
        # lidRmetrics: a class with no voxel is NA (left join), and all are NA when the voxel box has < 3 voxels
        vxmin = B.scatter_min(ncell, vc, (colk // VY) % VX, big)
        vxmax = B.scatter_max(ncell, vc, (colk // VY) % VX, -big)
        vymin = B.scatter_min(ncell, vc, colk % VY, big)
        vymax = B.scatter_max(ncell, vc, colk % VY, -big)
        filled = vzmax >= vzmin
        nbox = (vxmax - vxmin + 1) * (vymax - vymin + 1) * (vzmax - vzmin + 1)
        few = ~filled | (nbox <= 2)
        with np.errstate(all="ignore"):
            for name, v in (("OpenGapSpace", open_), ("ClosedGapSpace", closed), ("Euphotic", euph),
                            ("Oligophotic", olig)):
                out[name] = B.where(few | (v == 0), B.nan, v / tot * 100)
        zr = B.where(filled, vzmax - vzmin, -1)
        NZ = int(B.item(B.amax(zr))) + 2
        hv = B.astype(B.bincount(vc * NZ + B.clip(vzz - vzmin[vc], 0, NZ - 1), minlength=ncell * NZ)
                      .reshape(ncell, NZ), B.f)
        F = hv[:, 1:] + 0
        F[:, 0] = F[:, 0] + hv[:, 0]                         # first class holds zmin and zmin + 1
        jj = B.arange(1, NZ, dtype=B.i64)[None, :]
        vm = jj <= zr[:, None]
        Fmin = B.amin(B.where(vm, F, B.inf), 1)
        Fmax = B.amax(B.where(vm, F, -B.inf), 1)
        with np.errstate(all="ignore"):
            Fn = (F - Fmin[:, None]) / (Fmax - Fmin)[:, None]
            k_ = _rsum(B, B.astype(vm, B.f))
            fmean = _rsum(B, B.where(vm, Fn, 0.0)) / k_
            fsd = B.sqrt(_rsum(B, B.where(vm, (Fn - fmean[:, None]) ** 2, 0.0)) / (k_ - 1))
        nfill = B.bincount(vc, minlength=ncell)
        bad = (zr < 1) | (Fmax == Fmin) | (k_ < 2) | (nfill <= 2)
        out["vzsd"] = B.where(bad, B.nan, fsd)
        out["vzcv"] = B.where(bad, B.nan, fsd / fmean * 100)

    return out, enl


def area_metrics(B, cell, z, rn, nr, vx, vy, ncell, zmin=ZMIN, bw=2.0, above=30.0):
    """Metrics from points above zmin. Returns dict name -> numpy array (ncell)."""
    out = {}
    vox, enl = _voxel_metrics(B, cell, z, vx, vy, ncell, zmin)     # rounded heights of every point
    m = z > zmin
    cell, zs, rns, nrs, vx, vy = cell[m], z[m], rn[m], nr[m], vx[m], vy[m]
    npts = int(cell.shape[0])
    if npts == 0:
        res = {k: np.full(ncell, np.nan) for k in AREA_COLUMNS}
        res["n"] = np.full(ncell, np.nan)                   # metrics_basic: NA without points above zmin
        res["pz_above_30"] = np.zeros(ncell)                # metrics_interval: 0 without points
        res.update({k: B.to_numpy(v).astype(np.float64) for k, v in vox.items()})
        res["enl"] = enl
        return res
    # kde_peaks_count: R's density() of each cell's heights in file order (the order lidR passes them)
    kde = _kde_peaks(B, cell, zs, ncell, bw)
    o = B.lexsort([zs, cell])
    cell, zs, rns, nrs, vx, vy = cell[o], zs[o], rns[o], nrs[o], vx[o], vy[o]
    n = B.bincount(cell, minlength=ncell)
    nf = B.astype(n, B.f)
    start = B.concatenate([B.zeros(1, B.i64), B.cumsum(n, 0)[:-1]])
    has = n > 0
    last_i = B.maximum(start + n - 1, 0)
    zmin_c = B.where(has, zs[B.minimum(start, npts - 1)], B.nan)
    zmax_c = B.where(has, zs[B.minimum(last_i, npts - 1)], B.nan)

    # percentile 20 (type 7)
    h = (nf - 1) * 0.2
    lo = B.astype(B.floor(h), B.i64)
    i0 = B.minimum(start + lo, npts - 1)
    i1 = B.minimum(i0 + 1, B.minimum(last_i, npts - 1))
    out["zq20"] = B.where(has, zs[i0] + (h - B.astype(lo, B.f)) * (zs[i1] - zs[i0]), B.nan)
    s1 = B.segsum(zs, start, n)
    mean = s1 / nf
    zc = zs - mean[cell]                                  # two-pass variance, as R's var()
    sd = B.sqrt(B.segsum(zc * zc, start, n) / (nf - 1))
    out["zcv"] = sd / mean * 100
    out["CRR"] = (mean - zmin_c) / (zmax_c - zmin_c)

    # entropy of 1 m layers: metrics_dispersion passes z1 = z[z > 0] to lidR::entropy (NA when zmax < 2);
    # breaks 0..ceiling(zmax); a point exactly at an integer zmax falls outside (findInterval)
    nbin = B.astype(B.ceil(B.maximum(B.nan_to_num(zmax_c, nan=0.0), 0.0)), B.i64)
    NB = int(B.item(B.amax(nbin))) + 1
    zb = B.astype(B.floor(zs), B.i64)
    inb = (zs > 0) & (zb < nbin[cell])
    hb = B.bincount(cell[inb] * NB + zb[inb], minlength=ncell * NB).reshape(ncell, NB)
    hbf = B.astype(hb, B.f)
    p = hbf / B.maximum(_rsum(B, hbf)[:, None], 1)
    plogp = B.where(p > 0, p * B.log(B.where(p > 0, p, 1.0)), 0.0)
    Hs = -_rsum(B, plogp)
    ent = Hs / B.log(B.astype(B.maximum(nbin, 2), B.f))
    counted = _rsum(B, hbf) > 0                     # all points at an integer zmax fall outside: 0/0 -> NA
    out["zentropy"] = B.where((B.nan_to_num(zmax_c, nan=0.0) < 2) | ~has | ~counted, B.nan, ent)

    # cumulative proportions in 10 height slices between zmin and zmax
    # lidRmetrics::metrics_canopydensity: breaks <- seq(zmin, zmax, (zmax - zmin) / 10); findInterval
    rng = zmax_c - zmin_c
    by = B.sdiv(rng, 10.0)
    nbr = B.floor(rng / by + 1e-10)                       # seq(): n = floor((to - from) / by + 1e-10)
    zc0, byc, ztop = zmin_c[cell], by[cell], zmax_c[cell]
    kf = B.floor((zs - zc0) / byc) + 1
    # exact against R's breaks: from + (0:n) * by, clipped by pmin(x, to)
    kf = B.where(zs < B.minimum(zc0 + (kf - 1) * byc, ztop), kf - 1, kf)
    kf = B.where(zs >= B.minimum(zc0 + kf * byc, ztop), kf + 1, kf)
    kf = B.minimum(kf, nbr[cell] + 1)
    kk = B.astype(B.nan_to_num(kf, nan=0.0), B.i64)
    valid = (zs > zc0) & (kk >= 1) & (kk <= 10)
    hk = B.astype(B.bincount(cell[valid] * 10 + (kk[valid] - 1), minlength=ncell * 10).reshape(ncell, 10), B.f)
    cum = B.cumsum(hk, 1) / B.sum(hk, 1, keepdims=True) * 100
    flat = has & (zmax_c <= zmin_c)                       # metrics_canopydensity: max <= min -> all 0
    out["zpcum7"] = B.where(flat, 0.0, cum[:, 6])
    out["zpcum9"] = B.where(flat, 0.0, cum[:, 8])
    # metrics_interval(zintervals = c(0, 0.5, 2, 5, 10, 20, 30)): hist(right = FALSE) shifts every break down
    # by 1e-7 * median(diff(breaks)) = 7.5e-7, so the last class is z >= 30 - 7.5e-7; 0 without points
    thr = above + (-(1e-07 * PZ_MEDIAN_DIFF))
    out["pz_above_30"] = B.segsum(B.astype(zs >= thr, B.f), start, n) / B.maximum(nf, 1) * 100

    # L-moments (unbiased probability-weighted moments)
    j = B.astype(B.arange(npts, dtype=B.i64) - start[cell] + 1, B.f)
    nn = nf[cell]
    w1 = (j - 1) / (nn - 1)
    w2 = w1 * (j - 2) / (nn - 2)
    w3 = w2 * (j - 3) / (nn - 3)
    # L2..L4 are shift invariant: use centred heights (smaller rounding error, matters in float32)
    b0 = B.segsum(zc, start, n) / nf
    b1 = B.segsum(w1 * zc, start, n) / nf
    b2 = B.segsum(w2 * zc, start, n) / nf
    b3 = B.segsum(w3 * zc, start, n) / nf
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0
    lm_ok = nf >= 4                           # Lmoments() returns min(n, 4) moments; metrics_Lmoments needs 4
    ratio_ok = lm_ok & (zmax_c > zmin_c)      # equal heights: L2 = 0 and R's ratios are rounding noise -> NA
    out["L4"] = B.where(lm_ok, l4, B.nan)
    out["Lskew"] = B.where(ratio_ok, l3 / l2, B.nan)
    out["Lkurt"] = B.where(ratio_ok, l4 / l2, B.nan)

    # leaf area density: lidRmetrics::metrics_lad (n > 2) -> lidR::LAD(dz = 1, k = 0.5, z0 = 2)
    # -> gap_fraction_profile: hist(z, breaks = c(-Inf, seq(z0, top, 1))) with R's boundary fuzz
    z0 = B.where(B.nan_to_num(zmin_c, nan=0.0) > 2, B.floor(zmin_c - 2) + 2, 2.0)
    top = B.ceil(zmax_c - z0) + z0
    lad_ok = has & (z0 < zmax_c) & (nf > 2)
    nbk = B.where(lad_ok, B.astype(top - z0, B.i64) + 1, 0)      # length of seq(z0, top, 1)
    # hist.default: diddle = 1e-7 * (median(diff(breaks)) if nB > 5, diff(range(x)) if nB <= 3, else the
    # smallest positive diff); here nB = length(breaks) = nbk + 1 and every finite diff is 1
    dd = B.where(nbk + 1 <= 3, 1e-07 * (zmax_c - zmin_c), 1e-07)
    zb0, ddc = z0[cell], dd[cell]
    f0 = B.floor(zs - zb0)
    bidx = B.where(f0 < 0, 0.0, f0 + B.astype((zb0 + f0) + ddc < zs, B.f))   # breaks below x (fuzzy)
    NL = max(2, int(B.item(B.amax(nbk))) + 1)
    bidx = B.minimum(B.astype(bidx, B.i64), B.maximum(nbk[cell] - 1, 0))
    hc = B.astype(B.bincount(cell * NL + bidx, minlength=ncell * NL).reshape(ncell, NL), B.f)
    cs_ = B.cumsum(hc, 1)
    with np.errstate(all="ignore"):
        gf = cs_[:, :-1] / cs_[:, 1:]
        lad = B.sdiv(-B.log(gf), 0.5)
    lad = B.where(B.isfinite(lad), lad, B.nan)                   # 0/0 -> 0 -> Inf -> NA; Inf -> NA
    lad = B.where(B.arange(NL - 1, dtype=B.i64)[None, :] >= (nbk - 1)[:, None], B.nan, lad)
    fin = B.isfinite(lad)
    cnt = _rsum(B, B.astype(fin, B.f))
    lad0 = B.nan_to_num(lad, nan=0.0)
    with np.errstate(all="ignore"):
        lmean = _rsum(B, lad0) / cnt
        lvar = _rsum(B, B.where(fin, (lad0 - lmean[:, None]) ** 2, 0.0)) / (cnt - 1)
    # max(lad, na.rm = TRUE) is -Inf when every layer is NA; mean is NaN; sd needs two values
    out["lad_max"] = B.where(lad_ok, B.where(cnt > 0, B.amax(B.where(fin, lad, -B.inf), 1), -B.inf), B.nan)
    out["lad_mean"] = B.where(lad_ok & (cnt > 0), lmean, B.nan)
    out["lad_cv"] = B.where(lad_ok & (cnt > 1), B.sqrt(lvar) / lmean, B.nan)

    # echoes
    # metrics_echo keeps ReturnNumber > 0, NumberOfReturns > 0 and ReturnNumber <= NumberOfReturns
    ok_r = (rns > 0) & (nrs > 0) & (rns <= nrs)
    n_ok = B.segsum(B.astype(ok_r, B.f), start, n)
    n_first = B.segsum(B.astype(ok_r & (rns == 1), B.f), start, n)
    n_last = B.segsum(B.astype(ok_r & (rns == nrs) & (rns > 1), B.f), start, n)
    n_single = B.segsum(B.astype(ok_r & (nrs == 1), B.f), start, n)
    with np.errstate(all="ignore"):
        out["p_single"] = B.where(has, n_single / n_ok * 100, B.nan)
        out["p_last"] = B.where(has, n_last / n_ok * 100, B.nan)
    out["ratio_last_first"] = B.where(n_first > 0, n_last / n_first, B.nan)

    out["kde_peaks_count"] = kde
    out.update(vox)
    out["n"] = B.where(has, nf, B.nan)
    out["pz_above_30"] = B.where(has, out["pz_above_30"], 0.0)
    res = {k: B.to_numpy(v).astype(np.float64) for k, v in out.items()}
    res["enl"] = enl
    return res


def _kde_peaks(B, cell, z, ncell, bw, chunk=8_000_000):
    """kde_peaks_count per cell with the bit-exact port of R's density() (rdensity.py) on the host.
    cell, z: the points above zmin in file order. Cells are sent to the host in chunks of ~chunk points."""
    from .rdensity import kde_peaks_count
    npts = int(cell.shape[0])
    o = B.lexsort([B.arange(npts, dtype=B.i64), cell])          # by cell, file order kept within a cell
    n = B.to_numpy(B.bincount(cell, minlength=ncell)).astype(np.int64)
    start = np.concatenate([[0], np.cumsum(n)[:-1]])
    out = np.full(ncell, np.nan)
    c0 = 0
    while c0 < ncell:
        c1 = c0 + 1
        while c1 < ncell and start[c1] + n[c1] - start[c0] <= chunk:
            c1 += 1
        a, b = int(start[c0]), int(start[c1 - 1] + n[c1 - 1])
        zc = B.to_numpy(z[o[a:b]]).astype(np.float64)
        out[c0:c1] = kde_peaks_count(zc, start[c0:c1] - a, n[c0:c1], bw)
        c0 = c1
    return B.asarray(out)


# ---------------------------------------------------------------- rumple
def rumple_pixels(B, cell, x, y, ncell):
    """Pixel id of every point for lidR::decimate_points(highest(1)) applied to the points of its own cell.

    The id is unique per (cell, 1 m pixel), so neighbouring cells never compete for a pixel even when the
    grid is not aligned with whole metres. lidR snaps each cell's bbox with st_adjust_bbox: right edge
    round_any(xmax - 0.5) + 1, bottom edge round_any(ymin - 0.5), with C round(). A point lying exactly on
    that right / bottom edge goes to the last column / row; this happens when the cell's largest x is a
    negative integer, or its smallest y a positive integer. Elsewhere the pixel is floor(x), floor(-y)."""
    i64 = B.i64
    xr = B.round_away(B.scatter_max(ncell, cell, x, -B.inf) - 0.5) + 1.0
    yb = B.round_away(B.scatter_min(ncell, cell, y, B.inf) - 0.5)
    pc = B.astype(B.floor(x) - B.astype(x == xr[cell], x.dtype), i64)
    pr = B.astype(B.floor(-y) - B.astype(y == yb[cell], y.dtype), i64)
    big = 2 ** 62
    pr = pr - B.scatter_min(ncell, cell, pr, big)[cell]          # pixel offsets within the cell
    pc = pc - B.scatter_min(ncell, cell, pc, big)[cell]
    W = int(B.item(B.maximum(B.amax(pr), B.amax(pc)))) + 1
    return (cell * W + pr) * W + pc


def rumple_select(B, cell, z, px, order):
    """Indices of the highest point per (cell, 1 m pixel) (lidR::decimate_points(highest(1)) on one cell;
    the first point in file order wins ties). px: ids from rumple_pixels. Grouped max with exact atomics."""
    npix = int(B.item(B.amax(px))) + 1
    zmax = B.scatter_max(npix, px, z, -B.inf)
    cand = z == zmax[px]
    big = 2 ** 62
    order = B.astype(order, B.i64)
    omin = B.scatter_min(npix, px[cand], order[cand], big)
    first = cand & (order == omin[px])
    return B.to_numpy(B.arange(int(z.shape[0]), dtype=B.i64)[first])


def rumple_cells(cell, x, y, z, ncell, threads=8):
    """lidR::rumple_index per cell on CPU: Delaunay (qhull 'QbB Qt') of the decimated points,
    sum of 3D triangle areas / sum of projected areas. x, y: float64 global coordinates."""
    from scipy.spatial import Delaunay

    out = np.full(ncell, np.nan)
    o = np.argsort(cell, kind="stable")
    cs = cell[o]
    bounds = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1], True])

    def one(k):
        a, b = bounds[k], bounds[k + 1]
        if b - a <= 3:
            return cs[a], np.nan
        ii = o[a:b]
        P = np.column_stack([x[ii], y[ii], z[ii]])
        try:
            tri = Delaunay(P[:, :2], qhull_options="QbB Qt").simplices
        except Exception:  # noqa: BLE001  (qhull error → NA, as lidR)
            return cs[a], np.nan
        A, Bp, C = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
        nrm = np.cross(A - Bp, A - C)
        area = np.abs(0.5 * np.sqrt((nrm ** 2).sum(1))).sum()
        parea = np.abs(0.5 * nrm[:, 2]).sum()
        return cs[a], area / parea

    with ThreadPoolExecutor(threads) as ex:
        for c, v in ex.map(one, range(len(bounds) - 1)):
            out[c] = v
    return out
