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
KDE_N = 512


def area_metrics(B, cell, z, rn, nr, vx, vy, ncell, zmin=ZMIN, bw=2.0, above=30.0):
    """Metrics from points above zmin. Returns dict name -> numpy array (ncell)."""
    out = {}
    # voxels use every point: lidRmetrics rounds z first (round_any) and keeps round(z) > zmin,
    # so a point at exactly z = 0.5 becomes a voxel at 1 m
    cell_all, z_all, vx_all, vy_all = cell, z, vx, vy
    m = z > zmin
    cell, zs, rns, nrs, vx, vy = cell[m], z[m], rn[m], nr[m], vx[m], vy[m]
    npts = int(cell.shape[0])
    if npts == 0:
        return {"n": np.zeros(ncell)}
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

    # entropy of 1 m layers
    # lidR::entropy: breaks 0..ceiling(zmax); a point exactly at an integer zmax falls outside (findInterval)
    nbin = B.astype(B.ceil(B.nan_to_num(zmax_c, nan=0.0)), B.i64)
    zb = B.astype(B.floor(zs), B.i64)
    inb = zb < nbin[cell]
    hb = B.bincount(cell[inb] * 61 + zb[inb], minlength=ncell * 61).reshape(ncell, 61)
    hbf = B.astype(hb, B.f)
    p = hbf / B.maximum(B.sum(hbf, 1, keepdims=True), 1)
    plogp = B.where(p > 0, p * B.log(B.where(p > 0, p, 1.0)), 0.0)
    Hs = -B.sum(plogp, 1)
    ent = Hs / B.log(B.astype(B.maximum(nbin, 2), B.f))
    out["zentropy"] = B.where((B.nan_to_num(zmax_c, nan=0.0) < 2) | ~has, B.nan, ent)

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
    out["zpcum7"], out["zpcum9"] = cum[:, 6], cum[:, 8]
    out["pz_above_30"] = B.segsum(B.astype(zs >= above, B.f), start, n) / B.maximum(nf, 1) * 100

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
    out["L4"], out["Lskew"], out["Lkurt"] = l4, l3 / l2, l4 / l2

    # leaf area density (lidR::LAD, dz = 1, k = 0.5, z0 = 2)
    z0 = B.where(B.nan_to_num(zmin_c, nan=0.0) > 2, B.floor(zmin_c - 2) + 2, 2.0)
    top = B.ceil(zmax_c - z0) + z0
    nbk = B.where(has & (z0 < zmax_c), B.astype(top - z0, B.i64) + 1, 0)
    idx = B.clip(B.astype(B.ceil(zs - z0[cell]), B.i64), 0, 61)
    hc = B.astype(B.bincount(cell * 62 + idx, minlength=ncell * 62).reshape(ncell, 62), B.f)
    cs_ = B.cumsum(hc, 1)
    with np.errstate(all="ignore"):
        gf = cs_[:, :-1] / cs_[:, 1:]
        lad = B.sdiv(-B.log(gf), 0.5)
    lad = B.where(B.isfinite(lad), lad, B.nan)
    lad = B.where(B.arange(61, dtype=B.i64)[None, :] >= (nbk - 1)[:, None], B.nan, lad)
    fin = B.isfinite(lad)
    cnt = B.astype(B.sum(B.astype(fin, B.i64), 1), B.f)
    lad0 = B.nan_to_num(lad, nan=0.0)
    lmean = B.sum(lad0, 1) / cnt
    lvar = B.sum(B.where(fin, (lad0 - lmean[:, None]) ** 2, 0.0), 1) / (cnt - 1)
    out["lad_max"] = B.where(cnt > 0, B.amax(B.where(fin, lad, -B.inf), 1), B.nan)
    out["lad_mean"] = B.where(cnt > 0, lmean, B.nan)
    out["lad_cv"] = B.sqrt(lvar) / lmean

    # echoes
    n_first = B.segsum(B.astype(rns == 1, B.f), start, n)
    n_last = B.segsum(B.astype((rns == nrs) & (rns > 1), B.f), start, n)
    n_single = B.segsum(B.astype(nrs == 1, B.f), start, n)
    out["p_single"] = n_single / nf * 100
    out["p_last"] = n_last / nf * 100
    out["ratio_last_first"] = B.where(n_first > 0, n_last / n_first, B.nan)

    # voxels (1 m) — Lefsky light classes and vertical voxel distribution
    vz = B.astype(B.round_away(z_all), B.i64)
    vk = B.astype(vz, B.f) > zmin
    VX = int(B.item(B.amax(vx_all))) + 2                      # voxel column offsets within a block
    VY = int(B.item(B.amax(vy_all))) + 2
    key = ((cell_all[vk] * VX + vx_all[vk]) * VY + vy_all[vk]) * 64 + vz[vk]
    key = B.sort(key)
    keep = B.concatenate([B.full(1, True, dtype=B.bool), key[1:] != key[:-1]])
    key = key[keep]
    vzz = key % 64
    colk = key // 64
    vc = colk // VY // VX
    # forestr::calc_enl on the filled voxels (z > zmin): 1 / sum over 1 m layers of (n_layer / n_filled)^2
    venl = B.to_numpy(B.bincount(vc * 64 + vzz, minlength=ncell * 64))
    cnt_enl = venl.reshape(ncell, 64).astype(np.float64)
    tot_enl = cnt_enl.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        enl = np.where(tot_enl > 0, 1 / ((cnt_enl / tot_enl[:, None]) ** 2).sum(1), np.nan)
    vzmin = B.scatter_min(ncell, vc, vzz, 10 ** 9)
    vzmax = B.scatter_max(ncell, vc, vzz, -1)
    ucol, inv = B.unique_inverse(colk)
    ncol_ = int(ucol.shape[0])
    col_cell = ucol // VY // VX
    col_zmax = B.scatter_max(ncol_, inv, vzz, -1)
    col_nfill = B.bincount(inv, minlength=ncol_)
    eu = B.bincount(inv[B.astype(vzz, B.f) >= 0.65 * B.astype(col_zmax[inv], B.f)], minlength=ncol_)
    open_ = B.bincount(col_cell, B.astype(vzmax[col_cell] - col_zmax, B.f), ncell)
    closed = B.bincount(col_cell, B.astype((col_zmax - vzmin[col_cell] + 1) - col_nfill, B.f), ncell)
    euph = B.bincount(col_cell, B.astype(eu, B.f), ncell)
    olig = B.bincount(col_cell, B.astype(col_nfill - eu, B.f), ncell)
    tot = open_ + closed + euph + olig
    # lidRmetrics: a class with no voxel is NA (left join), and all are NA when the voxel box has < 3 voxels
    vxmin = B.scatter_min(ncell, vc, (colk // VY) % VX, 10 ** 9)
    vxmax = B.scatter_max(ncell, vc, (colk // VY) % VX, -1)
    vymin = B.scatter_min(ncell, vc, colk % VY, 10 ** 9)
    vymax = B.scatter_max(ncell, vc, colk % VY, -1)
    nbox = (vxmax - vxmin + 1) * (vymax - vymin + 1) * (vzmax - vzmin + 1)
    few = (vzmax < 0) | (nbox <= 2)
    for name, v in (("OpenGapSpace", open_), ("ClosedGapSpace", closed), ("Euphotic", euph), ("Oligophotic", olig)):
        out[name] = B.where(few | (v == 0), B.nan, v / tot * 100)
    zr = B.where(vzmax >= vzmin, vzmax - vzmin, -1)
    hv = B.astype(B.bincount(vc * 62 + B.clip(vzz - vzmin[vc], 0, 61), minlength=ncell * 62).reshape(ncell, 62), B.f)
    F = hv[:, 1:] + 0
    F[:, 0] = F[:, 0] + hv[:, 0]                         # first class holds zmin and zmin + 1
    jj = B.arange(1, 62, dtype=B.i64)[None, :]
    vm = jj <= zr[:, None]
    Fmin = B.amin(B.where(vm, F, B.inf), 1)
    Fmax = B.amax(B.where(vm, F, -B.inf), 1)
    with np.errstate(all="ignore"):
        Fn = (F - Fmin[:, None]) / (Fmax - Fmin)[:, None]
    k_ = B.astype(B.sum(B.astype(vm, B.i64), 1), B.f)
    fmean = B.sum(B.where(vm, Fn, 0.0), 1) / k_
    fsd = B.sqrt(B.sum(B.where(vm, (Fn - fmean[:, None]) ** 2, 0.0), 1) / (k_ - 1))
    nfill = B.bincount(vc, minlength=ncell)
    bad = (zr < 1) | (Fmax == Fmin) | (k_ < 2) | (nfill <= 2)
    out["vzsd"] = B.where(bad, B.nan, fsd)
    out["vzcv"] = B.where(bad, B.nan, fsd / fmean * 100)

    # kernel density of heights: exact port of stats::density(z, bw = 2) + lidRmetrics:::get_peaks
    out["kde_peaks_count"] = kde_peaks(B, cell, zs, n, zmin_c, zmax_c, ncell, bw)
    out["n"] = nf
    res = {k: B.to_numpy(v).astype(np.float64) for k, v in out.items()}
    res["enl"] = enl
    return res


def kde_peaks(B, cell, zs, n, zmin_c, zmax_c, ncell, bw, nG=KDE_N):
    """Number of local maxima of R's density(z, bw) (n = 512, cut = 3, ext = 4), per cell.

    R: lo = min - 7 bw, up = max + 7 bw; linear binning of the points onto nG nodes
    (C_BinDist), circular FFT convolution with a Gaussian of 2 nG nodes, then linear
    interpolation onto 512 points between min - 3 bw and max + 3 bw.
    """
    sel = n > 2
    ids = B.astype(B.cumsum(B.astype(sel, B.i64), 0) - 1, B.i64)     # rank among selected cells
    nsel = int(B.item(B.sum(B.astype(sel, B.i64))))
    res = B.full(ncell, B.nan)
    if nsel == 0:
        return res
    nf = B.astype(n, B.f)
    frm, to = zmin_c - 3 * bw, zmax_c + 3 * bw
    lo, up = frm - 4 * bw, to + 4 * bw
    xd = B.sdiv(up - lo, float(nG - 1))
    pm = sel[cell]
    c, z = cell[pm], zs[pm]
    r = ids[c]
    xpos = (z - lo[c]) / xd[c]
    ix = B.astype(B.floor(xpos), B.i64)
    fx = xpos - B.astype(ix, B.f)
    w = 1.0 / nf[c]
    L = 2 * nG
    y = B.bincount(r * L + ix, w * (1 - fx), nsel * L) + B.bincount(r * L + ix + 1, w * fx, nsel * L)
    y = y.reshape(nsel, L)
    # kernel at node distances (multiples of xd), negative half mirrored as in R
    kk = B.astype(B.arange(L, dtype=B.i64), B.f)
    kk = B.where(kk > nG, kk - L, kk)                       # 0..nG, then -(nG-1)..-1
    xs = xd[sel]
    kern = B.sdiv(B.exp(-0.5 * B.sdiv(kk[None, :] * xs[:, None], bw) ** 2), bw * math.sqrt(2 * math.pi))
    conv = B.real(B.ifft(B.fft(y, 1) * B.conj(B.fft(kern, 1)), 1))[:, :nG]
    kords = B.maximum(conv, 0.0)
    # approx(xords, kords, xout)
    lo_s, frm_s, to_s = lo[sel], frm[sel], to[sel]
    t = B.astype(B.arange(KDE_N, dtype=B.i64), B.f)
    xout = frm_s[:, None] + t[None, :] * B.sdiv(to_s - frm_s, float(KDE_N - 1))[:, None]
    pos = (xout - lo_s[:, None]) / xs[:, None]
    i0 = B.clip(B.astype(B.floor(pos), B.i64), 0, nG - 2)
    f = pos - B.astype(i0, B.f)
    rows = B.arange(nsel, dtype=B.i64)[:, None]
    flat = kords.reshape(-1)
    yv = flat[rows * nG + i0] * (1 - f) + flat[rows * nG + i0 + 1] * f
    s = B.sign(B.diff(yv, axis=1))
    dd = B.diff(s, axis=1)
    cnt = B.astype(B.sum(B.astype(dd < 0, B.i64), 1), B.f)
    res[sel] = cnt
    return res


# ---------------------------------------------------------------- rumple
def rumple_pixels(B, cell, x, y, ncell):
    """1 m pixel (row, col) of every point for lidR::decimate_points(highest(1)) on the points of its cell.
    lidR snaps each cell's bbox with st_adjust_bbox: right edge round_any(xmax - 0.5) + 1, bottom edge
    round_any(ymin - 0.5), with C round(). A point lying exactly on that right / bottom edge goes to the
    last column / row. This happens when the cell's largest x is a negative integer, or its smallest y a
    positive integer (round() takes the half away from zero). Elsewhere the pixel is floor(x), floor(-y)."""
    xr = B.round_away(B.scatter_max(ncell, cell, x, -B.inf) - 0.5) + 1.0
    yb = B.round_away(B.scatter_min(ncell, cell, y, B.inf) - 0.5)
    pc = B.floor(x) - B.astype(x == xr[cell], x.dtype)
    pr = B.floor(-y) - B.astype(y == yb[cell], y.dtype)
    return pr, pc


def rumple_select(B, cell, z, px, order):
    """Indices of the highest point per 1 m pixel (lidR::decimate_points(highest(1)); first point wins ties).
    px: pixel ids in 0 .. npix-1. Grouped max with exact atomics instead of a three-key sort."""
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
