"""Area-based metrics per grid cell: all 111 outputs of lidRmetrics::metrics_set3 (v0.1.5), same definitions.

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
PROBS = np.concatenate([[0.01], 0.05 + np.arange(19) * 0.05, [0.99]])   # c(0.01, seq(0.05, 0.95, 0.05), 0.99)
PROB_NAMES = ["zq" + format(float(p) * 100, ".15g") for p in PROBS]
THRESHOLDS = (2.0, 5.0, 10.0, 20.0, 30.0)                                  # metrics_percabove(threshold)
ZINTERVALS = (0.0, 0.5, 2.0, 5.0, 10.0, 20.0, 30.0)                         # metrics_interval(zintervals)
INTERVAL_NAMES = (["pz_below_0"] + [f"pz_{format(a, 'g')}-{format(b, 'g')}" for a, b in
                                    zip(ZINTERVALS[:-1], ZINTERVALS[1:])] + ["pz_above_30"])
KDE_NAMES = (["kde_peaks_count"] + [f"kde_peak{i}_elev" for i in range(1, 5)] +
             [f"kde_peak{i}_value" for i in range(1, 5)] + [f"kde_peak{i}_diff" for i in range(1, 4)])
# the 111 outputs of lidRmetrics::metrics_set3 (v0.1.5) in its order, with the arguments of
# validation/ref_lidr.R (zmin = 0.5, threshold = c(2, 5, 10, 20, 30), zintervals = c(0, 0.5, 2, 5, 10, 20, 30),
# pixel_size = 1, vox_size = 1); 'enl' (forestr) is computed with the voxels
SET3_COLUMNS = tuple(
    ["n", "zmax", "zmin", "zmean", "zvar", "zsd", "zcv", "zskew", "zkurt"] + PROB_NAMES
    + ["pzabovemean"] + [f"pzabove{format(t, 'g')}" for t in THRESHOLDS]
    + ["ziqr", "zMADmean", "zMADmedian", "CRR", "zentropy"] + [f"zpcum{i}" for i in range(1, 10)]
    + ["L1", "L2", "L3", "L4", "Lskew", "Lkurt", "Lcoefvar"]
    + ["lad_min", "lad_max", "lad_mean", "lad_cv", "lad_sum"] + INTERVAL_NAMES
    + ["n_first", "n_intermediate", "n_last", "n_single", "n_multiple", "p_first", "p_intermidiate", "p_last",
       "p_single", "p_multiple", "ratio_last_first", "ratio_intermediate_first", "ratio_multiple_single"]
    + [f"n_return_{k}" for k in range(1, 5)] + ["rumple"]
    + ["vn", "vFRall", "vFRcanopy", "vzrumple", "vzsd", "vzcv", "OpenGapSpace", "ClosedGapSpace", "Euphotic",
       "Oligophotic"] + KDE_NAMES + ["HOME"])
AREA_COLUMNS = tuple(c for c in SET3_COLUMNS if c not in ("n", "rumple")) + ("enl",)


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
    vox_names = ("vn", "vFRall", "vFRcanopy", "vzrumple", "OpenGapSpace", "ClosedGapSpace", "Euphotic",
                 "Oligophotic", "vzsd", "vzcv")
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
        # metrics_voxels: vn filled voxels; vFRall = vn / voxels of the box; vFRcanopy = vn / voxels from the
        # box bottom up to the top filled voxel of each filled column (NA without any voxel: create_voxels NULL)
        nvox = B.bincount(vc, minlength=ncell)
        below = B.bincount(col_cell, B.astype(col_zmax - vzmin[col_cell] + 1, B.f), ncell)
        with np.errstate(all="ignore"):
            out["vn"] = B.where(filled, B.astype(nvox, B.f), B.nan)
            out["vFRall"] = B.where(filled, B.astype(nvox, B.f) / B.astype(nbox, B.f) * 100, B.nan)
            out["vFRcanopy"] = B.where(filled, B.astype(nvox, B.f) / below * 100, B.nan)
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
        # metrics_voxstructure: vzrumple = sqrt(sum((freq - brks[1:k])^2)) / k, brks = seq(min(Z), max(Z), 1)
        brk = B.astype(vzmin[:, None] + (jj - 1), B.f)
        with np.errstate(all="ignore"):
            dist = B.sqrt(_rsum(B, B.where(vm, (Fn - brk) ** 2, 0.0)))
            out["vzrumple"] = B.where(bad, B.nan, dist / B.astype(B.maximum(zr, 1), B.f))

    return out, enl


def _quantiles(B, zs, start, n, has, probs, npts):
    """stats::quantile(type = 7) of each cell's sorted values: (1 - h) x[lo] + h x[hi] where index > lo and
    x[hi] != x[lo] (R's formula), NA for empty cells."""
    nf = B.astype(n, B.f)
    last_i = B.minimum(B.maximum(start + n - 1, 0), npts - 1)
    out = []
    for p in probs:
        index = 1 + B.maximum(nf - 1, 0.0) * float(p)
        lo = B.floor(index)
        hi = B.ceil(index)
        i0 = B.minimum(start + B.astype(lo, B.i64) - 1, last_i)
        i1 = B.minimum(start + B.astype(hi, B.i64) - 1, last_i)
        x0, x1 = zs[B.maximum(i0, 0)], zs[B.maximum(i1, 0)]
        h = index - lo
        q = B.where((index > lo) & (x1 != x0), (1 - h) * x0 + h * x1, x0)
        out.append(B.where(has, q, B.nan))
    return out


def _seg_sorted(B, cell, v, ncell):
    """Sort values by (cell, value); returns sorted values, start, count."""
    o = B.lexsort([v, cell])
    vs = v[o]
    n = B.bincount(cell, minlength=ncell)
    start = B.concatenate([B.zeros(1, B.i64), B.cumsum(n, 0)[:-1]])
    return vs, start, n


def _home(B, cell, z, it, ncell):
    """lidRmetrics::metrics_HOME: z of the point where |cumsum(i) by increasing z - cumsum(i) by decreasing z|
    is smallest; both orders are stable (ties in file order) and the first minimum in decreasing order wins.
    cell, z, it: points above zmin in file order. Intensities are integers, so the sums are exact."""
    npts = int(cell.shape[0])
    idx = B.arange(npts, dtype=B.i64)
    n = B.bincount(cell, minlength=ncell)
    start = B.concatenate([B.zeros(1, B.i64), B.cumsum(n, 0)[:-1]])
    itf = B.astype(it, B.f)

    def segcum(o):
        c = B.cumsum(itf[o], 0)
        c0 = B.concatenate([B.zeros(1, B.f), c])[start]          # sum before each cell's first point
        return c - c0[cell[o]]

    oa = B.lexsort([idx, z, cell])                                # increasing z, file order
    od = B.lexsort([idx, -z, cell])                               # decreasing z, file order
    c1 = B.zeros(npts, dtype=B.f)
    c2 = B.zeros(npts, dtype=B.f)
    c1[oa] = segcum(oa)
    c2[od] = segcum(od)
    dd = B.abs(c1 - c2)[od]                                       # in decreasing order
    cd = cell[od]
    dmin = B.scatter_min(ncell, cd, dd, B.inf)
    pos = B.arange(npts, dtype=B.i64)
    big = 2 ** 62
    first = B.scatter_min(ncell, cd[dd == dmin[cd]], pos[dd == dmin[cd]], big)
    has = n > 0
    zd = z[od]
    return B.where(has, zd[B.minimum(first, npts - 1)], B.nan)


def area_metrics(B, cell, z, rn, nr, vx, vy, ncell, zmin=ZMIN, bw=2.0, it=None):
    """lidRmetrics::metrics_set3 (all 111 outputs but rumple, which needs x and y; see rumple_cells) and
    forestr's ENL per cell. cell, z, rn, nr, vx, vy, it: points in file order (it: intensity, 1 if None,
    as in tests/make_oracle.R). Returns dict name -> numpy array (ncell)."""
    out = {}
    vox, enl = _voxel_metrics(B, cell, z, vx, vy, ncell, zmin)     # rounded heights of every point
    m = z > zmin
    cell, zs, rns, nrs = cell[m], z[m], rn[m], nr[m]
    its = it[m] if it is not None else None
    npts = int(cell.shape[0])
    if npts == 0:
        res = {k: np.full(ncell, np.nan) for k in AREA_COLUMNS}
        res["n"] = np.full(ncell, np.nan)
        for k in INTERVAL_NAMES:                                 # metrics_interval: 0 without points
            res[k] = np.zeros(ncell)
        res.update({k: B.to_numpy(v).astype(np.float64) for k, v in vox.items()})
        res["enl"] = enl
        return res
    kde = _kde_peaks(B, cell, zs, ncell, bw)                      # R's density() in file order
    out["HOME"] = _home(B, cell, zs, its if its is not None else B.full(npts, 1.0, dtype=B.f), ncell)
    o = B.lexsort([zs, cell])
    cell, zs, rns, nrs = cell[o], zs[o], rns[o], nrs[o]
    n = B.bincount(cell, minlength=ncell)
    nf = B.astype(n, B.f)
    start = B.concatenate([B.zeros(1, B.i64), B.cumsum(n, 0)[:-1]])
    has = n > 0
    last_i = B.maximum(start + n - 1, 0)
    zmin_c = B.where(has, zs[B.minimum(start, npts - 1)], B.nan)
    zmax_c = B.where(has, zs[B.minimum(last_i, npts - 1)], B.nan)

    # metrics_basic
    s1 = B.segsum(zs, start, n)
    mean = s1 / nf
    zc = zs - mean[cell]                                  # two-pass variance, as R's var()
    s2 = B.segsum(zc * zc, start, n)
    var = s2 / (nf - 1)
    sd = B.sqrt(var)
    with np.errstate(all="ignore"):
        out["zmax"], out["zmin"], out["zmean"] = zmax_c, zmin_c, B.where(has, mean, B.nan)
        out["zvar"], out["zsd"] = B.where(has, var, B.nan), B.where(has, sd, B.nan)
        out["zcv"] = sd / mean * 100
        out["zskew"] = (B.segsum(zc * zc * zc, start, n) / nf) / (s2 / nf) ** 1.5
        out["zkurt"] = nf * B.segsum(zc * zc * zc * zc, start, n) / (s2 * s2)

    # metrics_percentiles and the quartiles of IQR()
    qs = _quantiles(B, zs, start, n, has, list(PROBS) + [0.25, 0.75, 0.5], npts)
    for name, q in zip(PROB_NAMES, qs):
        out[name] = q
    q25, q75, med = qs[-3], qs[-2], qs[-1]

    # metrics_percabove: lidR:::fast_countover counts z > threshold; n = 0 gives 0/0
    with np.errstate(all="ignore"):
        out["pzabovemean"] = B.segsum(B.astype(zs > mean[cell], B.f), start, n) / nf * 100
        for t in THRESHOLDS:
            out[f"pzabove{format(t, 'g')}"] = B.segsum(B.astype(zs > t, B.f), start, n) / nf * 100

    # metrics_dispersion
    out["ziqr"] = q75 - q25
    out["zMADmean"] = B.where(has, B.segsum(B.abs(zc), start, n) / nf, B.nan)
    dev, dstart, dn = _seg_sorted(B, cell, B.abs(zs - med[cell]), ncell)
    out["zMADmedian"] = _quantiles(B, dev, dstart, dn, has, [0.5], npts)[0]
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

    # metrics_canopydensity: cumulative proportions in 10 slices; breaks <- seq(zmin, zmax, (zmax - zmin) / 10)
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
    for i in range(9):
        out[f"zpcum{i + 1}"] = B.where(flat, 0.0, cum[:, i])

    # metrics_interval(zintervals): hist(right = FALSE, include.lowest = FALSE) shifts every break down by
    # 1e-7 * median(diff(breaks)) (breaks with -Inf / Inf: 7.5 here); 0 everywhere without points
    br = (-np.inf,) + ZINTERVALS + (np.inf,)
    dfs = np.diff(np.asarray(br))
    fuzz = 1e-07 * float(np.median(dfs))
    edges = [b - fuzz for b in br]
    for k, name in enumerate(INTERVAL_NAMES):
        sel = (zs >= edges[k]) & (zs < edges[k + 1])
        out[name] = B.where(has, B.segsum(B.astype(sel, B.f), start, n) / B.maximum(nf, 1) * 100, 0.0)

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
    out["L1"] = B.where(lm_ok, mean, B.nan)
    out["L2"] = B.where(lm_ok, l2, B.nan)
    out["L3"] = B.where(lm_ok, l3, B.nan)
    out["L4"] = B.where(lm_ok, l4, B.nan)
    out["Lskew"] = B.where(ratio_ok, l3 / l2, B.nan)
    out["Lkurt"] = B.where(ratio_ok, l4 / l2, B.nan)
    out["Lcoefvar"] = B.where(lm_ok, l2 / mean, B.nan)

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
        lsum = _rsum(B, lad0)
        lmean = lsum / cnt
        lvar = _rsum(B, B.where(fin, (lad0 - lmean[:, None]) ** 2, 0.0)) / (cnt - 1)
    # min / max(lad, na.rm = TRUE) are Inf / -Inf when every layer is NA; mean is NaN; sd needs two values
    out["lad_min"] = B.where(lad_ok, B.where(cnt > 0, B.amin(B.where(fin, lad, B.inf), 1), B.inf), B.nan)
    out["lad_max"] = B.where(lad_ok, B.where(cnt > 0, B.amax(B.where(fin, lad, -B.inf), 1), -B.inf), B.nan)
    out["lad_mean"] = B.where(lad_ok & (cnt > 0), lmean, B.nan)
    out["lad_cv"] = B.where(lad_ok & (cnt > 1), B.sqrt(lvar) / lmean, B.nan)
    # sum(lad, na.rm = TRUE) is 0 when LAD() returns no layer (n > 2 but z0 >= zmax), the others are NA
    out["lad_sum"] = B.where(lad_ok, lsum, B.where(has & (nf > 2), 0.0, B.nan))

    # metrics_echo: keeps ReturnNumber > 0, NumberOfReturns > 0 and ReturnNumber <= NumberOfReturns
    ok_r = (rns > 0) & (nrs > 0) & (rns <= nrs)

    def count(mask):
        return B.segsum(B.astype(mask, B.f), start, n)

    n_ok = count(ok_r)
    cnts = {"n_first": count(ok_r & (rns == 1)), "n_intermediate": count(ok_r & (rns > 1) & (rns < nrs)),
            "n_last": count(ok_r & (rns == nrs) & (rns > 1)), "n_single": count(ok_r & (nrs == 1)),
            "n_multiple": count(ok_r & (nrs > 1))}
    with np.errstate(all="ignore"):
        for k, v in cnts.items():
            out[k] = B.where(has, v, B.nan)
        for k, nm in (("first", "p_first"), ("intermediate", "p_intermidiate"), ("last", "p_last"),
                      ("single", "p_single"), ("multiple", "p_multiple")):
            out[nm] = B.where(has, cnts["n_" + k] / n_ok * 100, B.nan)
        out["ratio_last_first"] = B.where(cnts["n_first"] > 0, cnts["n_last"] / cnts["n_first"], B.nan)
        out["ratio_intermediate_first"] = B.where(cnts["n_first"] > 0, cnts["n_intermediate"] / cnts["n_first"],
                                                  B.nan)
        out["ratio_multiple_single"] = B.where(cnts["n_single"] > 0, cnts["n_multiple"] / cnts["n_single"], B.nan)
    # metrics_echo2: as.data.frame(table(ReturnNumber)) and then as.numeric() of the factor, which gives the
    # level codes 1, 2, ... of the return numbers present, not the numbers: n_return_k is the count of the k-th
    # smallest return number present (the count of ReturnNumber == k only when 1..k are all present, as usual);
    # merged with KeepReturns = 1:4 -> NA beyond the number of distinct return numbers
    RMAX = int(B.item(B.amax(rns))) + 1
    tab = B.astype(B.bincount(B.astype(cell, B.i64) * RMAX + B.astype(rns, B.i64), minlength=ncell * RMAX)
                   .reshape(ncell, RMAX), B.f)
    present = tab > 0
    rank = B.cumsum(B.astype(present, B.i64), 1)                 # level code of each present return number
    for k in range(1, 5):
        sel = present & (rank == k)
        out[f"n_return_{k}"] = B.where(B.sum(B.astype(sel, B.i64), 1) > 0, B.sum(B.where(sel, tab, 0.0), 1), B.nan)

    for name, v in zip(KDE_NAMES, kde):
        out[name] = v
    out.update(vox)
    out["n"] = B.where(has, nf, B.nan)
    res = {k: B.to_numpy(v).astype(np.float64) for k, v in out.items()}
    res["enl"] = enl
    return res


def _kde_peaks(B, cell, z, ncell, bw, chunk=8_000_000):
    """metrics_kde per cell (count, elevations, values and differences of up to 4 peaks) with the bit-exact
    port of R's density() (rdensity.py) on the host. cell, z: the points above zmin in file order.
    Cells are sent to the host in chunks of ~chunk points. Returns a list of 12 arrays (KDE_NAMES)."""
    from .rdensity import kde_peaks
    npts = int(cell.shape[0])
    o = B.lexsort([B.arange(npts, dtype=B.i64), cell])          # by cell, file order kept within a cell
    n = B.to_numpy(B.bincount(cell, minlength=ncell)).astype(np.int64)
    start = np.concatenate([[0], np.cumsum(n)[:-1]])
    out = np.full((ncell, len(KDE_NAMES)), np.nan)
    c0 = 0
    while c0 < ncell:
        c1 = c0 + 1
        while c1 < ncell and start[c1] + n[c1] - start[c0] <= chunk:
            c1 += 1
        a, b = int(start[c0]), int(start[c1 - 1] + n[c1 - 1])
        zc = B.to_numpy(z[o[a:b]]).astype(np.float64)
        out[c0:c1] = kde_peaks(zc, start[c0:c1] - a, n[c0:c1], bw)
        c0 = c1
    return [B.asarray(out[:, k].copy()) for k in range(out.shape[1])]


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
