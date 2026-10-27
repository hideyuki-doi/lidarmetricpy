"""canopyLazR 2.0 (Kamoske et al. 2019, Forest Ecology and Management 433: 364-375) reproduced in numpy.

canopyLazR works on elevations (not heights above ground): every column of voxels takes its lowest
point as the ground, and its grid and outlier limits come from the whole input file. This module
therefore takes the raw points of one file (tile) and returns rasters on canopyLazR's own grid:

    laz.to.array(voxel.resolution, z.resolution, use.classified.returns = FALSE)
    canopy.height.levelr -> machorn.lad(beer.lambert.constant = k)
    rugosity.within.canopy, canopy.porosity.filled.ratio, lad.ht.max      (ht.cut)
    array.to.ground.and.canopy.rasters -> toc.rugosity

Details kept from the R code: Tukey's 1.5 IQR filter on z with R's type-7 quantiles; bins assigned
with round() (halves to even); points whose x or y index falls outside the levels are dropped (factor NA);
sd() with n - 1; round(..., 4) where the package rounds; the 3 x 3 windows skip the outer ring.
"""
from __future__ import annotations

import math

import numpy as np

from .trees import r_round


def _r_quartiles7(x):
    """R quantile(x, c(0.25, 0.75), type = 7): (1 - h) * x[lo] + h * x[hi], order statistics by partition."""
    n = len(x)
    idx = [1 + max(n - 1, 0) * p for p in (0.25, 0.75)]
    ks = sorted({math.floor(i) - 1 for i in idx} | {math.ceil(i) - 1 for i in idx})
    part = np.partition(x, ks)
    out = []
    for index in idx:
        lo, hi = math.floor(index), math.ceil(index)
        qs = part[lo - 1]
        h = index - lo
        if index > lo and part[hi - 1] != qs:
            qs = (1 - h) * qs + h * part[hi - 1]
        out.append(qs)
    return out


def _r_sd(v):
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return np.nan
    return float(np.sqrt(np.sum((v - v.mean()) ** 2) / (len(v) - 1)))


def canopylazr(x, y, z, xy_res=10.0, z_res=1.0, ht_cut=5, k=None):
    """Returns dict with 'x', 'y' (column centres, north-up order) and one array per output raster."""
    x, y, z = (np.asarray(a, np.float64) for a in (x, y, z))
    # Tukey outlier removal on z (summary(): type-7 quartiles)
    q1, q3 = _r_quartiles7(z)
    lo_, hi_ = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    keep = (z >= lo_) & (z <= hi_)
    x, y, z = x[keep], y[keep], z[keep]
    g, zg = float(xy_res), float(z_res)
    xr = (math.floor(x.min() / g) * g, math.ceil(x.max() / g) * g)
    yr = (math.floor(y.min() / g) * g, math.ceil(y.max() / g) * g)
    zr = (math.floor(z.min() / zg) * zg, math.ceil(z.max() / zg) * zg)
    nxb = int(round((xr[1] - xr[0]) / g)) + 1                 # length(x.bin)
    nyb = int(round((yr[1] - yr[0]) / g)) + 1
    L = int(round((zr[1] - zr[0]) / zg)) + 1                  # length(z.bin)
    nx, ny = nxb - 1, nyb - 1
    xc = np.rint((x - xr[0] + g / 2) / g).astype(np.int64)    # round(): halves to even
    yc = np.rint((y - yr[0] + g / 2) / g).astype(np.int64)
    zc = np.rint((z - zr[0] + zg / 2) / zg).astype(np.int64)
    ok = (xc >= 1) & (xc <= nx) & (yc >= 1) & (yc <= ny)       # other indices are not factor levels -> NA
    xc, yc, zc, z = xc[ok], yc[ok], zc[ok], z[ok]
    if zc.max() > L - 1:
        raise ValueError("a point falls in z bin length(z.bin); canopyLazR's array would be misaligned")
    col = (xc - 1) * ny + (yc - 1)                             # level order: x first, then y
    ncol_ = nx * ny
    ground = np.full(ncol_, np.inf)
    np.minimum.at(ground, col, z)
    canopy = np.full(ncol_, -np.inf)
    np.maximum.at(canopy, col, z)
    has = np.isfinite(ground)
    ground[~has] = np.nan
    canopy[~has] = np.nan
    counts = np.bincount(col * L + zc, minlength=ncol_ * L).reshape(ncol_, L).astype(np.float64)

    # canopy.height.levelr: shift every column down to its lowest return; ground must be > 0
    valid = has & (ground > 0)
    lev_counts = np.zeros((ncol_, L))
    cht = np.full(ncol_, np.nan)
    for i in np.flatnonzero(valid):
        r = counts[i]
        first = int(np.flatnonzero(r > 0)[0])
        lev_counts[i, :L - first] = r[first:]
        cht[i] = canopy[i] - ground[i]
    lev_counts[~valid] = 0.0                                   # NA columns become 0 in machorn.lad

    # machorn.lad
    accum = np.cumsum(lev_counts[:, ::-1], axis=1)[:, ::-1]    # pulses at or above each voxel
    allp = accum[:, :1]
    through = allp - accum
    shots_in = np.empty_like(through)
    shots_in[:, :-1] = through[:, 1:]
    shots_in[:, -1] = accum[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        rlad = np.log(shots_in / through) * (1 / zg) if k is None else np.log(shots_in / through) * (1 / (k * zg))
    rlad[~np.isfinite(rlad)] = np.nan
    for i in np.flatnonzero(np.isfinite(cht)):
        na_cut = math.ceil(cht[i]) + 2                         # 1-based in the R code
        if na_cut <= L:
            rlad[i, na_cut - 1:] = np.nan
    canopy_col = rlad[:, ht_cut:]                              # rLAD[(ht.cut + 1):nz]

    sd_lad = np.array([r_round(_r_sd(c), 4) if not np.isnan(_r_sd(c)) else np.nan for c in canopy_col])
    with np.errstate(invalid="ignore"):
        filled = np.nansum(canopy_col > 0, axis=1).astype(np.float64)
        poros = np.sum(canopy_col == 0, axis=1).astype(np.float64)
        filled_ratio = r_round(filled / (filled + poros), 4)
        poros_ratio = r_round(poros / (filled + poros), 4)
    max_lad = np.full(ncol_, np.nan)
    max_ht = np.full(ncol_, np.nan)
    for i in range(ncol_):
        c = canopy_col[i]
        if np.all(np.isnan(c)):
            continue
        m = np.nanmax(c)
        idx = np.flatnonzero(c == m) + 1 + (ht_cut - 1)
        max_lad[i] = m
        max_ht[i] = idx.max() if m > 0 else idx.min()

    # matrices [y index, x index] (row 1 = south), then flipped to north-up
    def mat(v):
        return v.reshape(nx, ny).T

    sdm = mat(sd_lad)
    rug = np.full((ny, nx), np.nan)
    for r in range(1, ny - 1):
        for c in range(1, nx - 1):
            rug[r, c] = _r_sd(sdm[r - 1:r + 2, c - 1:c + 2].ravel())
    chm = mat(canopy - ground)
    toc = np.full((ny, nx), np.nan)
    for r in range(1, ny - 1):
        for c in range(1, nx - 1):
            w = chm[r - 1:r + 2, c - 1:c + 2].ravel()
            mx = np.max(w)                                      # NA if any NA, then sum(na.rm) = 0
            toc[r, c] = np.nansum(mx - w) * g * g * zg if np.isfinite(mx) else 0.0
    out = {"clz_chm": chm, "clz_ground": mat(ground), "clz_vertical_sd_lad": sdm, "clz_rugosity": rug,
           "clz_filled_ratio": mat(filled_ratio), "clz_porosity_ratio": mat(poros_ratio),
           "clz_max_lad": mat(max_lad), "clz_max_lad_ht": mat(max_ht), "clz_toc_rugosity": toc}
    out = {k_: v[::-1, :].ravel() for k_, v in out.items()}
    rr, cc = np.mgrid[0:ny, 0:nx]
    out["x"] = (xr[0] + (cc + 0.5) * g).ravel()
    out["y"] = (yr[1] - (rr + 0.5) * g).ravel()
    return out


def canopylazr_file(path, xy_res=10.0, z_res=1.0, ht_cut=5, k=None):
    """canopylazr() on all points of a LAS/LAZ file (elevations, as canopyLazR reads them with rlas)."""
    import laspy
    import pandas as pd
    las = laspy.read(path)
    return pd.DataFrame(canopylazr(las.x, las.y, las.z, xy_res, z_res, ht_cut, k))
