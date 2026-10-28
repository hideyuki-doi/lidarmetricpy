"""Metrics defined by other packages, reproduced per grid cell.

leafR 0.3.5 (Almeida et al. 2019)          lad.voxels / lad.profile / lai / LAHV / FHD / GS / GC,
                                           applied to the points of one cell (grain.size = cell, k = 1)
PyForestScan 0.3.6 (JOSS 2025)           assign_voxels / calculate_pad / calculate_pai /
                                           calculate_canopy_cover / calculate_fhd, one cell = one column
forestr 2.0.2 (Atkins et al. 2018)         calc_enl on the filled 1 m voxels of lidRmetrics (see area.py)
ForestGapR 0.1.7 (Silva et al. 2019)       getForestGaps on the lidR canopy raster, summarised per cell
                                           (see gap_metrics)

The heavy part (counting points per cell and 1 m layer, rank sums for the Gini coefficient) runs on
the backend; the per-cell formulas are evaluated on small (cells x layers) matrices on the host with
the same operations as the original R / Python code, including their NA rules.
"""
from __future__ import annotations

import numpy as np

NS = 64                                    # 1 m layers 0..63 (heights are < 60 m after filtering)


def profile_counts(B, cell, z, ncell):
    """Counts per cell and 1 m layer for leafR and PyForestScan, plus leafR GC sums (all on the backend)."""
    out = {}
    i64 = B.i64
    # leafR: Z[Z < 0] <- 0; slice = as.integer(Z) (truncation)
    zl = B.maximum(z, 0.0)
    sl = B.clip(B.astype(B.floor(zl), i64), 0, NS - 1)
    out["leafr_counts"] = B.bincount(cell * NS + sl, minlength=ncell * NS).reshape(ncell, NS)
    # PyForestScan: points with HeightAboveGround >= 0; z edges arange(0, zmax + 1, 1) per cell
    m = z >= 0
    cp, zp = cell[m], z[m]
    zmax = B.scatter_max(ncell, cp, zp, -B.inf)
    nz = B.ceil(zmax)                         # number of 1 m bins = len(arange(0, zmax + 1)) - 1
    b = B.astype(B.floor(zp), i64)
    nzp = B.astype(nz[cp], i64)
    b = B.where(b >= nzp, nzp - 1, b)         # zmax exactly on an edge falls in the last (closed) bin
    b = B.clip(b, 0, NS - 1)
    out["pfs_counts"] = B.bincount(cp * NS + b, minlength=ncell * NS).reshape(ncell, NS)
    out["pfs_nz"] = nz
    # leafR GC: Gini of z >= 1 (-drop_z_below 1), G = 2 sum(x * 1:n) / sum(x) - (n + 1), GC = G / (n - 1)
    g = z >= 1
    cg, zg = cell[g], z[g]
    o = B.lexsort([zg, cg])
    cg, zg = cg[o], zg[o]
    n = B.bincount(cg, minlength=ncell)
    start = B.concatenate([B.zeros(1, i64), B.cumsum(n, 0)[:-1]])
    rank = B.astype(B.arange(int(cg.shape[0]), dtype=i64) - start[cg] + 1, B.f)
    out["gc_n"] = n
    out["gc_sx"] = B.segsum(zg * rank, start, n)
    out["gc_s"] = B.segsum(zg, start, n)
    return {k: B.to_numpy(v) for k, v in out.items()}


def _leafr(counts):
    """leafR lad.voxels -> lad.profile -> lai / LAHV / FHD / GS for every cell (row of counts)."""
    nc = counts.shape[0]
    res = {k: np.full(nc, np.nan) for k in ("leafr_lai", "leafr_lahv", "leafr_lahv_laiw", "leafr_lahv_hw",
                                            "leafr_lahv_laiw_hw", "leafr_fhd", "leafr_fhd_even", "leafr_gs",
                                            "leafr_gs_even")}
    for i in range(nc):
        c = counts[i].astype(np.float64)
        if c.sum() == 0:
            continue
        maxz = int(np.flatnonzero(c)[-1])
        if maxz < 2:
            # fewer than two LAD layers: lad.voxels drops to a vector and lad.profile fails in leafR -> NA
            continue
        cum = np.cumsum(c[:maxz + 1])         # points in slices <= s
        s = np.arange(1, maxz + 1)
        pin = cum[s]
        pout = cum[s - 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            lad = np.log(pin / pout) * 1.0 * 1.0
        lad[~np.isfinite(lad)] = np.nan       # Inf / NaN -> NA
        h = s + 0.5
        # lai(min = 1, max = 100): sum(lad[1:100], na.rm = TRUE)
        res["leafr_lai"][i] = np.nansum(lad[:100])
        if not np.isnan(lad).any():
            sl = lad.sum()
            res["leafr_lahv"][i] = np.sum(h * lad)
            res["leafr_lahv_laiw"][i] = np.sum(h * lad) / sl if sl != 0 else np.nan
            res["leafr_lahv_hw"][i] = np.sum(h * lad) / h.max()
            res["leafr_lahv_laiw_hw"][i] = (np.sum(h * lad) / sl) / h.max() if sl != 0 else np.nan
            # relative profile: lad / sum(lad) * 100 (NA as soon as one layer is NA)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = lad / sl * 100
            thr = 1 / len(h)
            keep = rel[rel >= thr] / 100
            nk = len(keep)
            with np.errstate(divide="ignore", invalid="ignore"):
                fhd = -np.sum(keep * np.log(keep))
                gs = 1 - np.sum(keep ** 2)
                res["leafr_fhd"][i] = fhd if np.isfinite(sl) and sl != 0 else np.nan
                res["leafr_fhd_even"][i] = fhd / np.log(nk) if sl != 0 else np.nan
                res["leafr_gs"][i] = gs if sl != 0 else np.nan
                res["leafr_gs_even"][i] = gs / (1 - 1 / nk) if sl != 0 else np.nan
    return res


def _pyforestscan(counts, nz):
    """PyForestScan calculate_pad / calculate_pai / calculate_canopy_cover / calculate_fhd per column."""
    from scipy.stats import entropy
    ncell = counts.shape[0]
    res = {k: np.full(ncell, np.nan) for k in ("pfs_pai", "pfs_cover", "pfs_fhd")}
    for i in range(ncell):
        if not np.isfinite(nz[i]) or nz[i] < 1:
            continue
        n = int(nz[i])
        vr = counts[i, :n].astype(np.float64)[None, None, :]
        # calculate_pad(voxel_returns, voxel_height = 1, beer_lambert_constant = 1, drop_ground = True)
        rev = vr[:, :, ::-1]
        total = np.sum(rev, axis=2, keepdims=True)
        csum = np.cumsum(rev, axis=2)
        shots_out = total - csum
        shots_in = np.concatenate((total, shots_out[:, :, :-1]), axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            pad_sky = np.log(shots_in / shots_out) / (1.0 * 1.0)
        pad_sky[~np.isfinite(pad_sky)] = np.nan
        pad = pad_sky[:, :, ::-1].copy()
        pad[:, :, 0] = np.nan
        if np.sum(vr) == 0:
            pad[:] = np.nan
        # calculate_pai(pad, 1, min_height = 1)
        res["pfs_pai"][i] = _pai(pad, 1.0, 1.0)
        # calculate_canopy_cover(pad, 1, min_height = 2, k = 0.5)
        eff = pad.shape[2] * 1.0
        if 2.0 >= eff:
            res["pfs_cover"][i] = 0.0
        else:
            pa = _pai(pad, 1.0, 2.0)
            core = pad[:, :, int(np.ceil(2.0)):int(np.floor(eff))]
            cov = 1.0 - np.exp(-0.5 * pa)
            cov = cov if np.isfinite(cov) else np.nan
            cov = float(np.clip(cov, 0.0, 1.0))
            res["pfs_cover"][i] = np.nan if np.all(np.isnan(core)) else cov
        # calculate_fhd(voxel_returns, 1, min_height = 0)
        core = vr[:, :, 0:n]
        sc = np.sum(core, axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            prop = np.divide(core, sc[..., None], out=np.zeros_like(core, dtype=float), where=sc[..., None] != 0)
        f = entropy(prop, axis=2)
        res["pfs_fhd"][i] = np.nan if sc[0, 0] == 0 else float(f[0, 0])
    return res


def _pai(pad, dz, min_height):
    eff = pad.shape[2] * dz
    if min_height >= eff:
        return 0.0
    a, b = int(np.ceil(min_height / dz)), int(np.floor(eff / dz))
    if a >= b:
        return 0.0
    core = pad[:, :, a:b]
    if np.all(np.isnan(core)):
        return np.nan
    return float(np.nansum(core, axis=2)[0, 0] * dz)


def extra_metrics(B, cell, z, ncell):
    """leafR, PyForestScan and leafR GC metrics for one block of cells. Returns dict of numpy arrays."""
    pc = profile_counts(B, cell, z, ncell)
    out = {}
    out.update(_leafr(pc["leafr_counts"]))
    out.update(_pyforestscan(pc["pfs_counts"], pc["pfs_nz"]))
    n = pc["gc_n"].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        G = 2 * pc["gc_sx"] / pc["gc_s"] - (n + 1)
        out["leafr_gc"] = G / (n - 1)
    return out


def enl_from_voxels(vc_np, vz_np, ncell):
    """forestr::calc_enl: 1 / sum(p_i^2), p_i = filled voxels in 1 m layer i / all filled voxels."""
    cnt = np.bincount(vc_np * 64 + vz_np, minlength=ncell * 64).reshape(ncell, 64).astype(np.float64)
    tot = cnt.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p2 = ((cnt / tot[:, None]) ** 2).sum(1)
        return np.where(tot > 0, 1 / p2, np.nan)


def gap_metrics(chm, layout, x0, y1, cell, nrow, ncol, threshold=10.0, size=(1.0, 1e4)):
    """ForestGapR::getForestGaps(chm, threshold, size) and a per-cell summary.

    gap pixels: chm <= threshold (NA stays non-gap); 8-connected patches (raster::clump directions = 8);
    patches with area < size[1] or > size[2] (m2) are dropped. Per grid cell (pixel centres):
    gap_frac = share of the cell's canopy-raster pixels in a gap, gap_n = gaps touching the cell,
    gap_area_mean = mean full area (m2) of those gaps."""
    from scipy import ndimage as ndi
    g = np.isfinite(chm) & (chm <= threshold)
    lab, n = ndi.label(g, structure=np.ones((3, 3), int))
    names = ("gap_frac", "gap_n", "gap_area_mean")
    out = {k: np.full(nrow * ncol, np.nan) for k in names}
    nr, nc = chm.shape
    rr, cc = np.mgrid[0:nr, 0:nc]
    cx = layout.xmin + (cc + 0.5) * layout.res
    cy = layout.ymax - (rr + 0.5) * layout.res
    ci = np.floor((cx - x0) / cell).astype(np.int64)
    ri = np.floor((y1 - cy) / cell).astype(np.int64)
    inside = (ci >= 0) & (ci < ncol) & (ri >= 0) & (ri < nrow)
    cid = np.where(inside, ri * ncol + ci, -1)
    npx = np.bincount(cid[inside], minlength=nrow * ncol).astype(np.float64)
    if n > 0:
        area = np.bincount(lab.ravel(), minlength=n + 1).astype(np.float64) * layout.res ** 2
        okg = (area >= size[0]) & (area <= size[1])
        okg[0] = False
        lab = np.where(okg[lab], lab, 0)
    m = inside & (lab > 0)
    gp = np.bincount(cid[m], minlength=nrow * ncol).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["gap_frac"] = np.where(npx > 0, gp / npx, np.nan)
    out["gap_n"] = np.where(npx > 0, 0.0, np.nan)
    out["gap_area_mean"] = np.full(nrow * ncol, np.nan)
    if m.any():
        pairs = np.unique(np.stack([cid[m], lab[m]], 1), axis=0)
        cnt = np.bincount(pairs[:, 0], minlength=nrow * ncol)
        areas = np.bincount(lab.ravel(), minlength=lab.max() + 1).astype(np.float64) * layout.res ** 2
        sa = np.bincount(pairs[:, 0], weights=areas[pairs[:, 1]], minlength=nrow * ncol)
        out["gap_n"] = np.where(npx > 0, cnt.astype(np.float64), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["gap_area_mean"] = np.where(cnt > 0, sa / cnt, np.nan)
    return out
