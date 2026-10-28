"""Grid metrics for one tile, with memory budgets.

    from lidarmetricpy import compute_tile
    df = compute_tile("tile.laz", x0=-62000, y1=-48750, ncol=40, nrow=30, cell=25,
                      dtm="dtm.tif", backend="auto", gpu_memory_gb=8, cpu_memory_gb=16)

Memory model (budgets in GiB, i.e. 1024**3 bytes)
------------
* GPU memory (``gpu_memory_gb``): points are processed in blocks of whole grid
  cells (rows of cells, split into column ranges when a single row is too large).
  The per-point working set is about ``BYTES_PER_POINT`` bytes, so an 8 GB budget
  handles ~20 M points per block; a 96–256 GB GPU takes a whole dense tile at once.
  On a GPU the whole tile is first prepared on the device (~40 bytes per point kept) when that and
  the blocks fit the budget; otherwise points are prepared on the host and sent block by block.
* CPU memory, i.e. RAM (``cpu_memory_gb``): points are kept as raw LAS integers (~23 bytes
  per point). If a tile does not fit, it is read in several passes, each keeping
  only some rows of cells. On the CPU backends (numpy, torch-cpu) this one budget also
  bounds the working set of the blocks.
* Apple MPS shares RAM with the GPU: both budgets apply.

All metrics are computed per cell, so blocks give the same result as one pass.
Tree detection uses a 0.5 m canopy raster of the whole tile (a few tens of MB).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .area import AREA_COLUMNS, area_metrics, rumple_cells, rumple_pixels, rumple_select
from .extra import EXTRA_COLUMNS, GAP_COLUMNS, extra_metrics, gap_metrics
from .backend import get_backend
from .trees import TREE_COLUMNS, CanopyAccumulator, CanopyLayout, local_maxima, tree_metrics

BYTES_PER_POINT = {64: 420, 32: 260}      # device working set per point (measured on 6–90 M point blocks, + margin)
HOST_BYTES_PER_POINT = 23                  # raw points kept on the host (X, Y, Z int32, returns, order, mask)
# device path (whole tile prepared on the GPU), measured as CuPy pool peaks on a 90.5 M point tile:
DEVICE_PREP_BYTES = 36                     # per point while reading / filtering (x, y, z, returns, order, masks)
DEVICE_CHUNK_BYTES = 100                   # per point of one read / filter slice (opt.chunk_points), incl. pool slack
DEVICE_KEEP_BYTES = 40                     # per point kept during the blocks (the above + cell row / column int32)
DEVICE_SLACK = 0.5 * 1024 ** 3             # memory-pool fragmentation and per-block arrays (measured 0.3–0.4 GiB)
HOST_BLOCK_BYTES = 170                     # host working set per point of the block being processed
HOST_READ_BYTES = 64                       # per point of one LAZ read chunk (decoded record + coordinates)
MIN_BUDGET_GB = 1 / 128                    # 8 MiB: smaller budgets are refused
MIN_BLOCK_POINTS = 1000


@dataclass
class Grid:
    x0: float          # left edge
    y1: float          # top edge
    ncol: int
    nrow: int
    cell: float = 25.0

    @property
    def x1(self):
        return self.x0 + self.ncol * self.cell

    @property
    def y0(self):
        return self.y1 - self.nrow * self.cell


@dataclass
class Options:
    zmin: float = 0.5
    zrange: tuple = (-1.0, 60.0)
    spike_filter: bool = True          # drop the top point of a 1 m column if it is > 8 m above the next one
    trees: bool = True
    crowns: bool = True
    kde_bw: float = 2.0
    threads: int = 8
    chunk_points: int = 20_000_000     # LAZ read chunk
    device_prepare: bool = True        # GPU: bucket/normalize/filter on the device when the tile fits
    other_packages: bool = True        # leafR, PyForestScan, ForestGapR metrics (enl is always computed)
    gap_threshold: float = 10.0        # ForestGapR getForestGaps(threshold)
    gap_size: tuple = (1.0, 1e4)       # ForestGapR getForestGaps(size), m2
    laz_threads: str = "auto"          # "parallel" | "single" | "auto" (= parallel lazrs decoder; measured fastest)
    closed_edges: bool = True          # points exactly on the right / bottom outer edge belong to the last column /
    #                                    row, as in lidR (terra); False: half-open grid, for tiles that share edges

    @classmethod
    def lidr(cls, **kw):
        """The settings under which the results equal lidR / lidRmetrics on the same points: no height range
        and no spike filter (lidarmetricpy's own preprocessing, on by default, removes points that lidR keeps)."""
        return cls(**{"spike_filter": False, "zrange": (-np.inf, np.inf), **kw})


def _in_grid(x, y, grid, closed):
    """Points inside the grid: [x0, x1) × (y0, y1], or [x0, x1] × [y0, y1] with closed outer edges (lidR)."""
    if closed:
        return (x >= grid.x0) & (x <= grid.x1) & (y >= grid.y0) & (y <= grid.y1)
    return (x >= grid.x0) & (x < grid.x1) & (y > grid.y0) & (y <= grid.y1)


# ------------------------------------------------------------------ reading
class _Buckets:
    """Points of one tile kept as raw integers per row of cells."""

    def __init__(self, nrow):
        self.parts = [[] for _ in range(nrow)]

    def add(self, row, arrays):
        if len(row) == 0:
            return
        # small integer keys → numpy uses a radix sort (stable, linear time)
        row = row.astype(np.uint8 if len(self.parts) < 256 else np.uint16)
        o = np.argsort(row, kind="stable")
        rs = row[o]
        cut = np.flatnonzero(np.r_[True, rs[1:] != rs[:-1], True])
        for a, b in zip(cut[:-1], cut[1:]):
            r = rs[a]
            self.parts[r].append({k: v[o[a:b]] for k, v in arrays.items()})

    def rows(self, r):
        ps = self.parts[r]
        if not ps:
            return None
        return {k: np.concatenate([p[k] for p in ps]) for k in ps[0]}

    def count(self, r):
        return sum(len(p["X"]) for p in self.parts[r])


def _read(source, grid, opt, rows_keep=None):
    """Read a LAZ/LAS file (or a dict of arrays) into row buckets of raw values."""
    import laspy
    bk = _Buckets(grid.nrow)
    meta = {"xmin": np.inf, "xmax": -np.inf, "ymin": np.inf, "ymax": -np.inf, "n_read": 0}
    if isinstance(source, dict):
        chunks = [source]
        scale = np.array([1.0, 1.0, 1.0])
        offset = np.zeros(3)
        raw = False
    else:
        f = laspy.open(source, laz_backend=_laz_backend(opt))
        scale, offset = f.header.scales, f.header.offsets
        chunks = f.chunk_iterator(opt.chunk_points)
        raw = True
    base = 0
    for ch in chunks:
        if raw:
            X, Y, Z = np.asarray(ch.X), np.asarray(ch.Y), np.asarray(ch.Z)
            x = X * scale[0] + offset[0]
            y = Y * scale[1] + offset[1]
            rn = np.asarray(ch.return_number).astype(np.uint8)
            nr = np.asarray(ch.number_of_returns).astype(np.uint8)
        else:
            x, y = np.asarray(ch["x"], np.float64), np.asarray(ch["y"], np.float64)
            X, Y, Z = x, y, np.asarray(ch["z"], np.float64)
            rn = np.asarray(ch.get("return_number", np.ones(len(x))), np.uint8)
            nr = np.asarray(ch.get("number_of_returns", np.ones(len(x))), np.uint8)
        order = np.arange(base, base + len(x), dtype=np.int64)
        base += len(x)
        k = _in_grid(x, y, grid, opt.closed_edges)
        row = np.floor((grid.y1 - y[k]) / grid.cell).astype(np.int64)
        row = np.clip(row, 0, grid.nrow - 1)
        arr = {"X": X[k], "Y": Y[k], "Z": Z[k], "rn": rn[k], "nr": nr[k], "order": order[k]}
        if rows_keep is not None:
            m = (row >= rows_keep[0]) & (row < rows_keep[1])
            row = row[m]
            arr = {kk: v[m] for kk, v in arr.items()}
        bk.add(row, arr)
        meta["n_read"] += int(k.sum())
    meta.update(scale=np.asarray(scale, np.float64), offset=np.asarray(offset, np.float64), raw=raw, n_total=base)
    return bk, meta


def _dtm_window(dtm, grid):
    """The DTM pixels covering the grid (plus one pixel), as float64 with NaN where the DTM has no data
    (nodata, mask, or outside the raster). Returns (array, x and y of the DTM's own top-left corner, x and y
    resolution, column and row of the window's first pixel in the DTM). A point takes the DTM pixel that
    contains it: column floor((x - x_dtm) / res_x), row floor((y_dtm - y) / res_y) on the DTM's own grid (as
    lidR), then shifted by the window's integer offset, so the grid origin never changes the pixel."""
    import rasterio
    from rasterio.windows import Window, from_bounds
    with rasterio.open(dtm) as s:
        t = s.transform
        if t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0:
            raise ValueError("the DTM must be north-up without rotation")
        w = from_bounds(grid.x0, grid.y0, grid.x1, grid.y1, t)
        c0, r0 = math.floor(w.col_off) - 1, math.floor(w.row_off) - 1
        c1, r1 = math.ceil(w.col_off + w.width) + 1, math.ceil(w.row_off + w.height) + 1
        a = np.full((r1 - r0, c1 - c0), np.nan)
        rr0, cc0, rr1, cc1 = max(r0, 0), max(c0, 0), min(r1, s.height), min(c1, s.width)
        if rr1 > rr0 and cc1 > cc0:
            d = s.read(1, window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0), masked=True)
            a[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = np.ma.filled(d.astype(np.float64), np.nan)
    a[a < -9000] = np.nan                     # undeclared nodata such as -9999
    return a, t.c, t.f, t.a, -t.e, c0, r0


# ------------------------------------------------------------------ planning
def plan_blocks(row_counts, ncol, budget_points, cell_counts=None):
    """Group rows of cells into blocks of at most budget_points; split a row by columns if needed.
    cell_counts (rows × ncol, optional) lets a crowded row be split by the points of each cell; a single cell
    above the budget then forms a block of its own (it cannot be split). Returns a list of (row0, row1, col0, col1)."""
    blocks = []
    r = 0
    n = len(row_counts)
    while r < n:
        if row_counts[r] > budget_points:
            if cell_counts is None:
                parts = math.ceil(row_counts[r] / budget_points)
                w = math.ceil(ncol / parts)
                for c0 in range(0, ncol, w):
                    blocks.append((r, r + 1, c0, min(ncol, c0 + w)))
            else:
                c0, tot = 0, 0
                for c in range(ncol):
                    v = int(cell_counts[r][c])
                    if c > c0 and tot + v > budget_points:
                        blocks.append((r, r + 1, c0, c))
                        c0, tot = c, 0
                    tot += v
                blocks.append((r, r + 1, c0, ncol))
            r += 1
            continue
        tot, r1 = 0, r
        while r1 < n and tot + row_counts[r1] <= budget_points:
            tot += row_counts[r1]
            r1 += 1
        blocks.append((r, r1, 0, ncol))
        r = r1
    return blocks


def _device_budget(B, gpu_memory_gb):
    if gpu_memory_gb is not None:
        return gpu_memory_gb * 1024 ** 3
    return 0.7 * B.mem_free_bytes()


def _budget_args(B, gpu_memory_gb, cpu_memory_gb, max_memory_gb, host_memory_gb):
    """Resolve the budgets, accepting the 0.1/0.2 names max_memory_gb (GPU) and host_memory_gb (RAM)."""
    import warnings
    if max_memory_gb is not None:
        warnings.warn("max_memory_gb is renamed gpu_memory_gb", DeprecationWarning, stacklevel=3)
        if B.device == "cpu" and cpu_memory_gb is None and host_memory_gb is None:
            cpu_memory_gb = max_memory_gb          # 0.1/0.2: on CPU backends it bounded the RAM working set
        elif gpu_memory_gb is None:
            gpu_memory_gb = max_memory_gb
    if host_memory_gb is not None:
        warnings.warn("host_memory_gb is renamed cpu_memory_gb", DeprecationWarning, stacklevel=3)
        if cpu_memory_gb is None:
            cpu_memory_gb = host_memory_gb
    if gpu_memory_gb is not None and B.device == "cpu":
        warnings.warn(f"gpu_memory_gb has no effect on the {B.name} CPU backend; use cpu_memory_gb", stacklevel=3)
        gpu_memory_gb = None
    for name, v in (("gpu_memory_gb", gpu_memory_gb), ("cpu_memory_gb", cpu_memory_gb)):
        if v is not None and v < MIN_BUDGET_GB:
            raise ValueError(f"{name} = {v} GiB is below the smallest supported budget ({MIN_BUDGET_GB} GiB = 8 MiB)")
    return gpu_memory_gb, cpu_memory_gb


# ------------------------------------------------------------------ main
def _fbits(B):
    return 32 if "float32" in str(B.f) else 64


def _xyz(P, meta):
    if meta["raw"]:
        return (P["X"] * meta["scale"][0] + meta["offset"][0], P["Y"] * meta["scale"][1] + meta["offset"][1],
                P["Z"] * meta["scale"][2] + meta["offset"][2])
    return P["X"], P["Y"], P["Z"]


def _prepare_row(P, meta, grid, opt, dtm_a, ok=None):
    """Heights above ground and the keep mask for one row of cells (host, float64)."""
    x, y, z = _xyz(P, meta)
    if dtm_a is not None:
        a, gx0, gy1, rx, ry, c0, r0 = dtm_a
        ri = np.floor((gy1 - y) / ry).astype(np.int64) - r0
        ci = np.floor((x - gx0) / rx).astype(np.int64) - c0
        inw = (ri >= 0) & (ri < a.shape[0]) & (ci >= 0) & (ci < a.shape[1])
        g = a[np.clip(ri, 0, a.shape[0] - 1), np.clip(ci, 0, a.shape[1] - 1)]
        z = z - np.where(inw, g, np.nan)
    if ok is not None:
        return x, y, z, ok
    ok = np.isfinite(z) & (z > opt.zrange[0]) & (z < opt.zrange[1])
    if opt.spike_filter and ok.any():
        idx = np.flatnonzero(ok)
        ok[idx[~_spike_filter(x[idx], y[idx], z[idx], grid)]] = False
    return x, y, z, ok


def compute_tile(source, x0, y1, ncol=40, nrow=30, cell=25.0, dtm=None, backend="auto",
                 gpu_memory_gb=None, cpu_memory_gb=None, options=None, return_trees=False, verbose=False,
                 max_memory_gb=None, host_memory_gb=None):
    """Metrics for one grid (usually one tile). Returns a DataFrame with one row per cell that has points.

    source : path to LAS/LAZ, or dict with x, y, z (height above ground if dtm is None), return_number,
             number_of_returns
    x0, y1 : left and top edge of the grid; ncol × nrow cells of size cell (m)
    dtm    : path to a terrain raster (heights are z - DTM at the point, nearest cell), or None
    backend: 'auto', 'cupy', 'torch' (cuda / mps / cpu), 'torch-cuda', 'mps', 'torch-cpu', 'numpy'
    gpu_memory_gb: GPU memory budget in GiB (Apple MPS: unified memory). None = 70 % of the free GPU memory.
                   No effect on the CPU backends.
    cpu_memory_gb: RAM budget in GiB (raw points; on CPU backends also the working set). None = 60 % of free RAM.
    return_trees : also return (canopy raster, raster layout, (rows, cols, crown areas) of the tree tops)
    max_memory_gb, host_memory_gb: old names of gpu_memory_gb and cpu_memory_gb (deprecated)
    """
    B = get_backend(backend)
    gpu_memory_gb, cpu_memory_gb = _budget_args(B, gpu_memory_gb, cpu_memory_gb, max_memory_gb, host_memory_gb)
    with np.errstate(divide="ignore", invalid="ignore"):     # empty cells and layers give NaN on purpose
        return _compute_tile(source, x0, y1, ncol, nrow, cell, dtm, B, gpu_memory_gb, cpu_memory_gb,
                             options or Options(), return_trees, verbose)


def _compute_tile(source, x0, y1, ncol, nrow, cell, dtm, B, gpu_memory_gb, cpu_memory_gb, opt, return_trees,
                  verbose):
    t0 = time.time()
    T = {"read": 0.0, "prepare": 0.0, "area": 0.0, "canopy": 0.0, "trees": 0.0}
    grid = Grid(x0, y1, ncol, nrow, cell)
    from .backend import _host_free_bytes
    host_budget = (cpu_memory_gb * 1024 ** 3) if cpu_memory_gb is not None else 0.6 * _host_free_bytes()
    npt = len(source["x"]) if isinstance(source, dict) else _point_count(source)
    raw = npt * HOST_BYTES_PER_POINT
    passes = [(0, nrow)]
    if not isinstance(source, dict) and raw > 0.5 * host_budget:
        k = max(1, math.ceil(raw / (0.5 * host_budget)))
        step = math.ceil(nrow / k)
        passes = [(r, min(nrow, r + step)) for r in range(0, nrow, step)]
    raw_kept = raw / len(passes)
    host_pts = max(MIN_BLOCK_POINTS, int((host_budget - raw_kept) / HOST_BLOCK_BYTES))
    # LAZ is decoded in chunks that also fit the RAM budget
    read_opt = replace(opt, chunk_points=min(opt.chunk_points, max(10_000, int(0.25 * host_budget / HOST_READ_BYTES))))
    if B.name == "numpy" or (B.name == "torch" and B.device in ("cpu", "mps")):
        # device memory is host memory: one budget for both
        dev_pts = max(MIN_BLOCK_POINTS, int((host_budget - raw_kept) / (HOST_BLOCK_BYTES + BYTES_PER_POINT[_fbits(B)])))
        if gpu_memory_gb is not None:                       # Apple MPS: GPU budget within the shared RAM
            dev_pts = min(dev_pts, int(gpu_memory_gb * 1024 ** 3 / (HOST_BLOCK_BYTES + BYTES_PER_POINT[_fbits(B)])))
        budget_pts = dev_pts
    else:
        budget_pts = min(host_pts, max(MIN_BLOCK_POINTS, int(_device_budget(B, gpu_memory_gb) / BYTES_PER_POINT[_fbits(B)])))
    dtm_a = _dtm_window(dtm, grid) if dtm is not None else None
    stats = {"blocks": 0, "points": 0, "passes": len(passes), "budget_points": budget_pts,
             "backend": f"{B.name}:{B.device}"}
    # device path: all points prepared on the device, metrics in blocks of cell rows. Taken only when both the
    # preparation and the blocks fit the budget; otherwise the host path below bounds the device memory.
    dev_budget = _device_budget(B, gpu_memory_gb)
    # read / filter slices shrink to what is left after the prepared points (same results, any slice size)
    chunk_pts = min(opt.chunk_points, int((dev_budget - DEVICE_SLACK - npt * DEVICE_PREP_BYTES) / DEVICE_CHUNK_BYTES))
    block_pts = int((dev_budget - DEVICE_SLACK - npt * DEVICE_KEEP_BYTES) / BYTES_PER_POINT[64])
    device_path = (opt.device_prepare and _fbits(B) == 64 and B.device != "cpu" and chunk_pts >= 1_000_000
                   and block_pts >= 1_000_000)
    if device_path:
        stats["path"] = "device"
        stats["budget_points"] = block_pts
        # LAZ is decoded in chunks that fit the RAM budget; the device works on slices that fit the GPU budget
        results, acc, npts = _device_tile(source, grid, replace(opt, chunk_points=chunk_pts), B, dtm_a, T,
                                          block_pts, read_points=read_opt.chunk_points)
        stats.update(blocks=len(results), points=npts, passes=1, cells_over_budget=T.pop("cells_over_budget", 0))
        B.free()
        return _finish(results, acc, B, grid, opt, nrow, ncol, cell, stats, T, t0, verbose, return_trees)
    stats["path"] = "host"

    # phase A: keep masks and the bbox of the kept points (the canopy raster layout depends on it)
    bbox = [np.inf, -np.inf, np.inf, -np.inf]
    masks = {}
    bk = None
    for p0, p1 in passes:
        t = time.time()
        bk, meta = _read(source, grid, read_opt, None if len(passes) == 1 else (p0, p1))
        T["read"] += time.time() - t
        t = time.time()
        for r in range(p0, p1):
            P = bk.rows(r)
            if P is None:
                continue
            x, y, z, ok = _prepare_row(P, meta, grid, opt, dtm_a)
            masks[r] = ok
            if ok.any():
                bbox = [min(bbox[0], x[ok].min()), max(bbox[1], x[ok].max()),
                        min(bbox[2], y[ok].min()), max(bbox[3], y[ok].max())]
        T["prepare"] += time.time() - t
        if len(passes) > 1:
            bk = None
    acc = None
    if opt.trees and np.isfinite(bbox[0]):
        acc = CanopyAccumulator(B, CanopyLayout(*bbox))

    # phase B: metrics block by block
    results = []
    for p0, p1 in passes:
        if bk is None:
            t = time.time()
            bk, meta = _read(source, grid, read_opt, (p0, p1))
            T["read"] += time.time() - t
        counts = [int(masks[r].sum()) if r in masks else 0 for r in range(p0, p1)]
        ccounts = []
        for r in range(p0, p1):
            cc_ = np.zeros(ncol, np.int64)
            if r in masks and counts[r - p0] > budget_pts:          # per-cell counts only for crowded rows
                P = bk.rows(r)
                xr_ = _xyz(P, meta)[0][masks[r]]
                cc_ = np.bincount(np.minimum(np.floor((xr_ - grid.x0) / cell).astype(np.int64), ncol - 1),
                                  minlength=ncol)
            ccounts.append(cc_)
        stats["cells_over_budget"] = stats.get("cells_over_budget", 0) + int(sum((c > budget_pts).sum() for c in ccounts))
        for (r0, r1, c0, c1) in plan_blocks(counts, ncol, budget_pts, ccounts):
            t = time.time()
            r0, r1 = r0 + p0, r1 + p0
            xs, ys, zs, Ps = [], [], [], []
            for r in range(r0, r1):
                P = bk.rows(r)
                if P is None:
                    continue
                x, y, z, ok = _prepare_row(P, meta, grid, opt, dtm_a, ok=masks[r])
                if c0 > 0 or c1 < ncol:
                    colc = np.minimum(np.floor((x - grid.x0) / cell).astype(np.int64), ncol - 1)
                    ok = ok & (colc >= c0) & (colc < c1)
                xs.append(x[ok]); ys.append(y[ok]); zs.append(z[ok])
                Ps.append({k: v[ok] for k, v in P.items() if k in ("rn", "nr", "order")})
            if not xs or sum(len(v) for v in zs) == 0:
                continue
            x, y, z = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
            P = {k: np.concatenate([p[k] for p in Ps]) for k in Ps[0]}
            del xs, ys, zs, Ps
            T["prepare"] += time.time() - t
            t = time.time()
            results.append(_block(B, grid, opt, x, y, z, P, r0, r1, c0, c1, T, acc))
            B.free()
            T["area"] += time.time() - t
            stats["blocks"] += 1
            stats["points"] += len(z)
        if len(passes) > 1:
            bk = None
    del bk

    return _finish(results, acc, B, grid, opt, nrow, ncol, cell, stats, T, t0, verbose, return_trees)


def _finish(results, acc, B, grid, opt, nrow, ncol, cell, stats, T, t0, verbose, return_trees):
    cols = output_columns(opt)
    if stats.get("cells_over_budget", 0):
        import warnings
        warnings.warn(f"{stats['cells_over_budget']} cell(s) hold more points than the memory budget allows for one "
                      "block; each was processed as a block of its own, above the budget", stacklevel=3)
    df = pd.concat(results, ignore_index=True) if results else pd.DataFrame({c: [] for c in cols}, dtype=float)
    tops = None
    if acc is not None:
        t = time.time()
        chm = acc.finish()
        rr, cc = local_maxima(B, chm)
        tm, tops = tree_metrics(chm, acc.L, grid.x0, grid.y1, cell, nrow, ncol, B=B, rr=rr, cc=cc,
                                crowns=opt.crowns)
        if opt.other_packages:
            tm.update(gap_metrics(chm, acc.L, grid.x0, grid.y1, cell, nrow, ncol,
                                  threshold=opt.gap_threshold, size=opt.gap_size))
        ii = np.arange(nrow * ncol)
        tdf = pd.DataFrame({"x": grid.x0 + (ii % ncol + 0.5) * cell, "y": grid.y1 - (ii // ncol + 0.5) * cell, **tm})
        df = df.merge(tdf, on=["x", "y"], how="left")       # tree / gap metrics of the cells with points
        tops = (chm, acc.L, tops)
        T["trees"] += time.time() - t
    df = df.reindex(columns=cols).sort_values(["y", "x"], ascending=[False, True], ignore_index=True)
    stats["seconds"] = round(time.time() - t0, 2)
    stats["stage_seconds"] = {k: round(v, 2) for k, v in T.items()}
    if verbose:
        print(stats)
    df.attrs["stats"] = stats
    return (df, tops) if return_trees else df


def _spike_key(x, y, grid):
    """1 m column of the spike filter: whole metres from the grid's top-left corner, split at the edges of the
    rows of cells, so that the host path (row of cells by row of cells) and the device path agree for any
    cell size. Same key as in _spike_filter_dev."""
    crow = np.minimum(np.floor((grid.y1 - y) / grid.cell).astype(np.int64), grid.nrow - 1)
    r = np.floor(grid.y1 - y).astype(np.int64)
    c = np.floor(x - grid.x0).astype(np.int64)
    return (crow * 1_000_000 + r) * 1_000_000 + c


def _spike_filter(x, y, z, grid):
    """Top point of a 1 m column more than 8 m above the second highest → noise."""
    col = _spike_key(x, y, grid)
    o = np.lexsort((z, col))
    cs, hs = col[o], z[o]
    last = np.r_[cs[1:] != cs[:-1], True]
    prev_same = np.r_[False, cs[1:] == cs[:-1]]
    jump = hs - np.r_[hs[:1], hs[:-1]] > 8
    keep = np.ones(len(z), bool)
    keep[o[last & prev_same & jump]] = False
    return keep


def _point_count(path):
    import laspy
    with laspy.open(path) as f:
        return f.header.point_count


def _round_away(a):
    t = np.trunc(a)
    return np.where(np.abs(a - t) == 0.5, t + np.sign(a), np.rint(a))


def _block(B, grid, opt, x, y, z, P, r0, r1, c0, c1, T=None, acc=None):
    """Area metrics (and canopy accumulation) for a block of cells, from host arrays."""
    if _fbits(B) == 64:
        return _block_dev(B, grid, opt, B.asarray(x), B.asarray(y), B.asarray(z), B.asarray(P["rn"]),
                          B.asarray(P["nr"]), B.asarray(P["order"]), r0, r1, c0, c1, T, acc)
    # float32 devices: integer indices in float64 on the host
    cell = grid.cell
    nr_, nc_ = r1 - r0, c1 - c0
    ncell = nr_ * nc_
    bx0 = math.floor(grid.x0 + c0 * cell) - 2
    by0 = math.floor(grid.y1 - r1 * cell) - 2
    i64 = B.i64
    row = np.minimum(np.floor((grid.y1 - y) / cell).astype(np.int32), grid.nrow - 1)     # closed outer edges
    col = np.minimum(np.floor((x - grid.x0) / cell).astype(np.int32), grid.ncol - 1)
    cid_h = (row - r0) * nc_ + (col - c0)
    del row, col
    vx = B.astype(B.asarray((_round_away(x) - bx0).astype(np.int32)), i64)
    vy = B.astype(B.asarray((_round_away(y) - by0).astype(np.int32)), i64)
    from .backend import NumpyLike
    pid = B.asarray(rumple_pixels(NumpyLike(np, "numpy"), cid_h.astype(np.int64), x, y, ncell))
    cid = B.astype(B.asarray(cid_h), i64)
    zd = B.asarray(z)
    rn = B.astype(B.asarray(P["rn"]), i64)
    nr = B.astype(B.asarray(P["nr"]), i64)
    m = area_metrics(B, cid, zd, rn, nr, vx, vy, ncell, zmin=opt.zmin, bw=opt.kde_bw)
    del vx, vy, rn, nr
    if opt.other_packages:
        m.update(extra_metrics(B, cid, zd, ncell))
    m["n_points"] = B.to_numpy(B.bincount(cid, minlength=ncell)).astype(np.float64)
    t = time.time()
    sel = rumple_select(B, cid, zd, pid, B.asarray(P["order"]))
    m["rumple"] = rumple_cells(cid_h[sel], x[sel], y[sel], z[sel], ncell, threads=opt.threads)
    if T is not None:
        T["rumple"] = T.get("rumple", 0.0) + time.time() - t
    if acc is not None:
        acc.add(x, y, z)
    return _cells_frame(grid, m, r0, r1, c0, c1)


def _cells_frame(grid, m, r0, r1, c0, c1):
    """Rows of the cells of a block that have at least one point after the preprocessing."""
    nc_ = c1 - c0
    ii = np.arange((r1 - r0) * nc_)
    gr = r0 + ii // nc_
    gc = c0 + ii % nc_
    df = pd.DataFrame({"x": grid.x0 + (gc + 0.5) * grid.cell, "y": grid.y1 - (gr + 0.5) * grid.cell, **m})
    return df[df.n_points > 0]


def output_columns(opt):
    """Columns of compute_tile's table for these options (fixed, whatever the data)."""
    cols = ["x", "y", "n_points", "n"] + list(AREA_COLUMNS) + ["rumple"]
    if opt.other_packages:
        cols += list(EXTRA_COLUMNS)
    if opt.trees:
        cols += list(TREE_COLUMNS)
        if opt.other_packages:
            cols += list(GAP_COLUMNS)
    return cols


def _block_dev(B, grid, opt, xd, yd, zd, rnd, nrd, ordd, r0, r1, c0, c1, T=None, acc=None):
    """Area metrics for a block of cells with all point arrays already on a float64 device.
    Indices are computed with the same IEEE operations as on the host (no fused multiply-add)."""
    cell = grid.cell
    nr_, nc_ = r1 - r0, c1 - c0
    ncell = nr_ * nc_
    bx0 = math.floor(grid.x0 + c0 * cell) - 2
    by0 = math.floor(grid.y1 - r1 * cell) - 2
    i64 = B.i64
    row = B.clip(B.astype(B.floor(B.sdiv(grid.y1 - yd, cell)), i64), 0, grid.nrow - 1)    # closed outer edges
    col = B.clip(B.astype(B.floor(B.sdiv(xd - grid.x0, cell)), i64), 0, grid.ncol - 1)
    cid = (row - r0) * nc_ + (col - c0)
    del row, col
    vx = B.astype(B.round_away(xd), i64) - bx0           # lidR:::round_any → C round()
    vy = B.astype(B.round_away(yd), i64) - by0
    pid = rumple_pixels(B, cid, xd, yd, ncell)              # (cell, 1 m pixel) ids of highest(1)
    m = area_metrics(B, cid, zd, B.astype(rnd, i64), B.astype(nrd, i64), vx, vy, ncell,
                     zmin=opt.zmin, bw=opt.kde_bw)
    del vx, vy
    if opt.other_packages:
        m.update(extra_metrics(B, cid, zd, ncell))
    m["n_points"] = B.to_numpy(B.bincount(cid, minlength=ncell)).astype(np.float64)
    t = time.time()
    sel = rumple_select(B, cid, zd, pid, ordd)
    del pid
    sd = B.asarray(sel)
    m["rumple"] = rumple_cells(B.to_numpy(cid[sd]), B.to_numpy(xd[sd]), B.to_numpy(yd[sd]), B.to_numpy(zd[sd]),
                               ncell, threads=opt.threads)
    if T is not None:
        T["rumple"] = T.get("rumple", 0.0) + time.time() - t
    if acc is not None:
        t = time.time()
        acc.add_device(xd, yd, zd)
        if T is not None:
            T["canopy"] = T.get("canopy", 0.0) + time.time() - t
    return _cells_frame(grid, m, r0, r1, c0, c1)


def _laz_backend(opt):
    import laspy
    order = (laspy.LazBackend.Lazrs, laspy.LazBackend.LazrsParallel) if opt.laz_threads == "single" \
        else (laspy.LazBackend.LazrsParallel, laspy.LazBackend.Lazrs)
    av = [b for b in order if b.is_available()]
    return av[0] if av else None


def _keep(B, P, mask):
    """Filter the point arrays in place, one at a time (peak = one extra array, not a second copy of all)."""
    if bool(B.item(B.all(mask))):
        return
    for i in range(len(P)):
        P[i] = P[i][mask]


def _spike_filter_dev(B, grid, opt, P):
    """Drop the highest point of a 1 m column when it is > 8 m above the next one (columns with >= 2 points).
    Bands of whole 1 m rows are sorted one at a time (lexsort by column, then z, then file position), which
    is the same order as one sort of the whole tile but needs memory only for one band."""
    x, y, z = P[0], P[1], P[2]
    n = int(z.shape[0])
    step = max(1, opt.chunk_points)
    r = B.zeros(n, dtype=B.i32)                                # 1 m row of the column key
    for s0 in range(0, n, step):
        s1 = min(n, s0 + step)
        r[s0:s1] = B.astype(B.floor(grid.y1 - y[s0:s1]), B.i32)
    rmin = int(B.item(B.amin(r)))
    counts = B.to_numpy(B.bincount(B.astype(r - rmin, B.i64)))
    bands, a, acc = [], 0, 0
    for i, c in enumerate(counts):                             # consecutive rows, about `step` points per band
        if acc and acc + c > step:
            bands.append((a, i))
            a, acc = i, 0
        acc += int(c)
    bands.append((a, len(counts)))
    keep = B.full(n, True, dtype=B.bool)
    for b0, b1 in bands:
        idx = B.nonzero((r >= rmin + b0) & (r < rmin + b1))    # increasing: file position order kept
        if int(idx.shape[0]) < 2:
            continue
        yi = y[idx]
        crow = B.minimum(B.astype(B.floor(B.sdiv(grid.y1 - yi, grid.cell)), B.i64), grid.nrow - 1)
        col = (crow * 1_000_000 + B.astype(r[idx], B.i64)) * 1_000_000 + B.astype(B.floor(x[idx] - grid.x0), B.i64)
        del yi
        zz = z[idx]
        o = B.lexsort([zz, col])
        cs = col[o]
        del col
        bad = B.concatenate([cs[1:] != cs[:-1], B.full(1, True, dtype=B.bool)])          # last of its column
        bad &= B.concatenate([B.full(1, False, dtype=B.bool), cs[1:] == cs[:-1]])        # not alone
        del cs
        hs = zz[o]
        bad &= (hs - B.concatenate([hs[:1], hs[:-1]])) > 8                               # jump > 8 m
        keep[idx[o[bad]]] = False
        del idx, zz, o, hs, bad
    del r, x, y, z
    B.free()
    _keep(B, P, keep)


def _device_tile(source, grid, opt, B, dtm_a, T, block_pts, read_points=None):
    """Whole tile on the device: decode LAZ chunks on the CPU, then coordinates, bounds, heights,
    filters, bbox and all metrics on the device. Returns (frames, canopy accumulator, n_points)."""
    import laspy
    t = time.time()
    if isinstance(source, dict):
        n0 = len(source["x"])
        chunks = [dict(x=np.asarray(source["x"], np.float64), y=np.asarray(source["y"], np.float64),
                       z=np.asarray(source["z"], np.float64),
                       rn=np.asarray(source.get("return_number", np.ones(n0)), np.uint8),
                       nr=np.asarray(source.get("number_of_returns", np.ones(n0)), np.uint8))]
        scale = None
    else:
        f = laspy.open(source, laz_backend=_laz_backend(opt))
        scale, offset = np.asarray(f.header.scales, np.float64), np.asarray(f.header.offsets, np.float64)
        chunks = f.chunk_iterator(read_points or opt.chunk_points)   # RAM budget (decoding on the host)
    # Points are written straight into preallocated device buffers and filtered one array at a time, so the
    # whole tile is never held twice (no concatenate, no tuple of filtered copies). Values and order are
    # the same as filtering every array with the same mask at once.
    cap = n0 if scale is None else int(f.header.point_count)
    P = None                                                  # x, y, z, return number, number of returns, order
    pos = base = 0
    for ch in chunks:
        if scale is None:
            x, y = B.asarray(ch["x"]), B.asarray(ch["y"])
        else:
            # raw integers to the device; x = X * scale + offset as two separate operations
            # (separate kernels cannot be fused into an FMA, so the rounding equals numpy's)
            x = (B.astype(B.asarray(np.ascontiguousarray(ch.X)), B.f) * float(scale[0])) + float(offset[0])
            y = (B.astype(B.asarray(np.ascontiguousarray(ch.Y)), B.f) * float(scale[1])) + float(offset[1])
        n = int(x.shape[0])
        k = _in_grid(x, y, grid, opt.closed_edges)
        m = int(B.item(B.sum(B.astype(k, B.i64))))

        def put(i, v):
            if P[i] is None:
                P[i] = B.zeros(cap, dtype=v.dtype)
            P[i][pos:pos + m] = v[k]

        if P is None:
            P = [None] * 6
        put(0, x)
        del x
        put(1, y)
        del y
        if scale is None:
            put(2, B.asarray(ch["z"]))
            put(3, B.asarray(ch["rn"]))
            put(4, B.asarray(ch["nr"]))
        else:
            put(2, (B.astype(B.asarray(np.ascontiguousarray(ch.Z)), B.f) * float(scale[2])) + float(offset[2]))
            put(3, B.asarray(np.ascontiguousarray(ch.return_number).astype(np.uint8)))
            put(4, B.asarray(np.ascontiguousarray(ch.number_of_returns).astype(np.uint8)))
        put(5, B.arange(base, base + n, dtype=B.i32))            # file order (ties), int32 saves memory
        base += n
        pos += m
        del k
    P = [a[:pos] for a in P] if P is not None else []
    B.free()
    T["read"] += time.time() - t
    t = time.time()
    if pos == 0:
        T["prepare"] += time.time() - t
        return [], None, 0
    x, y, z = P[0], P[1], P[2]
    if dtm_a is not None:
        a, gx0, gy1, rx, ry, c0, r0 = dtm_a
        ad = B.asarray(a).reshape(-1)
        step = max(1, opt.chunk_points)
        for s0 in range(0, pos, step):                         # in slices: bounded temporaries
            s1 = min(pos, s0 + step)
            ri = B.astype(B.floor(B.sdiv(gy1 - y[s0:s1], ry)), B.i64) - r0     # pixel on the DTM's own grid
            ci = B.astype(B.floor(B.sdiv(x[s0:s1] - gx0, rx)), B.i64) - c0
            inw = (ri >= 0) & (ri < a.shape[0]) & (ci >= 0) & (ci < a.shape[1])
            g = ad[B.clip(ri, 0, a.shape[0] - 1) * a.shape[1] + B.clip(ci, 0, a.shape[1] - 1)]
            z[s0:s1] = z[s0:s1] - B.where(inw, g, B.nan)
            del ri, ci, inw, g
        del ad
    ok = B.isfinite(z) & (z > opt.zrange[0]) & (z < opt.zrange[1])
    del x, y, z
    _keep(B, P, ok)
    del ok
    if opt.spike_filter and int(P[2].shape[0]):
        _spike_filter_dev(B, grid, opt, P)
    B.free()
    x, y, z, rn, nr, order = P
    del P
    T["prepare"] += time.time() - t
    npts = int(z.shape[0])
    if npts == 0:
        return [], None, 0
    acc = None
    if opt.trees:
        bbox = [B.item(B.amin(x)), B.item(B.amax(x)), B.item(B.amin(y)), B.item(B.amax(y))]
        acc = CanopyAccumulator(B, CanopyLayout(*bbox))
    t = time.time()
    if npts <= block_pts:
        frames = [_block_dev(B, grid, opt, x, y, z, rn, nr, order, 0, grid.nrow, 0, grid.ncol, T, acc)]
    else:
        # the prepared points stay on the device; the heavy work runs on blocks of cell rows
        row, col = B.zeros(npts, dtype=B.i32), B.zeros(npts, dtype=B.i32)   # int32: kept for all blocks
        for s0 in range(0, npts, max(1, opt.chunk_points)):                  # in slices: bounded temporaries
            s1 = min(npts, s0 + opt.chunk_points)
            row[s0:s1] = B.clip(B.astype(B.floor(B.sdiv(grid.y1 - y[s0:s1], grid.cell)), B.i32), 0, grid.nrow - 1)
            col[s0:s1] = B.clip(B.astype(B.floor(B.sdiv(x[s0:s1] - grid.x0, grid.cell)), B.i32), 0, grid.ncol - 1)
        B.free()
        counts = [int(v) for v in B.to_numpy(B.bincount(row, minlength=grid.nrow))]
        ccounts = np.zeros(grid.nrow * grid.ncol, np.int64)
        for s0 in range(0, npts, max(1, opt.chunk_points)):                  # in slices: bounded temporaries
            s1 = min(npts, s0 + opt.chunk_points)
            ccounts += B.to_numpy(B.bincount(B.astype(row[s0:s1], B.i64) * grid.ncol + B.astype(col[s0:s1], B.i64),
                                             minlength=grid.nrow * grid.ncol))
        ccounts = ccounts.reshape(grid.nrow, grid.ncol)
        T["cells_over_budget"] = int((ccounts > block_pts).sum())
        frames = []
        for (r0, r1, c0, c1) in plan_blocks(counts, grid.ncol, block_pts, ccounts):
            m = (row >= r0) & (row < r1) & (col >= c0) & (col < c1)
            if not bool(B.item(B.sum(B.astype(m, B.i64)) > 0)):
                continue
            frames.append(_block_dev(B, grid, opt, x[m], y[m], z[m], rn[m], nr[m], order[m], r0, r1, c0, c1, T, acc))
            del m
            B.free()
    T["area"] += time.time() - t
    return frames, acc, npts


# ------------------------------------------------------------------ many tiles
def _init_worker(threads):
    import os
    if threads is None:
        return
    os.environ["RAYON_NUM_THREADS"] = str(threads)       # lazrs parallel decoder (rayon)
    os.environ["NUMBA_NUM_THREADS"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)


def _worker(args):
    job, kw = args
    job = dict(job)
    name = job.pop("name", None)
    df = compute_tile(**job, **kw)
    return name, df


def compute_tiles(jobs, workers="auto", backend="auto", gpu_memory_gb=None, cpu_memory_gb=None,
                  points_per_tile=None, max_memory_gb=None, host_memory_gb=None, **kw):
    """Run compute_tile for many tiles in parallel processes sharing one machine.

    jobs: list of dicts with compute_tile arguments (source, x0, y1, ...) and an optional 'name'.
    gpu_memory_gb / cpu_memory_gb are the TOTAL budgets (GiB); each worker gets an equal share.
    None = 70 % of the free GPU memory / 60 % of the free RAM.
    workers='auto': as many as fit in the budget (points_per_tile points each), at most CPU/2.
    Yields (name, DataFrame) in the order of jobs.
    """
    import multiprocessing as mp
    import os
    from concurrent.futures import ProcessPoolExecutor

    from .backend import _host_free_bytes
    B = get_backend(backend)
    gpu_memory_gb, cpu_memory_gb = _budget_args(B, gpu_memory_gb, cpu_memory_gb, max_memory_gb, host_memory_gb)
    ram = cpu_memory_gb * 1024 ** 3 if cpu_memory_gb is not None else 0.6 * _host_free_bytes()
    if workers == "auto":
        ppt = points_per_tile or 60_000_000
        if B.device == "cpu":
            per = ppt * (HOST_BYTES_PER_POINT + HOST_BLOCK_BYTES + BYTES_PER_POINT[_fbits(B)])
            fit = ram // per
        else:
            fit = min(_device_budget(B, gpu_memory_gb) // (ppt * BYTES_PER_POINT[_fbits(B)]),
                      ram // (ppt * (HOST_BYTES_PER_POINT + HOST_BLOCK_BYTES)))
        workers = int(max(1, min((os.cpu_count() or 2) // 2, fit)))
    kw = dict(kw, backend=backend, cpu_memory_gb=ram / 1024 ** 3 / workers)
    if B.device != "cpu":
        # each process also holds a CUDA context and allocator slack (~0.7 GiB) outside the budget
        total = _device_budget(B, gpu_memory_gb)
        kw["gpu_memory_gb"] = (total / 1024 ** 3) / workers - (0.7 if B.device == "cuda" else 0.0)
    for k_ in ("cpu_memory_gb", "gpu_memory_gb"):
        if k_ in kw and kw[k_] < MIN_BUDGET_GB:
            raise ValueError(f"{k_} per worker = {kw[k_]:.4f} GiB is below {MIN_BUDGET_GB} GiB; use fewer workers")
    # decode_threads: limit the parallel LAZ decoder per worker (None = let every worker use all cores,
    # which measured fastest: 1.2 s vs 1.4 s per tile with 12 workers on 32 cores)
    threads = kw.pop("decode_threads", None)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker, initargs=(threads,)) as ex:
        for name, df in ex.map(_worker, [(j, kw) for j in jobs]):
            yield name, df
