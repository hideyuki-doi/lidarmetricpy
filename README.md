# lidarmetricpy

Per-cell forest structure metrics from airborne lidar point clouds on NVIDIA GPUs (CuPy or PyTorch), Apple
Silicon GPUs (PyTorch MPS, float32) or any CPU (NumPy), with explicit memory budgets. With `Options.lidr()` the
values equal those of **lidR 4.3.3 + lidRmetrics 0.1.5** for the 24 lidRmetrics metrics and the tree summaries
listed below (and those of leafR, PyForestScan, forestr, ForestGapR and canopyLazR for their metrics), to
floating-point rounding on every float64 backend.

```python
from lidarmetricpy import compute_tile, compute_tiles, Options

df = compute_tile("tile.laz", x0=-62000, y1=-57750, ncol=40, nrow=30, cell=25,
                  dtm="dtm.tif",            # or None if Z is already height above ground
                  backend="auto",           # cupy | torch-cuda | mps | numpy | torch-cpu
                  gpu_memory_gb=8,          # GPU memory budget (GiB); no effect on CPU backends
                  cpu_memory_gb=16,         # RAM budget (GiB); large tiles are read in several passes
                  options=Options.lidr())   # lidR-identical settings (see "Preprocessing")
```

`df` has one row per cell with at least one point after the preprocessing (`x`, `y` = cell centre), and the
same columns whatever the data (`lidarmetricpy.api.output_columns(options)`). `n_points` counts the points of
the cell; `n` counts those above `zmin` and is NA when there is none, as in lidRmetrics.
`compute_tiles(jobs, workers="auto", gpu_memory_gb=80, cpu_memory_gb=200)` runs many tiles in parallel
processes that share one GPU; the budgets are totals, split equally between the workers.

## Install

```bash
pip install "lidarmetricpy[cuda] @ git+https://github.com/hideyuki-doi/lidarmetricpy"     # NVIDIA, CuPy
pip install "lidarmetricpy[torch] @ git+https://github.com/hideyuki-doi/lidarmetricpy"    # PyTorch (CUDA, MPS or CPU)
pip install "lidarmetricpy @ git+https://github.com/hideyuki-doi/lidarmetricpy"           # NumPy only
```

Python ≥ 3.10. Core dependencies: numpy, scipy, pandas, laspy with lazrs, numba, shapely, rasterio.

## Backends and memory budgets

| backend | device | precision | budgets used |
|---|---|---|---|
| `cupy`, `torch-cuda` | NVIDIA GPU | float64 | `gpu_memory_gb` (GPU), `cpu_memory_gb` (raw points in RAM) |
| `mps` | Apple Silicon GPU | float32 | both; they share the unified memory |
| `numpy`, `torch-cpu` | CPU | float64 | `cpu_memory_gb` (raw points and working set) |

`backend="auto"` takes CuPy if a CUDA device is present, then PyTorch CUDA, then PyTorch MPS, then NumPy.
Budgets are in GiB (1024³ bytes); `None` uses 70 % of the free GPU memory and 60 % of the free RAM.
Points are processed in blocks of whole cells, so any budget gives the same values; on a GPU the whole tile
is prepared on the device when it fits, otherwise points are prepared in RAM and sent block by block.

What a budget bounds: the point arrays (raw points in RAM, prepared points, LAZ read chunks and the blocks).
It does not include fixed costs: Python and the libraries, the CUDA context (~0.7 GiB), the canopy raster of
the tile (8 bytes per 0.5 m pixel, ~24 MB for 1 km × 0.75 km) and its fill, qhull and numba compilation.
Measured GPU peaks stay under the budget from 2 to 16 GiB on a 90 M point tile (docs/data/memory_budget.csv).
Budgets below 8 MiB raise `ValueError`. Blocks hold whole cells: a cell holding more points than the budget
allows is processed as a block of its own, above the budget, with a warning (and `stats["cells_over_budget"]`).
The old argument names `max_memory_gb` / `host_memory_gb` still work and give a `DeprecationWarning`.

## Preprocessing

The defaults clean the points before the metrics: heights outside `zrange = (-1, 60)` are dropped, and
`spike_filter` drops the top point of a 1 m column when it is more than 8 m above the next one. lidR does
neither, so `Options.lidr()` (= `Options(spike_filter=False, zrange=(-inf, inf))`) is the setting to compare
with lidR; the defaults may remove real trees taller than 60 m or isolated crown tops.

With `dtm=`, heights are z minus the value of the DTM pixel that contains the point: column
floor((x - x_dtm) / res) and row floor((y_dtm - y) / res) on the DTM's own grid, as lidR does (north-up
rasters), whatever the origin of the metric grid. Points where the DTM has no data (nodata value, mask, outside
the raster) are dropped.

## Grid

Cells are `ncol × nrow` squares of `cell` m with the top-left corner at (`x0`, `y1`). As in lidR (terra), a
point on the right or bottom outer edge belongs to the last column or row. For tiles that share edges and
whose files overlap, `Options(closed_edges=False)` makes the grid half-open, [x0, x1) × (y0, y1], so that
no point is counted twice. Tree detection uses the points inside the grid; lidR's `rasterize_canopy` uses
every point of the file, which is the same when the file covers the grid.

## Metrics (per grid cell)

24 of the 111 outputs of `lidRmetrics::metrics_set3(zmin = 0.5, threshold = c(2, 5, 10, 20, 30),
zintervals = c(0, 0.5, 2, 5, 10, 20, 30), pixel_size = 1, vox_size = 1)`, with the same definitions and edge
cases (intensity metrics are not implemented):

| lidRmetrics function | columns |
|---|---|
| metrics_basic | n, zcv |
| metrics_percentiles | zq20 |
| metrics_dispersion | CRR, zentropy |
| metrics_canopydensity | zpcum7, zpcum9 |
| metrics_interval | pz_above_30 |
| metrics_Lmoments | L4, Lskew, Lkurt |
| metrics_lad (lidR::LAD, dz = 1, k = 0.5, z0 = 2) | lad_max, lad_mean, lad_cv |
| metrics_echo | p_single, p_last, ratio_last_first |
| metrics_voxels | OpenGapSpace, ClosedGapSpace, Euphotic, Oligophotic, vzsd, vzcv |
| metrics_kde (bw = 2) | kde_peaks_count |
| metrics_rumple | rumple |

Tree summaries per cell from the lidR workflow `rasterize_canopy(res = 0.5, p2r(0.15, knnidw()))` →
`locate_trees(lmf(ws = 2.5 + 0.15 h, clamped 3–8 m, hmin = 5))` → `dalponte2016(th_tree = 2)`:
tree_n, tree_h_mean / sd / cv / max / gini / shannon, crown_m2_mean / sd (the summaries are ours; tree tops
and crowns equal lidR's).

Metrics defined by other packages (`Options(other_packages=True)`, the default; `enl` is always computed),
each equal to the package applied to the points of one cell:

| package | columns |
|---|---|
| leafR 0.3.5 (`lad.voxels(k = 1)` with the cell as one pixel → `lad.profile` → `lai`, `LAHV`, `FHD`, `GS`, `GC`) | leafr_lai, leafr_lahv, leafr_lahv_laiw, leafr_lahv_hw, leafr_lahv_laiw_hw, leafr_fhd, leafr_fhd_even, leafr_gs, leafr_gs_even, leafr_gc |
| PyForestScan 0.3.6 (`assign_voxels(cell, cell, 1)` → `calculate_pad` → `calculate_pai(min 1 m)`, `calculate_canopy_cover(min 2 m, k = 0.5)`, `calculate_fhd`) | pfs_pai, pfs_cover, pfs_fhd |
| forestr 2.0.2 (`calc_enl` on the filled 1 m voxels of lidRmetrics) | enl |
| ForestGapR 0.1.7 (`getForestGaps(chm, threshold = 10, size = c(1, 1e4))`; per-cell summary is ours) | gap_frac, gap_n, gap_area_mean |

canopyLazR 2.0 works on elevations and on its own grid, so it is a separate CPU function:

```python
from lidarmetricpy import canopylazr_file
r = canopylazr_file("raw_tile.laz", xy_res=10, z_res=1, ht_cut=5)   # DataFrame: x, y, clz_* rasters
```

Details reproduced from the R/C++ sources so that results match to floating-point rounding:
C `round()` (halves away from zero) in `round_any` and in lidR's raster layouts (also the per-cell 1 m layout of
`highest()` for rumple), R's `round(x, 3)`, `seq()` / `findInterval` bin edges, `hist()` boundary fuzz (LAD,
`metrics_interval`), a bit-exact port of `stats::density` (BinDist in point order, `dnorm`, R's FFT, `approx`)
for the peak count, qhull Delaunay (`QbB Qt`) for rumple, the lmf tie rule, Dalponte's in-pass updates, and
lidR's knn search order with libstdc++ `std::sort` for knnidw ties. docs/tables/table5_reproduction_details.md
lists them.

Known differences: when all heights above zmin in a cell are equal, R's Lmoments package returns rounding noise
for Lskew and Lkurt (NaN, ±Inf or any number, since L2 = 0); lidarmetricpy returns NA. Float64 backends differ
from each other by ~1e-14 in sums over many points. The float32 path (Apple MPS) is not exact (see below).

## Tests

```bash
pip install pytest
pytest            # ~20 s; GPU backends are skipped when their device cannot be used
```

- tests/test_reference.py: a synthetic 100 m × 100 m tile (tests/make_synthetic.py: tree crowns, multiple
  returns, a gap, an empty cell, points on cell edges) against the reference values of lidR, lidRmetrics,
  leafR, forestr, ForestGapR, PyForestScan and canopyLazR (tests/make_references.sh; versions in
  tests/data/ref/versions.txt).
- tests/test_oracle.py: 17 single-cell edge cases (few points, equal heights, invalid returns, heights 1e-12
  from a break, tall canopies, zmin = 2) against `metrics_set3` run in R (tests/make_oracle.R).
- tests/test_regressions.py: cells computed together, alone or in blocks must agree (also for grids not aligned
  with whole metres and cells of 2.5 or 7.3 m); ground-only cells; DTM alignment and missing data; exact hull;
  sparse clouds; host and device paths; the strict comparator; budgets.
- tests/test_budgets.py: memory budgets, multi-pass reading and the device / host paths give identical values.

Comparisons use `lidarmetricpy.validate.compare_tables`: unique cell keys; the cells that must be present are
decided apart from the metric values (`occupied`, the cells holding points counted from the LAZ with
`occupied_cells`), so a missing cell fails even when its metrics are NA; an extra cell fails; non-numeric
values raise; the NA / +Inf / -Inf pattern must be equal, then finite values within 1e-6 relative; two empty
tables give status "empty", not equal.

## Validation on real data (validation/, docs/)

Kanagawa Prefecture 2024 airborne lidar (52–133 pts/m², up to 7 returns, JGD2011 Plane IX). The LAZ files are
not redistributed here; validation/make_testdata.py, make_fulltile.py and make_rawclips.py rebuild the test
clips from the original tiles (`PCM_LAZ_DIR`, `PCM_DEM_DIR`). The reference outputs of the R and Python
packages are included (validation/ref, validation/ref_full).

Checked with the strict comparator (no missing or extra cell, same NA pattern, values within 1e-6 relative):

| data | cells | trees | result |
|---|---|---|---|
| 4 clips, 52–133 pts/m² | 400 | 4,275 | 33/33 metrics equal in every cell (max relative diff 2.4e-12); all tree tops and crown areas equal — cupy, torch-cuda, torch-cpu, numpy |
| 2 full tiles (90 M and 48 M points) | 2,400 | 23,559 | all metrics, tree tops and crown areas equal (all four float64 backends) |
| other packages, 4 clips + 2 full tiles | 400 + 2,400 | – | leafR / PyForestScan / forestr ENL / ForestGapR: 17/17 equal on all backends (max diff 3.4e-12) |
| canopyLazR, 4 raw clips × 10 m and 5 m | 12,600 columns | – | all 9 rasters equal (max diff 3.1e-11) |
| float32 (MPS-like, tested as torch-cuda32) | 2,800 | 27,834 | trees, crowns and the 17 other-package metrics equal; 28/33 lidRmetrics metrics equal in every cell (kde_peaks_count is computed in float64); zpcum7/9 within 0.03 % points, L4, Lkurt, lad_cv within 1.3e-5 |

Apple MPS itself has not been run; the float32 path was tested on CUDA. Figures and tables for a methods
paper are in docs/ (docs/make_paper_figs.py), with notes in Japanese in docs/paper_notes.md.

## Speed and memory

Measured on an RTX PRO 6000 Blackwell Max-Q 96 GB and an AMD Threadripper PRO 7955WX (16 cores / 32 threads),
lidRmetrics metrics + trees (`Options.lidr(other_packages=False)`), one tile per process, nothing else running.
lidR was run twice with R 4.5.3: with `metrics_set3` (111 outputs, validation/ref_lidr.R) and with only the
lidRmetrics functions the 24 metrics need (validation/ref_lidr_subset.R; same 24 values). Two intervals, the
same on both sides (validation/run_timing_lidr.sh, run_timing_py.sh, backend_compare.py):

- **work**: reading the LAZ, metrics and trees inside one process (Python: `compute_tile` after a warm-up call;
  R: read_s + area_s + tree_s printed by the script).
- **whole process**: start, imports / package loading, work, and saving the cell table, tree tops and canopy
  raster (`/usr/bin/time`, fresh process; Python without a warm-up, so it includes loading compiled code).

| one tile (90.5 M / 48.3 M points) | work (s) | whole process (s) | peak GPU GiB | peak RAM GiB |
|---|---|---|---|---|
| CuPy | 5.5 / 3.8 | 6.9 / 4.9 | 22 / 13 | 2.2 / 2.2 |
| PyTorch CUDA | 5.5 / 3.8 | – | 22 / 13 | 2.9 / 2.9 |
| PyTorch CUDA float32 (MPS-like) | 20.4 / 12.4 | – | 15 / 9 | 11.2 / 6.8 |
| NumPy (CPU) | 53.4 / 31.2 | 54.4 / 32.4 | – | 23.0 / 13.8 |
| PyTorch CPU | 48.6 / 29.0 | – | – | 23.6 / 14.2 |
| lidR + lidRmetrics, 24 metrics (R, 1 core) | 165.7 / 133.9 | 205.3 / 156.2 | – | 9.6 / 5.3 |
| lidR + lidRmetrics, metrics_set3 (R, 1 core) | 174.9 / 140.8 | 214.6 / 163.3 | – | 9.6 / 5.3 |

For the same 24 metrics, CuPy is ~30 times faster than lidR in either interval (work 165.7 / 5.5 s, whole
process 205.3 / 6.9 s), NumPy 3–4 times (165.7 / 53.4 s, 205.3 / 54.4 s).

| many tiles | seconds per tile |
|---|---|
| CuPy, 1 / 4 / 12 workers | 4.59 / 1.72 / 1.44 |
| PyTorch CUDA, 12 workers | 1.53 |
| NumPy, 1 / 4 / 8 / 16 workers | 41.8 / 18.5 / 14.6 / 10.4 |
| lidR (metrics_set3, R 4.4.3), 8 R processes | 30.5 |

lidR on all cores was not measured. On the GPU path the GPU work is ~1 s per dense tile; the rest is CPU
work (LAZ decoding alone costs 23 CPU-seconds per dense tile and is the floor for throughput; the exact
`density()` port for kde_peaks_count runs on the CPU, 0.33 s measured on the dense tile).

## License

GPL-3.0-or-later (LICENSE): the tree detection, Dalponte segmentation, knnidw and spatial-index code are ports
of lidR (GPL-3, Jean-Romain Roussel). Metric definitions follow lidRmetrics (Tompalski et al.), leafR
(Almeida et al. 2019), PyForestScan (JOSS 2025), forestr (Atkins et al. 2018), ForestGapR
(Silva et al. 2019) and canopyLazR (Kamoske et al. 2019); see docs/paper_notes.md §9 for references.
