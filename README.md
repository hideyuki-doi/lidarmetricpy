# pcmetrics

Forest structure metrics from airborne lidar point clouds, **identical to lidR 4.3 + lidRmetrics 0.1.5**
(and to leafR, PyForestScan, forestr, ForestGapR and canopyLazR for their metrics), on NVIDIA GPUs (CuPy or
PyTorch), Apple Silicon GPUs (PyTorch MPS, float32) or any CPU (NumPy), with explicit memory budgets.

```python
from pcmetrics import compute_tile, compute_tiles, Options

df = compute_tile("tile.laz", x0=-62000, y1=-57750, ncol=40, nrow=30, cell=25,
                  dtm="dtm.tif",            # or None if Z is already height above ground
                  backend="auto",           # cupy | torch-cuda | mps | numpy | torch-cpu
                  gpu_memory_gb=8,          # GPU memory budget (GiB); no effect on CPU backends
                  cpu_memory_gb=16)         # RAM budget (GiB); large tiles are read in several passes
```

`df` has one row per 25 m cell that has points (`x`, `y` = cell centre) and one column per metric.
`compute_tiles(jobs, workers="auto", gpu_memory_gb=80, cpu_memory_gb=200)` runs many tiles in parallel
processes that share one GPU; the budgets are totals, split equally between the workers.

## Install

```bash
pip install "pcmetrics[cuda] @ git+https://github.com/hideyuki-doi/pcmetrics"     # NVIDIA, CuPy
pip install "pcmetrics[torch] @ git+https://github.com/hideyuki-doi/pcmetrics"    # PyTorch (CUDA, MPS or CPU)
pip install "pcmetrics @ git+https://github.com/hideyuki-doi/pcmetrics"           # NumPy only
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
is prepared on the device when it fits (measured peak ≤ budget from 2 to 16 GiB on a 90 M point tile), otherwise
points are prepared in RAM and sent block by block. The old argument names `max_memory_gb` / `host_memory_gb`
still work and give a `DeprecationWarning`.

## Grid

Cells are `ncol × nrow` squares of `cell` m with the top-left corner at (`x0`, `y1`). As in lidR (terra), a
point on the right or bottom outer edge belongs to the last column or row. For tiles that share edges and
whose files overlap, `Options(closed_edges=False)` makes the grid half-open, [x0, x1) × (y0, y1], so that
no point is counted twice. Tree detection uses the points inside the grid; lidR's `rasterize_canopy` uses
every point of the file, which is the same when the file covers the grid.

## Metrics (per grid cell)

Same definitions and edge cases as `lidRmetrics::metrics_set3` (zmin = 0.5) and the lidR tree workflow
`rasterize_canopy(p2r(0.15, knnidw()))` → `locate_trees(lmf(ws = 2.5 + 0.15 h, clamped 3–8 m, hmin = 5))` → `dalponte2016()`:

zq20, zcv, CRR, zentropy, zpcum7, zpcum9, pz_above_30, L4, Lskew, Lkurt, lad_max, lad_mean, lad_cv,
p_single, p_last, ratio_last_first, OpenGapSpace, ClosedGapSpace, Euphotic, Oligophotic, vzsd, vzcv,
kde_peaks_count, rumple, tree_n, tree_h_mean / sd / cv / max / gini / shannon, crown_m2_mean / sd.

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
from pcmetrics import canopylazr_file
r = canopylazr_file("raw_tile.laz", xy_res=10, z_res=1, ht_cut=5)   # DataFrame: x, y, clz_* rasters
```

Details reproduced from the R/C++ sources so that results match to floating-point rounding:
C `round()` (halves away from zero) in `round_any` and in lidR's raster layouts, R's `round(x, 3)`,
`seq()` / `findInterval` bin edges, `stats::density` (linear binning + FFT), qhull Delaunay (`QbB Qt`) for
rumple, the lmf tie rule, Dalponte's in-pass updates, and lidR's knn search order with libstdc++
`std::sort` for knnidw ties. docs/tables/table5_reproduction_details.md lists them.

## Tests

```bash
pip install pytest
pytest            # ~10 s; GPU backends are skipped when not installed
```

tests/data holds a synthetic 100 m × 100 m tile (made by tests/make_synthetic.py: tree crowns, multiple
returns, a gap, an empty cell, points on cell edges) and the reference values of lidR, lidRmetrics, leafR,
forestr, ForestGapR, PyForestScan and canopyLazR for it (tests/make_references.sh). The tests require every
float64 backend to reproduce every reference value and tree top, and every memory budget and processing
path to give identical results.

## Validation on real data (validation/, docs/)

Kanagawa Prefecture 2024 airborne lidar (52–133 pts/m², up to 7 returns, JGD2011 Plane IX). The LAZ files are
not redistributed here; validation/make_testdata.py, make_fulltile.py and make_rawclips.py rebuild the test
clips from the original tiles (`PCM_LAZ_DIR`, `PCM_DEM_DIR`). The reference outputs of the R and Python
packages are included (validation/ref, validation/ref_full).

| data | cells | trees | result |
|---|---|---|---|
| 4 clips, 52–133 pts/m² | 400 | 4,275 | 33/33 metrics equal in every cell (max diff 2.3e-12); all tree tops and crown areas equal — cupy, torch-cuda, torch-cpu, numpy |
| 2 full tiles (90 M and 48 M points) | 2,400 | 23,559 | all metrics, tree tops and crown areas equal (all four float64 backends) |
| other packages, 4 clips + 2 full tiles | 400 + 2,400 | – | leafR / PyForestScan / forestr ENL / ForestGapR: 17/17 equal on all backends (max diff 3.4e-12) |
| canopyLazR, 4 raw clips × 10 m and 5 m | 12,600 columns | – | all 9 rasters equal (max diff 3.1e-11) |
| float32 (MPS-like, tested as torch-cuda32) | 2,800 | 27,834 | trees, crowns and the 17 other-package metrics equal; 27/33 lidRmetrics metrics equal in every cell; zpcum7/9 within 0.03 % points, L4, Lkurt, lad_cv within 1.3e-5, kde_peaks_count off by one peak in 2 cells |

Apple MPS itself has not been run; the float32 path was tested on CUDA. Figures and tables for a methods
paper are in docs/ (docs/make_paper_figs.py), with notes in Japanese in docs/paper_notes.md.

## Speed and memory

Measured on an RTX PRO 6000 Blackwell Max-Q 96 GB and an AMD Threadripper PRO 7955WX (16 cores / 32 threads),
lidRmetrics set + trees, on a frozen copy of the package (validation/backend_compare/).

| one tile, one process (90.5 M / 48.3 M points) | seconds | peak GPU GiB | peak RAM GiB |
|---|---|---|---|
| CuPy | 5.2 / 3.6 | 25 / 16 | 2.3 / 2.3 |
| PyTorch CUDA | 5.1 / 3.4 | 24 / 15 | 2.9 / 2.9 |
| PyTorch CUDA float32 (MPS-like) | 18.2 / 10.9 | 18 / 11 | 10.4 / 6.4 |
| NumPy (CPU) | 50.6 / 30.1 | – | 25.7 / 15.5 |
| PyTorch CPU | 45.5 / 27.6 | – | 26.7 / 16.3 |
| lidR + lidRmetrics (R, 1 core) | 230.9 / 179.4 | – | 9.6 / 5.3 |

| many tiles | seconds per tile |
|---|---|
| CuPy, 1 / 4 / 12 workers | 4.40 / 1.71 / 1.33 |
| PyTorch CUDA, 12 workers | 1.48 |
| NumPy, 1 / 4 / 8 / 16 workers | 39.8 / 16.5 / 13.4 / 9.6 |
| lidR, 8 R processes | 30.5 |

lidR on all cores was not measured. On the GPU path the GPU work is ~1 s per dense tile; the rest is CPU
work. LAZ decoding alone costs 23 CPU-seconds per dense tile and is the floor for throughput.

## License

GPL-3.0-or-later (LICENSE): the tree detection, Dalponte segmentation, knnidw and spatial-index code are ports
of lidR (GPL-3, Jean-Romain Roussel). Metric definitions follow lidRmetrics (Tompalski et al.), leafR
(Almeida et al. 2019), PyForestScan (JOSS 2025), forestr (Atkins et al. 2018), ForestGapR
(Silva et al. 2019) and canopyLazR (Kamoske et al. 2019); see docs/paper_notes.md §9 for references.
