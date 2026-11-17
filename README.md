# lidarmetricpy

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24111785.svg)](https://doi.org/10.5281/zenodo.24111785)

> **AI coding assistants** (Claude Code, Codex, Cursor, …): read [AGENTS.md](AGENTS.md) before using or changing
> this package. Humans may find it a useful checklist too.

lidarmetricpy computes forest-structure metrics for every cell of a grid (typically 25 m) from airborne lidar
point clouds, on NVIDIA GPUs (CuPy or PyTorch) or any CPU (NumPy or PyTorch), within memory limits that you set.
Apple Silicon GPUs (PyTorch MPS) are supported in float32 but have not yet been run on Apple hardware.

It produces **146 outputs**:

- **137 metrics per grid cell** (`compute_tile`, `compute_tiles`):
  - all 111 outputs of `lidRmetrics::metrics_set3` (height distribution, cover, layers, L-moments, leaf area
    density, return types, canopy roughness, voxels, density peaks);
  - 9 tree summaries (number, heights and crown areas of the trees whose tops lie in the cell), computed from
    lidR's tree tops and Dalponte crowns;
  - 17 metrics defined by leafR (10), PyForestScan (3), forestr (1) and ForestGapR (3).
- **9 canopyLazR rasters** (`canopylazr_file`, a separate CPU function on canopyLazR's own grid).

With `Options.lidr()` each output equals the value of the package that defines it (lidR 4.3.3 + lidRmetrics
0.1.5, leafR 0.3.5, PyForestScan 0.3.6, forestr 2.0.2, ForestGapR 0.1.7, canopyLazR 2.0), to floating-point
rounding, on every float64 backend (canopyLazR runs on the CPU). For the tree and gap summaries, the tree tops, crowns and gaps are those of
lidR and ForestGapR, and the per-cell summaries are defined by lidarmetricpy.
**[List of all 146 outputs with their meaning](docs/tables/table1_metrics.md)**.

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

## Quick start

```bash
python examples/quickstart.py          # metrics of the synthetic test tile on a 10 m grid → examples/quickstart.png
```

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
On the device path, LAZ is decoded in chunks that fit both budgets, so each transfer to the GPU fits the GPU
budget.
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
rasters), whatever the origin of the metric grid; a point exactly on the right or bottom edge of the DTM takes
its last column or row, as in lidR. Points where the DTM has no data (nodata value, mask) are dropped, and so
are points outside the raster: there lidR's `normalize_height` takes the nearest pixel with a warning, so make
the DTM cover the tile.

## Grid

Cells are `ncol × nrow` squares of `cell` m with the top-left corner at (`x0`, `y1`). As in lidR (terra), a
point on the right or bottom outer edge belongs to the last column or row. For tiles that share edges and
whose files overlap, `Options(closed_edges=False)` makes the grid half-open, [x0, x1) × (y0, y1], so that
no point is counted twice. Tree detection uses the points inside the grid; lidR's `rasterize_canopy` uses
every point of the file, which is the same when the file covers the grid.

## Metrics

The output table has `x`, `y` (cell centre) and `n_points` (all points of the cell) and then the 137 metrics
below, in this order; canopyLazR's 9 rasters come from a separate function. The full list with a short meaning,
the definition and the points used is at the end of this section.

### lidRmetrics (111 columns)

All 111 outputs of `lidRmetrics::metrics_set3(zmin = 0.5, threshold = c(2, 5, 10, 20, 30),
zintervals = c(0, 0.5, 2, 5, 10, 20, 30), pixel_size = 1, vox_size = 1)`, with the same names, definitions and
edge cases (column order as in lidRmetrics; `lidarmetricpy.area.SET3_COLUMNS`):

| lidRmetrics function | columns |
|---|---|
| metrics_basic | n, zmax, zmin, zmean, zvar, zsd, zcv, zskew, zkurt |
| metrics_percentiles | zq1, zq5, zq10, …, zq95, zq99 |
| metrics_percabove | pzabovemean, pzabove2, pzabove5, pzabove10, pzabove20, pzabove30 |
| metrics_dispersion | ziqr, zMADmean, zMADmedian, CRR, zentropy |
| metrics_canopydensity | zpcum1 … zpcum9 |
| metrics_Lmoments | L1, L2, L3, L4, Lskew, Lkurt, Lcoefvar |
| metrics_lad (lidR::LAD, dz = 1, k = 0.5, z0 = 2) | lad_min, lad_max, lad_mean, lad_cv, lad_sum |
| metrics_interval | pz_below_0, pz_0-0.5, pz_0.5-2, pz_2-5, pz_5-10, pz_10-20, pz_20-30, pz_above_30 |
| metrics_echo | n_first, n_intermediate, n_last, n_single, n_multiple, p_first, p_intermidiate (sic), p_last, p_single, p_multiple, ratio_last_first, ratio_intermediate_first, ratio_multiple_single |
| metrics_echo2 (KeepReturns = 1:4) | n_return_1 … n_return_4 |
| metrics_rumple | rumple |
| metrics_voxels | vn, vFRall, vFRcanopy, vzrumple, vzsd, vzcv, OpenGapSpace, ClosedGapSpace, Euphotic, Oligophotic |
| metrics_kde (bw = 2, npeaks = 4) | kde_peaks_count, kde_peak1–4_elev, kde_peak1–4_value, kde_peak1–3_diff |
| metrics_HOME | HOME (height of median energy; uses the point intensities, 1 for in-memory input without them) |

lidRmetrics' own quirks are reproduced, not fixed: `n_return_k` counts the k-th smallest return number present
in the cell (metrics_echo2 converts the table's factor to its level codes), which equals the count of
ReturnNumber == k only when return numbers 1..k all occur; `p_intermidiate` keeps its spelling.

### Tree summaries (9 columns)

Tree summaries per cell from the lidR workflow `rasterize_canopy(res = 0.5, p2r(0.15, knnidw()))` →
`locate_trees(lmf(ws = 2.5 + 0.15 h, clamped 3–8 m, hmin = 5))` → `dalponte2016(th_tree = 2)`:
tree_n, tree_h_mean / sd / cv / max / gini / shannon, crown_m2_mean / sd. Tree tops and crowns equal lidR's; a
tree belongs to the cell that contains its top, and the per-cell summaries are defined by lidarmetricpy
(`Options(trees=True)`, the default).

### Metrics of other packages (17 columns)

Metrics defined by other packages (`Options(other_packages=True)`, the default; `enl` is always computed),
each equal to the package applied to the points of one cell:

| package | columns |
|---|---|
| leafR 0.3.5 (`lad.voxels(k = 1)` with the cell as one pixel → `lad.profile` → `lai`, `LAHV`, `FHD`, `GS`, `GC`) | leafr_lai, leafr_lahv, leafr_lahv_laiw, leafr_lahv_hw, leafr_lahv_laiw_hw, leafr_fhd, leafr_fhd_even, leafr_gs, leafr_gs_even, leafr_gc |
| PyForestScan 0.3.6 (`assign_voxels(cell, cell, 1)` → `calculate_pad` → `calculate_pai(min 1 m)`, `calculate_canopy_cover(min 2 m, k = 0.5)`, `calculate_fhd`) | pfs_pai, pfs_cover, pfs_fhd |
| forestr 2.0.2 (`calc_enl` on the filled 1 m voxels of lidRmetrics) | enl |
| ForestGapR 0.1.7 (`getForestGaps(chm, threshold = 10, size = c(1, 1e4))`; per-cell summary by lidarmetricpy) | gap_frac, gap_n, gap_area_mean |

### canopyLazR rasters (9, separate function)

canopyLazR 2.0 works on elevations and on its own grid, so it is a separate CPU function:

```python
from lidarmetricpy import canopylazr_file
r = canopylazr_file("raw_tile.laz", xy_res=10, z_res=1, ht_cut=5)   # DataFrame: x, y, clz_* rasters
```

### Full list

[docs/tables/table1_metrics.md](docs/tables/table1_metrics.md) (and [.csv](docs/tables/table1_metrics.csv)) lists
all 146 outputs (137 per cell, 9 canopyLazR rasters) one per row, with a short meaning, the package, function,
definition and points used.

<!-- metrics table: begin (docs/make_metrics_table.py) -->
<details>
<summary>All 146 outputs with a short meaning (click to open; definitions in <a href="docs/tables/table1_metrics.md">docs/tables/table1_metrics.md</a>)</summary>

| no | metric | meaning | package |
|---|---|---|---|
| 1 | n | how many returns lie above 0.5 m (vegetation returns) | lidRmetrics 0.1.5 |
| 2 | zmax | top height of the vegetation | lidRmetrics 0.1.5 |
| 3 | zmin | lowest vegetation return above 0.5 m | lidRmetrics 0.1.5 |
| 4 | zmean | average height of the vegetation returns | lidRmetrics 0.1.5 |
| 5 | zvar | spread of return heights (variance) | lidRmetrics 0.1.5 |
| 6 | zsd | spread of return heights | lidRmetrics 0.1.5 |
| 7 | zcv | spread of return heights relative to their mean | lidRmetrics 0.1.5 |
| 8 | zskew | whether returns are concentrated low (positive) or high (negative) in the canopy | lidRmetrics 0.1.5 |
| 9 | zkurt | how peaked the height distribution is (3 = normal) | lidRmetrics 0.1.5 |
| 10 | zq1 | height below which 1 % of the returns lie | lidRmetrics 0.1.5 |
| 11 | zq5 | height below which 5 % of the returns lie | lidRmetrics 0.1.5 |
| 12 | zq10 | height below which 10 % of the returns lie | lidRmetrics 0.1.5 |
| 13 | zq15 | height below which 15 % of the returns lie | lidRmetrics 0.1.5 |
| 14 | zq20 | height below which 20 % of the returns lie | lidRmetrics 0.1.5 |
| 15 | zq25 | height below which 25 % of the returns lie | lidRmetrics 0.1.5 |
| 16 | zq30 | height below which 30 % of the returns lie | lidRmetrics 0.1.5 |
| 17 | zq35 | height below which 35 % of the returns lie | lidRmetrics 0.1.5 |
| 18 | zq40 | height below which 40 % of the returns lie | lidRmetrics 0.1.5 |
| 19 | zq45 | height below which 45 % of the returns lie | lidRmetrics 0.1.5 |
| 20 | zq50 | height below which 50 % of the returns lie (median height) | lidRmetrics 0.1.5 |
| 21 | zq55 | height below which 55 % of the returns lie | lidRmetrics 0.1.5 |
| 22 | zq60 | height below which 60 % of the returns lie | lidRmetrics 0.1.5 |
| 23 | zq65 | height below which 65 % of the returns lie | lidRmetrics 0.1.5 |
| 24 | zq70 | height below which 70 % of the returns lie | lidRmetrics 0.1.5 |
| 25 | zq75 | height below which 75 % of the returns lie | lidRmetrics 0.1.5 |
| 26 | zq80 | height below which 80 % of the returns lie | lidRmetrics 0.1.5 |
| 27 | zq85 | height below which 85 % of the returns lie | lidRmetrics 0.1.5 |
| 28 | zq90 | height below which 90 % of the returns lie | lidRmetrics 0.1.5 |
| 29 | zq95 | height below which 95 % of the returns lie (≈ top height) | lidRmetrics 0.1.5 |
| 30 | zq99 | height below which 99 % of the returns lie (≈ top height) | lidRmetrics 0.1.5 |
| 31 | pzabovemean | share of returns above the average height | lidRmetrics 0.1.5 |
| 32 | pzabove2 | share of returns above 2 m (canopy cover above 2 m) | lidRmetrics 0.1.5 |
| 33 | pzabove5 | share of returns above 5 m (canopy cover above 5 m) | lidRmetrics 0.1.5 |
| 34 | pzabove10 | share of returns above 10 m (canopy cover above 10 m) | lidRmetrics 0.1.5 |
| 35 | pzabove20 | share of returns above 20 m (canopy cover above 20 m) | lidRmetrics 0.1.5 |
| 36 | pzabove30 | share of returns above 30 m (canopy cover above 30 m) | lidRmetrics 0.1.5 |
| 37 | ziqr | height range of the middle half of the returns | lidRmetrics 0.1.5 |
| 38 | zMADmean | average distance of return heights from their mean | lidRmetrics 0.1.5 |
| 39 | zMADmedian | typical distance of return heights from their median (robust spread) | lidRmetrics 0.1.5 |
| 40 | CRR | where the mean height lies between lowest and highest return (0 = low, 1 = top); canopy shape | lidRmetrics 0.1.5 |
| 41 | zentropy | how evenly returns spread over 1 m layers (0 = one layer, 1 = all layers equal) | lidRmetrics 0.1.5 |
| 42 | zpcum1 | share of returns in the lower 10 % of the height range (profile shape) | lidRmetrics 0.1.5 |
| 43 | zpcum2 | share of returns in the lower 20 % of the height range (profile shape) | lidRmetrics 0.1.5 |
| 44 | zpcum3 | share of returns in the lower 30 % of the height range (profile shape) | lidRmetrics 0.1.5 |
| 45 | zpcum4 | share of returns in the lower 40 % of the height range (profile shape) | lidRmetrics 0.1.5 |
| 46 | zpcum5 | share of returns in the lower 50 % of the height range (profile shape) | lidRmetrics 0.1.5 |
| 47 | zpcum6 | share of returns in the lower 60 % of the height range (profile shape) | lidRmetrics 0.1.5 |
| 48 | zpcum7 | share of returns in the lower 70 % of the height range (profile shape) | lidRmetrics 0.1.5 |
| 49 | zpcum8 | share of returns in the lower 80 % of the height range (profile shape) | lidRmetrics 0.1.5 |
| 50 | zpcum9 | share of returns in the lower 90 % of the height range (profile shape) | lidRmetrics 0.1.5 |
| 51 | L1 | mean height (robust moment) | lidRmetrics 0.1.5 |
| 52 | L2 | spread of heights (robust moment) | lidRmetrics 0.1.5 |
| 53 | L3 | asymmetry of heights (robust moment, unscaled) | lidRmetrics 0.1.5 |
| 54 | L4 | peakedness of heights (robust moment, unscaled) | lidRmetrics 0.1.5 |
| 55 | Lskew | asymmetry of the height distribution (robust) | lidRmetrics 0.1.5 |
| 56 | Lkurt | peakedness of the height distribution (robust) | lidRmetrics 0.1.5 |
| 57 | Lcoefvar | spread of heights relative to the mean (robust) | lidRmetrics 0.1.5 |
| 58 | lad_min | leaf area density of the sparsest 1 m layer | lidRmetrics 0.1.5 |
| 59 | lad_max | leaf area density of the densest 1 m layer | lidRmetrics 0.1.5 |
| 60 | lad_mean | average leaf area density over the layers | lidRmetrics 0.1.5 |
| 61 | lad_cv | how unequal leaf area is among layers | lidRmetrics 0.1.5 |
| 62 | lad_sum | total leaf area over the layers (an LAI-like value) | lidRmetrics 0.1.5 |
| 63 | pz_below_0 | share of returns below ground (0 here because only returns above 0.5 m are used) | lidRmetrics 0.1.5 |
| 64 | pz_0-0.5 | share of returns between 0 and 0.5 m | lidRmetrics 0.1.5 |
| 65 | pz_0.5-2 | share of returns between 0.5 and 2 m | lidRmetrics 0.1.5 |
| 66 | pz_2-5 | share of returns between 2 and 5 m | lidRmetrics 0.1.5 |
| 67 | pz_5-10 | share of returns between 5 and 10 m | lidRmetrics 0.1.5 |
| 68 | pz_10-20 | share of returns between 10 and 20 m | lidRmetrics 0.1.5 |
| 69 | pz_20-30 | share of returns between 20 and 30 m | lidRmetrics 0.1.5 |
| 70 | pz_above_30 | share of returns in the very tall canopy (30 m and above) | lidRmetrics 0.1.5 |
| 71 | n_first | number of first returns (pulses that hit something) | lidRmetrics 0.1.5 |
| 72 | n_intermediate | number of returns between first and last of a pulse (passed through the canopy) | lidRmetrics 0.1.5 |
| 73 | n_last | number of last returns of pulses with several returns | lidRmetrics 0.1.5 |
| 74 | n_single | number of returns from pulses with one return (solid surfaces, dense crowns) | lidRmetrics 0.1.5 |
| 75 | n_multiple | number of returns from pulses with several returns (penetrable canopy) | lidRmetrics 0.1.5 |
| 76 | p_first | share of first returns | lidRmetrics 0.1.5 |
| 77 | p_intermidiate | share of intermediate returns | lidRmetrics 0.1.5 |
| 78 | p_last | share of last returns | lidRmetrics 0.1.5 |
| 79 | p_single | share of single-return pulses' returns | lidRmetrics 0.1.5 |
| 80 | p_multiple | share of multiple-return pulses' returns | lidRmetrics 0.1.5 |
| 81 | ratio_last_first | last per first returns; higher where pulses pass through the canopy | lidRmetrics 0.1.5 |
| 82 | ratio_intermediate_first | intermediate per first returns; higher in deep, layered canopies | lidRmetrics 0.1.5 |
| 83 | ratio_multiple_single | multiple- per single-return returns; canopy penetrability | lidRmetrics 0.1.5 |
| 84 | n_return_1 | number of returns with return number 1 | lidRmetrics 0.1.5 |
| 85 | n_return_2 | number of returns with return number 2 | lidRmetrics 0.1.5 |
| 86 | n_return_3 | number of returns with return number 3 | lidRmetrics 0.1.5 |
| 87 | n_return_4 | number of returns with return number 4 | lidRmetrics 0.1.5 |
| 88 | rumple | roughness of the canopy surface (1 = flat; larger = more uneven) | lidRmetrics 0.1.5 |
| 89 | vn | number of occupied 1 m cubes (volume filled by vegetation) | lidRmetrics 0.1.5 |
| 90 | vFRall | share of the occupied cubes in the box around the vegetation | lidRmetrics 0.1.5 |
| 91 | vFRcanopy | share of space below the canopy top that is occupied | lidRmetrics 0.1.5 |
| 92 | vzrumple | complexity of the vertical profile of occupied cubes | lidRmetrics 0.1.5 |
| 93 | vzsd | unevenness of the vertical profile of occupied cubes | lidRmetrics 0.1.5 |
| 94 | vzcv | unevenness of the vertical profile of occupied cubes, relative | lidRmetrics 0.1.5 |
| 95 | OpenGapSpace | empty space above the canopy surface (between crowns) | lidRmetrics 0.1.5 |
| 96 | ClosedGapSpace | empty space inside or below the canopy | lidRmetrics 0.1.5 |
| 97 | Euphotic | occupied space in the upper, sunlit part of the canopy | lidRmetrics 0.1.5 |
| 98 | Oligophotic | occupied space in the lower, shaded part of the canopy | lidRmetrics 0.1.5 |
| 99 | kde_peaks_count | number of vegetation layers (peaks of the height distribution) | lidRmetrics 0.1.5 |
| 100 | kde_peak1_elev | height of vegetation layer 1, counted from the top | lidRmetrics 0.1.5 |
| 101 | kde_peak2_elev | height of vegetation layer 2, counted from the top | lidRmetrics 0.1.5 |
| 102 | kde_peak3_elev | height of vegetation layer 3, counted from the top | lidRmetrics 0.1.5 |
| 103 | kde_peak4_elev | height of vegetation layer 4, counted from the top | lidRmetrics 0.1.5 |
| 104 | kde_peak1_value | how strong vegetation layer 1 is | lidRmetrics 0.1.5 |
| 105 | kde_peak2_value | how strong vegetation layer 2 is | lidRmetrics 0.1.5 |
| 106 | kde_peak3_value | how strong vegetation layer 3 is | lidRmetrics 0.1.5 |
| 107 | kde_peak4_value | how strong vegetation layer 4 is | lidRmetrics 0.1.5 |
| 108 | kde_peak1_diff | vertical distance between layers 1 and 2 | lidRmetrics 0.1.5 |
| 109 | kde_peak2_diff | vertical distance between layers 2 and 3 | lidRmetrics 0.1.5 |
| 110 | kde_peak3_diff | vertical distance between layers 3 and 4 | lidRmetrics 0.1.5 |
| 111 | HOME | height that splits the returned energy in half; higher for top-heavy canopies | lidRmetrics 0.1.5 |
| 112 | enl | effective number of vegetation layers | forestr 2.0.2 |
| 113 | leafr_lai | leaf area index (leaf area per ground area) | leafR 0.3.5 |
| 114 | leafr_lahv | leaf area weighted by height (amount and height of foliage together) | leafR 0.3.5 |
| 115 | leafr_lahv_laiw | average height of the leaf area | leafR 0.3.5 |
| 116 | leafr_lahv_hw | LAHV relative to stand height | leafR 0.3.5 |
| 117 | leafr_lahv_laiw_hw | average height of the leaf area relative to stand height | leafR 0.3.5 |
| 118 | leafr_fhd | foliage height diversity: how evenly leaves spread over layers | leafR 0.3.5 |
| 119 | leafr_fhd_even | foliage height diversity scaled to 0–1 | leafR 0.3.5 |
| 120 | leafr_gs | diversity of the leaf profile (Gini–Simpson) | leafR 0.3.5 |
| 121 | leafr_gs_even | Gini–Simpson diversity scaled to 0–1 | leafR 0.3.5 |
| 122 | leafr_gc | inequality of return heights above 1 m (0 = all equal) | leafR 0.3.5 |
| 123 | pfs_pai | plant area index (leaves and wood) above 1 m | PyForestScan 0.3.6 |
| 124 | pfs_cover | canopy cover above 2 m (0–1) | PyForestScan 0.3.6 |
| 125 | pfs_fhd | foliage height diversity of the return profile | PyForestScan 0.3.6 |
| 126 | tree_n | number of trees (tree tops ≥ 5 m) | lidR 4.3.3 (tree tops, crowns); per-cell summary: lidarmetricpy |
| 127 | tree_h_mean | average tree height | lidR 4.3.3 (tree tops, crowns); per-cell summary: lidarmetricpy |
| 128 | tree_h_sd | variation of tree heights | lidR 4.3.3 (tree tops, crowns); per-cell summary: lidarmetricpy |
| 129 | tree_h_cv | variation of tree heights relative to their mean | lidR 4.3.3 (tree tops, crowns); per-cell summary: lidarmetricpy |
| 130 | tree_h_max | tallest tree | lidR 4.3.3 (tree tops, crowns); per-cell summary: lidarmetricpy |
| 131 | tree_h_gini | inequality of tree heights (0 = all equal) | lidR 4.3.3 (tree tops, crowns); per-cell summary: lidarmetricpy |
| 132 | tree_h_shannon | diversity of tree heights | lidR 4.3.3 (tree tops, crowns); per-cell summary: lidarmetricpy |
| 133 | crown_m2_mean | average crown size (m²) | lidR 4.3.3 (tree tops, crowns); per-cell summary: lidarmetricpy |
| 134 | crown_m2_sd | variation of crown size | lidR 4.3.3 (tree tops, crowns); per-cell summary: lidarmetricpy |
| 135 | gap_frac | share of the cell in canopy gaps (canopy ≤ 10 m) | ForestGapR 0.1.7 (gaps); per-cell summary: lidarmetricpy |
| 136 | gap_n | number of canopy gaps | ForestGapR 0.1.7 (gaps); per-cell summary: lidarmetricpy |
| 137 | gap_area_mean | average size of those gaps (m²) | ForestGapR 0.1.7 (gaps); per-cell summary: lidarmetricpy |
| 138 | clz_chm | canopy height (from the raw elevations) | canopyLazR 2.0 |
| 139 | clz_ground | ground elevation | canopyLazR 2.0 |
| 140 | clz_vertical_sd_lad | how unevenly leaves are spread with height in the column | canopyLazR 2.0 |
| 141 | clz_rugosity | within-canopy roughness: how much the vertical leaf structure changes between neighbours | canopyLazR 2.0 |
| 142 | clz_filled_ratio | share of canopy space filled with leaves | canopyLazR 2.0 |
| 143 | clz_porosity_ratio | share of canopy space that is empty (porosity) | canopyLazR 2.0 |
| 144 | clz_max_lad | leaf density of the densest layer | canopyLazR 2.0 |
| 145 | clz_max_lad_ht | height of the densest leaf layer | canopyLazR 2.0 |
| 146 | clz_toc_rugosity | roughness of the canopy top: how far the surface lies below its highest neighbour | canopyLazR 2.0 |

</details>
<!-- metrics table: end -->

### Exactness

Details reproduced from the R/C++ sources so that results match to floating-point rounding:
C `round()` (halves away from zero) in `round_any` and in lidR's raster layouts (also the per-cell 1 m layout of
`highest()` for rumple), R's `round(x, 3)`, `seq()` / `findInterval` bin edges, `hist()` boundary fuzz (LAD,
`metrics_interval`), a bit-exact port of `stats::density` (BinDist in point order, `dnorm`, R's FFT, `approx`)
for the peak count, qhull Delaunay (`QbB Qt`) for rumple, the lmf tie rule, Dalponte's in-pass updates, and
lidR's knn search order with libstdc++ `std::sort` for knnidw ties. docs/tables/table5_reproduction_details.md
lists them.

Known differences: when all heights above zmin in a cell are equal, R's Lmoments package returns rounding noise
for Lskew and Lkurt (NaN, ±Inf or any number, since L2 = 0); lidarmetricpy returns NA. Float64 backends differ
from each other by ~1e-14 in sums over many points. On float32 devices (Apple MPS) the lidRmetrics metrics are computed in float64 on the CPU by default (`Options(float32_exact=True)`), which makes them exact; `float32_exact=False` is faster but not exact (see below).

## Tests

```bash
pip install pytest
pytest            # ~20 s; GPU backends are skipped when their device cannot be used
```

- tests/test_reference.py: a synthetic 100 m × 100 m tile (tests/make_synthetic.py: tree crowns, multiple
  returns, a gap, an empty cell, points on cell edges) against the reference values of lidR, lidRmetrics,
  leafR, forestr, ForestGapR, PyForestScan and canopyLazR (tests/make_references.sh; versions in
  tests/data/ref/versions.txt).
- tests/test_oracle.py: 23 single-cell edge cases (few points, equal heights, invalid returns, heights 1e-12
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

Kanagawa Prefecture 2024 airborne lidar (52–133 pts/m², up to 7 returns, JGD2011 Plane IX; Kanagawa Prefecture,
CC BY, https://www.geospatial.jp/ckan/dataset/kanagawa-2024-pointcloud). The LAZ files are not redistributed here;
validation/make_testdata.py, make_fulltile.py and make_rawclips.py rebuild the test clips from the original tiles (`PCM_LAZ_DIR`, `PCM_DEM_DIR`). The reference outputs of the R and Python
packages are included (validation/ref, validation/ref_full).

Second data set: four whole tiles (400 m × 300 m) of the Shizuoka Prefecture airborne lidar (VIRTUAL SHIZUOKA,
CC BY 4.0; 2021 and 2025 releases, 77–293 pts/m², up to 5 returns, JGD2011 Plane VIII). validation/fetch_shizuoka.sh
downloads the point clouds and ground grids from the public bucket, validation/make_shizuoka.py normalizes them
with the 0.5 m ground grid, run_ref_shizuoka.sh makes the references (validation/ref_shizuoka), and
`compare.py --set shizuoka` / `compare_other.py --set shizuoka` compare.

Checked with the strict comparator (no missing or extra cell, same NA pattern, values within 1e-6 relative):

| data | cells | trees | result |
|---|---|---|---|
| 4 clips, 52–133 pts/m² | 400 | 4,275 | 120/120 columns (111 metrics_set3 + 9 tree summaries) equal in every cell (max relative diff 2.4e-12); all tree tops and crown areas equal — cupy, torch-cuda, torch-cpu, numpy (0.4.0) |
| 2 full tiles (90 M and 48 M points) | 2,400 | 23,559 | 120/120 columns, tree tops and crown areas equal (all four float64 backends; 0.4.0) |
| other packages, 4 clips + 2 full tiles | 400 + 2,400 | – | leafR / PyForestScan / forestr ENL / ForestGapR: 17/17 equal on all backends (max diff 3.4e-12) |
| Shizuoka, 4 whole tiles (84 M points) | 768 | 8,826 | 120/120 columns equal in every cell (max relative diff 8.2e-11); all tree tops and crown areas equal; 17/17 other-package metrics equal — all four float64 backends (0.4.0) |
| canopyLazR, 4 raw clips × 10 m and 5 m | 12,600 columns | – | all 9 rasters equal (max diff 3.1e-11) |
| float32 (MPS-like, tested as torch-cuda32), Kanagawa clips | 400 | 4,275 | with the default `float32_exact=True`: 120/120 columns and 17/17 other-package metrics equal; with `float32_exact=False` 94/111 equal, the others within 0.03 % except zpcum1–9 and HOME (up to 0.14 %) (0.4.0) |
| float32, Shizuoka | 768 | 8,826 | `float32_exact=True`: 120/120 equal; `False`: 85/111 equal, the others within 0.03 % except zpcum1–9 and HOME (up to 10 %, a point near a class boundary changes class) (0.4.0) |

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

**0.4.0 (all 111 metrics_set3 outputs + trees, idle machine, validation/time_work.py → timing/work_*.csv):**
CuPy 6.0 / 5.9 / 5.8 s for the 90.5 M point tile with GPU budgets of 16 / 32 / 96 GiB (3.9 s for 48.3 M), NumPy
99.2 s (58.6 s), against 174.9 s (140.8 s) for lidR's metrics_set3: CuPy is 30 times faster, NumPy 1.8 times.
12 CuPy workers: 1.59 s per tile. float32 (torch-cuda32) 84.9 s with the default `float32_exact=True`, 19.8 s
with `False`. The table above is from 0.3.2 (24 metrics). For the same 24 metrics, CuPy is ~30 times faster than lidR in either interval (work 165.7 / 5.5 s, whole
process 205.3 / 6.9 s), NumPy 3–4 times (165.7 / 53.4 s, 205.3 / 54.4 s).

| many tiles (the two full tiles in turn) | seconds per tile, 0.4.0 (111 metrics) | 0.3.2 (24 metrics) |
|---|---|---|
| CuPy, 1 / 4 / 12 workers | 4.96 / 1.89 / 1.59 | 4.59 / 1.72 / 1.44 |
| PyTorch CUDA, 12 workers | – | 1.53 |
| NumPy, 1 / 4 / 8 / 16 workers | 73.2 / 29.5 / 21.5 / 13.9 | 41.8 / 18.5 / 14.6 / 10.4 |
| lidR (metrics_set3, R 4.4.3), 8 R processes | 30.5 | 30.5 |

Memory budgets (0.4.0, 90.5 M point tile, docs/data/memory_budget.csv): with a 16 GiB GPU budget the peak GPU
memory pool is 11.6 GiB and the whole tile stays on the device (6.1 s); 5 GiB still runs on the device (7.7 s);
4 GiB and below bin the points in main memory and send them block by block (10.8 s). Values are identical for
every budget.

lidR on all cores was not measured. On the GPU path the GPU work is ~1 s per dense tile; the rest is CPU
work (LAZ decoding alone costs 23 CPU-seconds per dense tile and is the floor for throughput; the exact
`density()` port for kde_peaks_count runs on the CPU, 0.33 s measured on the dense tile).

## License

GPL-3.0-or-later (LICENSE): the tree detection, Dalponte segmentation, knnidw and spatial-index code are ports
of lidR (GPL-3, Jean-Romain Roussel). Metric definitions follow lidRmetrics (Tompalski et al.), leafR
(Almeida et al. 2019), PyForestScan (JOSS 2025), forestr (Atkins et al. 2018), ForestGapR
(Silva et al. 2019) and canopyLazR (Kamoske et al. 2019); see docs/paper_notes.md §9 for references.

## Citing

See CITATION.cff; each GitHub release is archived at Zenodo with its own DOI.

- version 1.0.0: https://doi.org/10.5281/zenodo.24111786
- all versions (resolves to the latest): https://doi.org/10.5281/zenodo.24111785

