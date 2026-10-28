# Changelog

## 0.3.0 (2026-10-28)

Fixes from an independent review of 0.2.0 (commit bfe8725). Every item below was reproduced first and has a
regression test (tests/test_regressions.py, tests/test_oracle.py).

Results that changed
- **DTM**: the DTM window is read on the DTM's own pixel grid. 0.2.0 read a window with a fractional offset
  and could subtract the value of a neighbouring DTM pixel when the grid origin was not aligned with the DTM.
  Points where the DTM has no data (nodata, mask, outside the raster; also integer rasters without a nodata
  value) are dropped; 0.2.0 could read them as a ground height of 0.
- **rumple**: the highest point per 1 m pixel is chosen within each cell. In 0.2.0 neighbouring cells
  competed for the same pixel when the grid was not aligned with whole metres.
- **Cells kept**: every cell with at least one point after the preprocessing has a row (0.2.0 dropped cells
  without points above zmin, e.g. bare ground). New column `n_points`; `n` is NA without points above zmin,
  as in lidRmetrics. The columns depend only on the options.
- **Height range**: no fixed number of 1 m layers any more (0.2.0 assumed heights < 61–64 m; with a wider
  `zrange` a tall cell could change its neighbour's metrics or raise an error).
- **LAD**: bins follow R's `hist()` with its boundary fuzz (a height 1e-12 above a break changed lad_* in 0.2.0).
- **kde_peaks_count**: bit-exact port of R's `density()` (BinDist in point order, `dnorm`, R's FFT, `approx`),
  so ties at a flat density top are counted as in R, on every backend.
- **lidRmetrics edge rules**: invalid return numbers are left out of the echo metrics; equal heights give
  zpcum = 0; LAD needs more than two points and L-moments four; `zentropy` is NA when every point lies on an
  integer zmax; `pz_above_30` uses the shifted break of `hist(right = FALSE)` and is 0 without points.
- **Convex hull** for the knnidw fill of the canopy raster is exact (0.2.0 kept only row/column extremes, which
  can miss hull vertices). One, two or collinear points no longer raise an error.
- **Spike filter**: 1 m columns are split at the rows of cells on both processing paths (they differed for
  cell sizes that are not whole metres).
- **Sums over height layers** are computed in a fixed order, so a cell's values do not depend on the block it
  is computed in.

API
- `Options.lidr()`: the settings under which results equal lidR / lidRmetrics (no height range, no spike filter).
- Memory budgets below 8 MiB raise `ValueError`; LAZ read chunks follow `cpu_memory_gb`; crowded rows are split
  by the points of each cell, and a single cell larger than the budget is processed on its own with a warning.
- `lidarmetricpy.validate.compare_tables`: strict comparison of cell tables (outer join, unique keys, NA / ±Inf
  classes, relative tolerance).

Validation
- All comparisons use `compare_tables`: 0.2.0's scripts used an inner join (missing cells were not counted) and
  accepted a reference +Inf against any value.
- Reference scripts stop at the first error and replace committed files only after their row counts are checked.
- GPU tests run only when the device can be used.
- The R timing now also covers the lidRmetrics functions that the 24 metrics need (validation/ref_lidr_subset.R).

Known differences from lidR (documented, tested)
- Cells whose heights above zmin are all equal: R's Lmoments returns rounding noise for Lskew and Lkurt
  (NaN, ±Inf or any number); lidarmetricpy returns NA.

## 0.2.0 (2026-10-28)

First public version (as pcmetrics, renamed lidarmetricpy).
