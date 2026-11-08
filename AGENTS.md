# AGENTS.md — guide for AI coding assistants

This file is for AI coding agents (Claude Code, Codex, Cursor, Copilot and others) that use or change
lidarmetricpy for a person. `AGENTS.md` is the file name these tools look for; humans may use it as a checklist.
AI agents can misread data or documentation: the person who runs the analysis must check the results
(see "Check before you trust a run").

## What the package does

`compute_tile` turns one airborne lidar tile (LAS/LAZ, or arrays) into a table with one row per grid cell
(usually 25 m) and a fixed set of forest-structure metrics. With `Options.lidr()` the values equal those of
lidR 4.3.3 + lidRmetrics 0.1.5 (all 111 outputs of `metrics_set3`, plus 9 tree summaries) and of leafR,
PyForestScan, forestr and ForestGapR, to floating-point rounding, on every float64 backend. `compute_tiles`
runs many tiles in parallel worker processes. README.md documents every option; this file lists what agents
tend to get wrong.

## Minimal correct call

```python
from lidarmetricpy import compute_tile, Options
df = compute_tile("tile.laz", x0=-62000, y1=-57750, ncol=40, nrow=30, cell=25,
                  dtm="dtm.tif",                 # ground model (DEM/DTM), NOT a surface model (DSM)
                  backend="auto", gpu_memory_gb=8, cpu_memory_gb=16,
                  options=Options.lidr())        # lidR-identical; the defaults also filter points
```

- `x0`, `y1` are the left and top edges of the grid in the coordinates of the file; `ncol × nrow` cells of
  `cell` m. Take them from the tile's sheet / extent, not from the median of the points.
- `dtm=None` means Z is already height above ground.
- Results contain only cells with at least one point; columns are fixed (`lidarmetricpy.api.output_columns`).

## Rules

1. **Ground model.** Give a terrain model. Open-data portals often publish DEM, DSM and DCHM grids side by
   side, sometimes with identical file names in different folders (e.g. Yamanashi: `2024/01/LP/Grid` is DSM1,
   `2024/03/LP/Grid` is the DEM). A DSM as `dtm` gives heights near zero everywhere.
2. **Options.** `Options.lidr()` reproduces lidR. The defaults (`zrange=(-1, 60)`, `spike_filter=True`) remove
   points that lidR keeps; say which one you used. `Options.lidr(zrange=(-1, 60))` equals lidR after
   `filter_poi(Z > -1 & Z < 60)`.
3. **Coordinates.** All inputs (LAZ, DTM, grid) must be in the same projected CRS, in metres. The package does
   not reproject.
4. **Memory.** Set `gpu_memory_gb` and `cpu_memory_gb` below what is free on a shared machine; results do not
   depend on them. With `compute_tiles` the budgets are totals for all workers. Scripts that use
   `compute_tiles` need an `if __name__ == "__main__":` guard (spawned workers).
5. **Precision.** Use a float64 backend (cupy, torch-cuda, numpy, torch-cpu) for lidR-identical values.
   Apple MPS runs in float32 and is not exact.
6. **Points outside the DTM are dropped** (lidR uses the nearest pixel). Make the DTM cover the tile.
7. **Do not "fix" the definitions.** Odd-looking behaviour copied from lidRmetrics is intended (e.g.
   `n_return_k` is the count of the k-th return number present; `p_intermidiate` is spelled as in lidRmetrics;
   Lskew/Lkurt are NA when all heights are equal). See README "Known differences".

## Check before you trust a run

- Open a few output cells and check that heights are plausible (canopy heights of tens of metres in forest,
  `zmax` not near 0 everywhere, not hundreds of metres).
- Compare ground-classified points (class 2, where available) with the DTM: their heights should be near 0.
- `n_points` per cell should match the point density you expect; cells missing at tile edges usually mean a
  grid or CRS mismatch.
- Metrics that depend on return counts (LAI, LAD, gap fractions) change with point density and sensor; check
  for survey boundaries in maps before comparing surveys.
- For lidR-identical claims, run `lidarmetricpy.validate.compare_tables` against a lidR reference on a sample
  (validation/ref_lidr.R makes one).

## Changing the package

- Every metric is tested against the R packages: `pytest` (tests/, ~4 min with a GPU; GPU tests skip without
  one) and validation/compare.py on real data. A change must keep all of them equal.
- tests/make_oracle.R and validation/ref_lidr.R regenerate references (R with lidR 4.3.3, lidRmetrics 0.1.5).
- Keep results independent of memory budgets and block splits (tests/test_budgets.py, test_regressions.py).
- docs/table5 and README list the R details that exact agreement depends on (rounding, `hist()` fuzz,
  `density()`, tie orders); read them before touching a metric.
