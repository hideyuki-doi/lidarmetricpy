# Tests

```bash
pytest          # from the repository root
```

- `test_reference.py`: every installed float64 backend (numpy, torch-cpu, cupy, torch-cuda) against the
  reference values of lidR 4.3.3 + lidRmetrics 0.1.5 (33 metrics, tree tops, crown areas), leafR 0.3.5,
  forestr 2.0.2, ForestGapR 0.1.7, PyForestScan 0.3.6 and canopyLazR 2.0.
- `test_budgets.py`: memory budgets, multi-pass reading, the device and host paths on the GPU, the old option
  names, empty grids and `compute_tiles` must not change any value.

## Data

`data/synthetic.laz` (heights above ground) and `data/synthetic_raw.laz` (the same points on a sloping
terrain, for canopyLazR): 100 m × 100 m, 4 × 4 cells of 25 m, 337,858 points from ~200 k pulses with 1–4
returns. Tree crowns are paraboloids; one cell has few trees (gap), one has no point at all, and some points
lie exactly on cell edges, including the right and bottom outer edges. Made by `make_synthetic.py` (seed 1).

`data/ref/`: reference outputs made with `make_references.sh` (needs R with the packages above and Python with
pyforestscan; it calls the scripts in ../validation):

| file | from |
|---|---|
| synthetic_cells.csv, synthetic_trees.csv | validation/ref_lidr.R (lidR + lidRmetrics) |
| synthetic_other.csv | validation/ref_other.R (leafR, forestr, ForestGapR) |
| synthetic_pfs.csv | validation/ref_pyforestscan.py (PyForestScan) |
| synthetic_xy10_clz.csv, synthetic_xy5_clz.csv | validation/ref_canopylazr.R (canopyLazR, 10 m and 5 m) |
