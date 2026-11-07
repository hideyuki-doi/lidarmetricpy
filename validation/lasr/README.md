lasR 0.21.3 (conda-forge, env `lasr`) on 09LC9423_full.laz (90.5 M points), `lasr_compare.R`:
- work time 47.8 s (1 core) / 42.4 s (16 cores) for 4 metrics + canopy raster + local maxima + region growing
- n and zq20 identical to lidR/lidRmetrics; zcv identical after ×100 (lasR returns a fraction, lidRmetrics
  percent; max relative difference 1.5e-7, float32 GeoTIFF); pz_above_30 ×100 equal within 1e-6 in 98.9 % of
  cells, largest difference 0.0025 percentage points (not investigated further)
- trees: 32,793 tops vs 13,109 in lidR — lasR has no p2r subcircle/knnidw fill and no variable lmf window, so
  lidR's tree workflow cannot be reproduced
