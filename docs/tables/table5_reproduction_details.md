# Table 5. Details needed for exact agreement

| detail of the reference implementation | affects | how lidarmetricpy matches it |
|---|---|---|
| lidR round_any uses C round() (halves away from zero) | voxel indices, canopy raster extent | C round emulated |
| terra closes the right / bottom outer edge of a raster: points on it belong to the last column / row | cell membership | closed_edges=True (default) |
| highest(1) snaps each cell's bbox with st_adjust_bbox (C round): a negative-integer max x (positive-integer min y) lies on the layout edge and goes to the last column (row) | rumple | per-cell layout edge rule |
| R round(x, 3) (fround.c) differs from numpy.round in ~6 % of values | canopy raster heights | fround port |
| seq() clips the last break with pmin(x, to) | zpcum | breaks clipped |
| findInterval drops a point exactly at an integer zmax | zentropy | same rule |
| lidRmetrics rounds z before z > 0.5 (z = 0.5 becomes a 1 m voxel) | voxel metrics | voxels from all points |
| Lefsky class with no voxel is NA (left join), not 0 | Lefsky classes | NA rule |
| stats::density: BinDist in point order, seq.int, dnorm4, R's own FFT (Singleton), approx1; get_peaks counts both ends of a flat top | kde_peaks_count | bit-exact port (density y identical to R) |
| lidR::LAD uses hist(breaks = c(-Inf, seq(z0, top))) with 1e-7 x median(diff) fuzz (data range when < 3 breaks) | lad_max, lad_mean, lad_cv | same fuzzy breaks |
| metrics_interval uses hist(right = FALSE): the 30 m break is shifted to 30 - 7.5e-7; empty z gives 0 | pz_above_30 | same threshold |
| metrics_echo drops ReturnNumber <= 0, NumberOfReturns <= 0, ReturnNumber > NumberOfReturns | p_single, p_last, ratio_last_first | same filter and denominator |
| metrics_lad needs > 2 points, metrics_Lmoments 4 L-moments (n >= 4), canopydensity gives 0 for equal heights, entropy NA when every point sits on an integer zmax | small cells | same NA / 0 rules |
| lmf: an equal height already tagged as maximum wins (raster order) | tree tops | sequential tie pass |
| dalponte2016 updates crown means within a pass; later regions overwrite | crown area | numba port |
| knnidw ties: GridPartition visit order + libstdc++ introsort | filled canopy pixels | ported, checked vs g++ |
| CuPy scatter_max / ufunc.at use float32 atomics for float64 | all max / min | int64 order-preserving atomics |
| PyTorch CUDA divides by a Python scalar via its reciprocal | bin edges | tensor divisor |
| fused multiply-add in X·scale + offset | coordinates | separate kernels |
| leafR: one NA layer makes LAHV / FHD NA; < 2 layers -> error | leafR metrics | same NA rules |
| PyForestScan: zmax on an integer edge falls in the last closed bin | pfs metrics | same binning |
| canopyLazR: file-wide Tukey filter, round() half-to-even bins, factor NA drops | canopyLazR rasters | ported |
