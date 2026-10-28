# GPU / CPU comparison (measured)

Machine: RTX PRO 6000 Blackwell Max-Q 96 GB, AMD Threadripper PRO 7955WX (16 cores / 32 threads), 251 GB RAM. lidRmetrics-equivalent metrics + trees
(other_packages = False). Every number below is a measurement; nothing is extrapolated.

## One full tile, one process (after warm-up)

| backend | tile | seconds | peak GPU GB | peak RAM GB | metrics equal to lidR | trees / crowns equal |
|---|---|---|---|---|---|---|
| cupy | 09LC9423 | 5.5 | 22.1 | 2.2 | 33/33 | 13109/13109, 13109/13109 |
| cupy | 09MC0523 | 3.8 | 13.3 | 2.2 | 33/33 | 10450/10450, 10450/10450 |
| torch-cuda | 09LC9423 | 5.5 | 21.6 | 2.9 | 33/33 | 13109/13109, 13109/13109 |
| torch-cuda | 09MC0523 | 3.8 | 12.6 | 2.9 | 33/33 | 10450/10450, 10450/10450 |
| torch-cuda32 | 09LC9423 | 20.4 | 15.0 | 11.2 | 28/33 | 13109/13109, 13109/13109 |
| torch-cuda32 | 09MC0523 | 12.4 | 8.9 | 6.8 | 29/33 | 10450/10450, 10450/10450 |
| numpy | 09LC9423 | 53.4 | 0.0 | 23.0 | 33/33 | 13109/13109, 13109/13109 |
| numpy | 09MC0523 | 31.2 | 0.0 | 13.8 | 33/33 | 10450/10450, 10450/10450 |
| torch-cpu | 09LC9423 | 48.6 | 0.0 | 23.6 | 33/33 | 13109/13109, 13109/13109 |
| torch-cpu | 09MC0523 | 29.0 | 0.0 | 14.2 | 33/33 | 10450/10450, 10450/10450 |
| lidR (R, 1 core) | 09LC9423 | 230.9 | – | 9.6 | reference | reference |
| lidR (R, 1 core) | 09MC0523 | 179.4 | – | 5.3 | reference | reference |

With the other-package metrics (leafR, PyForestScan, ForestGapR; other_packages = True):

| backend | tile | seconds | seconds without |
|---|---|---|---|
| cupy | 09LC9423 | 6.0 | 5.5 |
| numpy | 09LC9423 | 70.3 | 53.4 |

## Many tiles (throughput)

Jobs alternate the two tiles. s/tile = wall time / tiles, including worker start-up.

| backend | workers | tiles | seconds | s per tile |
|---|---|---|---|---|
| cupy | 1 | 12 | 55.0 | 4.59 |
| cupy | 4 | 24 | 41.2 | 1.72 |
| cupy | 12 | 24 | 34.6 | 1.44 |
| numpy | 1 | 4 | 167.2 | 41.80 |
| numpy | 4 | 8 | 147.9 | 18.49 |
| numpy | 8 | 16 | 234.0 | 14.63 |
| numpy | 16 | 16 | 166.1 | 10.38 |
| torch-cpu | 16 | 16 | 228.6 | 14.29 |
| torch-cuda | 12 | 24 | 36.8 | 1.53 |
| torch-cuda32 | 12 | 24 | 128.0 | 5.33 |
| lidR (8 R processes) | 8 | 8 | 244.2 | 30.52 |

lidR timings were measured on this machine with R 4.4.3 earlier the same day (ref_full/*.log, par/lidr_par.log); its outputs are identical under R 4.5.3.
