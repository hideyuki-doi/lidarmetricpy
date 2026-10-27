# GPU / CPU comparison (measured)

Machine: RTX PRO 6000 Blackwell Max-Q 96 GB, AMD Threadripper PRO 7955WX (16 cores / 32 threads), 251 GB RAM. lidRmetrics-equivalent metrics + trees
(other_packages = False). Every number below is a measurement; nothing is extrapolated.

## One full tile, one process (after warm-up)

| backend | tile | seconds | peak GPU GB | peak RAM GB | metrics equal to lidR | trees / crowns equal |
|---|---|---|---|---|---|---|
| cupy | 09LC9423 | 5.2 | 25.4 | 2.3 | 33/33 | 13109/13109, 13109/13109 |
| cupy | 09MC0523 | 3.6 | 16.3 | 2.3 | 33/33 | 10450/10450, 10450/10450 |
| torch-cuda | 09LC9423 | 5.1 | 24.3 | 2.9 | 33/33 | 13109/13109, 13109/13109 |
| torch-cuda | 09MC0523 | 3.4 | 14.9 | 2.9 | 33/33 | 10450/10450, 10450/10450 |
| torch-cuda32 | 09LC9423 | 18.2 | 17.8 | 10.4 | 27/33 | 13109/13109, 13109/13109 |
| torch-cuda32 | 09MC0523 | 10.9 | 10.9 | 6.4 | 29/33 | 10450/10450, 10450/10450 |
| numpy | 09LC9423 | 50.6 | 0.0 | 25.7 | 33/33 | 13109/13109, 13109/13109 |
| numpy | 09MC0523 | 30.1 | 0.0 | 15.5 | 33/33 | 10450/10450, 10450/10450 |
| torch-cpu | 09LC9423 | 45.5 | 0.0 | 26.7 | 33/33 | 13109/13109, 13109/13109 |
| torch-cpu | 09MC0523 | 27.6 | 0.0 | 16.3 | 33/33 | 10450/10450, 10450/10450 |
| lidR (R, 1 core) | 09LC9423 | 230.9 | – | 9.6 | reference | reference |
| lidR (R, 1 core) | 09MC0523 | 179.4 | – | 5.3 | reference | reference |

With the other-package metrics (leafR, PyForestScan, ForestGapR; other_packages = True):

| backend | tile | seconds | seconds without |
|---|---|---|---|
| cupy | 09LC9423 | 5.7 | 5.2 |
| numpy | 09LC9423 | 67.3 | 50.6 |

## Many tiles (throughput)

Jobs alternate the two tiles. s/tile = wall time / tiles, including worker start-up.

| backend | workers | tiles | seconds | s per tile |
|---|---|---|---|---|
| cupy | 1 | 12 | 52.8 | 4.40 |
| cupy | 4 | 24 | 41.1 | 1.71 |
| cupy | 12 | 24 | 32.0 | 1.33 |
| numpy | 1 | 4 | 159.2 | 39.80 |
| numpy | 4 | 8 | 131.8 | 16.47 |
| numpy | 8 | 16 | 214.6 | 13.41 |
| numpy | 16 | 16 | 154.1 | 9.63 |
| torch-cpu | 16 | 16 | 198.7 | 12.42 |
| torch-cuda | 12 | 24 | 35.4 | 1.48 |
| torch-cuda32 | 12 | 24 | 109.4 | 4.56 |
| lidR (8 R processes) | 8 | 8 | 244.2 | 30.52 |

lidR timings were measured on this machine with R 4.4.3 earlier the same day (ref_full/*.log, par/lidr_par.log); its outputs are identical under R 4.5.3.
