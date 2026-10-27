# GPU / CPU comparison (measured)

Machine: RTX PRO 6000 Blackwell Max-Q 96 GB, AMD Threadripper PRO 7955WX (16 cores / 32 threads), 251 GB RAM. lidRmetrics-equivalent metrics + trees
(other_packages = False). Every number below is a measurement; nothing is extrapolated.

## One full tile, one process (after warm-up)

| backend | tile | seconds | peak GPU GB | peak RAM GB | metrics equal to lidR | trees / crowns equal |
|---|---|---|---|---|---|---|
| cupy | 09LC9423 | 5.3 | 29.2 | 2.3 | 33/33 | 13109/13109, 13109/13109 |
| cupy | 09MC0523 | 3.6 | 18.3 | 2.2 | 33/33 | 10450/10450, 10450/10450 |
| torch-cuda | 09LC9423 | 5.1 | 27.3 | 2.9 | 33/33 | 13109/13109, 13109/13109 |
| torch-cuda | 09MC0523 | 3.5 | 15.8 | 2.9 | 33/33 | 10450/10450, 10450/10450 |
| torch-cuda32 | 09LC9423 | 18.1 | 17.8 | 10.4 | 28/33 | 13109/13109, 13109/13109 |
| torch-cuda32 | 09MC0523 | 11.2 | 10.9 | 6.3 | 29/33 | 10450/10450, 10450/10450 |
| numpy | 09LC9423 | 50.6 | 0.0 | 25.7 | 33/33 | 13109/13109, 13109/13109 |
| numpy | 09MC0523 | 30.0 | 0.0 | 15.4 | 33/33 | 10450/10450, 10450/10450 |
| torch-cpu | 09LC9423 | 47.4 | 0.0 | 26.7 | 33/33 | 13109/13109, 13109/13109 |
| torch-cpu | 09MC0523 | 28.6 | 0.0 | 16.3 | 33/33 | 10450/10450, 10450/10450 |
| lidR (R, 1 core) | 09LC9423 | 230.9 | – | 9.6 | reference | reference |
| lidR (R, 1 core) | 09MC0523 | 179.4 | – | 5.3 | reference | reference |

With the other-package metrics (leafR, PyForestScan, ForestGapR; other_packages = True):

| backend | tile | seconds | seconds without |
|---|---|---|---|
| cupy | 09LC9423 | 5.7 | 5.3 |
| numpy | 09LC9423 | 67.4 | 50.6 |

## Many tiles (throughput)

Jobs alternate the two tiles. s/tile = wall time / tiles, including worker start-up.

| backend | workers | tiles | seconds | s per tile |
|---|---|---|---|---|
| cupy | 1 | 12 | 53.0 | 4.41 |
| cupy | 4 | 24 | 41.5 | 1.73 |
| cupy | 12 | 24 | 32.7 | 1.36 |
| numpy | 1 | 4 | 159.0 | 39.75 |
| numpy | 4 | 8 | 130.6 | 16.32 |
| numpy | 8 | 16 | 212.6 | 13.28 |
| numpy | 16 | 16 | 151.9 | 9.49 |
| torch-cpu | 16 | 16 | 195.7 | 12.23 |
| torch-cuda | 12 | 24 | 34.5 | 1.44 |
| torch-cuda32 | 12 | 24 | 111.0 | 4.63 |
| lidR (8 R processes) | 8 | 8 | 244.2 | 30.52 |

lidR timings were measured on this machine with R 4.4.3 earlier the same day (ref_full/*.log, par/lidr_par.log); its outputs are identical under R 4.5.3.
