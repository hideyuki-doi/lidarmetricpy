# GPU / CPU comparison (measured)

Machine: RTX PRO 6000 Blackwell Max-Q 96 GB, AMD Threadripper PRO 7955WX (16 cores / 32 threads), 251 GB RAM. lidRmetrics-equivalent metrics + trees
(other_packages = False). Every number below is a measurement; nothing is extrapolated.

## One full tile, one process (after warm-up)

| backend | tile | seconds | peak GPU GB | peak RAM GB | metrics equal to lidR | trees / crowns equal |
|---|---|---|---|---|---|---|
| cupy | 09LC9423 | 5.5 | 27.8 | 2.3 | 33/33 | 13109/13109, 13109/13109 |
| cupy | 09MC0523 | 3.7 | 17.8 | 2.2 | 33/33 | 10450/10450, 10450/10450 |
| torch-cuda | 09LC9423 | 5.5 | 24.7 | 2.9 | 33/33 | 13109/13109, 13109/13109 |
| torch-cuda | 09MC0523 | 3.7 | 14.5 | 2.9 | 33/33 | 10450/10450, 10450/10450 |
| torch-cuda32 | 09LC9423 | 20.5 | 18.3 | 11.2 | 28/33 | 13109/13109, 13109/13109 |
| torch-cuda32 | 09MC0523 | 12.1 | 10.8 | 6.8 | 29/33 | 10450/10450, 10450/10450 |
| numpy | 09LC9423 | 53.5 | 0.0 | 24.4 | 33/33 | 13109/13109, 13109/13109 |
| numpy | 09MC0523 | 31.2 | 0.0 | 14.3 | 33/33 | 10450/10450, 10450/10450 |
| torch-cpu | 09LC9423 | 47.5 | 0.0 | 25.5 | 33/33 | 13109/13109, 13109/13109 |
| torch-cpu | 09MC0523 | 27.5 | 0.0 | 15.4 | 33/33 | 10450/10450, 10450/10450 |
| lidR (R, 1 core) | 09LC9423 | 230.9 | – | 9.6 | reference | reference |
| lidR (R, 1 core) | 09MC0523 | 179.4 | – | 5.3 | reference | reference |

With the other-package metrics (leafR, PyForestScan, ForestGapR; other_packages = True):

| backend | tile | seconds | seconds without |
|---|---|---|---|
| cupy | 09LC9423 | 6.0 | 5.5 |
| numpy | 09LC9423 | 70.2 | 53.5 |

## Many tiles (throughput)

Jobs alternate the two tiles. s/tile = wall time / tiles, including worker start-up.

| backend | workers | tiles | seconds | s per tile |
|---|---|---|---|---|
| cupy | 1 | 12 | 55.3 | 4.61 |
| cupy | 4 | 24 | 42.0 | 1.75 |
| cupy | 12 | 24 | 34.4 | 1.43 |
| numpy | 1 | 4 | 167.2 | 41.81 |
| numpy | 4 | 8 | 144.8 | 18.09 |
| numpy | 8 | 16 | 233.1 | 14.57 |
| numpy | 16 | 16 | 167.8 | 10.49 |
| torch-cpu | 16 | 16 | 229.4 | 14.34 |
| torch-cuda | 12 | 24 | 36.8 | 1.53 |
| torch-cuda32 | 12 | 24 | 127.5 | 5.31 |
| lidR (8 R processes) | 8 | 8 | 244.2 | 30.52 |

lidR timings were measured on this machine with R 4.4.3 earlier the same day (ref_full/*.log, par/lidr_par.log); its outputs are identical under R 4.5.3.
