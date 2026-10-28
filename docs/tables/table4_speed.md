# Table 4. Speed and memory (measured; RTX PRO 6000 Blackwell Max-Q 96 GB, Threadripper PRO 7955WX 16 cores)

| backend | one tile, 90.5 M points (s) | one tile, 48.3 M points (s) | peak GPU (GiB) | peak RAM (GiB) | throughput (s per tile, workers) |
|---|---|---|---|---|---|
| cupy | 5.46 | 3.74 | 27.81 | 2.31 | 4.61 (1); 1.75 (4); 1.43 (12) |
| torch-cuda | 5.54 | 3.66 | 24.7 | 2.87 | 1.53 (12) |
| torch-cuda32 | 20.51 | 12.13 | 18.26 | 11.19 | 5.31 (12) |
| numpy | 53.47 | 31.22 | – | 24.37 | 41.81 (1); 18.09 (4); 14.57 (8); 10.49 (16) |
| torch-cpu | 47.55 | 27.49 | – | 25.5 | 14.34 (16) |
| lidR + lidRmetrics, 24 metrics (R) | 205.16 | 155.83 | – | 9.6 | – |
| lidR + lidRmetrics, metrics_set3 (R) | 230.87 | 179.44 | – | 9.6 | 30.52 (8) |
