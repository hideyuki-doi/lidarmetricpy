# Table 4. Speed and memory (measured; RTX PRO 6000 Blackwell Max-Q 96 GB, Threadripper PRO 7955WX 16 cores). work = reading + metrics + trees inside one process (Python: after a warm-up call; R: read_s + area_s + tree_s); whole process = start, imports / package loading, work and saving the cell table, tree tops and canopy raster (/usr/bin/time, fresh process). lidR throughput: metrics_set3, 8 processes.

| backend | work, 90.5 M (s) | work, 48.3 M (s) | whole process, 90.5 M (s) | whole process, 48.3 M (s) | peak GPU (GiB) | peak RAM (GiB) | throughput (s per tile, workers) |
|---|---|---|---|---|---|---|---|
| cupy | 5.51 | 3.77 | 6.86 | 4.87 | 22.05 | 2.2 | 4.59 (1); 1.72 (4); 1.44 (12) |
| torch-cuda | 5.51 | 3.75 | – | – | 21.58 | 2.87 | 1.53 (12) |
| torch-cuda32 | 20.41 | 12.39 | – | – | 15.04 | 11.21 | 5.33 (12) |
| numpy | 53.43 | 31.17 | 54.38 | 32.38 | – | 22.99 | 41.80 (1); 18.49 (4); 14.63 (8); 10.38 (16) |
| torch-cpu | 48.6 | 28.97 | – | – | – | 23.57 | 14.29 (16) |
| lidR + lidRmetrics, 24 metrics (R) | 165.7 | 133.9 | 205.33 | 156.22 | – | 9.6 | – |
| lidR + lidRmetrics, metrics_set3 (R) | 174.9 | 140.8 | 214.61 | 163.29 | – | 9.6 | 30.52 (8) |
