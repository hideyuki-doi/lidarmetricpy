"""Throughput: the same 8 full-tile jobs as run_par_lidr.sh, with compute_tiles and N workers.

    python throughput.py <backend> <workers> [<total_gpu_gb>]
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from pcmetrics import Options, compute_tiles  # noqa: E402

if __name__ == "__main__":
    backend, workers = sys.argv[1], int(sys.argv[2])
    total = float(sys.argv[3]) if len(sys.argv) > 3 else None
    full = {d["mesh"]: d for d in json.loads((HERE / "data/full.json").read_text())}
    jobs = []
    for i in range(int(sys.argv[4]) if len(sys.argv) > 4 else 4):
        for m in ("09LC9423", "09MC0523"):
            c = full[m]
            jobs.append(dict(name=f"{m}_{i}", source=str(HERE / c["file"]), x0=c["x0"], y1=c["y1"], ncol=40, nrow=30))
    t = time.time()
    n = 0
    for name, df in compute_tiles(jobs, workers=workers, backend=backend, gpu_memory_gb=total,
                                  options=Options(spike_filter=False)):
        n += 1
    sec = time.time() - t
    print(f"{backend} workers={workers} total_gb={total}: {n} jobs in {sec:.1f} s ({sec / n:.1f} s per tile)")
