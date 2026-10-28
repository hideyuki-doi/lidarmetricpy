"""GPU / CPU backend comparison for the paper (measured values only).

    python backend_compare.py single <backend> <mesh>        # one full tile in a fresh process
    python backend_compare.py throughput <backend> <workers> <n_pairs> [host_gb_per_worker]

single: warm-up on a clip (JIT compile, CUDA context), then one full tile; records wall time,
stage times, peak device memory, peak RSS and agreement with the lidR reference (ref_full/).
throughput: compute_tiles over n_pairs × (09LC9423, 09MC0523) full tiles.
Rows are appended to validation/backend_compare/{single,throughput}.csv.
"""
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# LIDARMETRICPY_PATH: a frozen copy of the package, so that edits during a long benchmark cannot leak in
sys.path.insert(0, os.environ.get("LIDARMETRICPY_PATH", str(HERE.parent)))
from lidarmetricpy import Options, compute_tile, compute_tiles  # noqa: E402
from compare import METRICS  # noqa: E402
from refcheck import check  # noqa: E402

OUT = HERE / "backend_compare"
OUT.mkdir(exist_ok=True)
FULL = {d["mesh"]: d for d in json.loads((HERE / "data/full.json").read_text())}
CLIPS = {d["mesh"]: d for d in json.loads((HERE / "data/clips.json").read_text()) if d["n"] > 0}
OPT = Options(spike_filter=False, other_packages=False, zrange=(-np.inf, np.inf))      # lidRmetrics set (comparable with lidR)
OPT_ALL = Options(spike_filter=False, other_packages=True, zrange=(-np.inf, np.inf))   # + leafR / PyForestScan / ForestGapR


def _append(name, row):
    p = OUT / f"{name}.csv"
    df = pd.DataFrame([row])
    df.to_csv(p, mode="a", header=not p.exists(), index=False)
    print(row, flush=True)


class Peak:
    """Polls device memory held by this process (CuPy pool / PyTorch caching allocator)."""

    def __init__(self, backend):
        self.backend, self.peak, self.stop = backend, 0, False

    def run(self):
        try:
            if self.backend == "cupy":
                import cupy
                pool = cupy.get_default_memory_pool()
                f = pool.total_bytes
            elif self.backend.startswith("torch-cuda"):
                import torch
                f = torch.cuda.memory_reserved
            else:
                return
            while not self.stop:
                self.peak = max(self.peak, f())
                time.sleep(0.01)
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self):
        self.t = threading.Thread(target=self.run, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop = True
        self.t.join()


def agreement(df, mesh):
    """(number of metrics equal in every cell, {metric: share of compared cells that are equal}) — strict."""
    res = check(pd.read_csv(HERE / "ref_full" / f"{mesh}_cells.csv"), df, METRICS)
    worst = {r.metric: round(1 - (r.value_mismatch + r.class_mismatch) / max(1, r.n_finite + r.class_mismatch), 6)
             for r in res.itertuples() if not r.equal}
    return int(res.equal.sum()), worst


def single(backend, mesh, other=False):
    opt = OPT_ALL if other else OPT
    clip = CLIPS["09MC0523"]
    compute_tile(str(HERE / clip["file"]), clip["x0"], clip["y1"], clip["ncol"], clip["nrow"], backend=backend, options=opt)
    c = FULL[mesh]
    with Peak(backend) as pk:
        t = time.time()
        df, (chm, L, (rr, cc, area)) = compute_tile(str(HERE / c["file"]), c["x0"], c["y1"], 40, 30, backend=backend,
                                                    options=opt, return_trees=True)
        sec = time.time() - t
    exact, worst = agreement(df, mesh)
    rt = pd.read_csv(HERE / "ref_full" / f"{mesh}_trees.csv")
    pt = pd.DataFrame({"x": L.xmin + (cc + 0.5) * L.res, "y": L.ymax - (rr + 0.5) * L.res, "a": area})
    j = rt.merge(pt, on=["x", "y"], how="outer", indicator=True)
    st = df.attrs["stats"]
    _append("single", {
        "backend": backend, "mesh": mesh, "other_packages": other, "points": c["n"], "seconds": round(sec, 2),
        "path": st.get("path"),
        **{f"t_{k}": v for k, v in st["stage_seconds"].items()},
        "peak_device_gb": round(pk.peak / 1024 ** 3, 2),
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2),
        "metrics_exact": exact, "metrics_total": len(METRICS), "not_exact": json.dumps(worst),
        "trees_ref": len(rt), "trees_matched": int((j._merge == "both").sum()),
        "crowns_equal": int((np.abs(j.crown_m2 - j.a) < 1e-9).sum())})


def throughput(backend, workers, n_pairs, host_gb=None):
    jobs = [dict(name=f"{m}_{i}", source=str(HERE / FULL[m]["file"]), x0=FULL[m]["x0"], y1=FULL[m]["y1"], ncol=40, nrow=30)
            for i in range(n_pairs) for m in ("09LC9423", "09MC0523")]
    kw = {}
    total = 90 if backend in ("cupy", "torch-cuda", "torch-cuda32") else 200
    if host_gb is not None:
        kw["cpu_memory_gb"] = host_gb * workers                # compute_tiles takes the total RAM budget
    if backend in ("cupy", "torch-cuda", "torch-cuda32"):
        kw["gpu_memory_gb"] = total
    load0 = os.getloadavg()[0]
    t = time.time()
    n = sum(1 for _ in compute_tiles(jobs, workers=workers, backend=backend, options=OPT, **kw))
    sec = time.time() - t
    _append("throughput", {"backend": backend, "workers": workers, "tiles": n,
                           "points": sum(FULL[j["name"][:8]]["n"] for j in jobs), "seconds": round(sec, 1),
                           "s_per_tile": round(sec / n, 2), "cpu_count": os.cpu_count(),
                           "host_gb_per_worker": host_gb, "load_before": round(load0, 1)})


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "single":
        single(sys.argv[2], sys.argv[3], len(sys.argv) > 4 and sys.argv[4] == "other")
    else:
        throughput(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
                   float(sys.argv[5]) if len(sys.argv) > 5 else None)
