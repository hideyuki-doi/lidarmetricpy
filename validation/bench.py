"""Time and peak memory of one configuration on a full tile (run in a fresh process).

    python bench.py <mesh> <backend> <gpu_memory_gb|none> <cpu_memory_gb|none> <tag>
Writes bench/<mesh>_<tag>.parquet and appends a line to bench/bench.csv.
"""
import json
import resource
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lidarmetricpy import Options, compute_tile  # noqa: E402

mesh, backend, mem, host, tag = sys.argv[1:6]
mem = None if mem == "none" else float(mem)
host = None if host == "none" else float(host)
c = {d["mesh"]: d for d in json.loads((HERE / "data/full.json").read_text())}[mesh]
peak = [0]
stop = False


def poll():
    try:
        if backend.startswith("cupy"):
            import cupy
            pool = cupy.get_default_memory_pool()
            while not stop:
                peak[0] = max(peak[0], pool.total_bytes())
                time.sleep(0.02)
        elif backend.startswith("torch-cuda"):
            import torch
            while not stop:
                peak[0] = max(peak[0], torch.cuda.memory_reserved())
                time.sleep(0.02)
    except Exception:  # noqa: BLE001
        pass


th = threading.Thread(target=poll, daemon=True)
th.start()
t = time.time()
df = compute_tile(str(HERE / c["file"]), c["x0"], c["y1"], c["ncol"], c["nrow"], backend=backend, gpu_memory_gb=mem,
                  cpu_memory_gb=host, options=Options(spike_filter=False))
sec = time.time() - t
stop = True
th.join()
(HERE / "bench").mkdir(exist_ok=True)
df.to_parquet(HERE / "bench" / f"{mesh}_{tag}.parquet")
st = df.attrs["stats"]
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
line = (f"{mesh},{tag},{backend},{mem},{host},{c['n']},{sec:.1f},{st['blocks']},{st['passes']},"
        f"{peak[0] / 1024 ** 3:.2f},{rss:.2f}\n")
p = HERE / "bench" / "bench.csv"
if not p.exists():
    p.write_text("mesh,tag,backend,max_memory_gb,host_memory_gb,points,seconds,blocks,passes,peak_device_gb,peak_host_gb\n")
with open(p, "a") as f:
    f.write(line)
print(line, end="")
