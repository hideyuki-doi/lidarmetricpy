"""Whole-process timing of lidarmetricpy for comparison with the lidR scripts (run under /usr/bin/time):
start Python, import, compute one tile, save the cell table, the tree tops and the canopy raster
(the files ref_lidr.R saves).

    /usr/bin/time -f "%e s %M KB" python py_end_to_end.py <backend> <mesh> <out_prefix>
"""
import json
import sys
import time
from pathlib import Path

t_start = time.time()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lidarmetricpy import Options, compute_tile  # noqa: E402

t_import = time.time() - t_start
backend, mesh, out = sys.argv[1], sys.argv[2], sys.argv[3]
c = {d["mesh"]: d for d in json.loads((HERE / "data/full.json").read_text())}[mesh]
t0 = time.time()
df, (chm, L, (rr, cc, area)) = compute_tile(str(HERE / c["file"]), c["x0"], c["y1"], c["ncol"], c["nrow"],
                                            backend=backend, options=Options.lidr(other_packages=False),
                                            return_trees=True)
t_compute = time.time() - t0
t0 = time.time()
df.to_csv(f"{out}_cells.csv", index=False)
pd.DataFrame({"x": L.xmin + (cc + 0.5) * L.res, "y": L.ymax - (rr + 0.5) * L.res, "h": chm[rr, cc],
              "crown_m2": area}).to_csv(f"{out}_trees.csv", index=False)
import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
with rasterio.open(f"{out}_chm.tif", "w", driver="GTiff", height=chm.shape[0], width=chm.shape[1], count=1,
                   dtype="float64", transform=from_origin(L.xmin, L.ymax, L.res, L.res), nodata=np.nan) as f:
    f.write(chm, 1)
t_write = time.time() - t0
print(f"import_s {t_import:.2f} compute_s {t_compute:.2f} write_s {t_write:.2f} "
      f"stages {df.attrs['stats']['stage_seconds']}")
