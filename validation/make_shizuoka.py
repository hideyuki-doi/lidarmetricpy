"""Second validation dataset: whole tiles of the Shizuoka Prefecture ALS (VIRTUAL SHIZUOKA, CC BY 4.0,
s3://virtual-shizuoka, EPSG:6676), normalized like make_fulltile.py.

    python make_shizuoka.py 08MD4295 08OC0957 08NC0725 08MC7931     → data/<mesh>_sz.laz, data/shizuoka.json

Inputs in $PCM_SHIZUOKA_DIR (default ~/forest_data/shizuoka/raw):
  <mesh>*.las         LP "Original" point cloud (all returns, class 2 = ground), e.g. 2021/LP/Original/08/MD/42/08MD4295.zip
  grid/<mesh>*.txt    LP "Grid" ground model of the same tile: id, x, y, z, flag at 0.5 m pixel centres
(file names are lower or upper case and some have a suffix such as _org or _0.5g)
Each tile is 400 m × 300 m (16 × 12 cells of 25 m). Heights = z minus the grid pixel that contains the point,
kept for -1 < h < 60 m, then the spike filter (as the Kanagawa data). The grid is also written as a GeoTIFF
(<dir>/dtm/<mesh>.tif).
"""
import json
import os
import sys
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidarmetricpy.api import Grid, _spike_filter  # noqa: E402

SRC = Path(os.environ.get("PCM_SHIZUOKA_DIR", Path.home() / "forest_data/shizuoka/raw"))
OUT = Path(__file__).resolve().parent / "data"
RES = 0.5


def find(folder, mesh, ext):
    f = [p for p in folder.iterdir() if p.name.lower().startswith(mesh.lower()) and p.suffix.lower() == ext]
    if len(f) != 1:
        sys.exit(f"{mesh}: expected one {ext} file in {folder}, found {[p.name for p in f]}")
    return f[0]


def dtm_of(mesh):
    g = pd.read_csv(find(SRC / "grid", mesh, ".txt"), header=None, names=["id", "x", "y", "z", "flag"])
    x0, y1 = g.x.min() - RES / 2, g.y.max() + RES / 2
    nc, nr = int(round((g.x.max() + RES / 2 - x0) / RES)), int(round((y1 - (g.y.min() - RES / 2)) / RES))
    a = np.full((nr, nc), np.nan)
    a[np.round((y1 - g.y) / RES - 0.5).astype(int), np.round((g.x - x0) / RES - 0.5).astype(int)] = g.z
    a[a < -9000] = np.nan
    (SRC / "dtm").mkdir(exist_ok=True)
    with rasterio.open(SRC / "dtm" / f"{mesh}.tif", "w", driver="GTiff", width=nc, height=nr, count=1,
                       dtype="float64", nodata=np.nan, transform=from_origin(x0, y1, RES, RES)) as d:   # EPSG:6676
        d.write(a, 1)
    return a, float(x0), float(y1), nc, nr


res = []
for mesh in sys.argv[1:]:
    las = laspy.read(find(SRC, mesh, ".las"))
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    dem, x0, y1, nc, nr = dtm_of(mesh)
    ncol, nrow = int(nc * RES / 25), int(nr * RES / 25)
    k = (x >= x0) & (x < x0 + nc * RES) & (y > y1 - nr * RES) & (y <= y1)
    r = np.clip(np.floor((y1 - y[k]) / RES).astype(int), 0, nr - 1)
    c = np.clip(np.floor((x[k] - x0) / RES).astype(int), 0, nc - 1)
    hn = z[k] - dem[r, c]
    ok = np.isfinite(hn) & (hn > -1) & (hn < 60)
    idx = np.flatnonzero(k)[ok]
    keep = _spike_filter(x[idx], y[idx], hn[ok], Grid(x0, y1, ncol, nrow, 25))
    idx, h = idx[keep], hn[ok][keep]
    hdr = laspy.LasHeader(point_format=las.header.point_format.id, version=las.header.version)
    hdr.scales = np.array([las.header.scales[0], las.header.scales[1], 0.001])
    hdr.offsets = np.array([las.header.offsets[0], las.header.offsets[1], 0.0])
    out = laspy.LasData(hdr)
    for dim in ("X", "Y", "intensity", "return_number", "number_of_returns", "classification", "gps_time",
                "scan_angle_rank", "point_source_id"):
        setattr(out, dim, np.asarray(getattr(las, dim))[idx])
    out.Z = np.round(h / 0.001).astype(np.int32)
    p = OUT / f"{mesh}_sz.laz"
    out.write(p)
    res.append({"mesh": mesh, "file": f"data/{p.name}", "x0": x0, "y1": y1, "ncol": ncol, "nrow": nrow,
                "n": int(len(idx)), "pts_m2": round(len(idx) / (ncol * nrow * 625), 1),
                "z_max": round(float(h.max()), 2), "dtm_nan": int(np.isnan(dem).sum())})
    print(res[-1], flush=True)
(OUT / "shizuoka.json").write_text(json.dumps(res, indent=1))
