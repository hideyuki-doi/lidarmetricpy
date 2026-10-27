"""Whole tiles, normalized like make_testdata.py (for timing and a full-tile lidR check)."""
import os
import json, sys
from pathlib import Path
import laspy, numpy as np, rasterio
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcmetrics.api import Grid, _spike_filter
LAZ = Path(os.environ.get("PCM_LAZ_DIR", Path.home() / "forest_data/nakatsu/raw/laz"))          # Kanagawa 2024 ALS LAZ tiles
DEM = Path(os.environ.get("PCM_DEM_DIR", Path.home() / "forest_data/kanagawa2024/dem"))          # matching 0.5 m DEM tiles
OUT = Path(__file__).resolve().parent / "data"
res = []
for mesh in sys.argv[1:]:
    las = laspy.read(LAZ / f"{mesh}.laz")
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    x0 = float(np.floor(np.median(x) / 1000) * 1000); y1 = float(np.ceil(np.median(y) / 750) * 750)
    dp = DEM / f"{mesh}_m.tif"; dp = dp if dp.exists() else DEM / f"{mesh}.tif"
    with rasterio.open(dp) as s:
        dem = s.read(1, window=rasterio.windows.from_bounds(x0, y1 - 750, x0 + 1000, y1, s.transform),
                     boundless=True, fill_value=np.nan).astype(np.float64)
    dem[dem < -9000] = np.nan
    k = (x >= x0) & (x < x0 + 1000) & (y > y1 - 750) & (y <= y1)
    r = np.clip(((y1 - y[k]) / 0.5).astype(int), 0, 1499); c = np.clip(((x[k] - x0) / 0.5).astype(int), 0, 1999)
    hn = z[k] - dem[r, c]; ok = np.isfinite(hn) & (hn > -1) & (hn < 60)
    idx = np.flatnonzero(k)[ok]
    keep = _spike_filter(x[idx], y[idx], hn[ok], Grid(x0, y1, 40, 30, 25)); idx, h = idx[keep], hn[ok][keep]
    hdr = laspy.LasHeader(point_format=las.header.point_format.id, version=las.header.version)
    hdr.scales = np.array([las.header.scales[0], las.header.scales[1], 0.001]); hdr.offsets = np.array([las.header.offsets[0], las.header.offsets[1], 0.0])
    out = laspy.LasData(hdr)
    for dim in ("X", "Y", "intensity", "return_number", "number_of_returns", "classification", "gps_time", "scan_angle_rank", "point_source_id"):
        setattr(out, dim, np.asarray(getattr(las, dim))[idx])
    out.Z = np.round(h / 0.001).astype(np.int32)
    p = OUT / f"{mesh}_full.laz"; out.write(p)
    res.append({"mesh": mesh, "file": f"data/{p.name}", "x0": x0, "y1": y1, "ncol": 40, "nrow": 30, "n": int(len(idx))}); print(res[-1], flush=True)
(OUT / "full.json").write_text(json.dumps(res, indent=1))
