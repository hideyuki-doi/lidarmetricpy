"""Normalized test clips for the lidR comparison.

For each tile: heights above ground with the tile DEM (nearest 0.5 m cell), points between
-1 and 60 m, the 1 m-column spike filter, then a 250 x 250 m clip (10 x 10 cells of 25 m)
in the tile centre. Written as LAZ with Z = height above ground (scale 0.001), so lidR and
lidarmetricpy read exactly the same points.

    python make_testdata.py  → validation/data/<mesh>_clip.laz + clips.json
"""
import json
import os
import sys
from pathlib import Path

import laspy
import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidarmetricpy.api import Grid, _spike_filter  # noqa: E402

LAZ = Path(os.environ.get("PCM_LAZ_DIR", Path.home() / "forest_data/nakatsu/raw/laz"))          # Kanagawa 2024 ALS LAZ tiles
DEM = Path(os.environ.get("PCM_DEM_DIR", Path.home() / "forest_data/kanagawa2024/dem"))          # matching 0.5 m DEM tiles
OUT = Path(__file__).resolve().parent / "data"
TILES = ["09MC0623", "09LC9542", "09MC0523", "09LC9511", "09LC9423"]
SIZE = 250


def main(tiles=TILES):
    OUT.mkdir(exist_ok=True)
    clips = []
    for mesh in tiles:
        las = laspy.read(LAZ / f"{mesh}.laz")
        x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
        x0t = float(np.floor(np.median(x) / 1000) * 1000)
        y1t = float(np.ceil(np.median(y) / 750) * 750)
        dp = DEM / f"{mesh}_m.tif"
        dp = dp if dp.exists() else DEM / f"{mesh}.tif"
        with rasterio.open(dp) as s:
            win = rasterio.windows.from_bounds(x0t, y1t - 750, x0t + 1000, y1t, s.transform)
            dem = s.read(1, window=win, boundless=True, fill_value=np.nan).astype(np.float64)
        dem[dem < -9000] = np.nan
        # clip in the tile centre, aligned to the 25 m grid
        cx0 = x0t + 375
        cy1 = y1t - 250
        k = (x >= cx0) & (x < cx0 + SIZE) & (y > cy1 - SIZE) & (y <= cy1)
        r = np.clip(((y1t - y[k]) / 0.5).astype(int), 0, 1499)
        c = np.clip(((x[k] - x0t) / 0.5).astype(int), 0, 1999)
        hn = z[k] - dem[r, c]
        ok = np.isfinite(hn) & (hn > -1) & (hn < 60)
        idx = np.flatnonzero(k)[ok]
        keep = _spike_filter(x[idx], y[idx], hn[ok], Grid(cx0, cy1, 10, 10, 25))
        idx, h = idx[keep], hn[ok][keep]
        hdr = laspy.LasHeader(point_format=las.header.point_format.id, version=las.header.version)
        hdr.scales = np.array([las.header.scales[0], las.header.scales[1], 0.001])
        hdr.offsets = np.array([las.header.offsets[0], las.header.offsets[1], 0.0])
        out = laspy.LasData(hdr)
        for dim in ("X", "Y", "intensity", "return_number", "number_of_returns", "classification", "gps_time",
                    "scan_angle_rank", "point_source_id"):
            setattr(out, dim, np.asarray(getattr(las, dim))[idx])
        out.Z = np.round(h / 0.001).astype(np.int32)
        p = OUT / f"{mesh}_clip.laz"
        out.write(p)
        clips.append({"mesh": mesh, "file": f"data/{p.name}", "x0": cx0, "y1": cy1, "ncol": 10, "nrow": 10,
                      "n": int(len(idx)), "pts_m2": round(len(idx) / SIZE ** 2, 1)})
        print(clips[-1], flush=True)
    (OUT / "clips.json").write_text(json.dumps(clips, indent=1))


if __name__ == "__main__":
    main()
