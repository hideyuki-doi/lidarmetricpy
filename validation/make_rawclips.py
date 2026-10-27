"""Raw (not height-normalized) clips for canopyLazR, which works on elevations and treats the lowest
point of each column as the ground. Same 250 x 250 m areas as make_testdata.py, all points kept.

    python make_rawclips.py → data/<mesh>_raw.laz + data/rawclips.json
"""
import json
import os
from pathlib import Path

import laspy
import numpy as np

LAZ = Path(os.environ.get("PCM_LAZ_DIR", Path.home() / "forest_data/nakatsu/raw/laz"))          # Kanagawa 2024 ALS LAZ tiles
HERE = Path(__file__).resolve().parent


def main():
    clips = [c for c in json.loads((HERE / "data/clips.json").read_text()) if c["n"] > 0]
    out = []
    for c in clips:
        las = laspy.read(LAZ / f"{c['mesh']}.laz")
        x, y = np.asarray(las.x), np.asarray(las.y)
        k = (x >= c["x0"]) & (x < c["x0"] + 250) & (y > c["y1"] - 250) & (y <= c["y1"])
        sub = laspy.LasData(las.header)
        sub.points = las.points[np.flatnonzero(k)]
        p = HERE / "data" / f"{c['mesh']}_raw.laz"
        sub.write(p)
        out.append({"mesh": c["mesh"], "file": f"data/{p.name}", "x0": c["x0"], "y1": c["y1"], "n": int(k.sum())})
        print(out[-1], flush=True)
    (HERE / "data/rawclips.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
