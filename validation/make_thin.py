"""Low-density copies of the Kanagawa clips for the lidR comparison.

National surveys usually have 1-20 points per m², the clips 52-133. Each clip of data/clips.json is thinned
to 2, 5 and 10 points per m² by keeping a random subset of laser pulses (all returns of a kept pulse stay
together; a pulse is one GPS time), which is how a sparser survey of the same forest would look. The thinned
points are written as LAZ, so lidR and lidarmetricpy read exactly the same points.

    python make_thin.py  → validation/data/<mesh>_d<density>.laz + thin.json
"""
import json
from pathlib import Path

import laspy
import numpy as np

HERE = Path(__file__).resolve().parent
DENSITIES = (2, 5, 10)             # points per m²
SEED = 20261111


def main():
    out = []
    for c in json.loads((HERE / "data/clips.json").read_text()):
        if c["n"] == 0:
            continue
        las = laspy.read(HERE / c["file"])
        area = c["ncol"] * c["nrow"] * 25.0 ** 2
        pulse, inv = np.unique(np.asarray(las.gps_time), return_inverse=True)
        rng = np.random.default_rng(SEED)
        order = rng.permutation(len(pulse))          # one order per clip: the sparser sets are nested
        for d in DENSITIES:
            n_pulse = int(round(len(pulse) * d * area / len(las.points)))
            keep = np.zeros(len(pulse), bool)
            keep[order[:n_pulse]] = True
            sub = laspy.LasData(las.header)
            sub.points = las.points[keep[inv]]
            p = HERE / "data" / f"{c['mesh']}_d{d}.laz"
            sub.write(p)
            n = int(keep[inv].sum())
            out.append({"mesh": f"{c['mesh']}_d{d}", "file": f"data/{p.name}", "x0": c["x0"], "y1": c["y1"],
                        "ncol": c["ncol"], "nrow": c["nrow"], "n": n, "pts_m2": round(n / area, 2),
                        "pulses_m2": round(n_pulse / area, 2), "source": c["mesh"]})
            print(out[-1], flush=True)
    (HERE / "data/thin.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
