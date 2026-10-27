"""Reference values from PyForestScan (installed package, calculate.py loaded directly: the rest of the
package imports PDAL, which is not needed here), one call per 25 m cell of a normalized clip.

    python -I ref_pyforestscan.py <laz> <x0> <y1> <ncol> <nrow> <out_prefix>

Per cell: assign_voxels(points, (25, 25, 1)) -> calculate_pad (k = 1, dz = 1) -> calculate_pai(min_height = 1),
calculate_canopy_cover(min_height = 2, k = 0.5), calculate_fhd(voxel_returns).
"""
import importlib.metadata as md
import importlib.util
import pathlib
import sys

import laspy
import numpy as np
import pandas as pd


def load_calculate():
    d = md.distribution("pyforestscan")
    f = [f for f in d.files if str(f).endswith("pyforestscan/calculate.py")][0]
    spec = importlib.util.spec_from_file_location("pfs_calculate", pathlib.Path(d.locate_file(f)))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, d.version


def main(laz, x0, y1, nc, nr, out):
    pfs, ver = load_calculate()
    las = laspy.read(laz)
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    col = np.floor((x - x0) / 25).astype(int)
    row = np.floor((y1 - y) / 25).astype(int)
    col[x == x0 + nc * 25] = nc - 1                # closed outer edges as in lidR / terra
    row[y == y1 - nr * 25] = nr - 1
    rows = []
    for r in range(nr):
        for c in range(nc):
            k = (row == r) & (col == c) & (z >= 0)
            rec = {"x": x0 + (c + 0.5) * 25, "y": y1 - (r + 0.5) * 25}
            if k.sum():
                arr = np.zeros(int(k.sum()), dtype=[("X", "f8"), ("Y", "f8"), ("HeightAboveGround", "f8")])
                arr["X"], arr["Y"], arr["HeightAboveGround"] = x[k], y[k], z[k]
                vr, _ = pfs.assign_voxels(arr, (25.0, 25.0, 1.0))
                assert vr.shape[:2] == (1, 1), vr.shape
                pad = pfs.calculate_pad(vr, 1.0)
                rec["pfs_pai"] = float(pfs.calculate_pai(pad, 1.0, min_height=1.0)[0, 0])
                rec["pfs_cover"] = float(pfs.calculate_canopy_cover(pad, 1.0, min_height=2.0, k=0.5)[0, 0])
                rec["pfs_fhd"] = float(pfs.calculate_fhd(vr)[0, 0])
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.attrs["version"] = ver
    df.to_csv(out + "_pfs.csv", index=False)
    print("cells", len(df), "pyforestscan", ver)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], float(a[1]), float(a[2]), int(a[3]), int(a[4]), a[5])
