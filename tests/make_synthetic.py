"""Synthetic test tiles (no external data needed), written once and kept in tests/data.

    python tests/make_synthetic.py      → tests/data/synthetic.laz (heights above ground)
                                          tests/data/synthetic_raw.laz (same points on a sloping terrain)

100 m × 100 m (4 × 4 cells of 25 m) with ~200 k laser pulses: ground, paraboloid tree crowns with multiple
returns, a treeless gap, one cell without any point, and points exactly on cell edges. The reference values
in tests/data/ref were made from these files with the R / Python packages (see tests/README.md).
"""
from pathlib import Path

import laspy
import numpy as np

OUT = Path(__file__).resolve().parent / "data"
X0, Y1, SIDE = -62000.0, -57750.0, 100.0          # grid: x0, top edge, 4 × 4 cells of 25 m


def terrain(x, y):
    return 120.0 + 0.08 * (x - X0) + 3.0 * np.sin((Y1 - y) / 17.0)


def main(seed=1):
    rng = np.random.default_rng(seed)
    # trees: none in the north-east cell (empty) and few in the south-west cell (gap)
    nt = 260
    tx, ty = X0 + rng.uniform(0, SIDE, nt), Y1 - rng.uniform(0, SIDE, nt)
    empty = (tx >= X0 + 75) & (ty > Y1 - 25)
    gap = (tx < X0 + 25) & (ty <= Y1 - 75) & (rng.uniform(size=nt) < 0.85)
    tx, ty = tx[~empty & ~gap], ty[~empty & ~gap]
    th = rng.uniform(6, 32, len(tx))
    tr = 0.12 * th + rng.uniform(0.8, 2.0, len(tx))

    npulse = int(SIDE * SIDE * 20)
    px = np.round(X0 + rng.uniform(0, SIDE, npulse), 2)
    py = np.round(Y1 - rng.uniform(0, SIDE, npulse), 2)
    # some pulses exactly on cell edges (edge rules of the grid)
    e = rng.choice(npulse, 400, replace=False)
    px[e[:200]] = X0 + 25 * rng.integers(0, 5, 200)
    py[e[200:]] = Y1 - 25 * rng.integers(0, 5, 200)
    keep = ~((px >= X0 + 75) & (py > Y1 - 25))                   # north-east cell: no point at all
    px, py = px[keep], py[keep]
    npulse = len(px)

    # canopy surface height under each pulse (highest crown)
    top = np.zeros(npulse)
    for x, y, h, r in zip(tx, ty, th, tr):
        d2 = (px - x) ** 2 + (py - y) ** 2
        m = d2 < r * r
        top[m] = np.maximum(top[m], h - 0.6 * h * d2[m] / (r * r))
    canopy = top > 0
    nret = np.where(canopy, rng.choice([1, 2, 3, 4], npulse, p=[0.35, 0.3, 0.2, 0.15]), 1)
    nret = np.where(~canopy, rng.choice([1, 2], npulse, p=[0.95, 0.05]), nret)
    xs, ys, zs, rn, nr = [], [], [], [], []
    for i in range(1, 5):
        m = nret >= i
        if i == 1:
            z = np.where(canopy, top - rng.exponential(0.4, npulse), rng.normal(0, 0.04, npulse))
            prev = z
        else:
            drop = rng.exponential(3.0, npulse)
            z = np.where(nret == i, np.where(rng.uniform(size=npulse) < 0.5, rng.normal(0, 0.04, npulse),
                                             np.maximum(prev - drop, 0.0)), np.maximum(prev - drop, 0.0))
            prev = np.where(m, z, prev)
        xs.append(px[m]); ys.append(py[m]); zs.append(z[m])
        rn.append(np.full(m.sum(), i)); nr.append(nret[m])
    x, y, z = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
    rn, nr = np.concatenate(rn), np.concatenate(nr)
    z = np.maximum(z, -0.5)
    o = rng.permutation(len(x))                                   # file order is not spatial
    x, y, z, rn, nr = x[o], y[o], np.round(z[o], 3), rn[o], nr[o]

    OUT.mkdir(exist_ok=True)
    for name, zz in (("synthetic", z), ("synthetic_raw", np.round(z + terrain(x, y), 3))):
        hdr = laspy.LasHeader(point_format=1, version="1.2")
        hdr.scales = np.array([0.01, 0.01, 0.001])
        hdr.offsets = np.array([X0, Y1 - SIDE, 0.0])
        try:
            import pyproj
            hdr.add_crs(pyproj.CRS.from_epsg(6677))              # JGD2011 / Japan Plane IX (any metric CRS works)
        except Exception:  # noqa: BLE001
            pass
        las = laspy.LasData(hdr)
        las.x, las.y, las.z = x, y, zz
        las.return_number, las.number_of_returns = rn.astype(np.uint8), nr.astype(np.uint8)
        las.intensity = rng.integers(1, 2000, len(x)).astype(np.uint16)
        las.classification = np.where(z < 0.2, 2, 1).astype(np.uint8)
        las.write(OUT / f"{name}.laz")
        print(OUT / f"{name}.laz", len(x), "points")


if __name__ == "__main__":
    main()
