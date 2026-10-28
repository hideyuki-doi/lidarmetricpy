"""Small single-cell cases at the edges of the lidRmetrics definitions (from the 2026-10-28 review and more).

    python tests/oracle_cases.py      → tests/data/ref/oracle_input.txt (heights as exact hex doubles)
Then tests/make_oracle.R runs lidRmetrics::metrics_set3 on each case → tests/data/ref/oracle_cases.csv.
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "data" / "ref" / "oracle_input.txt"


def cases():
    rng = np.random.default_rng(7)
    c = {}

    def add(name, z, rn=None, nr=None, zmin=0.5):
        z = np.asarray(z, np.float64)
        n = len(z)
        rn = np.ones(n, np.int64) if rn is None else np.asarray(rn, np.int64)
        nr = np.ones(n, np.int64) if nr is None else np.asarray(nr, np.int64)
        x = np.round(rng.uniform(0.5, 24.5, n), 2)
        y = np.round(rng.uniform(0.5, 24.5, n), 2)
        c[name] = dict(x=x, y=y, z=z, rn=rn, nr=nr, zmin=zmin)

    add("invalid_returns", [1, 2, 3, 4], rn=[1, 2, 1, 0], nr=[2, 2, 0, 1])
    add("kde_symmetric", [1, 2, 3, 4])
    add("constant", [5, 5, 5, 5])
    add("lad_boundary_up", [1, 2, 3 + 1e-12, 4])
    add("lad_boundary_down", [1, 2, 3 - 1e-12, 4])
    add("lad_small_range", [2.2, 2.5, 3.0 + 5e-9])
    add("two_points", [1, 4])
    add("three_points", [1, 2, 3])
    add("four_points", [1, 2, 3, 7])
    add("high_canopy", [1, 2, 61.2])
    add("single_point", [7])
    add("zmin2_no_voxel", [2.1, 2.2, 2.4], zmin=2.0)
    add("with_negative", [-0.5, 0.3, 1, 5, 9])
    add("all_low", [0.1, 0.2, 0.4])
    add("at_zmin_only", [0.5, 0.5, 0.5, 0.5])                     # round(0.5) = 1 > zmin: voxels, no n
    add("entropy_zmin_negative_zero", [0, 1, 2, 3], zmin=-0.5)     # metrics_dispersion: entropy of z > 0
    add("entropy_zmin_negative", [-0.2, 1, 2, 3], zmin=-0.5)
    k = np.arange(1, 13, dtype=float)
    eps = np.array([1e-12, -1e-12, 1e-9, -1e-9, 1e-7, -1e-7, 2e-7, -2e-7, 0, 0, 5e-8, -5e-8])
    add("near_integers", np.concatenate([k + eps, rng.gamma(3, 3, 40) + 0.6]))
    g = rng.gamma(3, 4, 60) + 0.6
    nrr = rng.integers(1, 4, 60)
    add("random_60", g, rn=np.minimum(rng.integers(1, 4, 60), nrr), nr=nrr)
    t = rng.uniform(0.6, 82, 500)
    nrr = rng.integers(1, 5, 500)
    add("tall_500", t, rn=np.minimum(rng.integers(1, 5, 500), nrr), nr=nrr)
    return c


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for name, d in cases().items():
            f.write(name + "\t" + repr(d["zmin"]) + "\t"
                    + " ".join(float(v).hex() for v in d["x"]) + "\t"
                    + " ".join(float(v).hex() for v in d["y"]) + "\t"
                    + " ".join(float(v).hex() for v in d["z"]) + "\t"
                    + " ".join(str(int(v)) for v in d["rn"]) + "\t"
                    + " ".join(str(int(v)) for v in d["nr"]) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
