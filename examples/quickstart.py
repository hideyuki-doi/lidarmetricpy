"""Out-of-the-box example: metrics of the synthetic tile shipped with the tests (100 m × 100 m, heights already
normalised), on a 10 m grid, and four maps of them.

    python examples/quickstart.py [backend]        → examples/quickstart_cells.csv, examples/quickstart.png

Runs on a CPU (NumPy) in about a minute; pass cupy, torch-cuda or mps to use a GPU. Needs matplotlib for the
figure.
"""
import sys
from pathlib import Path

import numpy as np

from lidarmetricpy import Options, compute_tile

HERE = Path(__file__).resolve().parent
LAZ = HERE.parent / "tests" / "data" / "synthetic.laz"
backend = sys.argv[1] if len(sys.argv) > 1 else "numpy"

# grid: left edge x0, top edge y1, 10 × 10 cells of 10 m; Z is height above ground, so no DTM
df = compute_tile(str(LAZ), x0=-62000.0, y1=-57750.0, ncol=10, nrow=10, cell=10.0, dtm=None,
                  backend=backend, options=Options.lidr())
df.to_csv(HERE / "quickstart_cells.csv", index=False)
print(f"{len(df)} cells, {df.shape[1]} columns; backend {backend}")
print(df[["x", "y", "n", "zq95", "pzabove2", "leafr_fhd", "enl", "tree_n"]].head().to_string(index=False))

import matplotlib.pyplot as plt  # noqa: E402

panels = [("zq95", "95th height percentile (m)"), ("pzabove2", "returns above 2 m (%)"),
          ("leafr_fhd", "foliage height diversity"), ("tree_n", "tree tops per cell")]
fig, axes = plt.subplots(2, 2, figsize=(7, 6.4), constrained_layout=True)
for ax, (col, title) in zip(axes.ravel(), panels):
    a = np.full((10, 10), np.nan)
    r = np.round((-57750.0 - df.y) / 10 - 0.5).astype(int)
    c = np.round((df.x + 62000.0) / 10 - 0.5).astype(int)
    a[r, c] = df[col]
    im = ax.imshow(a, cmap="viridis", extent=(0, 100, 0, 100))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("m")
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.savefig(HERE / "quickstart.png", dpi=150)
print(HERE / "quickstart.png")
