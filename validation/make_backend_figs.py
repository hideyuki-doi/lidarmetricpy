"""Table and figures of the GPU / CPU comparison (measured values only).

Inputs: backend_compare/single.csv, backend_compare/throughput.csv (backend_compare.py),
lidR timings measured on the same machine: ref_full/<mesh>.log (/usr/bin/time, 1 core) and
par/lidr_par.log (8 processes, 8 full tiles).
Outputs: backend_compare/summary.md, fig_backend_single.png, fig_backend_throughput.png
"""
import re
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "backend_compare"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
# fixed categorical order (dataviz reference palette, light), lidR as neutral reference
COL = {"cupy": "#2a78d6", "torch-cuda": "#eb6834", "torch-cuda32": "#1baf7a", "numpy": "#eda100",
       "torch-cpu": "#e87ba4", "lidR": "#5f5e5a"}
LABEL = {"cupy": "CuPy (CUDA)", "torch-cuda": "PyTorch CUDA", "torch-cuda32": "PyTorch CUDA float32 (MPS-like)",
         "numpy": "NumPy (CPU)", "torch-cpu": "PyTorch CPU", "lidR": "lidR + lidRmetrics (R, 1 core)"}
ORDER = ["cupy", "torch-cuda", "torch-cuda32", "numpy", "torch-cpu"]
MESH = {"09LC9423": "dense tile (90.5 M points)", "09MC0523": "medium tile (48.3 M points)"}


def lidr_single():
    rows = []
    for m in MESH:
        txt = (HERE / "ref_full" / f"{m}.log").read_text(errors="ignore")
        s = re.search(r"([\d.]+) s (\d+) KB", txt)
        rows.append({"backend": "lidR", "mesh": m, "seconds": float(s.group(1)), "peak_rss_gb": int(s.group(2)) / 1024 ** 2,
                     "peak_device_gb": 0.0})
    return pd.DataFrame(rows)


def lidr_parallel():
    txt = (HERE / "par" / "lidr_par.log").read_text()
    s = float(re.search(r"\): ([\d.]+) s", txt).group(1))
    return {"backend": "lidR", "workers": 8, "tiles": 8, "seconds": s, "s_per_tile": s / 8}


def main():
    single = pd.read_csv(OUT / "single.csv")
    base = single[~single.other_packages.astype(bool)] if "other_packages" in single else single
    ld = lidr_single()
    thr = pd.read_csv(OUT / "throughput.csv")
    lp = lidr_parallel()

    # ---- table
    lines = ["# GPU / CPU comparison (measured)", "",
             "Machine: RTX PRO 6000 Blackwell Max-Q 96 GB, AMD Threadripper PRO 7955WX (16 cores / 32 threads), 251 GB RAM. lidRmetrics-equivalent metrics + trees",
             "(other_packages = False). Every number below is a measurement; nothing is extrapolated.", "",
             "## One full tile, one process (after warm-up)", "",
             "| backend | tile | seconds | peak GPU GB | peak RAM GB | metrics equal to lidR | trees / crowns equal |",
             "|---|---|---|---|---|---|---|"]
    for b in ORDER:
        for m in MESH:
            r = base[(base.backend == b) & (base.mesh == m)]
            if not len(r):
                continue
            r = r.iloc[-1]
            lines.append(f"| {b} | {m} | {r.seconds:.1f} | {r.peak_device_gb:.1f} | {r.peak_rss_gb:.1f} | "
                         f"{int(r.metrics_exact)}/{int(r.metrics_total)} | {int(r.trees_matched)}/{int(r.trees_ref)}, "
                         f"{int(r.crowns_equal)}/{int(r.trees_ref)} |")
    for _, r in ld.iterrows():
        lines.append(f"| lidR (R, 1 core) | {r.mesh} | {r.seconds:.1f} | – | {r.peak_rss_gb:.1f} | reference | reference |")
    if "other_packages" in single and single.other_packages.astype(bool).any():
        lines += ["", "With the other-package metrics (leafR, PyForestScan, ForestGapR; other_packages = True):", "",
                  "| backend | tile | seconds | seconds without |", "|---|---|---|---|"]
        for _, r in single[single.other_packages.astype(bool)].iterrows():
            w = base[(base.backend == r.backend) & (base.mesh == r.mesh)]
            lines.append(f"| {r.backend} | {r.mesh} | {r.seconds:.1f} | {w.seconds.iloc[-1]:.1f} |" if len(w) else
                         f"| {r.backend} | {r.mesh} | {r.seconds:.1f} | – |")
    lines += ["", "## Many tiles (throughput)", "",
              "Jobs alternate the two tiles. s/tile = wall time / tiles, including worker start-up.", "",
              "| backend | workers | tiles | seconds | s per tile |", "|---|---|---|---|---|"]
    for _, r in thr.sort_values(["backend", "workers"]).iterrows():
        lines.append(f"| {r.backend} | {int(r.workers)} | {int(r.tiles)} | {r.seconds:.1f} | {r.s_per_tile:.2f} |")
    lines.append(f"| lidR (8 R processes) | 8 | {lp['tiles']} | {lp['seconds']:.1f} | {lp['s_per_tile']:.2f} |")
    lines += ["", "lidR timings were measured on this machine with R 4.4.3 earlier the same day "
              "(ref_full/*.log, par/lidr_par.log); its outputs are identical under R 4.5.3."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")

    # ---- figure 1: single tile seconds and peak memory
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
    cats = ORDER + ["lidR"]
    allb = pd.concat([base, ld], ignore_index=True)
    y = range(len(cats))
    for j, m in enumerate(MESH):
        vals = [allb[(allb.backend == b) & (allb.mesh == m)].seconds.iloc[-1] if len(allb[(allb.backend == b) & (allb.mesh == m)]) else float("nan") for b in cats]
        off = -0.2 if j == 0 else 0.2
        axs[0].barh([i + off for i in y], vals, 0.36, color=[COL[b] for b in cats], alpha=1.0 if j == 0 else 0.55,
                    label=MESH[m])
        for i, v in zip(y, vals):
            axs[0].text(v * 1.08, i + off, f"{v:.1f} s", va="center", fontsize=7, color="#333")
    axs[0].set_xscale("log")
    axs[0].set_yticks(list(y), [LABEL[b] for b in cats], fontsize=8)
    axs[0].invert_yaxis()
    axs[0].set_xlabel("seconds per tile (log scale)")
    axs[0].set_title("(a) One tile, one process (dark: 90.5 M points, light: 48.3 M)", loc="left", fontsize=9)
    d = allb[allb.mesh == "09LC9423"]
    gpu = [d[d.backend == b].peak_device_gb.iloc[-1] if len(d[d.backend == b]) else 0 for b in cats]
    ram = [d[d.backend == b].peak_rss_gb.iloc[-1] if len(d[d.backend == b]) else 0 for b in cats]
    axs[1].barh([i - 0.2 for i in y], gpu, 0.36, color=[COL[b] for b in cats])
    axs[1].barh([i + 0.2 for i in y], ram, 0.36, color=[COL[b] for b in cats], alpha=0.45)
    for i, (g_, r_) in enumerate(zip(gpu, ram)):
        if g_ > 0:
            axs[1].text(g_ + 0.5, i - 0.2, f"GPU {g_:.0f}", va="center", fontsize=7)
        axs[1].text(r_ + 0.5, i + 0.2, f"RAM {r_:.0f}", va="center", fontsize=7)
    axs[1].set_yticks(list(y), [""] * len(cats))
    axs[1].invert_yaxis()
    axs[1].set_xlabel("peak memory, GB (90.5 M point tile)")
    axs[1].set_title("(b) Peak memory (dark: GPU, light: process RAM)", loc="left", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "fig_backend_single.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ---- figure 2: throughput vs workers
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    nudge = {"torch-cuda": 1.07, "cupy": 0.93}             # separate labels of points that nearly coincide
    for b in ORDER:
        t = thr[thr.backend == b].sort_values("workers")
        if not len(t):
            continue
        ax.plot(t.workers, t.s_per_tile, "-o", color=COL[b], lw=2, ms=5)
        ax.text(t.workers.iloc[-1] * 1.06, t.s_per_tile.iloc[-1] * nudge.get(b, 1.0),
                f"{LABEL[b]}  {t.s_per_tile.iloc[-1]:.2f} s", color="#333", fontsize=7.5, va="center")
    ax.plot([lp["workers"]], [lp["s_per_tile"]], "s", color=COL["lidR"], ms=7)
    ax.text(lp["workers"] * 1.06, lp["s_per_tile"], f"lidR, 8 R processes  {lp['s_per_tile']:.1f} s", fontsize=7.5,
            va="center", color="#333")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 4, 10, 20, 40], ["1", "2", "4", "10", "20", "40"])
    ax.minorticks_off()
    ax.set_xticks([1, 2, 4, 8, 12, 16], ["1", "2", "4", "8", "12", "16"])
    ax.set_xlabel("parallel worker processes")
    ax.set_ylabel("seconds per tile (wall time / tiles)")
    ax.set_title("Throughput on many full tiles (measured)", loc="left", fontsize=9)
    ax.set_xlim(0.8, 60)
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    plt.tight_layout()
    plt.savefig(OUT / "fig_backend_throughput.png", dpi=200, bbox_inches="tight")
    plt.close()
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
