"""Figures and tables for the lidarmetricpy methods paper (all values read from measured files).

Inputs
  docs/data/cells_vs_reference.parquet, canopylazr_vs_reference.parquet   (make_paper_data.py)
  docs/data/memory_budget.csv                                             (memory_budget.py)
  validation/backend_compare/single.csv, throughput.csv                  (backend_compare.py)
  validation/ref_full/*.log, validation/par/lidr_par.log                  (lidR timings, /usr/bin/time)
Outputs
  docs/figures/fig1_workflow … fig6_memory (.png 300 dpi + .pdf), docs/tables/table1–5 (.csv + .md)
"""
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
VAL = HERE.parent / "validation"
FIG, TAB, DAT = HERE / "figures", HERE / "tables", HERE / "data"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)
sys.path.insert(0, str(VAL))
from compare import METRICS as LIDR  # noqa: E402

OTHER = ["leafr_lai", "leafr_lahv", "leafr_lahv_laiw", "leafr_lahv_hw", "leafr_lahv_laiw_hw", "leafr_fhd",
         "leafr_fhd_even", "leafr_gs", "leafr_gs_even", "leafr_gc", "pfs_pai", "pfs_cover", "pfs_fhd", "enl",
         "gap_frac", "gap_n", "gap_area_mean"]
CLZ = ["clz_chm", "clz_ground", "clz_vertical_sd_lad", "clz_rugosity", "clz_filled_ratio", "clz_porosity_ratio",
       "clz_max_lad", "clz_max_lad_ht", "clz_toc_rugosity"]
TREES = [m for m in LIDR if m.startswith(("tree_", "crown_"))]
BACKENDS = ["cupy", "torch-cuda", "numpy", "torch-cpu", "torch-cuda32"]
BLABEL = {"cupy": "CuPy\n(CUDA)", "torch-cuda": "PyTorch\nCUDA", "numpy": "NumPy\n(CPU)", "torch-cpu": "PyTorch\nCPU",
          "torch-cuda32": "float32\n(MPS-like)"}
# fixed categorical order (dataviz reference palette); lidR as neutral reference
BCOL = {"cupy": "#2a78d6", "torch-cuda": "#eb6834", "torch-cuda32": "#1baf7a", "numpy": "#eda100",
        "torch-cpu": "#e87ba4", "lidR": "#5f5e5a"}
PKG_ORDER = ["lidRmetrics", "lidR trees", "leafR", "PyForestScan", "forestr", "ForestGapR", "canopyLazR"]
PCOL = dict(zip(PKG_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))
INK, INK2, GRID = "#1f1f1f", "#5f5e5a", "#e6e6e6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlesize": 8.5,
                     "axes.titleweight": "bold", "legend.frameon": False})
TOL = 1e-6


def package(m):
    if m in TREES:
        return "lidR trees"
    if m in LIDR:
        return "lidRmetrics"
    if m.startswith("leafr"):
        return "leafR"
    if m.startswith("pfs"):
        return "PyForestScan"
    if m == "enl":
        return "forestr"
    if m.startswith("gap"):
        return "ForestGapR"
    return "canopyLazR"


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def pair(df, m):
    a, b = df[m + "_ref"].astype(float), df[m + "_pcm"].astype(float)
    if m == "tree_n":                           # the R reference leaves NA where a cell has no tree
        a = a.fillna(0.0)
    return a, b


def agreement(df, m):
    a, b = pair(df, m)
    both = a.notna() & b.notna()
    nan_mis = int((a.isna() != b.isna()).sum())
    d = (a[both] - b[both]).abs()
    eq = (d <= TOL * np.maximum(1, a[both].abs())) | (a[both] == b[both])
    fin = np.isfinite(d)
    rel = (d[fin] / np.maximum(1, a[both][fin].abs()))
    return {"n": int(both.sum()), "nan_mismatch": nan_mis, "equal_share": float(eq.mean()) if len(eq) else np.nan,
            "max_abs_diff": float(d[fin].max()) if fin.any() else 0.0,
            "max_rel_diff": float(rel.max()) if len(rel) else 0.0}


# ------------------------------------------------------------------ figure 1: workflow
def fig_workflow():
    fig, ax = plt.subplots(figsize=(7.1, 3.1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")
    yc, yg, hb = 29, 9, 11
    for lab, y, col in [("CPU (host)", yc, "#f3f2ee"), ("device: GPU (CuPy / PyTorch) or CPU (NumPy / PyTorch)", yg, "#e8f0fb")]:
        ax.add_patch(FancyBboxPatch((0.2, y - 7), 79.5, 14, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=col, ec="none", zorder=0))
        ax.text(20.6, y + 8, lab, fontsize=6.5, color=INK2, va="bottom")

    def box(col, y, text, gpu=False, x=None, w=18.6):
        x = 1.0 + col * 19.6 if x is None else x
        ax.add_patch(FancyBboxPatch((x, y - hb / 2), w, hb, boxstyle="round,pad=0,rounding_size=1",
                                    fc="white", ec=BCOL["cupy"] if gpu else INK2, lw=1.0, zorder=2))
        ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=5.5, zorder=3, linespacing=1.25)
        return (x, y, w)

    def arrow(p1, p2, rad=0.0):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=INK2,
                                     connectionstyle=f"arc3,rad={rad}", zorder=1, shrinkA=0, shrinkB=0))

    def right(b): return (b[0] + b[2], b[1])
    def left(b): return (b[0], b[1])
    def top(b): return (b[0] + b[2] / 2, b[1] + hb / 2)
    def bot(b): return (b[0] + b[2] / 2, b[1] - hb / 2)

    c1 = box(0, yc, "LAZ decoding\n(lazrs, parallel chunks;\nmulti-pass if host budget)")
    g1 = box(0, yg, "X·scale + offset\n(no FMA), cell index", True)
    g2 = box(1, yg, "height above DTM,\nz range, spike filter,\nsort by cell", True)
    g3 = box(2, yg, "per-cell metrics:\nsegmented sums, KDE,\nvoxels, layer counts", True)
    g4 = box(3, yg, "0.5 m canopy raster,\nlocal maxima\n(int64 atomics)", True)
    c3 = box(2, yc, "rumple: Delaunay\n(qhull QbB Qt); leafR /\nPyForestScan formulas")
    c4 = box(3, yc, "tree ties, Dalponte\ncrowns (numba), knnidw;\nForestGapR gaps")
    ox, ow = 82.4, 17.0
    ax.add_patch(FancyBboxPatch((ox, yg - hb / 2), ow, yc - yg + hb, boxstyle="round,pad=0,rounding_size=1",
                                fc="white", ec=INK, lw=1.2, zorder=2))
    ax.text(ox + ow / 2, (yc + yg) / 2, "per-cell table\n\nlidRmetrics (24)\nlidR trees (9)\nleafR (10)\n"
            "PyForestScan (3)\nforestr ENL (1)\nForestGapR (3)", ha="center", va="center", fontsize=5.9,
            linespacing=1.3, zorder=3)
    arrow(bot(c1), top(g1))
    for p, q in [(g1, g2), (g2, g3), (g3, g4)]:
        arrow(right(p), left(q))
    arrow(top(g3), bot(c3))
    arrow(top(g4), bot(c4))
    xm = ox + ow / 2
    ax.plot([top(c3)[0], top(c3)[0], xm], [top(c3)[1], 38.2, 38.2], color=INK2, lw=0.8, zorder=1,
            solid_joinstyle="round")
    arrow((xm, 38.2), (xm, yc + hb / 2))
    arrow(right(c4), (ox, yc))
    arrow(right(g4), (ox, yg))
    ax.plot([bot(g3)[0], bot(g3)[0], xm], [bot(g3)[1], 0.6, 0.6], color=INK2, lw=0.8, zorder=1)
    arrow((xm, 0.6), (xm, yg - hb / 2))
    ax.text(1.0, 44, "with a memory budget the device stages run on blocks of cell rows",
            fontsize=6.0, color=INK2, va="top", style="italic")
    save(fig, "fig1_workflow")


# ------------------------------------------------------------------ figure 2: agreement scatter
def fig_scatter(cells, clz):
    sel = ["zq20", "zentropy", "L4", "Oligophotic", "vzcv", "rumple", "kde_peaks_count", "tree_n", "crown_m2_mean",
           "leafr_lai", "leafr_fhd", "pfs_pai", "pfs_fhd", "enl", "gap_frac", "clz_rugosity"]
    c64 = cells[cells.backend == "cupy"]
    fig, axs = plt.subplots(4, 4, figsize=(7.1, 7.3))
    for ax, m in zip(axs.ravel(), sel):
        df = clz if m in CLZ else c64
        a, b = pair(df, m)
        ok = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
        pk = package(m)
        ax.scatter(a[ok], b[ok], s=3, lw=0, color=PCOL[pk], alpha=0.6, rasterized=True)
        lo, hi = np.nanmin(a[ok]), np.nanmax(a[ok])
        pad = (hi - lo) * 0.05 or 1
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK2, lw=0.6, zorder=0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ag = agreement(df, m)
        ax.set_title(m, loc="left", fontsize=7.5)
        ax.text(0.04, 0.96, f"{pk}\nn = {ag['n']:,}\nmax |Δ| = {ag['max_abs_diff']:.1e}", transform=ax.transAxes,
                va="top", fontsize=5.8, color=INK2)
        ax.tick_params(labelsize=6)
    fig.supxlabel("reference package (R / Python)", fontsize=8)
    fig.supylabel("lidarmetricpy (CuPy, float64)", fontsize=8)
    plt.tight_layout()
    save(fig, "fig2_agreement_scatter")


# ------------------------------------------------------------------ figure 3: max relative difference heatmap
def fig_heatmap(cells, clz):
    rows = []
    for m in LIDR + OTHER:
        for bk in BACKENDS:
            ag = agreement(cells[cells.backend == bk], m)
            rows.append({"metric": m, "package": package(m), "backend": bk, **ag})
    for m in CLZ:
        rows.append({"metric": m, "package": "canopyLazR", "backend": "numpy", **agreement(clz, m)})
    res = pd.DataFrame(rows)
    metrics = LIDR + OTHER + CLZ
    mat = np.full((len(metrics), len(BACKENDS)), np.nan)
    flag = np.zeros_like(mat, bool)
    for i, m in enumerate(metrics):
        for j, bk in enumerate(BACKENDS):
            r = res[(res.metric == m) & (res.backend == bk)]
            if len(r):
                mat[i, j] = max(r.max_rel_diff.iloc[0], 1e-17)
                flag[i, j] = (r.equal_share.iloc[0] < 1) or (r.nan_mismatch.iloc[0] > 0)
    fig, ax = plt.subplots(figsize=(4.2, 9.2))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#f3f2ee")
    im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap=cmap, norm=LogNorm(1e-17, 1e-1))
    for i, j in zip(*np.nonzero(flag)):
        ax.text(j, i, "×", ha="center", va="center", fontsize=7, color="#c0392b", fontweight="bold")
    for i in range(len(metrics)):
        for j in range(len(BACKENDS)):
            if np.isnan(mat[i, j]):
                ax.text(j, i, "–", ha="center", va="center", fontsize=6, color=INK2)
    ax.set_yticks(range(len(metrics)), metrics, fontsize=5.8)
    ax.set_xticks(range(len(BACKENDS)), [BLABEL[b] for b in BACKENDS], fontsize=6.2)
    ax.xaxis.tick_top()
    bounds = np.cumsum([len(LIDR) - len(TREES), len(TREES), 10, 3, 1, 3])
    for y in bounds:
        ax.axhline(y - 0.5, color="white", lw=2)
    for (y0, y1), lab in zip(zip(np.r_[0, bounds], np.r_[bounds, len(metrics)]), PKG_ORDER):
        ax.text(len(BACKENDS) - 0.35, (y0 + y1 - 1) / 2, lab, fontsize=6.2, va="center", color=PCOL[lab],
                fontweight="bold")
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.025, pad=0.02, aspect=30)
    cb.set_label("max |lidarmetricpy − reference| / max(1, |reference|) over all cells", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    cb.ax.axvline(1e-6, color="#c0392b", lw=1)
    ax.set_title("Agreement with the reference packages\n(× : at least one cell outside 1e-6 or NA pattern differs; "
                 "– : not computed)", fontsize=6.8, loc="left", pad=28)
    save(fig, "fig3_agreement_heatmap")
    return res


# ------------------------------------------------------------------ figure 4: one tile time and stages
def lidr_times():
    out = {}
    for m in ("09LC9423", "09MC0523"):
        txt = (VAL / "ref_full" / f"{m}.log").read_text(errors="ignore")
        tot = float(re.search(r"([\d.]+) s (\d+) KB", txt).group(1))
        rss = int(re.search(r"([\d.]+) s (\d+) KB", txt).group(2)) / 1024 ** 2
        a = re.search(r"area_s ([\d.]+) tree_s ([\d.]+)", txt)
        out[m] = {"seconds": tot, "metrics": float(a.group(1)), "trees": float(a.group(2)), "rss": rss}
    return out


def fig_single(single):
    base = single[~single.other_packages.astype(bool)]
    ld = lidr_times()
    d = base[base.mesh == "09LC9423"].set_index("backend")
    order = ["cupy", "torch-cuda", "torch-cuda32", "numpy", "torch-cpu"]
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.5), gridspec_kw={"width_ratios": [1, 1.1]})
    tot = [d.loc[b, "seconds"] for b in order] + [ld["09LC9423"]["seconds"]]
    labs = [BLABEL[b].replace("\n", " ") for b in order] + ["lidR + lidRmetrics (R, 1 core)"]
    cols = [BCOL[b] for b in order] + [BCOL["lidR"]]
    y = np.arange(len(tot))
    axs[0].barh(y, tot, color=cols, height=0.62)
    for i, v in enumerate(tot):
        axs[0].text(v + 3, i, f"{v:.1f} s", va="center", fontsize=6.5)
    axs[0].set_yticks(y, labs, fontsize=6.5)
    axs[0].invert_yaxis()
    axs[0].set_xlabel("seconds, one 90.5 M point tile")
    axs[0].set_title("(a) Wall time, one process", loc="left")
    axs[0].set_xlim(0, 270)
    axs[0].grid(axis="x", color=GRID, lw=0.6)
    stages = [("read + LAZ decode", "t_read", "#8c8a83"), ("prepare", "t_prepare", "#c9c7c0"),
              ("cell metrics", None, "#2a78d6"), ("rumple (qhull)", "t_rumple", "#1baf7a"),
              ("canopy raster", "t_canopy", "#eda100"), ("trees", "t_trees", "#eb6834")]
    left = np.zeros(len(order))
    for lab, col, c in stages:
        if col is None:
            v = np.array([d.loc[b, "t_area"] - d.loc[b, "t_rumple"] - d.loc[b, "t_canopy"] for b in order])
        else:
            v = np.array([d.loc[b, col] for b in order])
        axs[1].barh(range(len(order)), v, left=left, color=c, height=0.62, label=lab,
                    edgecolor="white", linewidth=0.6)
        left += v
    other = np.array([d.loc[b, "seconds"] for b in order]) - left
    axs[1].barh(range(len(order)), np.maximum(other, 0), left=left, color="#e6e6e6", height=0.62, label="other",
                edgecolor="white", linewidth=0.6)
    axs[1].set_yticks(range(len(order)), [BLABEL[b].replace("\n", " ") for b in order], fontsize=6.5)
    axs[1].invert_yaxis()
    axs[1].set_xlabel("seconds")
    axs[1].set_title("(b) lidarmetricpy stages (same tile)", loc="left")
    axs[1].legend(fontsize=6, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5))
    axs[1].grid(axis="x", color=GRID, lw=0.6)
    plt.tight_layout()
    save(fig, "fig4_single_tile")


# ------------------------------------------------------------------ figure 5: throughput
def fig_throughput(thr):
    txt = (VAL / "par" / "lidr_par.log").read_text()
    lsec = float(re.search(r"\): ([\d.]+) s", txt).group(1)) / 8
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    lab = {"cupy": "CuPy", "torch-cuda": "PyTorch CUDA", "torch-cuda32": "float32 (MPS-like)", "numpy": "NumPy",
           "torch-cpu": "PyTorch CPU"}
    nudge = {"torch-cuda": 1.12, "cupy": 0.9}
    for b in ["cupy", "torch-cuda", "torch-cuda32", "numpy", "torch-cpu"]:
        t = thr[thr.backend == b].sort_values("workers")
        ax.plot(t.workers, t.s_per_tile, "-o", color=BCOL[b], lw=1.6, ms=3.5)
        ax.text(t.workers.iloc[-1] * 1.08, t.s_per_tile.iloc[-1] * nudge.get(b, 1), f"{lab[b]} {t.s_per_tile.iloc[-1]:.2f} s",
                fontsize=6, va="center")
    ax.plot([8], [lsec], "s", color=BCOL["lidR"], ms=5)
    ax.text(8 * 1.08, lsec, f"lidR, 8 processes {lsec:.1f} s", fontsize=6, va="center")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    ax.set_yticks([1, 2, 5, 10, 20, 40], ["1", "2", "5", "10", "20", "40"])
    ax.minorticks_off()
    ax.set_xlim(0.8, 90)
    ax.set_xlabel("parallel worker processes")
    ax.set_ylabel("seconds per tile")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_title("Throughput, full tiles (90.5 M and 48.3 M points)", loc="left", fontsize=7.5)
    save(fig, "fig5_throughput")


# ------------------------------------------------------------------ figure 6: memory budgets
def fig_memory(mb):
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.8), gridspec_kw={"width_ratios": [1.35, 1]})
    g = mb[mb.backend == "cupy"].copy()
    g["label"] = [("no limit" if pd.isna(r.gpu_memory_gb) else f"GPU {r.gpu_memory_gb:g} GiB")
                  + ("" if pd.isna(r.cpu_memory_gb) else f", RAM {r.cpu_memory_gb:g} GiB") for r in g.itertuples()]
    y = np.arange(len(g))
    axs[0].barh(y - 0.2, g.peak_gpu_gb, 0.38, color=BCOL["cupy"], label="peak GPU memory pool (GiB)")
    axs[0].barh(y + 0.2, g.peak_rss_gb, 0.38, color=BCOL["cupy"], alpha=0.4, label="peak process RAM (GiB)")
    for i, r in enumerate(g.itertuples()):
        axs[0].text(np.nanmax([r.peak_gpu_gb, r.peak_rss_gb, r.gpu_memory_gb]) + 0.8, i,
                    f"{r.seconds:.1f} s, {r.path}, {r.blocks} block{'s' if r.blocks > 1 else ''}"
                    + ("" if r.passes == 1 else f", {r.passes} passes"), va="center", fontsize=5.8)
    for i, r in enumerate(g.itertuples()):
        if not pd.isna(r.gpu_memory_gb):            # the requested device budget, to show where it is exceeded
            axs[0].plot([r.gpu_memory_gb] * 2, [i - 0.42, i + 0.02], color=INK, lw=1.2)
    axs[0].plot([], [], color=INK, lw=1.2, label="GPU budget requested")
    axs[0].set_yticks(y, g.label, fontsize=6.5)
    axs[0].invert_yaxis()
    axs[0].set_xlabel("GiB")
    axs[0].set_xlim(0, max(g.peak_gpu_gb.max(), g.peak_rss_gb.max()) * 1.75)
    axs[0].set_title("(a) CuPy: memory budgets", loc="left")
    axs[0].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, handlelength=1.2)
    n = mb[mb.backend == "numpy"].copy()
    n["label"] = ["no limit" if pd.isna(r.cpu_memory_gb) else f"RAM {r.cpu_memory_gb:g} GiB" for r in n.itertuples()]
    y = np.arange(len(n))
    axs[1].barh(y, n.peak_rss_gb, 0.55, color=BCOL["numpy"])
    for i, r in enumerate(n.itertuples()):
        axs[1].text(r.peak_rss_gb + 0.6, i, f"{r.seconds:.1f} s, {r.blocks} block{'s' if r.blocks > 1 else ''}"
                    + ("" if r.passes == 1 else f", {r.passes} passes"), va="center", fontsize=5.8)
    axs[1].set_yticks(y, n.label, fontsize=6.5)
    axs[1].invert_yaxis()
    axs[1].set_xlabel("peak process RAM (GiB)")
    axs[1].set_xlim(0, n.peak_rss_gb.max() * 1.7)
    axs[1].set_title("(b) NumPy (CPU): RAM budgets", loc="left")
    same = (mb.max_abs_diff_vs_unlimited.fillna(0) == 0).all()
    fig.text(0.01, -0.07, "One 90.5 M point tile, after warm-up. " + (
        "Every budgeted run gave the same values as the unlimited run (max |Δ| = 0 over all metrics).\n"
        "GPU peak = largest CuPy memory pool, sampled every 10 ms." if same
        else "Some budgeted runs differ from the unlimited run, see memory_budget.csv."), fontsize=6.2, color=INK2)
    plt.tight_layout()
    save(fig, "fig6_memory")


# ------------------------------------------------------------------ tables
DEFS = {
    "zq20": "20th percentile of heights > 0.5 m (type 7)", "zcv": "coefficient of variation of heights (%)",
    "CRR": "canopy relief ratio (mean − min) / (max − min)", "zentropy": "Shannon entropy of 1 m layers, normalized",
    "zpcum7": "cumulative % of points in the lower 7 of 10 height slices", "zpcum9": "as zpcum7, 9 slices",
    "pz_above_30": "% of points above 30 m", "L4": "4th L-moment", "Lskew": "L-skewness", "Lkurt": "L-kurtosis",
    "lad_max": "max leaf area density (lidR::LAD, dz 1 m, k 0.5, z0 2 m)", "lad_mean": "mean LAD",
    "lad_cv": "coefficient of variation of LAD", "p_single": "% single returns", "p_last": "% last returns",
    "ratio_last_first": "last / first returns", "OpenGapSpace": "Lefsky open gap voxels (%)",
    "ClosedGapSpace": "Lefsky closed gap voxels (%)", "Euphotic": "Lefsky euphotic voxels (%)",
    "Oligophotic": "Lefsky oligophotic voxels (%)", "vzsd": "SD of normalized voxel height histogram",
    "vzcv": "CV of normalized voxel height histogram", "kde_peaks_count": "peaks of density(z, bw = 2)",
    "rumple": "3D / 2D area of the Delaunay surface of highest points per 1 m",
    "tree_n": "tree tops in the cell (lmf, ws 2.5 + 0.15 h, 3–8 m, hmin 5)", "tree_h_mean": "mean tree height",
    "tree_h_sd": "SD of tree heights", "tree_h_cv": "CV of tree heights", "tree_h_max": "max tree height",
    "tree_h_gini": "Gini coefficient of tree heights", "tree_h_shannon": "Shannon index of 2 m height classes",
    "crown_m2_mean": "mean crown area (dalponte2016)", "crown_m2_sd": "SD of crown area",
    "leafr_lai": "LAI = sum of LAD (MacArthur–Horn on 1 m return slices, k = 1)",
    "leafr_lahv": "leaf area height volume", "leafr_lahv_laiw": "LAHV weighted by LAI",
    "leafr_lahv_hw": "LAHV weighted by height", "leafr_lahv_laiw_hw": "LAHV weighted by LAI and height",
    "leafr_fhd": "foliage height diversity of the relative LAD profile", "leafr_fhd_even": "FHD evenness",
    "leafr_gs": "Gini–Simpson index of the relative LAD profile", "leafr_gs_even": "Gini–Simpson evenness",
    "leafr_gc": "Gini coefficient of heights ≥ 1 m", "pfs_pai": "plant area index above 1 m (PAD, k = 1)",
    "pfs_cover": "canopy cover above 2 m, 1 − exp(−0.5 PAI)", "pfs_fhd": "entropy of 1 m voxel return counts",
    "enl": "effective number of layers, 1 / Σ p², filled 1 m voxels", "gap_frac": "share of canopy pixels in gaps",
    "gap_n": "gaps touching the cell", "gap_area_mean": "mean full area of those gaps (m²)",
    "clz_chm": "max − min elevation per column", "clz_ground": "lowest point per column",
    "clz_vertical_sd_lad": "SD of LAD above ht.cut (rounded to 4 digits)",
    "clz_rugosity": "SD of vertical SD of LAD in a 3 × 3 window (within-canopy rugosity)",
    "clz_filled_ratio": "share of filled voxels above ht.cut", "clz_porosity_ratio": "share of empty voxels",
    "clz_max_lad": "max LAD in the column", "clz_max_lad_ht": "voxel height of max LAD",
    "clz_toc_rugosity": "top-of-canopy rugosity volume, 3 × 3 window"}
POINTS = {"lidRmetrics": "heights > 0.5 m (voxels: rounded z > 0.5; rumple: all points)",
          "lidR trees": "canopy raster 0.5 m, p2r(subcircle 0.15) + knnidw fill",
          "leafR": "all points of the cell, z < 0 set to 0", "PyForestScan": "points with z ≥ 0",
          "forestr": "filled 1 m voxels (lidRmetrics voxelization)", "ForestGapR": "canopy raster ≤ 10 m, 1–10⁴ m²",
          "canopyLazR": "raw elevations of the file; Tukey 1.5 IQR filter; own grid"}
VERS = {"lidRmetrics": "lidRmetrics 0.1.5", "lidR trees": "lidR 4.3.3", "leafR": "leafR 0.3.5",
        "PyForestScan": "PyForestScan 0.3.6", "forestr": "forestr 2.0.2", "ForestGapR": "ForestGapR 0.1.7",
        "canopyLazR": "canopyLazR 2.0"}


def write_table(df, name, title):
    df.to_csv(TAB / f"{name}.csv", index=False)
    md = [f"# {title}", "", "| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    md += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    (TAB / f"{name}.md").write_text("\n".join(md) + "\n")


def tables(res, cells, single, thr, mb):
    t1 = pd.DataFrame([{"metric": m, "package": VERS[package(m)], "definition": DEFS.get(m, ""),
                        "points used": POINTS[package(m)]} for m in LIDR + OTHER + CLZ])
    write_table(t1, "table1_metrics", "Table 1. Metrics reproduced by lidarmetricpy")
    import json
    clips = [c for c in json.loads((VAL / "data/clips.json").read_text()) if c["n"] > 0]
    full = json.loads((VAL / "data/full.json").read_text())
    rows = [{"dataset": f"clip {c['mesh']}", "extent": "250 × 250 m (10 × 10 cells)", "points": f"{c['n']:,}",
             "points per m²": c["pts_m2"], "use": "lidR / lidRmetrics, leafR, PyForestScan, forestr, ForestGapR"}
            for c in clips]
    rows += [{"dataset": f"full tile {c['mesh']}", "extent": "1000 × 750 m (40 × 30 cells)", "points": f"{c['n']:,}",
              "points per m²": round(c["n"] / 750000, 1), "use": "same + speed and memory"} for c in full]
    rows.append({"dataset": "raw clips (4)", "extent": "250 × 250 m, elevations", "points": "3.3–8.3 M",
                 "points per m²": "52–133", "use": "canopyLazR (10 m and 5 m grids)"})
    write_table(pd.DataFrame(rows), "table2_datasets",
                "Table 2. Validation data (Kanagawa 2024 ALS, Nakatsu River; heights normalized once and written as LAZ)")
    t3 = (res.assign(exact=lambda d: (d.equal_share == 1) & (d.nan_mismatch == 0))
          .groupby(["package", "backend"]).agg(metrics=("metric", "nunique"), exact=("exact", "sum"),
                                                cells=("n", "max"), max_rel_diff=("max_rel_diff", "max"))
          .reset_index())
    t3["package"] = pd.Categorical(t3.package, PKG_ORDER, ordered=True)
    t3["backend"] = pd.Categorical(t3.backend, BACKENDS, ordered=True)
    t3 = t3.sort_values(["package", "backend"])
    t3["equal"] = t3.exact.astype(int).astype(str) + " / " + t3.metrics.astype(str)
    t3["max_rel_diff"] = t3.max_rel_diff.map(lambda v: f"{v:.1e}")
    write_table(t3[["package", "backend", "equal", "cells", "max_rel_diff"]], "table3_agreement",
                "Table 3. Agreement with the reference packages (all clip and full-tile cells; equal = every cell within "
                "1e-6 relative and same NA pattern)")
    ld = lidr_times()
    s = single[~single.other_packages.astype(bool)]
    rows = []
    for b in ["cupy", "torch-cuda", "torch-cuda32", "numpy", "torch-cpu"]:
        r1 = s[(s.backend == b) & (s.mesh == "09LC9423")].iloc[-1]
        r2 = s[(s.backend == b) & (s.mesh == "09MC0523")].iloc[-1]
        t = thr[thr.backend == b].sort_values("workers")
        rows.append({"backend": b, "one tile, 90.5 M points (s)": r1.seconds, "one tile, 48.3 M points (s)": r2.seconds,
                     "peak GPU (GiB)": r1.peak_device_gb if r1.peak_device_gb > 0 else "–", "peak RAM (GiB)": r1.peak_rss_gb,
                     "throughput (s per tile, workers)": "; ".join(f"{v:.2f} ({int(w)})" for w, v in zip(t.workers, t.s_per_tile))})
    txt = (VAL / "par" / "lidr_par.log").read_text()
    lsec = float(re.search(r"\): ([\d.]+) s", txt).group(1)) / 8
    rows.append({"backend": "lidR + lidRmetrics (R)", "one tile, 90.5 M points (s)": ld["09LC9423"]["seconds"],
                 "one tile, 48.3 M points (s)": ld["09MC0523"]["seconds"], "peak GPU (GiB)": "–",
                 "peak RAM (GiB)": round(ld["09LC9423"]["rss"], 1), "throughput (s per tile, workers)": f"{lsec:.2f} (8)"})
    write_table(pd.DataFrame(rows), "table4_speed",
                "Table 4. Speed and memory (measured; RTX PRO 6000 Blackwell Max-Q 96 GB, Threadripper PRO 7955WX 16 cores)")
    t5 = pd.DataFrame([
        ("lidR round_any uses C round() (halves away from zero)", "voxel indices, canopy raster extent", "C round emulated"),
        ("terra closes the right / bottom outer edge of a raster: points on it belong to the last column / row", "cell membership", "closed_edges=True (default)"),
        ("highest(1) snaps each cell's bbox with st_adjust_bbox (C round): a negative-integer max x (positive-integer min y) lies on the layout edge and goes to the last column (row)", "rumple", "per-cell layout edge rule"),
        ("R round(x, 3) (fround.c) differs from numpy.round in ~6 % of values", "canopy raster heights", "fround port"),
        ("seq() clips the last break with pmin(x, to)", "zpcum", "breaks clipped"),
        ("findInterval drops a point exactly at an integer zmax", "zentropy", "same rule"),
        ("lidRmetrics rounds z before z > 0.5 (z = 0.5 becomes a 1 m voxel)", "voxel metrics", "voxels from all points"),
        ("Lefsky class with no voxel is NA (left join), not 0", "Lefsky classes", "NA rule"),
        ("stats::density: linear binning (BinDist), FFT, approx", "kde_peaks_count", "ported"),
        ("lmf: an equal height already tagged as maximum wins (raster order)", "tree tops", "sequential tie pass"),
        ("dalponte2016 updates crown means within a pass; later regions overwrite", "crown area", "numba port"),
        ("knnidw ties: GridPartition visit order + libstdc++ introsort", "filled canopy pixels", "ported, checked vs g++"),
        ("CuPy scatter_max / ufunc.at use float32 atomics for float64", "all max / min", "int64 order-preserving atomics"),
        ("PyTorch CUDA divides by a Python scalar via its reciprocal", "bin edges", "tensor divisor"),
        ("fused multiply-add in X·scale + offset", "coordinates", "separate kernels"),
        ("leafR: one NA layer makes LAHV / FHD NA; < 2 layers -> error", "leafR metrics", "same NA rules"),
        ("PyForestScan: zmax on an integer edge falls in the last closed bin", "pfs metrics", "same binning"),
        ("canopyLazR: file-wide Tukey filter, round() half-to-even bins, factor NA drops", "canopyLazR rasters", "ported"),
    ], columns=["detail of the reference implementation", "affects", "how lidarmetricpy matches it"])
    write_table(t5, "table5_reproduction_details", "Table 5. Details needed for exact agreement")


def main():
    cells = pd.read_parquet(DAT / "cells_vs_reference.parquet")
    clz = pd.read_parquet(DAT / "canopylazr_vs_reference.parquet")
    single = pd.read_csv(VAL / "backend_compare/single.csv")
    thr = pd.read_csv(VAL / "backend_compare/throughput.csv")
    mb = pd.read_csv(DAT / "memory_budget.csv")
    fig_workflow()
    fig_scatter(cells, clz)
    res = fig_heatmap(cells, clz)
    res.to_csv(TAB / "agreement_per_metric_backend.csv", index=False)
    fig_single(single)
    fig_throughput(thr)
    fig_memory(mb)
    tables(res, cells, single, thr, mb)
    print("done")


if __name__ == "__main__":
    main()
