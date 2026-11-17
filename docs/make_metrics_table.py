"""docs/tables/table1_metrics.{csv,md}: every output of lidarmetricpy, one row each.

The 137 per-cell metrics of compute_tile (column order of lidarmetricpy.api.output_columns) and the 9 rasters
of canopylazr_file, with a short meaning, the package that defines them, the definition and the points used.
The same list with the short meanings is written into README.md between the "metrics table" markers. Settings
are those of the package defaults (the lidR workflow of the README). Run from the repository root:

    python docs/make_metrics_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lidarmetricpy import Options  # noqa: E402
from lidarmetricpy.api import output_columns  # noqa: E402

TAB = ROOT / "docs/tables"
CLZ = ["clz_chm", "clz_ground", "clz_vertical_sd_lad", "clz_rugosity", "clz_filled_ratio", "clz_porosity_ratio",
       "clz_max_lad", "clz_max_lad_ht", "clz_toc_rugosity"]

Z = "heights > 0.5 m"
RET = "returns with heights > 0.5 m and 0 < ReturnNumber ≤ NumberOfReturns"
VOX = "1 m voxels of points with rounded height > 0.5 m (x, y, z rounded half away from zero)"
CHM = "0.5 m canopy raster: highest return per pixel, returns as discs of 0.15 m, empty pixels filled by knnidw"
LEAFR = "all points of the cell, heights < 0 set to 0, 1 m layers (as.integer(z))"
PFS = "points with height ≥ 0, one column of 1 m voxels per cell"
CLZP = "elevations of all points of the file after Tukey's 1.5 IQR filter on z; canopyLazR's own grid"

# (function or step, definition, points used)
F = {"basic": "lidRmetrics::metrics_basic", "perc": "lidRmetrics::metrics_percentiles",
     "above": "lidRmetrics::metrics_percabove", "disp": "lidRmetrics::metrics_dispersion",
     "cd": "lidRmetrics::metrics_canopydensity", "lmom": "lidRmetrics::metrics_Lmoments",
     "lad": "lidRmetrics::metrics_lad", "int": "lidRmetrics::metrics_interval",
     "echo": "lidRmetrics::metrics_echo", "echo2": "lidRmetrics::metrics_echo2",
     "rumple": "lidRmetrics::metrics_rumple", "vox": "lidRmetrics::metrics_voxels",
     "kde": "lidRmetrics::metrics_kde", "home": "lidRmetrics::metrics_HOME"}
LADDEF = "of the LAD profile of lidR::LAD(z, dz = 1, k = 0.5, z0 = 2) (gap fractions of 1 m layers from 2 m); NA with ≤ 2 points"


def lidrmetrics_rows():
    d = {
        "n": (F["basic"], "number of points", Z),
        "zmax": (F["basic"], "maximum height (m)", Z),
        "zmin": (F["basic"], "minimum height (m)", Z),
        "zmean": (F["basic"], "mean height (m)", Z),
        "zvar": (F["basic"], "variance of heights (n − 1)", Z),
        "zsd": (F["basic"], "standard deviation of heights (n − 1)", Z),
        "zcv": (F["basic"], "coefficient of variation of heights, zsd / zmean × 100", Z),
        "zskew": (F["basic"], "skewness, m3 / m2^1.5 (moments with 1 / n)", Z),
        "zkurt": (F["basic"], "kurtosis, n Σ(z − mean)⁴ / (Σ(z − mean)²)² (not excess)", Z),
    }
    for p in [1] + list(range(5, 100, 5)) + [99]:
        d[f"zq{p}"] = (F["perc"], f"{p}th percentile of heights (R quantile type 7)", Z)
    d["pzabovemean"] = (F["above"], "% of points above the mean height", Z)
    for t in (2, 5, 10, 20, 30):
        d[f"pzabove{t}"] = (F["above"], f"% of points above {t} m", Z)
    d.update({
        "ziqr": (F["disp"], "interquartile range of heights (type 7)", Z),
        "zMADmean": (F["disp"], "mean absolute deviation from the mean height", Z),
        "zMADmedian": (F["disp"], "median absolute deviation from the median height (unscaled)", Z),
        "CRR": (F["disp"], "canopy relief ratio, (mean − min) / (max − min)", Z),
        "zentropy": (F["disp"], "normalised Shannon entropy of the counts in 1 m layers from 0 m (lidR::entropy)", Z),
    })
    for k in range(1, 10):
        d[f"zpcum{k}"] = (F["cd"], f"cumulative % of points in the lowest {k} of 10 equal height slices "
                          "between zmin and zmax (points at zmin excluded)", Z)
    d.update({
        "L1": (F["lmom"], "1st L-moment (mean)", Z),
        "L2": (F["lmom"], "2nd L-moment (L-scale)", Z),
        "L3": (F["lmom"], "3rd L-moment", Z),
        "L4": (F["lmom"], "4th L-moment", Z),
        "Lskew": (F["lmom"], "L-skewness, L3 / L2 (NA when all heights are equal; R gives rounding noise)", Z),
        "Lkurt": (F["lmom"], "L-kurtosis, L4 / L2 (NA when all heights are equal; R gives rounding noise)", Z),
        "Lcoefvar": (F["lmom"], "L-coefficient of variation, L2 / L1", Z),
        "lad_min": (F["lad"], "minimum " + LADDEF, Z),
        "lad_max": (F["lad"], "maximum " + LADDEF, Z),
        "lad_mean": (F["lad"], "mean " + LADDEF, Z),
        "lad_cv": (F["lad"], "coefficient of variation (sd / mean, not %) " + LADDEF, Z),
        "lad_sum": (F["lad"], "sum " + LADDEF + " (0 when the profile is empty)", Z),
        "pz_below_0": (F["int"], "% of points below 0 m (always 0 with zmin = 0.5)", Z),
    })
    for a, b in [("0", "0.5"), ("0.5", "2"), ("2", "5"), ("5", "10"), ("10", "20"), ("20", "30")]:
        d[f"pz_{a}-{b}"] = (F["int"], f"% of points with {a} ≤ height < {b} m (hist(), right = FALSE)", Z)
    d["pz_above_30"] = (F["int"], "% of points at or above 30 m", Z)
    d.update({
        "n_first": (F["echo"], "number of first returns", RET),
        "n_intermediate": (F["echo"], "number of intermediate returns (1 < ReturnNumber < NumberOfReturns)", RET),
        "n_last": (F["echo"], "number of last returns of multiple-return pulses (ReturnNumber = NumberOfReturns > 1)",
                   RET),
        "n_single": (F["echo"], "number of returns of single-return pulses", RET),
        "n_multiple": (F["echo"], "number of returns of multiple-return pulses", RET),
        "p_first": (F["echo"], "% first returns", RET),
        "p_intermidiate": (F["echo"], "% intermediate returns (spelling as in lidRmetrics)", RET),
        "p_last": (F["echo"], "% last returns (as n_last)", RET),
        "p_single": (F["echo"], "% returns of single-return pulses", RET),
        "p_multiple": (F["echo"], "% returns of multiple-return pulses", RET),
        "ratio_last_first": (F["echo"], "n_last / n_first", RET),
        "ratio_intermediate_first": (F["echo"], "n_intermediate / n_first", RET),
        "ratio_multiple_single": (F["echo"], "n_multiple / n_single", RET),
    })
    for k in range(1, 5):
        nth = {1: "smallest", 2: "2nd smallest", 3: "3rd smallest", 4: "4th smallest"}[k]
        same = "= count of ReturnNumber 1 when first returns occur" if k == 1 else f"= count of ReturnNumber {k} when 1–{k} all occur"
        d[f"n_return_{k}"] = (F["echo2"], f"number of returns with the {nth} return number present in the cell "
                              f"({same})",
                              "returns with heights > 0.5 m")
    d["rumple"] = (F["rumple"], "area of the Delaunay surface of the highest point per 1 m pixel / its projected "
                   "area (lidR::rumple_index)", "all points of the cell (no height limit)")
    d.update({
        "vn": (F["vox"], "number of filled voxels", VOX),
        "vFRall": (F["vox"], "% filled voxels in the bounding box of the filled voxels", VOX),
        "vFRcanopy": (F["vox"], "% filled voxels among voxels at or below the highest filled voxel of their column",
                      VOX),
        "vzrumple": (F["vox"], "Euclidean distance between the min–max-normalised histogram of filled-voxel "
                     "heights (1 m bins) and its bin starts / (number of bins × 1 m)",
                     VOX),
        "vzsd": (F["vox"], "standard deviation of that normalised histogram", VOX),
        "vzcv": (F["vox"], "coefficient of variation (%) of that normalised histogram", VOX),
        "OpenGapSpace": (F["vox"], "% of the voxels of columns with a filled voxel that are empty and above the "
                         "column's highest filled voxel (Lefsky et al. classes)", VOX),
        "ClosedGapSpace": (F["vox"], "% of those voxels that are empty and below the column's highest filled voxel", VOX),
        "Euphotic": (F["vox"], "% of those voxels that are filled and at or above 0.65 × the column's highest filled voxel", VOX),
        "Oligophotic": (F["vox"], "% of those voxels that are filled and below 0.65 × the column's highest filled voxel", VOX),
        "kde_peaks_count": (F["kde"], "number of peaks of stats::density(z, bw = 2); NA with ≤ 2 points", Z),
    })
    for k in range(1, 5):
        d[f"kde_peak{k}_elev"] = (F["kde"], f"height of peak {k}: of the 4 highest-density peaks, ordered from the "
                                  "highest elevation down (m)", Z)
    for k in range(1, 5):
        d[f"kde_peak{k}_value"] = (F["kde"], f"density at peak {k}", Z)
    for k in range(1, 4):
        d[f"kde_peak{k}_diff"] = (F["kde"], f"height difference between peaks {k} and {k + 1} (m)", Z)
    d["HOME"] = (F["home"], "height of median energy: height of the point where the intensity summed from below "
                 "and from above are closest (intensity 1 for input without intensities)", Z)
    return {k: ("lidRmetrics 0.1.5", *v) for k, v in d.items()}


def other_rows():
    lad = "leafR LAD profile (lad.voxels, k = 1, 1 m layers; lad.profile)"
    d = {
        "enl": ("forestr 2.0.2", "forestr::calc_enl", "effective number of layers, 1 / Σ p², p = share of the "
                "filled voxels in each 1 m layer", VOX),
        "leafr_lai": ("leafR 0.3.5", "leafR::lai(min = 1, max = 100)", f"leaf area index, sum of the {lad}", LEAFR),
        "leafr_lahv": ("leafR 0.3.5", "leafR::LAHV", "leaf area height volume, Σ layer height × LAD", LEAFR),
        "leafr_lahv_laiw": ("leafR 0.3.5", "leafR::LAHV(LAI.weighted = TRUE)", "LAHV / LAI", LEAFR),
        "leafr_lahv_hw": ("leafR 0.3.5", "leafR::LAHV(height.weighted = TRUE)", "LAHV / maximum layer height",
                          LEAFR),
        "leafr_lahv_laiw_hw": ("leafR 0.3.5", "leafR::LAHV(LAI.weighted = TRUE, height.weighted = TRUE)",
                               "LAHV / LAI / maximum layer height", LEAFR),
        "leafr_fhd": ("leafR 0.3.5", "leafR::FHD", "foliage height diversity, Shannon index of the relative LAD "
                      "profile (layers with share ≥ 1 / number of layers)", LEAFR),
        "leafr_fhd_even": ("leafR 0.3.5", "leafR::FHD(evenness = TRUE)", "FHD / ln(number of layers kept)", LEAFR),
        "leafr_gs": ("leafR 0.3.5", "leafR::GS", "Gini–Simpson index, 1 − Σ p², of the relative LAD profile", LEAFR),
        "leafr_gs_even": ("leafR 0.3.5", "leafR::GS(evenness = TRUE)", "GS / (1 − 1 / number of layers kept)",
                          LEAFR),
        "leafr_gc": ("leafR 0.3.5", "leafR::GC(threshold = 1)", "Gini coefficient of the heights ≥ 1 m",
                     "points with height ≥ 1 m"),
        "pfs_pai": ("PyForestScan 0.3.6", "calculate_pad → calculate_pai(min_height = 1)", "plant area index, "
                    "sum of the plant area density (Beer–Lambert, k = 1) of the 1 m layers from 1 m", PFS),
        "pfs_cover": ("PyForestScan 0.3.6", "calculate_canopy_cover(min_height = 2, k = 0.5)",
                      "canopy cover, 1 − exp(−0.5 × PAI above 2 m)", PFS),
        "pfs_fhd": ("PyForestScan 0.3.6", "calculate_fhd", "foliage height diversity, Shannon entropy of the "
                    "return counts of the 1 m layers", PFS),
    }
    tree = ("lidR 4.3.3 (tree tops, crowns); per-cell summary: lidarmetricpy", "rasterize_canopy → locate_trees(lmf) → dalponte2016")
    tops = ("tree tops (lmf: window 2.5 + 0.15 h m, 3–8 m; tops ≥ 5 m) on the " + CHM +
            "; a tree belongs to the cell that contains its top")
    crowns = "crowns of dalponte2016(th_tree = 2) grown from those tops on the same raster (pixels × 0.25 m²)"
    d.update({
        "tree_n": (*tree, "number of tree tops", tops),
        "tree_h_mean": (*tree, "mean tree height (m)", tops),
        "tree_h_sd": (*tree, "standard deviation of tree heights (n − 1)", tops),
        "tree_h_cv": (*tree, "tree_h_sd / tree_h_mean (not %)", tops),
        "tree_h_max": (*tree, "maximum tree height (m)", tops),
        "tree_h_gini": (*tree, "Gini coefficient of tree heights (NA with fewer than 2 trees)", tops),
        "tree_h_shannon": (*tree, "Shannon index of tree heights in 2 m classes", tops),
        "crown_m2_mean": (*tree, "mean crown area (m²)", crowns),
        "crown_m2_sd": (*tree, "standard deviation of crown area (m², n − 1)", crowns),
    })
    gap = ("ForestGapR 0.1.7 (gaps); per-cell summary: lidarmetricpy", "getForestGaps(chm, threshold = 10, size = c(1, 10^4))")
    gp = "gaps: connected areas of the canopy raster ≤ 10 m, 1 m² to 1 ha, found over the whole tile; " + CHM
    d.update({
        "gap_frac": (*gap, "share of the cell's canopy-raster pixels that lie in a gap", gp),
        "gap_n": (*gap, "number of gaps that touch the cell", gp),
        "gap_area_mean": (*gap, "mean whole area of the gaps that touch the cell (m²)", gp),
    })
    return d


def clz_rows():
    fn = "canopyLazR workflow (laz.to.array → canopy.height.levelr → machorn.lad); lidarmetricpy.canopylazr_file"
    lad = "LAD of MacArthur–Horn on the voxel columns (z_res 1 m), above ht_cut (5 m)"
    d = {
        "clz_chm": "canopy height, highest − lowest elevation of the column (m)",
        "clz_ground": "lowest elevation of the column (m), canopyLazR's ground",
        "clz_vertical_sd_lad": f"standard deviation of the {lad} (rounded to 4 digits)",
        "clz_rugosity": "within-canopy rugosity, standard deviation of clz_vertical_sd_lad in a 3 × 3 window",
        "clz_filled_ratio": f"share of voxels with LAD > 0 among those with a value, {lad}",
        "clz_porosity_ratio": f"share of voxels with LAD = 0 among those with a value, {lad}",
        "clz_max_lad": f"maximum {lad}",
        "clz_max_lad_ht": "voxel height (level) of clz_max_lad",
        "clz_toc_rugosity": "top-of-canopy rugosity, Σ (window maximum − clz_chm) × pixel area × z_res in a 3 × 3 "
                            "window",
    }
    return {k: ("canopyLazR 2.0", fn, v, CLZP) for k, v in d.items()}


MEANING = {
    "n": "how many returns lie above 0.5 m (vegetation returns)",
    "zmax": "top height of the vegetation", "zmin": "lowest vegetation return above 0.5 m",
    "zmean": "average height of the vegetation returns", "zvar": "spread of return heights (variance)",
    "zsd": "spread of return heights", "zcv": "spread of return heights relative to their mean",
    "zskew": "whether returns are concentrated low (positive) or high (negative) in the canopy",
    "zkurt": "how peaked the height distribution is (3 = normal)",
    "pzabovemean": "share of returns above the average height",
    "ziqr": "height range of the middle half of the returns",
    "zMADmean": "average distance of return heights from their mean",
    "zMADmedian": "typical distance of return heights from their median (robust spread)",
    "CRR": "where the mean height lies between lowest and highest return (0 = low, 1 = top); canopy shape",
    "zentropy": "how evenly returns spread over 1 m layers (0 = one layer, 1 = all layers equal)",
    "L1": "mean height (robust moment)", "L2": "spread of heights (robust moment)",
    "L3": "asymmetry of heights (robust moment, unscaled)", "L4": "peakedness of heights (robust moment, unscaled)",
    "Lskew": "asymmetry of the height distribution (robust)", "Lkurt": "peakedness of the height distribution (robust)",
    "Lcoefvar": "spread of heights relative to the mean (robust)",
    "lad_min": "leaf area density of the sparsest 1 m layer", "lad_max": "leaf area density of the densest 1 m layer",
    "lad_mean": "average leaf area density over the layers", "lad_cv": "how unequal leaf area is among layers",
    "lad_sum": "total leaf area over the layers (an LAI-like value)",
    "pz_below_0": "share of returns below ground (0 here because only returns above 0.5 m are used)",
    "pz_above_30": "share of returns in the very tall canopy (30 m and above)",
    "n_first": "number of first returns (pulses that hit something)",
    "n_intermediate": "number of returns between first and last of a pulse (passed through the canopy)",
    "n_last": "number of last returns of pulses with several returns",
    "n_single": "number of returns from pulses with one return (solid surfaces, dense crowns)",
    "n_multiple": "number of returns from pulses with several returns (penetrable canopy)",
    "p_first": "share of first returns", "p_intermidiate": "share of intermediate returns",
    "p_last": "share of last returns", "p_single": "share of single-return pulses' returns",
    "p_multiple": "share of multiple-return pulses' returns",
    "ratio_last_first": "last per first returns; higher where pulses pass through the canopy",
    "ratio_intermediate_first": "intermediate per first returns; higher in deep, layered canopies",
    "ratio_multiple_single": "multiple- per single-return returns; canopy penetrability",
    "rumple": "roughness of the canopy surface (1 = flat; larger = more uneven)",
    "vn": "number of occupied 1 m cubes (volume filled by vegetation)",
    "vFRall": "share of the occupied cubes in the box around the vegetation",
    "vFRcanopy": "share of space below the canopy top that is occupied",
    "vzrumple": "complexity of the vertical profile of occupied cubes",
    "vzsd": "unevenness of the vertical profile of occupied cubes",
    "vzcv": "unevenness of the vertical profile of occupied cubes, relative",
    "OpenGapSpace": "empty space above the canopy surface (between crowns)",
    "ClosedGapSpace": "empty space inside or below the canopy",
    "Euphotic": "occupied space in the upper, sunlit part of the canopy",
    "Oligophotic": "occupied space in the lower, shaded part of the canopy",
    "kde_peaks_count": "number of vegetation layers (peaks of the height distribution)",
    "HOME": "height that splits the returned energy in half; higher for top-heavy canopies",
    "enl": "effective number of vegetation layers",
    "leafr_lai": "leaf area index (leaf area per ground area)",
    "leafr_lahv": "leaf area weighted by height (amount and height of foliage together)",
    "leafr_lahv_laiw": "average height of the leaf area", "leafr_lahv_hw": "LAHV relative to stand height",
    "leafr_lahv_laiw_hw": "average height of the leaf area relative to stand height",
    "leafr_fhd": "foliage height diversity: how evenly leaves spread over layers",
    "leafr_fhd_even": "foliage height diversity scaled to 0–1",
    "leafr_gs": "diversity of the leaf profile (Gini–Simpson)", "leafr_gs_even": "Gini–Simpson diversity scaled to 0–1",
    "leafr_gc": "inequality of return heights above 1 m (0 = all equal)",
    "pfs_pai": "plant area index (leaves and wood) above 1 m", "pfs_cover": "canopy cover above 2 m (0–1)",
    "pfs_fhd": "foliage height diversity of the return profile",
    "tree_n": "number of trees (tree tops ≥ 5 m)", "tree_h_mean": "average tree height",
    "tree_h_sd": "variation of tree heights", "tree_h_cv": "variation of tree heights relative to their mean",
    "tree_h_max": "tallest tree", "tree_h_gini": "inequality of tree heights (0 = all equal)",
    "tree_h_shannon": "diversity of tree heights", "crown_m2_mean": "average crown size (m²)",
    "crown_m2_sd": "variation of crown size",
    "gap_frac": "share of the cell in canopy gaps (canopy ≤ 10 m)", "gap_n": "number of canopy gaps",
    "gap_area_mean": "average size of those gaps (m²)",
    "clz_chm": "canopy height (from the raw elevations)", "clz_ground": "ground elevation",
    "clz_vertical_sd_lad": "how unevenly leaves are spread with height in the column",
    "clz_rugosity": "within-canopy roughness: how much the vertical leaf structure changes between neighbours",
    "clz_filled_ratio": "share of canopy space filled with leaves",
    "clz_porosity_ratio": "share of canopy space that is empty (porosity)",
    "clz_max_lad": "leaf density of the densest layer", "clz_max_lad_ht": "height of the densest leaf layer",
    "clz_toc_rugosity": "roughness of the canopy top: how far the surface lies below its highest neighbour",
}


def meaning(m):
    if m in MEANING:
        return MEANING[m]
    if m.startswith("zq"):
        return f"height below which {m[2:]} % of the returns lie" + (" (≈ top height)" if m in ("zq95", "zq99")
                                                                      else " (median height)" if m == "zq50" else "")
    if m.startswith("pzabove"):
        return f"share of returns above {m[7:]} m (canopy cover above {m[7:]} m)"
    if m.startswith("zpcum"):
        return f"share of returns in the lower {m[5:]}0 % of the height range (profile shape)"
    if m.startswith("pz_"):
        a, b = m[3:].split("-")
        return f"share of returns between {a} and {b} m"
    if m.startswith("n_return_"):
        return f"number of returns with return number {m[-1]}"
    k = m[8]
    if m.endswith("_elev"):
        return f"height of vegetation layer {k}, counted from the top"
    if m.endswith("_value"):
        return f"how strong vegetation layer {k} is"
    if m.endswith("_diff"):
        return f"vertical distance between layers {k} and {int(k) + 1}"
    raise KeyError(m)


README = ROOT / "README.md"
BEGIN, END = "<!-- metrics table: begin (docs/make_metrics_table.py) -->", "<!-- metrics table: end -->"


def readme_block(df):
    md = [BEGIN, "<details>", "<summary>All 146 outputs with a short meaning (click to open; definitions in "
          "<a href=\"docs/tables/table1_metrics.md\">docs/tables/table1_metrics.md</a>)</summary>", "",
          "| no | metric | meaning | package |", "|---|---|---|---|"]
    md += [f"| {r.no} | {r.metric} | {r.meaning} | {r.package} |" for r in df.itertuples()]
    md += ["", "</details>", END]
    return "\n".join(md)


def main():
    cols = [c for c in output_columns(Options()) if c not in ("x", "y", "n_points")]
    rows = {**lidrmetrics_rows(), **other_rows()}
    missing, extra = set(cols) - set(rows), set(rows) - set(cols)
    assert not missing and not extra, (missing, extra)
    clz = clz_rows()
    assert list(clz) == CLZ
    out = []
    for i, c in enumerate(cols + CLZ, 1):
        pkg, fn, d, pts = (rows | clz)[c]
        out.append({"no": i, "metric": c, "meaning": meaning(c),
                    "output": "per cell" if c in rows else "raster (canopylazr_file)",
                    "package": pkg, "function": fn, "definition": d, "points used": pts})
    df = pd.DataFrame(out)
    assert len(df) == 146 and df.metric.is_unique
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / "table1_metrics.csv", index=False)
    md = ["# Table 1. Outputs of lidarmetricpy", "",
          "The 137 per-cell metrics of `compute_tile` / `compute_tiles` (in the column order of "
          "`lidarmetricpy.api.output_columns(Options())`, after `x`, `y` and `n_points`, the number of points in "
          "the cell) and the 9 rasters of `canopylazr_file`, with default settings. Heights are above the terrain "
          "model. Each value equals the named package applied to the same points (docs/tables/"
          "table3_agreement.md, validation/). Made by docs/make_metrics_table.py.", "",
          "| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    md += ["| " + " | ".join(str(v).replace("|", "\\|") for v in r) + " |" for r in df.itertuples(index=False)]
    (TAB / "table1_metrics.md").write_text("\n".join(md) + "\n")
    text = README.read_text()
    i, j = text.index(BEGIN), text.index(END) + len(END)
    README.write_text(text[:i] + readme_block(df) + text[j:])
    print(df.package.value_counts().to_string())


if __name__ == "__main__":
    main()
