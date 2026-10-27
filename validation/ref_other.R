# Reference values from other R packages, per 25 m cell of a normalized clip (Z = height above ground).
#   Rscript ref_other.R <laz> <x0> <y1> <ncol> <nrow> <out_prefix>
# Writes <out_prefix>_other.csv with, per cell:
#   leafR 0.3.5   : lad.voxels(cell file, grain.size = 26 (one pixel; LAD uses count ratios only), k = 1) -> lad.profile -> lai, LAHV (4 variants),
#                   FHD / GS (relative profile, plain and evenness), GC(threshold = 1)
#   forestr 2.0.2 : calc_enl on the filled 1 m voxels of lidRmetrics:::create_voxels(zmin = 0.5)
#   ForestGapR    : getForestGaps(lidR canopy raster, threshold = 10, size = c(1, 1e4)), summarised per cell
#                   (share of canopy-raster pixels in gaps, number of gaps touching the cell, their mean area)
# leafR reads a file, so every cell is written to its own LAS file (one leafR pixel per cell).
suppressPackageStartupMessages({ library(lidR); library(leafR); library(forestr); library(ForestGapR)
  library(lidRmetrics); library(terra); library(data.table) })
a <- commandArgs(trailingOnly = TRUE)
laz <- a[1]; x0 <- as.numeric(a[2]); y1 <- as.numeric(a[3]); nc <- as.integer(a[4]); nr <- as.integer(a[5]); out <- a[6]
CELL <- 25
set_lidr_threads(1); options(lidR.progress = FALSE)
las <- readLAS(laz, select = "xyzirn")
col <- floor((las$X - x0) / CELL); row <- floor((y1 - las$Y) / CELL)
# closed outer edges as in lidR / terra: points on the right / bottom edge belong to the last column / row
col[las$X == x0 + nc * CELL] <- nc - 1; row[las$Y == y1 - nr * CELL] <- nr - 1
tmp <- file.path(tempdir(), "cells"); dir.create(tmp, showWarnings = FALSE)
# points grouped by cell once (stable order, so each cell keeps the file order of its points)
key <- row * 100000 + col
ord <- order(key)
idx_by_cell <- split(ord, key[ord])
res <- list()
for (r in 0:(nr - 1)) for (c in 0:(nc - 1)) {
  sel <- idx_by_cell[[as.character(r * 100000 + c)]]
  if (is.null(sel) || length(sel) == 0) next
  sub <- LAS(las@data[sel], las@header, crs = st_crs(las), check = FALSE)
  f <- file.path(tmp, sprintf("c_%d_%d.las", r, c))
  writeLAS(sub, f)
  v <- tryCatch({
    vox <- lad.voxels(f, grain.size = CELL + 1, k = 1)   # one leafR pixel per cell, also when it spans exactly 25 m
    p <- lad.profile(vox)
    pr <- lad.profile(vox, relative = TRUE)
    list(leafr_lai = lai(p), leafr_lahv = LAHV(p), leafr_lahv_laiw = LAHV(p, LAI.weighting = TRUE),
         leafr_lahv_hw = LAHV(p, height.weighting = TRUE),
         leafr_lahv_laiw_hw = LAHV(p, LAI.weighting = TRUE, height.weighting = TRUE),
         leafr_fhd = FHD(pr), leafr_fhd_even = FHD(pr, evenness = TRUE),
         leafr_gs = GS(pr), leafr_gs_even = GS(pr, evenness = TRUE))
  }, error = function(e) list(leafr_error = conditionMessage(e)))
  gc_ <- tryCatch(GC(f, threshold = 1), error = function(e) NA_real_)
  vx <- lidRmetrics:::create_voxels(sub$X, sub$Y, sub$Z, vox_size = 1, zmin = 0.5)
  enl <- NA_real_
  if (!is.null(vx)) {
    m <- data.frame(zbin = vx$Z, vai = ifelse(is.na(vx$n), 0, 1))
    enl <- suppressMessages(capture.output(e <- calc_enl(m)))
    enl <- e
  }
  res[[length(res) + 1]] <- c(list(x = x0 + (c + 0.5) * CELL, y = y1 - (r + 0.5) * CELL, leafr_gc = gc_, enl = enl), v)
  unlink(f)
}
dt <- rbindlist(res, fill = TRUE)

# ForestGapR on the same canopy raster as ref_lidr.R
chm <- rasterize_canopy(las, res = 0.5, algorithm = p2r(subcircle = 0.15, na.fill = knnidw()))
gaps <- getForestGaps(raster::raster(chm), threshold = 10, size = c(1, 10^4))
g <- as.data.table(as.data.frame(rast(gaps), xy = TRUE, na.rm = FALSE))
setnames(g, c("px", "py", "gap"))
ch <- as.data.table(as.data.frame(chm, xy = TRUE, na.rm = FALSE))
g[, c := floor((px - x0) / CELL)][, r := floor((y1 - py) / CELL)]
area <- g[!is.na(gap), .(a = .N * 0.25), by = gap]        # full gap area (GapStats), before clipping to the grid
g <- g[c >= 0 & c < nc & r >= 0 & r < nr]
gs <- g[, .(gap_frac = mean(!is.na(gap)), gap_n = uniqueN(gap[!is.na(gap)]),
            gap_area_mean = { u <- unique(gap[!is.na(gap)]); if (length(u)) mean(area[match(u, area$gap), a]) else NA_real_ }),
        by = .(c, r)]
gs[, `:=`(x = x0 + (c + 0.5) * CELL, y = y1 - (r + 0.5) * CELL)][, c("c", "r") := NULL]
dt <- merge(dt, gs, by = c("x", "y"), all = TRUE)
fwrite(dt, paste0(out, "_other.csv"))
cat("cells", nrow(dt), "\n")
