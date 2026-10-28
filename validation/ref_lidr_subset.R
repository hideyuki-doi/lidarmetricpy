# Timing variant of ref_lidr.R: computes only what the 24 lidarmetricpy metrics need (same values).
#   Rscript ref_lidr_subset.R <laz> <x0> <y1> <ncol> <nrow> <out_prefix>
# Writes <out_prefix>_cells.csv (lidRmetrics::metrics_set3 + tree summaries per 25 m cell)
#    and <out_prefix>_trees.csv (tree tops with crown area).
# Single thread so that lidR's tie handling in lmf is deterministic.
suppressPackageStartupMessages({ library(lidR); library(lidRmetrics); library(terra); library(data.table) })
a <- commandArgs(trailingOnly = TRUE)
laz <- a[1]; x0 <- as.numeric(a[2]); y1 <- as.numeric(a[3]); nc <- as.integer(a[4]); nr <- as.integer(a[5]); out <- a[6]
CELL <- 25
set_lidr_threads(1); setDTthreads(1); options(lidR.progress = FALSE)
t0 <- Sys.time()
las <- readLAS(laz, select = "xyzirn")
t_read <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
tmpl <- rast(xmin = x0, xmax = x0 + nc * CELL, ymin = y1 - nr * CELL, ymax = y1, resolution = CELL, crs = st_crs(las)$wkt)

# Only the lidRmetrics functions that produce the 24 metrics of lidarmetricpy (metrics_set3 without
# metrics_percabove, metrics_echo2 and metrics_HOME), with the same arguments as in metrics_set3.
m3 <- function(x, y, z, i, rn, nr) {
  zi <- c(0, 0.5, 2, 5, 10, 20, 30)
  c(metrics_basic(z = z, zmin = 0.5), metrics_percentiles(z = z, zmin = 0.5),
    metrics_dispersion(z = z, dz = 1, zmin = 0.5), metrics_canopydensity(z = z, interval_count = 10, zmin = 0.5),
    metrics_Lmoments(z = z, zmin = 0.5), metrics_lad(z = z, zmin = 0.5), metrics_interval(z = z, zmin = 0.5, zintervals = zi),
    metrics_echo(ReturnNumber = rn, NumberOfReturns = nr, z = z, zmin = 0.5),
    metrics_rumple(x = x, y = y, z = z, pixel_size = 1), metrics_voxels(x = x, y = y, z = z, vox_size = 1, zmin = 0.5),
    metrics_kde(z = z, zmin = 0.5))
}
cid <- floor((las$X - x0) / CELL) + 1000 * floor((y1 - las$Y) / CELL)
best <- as.integer(names(which.max(table(cid))))
sel <- which(cid == best)
NM <- names(m3(las$X[sel], las$Y[sel], las$Z[sel], las$Intensity[sel], las$ReturnNumber[sel], las$NumberOfReturns[sel]))
safe_m3 <- function(x, y, z, i, rn, nr) {
  o <- setNames(as.list(rep(NA_real_, length(NM))), NM)
  r <- tryCatch(suppressWarnings(m3(x, y, z, i, rn, nr)), error = function(e) NULL)
  if (!is.null(r)) { k <- intersect(names(r), NM); o[k] <- lapply(r[k], function(v) if (length(v) == 1) as.numeric(v) else NA_real_) }
  o
}
t0 <- Sys.time()
m <- pixel_metrics(las, ~safe_m3(X, Y, Z, Intensity, ReturnNumber, NumberOfReturns), res = tmpl)
t_area <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
dt <- as.data.table(as.data.frame(m, xy = TRUE, na.rm = FALSE))

t0 <- Sys.time()
chm <- rasterize_canopy(las, res = 0.5, algorithm = p2r(subcircle = 0.15, na.fill = knnidw()))
ws_fun <- function(h) pmin(pmax(2.5 + 0.15 * h, 3), 8)
ttops <- locate_trees(chm, lmf(ws = ws_fun, hmin = 5))
tt <- as.data.table(sf::st_coordinates(ttops))[, .(x = X, y = Y)]
tt[, `:=`(treeID = ttops$treeID, h = ttops$Z)]
if (nrow(ttops) > 0) {
  crowns <- dalponte2016(chm, ttops, th_tree = 2)()
  ca <- as.data.table(freq(crowns))[, .(treeID = value, crown_m2 = count * 0.25)]
  tt <- merge(tt, ca, by = "treeID", all.x = TRUE)
}
t_tree <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
tt[, `:=`(cx = x0 + (floor((x - x0) / CELL) + 0.5) * CELL, cy = y1 - (floor((y1 - y) / CELL) + 0.5) * CELL)]
gini <- function(v) { v <- sort(v); n <- length(v); if (n < 2) return(NA_real_); sum((2 * seq_len(n) - n - 1) * v) / (n * sum(v)) }
shan <- function(v) { p <- tabulate(floor(v / 2) + 1); p <- p[p > 0] / length(v); -sum(p * log(p)) }
tm <- tt[, .(tree_n = .N, tree_h_mean = mean(h), tree_h_sd = sd(h), tree_h_cv = sd(h) / mean(h), tree_h_max = max(h),
             tree_h_gini = gini(h), tree_h_shannon = shan(h),
             crown_m2_mean = mean(crown_m2, na.rm = TRUE), crown_m2_sd = sd(crown_m2, na.rm = TRUE)), by = .(x = cx, y = cy)]
dt <- merge(dt, tm, by = c("x", "y"), all.x = TRUE)
fwrite(dt, paste0(out, "_cells.csv"))
fwrite(tt, paste0(out, "_trees.csv"))
writeRaster(chm, paste0(out, "_chm.tif"), overwrite = TRUE)
cat(sprintf("cells %d trees %d read_s %.1f area_s %.1f tree_s %.1f\n", nrow(dt), nrow(tt), t_read, t_area, t_tree))
