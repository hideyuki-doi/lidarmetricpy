# lasR (C++ rewrite of lidR's pipeline) on the dense Kanagawa tile: the metrics it shares with lidarmetricpy, and
# its own tree detection, timed like the other tools (work only; reading + metrics + trees).
#   ~/miniforge3/envs/lasr/bin/Rscript lasr_compare.R <laz> <out_prefix> <ncores>
# Writes <out_prefix>_cells.csv (25 m: n, zq20, zcv, pz_above_30 from points with Z > 0.5, as lidRmetrics),
# <out_prefix>_trees.gpkg (tree tops) and <out_prefix>_time.txt.
# Trees: canopy raster 0.5 m (highest point, no subcircle or knnidw fill: lasR has neither), lmf with a fixed 3 m
# window (lasR's local_maximum_raster has no variable window), min height 5 m, region growing (Dalponte-like).
suppressPackageStartupMessages(library(lasR))
a <- commandArgs(trailingOnly = TRUE)
laz <- a[1]; out <- a[2]; nc <- as.integer(a[3])
metrics <- rasterize(25, c("count", "z_p20", "z_cv", "z_above30"), filter = "Z > 0.5", ofile = paste0(out, "_m.tif"))
chm <- rasterize(0.5, "max", ofile = paste0(out, "_chm.tif"))
seeds <- local_maximum_raster(chm, ws = 3, min_height = 5, ofile = paste0(out, "_trees.gpkg"))
crowns <- region_growing(chm, seeds, ofile = paste0(out, "_crowns.tif"))
t0 <- Sys.time()
# the outputs are written to files; reading them back into R fails in this build (vreader), outside the timing
res <- tryCatch(exec(reader() + metrics + chm + seeds + crowns, on = laz, ncores = nc), error = function(e) NULL)
el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
writeLines(sprintf("%.2f", el), paste0(out, "_time.txt"))
cat("lasR", as.character(packageVersion("lasR")), "ncores", nc, "seconds", el, "\n")
