# Reference rasters from canopyLazR 2.0 (Kamoske et al. 2019) on a raw (not normalized) clip.
#   Rscript ref_canopylazr.R <raw laz> <out_prefix> [xy_res = 10] [z_res = 1] [ht_cut = 5]
# Workflow of the package README: laz.to.array(use.classified.returns = FALSE) -> canopy.height.levelr
# -> machorn.lad(k = NULL) -> rugosity.within.canopy, canopy.porosity.filled.ratio, lad.ht.max,
# array.to.ground.and.canopy.rasters -> toc.rugosity. Writes <out_prefix>_clz.csv (x, y, one column per raster).
suppressPackageStartupMessages({ library(canopyLazR); library(data.table) })
a <- commandArgs(trailingOnly = TRUE)
f <- a[1]; out <- a[2]
xy <- if (length(a) > 2) as.numeric(a[3]) else 10
zr <- if (length(a) > 3) as.numeric(a[4]) else 1
hc <- if (length(a) > 4) as.integer(a[5]) else 5L
epsg <- 6677
sink(tempfile())
arr <- laz.to.array(f, voxel.resolution = xy, z.resolution = zr, use.classified.returns = FALSE)
lev <- canopy.height.levelr(arr)
lad <- machorn.lad(lev, voxel.height = zr, beer.lambert.constant = NULL)
rug <- rugosity.within.canopy(lad, arr, ht.cut = hc, epsg.code = epsg)
por <- canopy.porosity.filled.ratio(lad, arr, ht.cut = hc, epsg.code = epsg)
mx <- lad.ht.max(lad, arr, ht.cut = hc, epsg.code = epsg)
gc_ <- array.to.ground.and.canopy.rasters(arr, epsg.code = epsg)
toc <- toc.rugosity(gc_$chm.raster, xy.res = xy, z.res = zr)
sink()
st <- raster::stack(list(clz_chm = gc_$chm.raster, clz_ground = gc_$ground.raster,
                         clz_vertical_sd_lad = rug$vertical.sd.lad.raster, clz_rugosity = rug$rugosity.raster,
                         clz_filled_ratio = por$filled.raster, clz_porosity_ratio = por$porosity.raster,
                         clz_max_lad = mx$max.lad.raster, clz_max_lad_ht = mx$max.lad.ht.raster,
                         clz_toc_rugosity = toc))
names(st) <- c("clz_chm", "clz_ground", "clz_vertical_sd_lad", "clz_rugosity", "clz_filled_ratio",
               "clz_porosity_ratio", "clz_max_lad", "clz_max_lad_ht", "clz_toc_rugosity")
d <- as.data.table(raster::as.data.frame(st, xy = TRUE))
fwrite(d, paste0(out, "_clz.csv"))
cat("columns", nrow(d), "dims", paste(dim(arr$array), collapse = "x"), "\n")
