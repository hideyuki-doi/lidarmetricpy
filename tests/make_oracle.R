# lidRmetrics::metrics_set3 (installed lidR 4.3.3, lidRmetrics 0.1.5) on the single-cell cases of oracle_cases.py,
# with the arguments of validation/ref_lidr.R. Heights are read as exact hex doubles.
#   Rscript make_oracle.R data/ref/oracle_input.txt data/ref/oracle_cases.csv
suppressPackageStartupMessages({ library(lidR); library(lidRmetrics); library(data.table) })
a <- commandArgs(trailingOnly = TRUE)
L <- readLines(a[1])
num <- function(s) as.numeric(strsplit(s, " ")[[1]])
rows <- list()
for (ln in L) {
  p <- strsplit(ln, "\t")[[1]]
  x <- num(p[3]); y <- num(p[4]); z <- num(p[5])
  rn <- as.integer(num(p[6])); nr <- as.integer(num(p[7]))
  m <- suppressWarnings(metrics_set3(x = x, y = y, z = z, i = rep(1L, length(z)), ReturnNumber = rn,
                                     NumberOfReturns = nr, zmin = as.numeric(p[2]),
                                     threshold = c(2, 5, 10, 20, 30), zintervals = c(0, 0.5, 2, 5, 10, 20, 30),
                                     pixel_size = 1, vox_size = 1))
  m <- lapply(m, function(v) if (length(v) == 1) as.numeric(v) else NA_real_)
  rows[[length(rows) + 1]] <- c(list(case = p[1]), m)
}
d <- rbindlist(rows, fill = TRUE)
fwrite(d, a[2])
cat("cases", nrow(d), "R", R.version.string, "lidR", as.character(packageVersion("lidR")),
    "lidRmetrics", as.character(packageVersion("lidRmetrics")), "\n")
