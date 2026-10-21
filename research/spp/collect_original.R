# Pull the SPP flex DAEP 20260701 feature frame (fair forecast-vintage features, DA energy price)
# for the "original" Kumo run (baseline rf730d features).
# Run from ~/dev/tios-quant-shiny:  Rscript <this file> <out.parquet> [start] [end]
suppressMessages({
  library(TiosQ)
  source('~/dev/tios-quant-shiny/includes/common_include.R')
})
dbset("slave")

args <- commandArgs(trailingOnly = TRUE)
out <- args[1]
start_dt <- as.Date(if (length(args) >= 2) args[2] else "2017-12-01")
end_dt <- as.Date(if (length(args) >= 3) args[3] else as.character(Sys.Date() - 1))

obj <- SppFlexDaep20260701$new()
df <- obj$do_collect_data(start_dt, end_dt)
df[] <- lapply(df, function(x) if (inherits(x, c("POSIXt", "Date"))) as.character(x) else x)
arrow::write_parquet(df, out, compression = "zstd")
cat("wrote", out, nrow(df), "rows,", ncol(df), "cols\n")
