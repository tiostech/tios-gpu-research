# Pull the full PJM flex DAEP 20261001 feature frame (forecast-vintage features,
# DA/RT prices, West Hub ICE/congestion) for Kumo feature engineering.
# Run from ~/dev/tios-quant-shiny:  Rscript <this file> <out.parquet> [start] [end]
library(TiosQ)
source('~/dev/tios-quant-shiny/includes/common_include.R')
dbset("slave")

args <- commandArgs(trailingOnly = TRUE)
out <- args[1]
start_dt <- if (length(args) >= 2) args[2] else "2017-12-01"
end_dt <- if (length(args) >= 3) args[3] else as.character(Sys.Date() - 1)

obj <- PjmFlexDaep20261001$new()
df <- obj$do_collect_data(start_dt, end_dt)
df <- df %>% dplyr::mutate(time = as.character(time), dt = as.character(dt))
arrow::write_parquet(df, out, compression = "zstd")
cat("wrote", out, nrow(df), "rows,", ncol(df), "cols\n")
