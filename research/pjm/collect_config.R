# Save every column of the dafc-20261001 config import (regional load/wind/solar
# forecasts etc.), which PjmFlexDaep20261001$collect_data() subsets.
# Run from ~/dev/tios-quant-shiny:  Rscript <this file> <out.parquet> [start] [end]
library(TiosQ)
source('~/dev/tios-quant-shiny/includes/common_include.R')
dbset("slave")

args <- commandArgs(trailingOnly = TRUE)
out <- args[1]
start_dt <- if (length(args) >= 2) args[2] else "2017-12-01"
end_dt <- if (length(args) >= 3) args[3] else as.character(Sys.Date() - 1)

cfg <- tios_epfc_config_reader(
  market = "pjm", config_name = "dafc-20261001",
  local_dir = here::here("lib/trader-forecasting/jsons")
)
data_list <- tios_epfc_data_collect.data_import(
  market = "pjm",
  train_start_time = paste(start_dt, "00:00:00"), train_end_time = paste(end_dt, "23:00:00"),
  test_start_time = paste(start_dt, "00:00:00"), test_end_time = paste(end_dt, "23:00:00"),
  config = cfg
)
df <- data_list$test_data %>% dplyr::mutate(time = as.character(time), dt = as.character(dt))
arrow::write_parquet(df, out, compression = "zstd")
cat("wrote", out, nrow(df), "rows,", ncol(df), "cols\n")
