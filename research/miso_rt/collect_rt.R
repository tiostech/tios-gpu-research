# MISO DA and RT system energy prices (miso_energy_prices, the table behind shiny's da_/rt_energy_prices()).
# Run from ~/dev/tios-quant-shiny:  Rscript ~/dev/tios-gpu-research/research/miso_rt/collect_rt.R [start] [end]
# Writes data/miso/miso_energy_prices.parquet (dt, hr, da_energy_price, rt_energy_price).
suppressMessages({
  library(TiosQ)
  source('~/dev/tios-quant-shiny/includes/common_include.R')
})
dbset("slave")
args <- commandArgs(trailingOnly = TRUE)
start <- if (length(args) >= 1) args[1] else "2017-12-01"
end <- if (length(args) >= 2) args[2] else "2026-10-20"
df <- query_ecore_mysql(sprintf(
  "select dt, hr, da_energy_price, rt_energy_price from miso_energy_prices where dt >= '%s' and dt <= '%s' order by dt, hr",
  start, end))
df$dt <- as.character(df$dt)
out <- "~/dev/tios-gpu-research/data/miso/miso_energy_prices.parquet"
arrow::write_parquet(df, out, compression = "zstd")
cat("wrote", out, nrow(df), "rows,", min(df$dt), "..", max(df$dt), ", rt NA", sum(is.na(df$rt_energy_price)), "\n")
