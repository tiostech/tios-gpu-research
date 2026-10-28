# Cache unit-level PJM DA generator offer curves (pjm_da_gs_generator_bids: per bidding group and hour, 10 cumulative
# MW / price segments, fuel) for a year. Run from ~/dev/tios-quant-shiny (needs the SSH tunnel):
#   Rscript ~/dev/tios-gpu-research/research/pjm_rt/collect_offers.R 2025
# Writes data/pjm_rt/offers/YYYY-MM.parquet (zstd); existing months are skipped.
source("~/dev/tios-quant-shiny/includes/common_include.R"); dbset("slave")
yr <- commandArgs(trailingOnly = TRUE)[1]
out <- "~/dev/tios-gpu-research/data/pjm_rt/offers"; dir.create(out, recursive = TRUE, showWarnings = FALSE)
for (m in sprintf("%s-%02d", yr, 1:12)) {
  f <- file.path(out, paste0(m, ".parquet")); if (file.exists(f)) next
  x <- query_ecore_mysql(sprintf(
    "select dt, hr, bidding_group_name unit, mapped_fuel_delivery fuel, %s, %s from pjm_da_gs_generator_bids
     where dt >= '%s-01' and dt < date_add('%s-01', interval 1 month)",
    paste0("bid", 1:10, collapse = ","), paste0("mw", 1:10, collapse = ","), m, m))
  x$dt <- as.character(x$dt)
  arrow::write_parquet(x, f, compression = "zstd"); cat(m, nrow(x), "rows\n")
}
