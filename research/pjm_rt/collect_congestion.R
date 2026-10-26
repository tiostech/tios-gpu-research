# Cache PJM hourly nodal congestion locally (read-only "slave" replica; needs the SSH tunnel). Run from ~/dev/tios-quant-shiny:
#   Rscript ~/dev/tios-gpu-research/research/pjm_rt/collect_congestion.R fc  [start end]
#   Rscript ~/dev/tios-gpu-research/research/pjm_rt/collect_congestion.R act [start end]
# fc : pjm_hourly_congestion_forecasts.rt_congestion_forecast_pure_shadow_based (trader shadow-price forecasts x shift
#      factors, no adjustments), keyed by trade_dt = the operating day; one stored value per (trade_dt, hr, node).
# act: pjm_lmp_combined rt_congestion and da_congestion (actual), restricted to nodes that appear in the forecasts.
# Writes one zstd parquet per month to data/pjm_rt/cong/<kind>/YYYY-MM.parquet (float32 values); existing months are
# skipped, so the pull resumes. Default range 2023-09-01 .. 2025-12-31 (the RT model's context and 2025; never 2026).
suppressMessages({
  library(TiosQ)
  source('~/dev/tios-quant-shiny/includes/common_include.R')
})
dbset("slave")
args <- commandArgs(trailingOnly = TRUE)
kind <- args[1]
start <- as.Date(if (length(args) >= 2) args[2] else "2023-09-01")
end <- as.Date(if (length(args) >= 3) args[3] else "2025-12-31")
stopifnot(kind %in% c("fc", "act"), end < as.Date("2026-01-01"))
out <- file.path("~/dev/tios-gpu-research/data/pjm_rt/cong", kind)
dir.create(out, recursive = TRUE, showWarnings = FALSE)

nodes <- NULL
if (kind == "act") {
  nf <- file.path("~/dev/tios-gpu-research/data/pjm_rt/cong/fc_nodes.csv")
  if (!file.exists(nf)) {
    n <- query_ecore_mysql(sprintf(
      "select distinct official_pnode_id from pjm_hourly_congestion_forecasts where trade_dt between '%s' and '%s'", start, end))
    write.csv(n, nf, row.names = FALSE)
  }
  nodes <- read.csv(nf)$official_pnode_id
  cat("act: restricting to", length(nodes), "forecast nodes\n")
}

months <- seq(as.Date(format(start, "%Y-%m-01")), end, by = "month")
for (m in as.list(months)) {
  f <- file.path(out, paste0(format(m, "%Y-%m"), ".parquet"))
  if (file.exists(f)) next
  days <- seq(max(m, start), min(seq(m, by = "month", length.out = 2)[2] - 1, end), by = "day")
  t0 <- Sys.time()
  parts <- lapply(days, function(d) {
    if (kind == "fc") {
      x <- query_ecore_mysql(sprintf(
        "select trade_dt as dt, hr, official_pnode_id, rt_congestion_forecast_pure_shadow_based as fc
         from pjm_hourly_congestion_forecasts where trade_dt = '%s'", d))
    } else {
      x <- query_ecore_mysql(sprintf(
        "select dt, hr, official_pnode_id, rt_congestion as rt, da_congestion as da from pjm_lmp_combined
         where dt = '%s' and official_pnode_id in (%s)", d, paste(nodes, collapse = ",")))
    }
    x
  })
  df <- do.call(rbind, parts)
  df$dt <- as.character(df$dt)
  for (c in intersect(c("fc", "rt", "da"), names(df))) df[[c]] <- arrow::Array$create(df[[c]], type = arrow::float32())
  tb <- arrow::Table$create(df)
  arrow::write_parquet(tb, paste0(f, ".tmp"), compression = "zstd")
  file.rename(paste0(f, ".tmp"), f)
  cat(format(Sys.time()), kind, format(m, "%Y-%m"), nrow(df), "rows", round(as.numeric(difftime(Sys.time(), t0, units = "secs"))), "s\n")
}
cat("done", kind, "\n")
