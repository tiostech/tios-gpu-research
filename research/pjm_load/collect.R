# Pull PJM ALL load-forecast inputs (read-only, ecore "slave"), keeping for each (dt, hr, key) the latest
# vintage issued by the cutoff 10:30 EPT on D-1 (forecast_time is EPT):
#   load            pjm_hourly_load zone ALL (actuals; only used as target and as lags >= 48h)
#   meteologica     pjm_meteologica_load_forecasts ALL + regions (one D-1 ~10:17-10:27 vintage per hour)
#   maxar           pjm_maxar_weather_forecasts for the 18 cities of blended_feelslike in
#                   ep-forecast.dafc-20261001.pjm.json (+ EWR, PHL)
# ISO pjm_load_forecast is NOT pulled: it keeps one vintage per hour, issued after 10:30 EPT on ~99% of
# 2019-2025 days, so no fair D-1 vintage exists (audit in research/pjm-kumo-feature-engineering.md).
# Run from ~/dev/tios-quant-shiny:  Rscript <this file> <out_dir> [start] [end]
suppressMessages({
  library(TiosQ)
  source('~/dev/tios-quant-shiny/includes/common_include.R')
})
dbset("slave")
args <- commandArgs(trailingOnly = TRUE)
out_dir <- args[1]
s <- if (length(args) >= 2) args[2] else "2017-11-01"
e <- if (length(args) >= 3) args[3] else "2025-12-31"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
fair <- "forecast_time <= timestamp(date_sub(dt, interval 1 day), '10:30:00')"
latest <- function(table, cols, key, where) {
  part <- paste(c("dt", "hr", key), collapse = ", ")
  sprintf(
    "select %s, %s, forecast_time from (
       select *, row_number() over (partition by %s order by forecast_time desc) rn
       from %s where dt >= '%s' and dt <= '%s' and %s and %s) x where rn = 1",
    part, cols, part, table, s, e, fair, where)
}
ids <- "3, 9, 10, 14, 15, 18, 20, 21, 27, 29, 30, 38, 70, 74, 114, 120, 162, 164, 11, 13"
queries <- list(
  load = sprintf("select dt, hr, zone, load_mwh from pjm_hourly_load where zone = 'ALL' and dt >= '%s' and dt <= '%s'", s, e),
  meteologica = latest("pjm_meteologica_load_forecasts", "power_mw", "region",
                       "region in ('ALL', 'MIDATL', 'SOUTH', 'WEST', 'WEST_CE', 'WEST_AEP', 'SOUTH_DOM', 'WEST_ATSI', 'MIDATL_PS')"),
  maxar = latest("pjm_maxar_weather_forecasts", "temp, dew_point, feelslike, wind_speed, tscv", "location_id",
                 sprintf("location_id in (%s)", ids))
)
for (name in names(queries)) {
  t0 <- Sys.time()
  df <- query_ecore_mysql(queries[[name]])
  df[] <- lapply(df, function(x) if (inherits(x, c("POSIXt", "Date"))) as.character(x) else x)
  arrow::write_parquet(df, file.path(out_dir, paste0(name, ".parquet")), compression = "zstd")
  cat(sprintf("%s: %d rows (%.0fs)\n", name, nrow(df), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
}
