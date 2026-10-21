# Pull OKGE load-forecast inputs (read-only, ecore "slave"). Each forecast table keeps a single
# vintage per (dt, hr[, key]), stamped D-1 ~09:00-09:22 market time, so Meteologica, Maxar and the
# SPP ISO forecast all reflect the same D-1 morning information set.
# Run from ~/dev/tios-quant-shiny:  Rscript <this file> <out_dir> [start] [end]
suppressMessages({
  library(TiosQ)
  source('~/dev/tios-quant-shiny/includes/common_include.R')
})
dbset("slave")
args <- commandArgs(trailingOnly = TRUE)
out_dir <- args[1]
s <- if (length(args) >= 2) args[2] else "2017-12-01"
e <- if (length(args) >= 3) args[3] else "2025-12-31"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
w <- sprintf("dt >= '%s' AND dt <= '%s'", s, e)
queries <- list(
  load = sprintf("SELECT dt, hr, zone, load_mwh FROM spp_hourly_load WHERE zone IN ('OKGE','ALL','GRDA','CSWS') AND %s", w),
  meteologica = sprintf("SELECT dt, hr, region, power_mw, lower_80, upper_80, forecast_time FROM spp_meteologica_load_forecasts WHERE region IN ('OKGE','ALL') AND %s", w),
  iso_fc = sprintf("SELECT dt, hr, load_mwh AS iso_load_fc, forecast_time FROM spp_load_forecast WHERE zone = 'ALL' AND %s", w),
  maxar = sprintf("SELECT dt, hr, location_id, temp, dew_point, rel_humidity, wchl, heat_index, wind_speed, tscv, precip, prob_of_precip, feelslike, forecast_time FROM spp_maxar_weather_forecasts WHERE location_id IN (13, 20, 203, 205) AND %s", w)
)
for (name in names(queries)) {
  df <- query_ecore_mysql(queries[[name]])
  df[] <- lapply(df, function(x) if (inherits(x, c("POSIXt", "Date"))) as.character(x) else x)
  arrow::write_parquet(df, file.path(out_dir, paste0(name, ".parquet")), compression = "zstd")
  cat(name, nrow(df), "rows\n")
}
