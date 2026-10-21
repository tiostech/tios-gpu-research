# Pull SPP ALL load-forecast inputs (read-only, ecore "slave"): Maxar weather for the 26 cities of the
# feelslike_spp_popweighted block in ep-forecast.daep-20260701.spp.json, and Meteologica load forecasts
# for every SPP zone (single D-1 ~09:20 vintage each). Load actuals / ISO fc / Meteologica ALL come from
# collect.R. Run from ~/dev/tios-quant-shiny:  Rscript <this file> <out_dir> [start] [end]
suppressMessages({
  library(TiosQ)
  source('~/dev/tios-quant-shiny/includes/common_include.R')
})
dbset("slave")
args <- commandArgs(trailingOnly = TRUE)
out_dir <- args[1]
s <- if (length(args) >= 2) args[2] else "2017-12-01"
e <- if (length(args) >= 3) args[3] else "2025-12-31"
w <- sprintf("dt >= '%s' AND dt <= '%s'", s, e)
ids <- "2, 60, 159, 89, 6, 7, 122, 236, 142, 135, 177, 146, 152, 59, 153, 32, 20, 21, 161, 209, 25, 26, 189, 208, 27, 28"
queries <- list(
  maxar_all = sprintf("SELECT dt, hr, location_id, temp, dew_point, rel_humidity, wind_speed, tscv, feelslike, forecast_time FROM spp_maxar_weather_forecasts WHERE location_id IN (%s) AND %s", ids, w),
  meteologica_zones = sprintf("SELECT dt, hr, region, power_mw, forecast_time FROM spp_meteologica_load_forecasts WHERE region IN ('CSWS','EDE','GRDA','INDN','KACY','KCPL','LES','MPS','NPPD','OKGE','OPPD','SECI','SPRM','SPS','WAUE','WFEC','WR') AND %s", w)
)
for (name in names(queries)) {
  df <- query_ecore_mysql(queries[[name]])
  df[] <- lapply(df, function(x) if (inherits(x, c("POSIXt", "Date"))) as.character(x) else x)
  arrow::write_parquet(df, file.path(out_dir, paste0(name, ".parquet")), compression = "zstd")
  cat(name, nrow(df), "rows\n")
}
