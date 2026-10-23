# Pull PJM ALL wind / solar forecast inputs (read-only, ecore "slave") for Artifact 4.
# For forecasts, keep for each (dt, hr, key) the latest vintage issued by the cutoff (forecast_time is EPT):
#   scenario and Maxar: D-1 10:30 EPT (shiny's fairness rule)
#   Meteologica wind / solar: D-1 11:00 EPT. These tables keep one vintage per hour, stamped 10:23-10:59 EPT
#   (after 10:30 on ~70% of 2023-2025 hours); v0's pjm_wind / pjm_solar (production inputs) equal it on every
#   hour, so this is what reproduces v0 (Gate 1).
#   ISO forecasts (wind, solar, load): D-1 11:00 EPT = PJM's day-ahead bid deadline (user decision 2026-10-23);
#   stamped ~10:55 (wind), ~10:00 (solar), ~10:47 (load) EPT, one vintage per hour.
# Tables:
#   wind_act / solar_act  pjm_windgen_outputs / pjm_solargen_outputs region ALL (actuals = targets; one row per
#                         hour, overwritten in place, no revision history -> used only as lags >= 48h)
#   met_wind / met_solar  pjm_meteologica_{wind,solar}_forecasts: ALL ("Meteologica") + regions (solar_type pv)
#   scen_wind / scen_solar pjm_meteologica_scenario_* (ECMWF-ENS, GEFS / ECMWF-ENS)
#   iso_wind / iso_solar / iso_load  pjm_windgen_output_forecasts ALL, pjm_solargen_output_forecasts ALL + BTM_ALL,
#                         pjm_load_forecast RTO COMBINED
#   maxar                 pjm_maxar_weather_forecasts at stations near the wind and solar clusters
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
cut <- function(t) sprintf("forecast_time <= timestamp(date_sub(dt, interval 1 day), '%s')", t)
latest <- function(table, cols, key, where, cutoff = "10:30:00") {
  part <- paste(c("dt", "hr", key), collapse = ", ")
  sprintf(
    "select %s, %s, forecast_time from (
       select *, row_number() over (partition by %s order by forecast_time desc) rn
       from %s where dt >= '%s' and dt <= '%s' and %s and %s) x where rn = 1",
    part, cols, part, table, s, e, cut(cutoff), where)
}
met_regions <- "'Meteologica', 'Meteologica_West', 'Meteologica_Mid-Atlantic', 'Meteologica_South'"
wind_sub <- "'Meteologica_West_CE', 'Meteologica_West_AEP', 'Meteologica_West_AP', 'Meteologica_West_ATSI',
             'Meteologica_Mid-Atlantic_PL', 'Meteologica_Mid-Atlantic_PN', 'Meteologica_Mid-Atlantic_AE',
             'Meteologica_South_DOM'"
# wind: Pontiac, Rockford, O'Hare, Lafayette IN, Fort Wayne, South Bend, Toledo, Dayton, Pittsburgh, Erie,
#       Charleston WV, Williamsport, Scranton
# solar: Richmond, Charlottesville, Norfolk, Dulles, Atlantic City, Trenton, BWI, Georgetown DE, Columbus,
#        Dayton, Harrisburg, Indianapolis, Lafayette IN, Pontiac
ids <- "165, 172, 162, 126, 10, 175, 29, 80, 164, 93, 21, 120, 14, 20, 71, 18, 114, 43, 12, 15, 102, 3, 30, 5"
queries <- list(
  wind_act = sprintf("select dt, hr, power_mw, preliminary, last_update_for_date from pjm_windgen_outputs
                      where region = 'ALL' and dt >= '%s' and dt <= '%s'", s, e),
  solar_act = sprintf("select dt, hr, power_mw from pjm_solargen_outputs
                       where region = 'ALL' and dt >= '%s' and dt <= '%s'", s, e),
  met_wind = latest("pjm_meteologica_wind_forecasts", "power_mw, lower_80, upper_80", "region",
                    sprintf("region in (%s, %s)", met_regions, wind_sub), "11:00:00"),
  met_solar = latest("pjm_meteologica_solar_forecasts", "power_mw, lower_80, upper_80", "region",
                     sprintf("region in (%s) and solar_type = 'pv'", met_regions), "11:00:00"),
  scen_wind = latest("pjm_meteologica_scenario_wind_forecasts",
                     "average, std_dev, quantile_5, quantile_50, quantile_95", "region",
                     "region in ('ECMWF-ENS', 'GEFS')"),
  scen_solar = latest("pjm_meteologica_scenario_solar_forecasts",
                      "average, std_dev, quantile_5, quantile_95", "region", "region in ('ECMWF-ENS')"),
  iso_wind = latest("pjm_windgen_output_forecasts", "power_mw", "region", "region = 'ALL'", "11:00:00"),
  iso_solar = latest("pjm_solargen_output_forecasts", "power_mw", "region", "region in ('ALL', 'BTM_ALL')", "11:00:00"),
  iso_load = latest("pjm_load_forecast", "load_mwh", "zone", "zone = 'RTO COMBINED'", "11:00:00"),
  maxar = latest("pjm_maxar_weather_forecasts", "temp, wind_speed, wind_dir, tscv, rel_humidity, prob_of_precip",
                 "location_id", sprintf("location_id in (%s)", ids))
)
only <- if (length(args) >= 4) strsplit(args[4], ",")[[1]] else names(queries)
for (name in only) {
  t0 <- Sys.time()
  df <- query_ecore_mysql(queries[[name]])
  df[] <- lapply(df, function(x) if (inherits(x, c("POSIXt", "Date"))) as.character(x) else x)
  arrow::write_parquet(df, file.path(out_dir, paste0(name, ".parquet")), compression = "zstd")
  cat(sprintf("%s: %d rows (%.0fs)\n", name, nrow(df), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
}
