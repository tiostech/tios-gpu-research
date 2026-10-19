# Pull extra PJM fundamentals for Kumo feature engineering (round 2), keeping
# for each (dt, hr, key) only the latest vintage issued before the cutoff
# 10:30 EPT on D-1 (PJM bids close 11:00 EPT; shiny's fairness cutoff is -30 min).
# forecast_time is market-local (EPT for PJM).
# Run from ~/dev/tios-quant-shiny:  Rscript <this file> <out_dir> [start]
suppressMessages({
  library(TiosQ)
  source('~/dev/tios-quant-shiny/includes/common_include.R')
})
dbset("slave")

args <- commandArgs(trailingOnly = TRUE)
out_dir <- args[1]
start_dt <- if (length(args) >= 2) args[2] else "2017-11-01"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

fair <- "forecast_time <= timestamp(date_sub(dt, interval 1 day), '10:30:00')"

latest <- function(table, cols, key = NULL, where = "1=1") {
  part <- paste(c("dt", "hr", key), collapse = ", ")
  sprintf(
    "select %s, %s, forecast_time from (
       select *, row_number() over (partition by %s order by forecast_time desc) rn
       from %s where dt >= '%s' and %s and %s) x where rn = 1",
    part, cols, part, table, start_dt, fair, where)
}

queries <- list(
  outages = latest(
    "pjm_totalgen_on_outages",
    "capacity_on_outage_mw, renewable_capacity_on_outage_mw, new_equipment_on_outage_mw,
     planned_outages_mw, maintenance_outages_mw, forced_outages_mw",
    key = "region", where = "region in ('ALL', 'Western', 'Mid Atlantic - Dominion')"),
  scen_load = latest(
    "pjm_meteologica_scenario_load_forecasts",
    "average, std_dev, quantile_5, quantile_50, quantile_95, minimum, maximum",
    key = "region", where = "region in ('ALL')"),
  scen_wind = latest(
    "pjm_meteologica_scenario_wind_forecasts",
    "average, std_dev, quantile_5, quantile_50, quantile_95",
    key = "region", where = "region in ('ECMWF-ENS', 'GEFS')"),
  scen_solar = latest(
    "pjm_meteologica_scenario_solar_forecasts",
    "average, std_dev, quantile_5, quantile_95",
    key = "region", where = "region in ('ECMWF-ENS')"),
  met_load = latest(
    "pjm_meteologica_load_forecasts", "power_mw, lower_80, upper_80",
    key = "region", where = "region in ('ALL')"),
  # ISO load forecast: mostly stamped ~10:47 EPT (after cutoff) -> pulled but unused.
  iso_load = latest(
    "pjm_load_forecast", "load_mwh", key = "zone",
    where = "zone in ('RTO COMBINED', 'WESTERN REGION', 'SOUTHERN REGION', 'PJME/MID ATLANTIC REGION')"),
  maxar = latest(
    "pjm_maxar_weather_forecasts",
    "temp, dew_point, feelslike, wind_speed, heat_index, wchl",
    key = "location_id",
    where = "location_id in (3, 9, 10, 11, 13, 15, 18, 20, 21, 27, 70, 74, 114, 162, 164)"),
  # Published on the operating day itself (05-08 EPT) -> only usable as a lag:
  # today's summary is known before the cutoff for tomorrow's auction.
  op_summary = sprintf(
    "select dt, peak_hr, scheduled_capacity_internal_mw, scheduled_tie_flow_total_mw,
       scheduled_capacity_total_mw, capacity_adjustments_mw, load_forecast_mw,
       operating_reserve_mw, unscheduled_steam_capacity_mw, forecast_time
     from (select *, row_number() over (partition by dt order by forecast_time desc) rn
           from pjm_op_summaries where dt >= '%s'
             and forecast_time <= timestamp(dt, '10:30:00')) x where rn = 1", start_dt)
)

for (name in names(queries)) {
  t0 <- Sys.time()
  df <- query_ecore_mysql(queries[[name]])
  df[] <- lapply(df, function(x) if (inherits(x, c("POSIXt", "Date"))) as.character(x) else x)
  path <- file.path(out_dir, paste0(name, ".parquet"))
  arrow::write_parquet(df, path, compression = "zstd")
  cat(sprintf("%s: %d rows (%.0fs) -> %s\n", name, nrow(df),
              as.numeric(difftime(Sys.time(), t0, units = "secs")), path))
}
