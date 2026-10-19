# Pull every West Hub ICE next-day contract price (last trade / bid-ask mid before
# each contract's own fairness cutoff), WITHOUT ice_price_computor's
# "contract starts the day after trading" filter, so Monday/post-holiday contracts
# traded days earlier are kept. trade_time is kept for as-of joins in Python.
# Run from ~/dev/tios-quant-shiny:  Rscript <this file> <out.parquet> [start] [end]
suppressMessages({
  library(TiosQ)
  source('~/dev/tios-quant-shiny/includes/common_include.R')
})
dbset("slave")

args <- commandArgs(trailingOnly = TRUE)
out <- args[1]
start_dt <- if (length(args) >= 2) args[2] else "2017-11-01"
end_dt <- if (length(args) >= 3) args[3] else as.character(Sys.Date())

pull <- function(instrument_id) {
  opts <- list(instrument_id = instrument_id, start_dt = start_dt, end_dt = end_dt,
               market = "pjm", stale_minute = 60, ddl_ahead_minute = 30)
  trades <- do.call(ice_price_computor.ice_trades, opts)
  intraday <- do.call(ice_price_computor.ice_intraday, opts)
  rows <- list()
  if (nrow(trades) > 0) {
    rows$t <- trades %>% dplyr::transmute(start_dt = begin_dt, end_dt, trade_time = last_ept,
                                          ice_price = last_price, data_source = 1)
  }
  if (nrow(intraday) > 0) {
    rows$i <- intraday %>% dplyr::transmute(start_dt = begin_dt, end_dt, trade_time = market_time,
                                            ice_price = avg_price, data_source = 2)
  }
  dplyr::bind_rows(rows) %>%
    dplyr::group_by(start_dt, end_dt) %>%
    dplyr::arrange(data_source, .by_group = TRUE) %>%
    dplyr::filter(dplyr::row_number() == 1) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(instrument_id = instrument_id)
}

df <- dplyr::bind_rows(pull(9), pull(10))  # 9 = nextday DA-settled, 10 = nextday RT-settled
df[] <- lapply(df, function(x) if (inherits(x, c("POSIXt", "Date"))) as.character(x) else x)
arrow::write_parquet(df, out, compression = "zstd")
cat("wrote", out, nrow(df), "rows\n")
print(df %>% dplyr::count(instrument_id, data_source))
