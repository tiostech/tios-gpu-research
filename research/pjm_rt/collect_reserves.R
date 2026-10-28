# Cache PJM RT reserve market clearing prices, hourly, from Snowflake (needs the SSH tunnel for dbset). Run from
# ~/dev/tios-quant-shiny:  Rscript ~/dev/tios-gpu-research/research/pjm_rt/collect_reserves.R
# Sources (copies of Data Miner feeds, no API key needed):
#   PJM_FIVEMIN_DISPATCHED_RESERVES  5-min RT reserve results (dispatched_reserves), zones ALL and Mid-Atlantic/Dominion,
#     res_type primary / synchronized / thirty-minute reserve (from 2022-10): MCP, quantity, requirement, shortage flag
#   PJM_HOURLY_REGULATION_RESERVES   hourly RT regulation results (RMCP, capability, performance)
# hr is hour ending EPT (dt + hr - 1 = hour beginning). Writes data/pjm_rt/reserves/{reserves,regulation}.parquet.
suppressMessages({library(TiosQ)})
dbset("slave"); snowflake_conn <<- snow_conn(warehouse = "TIOSWH_TIOSQ")
out <- "~/dev/tios-gpu-research/data/pjm_rt/reserves"
r <- snow_query_odbc(snowflake_conn, sql = "
  select dt, hr, zone, res_type, count(*) n_int,
         avg(mkt_clr_price) mcp_mean, max(mkt_clr_price) mcp_max, sum(is_shortage) n_short,
         avg(res_qty) qty, avg(res_req) req, min(res_qty - res_req) min_margin
  from TIOSDATA_PRODUCTION.TIOSDATA.PJM_FIVEMIN_DISPATCHED_RESERVES
  where dt >= '2021-05-01' group by 1,2,3,4")
arrow::write_parquet(as.data.frame(r), file.path(out, "reserves.parquet"), compression = "zstd")
g <- snow_query_odbc(snowflake_conn, sql = "
  select dt, hr, market_clear_price rmcp, capability_clear_price rmccp, performance_clear_price rmpcp,
         reserve_qty_mw reg_qty, reserve_req_mw reg_req
  from TIOSDATA_PRODUCTION.TIOSDATA.PJM_HOURLY_REGULATION_RESERVES where region = 'ALL'")
arrow::write_parquet(as.data.frame(g), file.path(out, "regulation.parquet"), compression = "zstd")
cat("reserves", nrow(r), "rows; regulation", nrow(g), "rows\n")
