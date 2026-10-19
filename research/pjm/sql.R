# Run a read-only SQL query on the ecore replica ("slave"; needs the SSH tunnel).
#   Rscript research/pjm/sql.R "<sql>" [out.parquet|out.csv]   (run from ~/dev/tios-quant-shiny)
# Without out.parquet, prints the result.
suppressMessages({
  library(TiosQ)
  source('~/dev/tios-quant-shiny/includes/common_include.R')
})
dbset("slave")
args <- commandArgs(trailingOnly = TRUE)
df <- query_ecore_mysql(args[1])
if (length(args) >= 2 && grepl("\\.csv$", args[2])) {
  write.csv(df, args[2], row.names = FALSE)
} else if (length(args) >= 2) {
  df[] <- lapply(df, function(x) if (inherits(x, c("POSIXt", "Date"))) as.character(x) else x)
  arrow::write_parquet(df, args[2], compression = "zstd")
  cat("wrote", args[2], nrow(df), "rows\n")
} else {
  print(as.data.frame(df), width = 250)
}
