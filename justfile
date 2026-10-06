# Data and outputs live on the instance store (wiped on stop).
data_dir := env_var_or_default("DATA_DIR", "/scratch/data")
out_dir := env_var_or_default("OUT_DIR", "/scratch/outputs")

# Run by salt on boot: install deps and pre-download Kumo-Tabular weights.
init:
    uv sync
    uv run python -c "import sdm; sdm.models.KumoTabular()"

# Pull train/val/test from S3, e.g. `just pull s3://bucket/prefix/my-task`
pull s3_uri:
    uvx --from awscli aws s3 sync {{s3_uri}} {{data_dir}}/$(basename {{s3_uri}})

# Run Kumo-Tabular on a pulled dataset, e.g. `just forecast my-task --target y --time-col ts`
forecast task *args:
    uv run python scripts/kumo_forecast.py --data-dir {{data_dir}}/{{task}} --out-dir {{out_dir}}/{{task}} {{args}}
