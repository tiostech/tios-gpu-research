# Data and outputs live on the instance store (wiped on stop).
data_dir := env_var_or_default("DATA_DIR", "/scratch/data")
out_dir := env_var_or_default("OUT_DIR", "/scratch/outputs")

# Run by salt on boot: install deps and pre-download Kumo-Tabular weights.
init:
    uv sync
    uv run python -c "import sdm; sdm.models.KumoTabular()"

# Pull train/val/test from S3, e.g. `just pull s3://bucket/prefix/my-task`
pull s3_uri:
    uvx --from awscli aws s3 sync {{s3_uri}} {{data_dir}}/$(basename {{s3_uri}}) --exclude 'outputs/*'

# Confirm the GPU and model work end to end on a toy dataset
smoke:
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
    uv run python scripts/kumo_forecast.py --smoke

# Run Kumo-Tabular on a pulled dataset, e.g. `just forecast my-task --target y --time-col ts [--walk-forward]`
forecast task *args:
    uv run python scripts/kumo_forecast.py --data-dir {{data_dir}}/{{task}} --out-dir {{out_dir}}/{{task}} {{args}}

# Copy results back next to the data, e.g. `just push s3://bucket/prefix/my-task`
# -> s3://bucket/prefix/my-task/outputs/<UTC timestamp>/ so runs never overwrite.
push s3_uri:
    uvx --from awscli aws s3 sync {{out_dir}}/$(basename {{s3_uri}}) {{trim_end_match(s3_uri, "/")}}/outputs/$(date -u +%Y%m%dT%H%M%SZ)/
