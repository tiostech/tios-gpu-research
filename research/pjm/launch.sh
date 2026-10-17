#!/usr/bin/env bash
# Launch a Kumo walk-forward run on a feature variant, in the background.
#   research/pjm/launch.sh <variant> <step_days> [run_name] [extra kumo_forecast args...]
# Data: /scratch/data/pjm-fe/<variant>; output: /scratch/outputs/pjm-fe/<run_name>
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
variant=$1; step=$2; name=${3:-${variant}}; shift $(( $# >= 3 ? 3 : $# ))
cd "$(dirname "$0")/../.."
mkdir -p /scratch/logs
log=/scratch/logs/${name}_${step}d.log
nohup uv run python scripts/kumo_forecast.py \
  --data-dir /scratch/data/pjm-fe/$variant --out-dir /scratch/outputs/pjm-fe/$name \
  --target y --time-col ts --walk-forward --step-days $step "$@" > $log 2>&1 &
echo "started $name (step $step d) pid $! -> $log"
