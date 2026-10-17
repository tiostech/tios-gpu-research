#!/usr/bin/env bash
# Run Kumo walk-forward jobs one after another (the box has only 4 vCPUs, so
# concurrent runs are slower in total). Each argument is one job:
#   "variant step_days [run_name] [extra kumo_forecast args...]"
# e.g. nohup research/pjm/queue.sh "v1_price_lags 7" "v2_lags_deltas 1 v2_daily" &
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."
mkdir -p /scratch/logs
for job in "$@"; do
  read -r variant step name extra <<< "$job"
  name=${name:-$variant}
  log=/scratch/logs/${name}_${step}d.log
  echo "$(date '+%F %T') start $name (step ${step}d)" >> /scratch/logs/queue.log
  # shellcheck disable=SC2086
  uv run python scripts/kumo_forecast.py \
    --data-dir /scratch/data/pjm-fe/$variant --out-dir /scratch/outputs/pjm-fe/$name \
    --target y --time-col ts --walk-forward --step-days $step $extra > $log 2>&1
  echo "$(date '+%F %T') done  $name: $(grep '^val' $log || tail -1 $log)" >> /scratch/logs/queue.log
done
