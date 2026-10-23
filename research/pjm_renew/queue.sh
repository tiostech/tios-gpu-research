#!/usr/bin/env bash
# Run Kumo walk-forward jobs one after another (as research/pjm/queue.sh, but the project dir is an
# argument). Each job: "variant step_days [run_name] [extra kumo_forecast args...]"
#   nohup research/pjm_load/queue.sh pjm-load-fe "P0_weather 1 P0_gap24 --eval-every 3 --context-gap-hours 24" &
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."
proj=$1; shift
mkdir -p /scratch/logs
for job in "$@"; do
  read -r variant step name extra <<< "$job"
  name=${name:-$variant}
  log=/scratch/logs/${name}_${step}d.log
  echo "$(date '+%F %T') start $name (step ${step}d)" >> /scratch/logs/queue.log
  # shellcheck disable=SC2086
  uv run python scripts/kumo_forecast.py \
    --data-dir /scratch/data/$proj/$variant --out-dir /scratch/outputs/$proj/$name \
    --target y --time-col ts --walk-forward --step-days $step $extra > $log 2>&1
  echo "$(date '+%F %T') done  $name: $(grep '^val' $log || tail -1 $log)" >> /scratch/logs/queue.log
done
