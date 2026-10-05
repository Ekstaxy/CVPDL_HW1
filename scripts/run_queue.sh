#!/usr/bin/env bash
# Run a list of experiments unattended, one after another, then summarize.
# usage: nohup scripts/run_queue.sh configs/queues/stage1.txt > results/queue.log 2>&1 &
# Queue file: one experiment id per line (# comments allowed). A line starting with "!" is run as a
# shell command (used for the inference sweep). The file is re-read before every step, so ids can be
# appended while the queue is running. Failed experiments are recorded and skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
QUEUE="$1"
mkdir -p results
STATE="results/.queue_$(basename "$QUEUE").done"
touch "$STATE"
while true; do
  next=""
  while IFS= read -r line; do
    line="${line%%#*}"; line="${line#"${line%%[![:space:]]*}"}"; line="${line%"${line##*[![:space:]]}"}"
    [ -z "$line" ] && continue
    grep -qxF -- "$line" "$STATE" || { next="$line"; break; }
  done < "$QUEUE"
  [ -z "$next" ] && break
  echo "===== $(date -Is) queue: $next ====="
  if [[ "$next" == "!"* ]]; then
    bash -c "${next:1}"
  else
    scripts/run_experiment.sh "$next"
  fi
  echo "$next" >> "$STATE"
  .venv/bin/python scripts/summarize.py > /dev/null 2>&1
done
.venv/bin/python scripts/summarize.py
echo "===== $(date -Is) queue finished ====="
