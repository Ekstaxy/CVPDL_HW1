#!/usr/bin/env bash
# Full chain for one experiment: train -> evaluate (val CSV + mAP) -> test CSV.
# usage: scripts/run_experiment.sh <exp_id> [key.sub=value ...]
# Re-running is safe: finished steps are skipped and an interrupted training resumes.
set -uo pipefail
cd "$(dirname "$0")/.."
ID="$1"; shift
PY=.venv/bin/python
CFG="configs/$ID.yaml"
OUT="results/$ID"
mkdir -p "$OUT"
[ -f "$OUT/DONE" ] && { echo "[$ID] already DONE"; exit 0; }
rm -f "$OUT/FAILED"

fail() { echo "$1" > "$OUT/FAILED"; echo "[$ID] FAILED: $1"; exit 1; }

$PY train.py --config "$CFG" "$@" 2>&1 | tee -a "$OUT/train.log"
rc=${PIPESTATUS[0]}
if [ "$rc" -eq 42 ]; then
  # out of memory under the VRAM cap: restart once from scratch with half the batch
  batch=$($PY -c "import sys; from utils.config import load_config; print(max(1, load_config('$CFG', sys.argv[1:])['train']['batch'] // 2))" "$@")
  echo "[$ID] OOM, retrying with batch=$batch" | tee -a "$OUT/train.log"
  rm -rf "$OUT/train"
  set -- "$@" "train.batch=$batch"
  $PY train.py --config "$CFG" "$@" 2>&1 | tee -a "$OUT/train.log"
  rc=${PIPESTATUS[0]}
fi
[ "$rc" -eq 0 ] || fail "train (exit $rc)"

# evaluate/test read the config saved by train.py, so any overrides stay in effect
$PY evaluate.py --config "$OUT/config.yaml" 2>&1 | tee "$OUT/eval.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "evaluate"
$PY test.py --config "$OUT/config.yaml" 2>&1 | tee "$OUT/test.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "test"
date -Is > "$OUT/DONE"
echo "[$ID] DONE"
