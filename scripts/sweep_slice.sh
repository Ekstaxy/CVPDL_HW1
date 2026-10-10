#!/usr/bin/env bash
# Slice-parameter sweep for one finished run (inference only): window size x overlap, then the best
# window combined with a 1.25x full-image view. Each variant lands in results/<id>/eval_<tag>/.
# usage: scripts/sweep_slice.sh s3_01_dfine_l_1536
set -uo pipefail
cd "$(dirname "$0")/.."
ID="$1"; CFG="results/$ID/config.yaml"; PY=.venv/bin/python
IMGSZ=$($PY -c "import yaml; c=yaml.safe_load(open('$CFG')); print(c['predict'].get('imgsz') or c['train']['imgsz'])")
UP=$(( (IMGSZ * 5 / 4 + 16) / 32 * 32 ))
run() {  # tag, overrides...
  local tag="$1"; shift
  [ -f "results/$ID/eval_$tag/metrics.json" ] && return
  $PY evaluate.py --config "$CFG" --tag "$tag" "$@" 2>&1 | grep -E "^\[eval\]|Error|Traceback"
}
for size in 640 800 1024; do
  for ov in 0.15 0.2 0.25 0.3; do
    run "sl${size}_o${ov#0.}" "predict.slice={size: $size, overlap: $ov, imgsz: $IMGSZ}"
  done
done
# best window from the grid, plus the full image at 1.25x
best=$($PY - "$ID" <<'PY'
import json, glob, sys, os
rows = [(json.load(open(f))["map"], os.path.basename(os.path.dirname(f))[5:]) for f in glob.glob(f"results/{sys.argv[1]}/eval_sl*_o*/metrics.json")]
print(max(rows)[1])
PY
)
size=${best#sl}; size=${size%%_*}; ov="0.${best##*_o}"
echo "[sweep] best window: $best (size $size, overlap $ov)"
run "${best}_up125" "predict.imgsz=$UP" "predict.slice={size: $size, overlap: $ov, imgsz: $IMGSZ}"
echo "[sweep] done"
