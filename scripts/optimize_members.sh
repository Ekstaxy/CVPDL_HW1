#!/usr/bin/env bash
# Inference-only tuning of ensemble members, then the final ensemble. No training.
# For each member: slice window 512 / 640 (overlap 0.2; overlap made no difference on D-FINE-L 1536),
# then the better window plus the full image at 1.25x. YOLO members also use hflip (it helped them with
# slicing; it did not help D-FINE). The best variant of each member gets a test CSV, then the 4-model WBF.
# usage: scripts/optimize_members.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
FIXED=s3_01_dfine_l_1536/sl512_o2_up125      # already tuned
MEMBERS="s2_05_dfine_l_1280:0 s2_04_yolo11m_1536_aug:1 s2_02_yolo11l_1280_aug:1"   # id:hflip

ev() {  # id tag overrides...
  local id="$1" tag="$2"; shift 2
  [ -f "results/$id/eval_$tag/metrics.json" ] && return
  $PY evaluate.py --config "results/$id/config.yaml" --tag "$tag" "$@" 2>&1 | grep -E "^\[eval\]|Traceback|OutOfMemory"
}
map() { $PY -c "import json; print(json.load(open('results/$1/eval_$2/metrics.json'))['map'])" 2>/dev/null || echo 0; }

chosen=("$FIXED")
for m in $MEMBERS; do
  id=${m%%:*}; flip=${m##*:}
  imgsz=$($PY -c "import yaml; c=yaml.safe_load(open('results/$id/config.yaml')); print(c['predict'].get('imgsz') or c['train']['imgsz'])")
  up=$(( (imgsz * 5 / 4 + 16) / 32 * 32 ))
  hf=(); sfx=""; [ "$flip" = 1 ] && { hf=("predict.hflip=true"); sfx="_hflip"; }
  for size in 512 640; do
    ev "$id" "sl${size}_o2${sfx}" "predict.slice={size: $size, overlap: 0.2, imgsz: $imgsz}" "${hf[@]}"
  done
  a=$(map "$id" "sl512_o2${sfx}"); b=$(map "$id" "sl640_o2${sfx}")
  size=$($PY -c "print(512 if $a >= $b else 640)")
  ev "$id" "sl${size}_o2${sfx}_up125" "predict.imgsz=$up" "predict.slice={size: $size, overlap: 0.2, imgsz: $imgsz}" "${hf[@]}"
  # best of: the old member variant and the three new ones
  best=$($PY - "$id" <<'PY'
import json, glob, os, sys
rid = sys.argv[1]
old = {"s2_05_dfine_l_1280": "slice800"}.get(rid, "slice800_hflip")
tags = [old] + [os.path.basename(os.path.dirname(f))[5:] for f in glob.glob(f"results/{rid}/eval_sl*_o2*/metrics.json")]
print(max(tags, key=lambda t: json.load(open(f"results/{rid}/eval_{t}/metrics.json"))["map"]))
PY
)
  echo "[opt] $(date +%H:%M) $id: best $best ($(map "$id" "$best"))"
  [ -f "results/$id/eval_$best/test_submission.csv" ] || $PY test.py --config "results/$id/config.yaml" --tag "$best" 2>&1 | grep -E "^\[test\]|Traceback"
  chosen+=("$id/$best")
done
echo "[opt] $(date +%H:%M) members: ${chosen[*]}"
$PY -u scripts/ensemble.py --name wbf4_opt --members "${chosen[@]}" --min-members 4 --iou 0.65 0.7 0.75 \
  --weights 1 1 1 1 --conf-type avg --workers 16 2>&1 | grep -E "^\s+0\.[0-9]{4}|best val|wrote|Traceback"
echo "[opt] $(date +%H:%M) all done"
