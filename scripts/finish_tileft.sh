#!/usr/bin/env bash
# After s3_03 (slicing-aided fine-tuning): score sliced inference with best.pt and last.pt (the trainer picks
# best.pt on full-image val, which is not how we infer), keep the better one, make its test CSV, rebuild the ensemble.
set -uo pipefail
cd "$(dirname "$0")/.."
ID=s3_03_dfine_l_1536_tileft; CFG=results/$ID/config.yaml; PY=.venv/bin/python
W=results/$ID/weights
ev() { local tag="$1"; shift; [ -f results/$ID/eval_$tag/metrics.json ] || \
  $PY evaluate.py --config $CFG --tag $tag "$@" 2>&1 | grep -E "^\[eval\]|Traceback|OutOfMemory"; }
ev sl512_o2_up125      "predict.imgsz=1920" "predict.slice={size: 512, overlap: 0.2, imgsz: 1536}"
ev sl640_o2_up125      "predict.imgsz=1920" "predict.slice={size: 640, overlap: 0.2, imgsz: 1536}"
ev sl512_o2_up125_last --weights $W/last.pt "predict.imgsz=1920" "predict.slice={size: 512, overlap: 0.2, imgsz: 1536}"
best=$($PY -c "
import json,glob,os
r=[(json.load(open(f))['map'],os.path.basename(os.path.dirname(f))[5:]) for f in glob.glob('results/$ID/eval_sl*/metrics.json')]
print(max(r)[1])")
echo "[tileft] $(date +%H:%M) best variant: $best"
case "$best" in *_last) $PY test.py --config $CFG --tag $best --weights $W/last.pt 2>&1 | grep -E "^\[test\]|Traceback";;
                *)      $PY test.py --config $CFG --tag $best 2>&1 | grep -E "^\[test\]|Traceback";; esac
$PY -u scripts/ensemble.py --name wbf_tileft --members $ID/$best s3_02_dfine_l_1536_short/sl512_o2_up125 \
  s3_01_dfine_l_1536/sl512_o2_up125 s2_05_dfine_l_1280/sl512_o2_up125 s2_04_yolo11m_1536_aug/slice800_hflip \
  s2_02_yolo11l_1280_aug/sl640_o2_hflip_up125 --min-members 5 --iou 0.7 --weights 1 1 1 1 1 1 --conf-type avg \
  --workers 16 2>&1 | grep -E "^\s+0\.[0-9]{4}|best val|wrote|Traceback"
echo "[tileft] $(date +%H:%M) all done"
