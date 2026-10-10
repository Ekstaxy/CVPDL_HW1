#!/usr/bin/env bash
# Reproduce the final submission (6-model WBF ensemble, val mAP50:95 0.4354, Kaggle public 0.38642).
#
#   scripts/reproduce_final.sh            # inference only, from results/<id>/weights/  (about 2 h on one RTX 4090)
#   scripts/reproduce_final.sh --train    # train the six models first                   (about 52 h)
#
# Output: submission_test.csv (repo root), plus results/_ensemble/<name>/ with the val CSV and its score.
# Set TAG_SUFFIX (e.g. TAG_SUFFIX=_repro) to write the member outputs next to the existing ones instead of over them.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
SFX="${TAG_SUFFIX:-}"
NAME="final${SFX}"

if [ "${1:-}" = "--train" ]; then
  # order matters: s3_03 continues from s3_01's best.pt
  for id in s2_02_yolo11l_1280_aug s2_04_yolo11m_1536_aug s2_05_dfine_l_1280 \
            s3_01_dfine_l_1536 s3_02_dfine_l_1536_short s3_03_dfine_l_1536_tileft; do
    scripts/run_experiment.sh "$id"
  done
fi

# member: run id | inference tag | weights file | inference overrides
# D-FINE: full image at 1.25x the train size + 512 px tiles (20 % overlap) at the train size, merged by NMS.
# YOLO:   the same with h-flip; YOLO11m keeps 800 px tiles (smaller tiles were worse for it).
MEMBERS=(
  "s3_03_dfine_l_1536_tileft|sl512_o2_up125_last|last.pt|predict.imgsz=1920|predict.slice={size: 512, overlap: 0.2, imgsz: 1536}"
  "s3_02_dfine_l_1536_short|sl512_o2_up125|best.pt|predict.imgsz=1920|predict.slice={size: 512, overlap: 0.2, imgsz: 1536}"
  "s3_01_dfine_l_1536|sl512_o2_up125|best.pt|predict.imgsz=1920|predict.slice={size: 512, overlap: 0.2, imgsz: 1536}"
  "s2_05_dfine_l_1280|sl512_o2_up125|best.pt|predict.imgsz=1600|predict.slice={size: 512, overlap: 0.2, imgsz: 1280}"
  "s2_04_yolo11m_1536_aug|slice800_hflip|best.pt|predict.hflip=true|predict.slice={size: 800, overlap: 0.2, imgsz: 1536}"
  "s2_02_yolo11l_1280_aug|sl640_o2_hflip_up125|best.pt|predict.imgsz=1600|predict.hflip=true|predict.slice={size: 640, overlap: 0.2, imgsz: 1280}"
)

refs=()
for m in "${MEMBERS[@]}"; do
  IFS='|' read -r id tag wfile ov1 ov2 ov3 <<< "$m"
  ov=("$ov1"); [ -n "${ov2:-}" ] && ov+=("$ov2"); [ -n "${ov3:-}" ] && ov+=("$ov3")
  cfg="configs/$id.yaml"; w="results/$id/weights/$wfile"
  echo "== $id / $tag$SFX"
  # val CSV + score (also writes eval_<tag>/predict.yaml, which test.py reuses), then the test CSV
  $PY evaluate.py --config "$cfg" --weights "$w" --tag "$tag$SFX" "${ov[@]}"
  $PY test.py     --config "$cfg" --weights "$w" --tag "$tag$SFX"
  refs+=("$id/$tag$SFX")
done

$PY scripts/ensemble.py --name "$NAME" --members "${refs[@]}" --min-members 6 \
  --iou 0.7 --weights 1 1 1 1 1 1 --conf-type avg
cp "results/_ensemble/$NAME/test_submission.csv" submission_test.csv
echo "wrote submission_test.csv (val mAP50:95 in results/_ensemble/$NAME/metrics.json)"
