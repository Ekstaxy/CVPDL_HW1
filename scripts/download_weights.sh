#!/usr/bin/env bash
# Download the trained weights of the six final-ensemble members from Google Drive (gdown) into
# results/<id>/weights/, where scripts/reproduce_final.sh reads them. About 2.1 GB in total.
# Files already present with the right MD5 are skipped; a file with a wrong MD5 is downloaded again.
#
#   scripts/download_weights.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

# run id | weights file | Google Drive file id | md5
WEIGHTS=(
  "s3_03_dfine_l_1536_tileft|last.pt|1PYwYhCB_BvMDKx_CDOpxZVeSnPASnIBh|42a07d99fc3fc0217799956be5a6200c"
  "s3_02_dfine_l_1536_short|best.pt|1PHu2ek3s_j1GK8yMJFnLJ8illlg1BiNx|d57b3e3fb272b3fcc54876e95794d793"
  "s3_01_dfine_l_1536|best.pt|1j3_mWtVjAtCcHLUuNP42CMQ3-iJ3bTj9|cbe70f96b18984ffdd5963c5fdf99fa2"
  "s2_05_dfine_l_1280|best.pt|1eI89xkhjtRPn4ocdi8i_yvyOGfDyjHYm|cc12fa53f7fd0e08a41799133015985c"
  "s2_02_yolo11l_1280_aug|best.pt|1WPeL91INgxTuWm28q1c8tn9u_N1gGEb7|9f216ab68f0d276dc3a3ce789f0b1aba"
  "s2_04_yolo11m_1536_aug|best.pt|1j0xp4lIPH1wKU5Gjb2b7zpptojae2x63|aa5e63ab042d13c01302127c42b4cbee"
)

for w in "${WEIGHTS[@]}"; do
  IFS='|' read -r id file gid md5 <<< "$w"
  out="results/$id/weights/$file"
  if [ -f "$out" ] && [ "$(md5sum "$out" | cut -d' ' -f1)" = "$md5" ]; then
    echo "== $out: already present"
    continue
  fi
  mkdir -p "$(dirname "$out")"
  echo "== $out"
  $PY -c "import sys, gdown; gdown.download(id=sys.argv[1], output=sys.argv[2], quiet=False)" "$gid" "$out"
  [ "$(md5sum "$out" | cut -d' ' -f1)" = "$md5" ] || { echo "MD5 mismatch for $out"; exit 1; }
done
echo "all six weights in results/<id>/weights/"
