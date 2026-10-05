"""Draw predicted boxes and labels from a prediction CSV onto images (report Q3 and debugging).

python scripts/visualize_predictions.py --csv results/<id>/val_pred.csv --split val --num 8
python scripts/visualize_predictions.py --csv results/<id>/test_submission.csv --split test --ids img_xxx img_yyy
python scripts/visualize_predictions.py --csv ... --split val --gt     # also save a ground-truth panel
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from utils.config import ROOT
from utils.metrics import load_gt, read_csv
from utils.visualize import draw_dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--gt", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--data", default="dataset")
    a = ap.parse_args()
    data = ROOT / a.data
    names = (data / "classes.txt").read_text().split()
    dets = read_csv(a.csv)
    ids = a.ids or random.Random(0).sample(sorted(dets), min(a.num, len(dets)))
    out = Path(a.out) if a.out else Path(a.csv).parent / "plots" / f"vis_{a.split}"
    out.mkdir(parents=True, exist_ok=True)
    gt = load_gt(data / "annotations" / f"instances_{a.split}.json")[1] if a.gt and a.split != "test" else None
    for i in ids:
        img = cv2.imread(str(data / "images" / a.split / f"{i}.jpg"))
        vis = draw_dets(img, dets[i], names, a.conf)
        if gt is not None:
            g = gt[i]
            g6 = np.stack([g[:, 0], g[:, 1], g[:, 0] + g[:, 2], g[:, 1] + g[:, 3], np.ones(len(g)), g[:, 4]], 1)
            vis = np.concatenate([vis, draw_dets(img, g6, names, 0, show_score=False)], 1)
        cv2.imwrite(str(out / f"{i}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"saved {len(ids)} images to {out}")


if __name__ == "__main__":
    main()
