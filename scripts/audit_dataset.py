"""Dataset audit: class balance, box sizes, objects per image, brightness, sample images with GT.

python scripts/audit_dataset.py    -> results/_audit/{audit.json, *.png, samples/}
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.config import ROOT
from utils.metrics import load_gt
from utils.visualize import draw_dets

DATA = ROOT / "dataset"
OUT = ROOT / "results" / "_audit"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for row, split in enumerate(["train", "val"]):
        ids, gt, sizes, names = load_gt(DATA / "annotations" / f"instances_{split}.json")
        allb = np.concatenate(list(gt.values()))
        area = allb[:, 2] * allb[:, 3]
        per_img = np.array([len(g) for g in gt.values()])
        inst = Counter(allb[:, 4].astype(int).tolist())
        imgs = Counter(c for g in gt.values() for c in set(g[:, 4].astype(int).tolist()))
        rng = random.Random(0)
        sample = rng.sample(ids, 300)
        bright = np.array([cv2.imread(str(DATA / "images" / split / f"{i}.jpg"), cv2.IMREAD_REDUCED_GRAYSCALE_4).mean() for i in sample])
        report[split] = {
            "images": len(ids),
            "boxes": int(len(allb)),
            "instances_per_class": {names[c]: inst[c] for c in range(len(names))},
            "images_per_class": {names[c]: imgs[c] for c in range(len(names))},
            "image_sizes": {f"{w}x{h}": n for (w, h), n in Counter(sizes.values()).most_common(8)},
            "frac_small_lt32": float((area < 32**2).mean()),
            "frac_medium": float(((area >= 32**2) & (area < 96**2)).mean()),
            "frac_large_ge96": float((area >= 96**2).mean()),
            "median_box_side_px": float(np.median(np.sqrt(area))),
            "objects_per_image": {"median": float(np.median(per_img)), "p95": float(np.percentile(per_img, 95)), "max": int(per_img.max())},
            "frac_night_mean_brightness_lt60": float((bright < 60).mean()),
        }
        ax = axes[row]
        ax[0].bar(names, [inst[c] for c in range(len(names))], color="#4477aa")
        ax[0].set_title(f"{split}: instances per class")
        ax[0].tick_params(axis="x", rotation=35)
        ax[1].hist(np.sqrt(area).clip(0, 200), bins=80, color="#4477aa")
        for v in (32, 96):
            ax[1].axvline(v, color="#cc3311", ls="--", lw=1)
        ax[1].set_title(f"{split}: sqrt(box area) in px (COCO small < 32, large ≥ 96)")
        ax[2].hist(per_img, bins=60, color="#4477aa")
        ax[2].set_title(f"{split}: objects per image")
        if split == "train":
            (OUT / "samples").mkdir(exist_ok=True)
            for i in rng.sample(ids, 16):
                g = gt[i]
                g6 = np.stack([g[:, 0], g[:, 1], g[:, 0] + g[:, 2], g[:, 1] + g[:, 3], np.ones(len(g)), g[:, 4]], 1)
                img = draw_dets(cv2.imread(str(DATA / "images" / split / f"{i}.jpg")), g6, names, 0, show_score=False)
                cv2.imwrite(str(OUT / "samples" / f"{i}.jpg"), img)
    fig.tight_layout()
    fig.savefig(OUT / "audit.png", dpi=120)
    (OUT / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
