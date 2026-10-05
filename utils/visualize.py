"""Drawing: predicted boxes on images, per-class AP bars, training curves."""
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# one BGR colour per class, readable on aerial imagery
COLORS = [
    (60, 60, 255), (60, 160, 255), (0, 220, 255), (80, 220, 80), (220, 200, 0),
    (255, 120, 40), (255, 60, 160), (200, 60, 255), (140, 255, 200), (255, 255, 255),
]


def draw_dets(img, dets, names, min_conf=0.3, show_score=True):
    """Draw (n, 6) x1, y1, x2, y2, score, class boxes with class name and score."""
    img = img.copy()
    t = max(1, round(min(img.shape[:2]) / 600))
    fs = 0.35 * t + 0.1
    for x1, y1, x2, y2, s, c in sorted(dets.tolist(), key=lambda d: d[4]):
        if s < min_conf:
            continue
        col = COLORS[int(c) % len(COLORS)]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, col, t)
        label = f"{names[int(c)]} {s:.2f}" if show_score else names[int(c)]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        y = max(p1[1], th + 3)
        cv2.rectangle(img, (p1[0], y - th - 3), (p1[0] + tw, y), col, -1)
        cv2.putText(img, label, (p1[0], y - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def plot_per_class_ap(per_class, title, path):
    names, vals = list(per_class), list(per_class.values())
    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.bar(names, vals, color="#4477aa")
    ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_ylabel("AP@[.5:.95]")
    ax.set_title(title)
    ax.set_ylim(0, max(vals + [0.1]) * 1.15)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_training_curves(results_csv, path):
    """Loss and validation curves from an Ultralytics results.csv."""
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    groups = {
        "train loss": [c for c in df.columns if c.startswith("train/")],
        "val loss": [c for c in df.columns if c.startswith("val/")],
        "val metrics": [c for c in df.columns if c.startswith("metrics/")],
        "learning rate": [c for c in df.columns if c.startswith("lr/")],
    }
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    for ax, (title, cols) in zip(axes, groups.items()):
        for c in cols:
            ax.plot(df["epoch"], df[c], label=c.split("/", 1)[1])
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_examples(image_paths, dets_by_id, names, out_dir, min_conf=0.3):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for iid, p in image_paths:
        img = draw_dets(cv2.imread(str(p)), dets_by_id[iid], names, min_conf)
        cv2.imwrite(str(out_dir / f"{iid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
