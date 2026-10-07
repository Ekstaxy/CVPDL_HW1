"""Kaggle CSV io and the TA's scoring function.

Detections are numpy arrays (n, 6): x1, y1, x2, y2, score, class_id in original-image pixels.
The CSV stores `class_id confidence x_min y_min width height` per detection.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision

EMPTY = np.zeros((0, 6), dtype=np.float32)


def finalize_dets(dets, width, height, per_class_topk=100, max_det=1000):
    """Clip to the image, drop degenerate boxes, keep the top-k scores per class (metric maxDets)."""
    if len(dets) == 0:
        return EMPTY
    d = np.asarray(dets, dtype=np.float32).copy()
    d[:, [0, 2]] = d[:, [0, 2]].clip(0, width)
    d[:, [1, 3]] = d[:, [1, 3]].clip(0, height)
    d = d[(d[:, 2] - d[:, 0] > 0.01) & (d[:, 3] - d[:, 1] > 0.01)]
    d = d[np.argsort(-d[:, 4], kind="stable")]
    if per_class_topk:
        keep = np.zeros(len(d), dtype=bool)
        for c in np.unique(d[:, 5]):
            keep[np.flatnonzero(d[:, 5] == c)[:per_class_topk]] = True
        d = d[keep]
    return d[:max_det] if max_det else d


def write_csv(path, image_ids, dets_by_id):
    """One row per image id, in the given order; `none` for images without detections."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "PredictionString"])
        for iid in image_ids:
            d = dets_by_id.get(iid, EMPTY)
            parts = [
                f"{int(c)} {s:.5f} {x1:.2f} {y1:.2f} {x2 - x1:.2f} {y2 - y1:.2f}"
                for x1, y1, x2, y2, s, c in d
            ]
            w.writerow([iid, " ".join(parts) if parts else "none"])


def read_csv(path):
    """-> {image_id: (n, 6) array of x1, y1, x2, y2, score, class_id}."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            s = row["PredictionString"].strip()
            if not s or s.lower() == "none":
                out[row["image_id"]] = EMPTY
                continue
            v = np.array(s.split(), dtype=np.float32).reshape(-1, 6)
            x2, y2 = v[:, 2] + v[:, 4], v[:, 3] + v[:, 5]
            out[row["image_id"]] = np.stack([v[:, 2], v[:, 3], x2, y2, v[:, 1], v[:, 0]], 1)
    return out


def load_gt(ann_json):
    """COCO json -> (image ids in file order, {id: (n, 5) x, y, w, h, class}, {id: (width, height)}, class names)."""
    d = json.loads(Path(ann_json).read_text())
    ids = [im["id"] for im in d["images"]]
    sizes = {im["id"]: (im["width"], im["height"]) for im in d["images"]}
    boxes = {i: [] for i in ids}
    for a in d["annotations"]:
        boxes[a["image_id"]].append([*a["bbox"], a["category_id"]])
    gt = {i: np.array(b, dtype=np.float32).reshape(-1, 5) for i, b in boxes.items()}
    names = [c["name"] for c in sorted(d["categories"], key=lambda c: c["id"])]
    return ids, gt, sizes, names


def score(dets_by_id, gt, names):
    """mAP exactly as the TA computes it: torchmetrics defaults, xywh boxes, per-class metrics."""
    metric = MeanAveragePrecision(box_format="xywh", iou_type="bbox", class_metrics=True)
    preds, targets = [], []
    for iid, g in gt.items():
        d = dets_by_id.get(iid, EMPTY)
        xywh = np.stack([d[:, 0], d[:, 1], d[:, 2] - d[:, 0], d[:, 3] - d[:, 1]], 1) if len(d) else np.zeros((0, 4))
        preds.append(
            {
                "boxes": torch.tensor(xywh, dtype=torch.float32).reshape(-1, 4),
                "scores": torch.tensor(d[:, 4], dtype=torch.float32),
                "labels": torch.tensor(d[:, 5], dtype=torch.int64),
            }
        )
        targets.append(
            {"boxes": torch.tensor(g[:, :4], dtype=torch.float32), "labels": torch.tensor(g[:, 4], dtype=torch.int64)}
        )
    metric.update(preds, targets)
    r = metric.compute()
    out = {k: float(r[k]) for k in ["map", "map_50", "map_75", "map_small", "map_medium", "map_large", "mar_100"]}
    classes = r["classes"].tolist()
    per_class = r["map_per_class"].tolist()
    if not isinstance(classes, list):
        classes, per_class = [classes], [per_class]
    out["per_class"] = {names[int(c)]: float(v) for c, v in zip(classes, per_class)}
    return out


def validate_csv(path, sample_csv, num_classes=10):
    """Check a submission against sample_submission.csv: same ids and order, well-formed detections."""
    with open(sample_csv, newline="") as f:
        want = [r["image_id"] for r in csv.DictReader(f)]
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    got = [r["image_id"] for r in rows]
    assert got == want, f"image ids differ from sample ({len(got)} rows vs {len(want)})"
    n = 0
    for r in rows:
        s = r["PredictionString"].strip()
        if s == "none":
            continue
        v = s.split()
        assert len(v) % 6 == 0 and len(v) > 0, f"{r['image_id']}: not a multiple of 6 values"
        a = np.array(v, dtype=np.float64).reshape(-1, 6)
        assert np.isfinite(a).all(), f"{r['image_id']}: non-finite value"
        assert ((a[:, 0] >= 0) & (a[:, 0] < num_classes) & (a[:, 0] % 1 == 0)).all(), f"{r['image_id']}: class id"
        assert ((a[:, 1] >= 0) & (a[:, 1] <= 1)).all(), f"{r['image_id']}: confidence"
        assert (a[:, 4:] > 0).all() and (a[:, 2:4] >= 0).all(), f"{r['image_id']}: box"
        n += len(a)
    return {"rows": len(rows), "detections": n}

