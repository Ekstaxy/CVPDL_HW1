"""Class-wise WBF weights for a fixed set of ensemble members, with an honest 2-fold estimate on val.

Weights are not searched freely (that would fit val). For member m and class c:
    w[m][c] = (AP[m][c] / max_m AP[m][c]) ** p
from each member's own per-class AP, so the only free knob is the sharpness p (p = 0 is equal weights).
To see whether this transfers, val is split into two halves: weights from one half score the other half,
and the two halves are averaged. The final weights use all of val, for the chosen p.

python scripts/ensemble_classwise.py --name cw4 --members s3_01_dfine_l_1536/slice800 s2_05_dfine_l_1280/slice800 \
    s2_04_yolo11m_1536_aug/slice800_hflip s2_02_yolo11l_1280_aug/slice800_hflip
"""
import argparse
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ensemble import DATA, RES, _init, fuse, member_dir
from utils.logger import read_json
from utils.metrics import load_gt, read_csv, score, validate_csv, write_csv


def per_class_ap(dets, gt, names, ids):
    m = score({i: dets[i] for i in ids if i in dets}, {i: gt[i] for i in ids}, names)
    return np.array([m["per_class"].get(n, 0.0) for n in names])


def weight_matrix(ap, p):
    """ap: (members, classes) -> weights (members, classes), best member of each class = 1."""
    rel = ap / np.maximum(ap.max(0, keepdims=True), 1e-6)
    return (rel**p).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--name", default="classwise")
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--conf-type", default="avg")
    ap.add_argument("--powers", type=float, nargs="*", default=[0, 1, 2, 4, 8])
    ap.add_argument("--per-class-topk", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    dirs = [member_dir(m) for m in a.members]
    val_ids, gt, val_sizes, names = load_gt(DATA / "annotations" / "instances_val.json")
    val_dets = [read_csv(d / "val_pred.csv") for d in dirs]
    idx = list(range(len(a.members)))
    folds = [sorted(val_ids)[0::2], sorted(val_ids)[1::2]]

    print("[classwise] member per-class AP on each val half")
    ap_fold = [np.stack([per_class_ap(d, gt, names, f) for d in val_dets]) for f in folds]
    ap_full = np.stack([np.array([read_json(d / "metrics.json")["per_class"][n] for n in names]) for d in dirs])
    print("    " + " ".join(f"{n[:7]:>7s}" for n in names))
    for m, row in zip(a.members, ap_full):
        print("    " + " ".join(f"{v:7.3f}" for v in row) + f"  {m}")

    rows = []
    with Pool(a.workers, initializer=_init, initargs=(val_dets, val_sizes, a.per_class_topk)) as pool:
        for p in a.powers:
            cross = []
            for k in (0, 1):  # weights from half k, scored on the other half
                w = weight_matrix(ap_fold[k], p)
                dets = fuse(pool, val_ids, idx, w, a.iou, a.conf_type)
                other = folds[1 - k]
                cross.append(score({i: dets[i] for i in other}, {i: gt[i] for i in other}, names)["map"])
            w_full = weight_matrix(ap_full, p)
            full = score(fuse(pool, val_ids, idx, w_full, a.iou, a.conf_type), gt, names)["map"]
            rows.append(dict(power=p, cross_half_a=cross[1], cross_half_b=cross[0], cross_mean=float(np.mean(cross)), full_val=full))
            print(f"    p {p:<4g} cross-fold {np.mean(cross):.4f} (halves {cross[1]:.4f} / {cross[0]:.4f})   full val {full:.4f}")

    base = next(r for r in rows if r["power"] == 0)
    best = max(rows, key=lambda r: r["cross_mean"])
    print(f"\n[classwise] equal weights cross-fold {base['cross_mean']:.4f}; best p {best['power']:g} cross-fold "
          f"{best['cross_mean']:.4f} ({best['cross_mean'] - base['cross_mean']:+.4f}), full val {best['full_val']:.4f}")

    out = RES / "_ensemble" / a.name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "grid.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    w_full = weight_matrix(ap_full, best["power"])
    with Pool(a.workers, initializer=_init, initargs=(val_dets, val_sizes, a.per_class_topk)) as pool:
        write_csv(out / "val_pred.csv", val_ids, fuse(pool, val_ids, idx, w_full, a.iou, a.conf_type))
    m_disk = score(read_csv(out / "val_pred.csv"), gt, names)

    sample = DATA / "sample_submission.csv"
    with open(sample, newline="") as f:
        test_ids = [r["image_id"] for r in csv.DictReader(f)]
    test_sizes = {i: Image.open(DATA / "images" / "test" / f"{i}.jpg").size for i in test_ids}
    test_dets = [read_csv(d / "test_submission.csv") for d in dirs]
    with Pool(a.workers, initializer=_init, initargs=(test_dets, test_sizes, a.per_class_topk)) as pool:
        write_csv(out / "test_submission.csv", test_ids, fuse(pool, test_ids, idx, w_full, a.iou, a.conf_type))
    info = validate_csv(out / "test_submission.csv", sample, len(names))

    cfg = dict(members=a.members, iou=a.iou, conf_type=a.conf_type, power=best["power"],
               cross_fold_map=best["cross_mean"], cross_fold_map_equal=base["cross_mean"],
               class_weights={n: [round(w_full[m][c], 4) for m in idx] for c, n in enumerate(names)},
               per_class_topk=a.per_class_topk)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    m_disk.update(exp_id=f"_ensemble/{a.name}", **{k: v for k, v in cfg.items() if k != "class_weights"})
    (out / "metrics.json").write_text(json.dumps(m_disk, indent=2))
    print(f"[classwise] wrote {out}: val {m_disk['map']:.4f}, test {info['rows']} rows / {info['detections']} dets, format OK")


if __name__ == "__main__":
    main()
