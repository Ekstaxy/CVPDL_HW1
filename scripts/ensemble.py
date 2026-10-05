"""Weighted-boxes-fusion ensemble of prediction CSVs from several runs.

python scripts/ensemble.py --name ens_a --runs s1_09_yolo11m_1280 s1_10_dfine_s_960/eval_hflip --weights 1 1
Each run entry is a folder under results/ holding val_pred.csv (and test_submission.csv).
Writes results/<name>/{val_pred.csv, metrics.json, test_submission.csv}.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import ROOT
from utils.logger import write_json
from utils.metrics import finalize_dets, load_gt, read_csv, score, validate_csv, write_csv
from utils.postprocess import wbf


def fuse(csvs, weights, iou):
    preds = [read_csv(c) for c in csvs]
    ids = list(preds[0])
    # sizes are not needed for clipping here: every input CSV is already clipped
    return ids, {i: finalize_dets(wbf([p[i] for p in preds], weights, iou), 1e9, 1e9) for i in ids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float)
    ap.add_argument("--iou", type=float, default=0.65)
    ap.add_argument("--data", default="dataset")
    a = ap.parse_args()
    res, data = ROOT / "results", ROOT / a.data
    out = res / a.name
    out.mkdir(exist_ok=True)
    weights = a.weights or [1.0] * len(a.runs)

    ids, gt, _, names = load_gt(data / "annotations" / "instances_val.json")
    _, dets = fuse([res / r / "val_pred.csv" for r in a.runs], weights, a.iou)
    write_csv(out / "val_pred.csv", ids, dets)
    m = score(read_csv(out / "val_pred.csv"), gt, names)
    m.update(runs=a.runs, weights=weights, iou=a.iou)
    write_json(m, out / "metrics.json")
    print(f"[ensemble] {a.name}: val mAP50:95 {m['map']:.4f}  mAP50 {m['map_50']:.4f}")
    for r in a.runs:
        print(f"    member {r}")

    tests = [res / r / "test_submission.csv" for r in a.runs]
    if all(t.exists() for t in tests):
        tids, tdets = fuse(tests, weights, a.iou)
        write_csv(out / "test_submission.csv", tids, tdets)
        print("[ensemble] test:", validate_csv(out / "test_submission.csv", data / "sample_submission.csv"))
    else:
        print("[ensemble] no test CSV written: a member has no test_submission.csv")


if __name__ == "__main__":
    main()
