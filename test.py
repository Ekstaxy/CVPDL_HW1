"""Write the Kaggle submission for the 1,800 test images.

python test.py --config configs/<id>.yaml            # uses results/<id>/predict.yaml written by evaluate.py
python test.py --config configs/<id>.yaml --tag hflip # uses results/<id>/eval_hflip/predict.yaml
"""
import argparse
import csv

import yaml

from models import build_adapter
from utils.config import load_config, run_dir
from utils.data import class_names, data_root
from utils.logger import cap_vram
from utils.metrics import validate_csv, write_csv
from utils.postprocess import predict_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    root = run_dir(cfg)
    out = root / f"eval_{a.tag}" if a.tag else root
    saved = out / "predict.yaml"  # the exact settings that were scored on val
    pcfg = yaml.safe_load(saved.read_text()) if saved.exists() else cfg["predict"]
    imgsz = pcfg.get("imgsz") or cfg["train"]["imgsz"]

    cap_vram(cfg.get("vram_cap_gb"))
    sample = data_root(cfg) / "sample_submission.csv"
    with open(sample, newline="") as f:
        ids = [r["image_id"] for r in csv.DictReader(f)]
    items = [(i, data_root(cfg) / "images" / "test" / f"{i}.jpg") for i in ids]
    adapter = build_adapter(cfg).load(a.weights or root / "weights" / "best.pt")
    dets = predict_paths(adapter, items, pcfg, imgsz, pcfg.get("per_class_topk", 100), desc="test")
    path = out / "test_submission.csv"
    write_csv(path, ids, dets)
    info = validate_csv(path, sample, len(class_names(cfg)))
    print(f"[test] wrote {path}: {info['rows']} rows, {info['detections']} detections, format OK")


if __name__ == "__main__":
    main()
