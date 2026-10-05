"""Inference sweep without training: score inference variants on the best finished runs.

python scripts/tune_postprocess.py --top 2          # the 2 best runs by default-inference val mAP
python scripts/tune_postprocess.py --ids s1_03_yolo11s_1280 --test-best
Each variant is written to results/<id>/eval_<tag>/. With --test-best the best variant of each run
(if it beats the default) also gets a test CSV.
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from utils.config import ROOT
from utils.logger import read_json


def variants(imgsz):
    up = int(round(imgsz * 1.25 / 32) * 32)
    tile = {"size": 800, "overlap": 0.2, "imgsz": imgsz}
    sl = "predict.slice=" + yaml.safe_dump(tile, default_flow_style=True).strip()
    return {
        "iou50": ["predict.iou=0.5"],
        "iou70": ["predict.iou=0.7"],
        "up125": [f"predict.imgsz={up}"],
        "hflip": ["predict.hflip=true"],
        "hflip_wbf": ["predict.hflip=true", "predict.merge=wbf"],
        "slice800": [sl],
        "slice800_hflip": [sl, "predict.hflip=true"],
        "ms_hflip": ["predict.scales=[1.0, 1.25]", "predict.hflip=true"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--top", type=int, default=2)
    ap.add_argument("--test-best", action="store_true")
    a = ap.parse_args()
    res = ROOT / "results"
    ids = a.ids
    if not ids:
        scored = [(read_json(d / "metrics.json")["map"], d.name) for d in res.iterdir() if (d / "metrics.json").exists() and (d / "DONE").exists()]
        ids = [n for _, n in sorted(scored, reverse=True)[: a.top]]
    py = sys.executable
    for i in ids:
        cfg_path = res / i / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        imgsz = cfg["predict"].get("imgsz") or cfg["train"]["imgsz"]
        for tag, ov in variants(imgsz).items():
            if (res / i / f"eval_{tag}" / "metrics.json").exists():
                continue
            subprocess.run([py, "evaluate.py", "--config", str(cfg_path), "--tag", tag, *ov], cwd=ROOT)
        if a.test_best:
            base = read_json(res / i / "metrics.json")["map"]
            got = {p.parent.name[5:]: read_json(p)["map"] for p in (res / i).glob("eval_*/metrics.json")}
            if got and max(got.values()) > base:
                tag = max(got, key=got.get)
                if not (res / i / f"eval_{tag}" / "test_submission.csv").exists():
                    subprocess.run([py, "test.py", "--config", str(cfg_path), "--tag", tag], cwd=ROOT)


if __name__ == "__main__":
    main()
