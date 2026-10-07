"""Multi-model ensemble with weighted boxes fusion, tuned on val, written for test.

Members are finished inference outputs (results/<id>/[eval_<tag>/]val_pred.csv + test_submission.csv),
so this needs no GPU. The grid (member subsets x WBF IoU x weights x conf_type) is scored on val with the
TA's metric; the best setting is applied to the members' test CSVs.

python scripts/ensemble.py                                   # default members = best variant of the top runs
python scripts/ensemble.py --members s2_05_dfine_l_1280/slice800 s2_04_yolo11m_1536_aug/slice800_hflip
python scripts/ensemble.py --name final --iou 0.55 --weights 2 1 1 --conf-type avg   # one setting, no grid

Outputs: results/_ensemble/<name>/{config.yaml, metrics.json, val_pred.csv, test_submission.csv, grid.csv}
"""
import argparse
import csv
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import ROOT
from utils.logger import read_json
from utils.metrics import EMPTY, finalize_dets, load_gt, read_csv, score, validate_csv, write_csv
from utils.postprocess import wbf

RES = ROOT / "results"
DATA = ROOT / "dataset"


def member_dir(name):
    """'<id>' or '<id>/<tag>' -> results dir holding val_pred.csv and test_submission.csv."""
    run, _, tag = name.partition("/")
    return RES / run / f"eval_{tag}" if tag else RES / run


def best_variants(top):
    """Best (by val mAP) inference variant of each finished run, top runs first."""
    out = []
    for d in RES.iterdir():
        if not (d / "metrics.json").exists() or not (d / "DONE").exists():
            continue
        cands = [(read_json(d / "metrics.json")["map"], d.name)]
        cands += [(read_json(p)["map"], f"{d.name}/{p.parent.name[5:]}") for p in d.glob("eval_*/metrics.json")
                  if (p.parent / "test_submission.csv").exists()]
        out.append(max(cands))
    return [n for _, n in sorted(out, reverse=True)[:top]]


_G = {}  # per-process state for the pool


def _init(dets_list, sizes, per_class_topk):
    _G.update(dets=dets_list, sizes=sizes, topk=per_class_topk)


def _fuse_one(args):
    iid, idx, weights, iou, conf_type = args
    parts = [_G["dets"][i].get(iid, EMPTY) for i in idx]
    w = [weights[i] for i in idx]
    d = wbf(parts, weights=w, iou_thr=iou, conf_type=conf_type) if sum(len(p) > 0 for p in parts) > 1 else \
        next((p for p in parts if len(p)), EMPTY)
    width, height = _G["sizes"][iid]
    return iid, finalize_dets(d, width, height, _G["topk"])


def fuse(pool, ids, idx, weights, iou, conf_type):
    return dict(pool.imap_unordered(_fuse_one, [(i, idx, weights, iou, conf_type) for i in ids], chunksize=16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="*", default=None, help="<id> or <id>/<tag>; default: best variant of --top runs")
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--name", default="wbf")
    ap.add_argument("--iou", type=float, nargs="*", default=[0.5, 0.55, 0.6, 0.7])
    ap.add_argument("--weights", type=float, nargs="*", default=None, help="fixed per-member weights (skips weight grid)")
    ap.add_argument("--conf-type", nargs="*", default=["avg", "max", "box_and_model_avg"])
    ap.add_argument("--min-members", type=int, default=2)
    ap.add_argument("--per-class-topk", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    members = a.members or best_variants(a.top)
    dirs = [member_dir(m) for m in members]
    for m, d in zip(members, dirs):
        assert (d / "val_pred.csv").exists() and (d / "test_submission.csv").exists(), f"{m}: missing CSVs in {d}"
    member_map = [read_json(d / "metrics.json")["map"] for d in dirs]
    print("[ensemble] members:")
    for m, v in zip(members, member_map):
        print(f"    {m:45s} val {v:.4f}")

    val_ids, gt, val_sizes, names = load_gt(DATA / "annotations" / "instances_val.json")
    val_dets = [read_csv(d / "val_pred.csv") for d in dirs]

    # weight candidates: equal, proportional to val mAP, best member doubled
    n = len(members)
    if a.weights:
        assert len(a.weights) == n
        weight_sets = {"fixed": list(a.weights)}
    else:
        order = np.argsort(member_map)[::-1]
        boosted = [1.0] * n
        boosted[order[0]] = 2.0
        weight_sets = {"equal": [1.0] * n, "map_prop": [v / max(member_map) for v in member_map], "best_x2": boosted}

    subsets = [s for k in range(a.min_members, n + 1) for s in itertools.combinations(range(n), k)]
    grid = list(itertools.product(subsets, a.iou, weight_sets.items(), a.conf_type))
    print(f"[ensemble] scoring {len(grid)} settings on val")

    rows, best = [], None
    with Pool(a.workers, initializer=_init, initargs=(val_dets, val_sizes, a.per_class_topk)) as pool:
        for idx, iou, (wname, weights), conf_type in grid:
            dets = fuse(pool, val_ids, idx, weights, iou, conf_type)
            m = score(dets, gt, names)
            row = dict(members="+".join(members[i] for i in idx), iou=iou, weights=wname, conf_type=conf_type,
                       map=round(m["map"], 5), map_50=round(m["map_50"], 5), map_small=round(m["map_small"], 5))
            rows.append(row)
            flag = ""
            if best is None or m["map"] > best[0]:
                best, flag = (m["map"], idx, iou, wname, weights, conf_type, m), "  <- best"
            print(f"    {row['map']:.4f}  iou {iou:.2f}  {wname:8s} {conf_type:17s} {row['members']}{flag}")

    out = RES / "_ensemble" / a.name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "grid.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: -r["map"]))

    _, idx, iou, wname, weights, conf_type, m = best
    chosen = [members[i] for i in idx]
    print(f"\n[ensemble] best val mAP50:95 {m['map']:.4f} (best single member {max(member_map):.4f}, "
          f"{m['map'] - max(member_map):+.4f}): {' + '.join(chosen)}, iou {iou}, weights {wname}, conf {conf_type}")
    for k, v in m["per_class"].items():
        print(f"         {k:16s} {v:.4f}")

    # rebuild the chosen val fusion and write it, then apply the same setting to the test CSVs
    with Pool(a.workers, initializer=_init, initargs=(val_dets, val_sizes, a.per_class_topk)) as pool:
        write_csv(out / "val_pred.csv", val_ids, fuse(pool, val_ids, idx, weights, iou, conf_type))
    m_disk = score(read_csv(out / "val_pred.csv"), gt, names)

    sample = DATA / "sample_submission.csv"
    with open(sample, newline="") as f:
        test_ids = [r["image_id"] for r in csv.DictReader(f)]
    test_sizes = {i: Image.open(DATA / "images" / "test" / f"{i}.jpg").size for i in test_ids}
    test_dets = [read_csv(d / "test_submission.csv") for d in dirs]
    with Pool(a.workers, initializer=_init, initargs=(test_dets, test_sizes, a.per_class_topk)) as pool:
        write_csv(out / "test_submission.csv", test_ids, fuse(pool, test_ids, idx, weights, iou, conf_type))
    info = validate_csv(out / "test_submission.csv", sample, len(names))

    cfg = dict(members=chosen, member_val_map={members[i]: member_map[i] for i in idx}, iou=iou, weights_name=wname,
               weights=[weights[i] for i in idx], conf_type=conf_type, per_class_topk=a.per_class_topk)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    m_disk.update(exp_id=f"_ensemble/{a.name}", **cfg)
    (out / "metrics.json").write_text(json.dumps(m_disk, indent=2))
    print(f"[ensemble] wrote {out}: val {m_disk['map']:.4f}, test {info['rows']} rows / {info['detections']} dets, format OK")


if __name__ == "__main__":
    main()
