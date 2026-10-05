"""Dataset paths and the generated training sets (repeat-factor lists, tiles)."""
import json
import random
from collections import Counter, defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import yaml

from .config import ROOT, resolve
from .postprocess import slice_windows

CACHE = ROOT / "data_cache"


def data_root(cfg):
    return resolve(cfg["data"]["root"])


def class_names(cfg):
    return (data_root(cfg) / "classes.txt").read_text().split()


def split_images(cfg, split):
    """Sorted (image_id, path) pairs of a split; test follows sample_submission.csv order by id sort."""
    d = data_root(cfg) / "images" / split
    return [(p.stem, p) for p in sorted(d.glob("*.jpg"))]


def ann_json(cfg, split):
    return data_root(cfg) / "annotations" / f"instances_{split}.json"


def rfs_list(cfg, thresh, seed=0):
    """LVIS-style repeat-factor sampling: r(img) = max over its classes of max(1, sqrt(t / f_c)),
    f_c = fraction of train images containing class c. Returns image paths with repeats."""
    d = json.loads(ann_json(cfg, "train").read_text())
    per_img = defaultdict(set)
    for a in d["annotations"]:
        per_img[a["image_id"]].add(a["category_id"])
    n = len(d["images"])
    freq = Counter(c for cs in per_img.values() for c in cs)
    rep = {c: max(1.0, (thresh / (k / n)) ** 0.5) for c, k in freq.items()}
    rng = random.Random(seed)
    out = []
    img_dir = data_root(cfg) / "images" / "train"
    for im in d["images"]:
        r = max((rep[c] for c in per_img[im["id"]]), default=1.0)
        k = int(r) + (rng.random() < r - int(r))
        out += [str(img_dir / im["file_name"])] * k
    return out


def _tile_one(item, size, overlap, min_vis, out_dir):
    img_path, boxes = item  # boxes: (n, 5) x, y, w, h, cls in pixels
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    paths = []
    for x1, y1, x2, y2 in slice_windows(w, h, size, overlap):
        if (x2 - x1) >= w and (y2 - y1) >= h:
            continue
        bx1 = np.clip(boxes[:, 0], x1, x2)
        by1 = np.clip(boxes[:, 1], y1, y2)
        bx2 = np.clip(boxes[:, 0] + boxes[:, 2], x1, x2)
        by2 = np.clip(boxes[:, 1] + boxes[:, 3], y1, y2)
        bw, bh = bx2 - bx1, by2 - by1
        vis = bw * bh / np.maximum(boxes[:, 2] * boxes[:, 3], 1e-6)
        keep = (vis >= min_vis) & (bw >= 2) & (bh >= 2)
        if not keep.any():
            continue
        tw, th = x2 - x1, y2 - y1
        name = f"{img_path.stem}_{x1}_{y1}"
        cv2.imwrite(str(out_dir / "images" / "train" / f"{name}.jpg"), img[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = [
            f"{int(c)} {((a + b) / 2 - x1) / tw:.6f} {((e + f) / 2 - y1) / th:.6f} {(b - a) / tw:.6f} {(f - e) / th:.6f}"
            for c, a, b, e, f in zip(boxes[keep, 4], bx1[keep], bx2[keep], by1[keep], by2[keep])
        ]
        (out_dir / "labels" / "train" / f"{name}.txt").write_text("\n".join(lines) + "\n")
        paths.append(str(out_dir / "images" / "train" / f"{name}.jpg"))
    return paths


def make_tiles(cfg, size, overlap=0.2, min_vis=0.4):
    """Cut the TRAIN images into overlapping tiles (YOLO layout). Cached on disk. -> tile image paths."""
    out_dir = CACHE / f"tiles_{size}_{int(overlap * 100)}"
    index = out_dir / "tiles.txt"
    if index.exists():
        return index.read_text().split("\n")[:-1]
    (out_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    d = json.loads(ann_json(cfg, "train").read_text())
    boxes = defaultdict(list)
    for a in d["annotations"]:
        boxes[a["image_id"]].append([*a["bbox"], a["category_id"]])
    img_dir = data_root(cfg) / "images" / "train"
    items = [(img_dir / im["file_name"], np.array(boxes[im["id"]], dtype=np.float32).reshape(-1, 5)) for im in d["images"]]
    with Pool(16) as pool:
        res = pool.map(partial(_tile_one, size=size, overlap=overlap, min_vis=min_vis, out_dir=out_dir), items, chunksize=16)
    paths = [p for r in res for p in r]
    index.write_text("\n".join(paths) + "\n")
    return paths


def train_image_list(cfg):
    """Training image paths per the data config: full images (optionally RFS-repeated) plus optional tiles.
    Returns None when the plain images/train folder is enough."""
    dc = cfg["data"]
    tiles, rfs = dc.get("tiles"), dc.get("rfs_thresh")
    if not tiles and not rfs:
        return None
    full = rfs_list(cfg, rfs, cfg.get("seed", 0)) if rfs else [str(p) for _, p in split_images(cfg, "train")]
    out = []
    if tiles:
        out += make_tiles(cfg, tiles["size"], tiles.get("overlap", 0.2))
        if not tiles.get("include_full", True):
            full = []
    return out + full


def write_yolo_data_yaml(cfg, path):
    """Ultralytics dataset yaml for this run; train on a generated list when the config asks for one."""
    root = data_root(cfg)
    train = "images/train"
    lst = train_image_list(cfg)
    if lst is not None:
        train = str(Path(path).with_name("train_list.txt"))
        Path(train).write_text("\n".join(lst) + "\n")
    names = dict(enumerate(class_names(cfg)))
    Path(path).write_text(yaml.safe_dump({"path": str(root), "train": train, "val": "images/val", "names": names}))
    return path
