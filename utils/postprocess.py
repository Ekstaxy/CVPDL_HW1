"""Inference-time machinery shared by all models: views (full / scales / hflip / slices) and merging.

Detections are (n, 6) arrays: x1, y1, x2, y2, score, class_id in original-image pixels.
"""
import numpy as np
import torch
from torchvision.ops import batched_nms

from .metrics import EMPTY


def slice_windows(width, height, size, overlap=0.2):
    """Overlapping square windows covering the image; the last row/column is shifted to stay inside."""

    def starts(length):
        if length <= size:
            return [0]
        step = max(1, int(size * (1 - overlap)))
        s = list(range(0, length - size, step)) + [length - size]
        return sorted(set(s))

    return [(x, y, min(x + size, width), min(y + size, height)) for y in starts(height) for x in starts(width)]


def build_views(img, pcfg, base_imgsz):
    """Each view is (array, imgsz, back) where back(dets) maps detections to original coordinates."""
    h, w = img.shape[:2]
    views = []
    if pcfg.get("native_scale"):
        # infer at the image's own resolution (times a factor), never below the configured size
        base_imgsz = max(base_imgsz, int(round(max(h, w) * pcfg["native_scale"] / 32) * 32))
    for scale in pcfg.get("scales") or [1.0]:
        imgsz = int(round(base_imgsz * scale / 32) * 32)
        views.append((img, imgsz, lambda d: d))
        if pcfg.get("hflip"):

            def unflip(d, w=w):
                d = d.copy()
                d[:, [0, 2]] = w - d[:, [2, 0]]
                return d

            views.append((np.ascontiguousarray(img[:, ::-1]), imgsz, unflip))
    sl = pcfg.get("slice")
    if sl:
        for x1, y1, x2, y2 in slice_windows(w, h, sl["size"], sl.get("overlap", 0.2)):
            if (x2 - x1) >= w and (y2 - y1) >= h:
                continue  # window is the whole image, already covered

            def back(d, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h):
                d = d.copy()
                d[:, [0, 2]] += x1
                d[:, [1, 3]] += y1
                # boxes cut by an inner window edge are partial objects; another view sees them whole
                m = 2.0
                cut = (
                    ((d[:, 0] <= x1 + m) & (x1 > 0))
                    | ((d[:, 1] <= y1 + m) & (y1 > 0))
                    | ((d[:, 2] >= x2 - m) & (x2 < w))
                    | ((d[:, 3] >= y2 - m) & (y2 < h))
                )
                return d[~cut]

            views.append((np.ascontiguousarray(img[y1:y2, x1:x2]), sl.get("imgsz", base_imgsz), back))
    return views


def merge(parts, pcfg):
    """Merge detections of several views of one image (class-wise NMS, or WBF)."""
    parts = [p for p in parts if len(p)]
    if not parts:
        return EMPTY
    if pcfg.get("merge", "nms") == "wbf" and len(parts) > 1:
        return wbf(parts, iou_thr=pcfg.get("merge_iou", 0.6))
    d = np.concatenate(parts)
    t = torch.from_numpy(d)
    keep = batched_nms(t[:, :4], t[:, 4], t[:, 5].long(), pcfg.get("merge_iou", 0.6))
    return d[keep.numpy()]


def wbf(parts, weights=None, iou_thr=0.6, skip_box_thr=0.001, conf_type="avg"):
    """Weighted boxes fusion over a list of detection arrays (views or models) of one image."""
    from ensemble_boxes import weighted_boxes_fusion

    parts = [p for p in parts if len(p)]
    if not parts:
        return EMPTY
    scale = float(max(p[:, :4].max() for p in parts)) + 1.0
    boxes, scores, labels = weighted_boxes_fusion(
        [(p[:, :4] / scale).clip(0, 1).tolist() for p in parts],
        [p[:, 4].tolist() for p in parts],
        [p[:, 5].tolist() for p in parts],
        weights=weights,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
        conf_type=conf_type,
    )
    if len(boxes) == 0:
        return EMPTY
    return np.concatenate([boxes * scale, scores[:, None], labels[:, None]], 1).astype(np.float32)


def predict_images(adapter, images, pcfg, base_imgsz, batch=8):
    """Run every view of every image through the model and merge per image. -> list of (n, 6)."""
    jobs = []  # (image index, array, imgsz, back)
    for i, img in enumerate(images):
        jobs += [(i, a, s, b) for a, s, b in build_views(img, pcfg, base_imgsz)]
    parts = [[] for _ in images]
    for imgsz in sorted({j[2] for j in jobs}):
        group = [j for j in jobs if j[2] == imgsz]
        for k in range(0, len(group), batch):
            chunk = group[k : k + batch]
            outs = adapter.predict_arrays([c[1] for c in chunk], imgsz, pcfg)
            for (i, _, _, back), d in zip(chunk, outs):
                parts[i].append(back(d))
    return [p[0] if len(p) == 1 else merge(p, pcfg) for p in parts]


def predict_paths(adapter, items, pcfg, base_imgsz, per_class_topk=100, chunk=8, desc="predict"):
    """items: (image_id, path) pairs. -> {image_id: final (n, 6) detections}, ready for the CSV."""
    from concurrent.futures import ThreadPoolExecutor

    import cv2
    from tqdm import tqdm

    from .metrics import finalize_dets

    out = {}
    with ThreadPoolExecutor(8) as pool:
        for k in tqdm(range(0, len(items), chunk), desc=desc):
            part = items[k : k + chunk]
            images = list(pool.map(lambda it: cv2.imread(str(it[1])), part))
            for (iid, _), img, d in zip(part, images, predict_images(adapter, images, pcfg, base_imgsz)):
                h, w = img.shape[:2]
                out[iid] = finalize_dets(d, w, h, per_class_topk, pcfg.get("max_det", 1000))
    return out
