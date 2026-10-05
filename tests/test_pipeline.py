"""Fast checks of the parts every score depends on. Run: .venv/bin/pytest tests/"""
import numpy as np
import pytest

from utils.config import ROOT, load_config
from utils.metrics import finalize_dets, load_gt, read_csv, score, validate_csv, write_csv
from utils.postprocess import build_views, merge, slice_windows, wbf

VAL = ROOT / "dataset" / "annotations" / "instances_val.json"


def gt_as_dets(gt, n=None):
    out = {}
    for i, g in list(gt.items())[:n]:
        out[i] = np.stack([g[:, 0], g[:, 1], g[:, 0] + g[:, 2], g[:, 1] + g[:, 3], np.ones(len(g)), g[:, 4]], 1).astype(np.float32)
    return out


@pytest.fixture(scope="module")
def val():
    ids, gt, sizes, names = load_gt(VAL)
    keep = ids[:60]
    return keep, {i: gt[i] for i in keep}, sizes, names


def test_config_inheritance(tmp_path):
    (tmp_path / "a.yaml").write_text("train: {imgsz: 640, epochs: 5}\nseed: 1\n")
    (tmp_path / "b.yaml").write_text("base: a.yaml\ntrain: {imgsz: 1280}\n")
    cfg = load_config(tmp_path / "b.yaml", ["train.batch=4", "predict.slice={size: 800}"])
    assert cfg["train"] == {"imgsz": 1280, "epochs": 5, "batch": 4}
    assert cfg["seed"] == 1 and cfg["exp_id"] == "b" and cfg["predict"]["slice"] == {"size": 800}


def test_all_experiment_configs_load():
    for p in (ROOT / "configs").glob("*.yaml"):
        cfg = load_config(p)
        assert {"model", "train", "predict", "data"} <= set(cfg), p.name
        parent = cfg.get("parent")
        assert parent is None or (ROOT / "configs" / f"{parent}.yaml").exists(), f"{p.name}: parent {parent}"


def test_csv_round_trip(tmp_path):
    d = {"a": np.array([[1.5, 2.25, 11.5, 22.25, 0.9, 3], [0, 0, 5, 5, 0.5, 9]], dtype=np.float32)}
    write_csv(tmp_path / "p.csv", ["a", "b"], d)
    text = (tmp_path / "p.csv").read_text().splitlines()
    assert text[1] == "a,3 0.90000 1.50 2.25 10.00 20.00 9 0.50000 0.00 0.00 5.00 5.00"
    assert text[2] == "b,none"
    back = read_csv(tmp_path / "p.csv")
    assert np.allclose(back["a"], d["a"]) and len(back["b"]) == 0


def test_gt_scores_one_through_csv(val, tmp_path):
    ids, gt, _, names = val
    write_csv(tmp_path / "gt.csv", ids, gt_as_dets(gt))
    m = score(read_csv(tmp_path / "gt.csv"), gt, names)
    # not exactly 1: the metric scores at most 100 detections per class per image
    assert m["map_50"] > 0.9 and m["map"] > 0.9


def test_wrong_classes_score_zero(val):
    ids, gt, _, names = val
    d = gt_as_dets(gt)
    for v in d.values():
        v[:, 5] = (v[:, 5] + 5) % 10
    assert score(d, gt, names)["map"] < 0.02


def test_finalize_clips_and_keeps_top100_per_class():
    rng = np.random.default_rng(0)
    d = np.zeros((300, 6), dtype=np.float32)
    d[:, 0] = rng.uniform(0, 90, 300)
    d[:, 1] = rng.uniform(0, 90, 300)
    d[:, 2] = d[:, 0] + 20
    d[:, 3] = d[:, 1] + 20
    d[:, 4] = rng.uniform(0, 1, 300)
    d[:250, 5] = 3
    d[250:, 5] = 7
    out = finalize_dets(d, 100, 100)
    assert (out[:, 5] == 3).sum() == 100 and (out[:, 5] == 7).sum() == 50
    assert out[:, 2].max() <= 100 and out[:, 3].max() <= 100
    assert out[(out[:, 5] == 3), 4].min() >= np.sort(d[:250, 4])[-100] - 1e-6
    assert len(finalize_dets(np.array([[50, 50, 50, 80, 0.9, 1]]), 100, 100)) == 0


def test_slice_windows_cover_image():
    for w, h in [(1400, 1050), (2000, 1500), (960, 540), (700, 500)]:
        cover = np.zeros((h, w), dtype=bool)
        for x1, y1, x2, y2 in slice_windows(w, h, 800, 0.2):
            assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h and x2 - x1 <= 800 and y2 - y1 <= 800
            cover[y1:y2, x1:x2] = True
        assert cover.all()


def test_views_map_back_to_original_coordinates():
    img = np.zeros((1000, 1500, 3), dtype=np.uint8)
    box = np.array([900.0, 300.0, 960.0, 380.0])  # an object in original coordinates
    pcfg = {"scales": [1.0], "hflip": True, "slice": {"size": 800, "overlap": 0.2}}
    views = build_views(img, pcfg, 1280)
    assert len(views) > 3
    hits = 0
    for arr, _, back in views:
        h, w = arr.shape[:2]
        if arr.shape == img.shape and back(np.array([[0, 0, 10, 10, 1, 0]], dtype=np.float32))[0, 0] != 0:
            local = np.array([1500 - box[2], box[1], 1500 - box[0], box[3]])  # flipped view
        elif arr.shape == img.shape:
            local = box
        else:
            # find this window's offset by mapping a point at the window centre back
            ox, oy = back(np.array([[w / 2, h / 2, w / 2 + 1, h / 2 + 1, 1, 0]], dtype=np.float32))[0, :2] - [w / 2, h / 2]
            local = box - [ox, oy, ox, oy]
            if local[0] < 3 or local[1] < 3 or local[2] > w - 3 or local[3] > h - 3:
                continue  # object not fully inside this window
        got = back(np.array([[*local, 0.9, 2]], dtype=np.float32))
        assert len(got) == 1 and np.allclose(got[0, :4], box, atol=1e-3)
        hits += 1
    assert hits >= 3


def test_slice_view_drops_boxes_cut_by_inner_edge():
    img = np.zeros((1000, 1500, 3), dtype=np.uint8)
    views = build_views(img, {"slice": {"size": 800, "overlap": 0.2}}, 1280)
    arr, _, back = views[1]  # first window, at the top-left corner
    h, w = arr.shape[:2]
    d = np.array([[0, 0, 50, 50, 0.9, 1], [w - 40, 100, w, 160, 0.9, 1]], dtype=np.float32)
    out = back(d)
    assert len(out) == 1 and out[0, 2] == 50  # image-border box kept, inner-edge box dropped


def test_merge_nms_and_wbf_remove_duplicates():
    a = np.array([[10, 10, 50, 50, 0.9, 1], [100, 100, 140, 140, 0.8, 2]], dtype=np.float32)
    b = np.array([[11, 11, 51, 51, 0.7, 1], [10, 10, 50, 50, 0.6, 3]], dtype=np.float32)
    n = merge([a, b], {"merge": "nms", "merge_iou": 0.6})
    assert len(n) == 3  # the class-1 duplicate is suppressed, the class-3 box at the same place stays
    f = wbf([a, b], iou_thr=0.6)
    assert len(f) == 3 and set(f[:, 5].astype(int)) == {1, 2, 3}
    fused = f[f[:, 5] == 1][0]
    assert 10 <= fused[0] <= 11 and 50 <= fused[2] <= 51


def test_validate_csv_accepts_sample_and_rejects_bad(tmp_path):
    sample = ROOT / "dataset" / "sample_submission.csv"
    assert validate_csv(sample, sample)["rows"] == 1800
    lines = sample.read_text().splitlines()
    (tmp_path / "bad.csv").write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(AssertionError):
        validate_csv(tmp_path / "bad.csv", sample)
    lines[1] = lines[1].split(",")[0] + ",12 0.5 1 1 5 5"
    (tmp_path / "bad2.csv").write_text("\n".join(lines) + "\n")
    with pytest.raises(AssertionError):
        validate_csv(tmp_path / "bad2.csv", sample)
