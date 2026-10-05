"""Model adapters. Each adapter exposes:

    train(run_dir) -> dict            train from the config, weights end up in run_dir/weights/{best,last}.pt
    load(weights)                     load trained weights for inference
    predict_arrays(images, imgsz, pcfg) -> list of (n, 6) arrays: x1, y1, x2, y2, score, class_id
"""


def build_adapter(cfg):
    family = cfg["model"]["family"]
    if family == "yolo":
        from .yolo import YoloAdapter

        return YoloAdapter(cfg)
    if family == "dfine":
        from .dfine import DFineAdapter

        return DFineAdapter(cfg)
    raise ValueError(f"unknown model family: {family}")
