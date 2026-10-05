"""Ultralytics YOLO adapter (YOLO11 / YOLO12 / YOLO26, optional custom architecture yaml)."""
from pathlib import Path

import numpy as np

from utils.config import ROOT, resolve
from utils.data import write_yolo_data_yaml


class YoloAdapter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None

    def _wandb_callback(self, trainer):
        import wandb

        if wandb.run is None:
            return
        log = {**trainer.label_loss_items(trainer.tloss, prefix="train"), **trainer.metrics, **trainer.lr}
        wandb.log({k: float(v) for k, v in log.items()}, step=trainer.epoch + 1)

    def train(self, run_dir):
        from ultralytics import YOLO, settings

        m, t = self.cfg["model"], self.cfg["train"]
        train_dir = run_dir / "train"
        last = train_dir / "weights" / "last.pt"
        # Ultralytics' auxiliary downloads (AMP check weights) go next to the other pretrained weights
        settings.update(weights_dir=str(ROOT / "checkpoints"), runs_dir=str(ROOT / "results"))
        if last.exists():
            model = YOLO(str(last))
            args = dict(resume=True)
        else:
            weights = str(resolve(m["weights"])) if m.get("weights") else None
            if m.get("arch"):
                # the scale letter in the file name (yolo11s-p2.yaml) selects the scale of yolo11-p2.yaml
                model = YOLO(str(resolve(m["arch"])))
                if weights:
                    model.load(weights)
            else:
                model = YOLO(weights)
            args = dict(
                data=str(write_yolo_data_yaml(self.cfg, run_dir / "data.yaml")),
                project=str(run_dir),
                name="train",
                exist_ok=True,
                seed=self.cfg.get("seed", 0),
                device=0,
                max_det=1000,
                plots=True,
                **{k: v for k, v in t.items() if k != "extra"},
                **(t.get("extra") or {}),
            )
        model.add_callback("on_fit_epoch_end", self._wandb_callback)
        try:
            model.train(**args)
        except AssertionError as e:
            # last.pt of a training that already ran all its epochs cannot be resumed; it is finished
            if "nothing to resume" not in str(e):
                raise
        link = run_dir / "weights"
        if not link.exists():
            link.symlink_to(Path("train") / "weights")
        metrics = getattr(getattr(model, "trainer", None), "metrics", None) or {}
        return {"framework_metrics": {k: float(v) for k, v in metrics.items()}}

    def load(self, weights):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        return self

    def predict_arrays(self, images, imgsz, pcfg):
        res = self.model.predict(
            images,
            imgsz=imgsz,
            conf=pcfg.get("conf", 0.001),
            iou=pcfg.get("iou", 0.6),
            max_det=pcfg.get("max_det", 1000),
            # full precision: fp16 box decoding rounds coordinates to ~1 px at 1280 input, which costs
            # mAP at the high IoU thresholds on tiny objects
            device=0,
            verbose=False,
        )
        out = []
        for r in res:
            b = r.boxes
            out.append(np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], 1).astype(np.float32))
        return out
