"""D-FINE adapter. Trains with the official repo (third_party/D-FINE) in a subprocess, fine-tuning the
COCO checkpoint; inference loads the trained model in-process.

Setup (once):
    git clone https://github.com/Peterande/D-FINE third_party/D-FINE
    wget -P checkpoints https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from utils.config import ROOT, resolve, run_dir
from utils.data import CACHE, ann_json, class_names, data_root

REPO = ROOT / "third_party" / "D-FINE"
METRICS = ["mAP50-95", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large"]


def int_id_annotations(cfg, split):
    """The repo needs integer image ids; ours are strings. Same boxes, ids replaced by the image index."""
    out = CACHE / "dfine" / f"instances_{split}.json"
    if not out.exists():
        d = json.loads(ann_json(cfg, split).read_text())
        idx = {im["id"]: i for i, im in enumerate(d["images"])}
        for im in d["images"]:
            im["id"] = idx[im["id"]]
        for a in d["annotations"]:
            a["image_id"] = idx[a["image_id"]]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(d))
    return out


class DFineAdapter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.weights = None
        self._models = {}

    def repo_config(self, out_dir):
        """Our YAML -> a D-FINE config that includes the repo's custom-dataset recipe for this size."""
        m, t = self.cfg["model"], self.cfg["train"]
        s, lr = t["imgsz"], t.get("lr", 1e-4)
        stop = t["epochs"] - t.get("no_aug_epochs", 8)
        root = data_root(self.cfg)
        resize = {"type": "Resize", "size": [s, s]}
        to_float = {"type": "ConvertPILImage", "dtype": "float32", "scale": True}
        # the repo's RandomZoomOut is left out: it shrinks objects that are already tiny here
        train_ops = [
            {"type": "RandomPhotometricDistort", "p": 0.5},
            {"type": "RandomIoUCrop", "p": 0.8},
            {"type": "SanitizeBoundingBoxes", "min_size": 1},
            {"type": "RandomHorizontalFlip"},
            resize,
            {"type": "SanitizeBoundingBoxes", "min_size": 1},
            to_float,
            {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
        ]
        return {
            "__include__": [str(REPO / "configs" / "dfine" / "custom" / f"dfine_hgnetv2_{m['size']}_custom.yml")],
            "output_dir": str(out_dir),
            "num_classes": len(class_names(self.cfg)),
            "remap_mscoco_category": False,
            "eval_spatial_size": [s, s],
            "epochs": t["epochs"],
            "use_amp": bool(t.get("amp", True)),
            "HGNetv2": {"pretrained": False},  # everything is initialised from the COCO checkpoint
            "DFINETransformer": {"num_queries": m.get("num_queries", 300)},
            "DFINEPostProcessor": {"num_top_queries": m.get("num_queries", 300)},
            "optimizer": {
                "lr": lr,
                "params": [
                    {"params": "^(?=.*backbone)(?!.*norm|bn).*$", "lr": lr * 0.5},
                    {"params": "^(?=.*backbone)(?=.*norm|bn).*$", "lr": lr * 0.5, "weight_decay": 0.0},
                    {"params": "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$", "weight_decay": 0.0},
                ],
            },
            "train_dataloader": {
                "total_batch_size": t["batch"],
                "num_workers": t.get("workers", 8),
                "dataset": {
                    "img_folder": str(root / "images" / "train"),
                    "ann_file": str(int_id_annotations(self.cfg, "train")),
                    "transforms": {"ops": train_ops, "policy": {"epoch": stop}},
                },
                "collate_fn": {"base_size": s, "base_size_repeat": 20, "stop_epoch": stop},
            },
            "val_dataloader": {
                "total_batch_size": t["batch"],
                "num_workers": t.get("workers", 8),
                "dataset": {
                    "img_folder": str(root / "images" / "val"),
                    "ann_file": str(int_id_annotations(self.cfg, "val")),
                    "transforms": {"ops": [resize, to_float]},
                },
            },
        }

    def _write_repo_config(self, rd):
        path = rd / "dfine.yml"
        path.write_text(yaml.safe_dump(self.repo_config(rd / "train"), sort_keys=False))
        return path

    def _sync_log(self, log_path, seen):
        """Mirror new epochs of the repo's log.txt to wandb and to train/results.csv. -> epochs seen."""
        if not log_path.exists():
            return seen
        rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        if len(rows) > seen:
            import wandb

            for r in rows[seen:]:
                if wandb.run is not None:
                    log = {f"metrics/{k}": v for k, v in zip(METRICS, r.get("test_coco_eval_bbox", []))}
                    log.update({"train/loss": r.get("train_loss"), "lr/lr": r.get("train_lr")})
                    wandb.log(log, step=int(r["epoch"]) + 1)
            lines = ["epoch,train/loss,metrics/mAP50-95,metrics/mAP50,lr/lr"]
            for r in rows:
                ev = r.get("test_coco_eval_bbox", [0, 0])
                lines.append(f"{int(r['epoch']) + 1},{r.get('train_loss', 0)},{ev[0]},{ev[1]},{r.get('train_lr', 0)}")
            (log_path.parent / "results.csv").write_text("\n".join(lines) + "\n")
        return len(rows)

    def train(self, rd):
        m = self.cfg["model"]
        out = rd / "train"
        cfg_path = self._write_repo_config(rd)
        cap = self.cfg.get("vram_cap_gb")
        # same hard VRAM cap as our own processes, applied before the repo's train.py runs
        launcher = (
            "import sys, runpy, torch\n"
            f"cap = {cap!r}\n"
            "if cap: torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 1024**3 / torch.cuda.get_device_properties(0).total_memory), 0)\n"
            "sys.argv = ['train.py'] + sys.argv[1:]\n"
            "runpy.run_path('train.py', run_name='__main__')\n"
            "print('PEAK_VRAM_GB', round(torch.cuda.max_memory_reserved() / 1024**3, 2))\n"
        )
        args = ["-c", str(cfg_path), "--seed", str(self.cfg.get("seed", 0)), "--output-dir", str(out)]
        if self.cfg["train"].get("amp", True):
            args.append("--use-amp")
        last = out / "last.pth"
        args += ["-r", str(last)] if last.exists() else ["-t", str(resolve(m["weights"]))]
        log_file = rd / "dfine_stdout.log"
        seen = 0
        with open(log_file, "a") as lf:
            proc = subprocess.Popen([sys.executable, "-c", launcher, *args], cwd=REPO, stdout=lf, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                time.sleep(20)
                seen = self._sync_log(out / "log.txt", seen)
                print(f"[dfine] epochs finished: {seen}/{self.cfg['train']['epochs']}", flush=True)
        self._sync_log(out / "log.txt", seen)
        tail = log_file.read_text()[-6000:]
        if proc.returncode != 0:
            print(tail)
            if "out of memory" in tail.lower():
                raise RuntimeError("CUDA out of memory in D-FINE training")
            raise RuntimeError(f"D-FINE training failed (exit {proc.returncode}), see {log_file}")

        wdir = rd / "weights"
        wdir.mkdir(exist_ok=True)
        # best_stg2 exists only if the final no-augmentation stage beat the best of stage 1
        best = out / "best_stg2.pth" if (out / "best_stg2.pth").exists() else out / "best_stg1.pth"
        shutil.copy(best, wdir / "best.pt")
        shutil.copy(last, wdir / "last.pt")
        rows = [json.loads(line) for line in (out / "log.txt").read_text().splitlines() if line.strip()]
        best_row = max(rows, key=lambda r: r["test_coco_eval_bbox"][0])
        peak = [line.split()[-1] for line in tail.splitlines() if line.startswith("PEAK_VRAM_GB")]
        result = {"framework_metrics": dict(zip(METRICS, best_row["test_coco_eval_bbox"])), "best_epoch": int(best_row["epoch"]) + 1}
        if peak:
            result["peak_vram_gb_train"] = float(peak[-1])
        return result

    def load(self, weights):
        self.weights = Path(weights)
        self._models = {}
        return self

    def _model(self, imgsz, top_k):
        key = (imgsz, top_k)
        if key not in self._models:
            if str(REPO) not in sys.path:
                sys.path.insert(0, str(REPO))
            from src.core import YAMLConfig

            rd = run_dir(self.cfg)
            cfg_path = rd / "dfine.yml"
            if not cfg_path.exists():
                self._write_repo_config(rd)
            # positional embeddings and anchors are built for one input size, so one model per size
            c = YAMLConfig(str(cfg_path), eval_spatial_size=[imgsz, imgsz], DFINEPostProcessor={"num_top_queries": top_k})
            state = torch.load(self.weights, map_location="cpu", weights_only=False)
            c.model.load_state_dict(state["ema"]["module"] if "ema" in state else state["model"])
            self._models = {key: (c.model.deploy().cuda().eval(), c.postprocessor.deploy().cuda())}
        return self._models[key]

    @torch.no_grad()
    def predict_arrays(self, images, imgsz, pcfg):
        nq = self.cfg["model"].get("num_queries", 300)
        top_k = min(pcfg.get("max_det", 1000), nq * len(class_names(self.cfg)))
        model, post = self._model(imgsz, top_k)
        batch, sizes = [], []
        for im in images:  # BGR uint8 -> RGB float in [0, 1], plain resize to a square like the repo does
            t = torch.from_numpy(np.ascontiguousarray(im[:, :, ::-1])).cuda().permute(2, 0, 1)[None].float() / 255
            batch.append(F.interpolate(t, size=(imgsz, imgsz), mode="bilinear", antialias=True, align_corners=False))
            sizes.append([im.shape[1], im.shape[0]])
        labels, boxes, scores = post(model(torch.cat(batch)), torch.tensor(sizes, device="cuda"))
        out = []
        conf = pcfg.get("conf", 0.001)
        for lab, box, sc in zip(labels, boxes, scores):
            keep = sc >= conf
            out.append(torch.cat([box[keep], sc[keep, None], lab[keep, None].float()], 1).float().cpu().numpy())
        return out
