"""Score a trained experiment on the validation set, exactly like Kaggle does (CSV -> torchmetrics mAP).

python evaluate.py --config configs/<id>.yaml                       # default inference settings
python evaluate.py --config configs/<id>.yaml --tag hflip predict.hflip=true   # a variant, in eval_<tag>/
"""
import argparse
import random
import time

from models import build_adapter
from utils.config import load_config, run_dir, save_config
from utils.data import ann_json, data_root
from utils.logger import cap_vram, init_wandb, peak_vram_gb, write_json
from utils.metrics import load_gt, read_csv, score, write_csv
from utils.postprocess import predict_paths
from utils.visualize import plot_per_class_ap, plot_training_curves, save_examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", default=None, help="default: results/<id>/weights/best.pt")
    ap.add_argument("--tag", default=None, help="name of an inference variant; outputs go to eval_<tag>/")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    root = run_dir(cfg)
    out = root / f"eval_{a.tag}" if a.tag else root
    out.mkdir(exist_ok=True)
    pcfg = cfg["predict"]
    imgsz = pcfg.get("imgsz") or cfg["train"]["imgsz"]
    save_config({**pcfg, "imgsz": imgsz}, out / "predict.yaml")

    cap_vram(cfg.get("vram_cap_gb"))
    ids, gt, _, names = load_gt(ann_json(cfg, "val"))
    items = [(i, data_root(cfg) / "images" / "val" / f"{i}.jpg") for i in ids]
    adapter = build_adapter(cfg).load(a.weights or root / "weights" / "best.pt")
    t0 = time.time()
    dets = predict_paths(adapter, items, pcfg, imgsz, pcfg.get("per_class_topk", 100), desc="val")
    secs = time.time() - t0
    write_csv(out / "val_pred.csv", ids, dets)

    # score what was written to disk, so rounding in the CSV is part of the measurement
    m = score(read_csv(out / "val_pred.csv"), gt, names)
    m.update(exp_id=cfg["exp_id"], tag=a.tag, imgsz=imgsz, peak_vram_gb_infer=peak_vram_gb(), sec_per_image=round(secs / len(ids), 4))
    write_json(m, out / "metrics.json")
    print(f"[eval] {cfg['exp_id']}{' / ' + a.tag if a.tag else ''}: mAP50:95 {m['map']:.4f}  mAP50 {m['map_50']:.4f}  small {m['map_small']:.4f}")
    for k, v in m["per_class"].items():
        print(f"         {k:16s} {v:.4f}")

    plots = out / "plots"
    plots.mkdir(exist_ok=True)
    plot_per_class_ap(m["per_class"], f"{cfg['exp_id']}  mAP50:95 = {m['map']:.4f}", plots / "per_class_ap.png")
    results_csv = root / "train" / "results.csv"
    if results_csv.exists() and not a.tag:
        plot_training_curves(results_csv, plots / "training_curves.png")
    save_examples(random.Random(0).sample(items, 8), dets, names, plots / "val_examples")

    if not a.tag:
        run = init_wandb(cfg, "eval")
        if run:
            run.summary.update({f"val_csv/{k}": v for k, v in m.items() if isinstance(v, float)})
            run.summary.update({f"val_csv/AP_{k}": v for k, v in m["per_class"].items()})
            run.finish()


if __name__ == "__main__":
    main()
