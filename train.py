"""Train one experiment.  python train.py --config configs/s1_01_yolo11s_640.yaml [key.sub=value ...]

Exit code 42 means CUDA out of memory under the VRAM cap (the caller may retry with a smaller batch).
"""
import argparse
import sys
import time

from models import build_adapter
from utils.config import load_config, run_dir, save_config
from utils.logger import cap_vram, init_wandb, is_oom, peak_vram_gb, read_json, seed_everything, write_env, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    out = run_dir(cfg)
    summary_path = out / "train_summary.json"
    if summary_path.exists():
        print(f"[train] {cfg['exp_id']} already trained, skipping")
        return

    save_config(cfg, out / "config.yaml")
    write_env(out / "env.txt")
    seed_everything(cfg.get("seed", 0))
    cap_vram(cfg.get("vram_cap_gb"))
    run = init_wandb(cfg, "train")

    # elapsed time accumulates across resumed sessions
    prev = read_json(out / "train_time.json", {"seconds": 0})["seconds"]
    t0 = time.time()
    try:
        result = build_adapter(cfg).train(out)
    except Exception as e:
        write_json({"seconds": prev + time.time() - t0}, out / "train_time.json")
        if is_oom(e):
            print(f"[train] OOM under the {cfg.get('vram_cap_gb')} GB cap: {e}")
            sys.exit(42)
        raise
    summary = {
        "train_hours": round((prev + time.time() - t0) / 3600, 3),
        "peak_vram_gb_train": peak_vram_gb(),
        **result,  # an adapter that trains in a subprocess reports its own peak
    }
    write_json(summary, summary_path)
    if run:
        run.summary.update({"train_hours": summary["train_hours"], "peak_vram_gb_train": summary["peak_vram_gb_train"]})
        run.finish()
    print(f"[train] done: {summary['train_hours']} h, peak VRAM {summary['peak_vram_gb_train']} GB")


if __name__ == "__main__":
    main()
