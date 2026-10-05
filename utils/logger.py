"""Run bookkeeping: seeding, the 12 GB VRAM cap, environment records, wandb."""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from .config import ROOT


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cap_vram(gb):
    """Hard-limit this process's GPU memory so a run can never exceed the HW limit."""
    if gb and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, gb * 1024**3 / total), 0)


def peak_vram_gb():
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_reserved() / 1024**3, 2)


def is_oom(exc):
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, cwd=ROOT, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "n/a"


def write_env(path):
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    lines = [
        f"git: {_sh('git rev-parse HEAD')} dirty={bool(_sh('git status --porcelain'))}",
        f"python: {sys.version.split()[0]}",
        f"torch: {torch.__version__} cuda={torch.version.cuda}",
        f"gpu: {gpu}",
        "",
        _sh(f"{sys.executable} -m pip freeze"),
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def write_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=float))


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def init_wandb(cfg, job_type="train"):
    """One wandb run per experiment id; later stages resume the same run. Returns None if disabled."""
    w = cfg.get("wandb", {})
    if not w.get("enabled", False):
        return None
    try:
        import wandb

        return wandb.init(
            project=w.get("project", "CVPDL_HW1"),
            entity=w.get("entity"),
            id=cfg["exp_id"].replace("/", "_"),
            name=cfg["exp_id"],
            group=cfg.get("stage"),
            job_type=job_type,
            notes=cfg.get("notes"),
            config=cfg,
            resume="allow",
            dir=str(ROOT / "results"),
            settings=wandb.Settings(init_timeout=60),
        )
    except Exception as e:  # network down etc.: keep training
        print(f"[wandb] disabled for this run: {e}")
        os.environ["WANDB_MODE"] = "disabled"
        return None
