"""YAML experiment configs with single-parent inheritance and CLI overrides."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _merge(parent, child):
    out = dict(parent)
    for k, v in child.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _load(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = _merge(_load(path.parent / base), cfg)
    return cfg


def load_config(path, overrides=()):
    """Load `path`, resolve its `base:` chain, then apply `a.b=value` overrides."""
    path = Path(path)
    cfg = _load(path)
    # a resolved config saved in results/<id>/config.yaml keeps its own exp_id
    if path.parent.name == "configs" or "exp_id" not in cfg:
        cfg["exp_id"] = path.stem
    for item in overrides:
        key, _, value = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def save_config(cfg, path):
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))


def run_dir(cfg):
    d = ROOT / "results" / cfg["exp_id"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve(path):
    """Project-relative path -> absolute."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
