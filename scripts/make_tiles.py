"""Pre-build the training tiles (training does this on demand; this just makes it explicit).

python scripts/make_tiles.py --size 800 --overlap 0.2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import ROOT, load_config
from utils.data import make_tiles

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=800)
    ap.add_argument("--overlap", type=float, default=0.2)
    a = ap.parse_args()
    paths = make_tiles(load_config(ROOT / "configs" / "base.yaml"), a.size, a.overlap)
    print(f"{len(paths)} tiles in {Path(paths[0]).parents[2]}")
