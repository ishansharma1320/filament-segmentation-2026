"""Train one fold locally (Apple GPU) - for smoke tests; real runs go to Kaggle.

    FILAMENT_DEBUG=1 python scripts/train.py --fold 0          # 20 steps, 8 val files
    python scripts/train.py --fold 0 --epochs 3
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import EPOCHS, EVAL_FILES, MAX_STEPS, RUNS  # noqa: E402
from dataset import load_gt_rles, load_meta  # noqa: E402
from model import get_device  # noqa: E402
from pipeline import train_fold  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--eval-files", type=int, default=EVAL_FILES)
    ap.add_argument("--name", default=time.strftime("%Y%m%d-%H%M%S"))
    a = ap.parse_args()
    out = RUNS / a.name
    out.mkdir(parents=True, exist_ok=True)
    r = train_fold(a.fold, load_meta(), get_device(), out, load_gt_rles(), epochs=a.epochs,
                   max_steps=a.max_steps, eval_files=a.eval_files)
    print("best PQ", r["best_pq"], "->", r["ckpt"])
