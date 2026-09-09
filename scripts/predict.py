"""Write a submission from local or pulled checkpoints (fold ensemble).

    python scripts/predict.py runs/20260908-1200/fold*_best.pt --out submissions/x.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import EPOCH_THR, MIN_AREA, REPO  # noqa: E402
from dataset import load_meta  # noqa: E402
from model import get_device, load_checkpoint  # noqa: E402
from pipeline import predict_test, write_submission  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--min-area", type=int, default=None)
    ap.add_argument("--close", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "submissions" / "submission.csv"))
    a = ap.parse_args()
    device = get_device()
    models, cks = zip(*[load_checkpoint(c, device) for c in a.ckpts])
    thr = a.thr if a.thr is not None else cks[0].get("thr", EPOCH_THR)
    area = a.min_area if a.min_area is not None else cks[0].get("min_area", MIN_AREA)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    probs = predict_test(list(models), load_meta()["size"], device)
    df = write_submission(probs, thr, area, out.parent, a.close)
    (out.parent / "submission.csv").rename(out)
    print("wrote", out, len(df), "rows")
