"""Offline post-processing sweep on pulled OOF prob maps (runs/<stamp>/oof/*.png).

Threshold, min_area and closing-kernel are cheap to sweep locally; no GPU needed.
    python scripts/sweep_postprocess.py runs/20260908-1200 --thr 0.4 0.5 0.6 --min-area 100 200 400 --close 0 5
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset import load_gt_rles, load_meta  # noqa: E402
from evaluate import score_probs  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--thr", type=float, nargs="+", default=[0.4, 0.5, 0.6])
    ap.add_argument("--min-area", type=int, nargs="+", default=[100, 200, 400])
    ap.add_argument("--close", type=int, nargs="+", default=[0])
    ap.add_argument("--min-score", type=float, nargs="+", default=[0.0])
    ap.add_argument("--dilate", type=int, nargs="+", default=[0])
    a = ap.parse_args()
    meta, gt = load_meta(), load_gt_rles()
    oof = {}
    for p in sorted(Path(a.run, "oof").glob("*.png")):
        oof[p.stem + ".jpeg"] = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    print(f"{len(oof)} OOF files")
    rows = []
    for t in a.thr:
        for m in a.min_area:
            for c in a.close:
                for ms in a.min_score:
                    for d in a.dilate:
                        s = score_probs(oof, meta, gt, t, m, c, ms, d)
                        rows.append((s["pq"], t, m, c, ms, d, s))
                        print(f"thr {t:.2f} min_area {m:4d} close {c} min_score {ms:.2f} dilate {d:+d}: "
                              f"PQ {s['pq']:.4f} SQ {s['sq']:.3f} RQ {s['rq']:.3f} "
                              f"tp {s['tp']} fp {s['fp']} fn {s['fn']}", flush=True)
    best = max(rows)
    print("BEST", best[:6])
