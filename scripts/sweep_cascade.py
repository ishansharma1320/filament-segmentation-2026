"""Offline sweep of the cascade knobs on pulled OOF crop records (runs/<stamp>/oof_crops/*.npz).

    python3 scripts/sweep_cascade.py runs/<stamp> --conf 0.2 0.3 0.4 --thr 0.5 0.6 --min-area 100 200 --dup 0.5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cascade  # noqa: E402
from dataset import load_gt_rles, load_meta  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--conf", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--thr", type=float, nargs="+", default=[0.5, 0.6, 0.7])
    ap.add_argument("--min-area", type=int, nargs="+", default=[100, 200])
    ap.add_argument("--dup", type=float, nargs="+", default=[cascade.DUP_FRACTION])
    a = ap.parse_args()
    meta, gt = load_meta(), load_gt_rles()
    oof = cascade.load_crops(Path(a.run, "oof_crops"))
    print(f"{len(oof)} OOF files, {sum(len(v) for v in oof.values())} crops")
    rows = []
    for d in a.dup:
        cascade.DUP_FRACTION = d
        for c in a.conf:
            for t in a.thr:
                for m in a.min_area:
                    s = cascade.score_crops(oof, meta, gt, c, t, m)
                    rows.append((s["pq"], c, t, m, d))
                    print(f"conf {c:.2f} thr {t:.2f} min_area {m:4d} dup {d:.2f}: PQ {s['pq']:.4f} "
                          f"SQ {s['sq']:.3f} RQ {s['rq']:.3f} tp {s['tp']} fp {s['fp']} fn {s['fn']}", flush=True)
    print("BEST", max(rows))
