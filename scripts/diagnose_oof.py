"""OOF diagnostics for the OOF-vs-LB gap.

  python scripts/diagnose_oof.py runs/<stamp> --thr 0.7 --min-area 400

Reports (a) pooled PQ vs mean-per-record vs mean-per-file, (b) PQ restricted to
single- vs multi-annotator files, (c) where FPs sit (radius from disk centre,
area) and (d) bootstrap spread of PQ over random 90-image subsets (public-LB noise).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset import load_gt_rles, load_meta  # noqa: E402
from metric import IOU_THRESHOLD, PQAccumulator, iou_matrix  # noqa: E402
from postprocess import probs_to_instances  # noqa: E402
from rle import masks_to_rles  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--thr", type=float, default=0.7)
ap.add_argument("--min-area", type=int, default=400)
a = ap.parse_args()
meta, gt = load_meta(), load_gt_rles()

per_record = []   # (file, record, tp_sum_iou, tp, fp, fn)
fp_radius, fp_area, tp_radius = [], [], []
for p in sorted(Path(a.run, "oof").glob("*.png")):
    name = p.stem + ".jpeg"
    prob = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    inst = probs_to_instances(prob, a.thr, a.min_area)
    preds = masks_to_rles(inst)
    cents = [np.argwhere(m).mean(0) for m in inst]
    for rid in meta["records_by_file"][name]:
        g = gt[rid]
        if not g or not preds:
            per_record.append((name, rid, 0.0, 0, len(preds), len(g)))
            continue
        m = iou_matrix(g, preds)
        hit = m > IOU_THRESHOLD
        tp = int(hit.sum()); fp = int((hit.sum(0) == 0).sum()); fn = int((hit.sum(1) == 0).sum())
        per_record.append((name, rid, float(m[hit].sum()), tp, fp, fn))
        for j in range(len(preds)):
            r = float(np.hypot(*(cents[j] - 1024)))
            (tp_radius if hit[:, j].any() else fp_radius).append(r)
            if not hit[:, j].any():
                fp_area.append(int(inst[j].sum()))

def pq(rows):
    s = sum(r[2] for r in rows); tp = sum(r[3] for r in rows)
    d = tp + 0.5 * sum(r[4] for r in rows) + 0.5 * sum(r[5] for r in rows)
    return s / d if d else 0.0

print(f"thr {a.thr} min_area {a.min_area}: records {len(per_record)}")
print(f"pooled PQ            {pq(per_record):.4f}")
print(f"mean per record PQ   {np.mean([pq([r]) for r in per_record]):.4f}")
files = sorted({r[0] for r in per_record})
print(f"mean per file PQ     {np.mean([pq([r for r in per_record if r[0] == f]) for f in files]):.4f}")
multi = {f for f in files if len(meta['records_by_file'][f]) > 1}
print(f"pooled PQ single-annotator files {pq([r for r in per_record if r[0] not in multi]):.4f} "
      f"({len(files) - len(multi)} files)")
print(f"pooled PQ multi-annotator files  {pq([r for r in per_record if r[0] in multi]):.4f} "
      f"({len(multi)} files)")
rng = np.random.default_rng(0)
boots = [pq([r for r in per_record if r[0] in set(rng.choice(files, 90, replace=False))]) for _ in range(200)]
print(f"90-file subsets: PQ mean {np.mean(boots):.4f} sd {np.std(boots):.4f} "
      f"5-95% {np.percentile(boots, 5):.3f}-{np.percentile(boots, 95):.3f}")
fp_radius, tp_radius, fp_area = map(np.array, (fp_radius, tp_radius, fp_area))
print(f"FP radius pct 50/90/99: {np.percentile(fp_radius, [50, 90, 99]).round(0)} | "
      f"TP radius: {np.percentile(tp_radius, [50, 90, 99]).round(0)}")
print(f"FP beyond r=868: {(fp_radius > 868).sum()} of {len(fp_radius)}; "
      f"FP area pct 50/90: {np.percentile(fp_area, [50, 90]).round(0)}")
