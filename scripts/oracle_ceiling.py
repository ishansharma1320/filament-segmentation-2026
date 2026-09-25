"""How high can PQ go on this label set? Oracles that know every annotator record.

For each file, filaments across its records are clustered (greedy, mask IoU >= 0.5).
A cluster's support = records containing it / records of the file. An oracle predicts
one mask per cluster whose support clears a floor; the mask is the member from the
record that agrees best with the others (medoid record), else the largest member.
Scored with src/metric.py against ALL records, pooled - the evaluator's view.

    python scripts/oracle_ceiling.py            # support floors 0 (union), 1/2, 2/3, 1 (intersection)

Single-annotator files (411/707) score 1.0 for every oracle (the record is its own
consensus), so the multi-annotator subset is the informative number; the pooled figure
shows what a test set with the train mix would allow a perfect consensus predictor.
"""
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset import load_gt_rles, load_meta  # noqa: E402
from metric import PQAccumulator, iou_matrix  # noqa: E402

meta, gt = load_meta(), load_gt_rles()
by_file = meta["records_by_file"]
medoid = meta.get("medoid", {})
FLOORS = (0.0, 0.5, 0.67, 1.0)


def clusters_for(file):
    rids = by_file[file]
    items = [(r, i, gt[r][i]) for r in rids for i in range(len(gt[r]))]
    clusters = []  # each: list of (rid, idx, rle)
    for it in items:
        best, best_iou = None, 0.5
        for c in clusters:
            if any(m[0] == it[0] for m in c):
                continue  # one filament per record per cluster
            m = iou_matrix([it[2]], [c[0][2]])[0, 0]
            if m >= best_iou:
                best, best_iou = c, m
        (best if best is not None else clusters.append([]) or clusters[-1]).append(it)
    return rids, clusters


def oracle_masks(file, floor):
    rids, clusters = clusters_for(file)
    n = len(rids)
    out = []
    for c in clusters:
        if len(c) / n < floor or (floor == 0.0 and len(c) < 1):
            continue
        rep = next((m for m in c if m[0] == medoid.get(file)), None)
        if rep is None:
            rep = max(c, key=lambda m: mask_util.area(m[2]))
        out.append(rep[2])
    return out


accs = {f: {"all": PQAccumulator(), "multi": PQAccumulator(), "single": PQAccumulator()} for f in FLOORS}
files = sorted(by_file)
for k, file in enumerate(files):
    multi = len(by_file[file]) > 1
    for floor in FLOORS:
        preds = oracle_masks(file, floor)
        for r in by_file[file]:
            accs[floor]["all"].add(gt[r], preds)
            accs[floor]["multi" if multi else "single"].add(gt[r], preds)
    if k % 100 == 0:
        print(f"{k}/{len(files)}", flush=True)

print("\noracle (knows every record), pooled PQ against all records:")
print(f"{'support floor':>14} | {'all 707':>8} | {'multi-annot 296':>15} | {'single 411':>10} | TP/FP/FN (multi)")
for floor in FLOORS:
    a, m, s = (accs[floor][k].summary() for k in ("all", "multi", "single"))
    print(f"{floor:>14.2f} | {a['pq']:>8.4f} | {m['pq']:>15.4f} | {s['pq']:>10.4f} | {m['tp']}/{m['fp']}/{m['fn']}  SQ {m['sq']:.3f} RQ {m['rq']:.3f}")
