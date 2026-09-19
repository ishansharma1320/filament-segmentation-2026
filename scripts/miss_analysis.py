"""Where does the cascade's detector miss? Box recall of every GT filament of every annotator
record vs the fold detector's OOF boxes (runs/<stamp>/fold*_oof_boxes.json), broken down by
annotator support (how many of the file's annotators drew that filament) and by size.

    python3 scripts/miss_analysis.py runs/<stamp> [--conf 0.3]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset import load_gt_rles, load_meta  # noqa: E402
from detector import box_iou, box_of_rle  # noqa: E402


def main(run: str, conf: float):
    meta, gt = load_meta(), load_gt_rles()
    boxes = {}
    for p in sorted(Path(run).glob("fold*_oof_boxes.json")):
        boxes.update({f: np.array(v, np.float32).reshape(-1, 5) for f, v in json.load(open(p)).items()})
    rows = []
    for f, rids in meta["records_by_file"].items():
        if f not in boxes:
            continue
        b, n = boxes[f], len(rids)
        for rid in rids:
            for rle in gt[rid]:
                x, y, w, h = box_of_rle(rle)
                g = np.array([[x, y, x + w, y + h]], np.float32)
                ious = box_iou(g, b[:, :4])[0] if len(b) else np.zeros(0)
                best = float(b[ious > 0.5, 4].max()) if (ious > 0.5).any() else 0.0
                k = 1 + sum((np.asarray(mask_util.iou(gt[o], [rle], [0])) > 0.5).any()
                            for o in rids if o != rid and gt[o])
                rows.append((max(w, h), k, n, best))
    a = np.array(rows, float)
    size, k, n, best = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    hit, low = best >= conf, best >= 0.1
    print(f"{len(a)} GT filaments over {len(boxes)} files; box recall @conf {conf}: {hit.mean():.3f}, "
          f"@0.1: {low.mean():.3f}; never proposed: {(best < 0.1).mean():.1%}")
    print("recall by annotator support k/n:")
    for kk, nn in ((1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3)):
        m = (k == kk) & (n == nn)
        if m.any():
            print(f"  k={kk}/n={nn}: n_gt={m.sum():5d}  recall@{conf} {hit[m].mean():.3f}  @0.1 {low[m].mean():.3f}")
    print("recall by long side (native px), consensus (k==n, multi files) / contested / single-annotator:")
    multi = n > 1
    for lo, hi in ((0, 64), (64, 150), (150, 2048)):
        s = (size >= lo) & (size < hi)
        c, x, o = s & multi & (k == n), s & multi & (k < n), s & ~multi
        print(f"  {lo:3d}-{hi:4d}: consensus {hit[c].mean():.3f} (n={c.sum()})  contested {hit[x].mean():.3f} "
              f"(n={x.sum()})  single {hit[o].mean():.3f} (n={o.sum()})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--conf", type=float, default=0.3)
    a = ap.parse_args()
    main(a.run, a.conf)
