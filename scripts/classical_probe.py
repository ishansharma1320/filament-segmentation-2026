"""Score the public no-training 'dark residual' baseline (vmerckle, LB 0.46) on the TRAIN
images with our PQ metric, at several component caps. If our metric gives it ~0.46 the method
is genuinely competitive; if it gives far less, the LB evaluator differs from src/metric.py.

    python3 scripts/classical_probe.py [--files N] [--caps 120 30 12 6]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import train_files, train_img_dir  # noqa: E402
from dataset import load_gt_rles, load_meta  # noqa: E402
from metric import PQAccumulator  # noqa: E402
from rle import masks_to_rles  # noqa: E402


def components(path, pct=90.0):
    image = cv2.imread(str(path), 0).astype(np.float32)
    h, w = image.shape
    y, x = np.ogrid[:h, :w]
    disk = (image > 5) & (((y - (h - 1) / 2) ** 2 + (x - (w - 1) / 2) ** 2) < 1000.0 ** 2)
    bg = cv2.GaussianBlur(image, (0, 0), 40.0, borderType=cv2.BORDER_REFLECT)
    res = bg - image
    thr = max(1.0, float(np.percentile(res[disk], pct)))
    m = ((res > thr) & disk).astype(np.uint8)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    comps = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA]); bw, bh = int(stats[i, 2]), int(stats[i, 3])
        if a < 80 or a > 150_000 or a / max(1, bw * bh) > 0.99:
            continue
        comps.append((a, i))
    comps.sort(reverse=True)
    return [(labels == i).astype(np.uint8) for _, i in comps]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=0)
    ap.add_argument("--caps", type=int, nargs="+", default=[120, 30, 12, 6])
    a = ap.parse_args()
    meta, gt = load_meta(), load_gt_rles()
    files = train_files()[:a.files] if a.files else train_files()
    accs = {c: PQAccumulator() for c in a.caps}
    n_comp = []
    for i, f in enumerate(files):
        masks = components(train_img_dir() / f)
        n_comp.append(len(masks))
        rles_all = masks_to_rles(masks[:max(a.caps)])
        for c in a.caps:
            for rid in meta["records_by_file"][f]:
                accs[c].add(gt[rid], rles_all[:c])
        if i % 100 == 0:
            print(f"{i}/{len(files)} comps/img so far {np.mean(n_comp):.1f}", flush=True)
    print(f"components per image: mean {np.mean(n_comp):.1f} median {np.median(n_comp):.0f}")
    for c in a.caps:
        s = accs[c].summary()
        print(f"cap {c:3d}: PQ {s['pq']:.4f} SQ {s['sq']:.3f} RQ {s['rq']:.3f} tp {s['tp']} fp {s['fp']} fn {s['fn']}")
