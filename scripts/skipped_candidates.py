"""What do the candidates annotators did NOT draw have in common, and is that learnable?

From the pulled OOF detector boxes (runs/<stamp>/fold*_oof_boxes.json, one entry per train
file, [x0,y0,x1,y1,conf] native px) and the training images, per candidate:
  conf, size, aspect, area, limb distance (rho), contrast (mean inside box vs surrounding
  ring), local brightness / plage fraction (bright pixels in a window around the box),
  darkness of the darkest core, and support = annotator records with a GT box IoU > 0.5.
Reports: drawn vs undrawn feature means per conf band, single-feature AUCs, and a
gradient-boosted "would an annotator draw this" classifier under grouped 5-fold CV by file
(AUC vs conf alone). If the classifier clearly beats conf, the skip rule is learnable from
single-image features and rescoring the cascade is the next run.

    python scripts/skipped_candidates.py runs/20260927-130912
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset import load_gt_rles, load_meta  # noqa: E402
from detector import box_iou, box_of_rle  # noqa: E402

TRAIN_IMG = Path(__file__).resolve().parents[1] / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"


def auc(score, label):
    from sklearn.metrics import roc_auc_score
    label = np.asarray(label, bool)
    if label.all() or (~label).all():
        return float("nan")
    return float(roc_auc_score(label, score))


def disk(img):
    small = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    _, b = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, st, ce = cv2.connectedComponentsWithStats(b, connectivity=8)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return ce[k][0] * 4, ce[k][1] * 4, float(np.sqrt(st[k, cv2.CC_STAT_AREA] / np.pi) * 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    meta, gt = load_meta(), load_gt_rles()
    boxes = {}
    for k in range(5):
        boxes.update(json.load(open(run / f"fold{k}_oof_boxes.json")))
    rows = []
    for n, (f, cand) in enumerate(sorted(boxes.items())):
        if not cand:
            continue
        img = cv2.imread(str(TRAIN_IMG / f), cv2.IMREAD_GRAYSCALE)
        cx, cy, R = disk(img)
        yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
        on_disk = ((xx - cx) ** 2 + (yy - cy) ** 2) < (0.97 * R) ** 2
        disk_med = float(np.median(img[on_disk]))
        bright_thr = np.percentile(img[on_disk], 97)   # plage-ish pixels
        g_sets = []
        for rid in meta["records_by_file"][f]:
            g = np.array([box_of_rle(r) for r in gt[rid]], np.float32).reshape(-1, 4)
            g[:, 2:] += g[:, :2]
            g_sets.append(g)
        cand = np.array(cand, np.float32)
        for b in cand:
            x0, y0, x1, y1, conf = b
            x0i, y0i, x1i, y1i = int(max(x0, 0)), int(max(y0, 0)), int(min(x1, 2047)), int(min(y1, 2047))
            if x1i <= x0i or y1i <= y0i:
                continue
            inside = img[y0i:y1i + 1, x0i:x1i + 1].astype(np.float32)
            pad = int(max(20, 0.5 * max(x1 - x0, y1 - y0)))
            X0, Y0, X1, Y1 = max(x0i - pad, 0), max(y0i - pad, 0), min(x1i + pad, 2047), min(y1i + pad, 2047)
            ctx = img[Y0:Y1 + 1, X0:X1 + 1].astype(np.float32)
            ring_mask = np.ones(ctx.shape, bool)
            ring_mask[y0i - Y0:y1i - Y0 + 1, x0i - X0:x1i - X0 + 1] = False
            ring = ctx[ring_mask]
            w, h = x1 - x0, y1 - y0
            bx, by = (x0 + x1) / 2, (y0 + y1) / 2
            support = sum(int((box_iou(b[None, :4], g)[0] > 0.5).any()) if len(g) else 0 for g in g_sets)
            rows.append({
                "file": f, "conf": float(conf), "support": support, "n_records": len(g_sets),
                "w": w, "h": h, "long": max(w, h), "short": min(w, h), "aspect": max(w, h) / max(min(w, h), 1),
                "area": w * h, "rho": float(np.hypot(bx - cx, by - cy) / R),
                "contrast": float((ring.mean() - inside.mean()) / max(disk_med, 1)),
                "core_dark": float((ring.mean() - np.percentile(inside, 5)) / max(disk_med, 1)),
                "inside_rel": float(inside.mean() / max(disk_med, 1)),
                "ring_rel": float(ring.mean() / max(disk_med, 1)),
                "plage_frac": float((ctx > bright_thr).mean()),
                "dark_frac": float((inside < ring.mean() - 0.08 * disk_med).mean()),
                "ring_std": float(ring.std() / max(disk_med, 1)),
            })
        if n % 100 == 0:
            print(f"{n}/{len(boxes)} files, {len(rows)} candidates", flush=True)
    df = pd.DataFrame(rows)
    out = a.out or (run / "skipped_candidates.csv")
    df.to_csv(out, index=False)
    drawn = df.support >= 1
    print(f"\n{len(df)} candidates from {df.file.nunique()} files: drawn {int(drawn.sum())}, undrawn {int((~drawn).sum())}")

    feats = ["conf", "long", "short", "aspect", "area", "rho", "contrast", "core_dark", "inside_rel",
             "ring_rel", "plage_frac", "dark_frac", "ring_std"]
    print("\nsingle-feature AUC for 'drawn by >= 1 annotator' (all | conf 0.1-0.5 only):")
    mid = (df.conf >= 0.1) & (df.conf < 0.5)
    for c in feats:
        print(f"  {c:11s} {auc(df[c], drawn):.3f} | {auc(df.loc[mid, c], drawn[mid]):.3f}")

    print("\nmean feature by group, candidates with conf 0.1-0.5 (drawn / undrawn):")
    g = df[mid].groupby(drawn[mid]).mean(numeric_only=True)
    for c in ["long", "aspect", "rho", "contrast", "core_dark", "plage_frac", "dark_frac", "ring_std"]:
        print(f"  {c:11s} {g.loc[True, c]:.3f} / {g.loc[False, c]:.3f}")

    # learned "would an annotator draw this", grouped CV by file
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    X = df[feats].values; y = drawn.values; groups = df.file.values
    oof = np.zeros(len(df))
    for tr, te in GroupKFold(5).split(X, y, groups):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    df["learned"] = oof
    print(f"\nlearned classifier (grouped 5-fold OOF): AUC {auc(oof, y):.3f} vs conf alone {auc(df.conf, y):.3f}")
    print(f"  conf 0.1-0.5 only: learned {auc(oof[mid], y[mid]):.3f} vs conf {auc(df.conf[mid], y[mid]):.3f}")
    # what does the learned score buy at equal candidate count?
    for q in (0.4, 0.5, 0.6):
        k = int(q * len(df))
        top_c = df.nlargest(k, "conf"); top_l = df.nlargest(k, "learned")
        print(f"  top {int(q * 100)}% of candidates: precision by conf {top_c.support.ge(1).mean():.3f} | by learned {top_l.support.ge(1).mean():.3f}")
    df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
