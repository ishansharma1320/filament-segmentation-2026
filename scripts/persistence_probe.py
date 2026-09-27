"""Go/no-go for the sequence idea: does temporal persistence separate real faint filaments
from spurious detector candidates on the TRAINING images?

For each train file with downloaded neighbours (data/sequences/<stem>/), the OOF fold
detector (the fold the file is held out of) predicts boxes at a low conf on the labelled
frame and on every neighbour. For each candidate on the labelled frame:
  persistence = fraction of neighbour frames with a box of IoU >= --match to it
  support     = number of annotator records containing a GT filament with box IoU > 0.5
Then: AUC of persistence vs AUC of detector conf for "support >= 1" (any annotator drew
it) overall and among faint candidates (conf < --faint), and a simulated rule
  keep if conf >= c_hi  OR  (conf >= c_lo and persistence >= p)
scored as box P/R against all records, versus the plain conf >= c_hi cut.

    python scripts/persistence_probe.py --weights 'runs/<v9>/fold{k}_det_best.pt' --limit 150
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import DET_IMGSZ, REPO  # noqa: E402
from dataset import load_gt_rles, load_meta  # noqa: E402
from detector import _device_arg, box_iou, box_of_rle  # noqa: E402

SEQ = REPO / "data" / "sequences"


def auc(score, label):
    """Rank AUC (Mann-Whitney)."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    if label.all() or (~label).all():
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    n1 = label.sum(); n0 = len(label) - n1
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def gt_boxes(meta, gt, f):
    out = []
    for rid in meta["records_by_file"][f]:
        g = np.array([box_of_rle(r) for r in gt[rid]], np.float32).reshape(-1, 4)
        g[:, 2:] += g[:, :2]
        out.append(g)
    return out


def predict(model, paths, conf):
    from ultralytics import YOLO  # noqa: F401
    res = model.predict(source=paths, imgsz=DET_IMGSZ, conf=conf, iou=0.5, max_det=100,
                        verbose=False, device=_device_arg())
    out = []
    for r in res:
        if r.boxes is None or len(r.boxes) == 0:
            out.append(np.zeros((0, 5), np.float32)); continue
        b = np.concatenate([r.boxes.xyxy.float().cpu().numpy(), r.boxes.conf.float().cpu().numpy()[:, None]], 1)
        out.append(b.astype(np.float32))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="pattern with {k} for the fold, e.g. runs/x/fold{k}_det_best.pt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--match", type=float, default=0.3)
    ap.add_argument("--faint", type=float, default=0.3)
    ap.add_argument("--min-neighbours", type=int, default=6)
    ap.add_argument("--out", default=str(REPO / "runs" / "persistence_probe.csv"))
    a = ap.parse_args()
    from ultralytics import YOLO
    meta, gt = load_meta(), load_gt_rles()
    files = [f for f in sorted(meta["folds"]) if (SEQ / f.rsplit(".", 1)[0]).is_dir()
             and len(list((SEQ / f.rsplit(".", 1)[0]).glob("*.jpg"))) >= a.min_neighbours]
    if a.limit:
        files = files[:a.limit]
    print(f"{len(files)} train files with >= {a.min_neighbours} neighbours", flush=True)
    models = {}
    rows = []
    for n, f in enumerate(files, 1):
        k = meta["folds"][f]
        if k not in models:
            models[k] = YOLO(a.weights.format(k=k))
        stem = f.rsplit(".", 1)[0]
        neigh = sorted((SEQ / stem).glob("*.jpg"))
        paths = [str(REPO / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images" / f)] + [str(p) for p in neigh]
        preds = predict(models[k], paths, a.conf)
        cand, others = preds[0], preds[1:]
        g_sets = gt_boxes(meta, gt, f)
        for i, b in enumerate(cand):
            hits = [float((box_iou(b[None, :4], o[:, :4])[0] >= a.match).any()) if len(o) else 0.0 for o in others]
            best = [float(box_iou(b[None, :4], o[:, :4])[0].max()) if len(o) else 0.0 for o in others]
            support = sum(int((box_iou(b[None, :4], g)[0] > 0.5).any()) if len(g) else 0 for g in g_sets)
            rows.append({"file": f, "fold": k, "conf": float(b[4]), "persistence": float(np.mean(hits)),
                         "mean_best_iou": float(np.mean(best)), "n_neigh": len(others),
                         "support": support, "n_records": len(g_sets),
                         "w": float(b[2] - b[0]), "h": float(b[3] - b[1])})
        if n % 20 == 0:
            print(f"{n}/{len(files)} files, {len(rows)} candidates", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(df)} candidates from {df.file.nunique()} files\n")

    real = df.support >= 1
    print(f"candidates: {len(df)} | drawn by >= 1 annotator: {int(real.sum())} | by none: {int((~real).sum())}")
    print(f"AUC(support>=1)  conf {auc(df.conf, real):.3f} | persistence {auc(df.persistence, real):.3f} "
          f"| mean_best_iou {auc(df.mean_best_iou, real):.3f}")
    faint = df.conf < a.faint
    print(f"faint (conf < {a.faint}) n={int(faint.sum())}: AUC conf {auc(df.conf[faint], real[faint]):.3f} | "
          f"persistence {auc(df.persistence[faint], real[faint]):.3f} | mean_best_iou {auc(df.mean_best_iou[faint], real[faint]):.3f}")
    for lo, hi in ((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.01)):
        s = df[(df.conf >= lo) & (df.conf < hi)]
        if len(s):
            print(f"  conf [{lo:.1f},{hi:.1f}): n {len(s):5d} | P(real) {s.support.ge(1).mean():.2f} | "
                  f"persistence real {s[s.support >= 1].persistence.mean():.2f} vs none {s[s.support == 0].persistence.mean():.2f}")

    # simulated rule vs plain cut, box-level pooled over records
    def pr(keep):
        tp = fp = 0
        for f, s in df[keep].groupby("file"):
            tp += int(s.support.sum()); fp += int((s.support == 0).sum()) * s.n_records.iloc[0]
        total_gt = sum(len(g) for f in df.file.unique() for g in gt_boxes(meta, gt, f))
        return tp / max(tp + fp, 1), tp / max(total_gt, 1)
    print("\nrule: conf >= c_hi OR (conf >= c_lo AND persistence >= p)   [precision, recall over records]")
    for c_hi in (0.25, 0.3):
        p0, r0 = pr(df.conf >= c_hi)
        print(f"  plain conf >= {c_hi}: P {p0:.3f} R {r0:.3f}")
        for c_lo in (0.1, 0.15):
            for p in (0.5, 0.7, 0.9):
                p1, r1 = pr((df.conf >= c_hi) | ((df.conf >= c_lo) & (df.persistence >= p)))
                print(f"    + conf >= {c_lo} & persistence >= {p}: P {p1:.3f} R {r1:.3f}  (dP {p1 - p0:+.3f}, dR {r1 - r0:+.3f})")


if __name__ == "__main__":
    main()
