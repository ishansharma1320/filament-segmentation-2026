"""Go/no-go for the sequence idea: does temporal persistence separate candidates that some
annotator drew from candidates nobody drew, on the TRAINING images?

For each train file with downloaded neighbours (--seq-dir/<stem>/*.jpg), the OOF fold
detector predicts boxes at a low conf on the labelled frame and on every neighbour. Each
candidate box on the labelled frame is PROJECTED into every neighbour frame (disk fit per
frame; solar differential rotation for the elapsed time, Snodgrass synodic rates) and
matched at box IoU >= --match. Neighbours are bucketed by baseline:
  same-site |dt| ~ 1 h / 3 h / 6 h, and cross-site dt ~ 0 (other telescopes, same minute).
Per bucket: persistence = fraction of that bucket's frames with a match;
support = annotator records with a GT box at IoU > 0.5 to the candidate.

Reports, per bucket: AUC of persistence vs detector conf for "support >= 1" (all and faint
candidates), persistence of drawn vs undrawn candidates by conf band, an ALIGNMENT CHECK
(mean best IoU of confident drawn candidates - must stay high at 6 h or the projection is
wrong), and the gated rule conf >= c_hi OR (conf >= c_lo AND persistence >= p) as box P/R.

    python scripts/persistence_probe.py --seq-dir data/sequences_long --weights 'runs/<v9>/fold{k}_yolo.pt'
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import DET_IMGSZ, REPO  # noqa: E402
from dataset import load_gt_rles, load_meta  # noqa: E402
from detector import _device_arg, box_iou, box_of_rle  # noqa: E402

TRAIN_IMG = REPO / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
_disk_cache = {}


# --- geometry ---------------------------------------------------------------------------
def disk_fit(path) -> tuple[float, float, float]:
    """(cx, cy, R) in px of the solar disk: Otsu on a 4x-downsampled frame, largest blob."""
    if path in _disk_cache:
        return _disk_cache[path]
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(im, (im.shape[1] // 4, im.shape[0] // 4), interpolation=cv2.INTER_AREA)
    _, b = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(b, connectivity=8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy = cents[k] * 4 + 1.5
    R = float(np.sqrt(stats[k, cv2.CC_STAT_AREA] / np.pi) * 4)
    _disk_cache[path] = (float(cx), float(cy), R)
    return _disk_cache[path]


def omega_synodic_deg_per_day(lat: np.ndarray) -> np.ndarray:
    s2 = np.sin(lat) ** 2
    return 14.713 - 2.396 * s2 - 1.787 * s2 * s2 - 0.9856


def project_boxes(boxes: np.ndarray, disk_a, disk_b, dt_days: float) -> tuple[np.ndarray, np.ndarray]:
    """Boxes (n,>=4) xyxy in frame A -> xyxy in frame B after dt_days of rotation.
    Returns (boxes_b, valid) - invalid = near/over the limb where the projection is unusable."""
    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, bool)
    cxa, cya, Ra = disk_a
    cxb, cyb, Rb = disk_b
    x = (boxes[:, 0] + boxes[:, 2]) / 2
    y = (boxes[:, 1] + boxes[:, 3]) / 2
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    X = (x - cxa) / Ra
    Y = (cya - y) / Ra
    rho2 = X * X + Y * Y
    valid = rho2 < 0.94
    lat = np.arcsin(np.clip(Y, -1, 1))
    coslat = np.maximum(np.cos(lat), 1e-6)
    lon = np.arcsin(np.clip(X / coslat, -1, 1))
    lon2 = lon + np.deg2rad(omega_synodic_deg_per_day(lat) * dt_days)
    valid &= np.abs(lon2) < np.deg2rad(80)
    X2 = coslat * np.sin(lon2)
    s = Rb / Ra
    fore = np.maximum(np.cos(lon2), 1e-3) / np.maximum(np.cos(lon), 1e-3)
    x2 = cxb + X2 * Rb
    y2 = cyb - Y * Rb
    w2 = w * s * fore
    h2 = h * s
    out = np.stack([x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2], 1).astype(np.float32)
    return out, valid


# --- helpers ----------------------------------------------------------------------------
def auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    if len(score) < 10 or label.all() or (~label).all():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
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
    res = model.predict(source=paths, imgsz=DET_IMGSZ, conf=conf, iou=0.5, max_det=100,
                        verbose=False, device=_device_arg())
    out = []
    for r in res:
        if r.boxes is None or len(r.boxes) == 0:
            out.append(np.zeros((0, 5), np.float32)); continue
        b = np.concatenate([r.boxes.xyxy.float().cpu().numpy(), r.boxes.conf.float().cpu().numpy()[:, None]], 1)
        out.append(b.astype(np.float32))
    return out


def t_of(name: str) -> datetime:
    return datetime.strptime(name[:14], "%Y%m%d%H%M%S")


def bucket_of(stem: str, neigh_name: str) -> str:
    dt_h = abs((t_of(neigh_name) - t_of(stem)).total_seconds()) / 3600
    if neigh_name[14] != stem[14]:
        return "cross-site 0h"
    if dt_h < 0.5:
        return "same-site <30min"
    return f"same-site {int(round(dt_h))}h"


# --- main -------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="pattern with {k} for the fold")
    ap.add_argument("--seq-dir", default=str(REPO / "data" / "sequences_long"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--match", type=float, default=0.3)
    ap.add_argument("--faint", type=float, default=0.3)
    ap.add_argument("--min-neighbours", type=int, default=2)
    ap.add_argument("--out", default=str(REPO / "runs" / "persistence_probe_long.csv"))
    a = ap.parse_args()
    from ultralytics import YOLO
    meta, gt = load_meta(), load_gt_rles()
    seq = Path(a.seq_dir)
    files = [f for f in sorted(meta["folds"]) if (seq / f.rsplit(".", 1)[0]).is_dir()
             and len(list((seq / f.rsplit(".", 1)[0]).glob("*.jpg"))) >= a.min_neighbours]
    if a.limit:
        files = files[:a.limit]
    print(f"{len(files)} train files with >= {a.min_neighbours} neighbours in {seq}", flush=True)
    models, rows = {}, []
    for n, f in enumerate(files, 1):
        k = meta["folds"][f]
        if k not in models:
            models[k] = YOLO(a.weights.format(k=k))
        stem = f.rsplit(".", 1)[0]
        neigh = sorted((seq / stem).glob("*.jpg"))
        main_path = TRAIN_IMG / f
        preds = predict(models[k], [str(main_path)] + [str(p) for p in neigh], a.conf)
        cand, others = preds[0], preds[1:]
        if len(cand) == 0:
            continue
        g_sets = gt_boxes(meta, gt, f)
        disk_a = disk_fit(main_path)
        support = np.array([sum(int((box_iou(b[None, :4], g)[0] > 0.5).any()) if len(g) else 0 for g in g_sets)
                            for b in cand])
        per_bucket_hits = {}   # bucket -> list per candidate of [hits, frames, best_iou_sum]
        for p, o in zip(neigh, others):
            bkt = bucket_of(stem, p.name)
            dt_days = (t_of(p.name) - t_of(stem)).total_seconds() / 86400
            proj, valid = project_boxes(cand, disk_a, disk_fit(p), dt_days)
            acc = per_bucket_hits.setdefault(bkt, np.zeros((len(cand), 3)))
            if len(o):
                m = box_iou(proj, o[:, :4])
                best = m.max(1)
            else:
                best = np.zeros(len(cand))
            acc[valid, 0] += (best[valid] >= a.match)
            acc[valid, 1] += 1
            acc[valid, 2] += best[valid]
        for i, b in enumerate(cand):
            row = {"file": f, "fold": k, "conf": float(b[4]), "support": int(support[i]),
                   "n_records": len(g_sets), "w": float(b[2] - b[0]), "h": float(b[3] - b[1]),
                   "rho": float(np.hypot((b[0] + b[2]) / 2 - disk_a[0], (b[1] + b[3]) / 2 - disk_a[1]) / disk_a[2])}
            for bkt, acc in per_bucket_hits.items():
                if acc[i, 1] > 0:
                    row[f"pers|{bkt}"] = acc[i, 0] / acc[i, 1]
                    row[f"iou|{bkt}"] = acc[i, 2] / acc[i, 1]
                    row[f"n|{bkt}"] = acc[i, 1]
            rows.append(row)
        if n % 20 == 0:
            print(f"{n}/{len(files)} files, {len(rows)} candidates", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    real = df.support >= 1
    print(f"\nwrote {a.out}: {len(df)} candidates from {df.file.nunique()} files | drawn by >= 1 annotator "
          f"{int(real.sum())} | by none {int((~real).sum())} | AUC(conf) {auc(df.conf, real):.3f}, "
          f"faint (conf < {a.faint}) AUC(conf) {auc(df.conf[df.conf < a.faint], real[df.conf < a.faint]):.3f}")

    buckets = sorted({c.split("|", 1)[1] for c in df.columns if c.startswith("pers|")})
    for bkt in buckets:
        pc, ic = f"pers|{bkt}", f"iou|{bkt}"
        s = df[df[pc].notna()]
        r = s.support >= 1
        faint = s.conf < a.faint
        strong = (s.conf >= 0.5) & r
        print(f"\n=== {bkt}: {len(s)} candidates from {s.file.nunique()} files, "
              f"{s[f'n|{bkt}'].mean():.1f} frames/candidate")
        print(f"  alignment check: confident drawn candidates (n={int(strong.sum())}) mean best IoU "
              f"{s.loc[strong, ic].mean():.3f}, persistence {s.loc[strong, pc].mean():.2f}  (must be high)")
        print(f"  AUC(support>=1): conf {auc(s.conf, r):.3f} | persistence {auc(s[pc], r):.3f} | best IoU {auc(s[ic], r):.3f}")
        print(f"  faint only (n={int(faint.sum())}): conf {auc(s.conf[faint], r[faint]):.3f} | "
              f"persistence {auc(s.loc[faint, pc], r[faint]):.3f} | best IoU {auc(s.loc[faint, ic], r[faint]):.3f}")
        for lo, hi in ((0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.01)):
            b = s[(s.conf >= lo) & (s.conf < hi)]
            if len(b) >= 10:
                print(f"    conf [{lo:.2f},{hi:.1f}): n {len(b):5d} | P(drawn) {b.support.ge(1).mean():.2f} | "
                      f"persistence drawn {b.loc[b.support >= 1, pc].mean():.2f} vs undrawn {b.loc[b.support == 0, pc].mean():.2f}")

        def pr(keep):
            tp = fp = 0
            for f_, g_ in s[keep].groupby("file"):
                tp += int(g_.support.sum()); fp += int((g_.support == 0).sum()) * int(g_.n_records.iloc[0])
            total_gt = sum(len(g) for f_ in s.file.unique() for g in gt_boxes(meta, gt, f_))
            return tp / max(tp + fp, 1), tp / max(total_gt, 1)
        p0, r0 = pr(s.conf >= 0.3)
        print(f"  rule vs plain conf >= 0.3 (P {p0:.3f} R {r0:.3f}):")
        for c_lo in (0.1, 0.15):
            for p in (0.5, 0.8, 1.0):
                p1, r1 = pr((s.conf >= 0.3) | ((s.conf >= c_lo) & (s[pc] >= p)))
                print(f"    OR conf >= {c_lo} & persistence >= {p}: P {p1:.3f} R {r1:.3f}  (dP {p1 - p0:+.3f}, dR {r1 - r0:+.3f})")
        p2, r2 = pr((s.conf >= 0.3) & ~((s.conf < 0.6) & (s[pc] < 0.5)))
        print(f"    AND-prune: drop conf in [0.3,0.6) with persistence < 0.5: P {p2:.3f} R {r2:.3f}  (dP {p2 - p0:+.3f}, dR {r2 - r0:+.3f})")


if __name__ == "__main__":
    main()
