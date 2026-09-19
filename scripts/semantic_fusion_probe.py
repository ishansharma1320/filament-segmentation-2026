"""Offline: does the semantic U-Net (a full-disk view) add information to the cascade?
Uses pulled semantic OOF prob maps (runs/<sem>/oof/*.png, 2048 uint8) on the files that
have them, and the cascade's OOF crop records (runs/<cas>/oof_crops).
(a) ranking: mean semantic prob inside each refined mask as a rescoring feature
    (AUC vs the detector conf; PQ sweep of conf x sem^b on the same file subset);
(b) recall complement: GT filaments with no detector box at conf >= 0.1 -> fraction that a
    semantic instance (thr/min_area/min_score) matches at IoU > 0.5, and the precision of
    semantic instances that overlap no detector box (candidate additions).

    python3 scripts/semantic_fusion_probe.py runs/<cascade> runs/<semantic>
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cascade  # noqa: E402
from dataset import load_gt_rles, load_meta  # noqa: E402
from detector import box_iou, box_of_rle  # noqa: E402
from metric import PQAccumulator  # noqa: E402
from postprocess import probs_to_instances  # noqa: E402
from refiner import seed_mask  # noqa: E402
from rle import masks_to_rles  # noqa: E402


def auc(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cas"); ap.add_argument("sem")
    ap.add_argument("--thr", type=float, default=0.55); ap.add_argument("--min-area", type=int, default=300)
    ap.add_argument("--min-score", type=float, default=0.9); ap.add_argument("--close", type=int, default=5)
    a = ap.parse_args()
    meta, gt = load_meta(), load_gt_rles()
    crops = cascade.load_crops(Path(a.cas, "oof_crops"))
    files = sorted(f for f in crops if Path(a.sem, "oof", f.rsplit(".", 1)[0] + ".png").exists())
    print(f"{len(files)} files with both cascade crops and semantic maps")
    feats, ys, confs = [], [], []
    rescored = {b: {} for b in (0, 0.5, 1, 2)}
    miss_total = miss_found = 0
    add_n = add_hit = 0
    for f in files:
        sem = cv2.imread(str(Path(a.sem, "oof", f.rsplit(".", 1)[0] + ".png")), 0).astype(np.float32) / 255
        g = np.array([box_of_rle(r) for rid in meta["records_by_file"][f] for r in gt[rid]], np.float32).reshape(-1, 4)
        g[:, 2:] += g[:, :2]
        recs = []
        for c in crops[f]:
            x0, y0, side = int(c["x0"]), int(c["y0"]), int(c["side"])
            p = c["prob"].astype(np.float32) / 255
            if p.shape[0] != side:
                p = cv2.resize(p, (side, side), interpolation=cv2.INTER_LINEAR)
            m = p >= 0.5
            sv = float(sem[y0:y0 + side, x0:x0 + side][m].mean()) if m.any() else 0.0
            hit = bool(box_iou(np.array(c["box"], np.float32)[None], g).max() > 0.5) if len(g) else False
            feats.append(sv); ys.append(hit); confs.append(c["conf"])
            recs.append((c, sv))
        for b in rescored:
            rescored[b][f] = [dict(c, conf=c["conf"] * max(sv, 1e-3) ** b) for c, sv in recs]
        # (b) recall complement
        boxes = np.array([c["box"] for c in crops[f]], np.float32).reshape(-1, 4)
        sem_inst = probs_to_instances(sem, a.thr, a.min_area, a.close, a.min_score)
        sem_rles = masks_to_rles(sem_inst)
        sem_boxes = np.array([mask_util.toBbox(r) for r in sem_rles], np.float32).reshape(-1, 4)
        sem_boxes[:, 2:] += sem_boxes[:, :2]
        for rid in meta["records_by_file"][f]:
            for r in gt[rid]:
                x, y, w, h = box_of_rle(r); gb = np.array([[x, y, x + w, y + h]], np.float32)
                if len(boxes) and box_iou(gb, boxes).max() > 0.5:
                    continue
                miss_total += 1
                if sem_rles and (np.asarray(mask_util.iou(sem_rles, [r], [0])) > 0.5).any():
                    miss_found += 1
        if len(sem_boxes):
            free = (box_iou(sem_boxes, boxes).max(1) < 0.3) if len(boxes) else np.ones(len(sem_boxes), bool)
            for i in np.nonzero(free)[0]:
                add_n += 1
                for rid in meta["records_by_file"][f]:
                    if gt[rid] and (np.asarray(mask_util.iou(gt[rid], [sem_rles[i]], [0])) > 0.5).any():
                        add_hit += 1; break
    feats, ys, confs = np.array(feats), np.array(ys, bool), np.array(confs)
    print(f"(a) crops {len(ys)}: AUC conf {auc(confs, ys):.3f} | sem {auc(feats, ys):.3f} | conf*sem {auc(confs * feats, ys):.3f} "
          f"| conf*sem^0.5 {auc(confs * np.sqrt(feats), ys):.3f}; sem mean pos {feats[ys].mean():.3f} neg {feats[~ys].mean():.3f}; corr {np.corrcoef(confs, feats)[0, 1]:.3f}")
    for b, d in rescored.items():
        for cut in (0.2, 0.25, 0.3):
            s = cascade.score_crops(d, meta, gt, cut, 0.5, 100)
            print(f"    b={b} cut={cut}: PQ {s['pq']:.4f} SQ {s['sq']:.3f} RQ {s['rq']:.3f} tp {s['tp']} fp {s['fp']} fn {s['fn']}", flush=True)
    print(f"(b) GT filaments with no detector box (conf>=0.1): {miss_total}; matched by a semantic instance: {miss_found} ({miss_found / max(miss_total, 1):.1%})")
    print(f"    semantic instances overlapping no detector box: {add_n}; of which match some annotator: {add_hit} (precision {add_hit / max(add_n, 1):.3f})")
