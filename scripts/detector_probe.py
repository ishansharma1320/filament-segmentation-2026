"""Offline (CPU) probe of inference-time detector knobs on the pulled fold YOLO weights:
flip/scale TTA and the NMS IoU. Box-level P/R against every annotator record, pooled over
the 5 folds' OOF files; boxes saved per config for a later refiner pass.

    python3 scripts/detector_probe.py runs/<stamp> [--configs base tta nms07 tta_nms07] [--files-per-fold N]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset import load_gt_rles, load_meta  # noqa: E402
from detector import box_pr, predict_boxes  # noqa: E402

CONFIGS = {"base": dict(augment=False, nms_iou=0.5), "tta": dict(augment=True, nms_iou=0.5),
           "nms07": dict(augment=False, nms_iou=0.7), "tta_nms07": dict(augment=True, nms_iou=0.7)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--files-per-fold", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(2)
    from ultralytics import YOLO
    meta, gt = load_meta(), load_gt_rles()
    out = {c: {} for c in a.configs}
    for k in range(5):
        va = sorted(f for f, j in meta["folds"].items() if j == k)
        if a.files_per_fold:
            va = va[:a.files_per_fold]
        model = YOLO(f"{a.run}/fold{k}_yolo.pt")
        for c in a.configs:
            out[c].update(predict_boxes(None, va, meta["size"], model=model, **CONFIGS[c]))
            print(f"fold {k} {c} done", flush=True)
    for c in a.configs:
        for conf in (0.2, 0.3, 0.4):
            r = box_pr(out[c], meta, gt, conf)
            print(f"{c:10s} conf>={conf}: P {r['precision']:.3f} R {r['recall']:.3f} tp {r['tp']} fp {r['fp']} fn {r['fn']}", flush=True)
    if a.out:
        json.dump({c: {f: b.tolist() for f, b in d.items()} for c, d in out.items()}, open(a.out, "w"))
