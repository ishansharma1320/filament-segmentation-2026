"""Reproduce the final submission from the committed weights - no training, no cache.

The final entry ("fused v9+v10+v11+v13", public LB 0.40, OOF PQ 0.4513) is built as:

  1. 20 YOLO11s fold detectors (weights/detectors/v09, v10, v11, v13; 5 grouped folds each)
     each predict boxes on every native 2048x2048 test image at imgsz 1280, conf >= 0.05,
     class-agnostic NMS 0.5. A box's score is its confidence times its annotator-agreement
     class weight (1.0 / 0.75 / 0.5 for "drawn by most / half / a minority"; v9-v11 are
     single-class, weight 1.0).
  2. The 20 box sets of one image are merged by weighted boxes fusion (IoU 0.55); a fused
     box's score is the sum of each detector's best score / 20, so agreement is rewarded.
  3. The 5 v13 crop refiners (U-Net/ResNet34 on [image, CLAHE, box seed] crops at native
     resolution) average their flip-TTA probabilities inside every fused box.
  4. Instances, in score order: keep fused boxes with score >= 0.25, threshold the refined
     probability at 0.6, keep the components touching the box seed, drop < 100 px, drop
     masks more than 50 % already taken, clip the rest - no two instances share a pixel.
  5. One RLE row per instance -> submission.csv.

Everything runs on CPU (~2.5 h on a laptop) or GPU. Usage:

    from reproduce import reproduce
    df = reproduce(data_root="data/MAGFiLO_1.0_Kaggle_2026", out_csv="submission.csv")
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = REPO / "weights"
DETECTOR_RUNS = ("v09", "v10", "v11", "v13")   # order matters only for floating-point ties in WBF
FINAL_KNOBS = {"imgsz": 1280, "cut": 0.25, "thr": 0.6, "min_area": 100}


def verify_weights(weights: Path = WEIGHTS) -> int:
    """Check every weight file against weights/SHA256SUMS; returns the number verified."""
    n = 0
    for line in (weights / "SHA256SUMS").read_text().splitlines():
        digest, rel = line.split()
        h = hashlib.sha256((weights / rel).read_bytes()).hexdigest()
        if h != digest:
            raise ValueError(f"checksum mismatch: {rel}")
        n += 1
    return n


def reproduce(data_root: str | Path, out_csv: str | Path = "submission.csv", weights: Path = WEIGHTS,
              device: str | None = None, tta: bool = True, limit: int = 0, log_every: int = 20):
    """Build the final submission from the committed weights. `limit` > 0 runs on the first N
    test images only (a quick check); the full set reproduces the submitted file."""
    os.environ["FILAMENT_DATA"] = str(Path(data_root).resolve())   # before src modules resolve paths
    import cascade
    from config import DET_CLASS_WEIGHTS, DET_CONF_MIN, DET_MAX_DET, DET_NMS_IOU
    from data import test_files, test_img_dir
    from detector import fuse_boxes
    from refiner import load_refiner, read_native, refine_boxes
    from ultralytics import YOLO

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    yolo_dev = 0 if dev.type == "cuda" else "cpu"
    files = test_files()
    if limit:
        files = files[:limit]
    t0 = time.time()
    print(f"{len(files)} test images, device {dev}, weights verified: {verify_weights(weights)}", flush=True)

    # 1. detectors
    box_sets = []
    for run in DETECTOR_RUNS:
        for k in range(5):
            model = YOLO(str(weights / "detectors" / run / f"fold{k}_yolo.pt"))
            out = {}
            paths = [str(test_img_dir() / f) for f in files]
            for i in range(0, len(paths), 8):
                res = model.predict(source=paths[i:i + 8], imgsz=FINAL_KNOBS["imgsz"], conf=DET_CONF_MIN,
                                    iou=DET_NMS_IOU, max_det=DET_MAX_DET, verbose=False, device=yolo_dev,
                                    agnostic_nms=True)
                for f, r in zip(files[i:i + 8], res):
                    if r.boxes is None or len(r.boxes) == 0:
                        out[f] = np.zeros((0, 5), np.float32)
                        continue
                    c = r.boxes.conf.float().cpu().numpy()
                    cls = r.boxes.cls.int().cpu().numpy()
                    c = c * np.asarray(DET_CLASS_WEIGHTS, np.float32)[np.clip(cls, 0, len(DET_CLASS_WEIGHTS) - 1)]
                    b = np.concatenate([r.boxes.xyxy.float().cpu().numpy(), c[:, None]], 1).astype(np.float32)
                    out[f] = b[np.argsort(-b[:, 4])]
            box_sets.append(out)
            print(f"[{time.time() - t0:6.0f}s] detector {run} fold {k} done ({len(box_sets)}/20)", flush=True)

    # 2. weighted boxes fusion
    fused = {f: fuse_boxes([s[f] for s in box_sets]) for f in files}

    # 3. refiners
    refiners = [load_refiner(weights / "refiners" / f"fold{k}_refiner.pt", dev)[0] for k in range(5)]
    crops = {}
    for i, f in enumerate(files):
        b = fused[f]
        crops[f] = refine_boxes(refiners, read_native(f, "test"), b[b[:, 4] >= DET_CONF_MIN], dev, tta=tta)
        if log_every and i % log_every == 0:
            print(f"[{time.time() - t0:6.0f}s] refined {i}/{len(files)}", flush=True)

    # 4-5. instances -> submission
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = cascade.write_cascade_submission(crops, FINAL_KNOBS["cut"], FINAL_KNOBS["thr"], FINAL_KNOBS["min_area"],
                                          out_csv.parent)
    if out_csv.name != "submission.csv":
        (out_csv.parent / "submission.csv").replace(out_csv)
    print(f"[{time.time() - t0:6.0f}s] wrote {out_csv}: {len(df)} rows over {df.filament_id.str.rsplit('_', n=1).str[0].nunique()} images")
    return df, crops


def compare_submissions(a: str | Path, b: str | Path) -> dict:
    """Instance-level agreement of two submission CSVs (same test images): identical rows,
    and per-image mask IoU of the matched instances."""
    import pandas as pd
    from pycocotools import mask as mu
    A, B = pd.read_csv(a), pd.read_csv(b)
    same = len(set(A.segmentation_rle) & set(B.segmentation_rle))
    A["img"] = A.filament_id.str.rsplit("_", n=1).str[0]
    B["img"] = B.filament_id.str.rsplit("_", n=1).str[0]
    ious = []
    for img, ga in A.groupby("img"):
        gb = B[B.img == img]
        ra = [{"size": [2048, 2048], "counts": c.encode()} for c in ga.segmentation_rle]
        rb = [{"size": [2048, 2048], "counts": c.encode()} for c in gb.segmentation_rle]
        if ra and rb:
            m = np.asarray(mu.iou(ra, rb, [0] * len(rb)))
            ious.extend(m.max(1).tolist())
    return {"rows_a": len(A), "rows_b": len(B), "identical_rles": same,
            "mean_best_iou": float(np.mean(ious)) if ious else 0.0,
            "share_iou_gt_0.99": float(np.mean(np.array(ious) > 0.99)) if ious else 0.0}
