"""Rung 4, stage 1: YOLO box detector (ultralytics) trained on the 1024 px cache.

One box per filament is the whole point: NMS gives instance separation (no merged
neighbours) and a per-object confidence (the FP/FN knob), which the semantic model
never had. Labels are written once (YOLO reads static txt files, so one annotator
record per file: the consensus / medoid one); per-fold train/val lists reuse the
grouped folds of the cache. Boxes are returned in NATIVE (2048) coordinates.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_util

from config import (CACHE, DET_BATCH, DET_CONF_MIN, DET_EPOCHS, DET_FUSE_IOU, DET_IMGSZ, DET_LABEL_MIN_SUPPORT,
                    DET_LABEL_RECORD, DET_MAX_DET, DET_MODEL, DET_NMS_IOU, H, SEED, W, WORKERS, log)
from data import test_files, train_files


def yolo_root() -> Path:
    return CACHE / "yolo"


def box_of_rle(rle) -> tuple[float, float, float, float]:
    """(x, y, w, h) in native px of one GT RLE."""
    x, y, w, h = mask_util.toBbox(rle)
    return float(x), float(y), float(w), float(h)


def union_boxes(rles_by_record: list[list], min_support: float = DET_LABEL_MIN_SUPPORT,
                iou_thr: float = 0.5) -> list[tuple[float, float, float, float]]:
    """Filaments of all annotator records of one file, deduplicated across records.
    Greedy clustering: a filament joins the first cluster that holds no mask of its own record
    yet and holds a mask of another record with IoU > iou_thr (one filament per record per
    cluster, so an annotator who split a filament in two keeps two boxes). A cluster's support
    is the number of records in it; clusters with support / n_records >= min_support are kept,
    each represented by the box of its member with the highest total IoU to the other members
    (the consensus outline)."""
    n = len(rles_by_record)
    clusters: list[list[tuple[int, object]]] = []   # [(record index, rle), ...]
    for r, rles in enumerate(rles_by_record):
        for rle in rles:
            placed = False
            for cl in clusters:
                if any(k == r for k, _ in cl):
                    continue
                others = [m for _, m in cl]
                if (np.asarray(mask_util.iou(others, [rle], [0] * 1)) > iou_thr).any():
                    cl.append((r, rle)); placed = True
                    break
            if not placed:
                clusters.append([(r, rle)])
    out = []
    for cl in clusters:
        if len({k for k, _ in cl}) / n < min_support:
            continue
        if len(cl) == 1:
            out.append(box_of_rle(cl[0][1])); continue
        ms = [m for _, m in cl]
        tot = np.asarray(mask_util.iou(ms, ms, [0] * len(ms))).sum(1)
        out.append(box_of_rle(ms[int(tot.argmax())]))
    return out


def build_yolo_labels(meta: dict, gt_rles: dict) -> Path:
    """images/<stem>.png -> the cached S x S PNGs (symlinks); labels/<stem>.txt in normalised
    YOLO format. Label files are always rewritten (the label recipe is a config knob)."""
    root, S = yolo_root(), meta["size"]
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(exist_ok=True)
    (root / "labels.cache").unlink(missing_ok=True)   # ultralytics' scan cache of the old labels
    n_box, n_files, support_hist = 0, 0, {}
    for f in train_files():
        stem = f.rsplit(".", 1)[0]
        dst = root / "images" / f"{stem}.png"
        if not dst.exists():
            os.symlink((CACHE / f"img{S}" / f"{stem}.png").resolve(), dst)
        rids = meta["records_by_file"][f]
        if DET_LABEL_RECORD == "union":
            boxes = union_boxes([gt_rles[r] for r in rids])
        else:
            rid = meta["medoid"][f] if DET_LABEL_RECORD == "medoid" else rids[0]
            boxes = [box_of_rle(rle) for rle in gt_rles[rid]]
        lines = []
        for x, y, w, h in boxes:
            if w < 1 or h < 1:
                continue
            lines.append(f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
        (root / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_box += len(lines); n_files += 1
        support_hist[len(rids)] = support_hist.get(len(rids), 0) + len(lines)
    log(f"YOLO labels ({DET_LABEL_RECORD}): {n_box} boxes over {n_files} files "
        f"({n_box / max(n_files, 1):.2f}/file); boxes by n_annotators {support_hist}")
    return root


def fold_yaml(fold: int, meta: dict, train_limit: int = 0, val_limit: int = 0) -> Path:
    root, folds = yolo_root(), meta["folds"]
    tr = sorted(f for f, k in folds.items() if k != fold)
    va = sorted(f for f, k in folds.items() if k == fold)
    if train_limit:
        tr = tr[:train_limit]
    if val_limit:
        va = va[:val_limit]
    paths = lambda fs: "\n".join(str(root / "images" / (f.rsplit(".", 1)[0] + ".png")) for f in fs) + "\n"
    (root / f"fold{fold}_train.txt").write_text(paths(tr))
    (root / f"fold{fold}_val.txt").write_text(paths(va))
    y = root / f"fold{fold}.yaml"
    y.write_text(f"path: {root}\ntrain: {root / f'fold{fold}_train.txt'}\nval: {root / f'fold{fold}_val.txt'}\n"
                 f"names:\n  0: filament\n")
    return y


def _device_arg():
    return 0 if torch.cuda.is_available() else "cpu"


def train_detector(fold: int, meta: dict, out_dir: Path, epochs: int = DET_EPOCHS, train_limit: int = 0,
                   val_limit: int = 0) -> Path:
    """Train YOLO on the fold's train files; returns the copied best.pt (fold{k}_yolo.pt)."""
    from ultralytics import YOLO
    from ultralytics import settings as yolo_settings
    yolo_settings.update({"sync": False, "wandb": False, "mlflow": False, "tensorboard": False})
    yaml = fold_yaml(fold, meta, train_limit, val_limit)
    log(f"fold {fold}: YOLO {DET_MODEL} imgsz {DET_IMGSZ} epochs {epochs}")
    model = YOLO(DET_MODEL)
    model.train(data=str(yaml), epochs=epochs, imgsz=DET_IMGSZ, batch=DET_BATCH, workers=WORKERS,
                seed=SEED, deterministic=False, project=str(out_dir / "yolo"), name=f"fold{fold}",
                exist_ok=True, plots=False, verbose=False, device=_device_arg(), single_cls=True,
                flipud=0.5, fliplr=0.5, patience=epochs, save_period=-1, val=True)
    best = out_dir / "yolo" / f"fold{fold}" / "weights" / "best.pt"
    if not best.exists():
        best = out_dir / "yolo" / f"fold{fold}" / "weights" / "last.pt"
    dst = out_dir / f"fold{fold}_yolo.pt"
    shutil.copy2(best, dst)
    # ultralytics may leave deterministic/cudnn flags behind; the refiner wants the defaults
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    shutil.rmtree(out_dir / "yolo" / f"fold{fold}", ignore_errors=True)
    return dst


@torch.no_grad()
def predict_boxes(weights, files: list[str], size: int, split: str = "train",
                  conf: float = DET_CONF_MIN) -> dict[str, np.ndarray]:
    """{file: float32 (n, 5) [x0, y0, x1, y1, conf] in native px, conf-descending}."""
    from ultralytics import YOLO
    model = YOLO(str(weights))
    d = CACHE / (f"img{size}" if split == "train" else f"test{size}")
    paths = [str(d / (f.rsplit(".", 1)[0] + ".png")) for f in files]
    scale = H / size
    out = {}
    for i in range(0, len(paths), 16):
        res = model.predict(source=paths[i:i + 16], imgsz=DET_IMGSZ, conf=conf, iou=DET_NMS_IOU,
                            max_det=DET_MAX_DET, verbose=False, device=_device_arg())
        for f, r in zip(files[i:i + 16], res):
            if r.boxes is None or len(r.boxes) == 0:
                out[f] = np.zeros((0, 5), np.float32)
                continue
            xyxy = r.boxes.xyxy.float().cpu().numpy() * scale
            c = r.boxes.conf.float().cpu().numpy()
            b = np.concatenate([xyxy, c[:, None]], 1).astype(np.float32)
            out[f] = b[np.argsort(-b[:, 4])]
    return out


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (n,4), b (m,4) xyxy -> (n,m)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0]); iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2]); iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-6)


def fuse_boxes(box_sets: list[np.ndarray], iou_thr: float = DET_FUSE_IOU) -> np.ndarray:
    """Weighted boxes fusion over the fold detectors of one image. Boxes of all models are
    clustered greedily (conf-descending) at IoU > iou_thr against the running fused box;
    fused box = conf-weighted mean, fused conf = sum over models of their best conf / n_models,
    so a box only one of five detectors sees is down-weighted (fold agreement).
    Returns (n, 5) conf-descending; a single set is returned unchanged."""
    n_models = len(box_sets)
    if n_models == 1:
        return box_sets[0]
    rows = [np.concatenate([b, np.full((len(b), 1), m, np.float32)], 1) for m, b in enumerate(box_sets) if len(b)]
    if not rows:
        return np.zeros((0, 5), np.float32)
    allb = np.concatenate(rows)
    allb = allb[np.argsort(-allb[:, 4])]
    fused, members = [], []
    for b in allb:
        if fused:
            ious = box_iou(b[None, :4], np.array([f[:4] for f in fused]))[0]
            j = int(ious.argmax())
            if ious[j] > iou_thr:
                members[j].append(b)
                ms = np.array(members[j])
                wsum = ms[:, 4].sum()
                fused[j][:4] = (ms[:, :4] * ms[:, 4:5]).sum(0) / wsum
                continue
        fused.append(b[:5].copy())
        members.append([b])
    out = []
    for f, ms in zip(fused, members):
        ms = np.array(ms)
        best_per_model = {}
        for r in ms:
            best_per_model[int(r[5])] = max(best_per_model.get(int(r[5]), 0.0), float(r[4]))
        out.append([*f[:4], sum(best_per_model.values()) / n_models])
    out = np.array(out, np.float32)
    return out[np.argsort(-out[:, 4])]


def box_pr(pred: dict[str, np.ndarray], meta: dict, gt_rles: dict, conf: float,
           iou_thr: float = 0.5) -> dict:
    """Box-level precision/recall against every annotator record (greedy 1-1 at IoU > iou_thr).
    Diagnostic only: says whether the detector or the refiner is the bottleneck."""
    tp = fp = fn = 0
    for f, b in pred.items():
        b = b[b[:, 4] >= conf]
        for rid in meta["records_by_file"][f]:
            g = np.array([box_of_rle(r) for r in gt_rles[rid]], np.float32).reshape(-1, 4)
            g[:, 2:] += g[:, :2]
            m = box_iou(b[:, :4], g)
            used_g = set()
            hit = 0
            for i in np.argsort(-b[:, 4]) if len(b) else []:
                order = np.argsort(-m[i])
                for j in order:
                    if m[i, j] <= iou_thr:
                        break
                    if j not in used_g:
                        used_g.add(j); hit += 1
                        break
            tp += hit; fp += len(b) - hit; fn += len(g) - hit
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return {"conf": conf, "precision": p, "recall": r, "tp": tp, "fp": fp, "fn": fn}
