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

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_util

from config import (CACHE, DET_ARCH, DET_BATCH, DET_CONF_MIN, DET_EPOCHS, DET_EXTRA_VIEWS, DET_FUSE_IOU,
                    DET_IMGSZ, DET_LABEL_MIN_SUPPORT, DET_LABEL_RECORD, DET_MAX_DET, DET_MODEL, DET_NMS_IOU,
                    DET_COPY_PASTE, DET_COPY_PASTE_MODE, DET_LABEL_SEGMENTS, DET_PSEUDO_VIEWS, DET_SOURCE,
                    DET_SUPPORT_BINS, DET_SUPPORT_CLASSES, DET_CLASS_WEIGHTS,
                    DET_VIEWS_MAX_DT_H, H, PSEUDO_BINS, PSEUDO_DETECTORS, PSEUDO_MIN_SCORE, REPO, RFDETR_BATCH, RFDETR_EPOCHS, RFDETR_EVAL_INTERVAL,
                    RFDETR_ACCELERATOR, RFDETR_EXPANDED_SCALES, RFDETR_GRAD_ACCUM, RFDETR_LR, RFDETR_RES,
                    RFDETR_SIZE, SEED, W, WORKERS, log)
from data import test_files, test_img_dir, train_files, train_img_dir
from views import COMPETITION_SITES, disk_fit, ecc_align, make_view, project_boxes, t_of


def yolo_root() -> Path:
    return CACHE / ("yolo" if DET_SOURCE == "cache" else "yolo_native")


LABEL_BOXES: dict = {}   # {train file: [(x, y, w, h) native px]} - filled by build_yolo_labels


# --- RF-DETR (v12): same labels / folds / outputs as the YOLO path --------------------------
def _rfdetr_cls():
    import rfdetr
    return {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium,
            "base": rfdetr.RFDETRBase}[RFDETR_SIZE]


def rfdetr_fold_dataset(fold: int, meta: dict, train_limit: int = 0, val_limit: int = 0) -> Path:
    """Roboflow/COCO layout RF-DETR expects: <root>/{train,valid,test}/_annotations.coco.json with
    the images beside it (symlinks to the native 2048 JPEGs). One category, id 0, 'filament'.
    test = the fold's val files (RF-DETR wants the split; run_test stays off)."""
    import json
    root = CACHE / "rfdetr" / f"fold{fold}"
    shutil.rmtree(root, ignore_errors=True)
    folds = meta["folds"]
    tr = sorted(f for f, k in folds.items() if k != fold)
    va = sorted(f for f, k in folds.items() if k == fold)
    if train_limit:
        tr = tr[:train_limit]
    if val_limit:
        va = va[:val_limit]
    n_ann = 0
    for split, fs in (("train", tr), ("valid", va), ("test", va)):
        d = root / split
        d.mkdir(parents=True)
        images, anns = [], []
        for i, f in enumerate(fs):
            os.symlink((train_img_dir() / f).resolve(), d / f)
            images.append({"id": i, "file_name": f, "width": W, "height": H})
            for x, y, w, h in LABEL_BOXES.get(f, []):
                anns.append({"id": len(anns), "image_id": i, "category_id": 0,
                             "bbox": [float(x), float(y), float(w), float(h)], "area": float(w * h), "iscrowd": 0})
        json.dump({"images": images, "annotations": anns,
                   "categories": [{"id": 0, "name": "filament", "supercategory": "none"}]},
                  open(d / "_annotations.coco.json", "w"))
        if split == "train":
            n_ann = len(anns)
    log(f"fold {fold}: RF-DETR dataset {len(tr)} train files ({n_ann} boxes) / {len(va)} val files")
    return root


def _train_rfdetr(fold: int, meta: dict, out_dir: Path, train_limit: int = 0, val_limit: int = 0) -> Path:
    ds = rfdetr_fold_dataset(fold, meta, train_limit, val_limit)
    od = out_dir / "rfdetr" / f"fold{fold}"
    log(f"fold {fold}: RF-DETR {RFDETR_SIZE} res {RFDETR_RES} epochs {RFDETR_EPOCHS} batch "
        f"{RFDETR_BATCH}x{RFDETR_GRAD_ACCUM} lr {RFDETR_LR}")
    model = _rfdetr_cls()(resolution=RFDETR_RES)
    model.train(dataset_dir=str(ds), epochs=RFDETR_EPOCHS, batch_size=RFDETR_BATCH,
                grad_accum_steps=RFDETR_GRAD_ACCUM, lr=RFDETR_LR, output_dir=str(od), tensorboard=False,
                num_workers=WORKERS, eval_interval=RFDETR_EVAL_INTERVAL, seed=SEED,
                accelerator=RFDETR_ACCELERATOR, expanded_scales=RFDETR_EXPANDED_SCALES)
    best = next((od / n for n in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth",
                                  "checkpoint_best_regular.pth", "last_ema.pth", "checkpoint.pth")
                 if (od / n).exists()), None)
    if best is None:
        raise FileNotFoundError(f"no RF-DETR checkpoint in {od}: {sorted(p.name for p in od.glob('*'))}")
    dst = out_dir / f"fold{fold}_rfdetr.pth"
    shutil.copy2(best, dst)
    shutil.rmtree(od, ignore_errors=True)
    return dst


def _load_rfdetr(weights):
    cls = _rfdetr_cls()
    try:
        return cls.from_checkpoint(str(weights))
    except Exception:  # noqa: BLE001 - older API: weights through the constructor
        return cls(pretrain_weights=str(weights), resolution=RFDETR_RES)


def _predict_rfdetr(weights, files: list[str], split: str, conf: float, model=None) -> dict:
    from PIL import Image
    m = model if model is not None else _load_rfdetr(weights)
    d = train_img_dir() if split == "train" else test_img_dir()
    out = {}
    for i in range(0, len(files), 8):
        batch = files[i:i + 8]
        imgs = [Image.open(d / f).convert("RGB") for f in batch]
        dets = m.predict(imgs, threshold=conf)
        if not isinstance(dets, list):
            dets = [dets]
        for f, det in zip(batch, dets):
            if det is None or len(det) == 0:
                out[f] = np.zeros((0, 5), np.float32)
                continue
            b = np.concatenate([np.asarray(det.xyxy, np.float32),
                                np.asarray(det.confidence, np.float32)[:, None]], 1)
            out[f] = b[np.argsort(-b[:, 4])][:DET_MAX_DET]
    return out


VIEWS_DATASET = "gong-halpha-train-views-long"


# --- v14: self-training views (pseudo-labelled by the fold-k ensemble) -----------------------
def pseudo_detectors(fold: int) -> list[Path]:
    """fold{k}_yolo.pt of every run in the detectors dataset (locally data/fold_detectors/<run>/)."""
    roots = [REPO / "data" / "fold_detectors"]
    kin = Path("/kaggle/input")
    if kin.is_dir():
        roots += [p for p in kin.glob(f"*/{PSEUDO_DETECTORS}")] + [p for p in kin.glob(f"*/*/{PSEUDO_DETECTORS}")] + \
                 [kin / PSEUDO_DETECTORS]
    for r in roots:
        ws = sorted(set(r.glob(f"*/fold{fold}_yolo.pt")) | set(r.glob(f"*/*/fold{fold}_yolo.pt")))
        if ws:
            return ws
    return []


def _predict_paths(model, paths: list[str]) -> dict:
    """{path: (n, 5) xyxy+score in the image's own px}, same scoring as predict_boxes."""
    out = {}
    for i in range(0, len(paths), 16):
        res = model.predict(source=paths[i:i + 16], imgsz=DET_IMGSZ, conf=0.05, iou=DET_NMS_IOU,
                            max_det=DET_MAX_DET, verbose=False, device=_device_arg(), agnostic_nms=DET_SUPPORT_CLASSES)
        for p, r in zip(paths[i:i + 16], res):
            if r.boxes is None or len(r.boxes) == 0:
                out[p] = np.zeros((0, 5), np.float32); continue
            c = r.boxes.conf.float().cpu().numpy()
            if DET_SUPPORT_CLASSES:
                cls = r.boxes.cls.int().cpu().numpy()
                c = c * np.asarray(DET_CLASS_WEIGHTS, np.float32)[np.clip(cls, 0, len(DET_CLASS_WEIGHTS) - 1)]
            out[p] = np.concatenate([r.boxes.xyxy.float().cpu().numpy(), c[:, None]], 1).astype(np.float32)
    return out


def build_pseudo_views(fold: int, meta: dict, train_limit: int = 0) -> Path:
    """pv_f{fold}/images/<stem>__<view>.jpg (symlinks to LUT-matched views, materialised once in
    views_lut/) + pv_f{fold}/labels/*.txt pseudo-labelled by the fold's detectors (WBF-fused, class by
    fused score). Only views of the fold's TRAIN files."""
    from ultralytics import YOLO
    root = yolo_root()
    pv = root / f"pv_f{fold}"
    shutil.rmtree(pv, ignore_errors=True)
    (pv / "images").mkdir(parents=True); (pv / "labels").mkdir()
    lut_dir = root / "views_lut"; lut_dir.mkdir(exist_ok=True)
    tr = sorted(f for f, k in meta["folds"].items() if k != fold)
    if train_limit:
        tr = tr[:train_limit]
    paths = []
    for f in tr:
        stem = f.rsplit(".", 1)[0]
        n = 0
        for v in views_of(stem):
            if n >= DET_PSEUDO_VIEWS or v.stem[14] not in COMPETITION_SITES:
                continue
            dst = lut_dir / f"{stem}__{v.stem}.jpg"
            if not dst.exists():
                img = cv2.imread(str(v), cv2.IMREAD_GRAYSCALE)
                if img is None or img.shape != (H, W):
                    continue
                cv2.imwrite(str(dst), make_view(img), [cv2.IMWRITE_JPEG_QUALITY, 95])
            link = pv / "images" / dst.name
            if not link.exists():
                os.symlink(dst.resolve(), link)
            paths.append(str(link)); n += 1
    weights = pseudo_detectors(fold)
    if not weights or not paths:
        log(f"fold {fold}: pseudo views skipped (detectors {len(weights)}, views {len(paths)})")
        return pv
    sets = []
    for w in weights:
        sets.append(_predict_paths(YOLO(str(w)), paths))
    n_box, hist = 0, {}
    for p in paths:
        fused = fuse_boxes([s[p] for s in sets])
        fused = fused[fused[:, 4] >= PSEUDO_MIN_SCORE] if len(fused) else fused
        lines = []
        for x0, y0, x1, y1, s in fused:
            c = 0 if s >= PSEUDO_BINS[0] else 1 if s >= PSEUDO_BINS[1] else 2
            if not DET_SUPPORT_CLASSES:
                c = 0
            hist[c] = hist.get(c, 0) + 1
            lines.append(f"{c} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
        n_box += len(lines)
        (pv / "labels" / (Path(p).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    log(f"fold {fold}: pseudo views {len(paths)} ({len(paths) / max(len(tr), 1):.1f}/file) labelled by {len(weights)} "
        f"detectors: {n_box} boxes ({n_box / max(len(paths), 1):.1f}/view), classes {dict(sorted(hist.items()))}")
    return pv


def views_of(stem: str) -> list[Path]:
    """Archive frames around a train file (training-time extra views), wherever they live:
    data/sequences_long/<stem>/ locally, or the attached Kaggle dataset (auto-unzipped per stem)."""
    roots = [REPO / "data" / "sequences_long"]
    kin = Path("/kaggle/input")
    if kin.is_dir():
        roots += [p for p in kin.glob(f"*/{VIEWS_DATASET}")] + [p for p in kin.glob(f"*/*/{VIEWS_DATASET}")] + \
                 [kin / VIEWS_DATASET]
    for r in roots:
        for d in (r / stem, r / stem / stem):
            if d.is_dir():
                return sorted(p for p in d.glob("*.jpg") if p.stem != stem)
    return []


def build_views(stem: str, ref_img: np.ndarray, boxes_xywh: np.ndarray, max_views: int):
    """Yield (view_name, view_image uint8 HxW, boxes xywh) for the usable archive frames of a
    train file: other competition sites at the same minute (ECC-aligned onto the reference,
    labels unchanged) and the same site within DET_VIEWS_MAX_DT_H (labels rotated)."""
    disk_ref = None
    n = 0
    for p in views_of(stem):
        if n >= max_views or p.stem[14] not in COMPETITION_SITES:
            continue
        dt_days = (t_of(p.stem) - t_of(stem)).total_seconds() / 86400
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape != (H, W):
            continue
        v = make_view(img)
        if p.stem[14] != stem[14]:
            if abs(dt_days) * 24 > 0.2:
                continue
            v, info = ecc_align(ref_img, v)
            if v is None:
                continue
            bx = boxes_xywh.copy()
        else:
            if abs(dt_days) * 24 > DET_VIEWS_MAX_DT_H:
                continue
            disk_ref = disk_ref or disk_fit(ref_img)
            bx, valid = project_boxes(boxes_xywh, disk_ref, dt_days)
            bx = bx[valid]
        n += 1
        yield p.stem, v, bx


def image_source(f: str, size: int, split: str = "train") -> Path:
    """The image the detector trains on / predicts from for file `f`."""
    if DET_SOURCE == "native":
        return (train_img_dir() if split == "train" else test_img_dir()) / f
    return CACHE / (f"img{size}" if split == "train" else f"test{size}") / (f.rsplit(".", 1)[0] + ".png")


def box_of_rle(rle) -> tuple[float, float, float, float]:
    """(x, y, w, h) in native px of one GT RLE."""
    x, y, w, h = mask_util.toBbox(rle)
    return float(x), float(y), float(w), float(h)


def union_boxes(rles_by_record: list[list], min_support: float = DET_LABEL_MIN_SUPPORT,
                iou_thr: float = 0.5, with_support: bool = False):
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
        support = len({k for k, _ in cl}) / n
        if support < min_support:
            continue
        if len(cl) == 1:
            rep = cl[0][1]
        else:
            ms = [m for _, m in cl]
            tot = np.asarray(mask_util.iou(ms, ms, [0] * len(ms))).sum(1)
            rep = ms[int(tot.argmax())]
        box = box_of_rle(rep)
        out.append((box, support, rep) if with_support else box)
    return out


def rle_polygon(rle, eps: float = 1.5) -> list[float]:
    """Largest outer contour of a mask as a flat [x1, y1, x2, y2, ...] polygon in native px
    (simplified with Douglas-Peucker at `eps`); falls back to the box corners."""
    m = mask_util.decode(rle).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cs:
        c = max(cs, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(c) >= 3:
            return [float(v) for v in c.ravel()]
    x, y, w, h = box_of_rle(rle)
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def support_class(support: float) -> int:
    """Annotator-agreement class of a label box: 0 = drawn by >= DET_SUPPORT_BINS[0] of the file's
    annotators (incl. every box of a single-annotator file), 1 = at least DET_SUPPORT_BINS[1],
    2 = fewer (a minority). The detector learns to predict it; at inference the class scales the
    confidence by DET_CLASS_WEIGHTS - the v13 'agreement-weighted' lever."""
    if support >= DET_SUPPORT_BINS[0]:
        return 0
    if support >= DET_SUPPORT_BINS[1]:
        return 1
    return 2


def build_yolo_labels(meta: dict, gt_rles: dict) -> Path:
    """images/<stem>.<ext> -> symlinks to the DET_SOURCE images (cached S x S PNGs or the native
    JPEGs); labels/<stem>.txt in normalised YOLO format. Label files are always rewritten
    (the label recipe is a config knob)."""
    root, S = yolo_root(), meta["size"]
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(exist_ok=True)
    (root / "labels.cache").unlink(missing_ok=True)   # ultralytics' scan cache of the old labels
    n_box, n_files, n_views, support_hist, class_hist = 0, 0, 0, {}, {}
    for f in train_files():
        stem = f.rsplit(".", 1)[0]
        src = image_source(f, S)
        dst = root / "images" / (stem + src.suffix)
        if not dst.exists():
            os.symlink(src.resolve(), dst)
        rids = meta["records_by_file"][f]
        if DET_LABEL_RECORD == "union" and DET_SUPPORT_CLASSES:
            # every cluster, including minority ones, each with its agreement class
            pairs = union_boxes([gt_rles[r] for r in rids], min_support=0.0, with_support=True)
            boxes = [b for b, _, _ in pairs]
            classes = [support_class(s) for _, s, _ in pairs]
            polys = [rle_polygon(r) for _, _, r in pairs] if DET_LABEL_SEGMENTS else None
        elif DET_LABEL_RECORD == "union":
            boxes = union_boxes([gt_rles[r] for r in rids])
            classes = [0] * len(boxes); polys = None
        else:
            rid = meta["medoid"][f] if DET_LABEL_RECORD == "medoid" else rids[0]
            boxes = [box_of_rle(rle) for rle in gt_rles[rid]]
            classes = [0] * len(boxes); polys = None
        LABEL_BOXES[f] = [(x, y, w, h) for x, y, w, h in boxes if w >= 1 and h >= 1]   # native xywh, reused by RF-DETR
        lines = []
        for i, ((x, y, w, h), c) in enumerate(zip(boxes, classes)):
            if w < 1 or h < 1:
                continue
            class_hist[c] = class_hist.get(c, 0) + 1
            if polys is not None:   # segment format: class + normalised polygon (copy-paste needs outlines)
                pts = polys[i]
                lines.append(f"{c} " + " ".join(f"{pts[j] / W:.6f} {pts[j + 1] / H:.6f}" for j in range(0, len(pts), 2)))
            else:
                lines.append(f"{c} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
        label_txt = "\n".join(lines) + ("\n" if lines else "")
        (root / "labels" / f"{stem}.txt").write_text(label_txt)
        n_box += len(lines); n_files += 1
        support_hist[len(rids)] = support_hist.get(len(rids), 0) + len(lines)
        if DET_EXTRA_VIEWS > 0:
            # extra training views from the archive (src/views.py): materialised as JPEGs with
            # their own (aligned / rotated) box labels; masks are never reused
            ref_img = cv2.imread(str(train_img_dir() / f), cv2.IMREAD_GRAYSCALE)
            bxy = np.array([[x, y, w, h] for x, y, w, h in boxes if w >= 1 and h >= 1], np.float32).reshape(-1, 4)
            for vname, vimg, vboxes in build_views(stem, ref_img, bxy, DET_EXTRA_VIEWS):
                vdst = root / "images" / f"{stem}__{vname}.jpg"
                if not vdst.exists():
                    cv2.imwrite(str(vdst), vimg, [cv2.IMWRITE_JPEG_QUALITY, 95])
                vl = [f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}"
                      for x, y, w, h in vboxes if w >= 1 and h >= 1]
                (root / "labels" / f"{stem}__{vname}.txt").write_text("\n".join(vl) + ("\n" if vl else ""))
                n_views += 1
    log(f"YOLO labels ({DET_LABEL_RECORD}): {n_box} boxes over {n_files} files "
        f"({n_box / max(n_files, 1):.2f}/file); boxes by n_annotators {support_hist}"
        + (f"; extra views {n_views} ({n_views / max(n_files, 1):.1f}/file)" if DET_EXTRA_VIEWS else "")
        + (f"; agreement classes {dict(sorted(class_hist.items()))} (0 high / 1 half / 2 minority)" if DET_SUPPORT_CLASSES else ""))
    return root


def fold_yaml(fold: int, meta: dict, train_limit: int = 0, val_limit: int = 0) -> Path:
    root, folds = yolo_root(), meta["folds"]
    tr = sorted(f for f, k in folds.items() if k != fold)
    va = sorted(f for f, k in folds.items() if k == fold)
    if train_limit:
        tr = tr[:train_limit]
    if val_limit:
        va = va[:val_limit]
    ext = image_source(next(iter(folds)), meta["size"]).suffix
    paths = lambda fs: "\n".join(str(root / "images" / (f.rsplit(".", 1)[0] + ext)) for f in fs) + "\n"
    train_txt = paths(tr)
    if DET_EXTRA_VIEWS > 0:   # views of TRAIN-fold files only; validation stays the labelled frames
        extra = [p for f in tr for p in sorted((root / "images").glob(f"{f.rsplit('.', 1)[0]}__*{ext}"))]
        train_txt += "\n".join(str(p) for p in extra) + ("\n" if extra else "")
    if DET_PSEUDO_VIEWS > 0:  # self-training views of this fold (built by build_pseudo_views)
        pvs = sorted((root / f"pv_f{fold}" / "images").glob("*.jpg"))
        train_txt += "\n".join(str(p) for p in pvs) + ("\n" if pvs else "")
    (root / f"fold{fold}_train.txt").write_text(train_txt)
    (root / f"fold{fold}_val.txt").write_text(paths(va))
    y = root / f"fold{fold}.yaml"
    y.write_text(f"path: {root}\ntrain: {root / f'fold{fold}_train.txt'}\nval: {root / f'fold{fold}_val.txt'}\n"
                 + ("names:\n  0: filament_high\n  1: filament_half\n  2: filament_minority\n" if DET_SUPPORT_CLASSES
                    else "names:\n  0: filament\n"))
    return y


def _device_arg():
    return 0 if torch.cuda.is_available() else "cpu"


def train_detector(fold: int, meta: dict, out_dir: Path, epochs: int = DET_EPOCHS, train_limit: int = 0,
                   val_limit: int = 0) -> Path:
    """Train the fold's detector: YOLO -> fold{k}_yolo.pt, or RF-DETR (DET_ARCH) -> fold{k}_rfdetr.pth."""
    if DET_ARCH == "rfdetr":
        return _train_rfdetr(fold, meta, out_dir, train_limit, val_limit)
    from ultralytics import YOLO
    from ultralytics import settings as yolo_settings
    yolo_settings.update({"sync": False, "wandb": False, "mlflow": False, "tensorboard": False})
    if DET_PSEUDO_VIEWS > 0:
        build_pseudo_views(fold, meta, train_limit)
    yaml = fold_yaml(fold, meta, train_limit, val_limit)
    log(f"fold {fold}: YOLO {DET_MODEL} imgsz {DET_IMGSZ} epochs {epochs}")
    model = YOLO(DET_MODEL)
    model.train(data=str(yaml), epochs=epochs, imgsz=DET_IMGSZ, batch=DET_BATCH, workers=WORKERS,
                seed=SEED, deterministic=False, project=str(out_dir / "yolo"), name=f"fold{fold}",
                exist_ok=True, plots=False, verbose=False, device=_device_arg(), single_cls=not DET_SUPPORT_CLASSES,
                flipud=0.5, fliplr=0.5, patience=epochs, save_period=-1, val=True,
                copy_paste=DET_COPY_PASTE, copy_paste_mode=DET_COPY_PASTE_MODE)
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
                  conf: float = DET_CONF_MIN, augment: bool = False, nms_iou: float = DET_NMS_IOU,
                  model=None) -> dict[str, np.ndarray]:
    """{file: float32 (n, 5) [x0, y0, x1, y1, conf] in native px, conf-descending}.
    augment=True: ultralytics flip/scale TTA. nms_iou: NMS overlap threshold.
    RF-DETR weights (*_rfdetr.pth) are dispatched to the RF-DETR path (no NMS, no TTA)."""
    if str(weights).endswith("_rfdetr.pth"):
        return _predict_rfdetr(weights, files, split, conf, model)
    from ultralytics import YOLO
    model = model if model is not None else YOLO(str(weights))
    paths = [str(image_source(f, size, split)) for f in files]
    scale = 1.0 if DET_SOURCE == "native" else H / size
    out = {}
    for i in range(0, len(paths), 16):
        res = model.predict(source=paths[i:i + 16], imgsz=DET_IMGSZ, conf=conf, iou=nms_iou,
                            max_det=DET_MAX_DET, verbose=False, device=_device_arg(), augment=augment,
                            agnostic_nms=DET_SUPPORT_CLASSES)
        for f, r in zip(files[i:i + 16], res):
            if r.boxes is None or len(r.boxes) == 0:
                out[f] = np.zeros((0, 5), np.float32)
                continue
            xyxy = r.boxes.xyxy.float().cpu().numpy() * scale
            c = r.boxes.conf.float().cpu().numpy()
            if DET_SUPPORT_CLASSES:   # agreement class scales the score: expected "would annotators draw it"
                cls = r.boxes.cls.int().cpu().numpy()
                c = c * np.asarray(DET_CLASS_WEIGHTS, np.float32)[np.clip(cls, 0, len(DET_CLASS_WEIGHTS) - 1)]
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
