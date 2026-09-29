"""Shared constants, environment discovery and logging.

The same modules run in two places: this repo (Apple-silicon Mac, smoke tests
and post-processing sweeps) and a Kaggle GPU kernel assembled from them by
scripts/build_notebook.py. Everything path-like is resolved here.
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np
import torch

T0 = time.time()
SEED = 2026


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


ON_KAGGLE = Path("/kaggle/working").is_dir()
try:
    REPO = Path(__file__).resolve().parents[1]
except NameError:  # notebook cell: no __file__
    REPO = Path.cwd()
WORK = Path("/kaggle/working") if ON_KAGGLE else REPO
CACHE = Path("/tmp/filament_cache") if ON_KAGGLE else REPO / "cache"
RUNS = WORK if ON_KAGGLE else REPO / "runs"

COMP = "filament-segmentation-2026"
DATA_DIRNAME = "MAGFiLO_1.0_Kaggle_2026"
ANN_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def find_data_root() -> Path:
    env = os.environ.get("FILAMENT_DATA")
    cands = [Path(env)] if env else []
    cands += [REPO / "data" / DATA_DIRNAME,
              Path("/kaggle/input/competitions") / COMP / DATA_DIRNAME,
              Path("/kaggle/input") / COMP / DATA_DIRNAME]
    for c in cands:
        if (c / "train" / "train_images").is_dir():
            return c
    base = Path("/kaggle/input")
    if base.is_dir():
        for pat in (f"*/{DATA_DIRNAME}/train/{ANN_NAME}", f"*/*/{DATA_DIRNAME}/train/{ANN_NAME}"):
            hits = sorted(base.glob(pat))
            if hits:
                return hits[0].parent.parent
    raise FileNotFoundError("competition data not found (set FILAMENT_DATA)")


# --- image / crops --------------------------------------------------------------------
H = W = 2048               # native GONG frame; the metric is computed at this size
IMG_SIZE = 1024            # working resolution. v1-v3: 1024 (thickness ~13 px native -> ~6 px);
                           # v4 tried native 2048: FAIL (-0.019 OOF, SQ unchanged) - see PORTING
CROP = 512                 # training crop taken from the IMG_SIZE image
POSITIVE_BIAS = 0.7        # fraction of crops centred on a filament pixel

# --- model / training -----------------------------------------------------------------
ARCH = "unet"              # smp architecture: unet | unetpp | fpn | deeplab | manet
ENCODER = "resnet34"
EPOCHS = 32                # v1 (20 epochs): still rising; v2/v3 (32): converged by 25-31
BATCH = 8
LR = 3e-4
WEIGHT_DECAY = 1e-4
POS_WEIGHT = 2.0           # BCE positive weight; ~0.3 % of pixels are filament
SAMPLES_PER_FILE = 4       # crops drawn per file per epoch
TARGET_MODE = "random"     # "random": one annotator record per crop at random (v1/v2);
                           # "medoid": the record with the highest mean mask-IoU to the
                           # other records of the same file (consensus target, rung 5)
WORKERS = 3 if ON_KAGGLE else 4

# --- folds / budget -------------------------------------------------------------------
N_FOLDS = 5                # grouped by physical jpeg (annotator records never split)
MAX_FOLDS = "auto"         # "auto" trains folds until TIME_BUDGET would be exceeded
START_FOLD = 0
TIME_BUDGET = 8.5 * 3600   # seconds; Kaggle GPU sessions are capped at 12 h

# --- post-processing / selection ------------------------------------------------------
EPOCH_THR = 0.5            # threshold used for per-epoch model selection (no TTA)
MIN_AREA = 150             # component area (native px) below which a prediction is dropped
THRESHOLDS = (0.55, 0.65, 0.75)         # final OOF sweep grid (v1 sweeps: plateau 0.6-0.7) ...
MIN_AREAS = (300, 450)                  # ... x this (v1: 300-400 best, 600 too aggressive) ...
MIN_SCORES = (0.8, 0.9)                 # ... x per-component mean-prob floor (v1 OOF: peak 0.90,
                                        #     +0.015 over none; 0.92 already collapses)
CLOSE_KERNEL = 5           # morphological closing before components (v1 OOF: +0.002)
TTA = True                 # 4-way flip TTA for OOF and test

# --- smoke mode -----------------------------------------------------------------------
DEBUG = os.environ.get("FILAMENT_DEBUG", "0") == "1"
MAX_STEPS = 0              # 0 = full epoch
EVAL_FILES = 0             # 0 = whole validation fold
if DEBUG:
    EPOCHS, MAX_STEPS, EVAL_FILES, MAX_FOLDS = 1, 20, 8, 2
    THRESHOLDS, MIN_AREAS, MIN_SCORES = (0.45, 0.55), (150,), (0.0,)
    WORKERS = 0            # macOS spawn cannot pickle notebook-namespace classes; Kaggle forks
    TTA = False            # 4 passes x 180 test images at 2048 is too slow for a smoke test

# --- rung 4: detector -> crop-refiner cascade ------------------------------------------
# PIPELINE selects the recipe: "semantic" = v1-v4 (U-Net prob map -> components);
# "cascade" = YOLO box detector -> per-box crop refiner -> one mask per detection.
PIPELINE = "cascade"
DET_ARCH = "rfdetr"         # v12: "rfdetr" (query-based, no NMS, DINOv2 backbone; RF100-VL small-object AP
                            # 35.8 vs YOLO11's 7.3) | "yolo" (v5-v11). Same union labels, folds, refiner.
RFDETR_SIZE = "medium"      # nano | small | medium | base (Apache-2.0 variants)
RFDETR_RES = 896            # must be divisible by patch 16 x 2 windows = 32 (and 64 for square resize);
                            # median filament 96 native px -> 42 px. v12a at 1024 + expanded scales (train sizes
                            # up to 1184) went OOM on the T4 at batch 2
RFDETR_EPOCHS = 25            # fine-tuned from the COCO-pretrained RF-DETR; 40 at 1024 px risks the 12 h cap
RFDETR_BATCH = 1             # per step (T4 16 GB); effective batch via accumulation
RFDETR_GRAD_ACCUM = 16      # effective batch 16 (RF-DETR recommended total)
RFDETR_EXPANDED_SCALES = False  # multi-scale jitter 800-1024 at res 896 (expanded would reach 1056)
RFDETR_LR = 1e-4
RFDETR_EVAL_INTERVAL = 5    # validation every N epochs (saves ~20 % of T4 time)
RFDETR_ACCELERATOR = "auto" # Lightning accelerator; the local Mac smoke test uses "cpu" (MPS AMP needs float64)
DET_MODEL = "yolo11s.pt"    # v10 tried yolo11m: WASH (0.4406 vs 0.4449, contested recall flat). COCO-pretrained
DET_IMGSZ = 1280            # v5-v8: 1024 on the cached PNGs (no resize). v9: 1280 from the native
                            # 2048 JPEGs (small filaments: consensus recall 0.79 < 64 px vs 0.89)
DET_SOURCE = "native"       # "cache": img{IMG_SIZE}.png (v5-v8) | "native": the 2048 JPEGs
DET_EPOCHS = 40             # YOLO epochs (v11 used 15 with views)
DET_BATCH = 8
DET_LABEL_RECORD = "union"  # YOLO labels are static files. "medoid": the consensus record (v5).
                            # "union": every record's filaments, deduplicated across annotators at
                            # mask IoU > 0.5, kept when drawn by >= DET_LABEL_MIN_SUPPORT of the
                            # file's annotators (v6; the pooled metric rewards predicting a filament
                            # iff support > PQ / (2 SQ) ~ 0.32, see PORTING 2026-09-19)
DET_LABEL_MIN_SUPPORT = 0.5
DET_EXTRA_VIEWS = 0         # v11 lever (FAIL, -0.011): archive views per TRAIN-fold file added to the detector's
                            # training list (val = the labelled frames only). 0 = off (v5-v10);
                            # N > 0 = up to N views per file from data/sequences_long/<stem>/ or the
                            # attached Kaggle dataset ishansharma13/gong-halpha-train-views-long:
                            # other competition sites at the same minute (ECC-aligned, labels
                            # unchanged) and the same site within DET_VIEWS_MAX_DT_H hours (labels
                            # rotated). Views are LUT-matched to the competition processing
                            # (src/views.py). Host ruling: sequences at training time only.
DET_VIEWS_MAX_DT_H = 1.5    # same-site views beyond this are dropped: label fidelity vs the v9
                            # detector is 0.47 at 1 h, 0.39 at 3 h, 0.31 at 6 h (original frame 0.59)
DET_CONF_MIN = 0.05         # v12: 0.05 (RF-DETR scores sit on a different scale than YOLO; v5-v11: 0.1)
DET_NMS_IOU = 0.5
DET_MAX_DET = 100
DET_CONFS = (0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6)   # OOF sweep: per-instance existence cut (the FP/FN knob)
DET_FUSE_IOU = 0.55         # test time: boxes of the fold detectors are fused (WBF) at this IoU
REF_CROP = 256              # refiner input side in native px; boxes whose context exceeds it
REF_CONTEXT = 1.4           #   (REF_CONTEXT x long side) are downscaled, probs upscaled back
REF_PAD = 0.25              # seed channel = detector box grown by this fraction per side
REF_ENCODER = "resnet34"
REF_EPOCHS = 12
REF_BATCH = 16
REF_LR = 3e-4
REF_THRS = (0.6,)           # OOF sweep: refiner mask threshold (v5-v11: flat across 0.5-0.7) ...
REF_MIN_AREAS = (100,)      # ... x minimum instance area (flat 100 vs 200)
REF_NEG_FRAC = 0.0          # v7/v8: 0.25 (hard negatives, WASH both times); 0 = the v6 refiner.
                            # fraction of refiner training samples that are hard negatives -
                            # detector OOF boxes with IoU < REF_NEG_IOU to every GT box of every
                            # record of the file, target all-zero - so the refiner learns to reject
                            # detector FPs (v5/v6 refiner: positives only, 0.95 mean prob on FPs)
REF_NEG_IOU = 0.3
REF_NEG_CONF = 0.2          # only FP boxes above this conf are used (the ones that reach the cut)
REF_OBJ_HEAD = False        # v8: True (objectness head; AUC 0.658 vs detector conf 0.774: WASH)
DUP_FRACTION = 0.5          # a refined mask with more than this fraction of its pixels already
                            # taken by a higher-confidence instance is a duplicate: dropped
if DEBUG:
    DET_EPOCHS, DET_BATCH, REF_EPOCHS = 1, 4, 1
    RFDETR_EPOCHS, RFDETR_RES, RFDETR_BATCH, RFDETR_GRAD_ACCUM, RFDETR_EVAL_INTERVAL = 1, 512, 2, 1, 1
    if not ON_KAGGLE:
        RFDETR_ACCELERATOR = "cpu"
    DET_CONFS, REF_THRS, REF_MIN_AREAS = (0.2, 0.3), (0.5,), (100,)
