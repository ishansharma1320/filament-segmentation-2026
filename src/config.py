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
