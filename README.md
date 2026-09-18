# Solar Filament Segmentation Challenge 2026

Kaggle: https://www.kaggle.com/competitions/filament-segmentation-2026
(IEEE BigData Cup 2026). Deadline **2026-11-15 06:00 UTC**, $3,000 prize.

## Task

Instance segmentation of solar filaments in 2048x2048 grayscale GONG H-alpha
full-disk images (MAGFiLO v1.0 dataset). Chirality labels exist in the
annotations but are **not** scored; only the filament masks are.

## Data (`data/MAGFiLO_1.0_Kaggle_2026/`)

| split | files | notes |
|-------|-------|-------|
| train | 707 jpeg | 1154 annotator *records* (296 files have 2-3 independent annotators), 8199 filament polygons |
| test  | 180 jpeg | scored against every annotator record of each image |

COCO-style JSON: `segmentation` (one polygon per filament), `bbox`, `spine`,
`category_id` (1 Left, 2 Right, 3 Unidentifiable, 4 Ambiguous). `images[].id`
is `<annotator>-<file_stem>`, so the same jpeg appears up to 3 times.

Filament stats at full res: bbox long side median 96 px (5-95%: 38-315),
thickness ~13 px, ~0.3% of pixels per image are filament.

## Metric: Panoptic Quality (changed from Dice on 2026-08-07)

```
PQ = sum(IoU over TP pairs) / (TP + 0.5 FP + 0.5 FN),   TP iff IoU > 0.5
```

Pooled over all annotator records of all test images. A filament predicted
as two fragments scores 0 TP + 1 FN + 2 x 0.5 FP, so **instance integrity and
FP/FN control matter more than pixel accuracy**. `src/metric.py` reproduces
the organiser's self-evaluation notebook (`azimahmadzadeh/self-evaluation-notebook`).

Inter-annotator ceiling (measured with `src/metric.py`): one annotator's masks
scored against the *other* annotators of the same files reach PQ 0.34
(SQ 0.63, RQ 0.53). Public LB top is 0.56.

## Submission

CSV, columns `filament_id` (`<file_stem>_<k>`, one row per predicted
filament) and `segmentation_rle` (pycocotools compressed RLE `counts` string of
a 2048x2048 uint8 mask). Every test image should have at least one row.

```
kaggle competitions submit -c filament-segmentation-2026 -f submissions/xxx.csv -m "msg"
```

Note: public MAGFiLO mirrors contain the test annotations. Using them is
against the rules; this repo uses only the official train JSON.

## Quick start

Read `CLAUDE.md` first: pinned competition facts, data hazards, workflow, guardrails.
`PORTING.md` is the live log, `FUTURE.md` the idea parking lot.

1. Kaggle API token at `~/.kaggle/` (username `ishansharma13` is baked into
   `notebooks/kernel-metadata.json`)
2. `./setup.sh` — conda env `filament`, data download, reference notebooks, 1024 px cache
3. `claude` — start Claude Code here

Cloud / remote session instead (no laptop): set the Kaggle credential as an environment
secret (`KAGGLE_ACCESS_TOKEN`, or `KAGGLE_USERNAME` + `KAGGLE_KEY`), then `./setup_cloud.sh`
(pip, data, cache, latest kernel outputs). Everything else is identical.

## Loop (same as ../rsna-knee)

develop in `src/` → `scripts/push.sh` (rebuilds `notebooks/submission.ipynb` from `src/`,
never edit it by hand, then `kaggle kernels push`) → `scripts/status.sh` →
`scripts/pull_logs.sh` (log, `submission.csv`, `fold*_best.pt`, `oof/*.png`) →
`scripts/sweep_postprocess.py runs/<stamp>` → `scripts/submit.sh runs/<stamp>/submission.csv "msg"`.

Local smoke test of the exact kernel path (about 3 minutes on the M1 Pro):

```
FILAMENT_DEBUG=1 conda run -n filament python -c "import sys; sys.path.insert(0,'src'); import pipeline; pipeline.run()"
```

## Layout

- `src/` — pipeline modules (config, data, rle, metric, postprocess, dataset, model,
  evaluate, pipeline); flat imports so the notebook builder can paste them
- `notebooks/` — the one generated kernel notebook + kernel metadata
- `scripts/` — build/push/status/pull/submit, local train/predict/sweep/check helpers
- `reference/` — pulled public notebooks, read-only (organiser's metric notebook included)
- `data/`, `cache/`, `runs/` — gitignored

Folds are grouped by physical jpeg so annotator duplicates never leak across
train/val. Training samples one annotator record at random per crop.

## Ideas ranked by expected PQ gain

1. Instance separation: watershed / distance-transform splitting of touching
   filaments, morphological closing to avoid fragments, tune `min_area`.
2. Detector + refiner cascade (YOLO box -> crop U-Net), what top public kernels do.
3. Higher resolution (full 2048 sliding window) for thin barbs.
4. Fold ensembling of probability maps; TTA is already in.
5. Consensus targets across annotators (medoid record) to reduce FP from
   annotator-specific faint filaments.
