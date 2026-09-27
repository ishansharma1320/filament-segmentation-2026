# Solar Filament Segmentation Challenge 2026 — Project Memory

Kaggle competition: `filament-segmentation-2026` (IEEE BigData Cup 2026 track).
Owner: Ishan (Kaggle `ishansharma13`). Final deadline: **2026-11-15 06:00 UTC**. $3,000 prize.
Process mirrors `../rsna-knee` (same owner): src/ is the truth, the notebook is generated,
every kernel push is a commit, PORTING.md is the live log.

## The task

Instance segmentation of solar filaments in 2048x2048 grayscale GONG H-alpha full-disk
images (MAGFiLO v1.0). Chirality labels (`category_id` 1 Left / 2 Right / 3 Unidentifiable /
4 Ambiguous) exist in the JSON but are **not scored**. Only masks are.

Submission: CSV with `filament_id` (`<file_stem>_<k>`, one row per predicted filament) and
`segmentation_rle` (pycocotools compressed RLE `counts` string of a 2048x2048 uint8 mask).
CSV upload competition (NOT a code competition): submit with the CLI, `scripts/submit.sh`.

**Metric: Panoptic Quality, pooled over every annotator record of every test image**
(changed from Dice on 2026-08-07; reference implementation is the organiser's
`reference/self-evaluation-notebook.ipynb`, reproduced byte-for-byte in `src/metric.py`):

    PQ = sum(IoU of TP pairs) / (TP + 0.5 FP + 0.5 FN),   TP iff IoU > 0.5 (strict)

Consequences (do not re-derive these, do not violate them):
- One filament predicted as two fragments = 0 TP + 1 FN + 2 x 0.5 FP. Fragmentation is
  the most expensive failure. Instance integrity beats pixel accuracy.
- Every extra predicted blob costs 0.5 FP; every missed filament 0.5 FN. FP/FN balance is
  a first-class knob (threshold, min_area), selected on OOF PQ, never on Dice.
- Boundary accuracy only matters once IoU clears 0.5, then linearly (SQ).
- Pixel Dice / BCE are training losses only. Model selection = OOF PQ.
- **Predicted instances of one image must not overlap by a single pixel.** A submission
  with 2 overlapping px in 1 image came back `SubmissionStatus.ERROR` (2026-09-10).
  Connected-component instances are safe; anything that grows masks must clip against
  pixels already taken. `scripts/check_submission.py` asserts zero overlap.
- OOF is very boundary-sensitive: growing or shrinking every mask by 1 px costs 0.02 PQ,
  2 px costs 0.09. Keep post-processing at native resolution and never resample masks.

## Data facts (measured, from the corpus — treat as ground truth)

- 707 train jpegs, **1154 annotator records** (411 files x1, 145 x2, 151 x3 annotators),
  8199 filaments, ~7 per record, min 1 max 26. 180 test jpegs, same site/year mix.
- All images 2048x2048 8-bit grayscale. Sites (letter before `h` in the filename):
  B, C, L, M, T, U — exposure/contrast differ per site. Years 2011-2022, solar-max heavy.
- Filament bbox long side: median 96 px, 5-95% 38-315 px, max ~500. Thickness ~13 px.
  ~0.3% of pixels are filament (5-95%: 0.035-0.85%). Class imbalance is extreme.
- All filaments within r <= 868 px of image centre (disk radius ~ 900 px). Nothing off-disk.
- **Inter-annotator ceiling (measured with src/metric.py): one annotator's masks scored
  against the OTHER annotators of the same files reach PQ 0.34 (SQ 0.63, RQ 0.53).**
  Annotators disagree on filament count by a median of 2 and up to 13 per image. Scored
  against all records including their own: 0.76. Public LB top is 0.56. Label noise, not
  model capacity, is the dominant limit: optimise for consensus, control FP/FN.
- **Folds are grouped by physical jpeg** (`data.make_folds`, seed 2026): annotator
  duplicates never straddle train/val. Training samples one annotator record per crop
  at random (label augmentation). Never split by COCO `image_id`.
- Public MAGFiLO mirrors on Kaggle contain the **test annotations**. Using them is a rules
  violation. This repo reads only the official train JSON. Never attach such a dataset.
- **Unlabelled GONG H-alpha images are allowed.** Official rules §6 (confirmed by Ishan
  from the Kaggle rules page, 2026-09-25): External Data may be used if it is "publicly
  available and equally accessible to use by all Participants ... at no cost", and "the use
  of external data and models is acceptable unless specifically prohibited by the Host".
  The NSO GONG archive is public, free and unrestricted, so it qualifies. Host clarification
  (digested in srivatsav-kannan/solar-seg `docs/competition.md`): unlabelled GONG images
  are fine "for self-supervised pretraining, with no test-label leakage and supplied-image
  inference". Pretrained *filament* weights of unverifiable provenance are not.
  Final inference must run on the supplied test images only.
  **Host ruling on sequences (forum, 2026-09-26, asked by Ishan):** "Use of sequences of
  H-Alpha images is perfectly fine. Of course, your model may not depend on any 'sequence'
  of observations during inference." So: neighbouring archive frames may be used at
  TRAINING time (extra views, persistence-derived teacher labels, pseudo-labels), never as
  inference input. Test-time persistence voting is OFF the table. The host asked to see
  the report regardless of score. The archive is open and
  huge: `https://gong2.nso.edu/HA/hag/<YYYYMM>/<YYYYMMDD>/<stamp><site>h.jpg`, ~3,800 JPEGs
  per day, 196 months since 2010, same 2048x2048 format and filenames as the competition
  data (the train JSON's `url` field points there). Sites S and Z exist there beyond our
  B/C/L/M/T/U. Team size <= 5, 5 submissions/day, 2 final submissions.

## Reference implementations (in `reference/`, read-only)

- `self-evaluation-notebook.ipynb` (organiser, Azim Ahmadzadeh): the metric.
- `solar-filaments-u-net-baseline.ipynb`: U-Net++ semantic + connected components, 0.51.
- `lb-0-70-topology-safe-solar-filaments-handoff.ipynb`, `best-score-magfilo-yolo11s-detect-crop-u-net.ipynb`:
  YOLO detector -> per-box crop refiner cascades (top public approach after the PQ switch).
- `new-metric-solar-filament.ipynb`: PQ discussion. Do not edit files in `reference/`.

## Workflow

Local machine (M1 Pro, 17 GB, MPS) is for smoke tests and offline post-processing sweeps
(~8 min/epoch — never for real training). The loop is:
1. Develop in `src/`. `FILAMENT_DEBUG=1` runs the whole pipeline in ~3 min locally.
2. `scripts/push.sh` -> rebuilds `notebooks/submission.ipynb` from `src/` and pushes
   (`kaggle kernels push`, metadata in `notebooks/kernel-metadata.json`, T4, internet ON
   for pip + ImageNet encoder weights). Never hand-edit the notebook.
3. `scripts/status.sh` / `scripts/pull_logs.sh` -> poll, retrieve log + `submission.csv`
   + `fold*_best.pt` + `oof/*.png` + `summary.json` into `runs/<stamp>/`.
4. `scripts/sweep_postprocess.py runs/<stamp>` -> threshold / min_area / closing on OOF.
5. `scripts/submit.sh runs/<stamp>/submission.csv "msg"` -> CLI upload (validates first).
6. **Commit at every kernel push AND `git push origin main`** (message names the kernel
   version and what changed). Weights, data, cache, runs/ stay gitignored. Remote:
   `github.com/ishansharma1320/filament-segmentation-2026` (private).

**Remote / cloud sessions (hands-free, no laptop).** The repo is designed to run the whole
loop from a cloud Claude Code session: `./setup_cloud.sh` (pip deps, Kaggle credential from
the `KAGGLE_ACCESS_TOKEN` or `KAGGLE_USERNAME`+`KAGGLE_KEY` secrets, data download, cache,
pull of the latest kernel outputs). No GPU is needed there: training is on Kaggle; the local
smoke test (`FILAMENT_DEBUG=1 python3 scripts/run_notebook_local.py`) runs on CPU in
~20-30 min. Start such a session with "continue from PORTING.md" and the current rung.
The laptop and the cloud must not both drive the kernel at once (a push cancels a running
version): whoever pushes owns the run until it is pulled and logged.

Kernel modes: no checkpoints attached -> train folds until `TIME_BUDGET` (8.5 h), rewrite
`submission.csv` after every fold. A dataset of `fold*_best.pt` attached -> inference only.
Pushing while a version runs CANCELS it: check status first. Kaggle GPU quota ~30 h/week.

## Scope guardrails

- Every change is gated on OOF PQ over all annotator records (grouped folds), logged in
  PORTING.md with its delta, including failures. LB is confirmation, not selection.
  Measured 2026-09-10: public LB ≈ OOF − 0.06 and it tracks OOF under perturbations
  (±1 px probes), so the offset is the test set, not the evaluator. 5 submissions/day.
- Single-fold PQ differences under ~0.01 are noise at this label quality; use >= 2 folds
  or the full 5-fold OOF before promoting a change.
- Any idea not on the current rung goes to `FUTURE.md`, one line, move on.
- Nothing that bills Ishan runs without explicit sign-off; Kaggle GPU is the free budget.
