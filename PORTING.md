# Porting log — filament-segmentation-2026

Plain-language record of what was done and why, kept up to date as the work happens.

## Plan of record (rungs; each session moves one or re-ranks them)

1. **Milestone 1 — baseline on the leaderboard.** U-Net/ResNet34 semantic segmentation at
   1024 px, connected-components instances, 5 grouped folds on Kaggle T4, OOF-selected
   threshold/min_area, fold-ensemble submission. Expect ~0.45-0.52 (public baselines with
   this recipe: 0.51 at 512 px).
2. Post-processing for PQ on saved OOF maps (local, free): closing kernel, min_area,
   watershed splitting of merged filaments, hole filling. Log each delta.
3. Resolution: 2048 native sliding-window inference / 1024 crops of native images.
4. Detector -> crop-refiner cascade (the public 0.55+ recipe under PQ).
5. Consensus targets (medoid annotator record) vs random-record sampling.
6. Ensembling across architectures/encoders; TTA is already in.

## 2026-09-08 — repo bootstrap (no kernel pushed yet)

**What I found before writing code.** Kaggle pages are JS-rendered, so competition facts
came from the CLI (`kaggle competitions files/leaderboard`), the organiser's
self-evaluation notebook and the top public kernels (pulled to `reference/`). Metric is
instance Panoptic Quality with a strict IoU > 0.5 match, pooled over every annotator
record; the submission is a CSV of COCO-compressed RLEs. Details pinned in CLAUDE.md.

**Measured.** 707 files / 1154 records / 8199 filaments; ~0.3% positive pixels; filament
long side median 96 px. Inter-annotator oracle PQ = 0.34 against other annotators
(0.76 including self) — label noise dominates. Public LB top 0.56 (540 teams).

**Structure (mirrors ../rsna-knee).** `src/` modules with flat imports so
`scripts/build_notebook.py` can paste them into one notebook and strip sibling imports:

| module | job |
|---|---|
| `config.py` | env discovery (repo vs `/kaggle/input`), all knobs, `log()`, DEBUG mode |
| `data.py` | COCO parsing, polygon -> RLE/mask, grouped folds by physical jpeg |
| `rle.py` | submission RLE encode/decode |
| `metric.py` | PQ exactly as the organiser's notebook |
| `postprocess.py` | prob map -> instances (threshold, closing, 8-CC, min_area, never-empty guard) |
| `dataset.py` | 1024 px cache builder, random-record crop dataset, full-image dataset |
| `model.py` | smp models, BCE+Dice, TTA, checkpoint save/load with metadata |
| `evaluate.py` | model -> probs -> instances -> PQ; OOF map capture |
| `pipeline.py` | `main()`: placeholder -> cache -> folds -> time-budgeted training -> OOF sweep -> ensemble -> `submission.csv`; `run()` crash guard |

**Verified locally (M1 Pro, MPS).** Metric self-consistency (record vs itself = 1.0);
40-step smoke train -> valid 180-image submission passing `check_submission.py`;
the assembled notebook executed cell-by-cell in one namespace under `FILAMENT_DEBUG=1`.

**Decisions.**
- 1024 px working resolution (thickness ~13 px native -> ~6 px) with 512 crops, 70% of
  crops centred on a filament pixel; full-image inference at 1024, upsample probs to 2048
  before thresholding.
- One annotator record drawn at random per crop instead of union/medoid — cheapest
  unbiased treatment of multi-annotator files; medoid is rung 5.
- Per-epoch selection at thr 0.5 / min_area 150 without TTA (fast); final OOF sweep
  (4 thresholds x 2 min_areas) with TTA; the selected pair is stamped into the checkpoints
  so an inference-only run reuses it.
- OOF prob maps saved as uint8 PNGs at 1024 px so post-processing sweeps run locally.
- Internet ON in the kernel (pip smp + ImageNet weights); this is a CSV competition, so no
  code-competition constraints apply.

## Run history

| kernel version | date | change | OOF PQ | LB |
|---|---|---|---|---|
| — | — | — | — | — |
