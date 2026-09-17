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
| v1 | 2026-09-09 | Milestone-1 training push: U-Net/ResNet34 @1024, 512 crops, 20 epochs/fold, 5 grouped folds under 8.5 h budget, OOF sweep, fold ensemble | **0.3883** (thr 0.65, min_area 250) | **0.33** (public) |
| v1 + local post-proc | 2026-09-10 | same maps, thr 0.6 / min_area 300 / close 5 / min_score 0.9 (offline sweep) | **0.4086** | — (not submitted; test maps not saved by v1) |
| v2 | 2026-09-10 | EPOCHS 20→32; in-kernel sweep thr {0.55,0.65,0.75} x min_area {300,450} x min_score {0.8,0.9}, close 5; saves test/ prob maps | **0.4100** (thr 0.65, min_area 300, min_score 0.9) | **0.36** (public) |
| v3 | 2026-09-11 | TARGET_MODE medoid (consensus annotator record per file instead of random record); hard no-overlap guard + hole-fill fix in postprocess (neutral on v2 OOF: 0.4100) | 0.4082 (same knobs as v2) | **0.36** (= v2) |
| v4 | 2026-09-17 | Rung 3: native 2048 resolution (IMG_SIZE 1024→2048, same 512 crops so each crop sees 1/4 the context, batch-1 full-frame inference, float16 test accumulation); EPOCHS 28; TARGET_MODE random | pending | pending |

## 2026-09-17 — v3 verdict: medoid targets WASH (runs/20260917-143130)

Same recipe as v2 with each multi-annotator file trained on its consensus (medoid) record
instead of a random record. OOF at the same selected knobs: **0.4082 vs 0.4100** (-0.002,
inside noise). Per-fold raw (thr 0.5 / min_area 150, TTA): 0.381 / 0.360 / 0.374 / 0.387 /
0.381 vs v2 0.376 / 0.355 / 0.375 / 0.392 / 0.386 — three up, two down, all < 0.01.
FP/FN at the optimum essentially unchanged (3562/3080 vs 3561/3064). Validity: single
variable (TARGET_MODE), same folds, same sweep grid, training converged (best epochs
25-31). **Verdict: WASH. Rung 5 closed; TARGET_MODE reverts to "random".** Reading: the
consensus record removes ~1/3 of the label noise on 296 files, but the semantic model's
errors are not annotator-choice errors — they are merges, fragments and loose boundaries.

## 2026-09-10 — v2 results (runs/20260910-190750)

**Run.** 5.65 h on a T4 (~62 min/fold, sweep 21 min). Per-fold at thr 0.5 / min_area 150
(no TTA -> TTA): fold0 0.368 -> 0.376, fold1 0.348 -> 0.355, fold2 0.370 -> 0.375, fold3
0.387 -> 0.392, fold4 0.374 -> 0.386. Versus v1 (20 epochs): +0.004 to +0.013 raw. Best
epochs 23-31, last-5 means within 0.003 of best: **32 epochs is converged; more epochs are
not the lever.**

**Sweep (12 combos).** thr 0.65 / min_area 300 / min_score 0.9 / close 5 -> **0.4100**
(SQ 0.674, RQ 0.608, TP 5135 / FP 3561 / FN 3064). thr 0.55 with the same knobs 0.4089;
thr 0.75 loses 0.015. **After post-processing, v2 beats v1-with-the-same-post-processing by
only +0.0014 — noise.** The training-length rung is closed: the model family is the limit.

**Reading of the FP/FN balance.** At the optimum, ~40% of predictions are unmatched and
~37% of GT filaments are missed. SQ 0.67 means matched masks are loose. Both point at the
next rungs: instance-level quality (rung 4, detector -> refiner) and consensus targets
(rung 5) rather than more of the same semantic model.

## 2026-09-10 — v1 results (runs/20260910-111121)

**Run.** 4.0 h on a T4: cache 4.5 min, ~45 min per fold (epoch ~2 min incl. the 141-file
no-TTA eval), OOF sweep 21 min. All 5 folds trained; time budget was not hit.

**Per-fold** (per-epoch selection at thr 0.5 / min_area 150, no TTA -> with TTA):
fold0 0.365 -> 0.372, fold1 0.349 -> 0.353, fold2 0.361 -> 0.367, fold3 0.372 -> 0.379,
fold4 0.373 -> 0.380. Folds 1, 3, 4 were still rising at epoch 19: **20 epochs is short**.
TTA is worth ~+0.006.

**OOF sweep (5-fold, 1154 records).** PQ rises monotonically with threshold and min_area
across the whole grid and peaks at the corner (thr 0.65, min_area 250): 0.3883, SQ 0.671,
RQ 0.579, TP 5310 / FP 4831 / FN 2889. FP is the problem: the model emits ~1.7 blobs per GT
filament; FN barely moves with threshold. The optimum is outside the grid -> local sweeps
at thr 0.7-0.8, min_area up to 600, plus closing (rung 2).

**Submission.** 1580 rows over 180 images (8.8/image; GT is ~7/record). Submitted as v1.

**LB: 0.33 public vs 0.388 OOF.** Diagnostics on the OOF maps (`scripts/diagnose_oof.py`,
thr 0.7 / min_area 400): pooled 0.3937, mean-per-record 0.3932, mean-per-file 0.3916 —
aggregation choice cannot explain it. Single-annotator files 0.398 vs multi 0.391 — nor
annotator mix. Random 90-file subsets: sd 0.011 (5-95%: 0.376-0.410) — nor public-subset
noise (0.33 is ~5 sd out). No FP lies beyond r = 868 (FPs sit nearer the limb than TPs:
median r 613 vs 532, but on-disk). So the gap is systematic and unexplained; leading
suspects: boundary/rasterisation bias between our masks and the organiser's GT RLEs (SQ is
0.67, so many matches sit just above IoU 0.5), or a test annotation set unlike train.
Probe: per-component dilation/erosion swept on OOF, then one LB submission of the v1
masks dilated by the OOF-best amount — if LB moves much more than OOF, boundary bias it is.

**Local post-processing sweeps (rung 2, `scripts/sweep_postprocess.py`).**
thr 0.6 / min_area 300 / close 5: 0.3941; thr 0.7 / min_area 400: 0.3937; thr 0.8 or
min_area 600 fall off (FN grows faster than FP shrinks). Closing +0.002. Plateau ≈ 0.394
(+0.006 over the kernel's corner pick).

**Per-component mean-probability floor (`min_score`, rung 2) — PASS, +0.015.** At thr 0.6 /
min_area 300 / close 5: none 0.3941, 0.70 0.3942, 0.75 0.3978, 0.80 0.3991, 0.85 0.4029,
0.88 0.4054, **0.90 0.4086** (FP 3248, FN 3205 — balanced for the first time), 0.92 0.3886,
0.94 0.2648. Sharp peak: the knob must be re-swept for every new model. Validity: same OOF
maps, single variable, monotone until the cliff.

**Dilation / erosion (boundary probe).** Same settings: -2 px 0.259, -1 px 0.374, 0 px
0.394, +1 px 0.371, +2 px 0.307, +3 px 0.246. OOF sits on a knife edge at 0 px, so a
rasterisation offset in the organiser's GT would cost a lot. LB probe: v1 masks +1 px.
First attempt ERRORED — dilation made two instances share 2 px in one image, and the
evaluator rejects overlap (see CLAUDE.md). Resubmitted with overlaps clipped.

**Probe verdict — boundary bias RULED OUT.** LB: v1 0.33, +1 px 0.32, -1 px 0.31; OOF
predicted -0.02 for both (LB shows 2 decimals). The leaderboard responds to boundary
perturbations exactly as our OOF does, so (a) the evaluator matches `src/metric.py` and the
organiser's GT rasterisation matches ours, and (b) the 0.06 OOF→LB offset is a property
of the test annotations (harder images / a different annotator pool), i.e. roughly a
constant shift. **OOF remains the selector; expect LB ≈ OOF - 0.06.** Three submissions
spent on this; the daily limit is 5.

**Lessons (2).** A cleanup `rm -rf runs/2026*-1[0-9]...` meant for local debug runs also
matched the pulled kernel run and deleted 467 MB of v1 outputs mid-analysis (re-pulled
to `runs/20260910-125133`). Local runs now get a `local-` prefix; never glob-delete runs/.

**Lessons.** (1) `kaggle kernels output` paginates at 20 files; with 707 OOF PNGs the
submission/summary/log never arrived until `pull_logs.sh` looped over page tokens (fixed).
(2) OOF PQ 0.388 vs inter-annotator 0.34-vs-others: we're already above raw human
agreement on the noisy files, consistent with the model learning the consensus.
