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
4. Detector -> crop-refiner cascade (the public 0.55+ recipe under PQ). **DONE v5: 0.444 / LB 0.39.**
5. Consensus targets (medoid annotator record) vs random-record sampling.
6. Ensembling across architectures/encoders; TTA is already in.
7. Detector recall (the v5 bottleneck: box recall 0.62 at the selected conf). Order set by the
   miss analysis of 2026-09-19: (a) union-of-records labels, (b) bigger YOLO / imgsz 1280 /
   more epochs, (c) detector flip TTA. **CLOSED 2026-09-20: (a) v6 wash, (b) v9 wash,
   (c) same P/R curve; hard negatives v7 / objectness v8 wash. Ledger in the v9 verdict.**
8. Test-time fusion of every pulled fold detector (WBF) under the latest refiners: +0.005 OOF,
   LB 0.39 -> 0.40, free (CPU, `scripts/fuse_runs_test.py`). Any future run adds 5 detectors.

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
| v4 | 2026-09-17 | Rung 3: native 2048 resolution (IMG_SIZE 1024→2048, same 512 crops so each crop sees 1/4 the context, batch-1 full-frame inference, float16 test accumulation); EPOCHS 28; TARGET_MODE random | 0.3910 (thr 0.55, min_area 300, min_score 0.9) | not submitted (FAIL) |
| v5 | 2026-09-18 | **Rung 4: detector -> crop-refiner cascade** (PIPELINE=cascade). YOLO11s boxes @1024 on the medoid record, 40 epochs/fold; per-box 256 px native crop refiner U-Net/R34 with [img, CLAHE, box seed] input, 12 epochs on every (record, filament); OOF sweep conf x thr x min_area; test boxes fused across folds (WBF) | **0.4441** (conf 0.3, thr 0.6, min_area 100) | **0.39** (public) |
| v6 | 2026-09-19 | Detector labels: union of annotator records (dedup at mask IoU 0.5, one filament per record per cluster, kept when >= 1/2 of the file's annotators drew it): 5421 boxes vs medoid 5222 (+1/2 and 2/3-support filaments added, medoid-only 1/3-support ones dropped). YOLO11s @1024 / 40 epochs, refiner and sweep unchanged | 0.4441 (conf 0.3, thr 0.5, min_area 100) — WASH | not submitted (= v5) |
| v7 | 2026-09-19 | Refiner hard negatives: 25 % of refiner samples are detector OOF FP boxes (conf >= 0.2, box IoU < 0.3 to every annotator) with an all-zero target; pipeline reordered (all 5 detectors, then 5 refiners). Detector = v6 (union labels), sweep unchanged | 0.4450 (conf 0.3, thr 0.5, min_area 100) — WASH; neg-reject rate 0.000: the refiner never learned to reject | not submitted |
| v8 | 2026-09-19 | Refiner objectness head (smp aux head, one logit per crop, BCE vs positive/negative) + negatives get pixel-BCE-only seg loss (no Dice on empty targets); objectness saved per crop for offline rescoring (score = conf x obj^b). Same negatives, detector, sweep as v7 | 0.4421 in-kernel / 0.4432 offline (conf 0.25) — WASH/FAIL; obj AUC 0.658 vs conf 0.774, rescoring with obj only loses | not submitted |
| v9 | 2026-09-19 | Detector at imgsz 1280 from the native 2048 JPEGs (DET_SOURCE native; v5-v8 trained on the 1024 cache); refiner reverted to v6 (no negatives, no objectness head); union labels, YOLO11s, 40 epochs, sweep unchanged | 0.4449 in-kernel / 0.4456 offline (conf 0.25) — WASH; box recall +0.016 at equal precision, small-consensus cell unchanged (0.79 -> 0.76, n=372) | not submitted |
| v10 | 2026-09-28 | Detector capacity: YOLO11s -> **YOLO11m** (batch 8 -> 4 for the T4), everything else = v9 (imgsz 1280 native, union labels, 40 epochs, v6 refiner). Judged on OOF PQ and box recall on contested filaments | 0.4406 (conf 0.3 / thr 0.6 / min_area 100) | not submitted (WASH) |
| v11 | 2026-09-28 | **Archive views for the detector**: v9 base (YOLO11s @1280 native, union labels) + up to 8 LUT-matched views per train file (other competition sites at the same minute, ECC-aligned; same site <= 1.5 h, labels rotated) = ~1,730 extra training images; DET_EPOCHS 40 -> 15 (~3.4x data). Dataset gong-halpha-train-views-long attached. Refiner = v6 | 0.4341 (conf 0.3 / thr 0.6) | not submitted (FAIL) |
| fused v5-v9 | 2026-09-20 | No kernel run: the 25 pulled fold detectors (v5-v9) fused by WBF on the test images, v9's 5 refiners with flip TTA, cut 0.25 / thr 0.5 / min_area 100 (`scripts/fuse_runs_test.py`, CPU) | **0.4485** (5-run fusion on OOF, v9 refiner, no TTA) | **0.40** (public) |

## 2026-09-29 — where we sit: research survey + error decomposition against a human baseline

**Error decomposition, v9 OOF at the selected cut vs a second human annotator** (every
ordered annotator pair on the 296 multi-annotator files, same categories; per TP):

| error type | model | human vs human |
|---|---|---|
| near-miss: prediction on a drawn filament but IoU 0.2-0.5 (counts FP + FN) | 0.227 | 0.454 |
| isolated extra object (no drawn filament overlaps) | 0.172 | 0.331 |
| drawn filament no prediction touches (complete miss) | **0.383** | **0.331** |

Model FP 2,184 = 52 % near-miss, 39 % isolated, 7 % touching, 2 % fragments; FN 3,230 = 35 %
near-miss, 6 % weak overlap, 59 % untouched; 223 merges (one prediction over >= 2 drawn).
**Reading.** Against a second expert, the model already has HALF the outline disagreements
and HALF the extra objects; it is worse on one thing only: complete misses, 16 % more than a
human (1,905 vs ~1,645 at our TP count). Bringing complete misses to the human rate at equal
precision is worth ~+0.014 PQ (3414+169 / 7676+130 = 0.459). That, not the 0.1 of the
self-matching oracle, is the realistic model-side headroom.

**Research survey (three agents, sources in the session log; summary).**
- *This competition / MAGFiLO:* no verifiable honest work beats our OOF 0.449. Closest
  peers: Anon-Tokyo OOF 0.4427 / LB 0.38 (ConvNeXt proposals -> refiners), msraaghavan 0.417,
  nicknnamsa Mask R-CNN 0.417 (merging near-duplicates +0.008), HeShen 0.359, Dharun 0.419.
  Learned keep/drop scorers failed publicly (ugurk-0 lost PQ, tushhaaarrrr LB 0.37).
  EdgeAttNet, SABR, ducanhle156 (SAM2 LoRA, per-image PQ 0.33) trained on public MAGFiLO
  splits that overlap our test set - not comparable. The 0.55 block is a hard-coded payload
  (and one team's YOLOv8l weights of unverifiable provenance); the 0.62 entries have no
  write-up. Papers report Dice/IoU/AP on random single-annotator splits, never pooled PQ.
- *Thin/small-object instance methods:* the only strong signal is on the detector:
  RF-DETR vs YOLO11 on RF100-VL (100 small real-world datasets), small-object AP 35.8 vs 7.3
  and YOLO11 flat from N to M (matches our v10). Caveat: detection mAP, not PQ, and an NMS
  setting may depress YOLO there. Mask2Former and SAM fine-tuning are weak on small, thin,
  low-contrast objects; a detect-then-refine cascade won the comparable Kaggle Sartorius
  contest. Copy-paste augmentation: +4.8-6.9 box AP at 10 % of COCO.
- *Multi-annotator learning:* object-level gains under real disagreement are small
  (EARL / Bayesian Detector Combination +0.5-1.8 AP on VinDr-CXR; majority-vote labels LOST
  3.5 AP vs training on all annotations - consistent with our flat label variants).
  Missing-label losses help only under random deletion. Decision theory (Lipton F*/2
  analogue): keep a candidate iff q x u > PQ/2 (q = fraction of records it would match,
  u = its IoU); with pooled PQ over ~8k objects a single global cut on a well-ranked score
  is near-optimal - which is what our conf sweep already does.

**Ranked next steps, by evidence x cost, all aimed at complete misses:** (1) RF-DETR as the
detector, refiner unchanged (1-2 runs); (2) copy-paste of drawn filaments within the disk
(1 run); (3) BDC/EARL-style annotator-agreement loss weighting (1 run, small expected).
Not worth it: Mask2Former, SAM fine-tune, probabilistic segmentation, missing-label losses,
semi-supervised archive pseudo-labels (recall at the cost of precision), learned keep/drop.

## 2026-09-29 — v11 verdict: archive views FAIL (runs/20260929-120323)

1,736 aligned views built in-kernel (2.5/file). **OOF PQ 0.4341 vs v9 0.4449 (-0.011)**:
SQ 0.688 (= v9), RQ 0.631 vs 0.648; TP 4779 vs 4969, FN 3420 vs 3223, FP ~equal. The
views cost recall. Most likely causes, not separable from one run: (a) ~22 % label noise on
the views (fidelity 0.45-0.47 vs 0.59 on originals) teaching the detector to under-fire,
(b) the archive/competition blur gap shifting the detector's input distribution,
(c) 15 epochs over 3.4x data is not equivalent to 40 over the originals. **Verdict: FAIL.
Training-time sequences closed as extra views, as teacher signal (persistence), and as
rescoring feature.** Report material: all three measured, all flat or negative.

## 2026-09-28 — v10 verdict: detector capacity WASH (runs/20260928-182903); capacity lever closed

YOLO11m (20.1M params) for YOLO11s (9.4M), single variable vs v9. 6.5 h on a T4 (YOLO
~44 min/fold). **OOF PQ 0.4406 vs v9 0.4449 (-0.004)**; box P/R at conf 0.3 pooled
P 0.664 / R 0.620 vs v9 0.661 / 0.639. Miss analysis, recall by size for consensus /
contested / single-annotator filaments: v10 0.785 / 0.892 / 0.864, 0.348 / 0.524 / 0.505,
0.380 / 0.645 / 0.686 vs v9 0.758 / 0.908 / 0.908, 0.354 / 0.521 / 0.581, 0.393 / 0.664 /
0.724. The bigger detector trades a little small-consensus recall for losses on large
filaments; contested recall is unchanged. **Verdict: WASH. The "a small model cannot see
the cue" hypothesis is dead; the cue is not in the single image at any capacity we can
run.** Lever ledger for the detector is now complete: labels, negatives, objectness,
resolution, capacity, rescoring, persistence - all flat. Remaining: views (v11), then the
report.

## 2026-09-28 — run 2 design: archive views for the detector (v11), measured before building

**Frames.** For all 707 train files: other GONG sites at the same minute and the same site
at ±1 h / ±3 h (3,237 frames, `fetch_sequences.py --long`; Kaggle dataset
`ishansharma13/gong-halpha-train-views-long`, private). Sites S and Z are not in the
competition and sit at 4-33 deg of roll: excluded.

**What the competition did to the archive frames** (24 same-frame pairs): no shift, no
rotation, no scale (my earlier "0.5 % rescale" was disk-fit noise); highlights clipped by a
monotone LUT (identity to ~150, flat at ~152); and their JPEGs carry 1.5-2x more
high-frequency content that the archive JPEGs lack - no unsharp setting recovers it (it is
a less-compressed source). Views = LUT(archive), a mild residual blur gap.

**Label fidelity on views** (union GT boxes vs the v9 fold detector's boxes on the view, 10
files; the ORIGINAL frame scores recall 0.59 / mean IoU 0.54): other-site frames raw 0.31
(centring differs by 2-15 px between telescopes) -> **0.45 after a per-frame ECC rigid
alignment**; same-site 1 h with rotation-projected labels **0.47** (median offset < 3 px:
the projection is right; the spread is evolution + detection variance); 3 h 0.39; 6 h 0.31.
Decision: use other-site (aligned) and same-site <= 1.5 h views only; drop 3 h / 6 h.
Roughly 3-4 views per file, ~2,500 extra training images of genuinely different seeing,
optics and exposure, labels ~78 % as consistent as the originals. Expected value stated
before the run: +0.005-0.01 OOF; a coin-flip on being measurable at all.

## 2026-09-28 (final) — persistence under the metric-true label: adds nothing. Sequence idea closed for good.

`scripts/persistence_pq_test.py`: the 158 long-baseline files; 1,783 of their 1,801 v9 OOF
candidates joined (box IoU >= 0.85) to the probe's candidates, so each carries the
metric-true label (`tp_frac`), the 17 single-image/refiner/density features and the
per-baseline persistence (cross-site 0 h, same-site 1 / 3 / 6 h, means). Fold-grouped GBMs:

| ranking score | AUC (metric TP >= half) | best pooled OOF PQ on the 158 files |
|---|---|---|
| detector conf (baseline) | 0.837 | **0.4412** (cut 0.25) |
| A: conf + image/refiner features | 0.844 | 0.4326 |
| B: A + persistence | 0.841 | 0.4316 |
| P: conf + persistence only | 0.808 | 0.4300 |

Persistence lowers the AUC when added to conf alone (0.808 < 0.837) and is flat on top of
the image features (0.841 vs 0.844); every rescored ranking loses 0.01 PQ to the plain
detector confidence, at every cut. The small persistence gap seen under the coverage label
(3 h: 0.67 vs 0.56) does not exist for the label the metric actually pays.

**Verdicts, final and label-clean.** (1) Temporal persistence of a candidate carries no
information about whether its refined mask will match an annotator's mask beyond what the
detector's confidence already says. Sequence idea closed at 10 min, 1/3/6 h and cross-site,
with alignment verified and the metric's own label. (2) Single-image rescoring on top of
the detector, with hand features, refiner confidence, density, or persistence, does not
raise PQ. (3) What the box-IoU labelling artefact changed: the guidelines' size rule IS
visible in consensus data and the detector has learned it; it did not change any
verdict about score. Method note for the report: candidate-level labels for a mask metric
must be built from the refined masks, not from boxes; box IoU mislabelled 39 %.

## 2026-09-28 (later) — the "undrawn" label was wrong for 4 in 10; rescoring re-done with the metric's own label

**Artefact found (Ishan's challenge: "is it the way the data was prepared?").** Every
candidate-level diagnostic since v5 (miss analysis, persistence probes, the skipped-
candidate audit) labelled a candidate "drawn" by BOX IoU > 0.5 to a GT box. For thin,
tilted filaments that test fails often: re-labelling by mask coverage (>= 50 % of a GT
filament's pixels inside the box) shows **1,532 of the 3,918 "undrawn" v9 candidates were
drawn** (39 %). With that noise removed, on three-annotator files, "drawn by all three"
vs "drawn by none" separate cleanly: length 161 vs 71 px (AUC 0.83), image features alone
AUC 0.87, detector conf 0.87, conf + features **0.93**. So rule 1 of the guidelines
("too small / faint") IS visible in consensus data, and the earlier "no rule exists"
paragraph above is withdrawn as stated.

**But the metric does not pay for that label.** Coverage rewards big boxes that swallow a
small filament; the rescorer trained on it (AUC 0.82 vs conf 0.76) LOWERED pooled OOF PQ,
0.4456 -> 0.4256. The only honest label is the metric's own: for each v9 OOF candidate,
its refined mask built exactly as in the pipeline, matched at IoU > 0.5 per record ->
`tp_frac` (3,839 of 8,821 are a TP in >= half their records; coverage and metric labels
disagree on 26 % of candidates). Retrained on `tp_frac >= 0.5` with conf + 12 image + 3
refiner + 2 density features, fold-grouped (`scripts/rescore_learned.py --target tp`):
AUC 0.864 vs conf 0.846 (+0.018); **pooled OOF PQ best 0.4423 vs baseline 0.4456 (-0.003)**
across cuts 0.25-0.55. The features carry a little information the detector lacks, and
none of it survives the instance metric (ordering, overlap resolution, per-record pooling).
**Learned rescoring on single-image features: closed, this time with the right label.**

**Persistence, re-labelled by mask coverage** (`persistence_probe_long_mask.csv`): AUC
persistence 0.62-0.64 vs conf 0.73 (all baselines); within conf 0.2-0.3, drawn vs undrawn
persistence at 3 h 0.67 vs 0.56, at 1 h 0.82 vs 0.75, at 6 h 0.46 vs 0.40 - a small gap the
box label had hidden. The probe's P/R rule table is invalid under a coverage label (recall
> 1: several boxes "contain" one filament), so the decisive test is the same as above:
persistence as an extra feature in the metric-true rescorer on the 158 long-baseline files,
judged by pooled PQ. Queued.

## 2026-09-28 — why annotators skip: the rules are known, already learned, and the rest is subjective

**Source.** MAGFiLO paper (Sci. Data 2024): annotators were told to ignore (1) filaments "too
small or too faint to exhibit any feature", (2) "small, close-by filaments which form
groups", (3) filaments "roughly beyond ±70° from the central meridian"; disagreements were
deliberately not relabelled ("the labeling task is, to a degree, subjective"); kappa 0.66.

**Rule 3 (limb) - already learned.** 0.0 % of the 8,199 GT filaments lie beyond |lon| 70°
(1.6 % beyond 60°), and the v9 OOF detector proposes 0.1 % of its conf >= 0.1 candidates
there, none above conf 0.3. A geometric cut removes nothing.

**Rules 1-2 (size / faintness / grouping) - not separable by anything measurable.**
`scripts/skipped_candidates.py` on all 8,821 v9 OOF candidates (4,903 drawn by >= 1
annotator, 3,918 by none): per candidate, box length / width / aspect / area, limb distance,
contrast vs surrounding ring, darkest-core depth, plage fraction around it, dark-pixel
fraction, ring texture, plus candidate density within 150 / 300 px and nearest-neighbour
distance. Single-feature AUCs for "drawn" among conf 0.1-0.5 candidates: every one in
0.47-0.53 (chance); means drawn vs undrawn: length 107 vs 111 px, aspect 2.01 vs 2.01,
rho 0.57 vs 0.58, contrast 0.017 vs 0.012, plage 0.058 vs 0.061, density 1.18 vs 1.28.
A gradient-boosted classifier on all 16 features, grouped 5-fold CV by file: **AUC 0.775
vs detector conf alone 0.778** (mid-conf: 0.666 vs 0.674). Adds nothing; at any fixed
candidate count its precision equals the detector's.

**Reading.** Among plausible candidates the annotators drew and did not draw look the same
in size, darkness, contrast, position and crowding. The detector's confidence already
encodes whatever is visible (it is the best single predictor and the crop-level objectness
head, v8, was worse). What remains is the "to a degree, subjective" part the paper names:
which borderline filaments a given annotator marks. That is annotator randomness, not a
rule, and no single-image feature - hand-made here, learned by the refiner in v8, or
temporal in the persistence probes - recovers it. Combined with the oracle ceiling, the
model-side headroom that is actually reachable is the capacity question (v10) and small
augmentation gains (views); the rest of the 0.1 is annotator dice.

## 2026-09-27 (evening) — long-baseline retest: NO-GO stands; sequence idea closed

The morning probe was challenged (rightly): a ±10 min window cannot separate day-lived
filaments from minute-lived fibrils, the sample was 2011-only, and persistence was
measured by our own detector on near-duplicate frames. Retest: **160 random train files
across 2011-2022** (`fetch_sequences.py --long --sample 160 --seed 2026`, 838 frames),
same site at ±1 h / ±3 h / ±6 h plus every other site at the same minute; candidate boxes
projected into each neighbour with a per-frame disk fit and Snodgrass differential
rotation (`persistence_probe.py`). Alignment validated: confident drawn candidates keep
mean best IoU 0.72 / 0.62 / 0.52 at 1 / 3 / 6 h (0.55 cross-site), i.e. the projection
works and the residual decay is real evolution + foreshortening.

| baseline | n cand. | AUC conf | AUC persistence | drawn vs undrawn persistence, conf 0.2-0.3 | conf 0.3-0.5 | conf >= 0.5 |
|---|---|---|---|---|---|---|
| cross-site 0 h | 2173 | 0.801 | 0.606 | 0.58 / 0.52 | 0.64 / 0.62 | 0.73 / 0.77 |
| same-site 1 h | 2325 | 0.800 | 0.610 | 0.79 / 0.80 | 0.89 / 0.88 | 0.94 / 0.93 |
| same-site 3 h | 2265 | 0.799 | 0.609 | 0.67 / 0.59 | 0.73 / 0.73 | 0.86 / 0.85 |
| same-site 6 h | 1905 | 0.797 | 0.593 | 0.45 / 0.44 | 0.55 / 0.61 | 0.78 / 0.79 |

At every baseline, and at every confidence above 0.2, the candidates nobody annotated
persist exactly as long as the annotated ones - through six hours of solar rotation and
across independent telescopes. They are not fibrils, seeing or noise; they are stable
solar features (faint filaments, dark lanes, plage boundaries) the annotators did not
draw. The only band with a persistence gap is conf 0.05-0.10 (0.53 vs 0.33 at 6 h), where
84 % of candidates are undrawn, so admitting the persistent ones still adds ~2 FP per TP.
Every gated rule loses precision at least as fast as it gains recall (e.g. 6 h, conf >=
0.15 & persistence >= 0.8: dP -0.064, dR +0.083; PQ charges 0.5 for each); the AND-prune
of unstable mid-confidence boxes loses recall 2-4x faster than it gains precision.

**Verdict: NO-GO, final.** Temporal persistence measures what is on the Sun; the metric
pays for what MAGFiLO annotators drew; on this label set those diverge for roughly half of
all plausible candidates, and the detector's own confidence already tracks annotator
behaviour (AUC 0.80) far better than physics does (0.61). Closed at both short and long
baselines, with alignment verified and a year-balanced sample. Frames kept under
data/sequences_long/ (gitignored) for the report. Remaining model-side lever: detector
capacity (one run).

## 2026-09-27 — sequence idea: NO-GO (temporal persistence does not separate what annotators draw)

**Host ruling (forum, twice):** sequences of neighbouring GONG frames may be used in
training only; inference is single-image. So the only admissible form was a teacher:
measure persistence on training images, distil it into the single-frame model.

**Go/no-go measurement (`scripts/persistence_probe.py`).** 160 train files, 12 same-site
frames within ±10 min each (`scripts/fetch_sequences.py`). The OOF v9 fold detector at
conf >= 0.05 on the labelled frame and its neighbours; per candidate, persistence =
fraction of neighbour frames with a box at IoU >= 0.3; support = annotator records with a
GT box at IoU > 0.5. 2,903 candidates: 1,344 drawn by >= 1 annotator, 1,559 by none.

| separating "drawn by someone" from "drawn by no one" | AUC |
|---|---|
| detector conf | **0.807** |
| persistence | 0.665 |
| mean best IoU to neighbours | 0.709 |
| faint candidates only (conf < 0.3, n=1,742): conf / persistence | 0.641 / 0.610 |

Persistence by conf band, real vs none: [0.2,0.3) 0.94 vs 0.94; [0.3,0.5) 0.97 vs 0.95;
[0.5,1) 0.99 vs 1.00. **The candidates nobody annotated persist across frames exactly as
much as the annotated ones.** They are not noise or seeing artefacts; they are physically
real dark features (faint filaments, fibril lanes, plage boundaries) that the annotators
chose not to draw. Persistence measures what is on the Sun; the metric pays for what the
annotators drew; those are different questions, and the detector's own confidence already
tracks the second one better than physics does.

Gated rule "conf >= 0.3 OR (conf >= 0.15 AND persistence >= 0.9)": P 0.772 -> 0.658,
R 0.633 -> 0.754 - precision falls faster than recall rises at every setting, and under
PQ an FP costs the same 0.5 as an FN. Net negative everywhere.

**Verdict: NO-GO.** Download had completed: 707 files, 6847 frames, 4.1 GB (kept under data/sequences/, gitignored).
Extra training views (label propagation) remain possible as augmentation, expected
+0.005-0.01 at most, parked in FUTURE. The finding is worth a paragraph in the final
report the host asked for: "unannotated" is not "absent" in MAGFiLO, and any objective
physical signal will over-predict against this label set.

## 2026-09-25 — oracle ceiling: the "label-noise ceiling" was wrong; headroom is ~0.1 of RQ

`scripts/oracle_ceiling.py`: filaments clustered across a file's records at mask IoU >= 0.5;
an oracle that knows every record predicts one mask per cluster (the medoid record's copy)
above a support floor, scored pooled against all records like the evaluator.

| support floor | all 707 | multi-annotator 296 | TP / FP / FN (multi) | SQ | RQ |
|---|---|---|---|---|---|
| >= 1 record (union) | 0.767 | 0.656 | 5111 / 3286 / 10 | 0.868 | 0.756 |
| >= 1/2 | 0.787 | **0.653** | 3974 / 1242 / 1019 | 0.838 | 0.779 |
| >= 2/3, all | 0.729 | 0.489 | 2227 / 10 / 2766 | 0.794 | 0.616 |

Single-annotator files score 1.0 for any oracle (the record is its own consensus), so the
multi-annotator column is the informative one. **A perfect consensus predictor reaches
0.65 on multi-annotated files; our cascade sits at ~0.44 there.** Two corrections to what
this log has been saying since v2:
1. The human-vs-human 0.34 is the ceiling for a model that behaves like ONE annotator.
   A model that predicts the consensus is bounded far higher. Our 0.449 beating 0.34 was
   never evidence of being at a ceiling.
2. The oracle's SQ 0.84 includes exact self-matches; a model cannot copy an annotator, so
   its realistic SQ is the medoid mask against the OTHER records, ~0.70 (we are at 0.687:
   SQ really is near its ceiling). RQ is not: oracle 0.78 vs ours 0.65. Realistic model
   ceiling on multi files ~0.70 x 0.78 = **~0.55**, i.e. ~0.1 PQ of headroom, all of it in
   existence decisions.

**Where the RQ gap is, precisely.** The >= 1/2 oracle predicts every 1/2- and 2/3-support
filament and eats their FPs in the dissenting record (1242 FP) because the metric pays for
it (k/n > 0.32 rule, v5 miss analysis). Our detector finds those at 0.39 / 0.66 recall and,
when its cut is lowered to catch them, also admits 0-support blobs nobody drew - which the
oracle never emits. The missing signal is "faint but some annotator would draw it" vs
"faint and nobody would". Three attempts to learn it were null (v6 union labels, v7 hard
negatives, v8 objectness head), so it is not a labelling fix; it is a model-capacity /
representation question, which is the one lever (detector capacity) never tested. The
honest LB 0.56-0.62 entries sit at OOF ~0.61-0.67 by our offset, i.e. at or above this
realistic ceiling: either they resolve contested existence far better than us, or the
test set's annotator mix is easier than train's, or they are not honest. Unknown.

## 2026-09-25 — pretrained-model and unlabelled-data audit (no run)

**Question.** Before pretraining anything on the open GONG archive (rules §6 allow it; the
archive is ~3,800 frames/day, 196 months, same JPEGs as ours), is there a pretrained solar
H-alpha model with clean provenance, and is there evidence that unlabelled pretraining pays?

**Candidates found.**
- `adiercke/DeepFilamentSegmentation` (A&A 2024, GPL-3): YOLOv5 on ChroTel labels, U-Net
  semi-supervised on GONG pseudo-boxes. **Code only; the single release has no assets, no
  weights.** Dead end without retraining their pipeline.
- `ducanhle156/solar-filament-segmentation` (Anh Le, NDSU, 2026 benchmark): six
  architectures on **958 GONG disks with MAGFiLO annotations** = 887 filament disks + 71
  empty = exactly the competition's 707 train + 180 test. Their released checkpoint
  (`sam2_lora_fpn.pt`, SAM 2 Hiera-B+ LoRA + FPN) is therefore **trained on the test
  labels — unusable for us**, whatever its quality. Their numbers, on their own split and
  a per-image-mean PQ: U-Net 0.32, SAM 2 LoRA 0.33, best arms within 0.014 Dice of each
  other; "the hard cases are faint, short filaments near the limb, and every model misses
  them" (our miss analysis, independently).
- **Their stage 2 is our lever 1, already run:** SimMIM-style masked pretraining of the
  SAM 2 encoder on 106,871 unlabelled GONG frames (dates overlapping labelled disks
  excluded), then the identical LoRA fine-tune on the same folds. Project page, 2026-09:
  "pretraining bought efficiency, not accuracy ... Pretraining on the unlabelled archive
  has not yet carried through to the fine-tuned model." A clean, single-variable negative
  on 100k images of our exact instrument.
- `nasa-ibm-ai4science/Surya-1.0` (Apache-2): 366M-param transformer on SDO AIA/HMI at
  4096 px. Not H-alpha, not ground-based, no single-channel transfer story. Not pursued.

**Verdict.** No usable pretrained H-alpha model exists off the shelf; the one clean
experiment on unlabelled GONG pretraining is null. **Lever 1 (unlabelled-data pretraining)
is demoted from first to last**: the prior is now negative and the cost (15-30 GPU-h) is
the highest of the open levers. Caveat kept: Le pretrained a ViT for a semantic head; a
CNN detector backbone might respond differently. If it is ever tried, it is one run, not a
programme. Open levers, re-ranked: detector capacity (YOLO11m / RF-DETR, one run), soft
consensus refiner targets + encoder diversity (one run), oracle-ceiling measurement
(local, decides whether any of it matters).

## 2026-09-20 — public leaderboard audit: the 0.55 wall is a copied payload

Rank after the fused entry: **56 / 654** (0.40; 9 teams tied at 0.40, ranks 50-58; median
0.33). Score distribution: 1 team at 0.62, 1 at 0.58, 3 at 0.56, **41 at exactly 0.55**,
then nothing until 0.42. Pulled the public kernels behind it:

- `lamhuy8904/solar-filament-unet-segmentation-0-55` ("0.55+", most-copied): trains a
  3-level toy U-Net on 60 images for 5 epochs, then never uses it — the submission is
  decoded from a hard-coded lzma+base64 blob (`CHAMPION_PAYLOAD`, "pre-compiled inference
  database, verified 0.55 on Leaderboard") holding RLEs for all 180 test images. A
  pre-baked test submission, i.e. the leaked MAGFiLO test annotations in another wrapper.
  Not decoded, not used. This is the 41-team block; it is not a recipe and it will not
  survive an honest private split any better than it deserves.
- `vmerckle/simple-dark-filament-segmentation-0-46`: genuinely no training — disk mask,
  Gaussian background (sigma 40), dark residual above the 90th percentile, 3x3 dilate,
  connected components (80 <= area <= 150k), **up to 120 components per image, largest
  first**, LB 0.46. That is above our trained 0.40 while emitting ~10x our instances per
  image; under `src/metric.py` (0.5 per FP) that cannot score 0.46 unless the components
  are mostly right. Scored on the 707 train images with our metric
  (`scripts/classical_probe.py`) to settle whether the LB evaluator penalises FPs the way
  the organiser's notebook does — moot: the kernel's last run is **2026-07-09, a month
  before the 2026-08-07 switch from the Dice-style evaluator to PQ**. Its 0.46 is a score on
  the old, recall-rewarding evaluator (the one that gave empty masks 0.88); the probe was
  stopped. Public scores dated before 2026-08-07 are not comparable to ours.
- `nishantkharga/...-honest-sota-v9`: UNet++/EfficientNet-B4 @1024, consensus masks, TTA,
  disk masking, components — the same family as our v2 (0.41 OOF / 0.36 LB).

**Evaluator probe (2026-09-20, submission 56400454).** The fused entry rebuilt at cut 0.10
(1665 rows, 9.25/image vs 6.9 at cut 0.25): `src/metric.py` on OOF predicts 0.4451 - 0.4485 =
**-0.03; LB 0.40 -> 0.37, exactly -0.03.** The leaderboard charges false positives as the
organiser's PQ does; the OOF->LB offset stays ~0.05. Any "emit more candidates" strategy is
dead on arrival, and the classical 0.46 cannot be an evaluator artefact of FP leniency.

## 2026-09-20 — 5-run detector fusion on OOF: +0.005, the one lever that cleared its bar

`scripts/fuse_runs_oof.py`: for each fold, the v5-v9 fold-k detectors' OOF boxes (all
five trained without fold k; already on disk) fused by weighted boxes fusion, refined by
the v9 fold refiner without TTA, vs the v9 boxes alone under the same refiner:

| | cut 0.2 | cut 0.25 | cut 0.3 | cut 0.35 |
|---|---|---|---|---|
| single (v9) | 0.4394 | 0.4433 | 0.4430 | 0.4378 |
| fused (v5-v9) | 0.4451 | **0.4485** | 0.4471 | 0.4423 |

Box level at conf 0.3: P 0.681 R 0.628 fused vs 0.661 / 0.639 single — fusion buys
precision (fold agreement demotes singletons), and the optimum moves to cut 0.25 where it
is +0.0052 paired on identical files and refiner. Consistent with the 2-detector probe
(+0.004): diminishing, but real. It costs no GPU: the 25 fold detectors and the 5 v9
refiners are pulled. **Built locally as `runs/fused-v5-v9/submission_fused.csv`
(`scripts/fuse_runs_test.py`: 25 detectors WBF at their own imgsz/source, v9 refiners with
flip TTA, cut 0.25 / thr 0.5 / min_area 100) and submitted as the first non-kernel
submission.** Expected LB: v5's 0.39 + ~0.005, i.e. 0.39 or 0.40 at 2 decimals. **LB: 0.40** (1240 rows, 6.9/image;
v5: 1092 rows at cut 0.3). The OOF->LB offset holds (0.4485 -> 0.40 ~ -0.05); the current best entry.

## 2026-09-20 — v9 verdict: 1280 px native detector WASH (0.4449); resolution lever closed

**Run.** 4.7 h (YOLO 23 min/fold at 1280, refiners 22 min/fold, sweep 32 min). OOF PQ
**0.4449 in-kernel / 0.4456 offline at conf 0.25** vs v6 0.4441 / 0.4452: +0.0008 / +0.0004.
Per fold 0.436 / 0.436 / 0.450 / 0.454 / 0.448. Refiner val Dice 0.811-0.818 (= v6). Not
submitted (v5's 0.39 LB stands; three submissions left today unused).

**Box level (pooled, conf 0.3): P 0.661 R 0.639 vs v6 P 0.660 R 0.623** — recall +0.016 at
the same precision, the first detector change that moved the P/R curve at all. But the
matched instances did not follow: TP 4971 vs 4968, FP 2185 vs 2202. The extra boxes land on
filaments whose refined masks miss IoU 0.5 or that a neighbour already claims.

**Size x agreement (`scripts/miss_analysis.py`), v9 vs v6 box recall at conf 0.3:**
consensus < 64 px **0.758 vs 0.790** (n=372: -0.03, inside the +-0.02 noise of that cell),
consensus 64-150 px 0.908 vs 0.885, > 150 px 0.908 vs 0.885; contested 0.354 / 0.521 /
0.581 vs 0.368 / 0.507 / 0.531; single-annotator 0.393 / 0.664 / 0.724 vs 0.370 / 0.647 /
0.702. 1280 px helps every cell EXCEPT the one it was aimed at. The small consensus
filaments the detector misses are not a resolution problem (they are ~25 px at 1280 and
~20 px at 1024, both well inside YOLO's stride-8 range); they are the faint/short ones on
which the detector's confidence is low for the same reason it is low on contested ones.
**The resolution lever is closed at this model size.**

**Where this leaves the cascade after v5-v9 (five runs, 24 GPU-hours):**

| lever | measured effect on OOF PQ |
|---|---|
| detector labels medoid -> union (v6) | 0 |
| refiner hard negatives (v7) | 0 (loss could not learn them) |
| refiner objectness head (v8) | 0 (AUC 0.658 < detector conf 0.774) |
| detector 1024 -> 1280 native (v9) | 0 (+0.016 box recall, +3 TP) |
| mask-confidence / objectness / semantic rescoring (offline) | 0 |
| semantic instances as extra candidates (offline) | -0.008 |
| detector flip TTA, NMS 0.7 (offline) | same P/R curve |
| 2-detector fusion (offline) | +0.004 |
| post-processing cuts (offline, every run) | plateau conf 0.25-0.30, thr/min_area flat |

Every variant sits at 0.442-0.446 OOF; v5 is the LB entry at 0.39. The failure profile is
unchanged since v5: consensus filaments are found (0.85-0.92), contested ones are found at
the rate annotators agree on them, SQ 0.687 is above the human-vs-human 0.63. The recipe
is at the ceiling that this label noise sets for a detector-plus-refiner; more single-knob
runs of this recipe are not expected to pay. The last free measurement in flight is the
5-run detector fusion on OOF (`scripts/fuse_runs_oof.py`, v5-v9 fold detectors under the
v9 refiners, boxes already on disk): if it clears +0.005 it is a free test-time gain built
locally from the pulled weights, no kernel run needed.

## 2026-09-19 — semantic-fusion probe NULL -> v9 = detector resolution (1280, native frames)

**Probe (`scripts/semantic_fusion_probe.py`, 590 files with v4 semantic OOF maps + v8 crops).**
(a) Ranking: mean semantic prob inside each refined mask — AUC 0.647 alone, 0.770 combined
vs 0.774 for the detector conf; PQ sweep 0.4446 (conf only) vs 0.4445 best combined. Null,
like every other second opinion. (b) Recall: of the 1476 GT filaments with no detector box,
a semantic instance (v4 knobs thr 0.55 / min_area 300 / min_score 0.9) matches 298 (20 %),
but the 766 semantic instances that overlap no detector box match an annotator only 100
times (precision 0.13): adding them = +100 TP, +666 FP -> PQ 0.437 (-0.008). **The
semantic model sees nothing the detector does not; the misses are the contested class.**

**Ledger of measured levers (all on 5-fold OOF, 1154 records):** medoid vs union labels 0,
hard negatives 0, objectness head 0, mask-confidence rescoring 0, semantic rescoring 0,
semantic candidates -0.008, detector TTA / NMS 0.7: same P/R curve, 2-detector fusion
+0.004. Remaining measured gap with an untested cause: **small filaments** — consensus box
recall 0.79 under 64 px vs 0.89 above (v5 miss analysis), 26 % of GT is under 64 px, and
the detector has only ever seen them at 1024 px (a 40 px filament is 20 px wide).

**v9: DET_IMGSZ 1280 with DET_SOURCE "native"** — the detector trains on and predicts from
the 2048 JPEGs (ultralytics resizes to 1280; v5-v8 used the INTER_AREA 1024 cache, so 1280
needs a new source), boxes come out in native coordinates (scale 1). Refiner reverted to
v6 exactly (`REF_NEG_FRAC 0`, `REF_OBJ_HEAD False`; the head is now a config knob and both
checkpoint kinds load), so the comparison is v6 (0.4441 / 0.4452 offline) + one variable.
Re-measure the size x agreement recall table (`scripts/miss_analysis.py`) to see whether
the small-consensus cell moves. Cost: YOLO ~22 min/fold at 1.56x the pixels -> ~4.5 h.
**Gate: OOF PQ > 0.4452 + 0.01, or the < 64 px consensus recall up by >= 0.05 with
precision held (then the next knob is 1536 / YOLO11m).**

## 2026-09-19 — v8 verdict: objectness head trains but adds nothing (WASH, 0.4421)

**Run.** 4.9 h. In-kernel 0.4421 (conf 0.3 / thr 0.6), offline 0.4432 at conf 0.25 vs v7's
0.4461: -0.003, noise (val Dice 0.804-0.813 vs 0.809-0.817, the head costs the mask a
little). Per fold 0.439 / 0.435 / 0.442 / 0.447 / 0.446. Not submitted.

**The head did learn this time**: val negatives rejected (obj < 0.5) 0.19-0.26 per fold
(v7: 0.000), obj mean on val positives 0.74-0.78 vs negatives 0.64-0.67. But as a ranking
signal over all 8835 OOF crops (label = box matches some annotator's filament at IoU > 0.5,
55 % positive): **AUC detector conf 0.774, objectness 0.658, conf x obj 0.773, conf x obj^2
0.759**; corr(conf, obj) 0.44. Rescoring sweep (score = conf x obj^b): b=0 0.4432, b=0.5
0.4427, b=1 0.4419, b=2 0.4358, b=4 0.4143. The refiner's second opinion is a noisier copy
of the detector's first one. **Verdict: the rejection lever is exhausted — three different
second-opinion signals (mask confidence, hard-negative mask, objectness head) all fail to
beat the detector's own confidence, because the "false positives" are mostly filaments
some annotator would have drawn (the v5 miss analysis: 88 % of never-proposed GT and, by
symmetry, most FPs sit in the contested class).**

**Detector inference knobs probed offline on the v8 fold weights (`scripts/detector_probe.py`,
box P/R pooled over folds):** base conf 0.3 P 0.663 R 0.623; flip/scale TTA conf 0.3 P 0.623
R 0.672, conf 0.4 P 0.702 R 0.585 — the same P/R curve shifted along conf (at P = 0.663
TTA gives R ~0.63); NMS 0.7 strictly worse (P 0.633 at R 0.627). Neither moves the curve.
Not worth a run.

**Ledger after v5-v8.** Cascade 0.444-0.446 OOF / 0.39 LB across four variants; every
lever inside the cascade (labels, hard negatives, objectness, TTA, NMS, rescoring,
2-detector fusion +0.004) is flat. What has NOT been measured: a second, independent view
of the image — the semantic U-Net (full-disk context at 1024, different errors) — as a
ranking feature or as a candidate source (`scripts/semantic_fusion_probe.py`, running on
the 590 files whose v4 OOF maps are on disk), and detector capacity / resolution for the
small-filament gap (consensus recall 0.79 under 64 px vs 0.89 above).

## 2026-09-19 — v7 verdict: hard negatives WASH (0.4450), because the loss could not learn them

**Run.** 5.0 h (detectors 77 min, refiners 5 x 31 min with the negatives + diagnostic, sweep
55 min). OOF PQ **0.4450 vs v6 0.4441** (+0.001, noise); offline conf 0.25: 0.4461 vs
0.4452 — the same +0.001. Per fold 0.442 / 0.437 / 0.447 / 0.451 / 0.448. Not submitted.
1437 hard negatives over 523 files (conf >= 0.2, IoU < 0.3 to every annotator).

**Mechanism: neg-reject 0.000 at every epoch of every fold.** The refiner emitted >= 100 px on
every one of the 400 val negatives, in every fold, at every epoch, and training loss sat at
0.45 vs 0.24 in v6 — exactly the negatives' Dice term (~1 each, 25 % of samples) never
coming down. The soft Dice on an all-zero target is 1 - 1/(area+1): its gradient is
1/(area+1)^2, i.e. ~1e-6 for a 1000 px mask, and the pixel-mean BCE spreads the remaining
signal over 65k pixels. The negatives carried no usable gradient; the v7 refiner is the v6
refiner plus noise. Rescoring probe on v7 crops: null again (0.4461 -> 0.4451 / 0.4453).
**Not a data verdict: the experiment was not run.**

## 2026-09-19 — v8 push: objectness head, the rejection channel v7 lacked

Same negatives, same detector, same sweep. Two changes that are one mechanism:
- `build_refiner` adds smp's auxiliary classification head (global-average-pooled encoder
  features -> 1 logit): "is there a filament under this seed?", trained with BCE on the
  positive / negative label — a per-sample gradient that cannot be starved.
- `refiner_loss`: positives keep BCE + Dice; negatives get pixel BCE only (Dice on an empty
  target is undefined).
The head's sigmoid is stored per crop (`obj`, 9th meta column) for OOF and test, so the
instance score conf x obj^b can be swept offline (`scripts/rescore_cascade.py --feature obj`)
and applied to the test crops without a kernel run (`scripts/build_submission.py`). The
in-kernel sweep stays on detector conf alone, so the kernel's own number is comparable to
v7. Per-epoch diagnostic: mean obj on val positives / negatives, negatives rejected
(obj < 0.5), negatives with a < 100 px mask. Debug smoke test: obj pos 0.885 / neg 0.009
after one epoch on 16 files — the head trains where the mask head could not.
**Gate: offline conf x obj^b sweep > 0.4461 + 0.01; kernel-only number is expected ~flat.**
Cost ~5 h.

## 2026-09-19 — offline probes on v6 crops -> v7 decision: refiner hard negatives

Three levers were measured offline before spending GPU hours:

1. **Refiner-confidence rescoring (`scripts/rescore_cascade.py`) — NULL.** Score =
   conf x meanprob^b, b in {0, 1, 2, 4}, cut swept 0.15-0.40: best 0.4452 (b=0, cut 0.25),
   0.4447 (b=1), 0.4442 (b=2), 0.4449 (b=4). The refiner's mean prob inside its mask is
   0.918 / 0.950 / 0.964 (10/50/90 %) with corr 0.13 to the detector conf: it is equally
   sure on true and false boxes. Trained on positive seeds only, it never learned to reject.
2. **Two-detector fusion (`scripts/fuse_oof.py`, v5 + v6 fold detectors under the v5 fold
   refiners, no TTA) — +0.004.** single 0.4417 (cut 0.3) vs fused 0.4460 (cut 0.25). Real
   (paired, same refiner) but small; a second detector per fold costs +14 min/fold. Parked.
3. **Test-time calibration — not a lever** (see v6 verdict).

**Where the loss is.** At the v6 optimum: TP 4968, FP 2202, FN 3231, sum IoU 3413. FP is
31 % of predictions; removing a third of them at constant TP is +0.02, and it would also let
the cut move down (recall) with the refiner absorbing the extra FPs. Nothing in the pipeline
looks at a box twice today.

**v7 = refiner hard negatives** (`REF_NEG_FRAC 0.25`, `REF_NEG_CONF 0.2`, `REF_NEG_IOU 0.3`).
The negatives are the detectors' own OOF false positives — the exact distribution the
refiner meets at test time — with an all-zero target; the crop pipeline is otherwise
unchanged (same jitter, same seed channel). To have OOF FPs for every training file, the
kernel now trains all five detectors first (phase 1, ~70 min), then the five refiners
(phase 2). Per-epoch diagnostic: fraction of 400 val negatives the refiner rejects
(< 100 px at thr 0.5), logged next to val Dice (positives only, so comparable to v5/v6).
Expected cost ~4.7 h. Single variable vs v6. **Gate: OOF PQ > 0.4452 + 0.01 at the
in-kernel sweep, and the offline rescoring must show the refiner's mask area / mean prob
now separating TP from FP (a nonzero neg-reject rate).** If it passes, the next knob is the
conf cut (lower) with area/mean-prob floors, swept offline on the saved crops.

## 2026-09-19 — v6 verdict: union labels WASH, exactly 0.4441 (runs/20260919-093949)

**Run.** 3.9 h. OOF PQ **0.4441 vs v5 0.4441** (SQ 0.687 / RQ 0.646, TP 4968 / FP 2202 / FN
3231 vs 4921 / 2076 / 3278): +47 TP bought with +126 FP. Per fold 0.441 / 0.435 / 0.446 /
0.453 / 0.445 vs 0.435 / 0.440 / 0.444 / 0.452 / 0.445 — three up, one down, all within
noise. Sweep shape identical (conf 0.3 optimum, thr and min_area flat). Not submitted.

**Mechanism (scripts/miss_analysis.py, v6 vs v5 box recall at conf 0.3).** support 1/2:
0.399 vs 0.389; 2/3: 0.679 vs 0.661; 2/2: 0.832 vs 0.852; 3/3: 0.905 vs 0.904; 1/3 (removed
from the labels!): 0.355 vs 0.337; single-annotator: 0.577 vs 0.575. Adding ~470 contested
filaments to the labels moved their recall by +0.01-0.02, and removing 250 others moved
theirs up. **Label presence is not what limits recall on contested filaments.** They are
proposed (recall 0.65-0.87 at conf 0.1) but at low confidence, because features that look
like them are unlabelled background in most other files: the detector's confidence on them
is the annotators' disagreement rate, learned faithfully. The v5 miss analysis identified
where the misses are; v6 shows they are not recoverable by relabelling. The metric's k/n
arithmetic still holds, but the detector already ranks by k/n on its own.

**Consequence for the plan.** Recall on consensus filaments is 0.83-0.91; on contested ones
it is capped by label noise; the FP/FN cut is on its plateau. The remaining levers are
(a) the ranking itself — fewer FP at equal recall — and (b) SQ. Both are testable offline
before any GPU run: refiner-confidence rescoring (`scripts/rescore_cascade.py`) and
two-detector fusion on OOF (`scripts/fuse_oof.py`, v5 + v6 detectors under the v5
refiners). Test-time calibration was checked and is not a lever: fused test confidences
match the OOF distribution (6.43 vs 6.46 boxes/image at conf 0.3, quantiles within 0.02).

## 2026-09-19 — v6 push: union-of-records detector labels (single variable)

`DET_LABEL_RECORD = "union"`, `DET_LABEL_MIN_SUPPORT = 0.5` (`detector.union_boxes`). Every
record's filaments are clustered across annotators at mask IoU > 0.5 with at most one
filament per record per cluster (an annotator who split a filament keeps two boxes; chain
merges cannot swallow them). Clusters drawn by >= half of the file's annotators are kept,
represented by the box of the member with the highest total IoU to the others.

Label count: 5421 boxes (7.67/file) vs 5222 medoid; 3206 from single-record files
(unchanged), 1428 from 2-annotator files, 787 from 3-annotator files. The full union at
support >= 1/3 would be 6448 — the extra ~1000 sit at the metric's break-even (k/n = 0.33
vs 0.32), so they are left out; if v6 raises recall on 1/2 and 2/3 filaments as intended,
"support >= 1/3" is a one-knob follow-up. Note the change is not purely additive: 88 files
lose a net 202 boxes that only the medoid annotator drew (1/3 support); the metric says
those are worth ~0.

Everything else is v5 byte-for-byte (YOLO11s @1024, 40 epochs, refiner, sweep grid, WBF).
Smoke test: debug notebook end-to-end, exit 0. **Gate: OOF PQ > 0.4441 + 0.01; the
per-support recall table of the miss analysis is re-measured to see the mechanism.**
Expected cost ~3.8 h.

## 2026-09-19 — v5 miss analysis: the detector misses what its labels never showed it

Offline, from `fold*_oof_boxes.json` (every GT filament of every record vs the fold
detector's boxes, hit = a box with IoU > 0.5; "support" k/n = how many of the file's n
annotators drew a mask with IoU > 0.5 to it):

| GT filaments (multi-annotator files) | n | box recall @conf 0.3 | @0.1 | present in medoid labels |
|---|---|---|---|---|
| support 2/2 | 1118 | 0.852 | 0.943 | 100 % |
| support 3/3 | 1152 | 0.904 | 0.957 | 100 % |
| support 2/3 | 868 | 0.661 | 0.866 | 82 % |
| support 1/2 | 869 | 0.389 | 0.625 | 55 % |
| support 1/3 | 986 | 0.337 | 0.625 | 25 % |

Single-annotator files (3206 filaments, no support info): 0.575. Size is secondary: within
consensus filaments recall is 0.78 (< 64 px) / 0.90 (64-150) / 0.90 (> 150); within contested
ones 0.35 / 0.50 / 0.51. Agreement explains a 0.4-0.5 gap, size ~0.1. Of the 1661 filaments
never proposed at any conf, 88 % are contested.

**Reading.** The detector is saturated on filaments every annotator agrees on (0.85-0.90 at
the selected conf, 0.95 at conf 0.1). Its recall on the rest tracks how often the medoid
labels contained them — it learned exactly what it was shown. The medoid record was the
right call for a semantic model (v3: wash) but for the detector it deletes 45 % of the
support-1/2 and 18 % of the support-2/3 filaments from the labels.

**What the pooled metric wants.** A filament drawn by k of n annotators, predicted with
IoU s, adds k*s to the numerator and k TP + 0.5(n-k) FP to the denominator; leaving it out
adds 0.5k FN. At the current operating point (PQ 0.44, SQ 0.68) predicting it wins iff
k/n > PQ/(2 SQ) = 0.32. So support 1/2 (869) and 2/3 (868) filaments are worth predicting
and the labels should contain them; 1/3 is marginal (0.33). Union-of-records labels
deduplicated at mask IoU > 0.5 are close to metric-optimal for the detector.

**Decision: v6 = detector labels from the union of records (support >= 1/2 of the file's
annotators, dedup at IoU 0.5), everything else frozen** (YOLO11s @1024, 40 epochs, same
refiner, same sweep). Same cost as v5 (~3.8 h), single variable. The bigger-model /
imgsz-1280 experiment is demoted to after this: it addresses the ~0.1 size effect, not the
0.4 agreement effect. Gate: OOF PQ > 0.4441 + 0.01.

## 2026-09-19 — v5 verdict: cascade PASS, +0.034 OOF, +0.03 LB (runs/20260919-014124)

**Run.** 3.8 h on a T4: cache 3.5 min; per fold YOLO 14 min (40 epochs), refiner 20.5 min
(12 x 102 s), OOF+test inference 1.5 min -> 36 min/fold; in-kernel sweep 30 combos 41 min
(60-110 s each, slower at low conf). Budget headroom: ~4 h.

**OOF PQ 0.4441 (SQ 0.686, RQ 0.648, TP 4921 / FP 2076 / FN 3278) vs v2 0.4100 -> +0.034.
LB 0.39 vs 0.36 -> +0.03. PASS.** The OOF->LB offset stays ~0.05, so OOF keeps selecting.
Per-fold at the default knobs (conf 0.3 / thr 0.5 / min_area 100): 0.435 / 0.440 / 0.444 /
0.452 / 0.445 — every fold above every v2 fold (v2 best fold 0.392 raw, 0.41 post-processed).

**Where the gain came from.** FP fell from 3561 to 2076 (-42 %) at nearly equal TP
(5135 -> 4921) and FN (3064 -> 3278): the detector's confidence is a far better existence
signal than the semantic model's mean-prob floor, and one-mask-per-box removed the merges.
SQ 0.686 vs 0.674: masks a little tighter, still at the annotator ceiling.

**Sweep shape.** conf is the only knob that matters: 0.15 -> 0.428, 0.2 -> 0.438, 0.3 ->
0.444, 0.4 -> 0.425, 0.5 -> 0.383 (FN explodes: box recall at conf 0.5 is only 0.41).
thr 0.6 beats 0.5 by 0.001, min_area 100 vs 200 is identical (the refiner never emits
tiny blobs). Offline finer conf sweep (`scripts/sweep_cascade.py`, thr 0.6): 0.25 -> 0.4441
(TP 5183 / FP 2598 / FN 3016), 0.30 -> 0.4441, 0.35 -> 0.4384: a flat plateau 0.25-0.30,
the kernel's pick stands. Nothing more to squeeze from post-processing on this run.

**Diagnostic: the detector is the bottleneck now.** Box-level P/R against all records,
mean over folds: conf 0.2 -> P 0.59 R 0.71; conf 0.3 -> P 0.68 R 0.62; conf 0.5 -> P 0.81
R 0.42. Final RQ 0.648 with box recall 0.62 at the selected conf means the refiner converts
almost every correct box into an IoU > 0.5 match; what is missing is boxes. Refiner val Dice
0.81 on clean GT boxes, still rising by ~0.001/epoch at epoch 11 (not converged; cheap to
extend, but SQ says the payoff is small). Every refiner epoch cost 102 s of which most is
JPEG decoding of 2048^2 frames (6.5k reads/epoch) — a native-crop cache would halve it.

**Next rung candidates, in order of expected value** (detector recall is the lever):
1. Detector capacity/resolution: YOLO11m, imgsz 1280 or native 2048 tiles, more epochs
   (40 epochs at 14 min/fold leaves 4 h of budget).
2. Detector labels: the medoid record has ~7 boxes/file; annotators disagree by a median
   of 2 filaments per image, so the union of records (deduplicated at IoU > 0.5) as the
   label set may lift recall on the consensus-ambiguous filaments at a precision cost that
   the conf sweep can trade back.
3. Test-time: lower conf on test is NOT free (fused conf is already an average over 5
   detectors); TTA for the detector (flips) is cheap and raises recall.

## 2026-09-18 — v5 push: rung 4, detector -> crop-refiner cascade (cloud session)

**Session note.** First hands-free cloud session. The environment's default network policy
blocked `api.kaggle.com` / `www.kaggle.com` (proxy 403 on CONNECT); fixed by Ishan switching
the cloud environment's network access, after which `setup_cloud.sh` ran clean (data, cache,
pull of v4 outputs). `download.pytorch.org` / `huggingface.co` stay blocked, so the local
smoke test builds the refiner without ImageNet weights (`pretrained=not DEBUG`); the kernel
has internet and downloads them.

**Why this rung.** v2's failure profile at the optimum is 40 % unmatched predictions, 37 %
missed GT and SQ 0.67; SQ is at the annotator ceiling (v4 proved resolution does not move
it), so the only headroom is RQ: per-object existence decisions and one mask per object.
A box detector gives exactly that (NMS = instance separation, confidence = FP/FN knob);
a per-box refiner gives one mask per detection that cannot merge with a neighbour.

**Design (src/detector.py, src/refiner.py, src/cascade.py; the semantic path is intact
behind `PIPELINE = "semantic"`).**
- Detector: ultralytics YOLO11s, imgsz 1024 on the cached 1024 PNGs (so no resize at all),
  40 epochs/fold, flips both axes, single class. Labels are static txt files, so one
  annotator record per file: the medoid (consensus) record — a union of records would
  teach the detector to emit duplicates. Predict at conf >= 0.1, NMS IoU 0.5, boxes scaled
  to native px.
- Refiner: smp U-Net/ResNet34, 3 input channels at native resolution: image crop, CLAHE of
  the crop, and a "seed" = the detector box grown 25 % per side, filled. The seed says WHICH
  filament to segment when several share the crop; the target is that one annotation's
  mask. Crop side = max(256, 1.4 x box long side), centred; the median filament (96 px)
  therefore trains and infers at native pixels with no resampling; only crops > 256 px are
  downscaled and their prob map upscaled back before thresholding (masks never resampled).
  Training samples every (record, filament) pair of the train files (~6.5k crops/fold,
  label augmentation across annotators), 12 epochs OneCycle, BCE+Dice; GT boxes are
  jittered (shift 15 %, scale 0.8-1.25, pad 0.5-1.5x) so seeds look like detector boxes.
  Epoch selection: val Dice on clean GT-box crops.
- Instances: confidence order; per crop keep only the components that touch the seed box
  (neighbours in the crop are dropped), drop < min_area; a mask with > 50 % of its pixels
  already taken by a higher-confidence instance is a duplicate and is dropped, otherwise it
  is clipped -> zero overlap by construction.
- OOF: each fold's detector + refiner on its val files; crop records (uint8 probs + bounds
  + conf) are saved to `oof_crops/*.npz` so every knob can be re-swept offline
  (`scripts/sweep_cascade.py`). In-kernel sweep: conf {0.15..0.5} x thr {0.5,0.6,0.7} x
  min_area {100,200}, selection by pooled PQ over all 1154 records, as always.
- Test: boxes of the fold detectors fused by weighted boxes fusion (IoU 0.55, fused conf =
  sum of per-model best conf / n_models -> fold agreement is rewarded), refined by the
  refiner ensemble with 4-flip TTA on the crops, thresholded with the OOF-selected knobs;
  `test_crops/*.npz` saved. After every fold an intermediate submission (default knobs) keeps
  the run scoreable.
- Diagnostics logged per fold: box-level P/R at conf 0.2/0.3/0.5 against all records —
  says whether the detector or the refiner is the bottleneck.

**Caveats stated up front.** (1) OOF confidences come from one detector per image, test
confidences from the 5-fold fusion; the selected conf cut is therefore slightly
conservative on test (fusion pulls singleton boxes down). (2) The refiner is trained on
positive seeds only: an FP box gets a hallucinated mask, so FP control is the detector's
conf cut alone (negative seeds are a FUTURE line). (3) Local budget estimate: ~45-60 min per
fold on a T4 (YOLO ~25 min, refiner ~15-25 min, inference few min) -> 5 folds well inside
8.5 h.

**Verified locally.** `FILAMENT_DEBUG=1 scripts/run_notebook_local.py` on CPU: 2 folds,
1 YOLO epoch on 16 files, 20 refiner steps, sweep, fused test submission, `oof_crops` /
`test_crops` written, exit 0 (2.5 min). Refiner targets lie 100 % inside their seed masks;
instance assembly unit-tested (duplicate dropped, off-seed blob dropped, zero overlap).

**Pushed as v5.** Gate: OOF PQ vs v2's 0.4100 at its selected knobs.

## 2026-09-18 — v4 verdict: native 2048 resolution FAIL (runs/20260918-075110)

7.4 h on a T4 (~80 min/fold: epoch 150-170 s, most of it the 141-image full-frame eval).
Raw per-fold (thr 0.5 / min_area 150, TTA): 0.341 / 0.332 / 0.364 / 0.356 / 0.362 vs v2
0.376 / 0.355 / 0.375 / 0.392 / 0.386 — every fold down 0.01-0.04. Best epochs 19-25 of 27
and the last-5 means sit 0.005-0.01 below best: training got noisier, not longer. OOF sweep
best **0.3910 vs v2 0.4100 (-0.019)**, with the optimum pushed to the low-threshold edge
(thr 0.55) and FP up 20% at equal FN.

**The decisive number: SQ 0.675 vs 0.674.** Matched masks are no tighter at 4x the pixels.
So boundary looseness is not a resolution artefact; it is annotator disagreement on the
outline (the ±1 px probes already showed the metric's boundary sensitivity is ~0.02/px, and
the inter-annotator SQ ceiling is 0.63). Resolution cannot buy SQ here.

Confound, stated: 512 crops at native res see 1/4 of the context of the v2 crops, which
plausibly explains part of the RQ loss (more FPs from local texture that looks filament-like
without the disk-scale context). A fair retest would use 1024 crops at native res at half
the batch, but with SQ flat the upside is bounded to recovering RQ, i.e. back to ~v2, at
2x the cost. **Verdict: FAIL. Rung 3 closed; IMG_SIZE reverts to 1024, EPOCHS to 32.**

Rungs 1, 2, 3, 5 are now closed on this model family; rung 6 (encoder ensembles) is
bounded by the same SQ ceiling. **Next: rung 4, the detector -> crop-refiner cascade.**
Rationale from the numbers: at the v2 optimum 40% of predictions are unmatched and 37% of
GT is missed; a per-object existence decision (detector confidence) and a per-object mask
(refiner) attack RQ directly, which is the only component with headroom.

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
