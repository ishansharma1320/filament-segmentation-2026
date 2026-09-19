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
   more epochs, (c) detector flip TTA.

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
