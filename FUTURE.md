# FUTURE.md — parking lot

Rules: one line per idea. No elaboration. Nothing here gets built while a rung is open.

- Watershed / distance-transform split of merged neighbouring filaments (PQ loses both when merged)
- Skeleton-based fragment re-joining: connect components whose spines align within a gap of N px
- Medoid annotator record as training target (highest mean IoU to the other records) vs random record
- Soft targets from annotator vote fraction on multi-annotated files
- Refiner negative seeds (random empty boxes, empty target) so FP boxes get an empty mask, not a hallucination
- Refiner 4th channel: the semantic U-Net prob map (fusion of rungs 1 and 4)
- YOLO11m / imgsz 1280 / more epochs once the box P/R diagnostic says the detector is the bottleneck
- Native 2048 inference with 1024 sliding window + overlap blending
- Disk mask prior: zero probabilities outside the solar disk (all GT lies within r <= 868 px)
- Per-site (B/C/L/M/T/U) intensity normalisation or CLAHE input channel
- Spine annotations as an auxiliary centreline head (barb-aware, EdgeAttNet paper arXiv 2509.02964)
- Chirality labels are free supervision for a multi-task head (not scored, may regularise)
- Encoder sweep: efficientnet-b4, convnext_tiny, mit_b2 (SegFormer) via smp
- Lovász-hinge / focal-Tversky loss for the thin-structure imbalance
- Confidence-ranked FP suppression: drop components whose mean prob < p_min (second knob after min_area)
- Test-time fold agreement: keep an instance only if >= 2 folds predict it (LB-0.70 notebook's fusion rule)
- Pseudo-label the 180 test images with the fold ensemble for a second training round (check rules first)
- Unlabelled GONG pretraining for the CNN detector backbone (Le's 100k-disk SimMIM on SAM 2 was null; only worth one run, last)
- Retrain adiercke's ChroTel-label YOLOv5 recipe as a diversity member (provenance clean, no weights shipped)
- Label-propagated training views from same-site ±10 min frames (29 files already fetched; pure augmentation, +0.005-0.01 at best)
- RF-DETR detector in place of YOLO11 (RF100-VL small-object AP 35.8 vs 7.3; refiner unchanged)
- Copy-paste augmentation of drawn filaments within the disk for the detector
- EARL / Bayesian Detector Combination annotator-agreement loss weighting (+0.5-1.8 AP on VinDr-CXR)
