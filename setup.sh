#!/usr/bin/env bash
# One-time environment setup. Run from repo root.
set -euo pipefail

command -v kaggle >/dev/null 2>&1 || pip install kaggle
if [ ! -f ~/.kaggle/kaggle.json ] && [ ! -f ~/.kaggle/access_token ]; then
  echo "ERROR: no Kaggle credentials (~/.kaggle/kaggle.json or ~/.kaggle/access_token)."; exit 1
fi
chmod 600 ~/.kaggle/* 2>/dev/null || true

COMP="filament-segmentation-2026"

echo "==> Local env (conda 'filament': torch/mps, smp, pycocotools, opencv)"
conda env list | grep -q '^filament ' || conda env create -f environment.yml -y

echo "==> Competition data (~750 MB) -> data/"
if [ ! -d data/MAGFiLO_1.0_Kaggle_2026 ]; then
  mkdir -p data && kaggle competitions download -c "$COMP" -p data/
  ( cd data && unzip -q -o "$COMP.zip" && rm "$COMP.zip" )
fi

echo "==> Reference notebooks (read-only) -> reference/"
mkdir -p reference
for k in azimahmadzadeh/self-evaluation-notebook realtalhacelik/solar-filaments-u-net-baseline \
         phuongncn/lb-0-70-topology-safe-solar-filaments-handoff \
         timgluhkikh/best-score-magfilo-yolo11s-detect-crop-u-net ektarr/new-metric-solar-filament; do
  kaggle kernels pull "$k" -p reference/ || true
done

echo "==> 1024 px cache + GT RLEs + grouped folds"
conda run -n filament python scripts/prepare_cache.py

echo "Done. Loop: edit src/ -> scripts/push.sh -> scripts/status.sh -> scripts/pull_logs.sh -> scripts/submit.sh"
