#!/usr/bin/env bash
# Bootstrap for a cloud / remote Claude Code session (no conda, no GPU needed).
# Prerequisite: a Kaggle credential in the environment, ONE of
#   KAGGLE_ACCESS_TOKEN   (new-style token; written to ~/.kaggle/access_token)
#   KAGGLE_USERNAME + KAGGLE_KEY   (legacy API key pair; the CLI reads them directly)
# Run from repo root:  ./setup_cloud.sh
set -euo pipefail

echo "==> Python deps (pip)"
python3 -m pip install -q -r requirements.txt

echo "==> Kaggle credentials"
mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
if [ -n "${KAGGLE_ACCESS_TOKEN:-}" ]; then
  printf '%s' "$KAGGLE_ACCESS_TOKEN" > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
elif [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; then
  echo "ERROR: set KAGGLE_ACCESS_TOKEN, or KAGGLE_USERNAME and KAGGLE_KEY, as environment secrets."; exit 1
fi
kaggle competitions list -s filament >/dev/null && echo "kaggle CLI OK"

COMP="filament-segmentation-2026"
echo "==> Competition data (~750 MB) -> data/"
if [ ! -d data/MAGFiLO_1.0_Kaggle_2026 ]; then
  mkdir -p data && kaggle competitions download -c "$COMP" -p data/
  ( cd data && unzip -q -o "$COMP.zip" && rm "$COMP.zip" )
fi

echo "==> 1024 px cache + GT RLEs + grouped folds"
PY=python3 python3 scripts/prepare_cache.py

echo "==> Latest kernel outputs (OOF/test maps for local sweeps; skip with NO_PULL=1)"
[ -n "${NO_PULL:-}" ] || scripts/pull_logs.sh || true

echo "Done. Smoke test:  FILAMENT_DEBUG=1 python3 scripts/run_notebook_local.py   (CPU: ~20-30 min)"
echo "Loop: edit src/ -> scripts/push.sh -> scripts/status.sh -> scripts/pull_logs.sh -> scripts/submit.sh"
