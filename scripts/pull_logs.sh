#!/usr/bin/env bash
# Retrieve the executed notebook + log + outputs (submission.csv, fold*_best.pt,
# fold*_history.json, oof/*.png, summary.json) from the latest kernel run into runs/<stamp>/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SLUG=$(python3 -c "import json;print(json.load(open('$ROOT/notebooks/kernel-metadata.json'))['id'])")
OUT="$ROOT/runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
kaggle kernels output "$SLUG" -p "$OUT"
echo "Outputs in $OUT"
ls -la "$OUT"
