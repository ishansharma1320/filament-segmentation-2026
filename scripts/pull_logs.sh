#!/usr/bin/env bash
# Retrieve the executed notebook + log + outputs (submission.csv, fold*_best.pt,
# fold*_history.json, oof/*.png, summary.json) from the latest kernel run into runs/<stamp>/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SLUG=$(python3 -c "import json;print(json.load(open('$ROOT/notebooks/kernel-metadata.json'))['id'])")
OUT="$ROOT/runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
# The listing is paginated (default 20 items, max 200): loop over page tokens or the
# submission/summary/log never arrive when oof/ holds hundreds of PNGs.
tok=""
while :; do
  out=$(kaggle kernels output "$SLUG" -p "$OUT" --page-size 200 ${tok:+--page-token "$tok"} 2>&1)
  echo "$out" | grep -v "oof/" | grep -v "Skipping" || true
  tok=$(echo "$out" | grep -o "Next Page Token = .*" | sed 's/Next Page Token = //' || true)
  [ -z "$tok" ] && break
done
echo "Outputs in $OUT"
ls -la "$OUT" | grep -v "^d"; echo "oof maps: $(ls "$OUT/oof" 2>/dev/null | wc -l)"
