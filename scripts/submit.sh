#!/usr/bin/env bash
# This is a CSV-upload competition (not a code competition): submit a pulled
# submission.csv straight from the CLI, after check_submission.py passes.
#   scripts/submit.sh runs/<stamp>/submission.csv "message"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FILE="${1:?usage: submit.sh <submission.csv> <message>}"
MSG="${2:?usage: submit.sh <submission.csv> <message>}"
PY="${PY:-conda run -n filament python}"   # checker needs pycocotools (conda env 'filament')
$PY "$ROOT/scripts/check_submission.py" "$FILE"
kaggle competitions submit -c filament-segmentation-2026 -f "$FILE" -m "$MSG"
echo "Submitted. Score: kaggle competitions submissions -c filament-segmentation-2026 | head -5"
