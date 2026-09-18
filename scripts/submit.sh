#!/usr/bin/env bash
# This is a CSV-upload competition (not a code competition): submit a pulled
# submission.csv straight from the CLI, after check_submission.py passes.
#   scripts/submit.sh runs/<stamp>/submission.csv "message"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FILE="${1:?usage: submit.sh <submission.csv> <message>}"
MSG="${2:?usage: submit.sh <submission.csv> <message>}"
# checker needs pycocotools (conda env 'filament'); call the env binary directly - `conda run`
# failed transiently once (2026-09-17) and silently blocked a submission
ENVPY="$(conda info --base 2>/dev/null || echo /nonexistent)/envs/filament/bin/python"
PY="${PY:-$([ -x "$ENVPY" ] && echo "$ENVPY" || echo python3)}"   # cloud: plain python3 (pip env)
$PY "$ROOT/scripts/check_submission.py" "$FILE"
kaggle competitions submit -c filament-segmentation-2026 -f "$FILE" -m "$MSG"
echo "Submitted. Score: kaggle competitions submissions -c filament-segmentation-2026 | head -5"
