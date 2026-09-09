#!/usr/bin/env bash
# Rebuild notebooks/submission.ipynb from src/ and push it to Kaggle as a kernel version.
#
# `kaggle kernels push` exits 0 even when the push is REJECTED (quota exhausted,
# bad metadata, ...) - it prints the error and returns success, so `set -e` never
# fires. The success string is checked explicitly and a rejection is a hard failure.
# Pushing while a version is still running CANCELS that run: check status.sh first.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python3 "$ROOT/scripts/build_notebook.py"
cd "$ROOT/notebooks"
[ -f kernel-metadata.json ] || { echo "kernel-metadata.json missing"; exit 1; }

out=$(kaggle kernels push -p . 2>&1)
echo "$out"
if ! grep -qi "successfully pushed" <<<"$out"; then
  echo "" >&2
  echo "PUSH REJECTED — no version was created. Nothing is running." >&2
  exit 1
fi
echo "Pushed. Poll with scripts/status.sh"
