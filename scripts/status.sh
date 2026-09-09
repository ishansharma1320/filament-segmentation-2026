#!/usr/bin/env bash
set -euo pipefail
SLUG=$(python3 -c "import json;print(json.load(open('$(dirname "$0")/../notebooks/kernel-metadata.json'))['id'])")
kaggle kernels status "$SLUG"
