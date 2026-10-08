#!/usr/bin/env bash
# Build an eval model dir that uses this repo's inference code (eaglevl/utils/locany/*.py) with a checkpoint's
# weights/configs, via symlinks (no copy of the weights). Mirrors the end-of-training copy in
# eaglevl/train/locany_finetune_magi_stream.py, but keeps the checkpoint's JSON configs.
# Usage: bash evaluation/tools/link_model_dir.sh <checkpoint_dir> <out_dir>
set -euo pipefail
CKPT=$(realpath "$1"); OUT=$2
REPO_CODE=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../../eaglevl/utils/locany")
mkdir -p "$OUT"
for f in "$CKPT"/*; do ln -sfn "$f" "$OUT/$(basename "$f")"; done
for f in "$REPO_CODE"/*.py; do
  [[ $(basename "$f") == __init__.py ]] && continue
  ln -sfn "$f" "$OUT/$(basename "$f")"
done
echo "linked $CKPT + $REPO_CODE/*.py -> $OUT"
