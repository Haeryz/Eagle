#!/usr/bin/env bash
# Single-GPU evaluation on the fixed, seeded subsets used for the fallback-reduction experiments.
# Every run uses the same decoding settings (eval defaults + per-sample seed); only the flags under test change.
#
# Usage:
#   RUN=E0 MODE=hybrid bash evaluation/scripts/eval_fallback_subsets.sh [extra inference args...]
#   e.g. RUN=E1 MODE=hybrid bash ... --constrained_block
#        RUN=E2 MODE=hybrid bash ... --lora_path work_dirs/seqkd_cf/llm_lora
# Env: MODEL_PATH, DATA_DIR, OUT_BASE, DATASETS, DTYPE, SEED
set -euo pipefail

RUN=${RUN:?set RUN name}
MODE=${MODE:-hybrid}
MODEL_PATH=${MODEL_PATH:-work_dirs/la3b_eval}
DATA_DIR=${DATA_DIR:-work_dirs/evaldata}
OUT_BASE=${OUT_BASE:-work_dirs/results}
DATASETS=${DATASETS:-"RefCOCOg_val RefCOCOg_test COCO LVIS Dense200 SROIE"}
DTYPE=${DTYPE:-float16}
SEED=${SEED:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for DS in $DATASETS; do
  OUT_DIR="$OUT_BASE/$RUN/$MODE/$DS"
  mkdir -p "$OUT_DIR"
  if [[ -f "$OUT_DIR/answer.jsonl" ]]; then
    echo "skip $OUT_DIR (exists)"; continue
  fi
  python "$EVAL_DIR/inference_grounding_ddp.py" \
    --model_path "$MODEL_PATH" \
    --test_jsonl_path "$DATA_DIR/subsets/$DS.jsonl" \
    --image_root_dir "$DATA_DIR/images" \
    --save_path "$OUT_DIR/answer.jsonl.tmp" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --eval_type box_eval \
    --generation_mode "$MODE" \
    --dtype "$DTYPE" \
    --seed "$SEED" \
    "$@" > "$OUT_DIR/inference_log.txt" 2>&1
  mv "$OUT_DIR/answer.jsonl.tmp" "$OUT_DIR/answer.jsonl"
  python "$EVAL_DIR/metrics/other_metric.py" --data_path "$OUT_DIR/answer.jsonl" \
    --output_path "$OUT_DIR/eval_results.json" > "$OUT_DIR/metric_log.txt" 2>&1
  python "$EVAL_DIR/tools/summarize_fallback.py" --run "$RUN/$MODE=$OUT_DIR/answer.jsonl" \
    --out "$OUT_DIR/summary.json" --wandb_id "eval-$RUN-$MODE-$DS" \
    --wandb_config "{\"experiment\": \"$RUN\", \"mode\": \"$MODE\", \"dataset\": \"$DS\", \"seed\": $SEED, \"dtype\": \"$DTYPE\", \"extra_args\": \"$*\"}" \
    > "$OUT_DIR/summary.md" 2>&1 || echo "summary/wandb logging failed for $OUT_DIR (see summary.md)"
  echo "done $OUT_DIR"
done
