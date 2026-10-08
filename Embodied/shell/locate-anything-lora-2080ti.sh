#!/usr/bin/env bash
# Single pre-Ampere GPU (e.g. RTX 2080 Ti, 11 GB) LoRA fine-tuning of LocateAnything for the
# fallback-reduction experiments (sequence-level self-distillation + dParallel certainty forcing).
#   - sdpa attention (no magi / flash-attn on sm75; the ViT falls back to its sdpa path automatically)
#   - fp16 AMP (no bf16 GEMM on sm75); trainable LoRA params are kept in fp32
#   - LoRA on the LLM only; ViT, MLP projector and base LLM frozen
# Optimizer/schedule follow the dParallel reference config for Dream (Qwen2.5-initialized):
# lr 2e-5, cosine, warmup 5%, weight decay 0.01, grad clip 1, LoRA r=32.
#
# Usage: META_PATH=recipe.json MAX_STEPS=N CF_BETA=1.0 bash shell/locate-anything-lora-2080ti.sh <OUTPUT_DIR>
set -euo pipefail

OUTPUT_DIR=${1:-"work_dirs/locany_lora_2080ti"}
MODEL_PATH=${MODEL_PATH:-"work_dirs/LocateAnything-3B"}
if [[ -z "${META_PATH:-}" ]]; then
  echo "Please set META_PATH to a training recipe json." >&2
  exit 1
fi

MAX_STEPS=${MAX_STEPS:?set MAX_STEPS}
CF_BETA=${CF_BETA:-0.0}
GRADIENT_ACC=${GRADIENT_ACC:-8}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-2048}
USE_LLM_LORA=${USE_LLM_LORA:-32}
PORT=${PORT:-29500}

mkdir -p "$OUTPUT_DIR"
script_name=$(basename "${BASH_SOURCE[0]}")

python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port="$PORT" \
  eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path "$MODEL_PATH" \
  --max_steps "$MAX_STEPS" \
  --output_dir "$OUTPUT_DIR" \
  --meta_path "$META_PATH" \
  --overwrite_output_dir False \
  --block_size 6 \
  --attn_implementation sdpa \
  --causal_attn False \
  --certainty_forcing_beta "$CF_BETA" \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --use_llm_lora "$USE_LLM_LORA" \
  --use_backbone_lora 0 \
  --vision_select_layer -1 \
  --dataloader_num_workers 2 \
  --fp16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "$GRADIENT_ACC" \
  --save_strategy "no" \
  --save_lora_adapter_only True \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --max_grad_norm 1.0 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --sample_log_interval 10 \
  --packing_buffer_size 16 \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --max_num_tokens_per_sample "$MAX_SEQ_LENGTH" \
  --max_num_tokens "$MAX_SEQ_LENGTH" \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --report_to "none" \
  --run_name "$script_name" \
  --use_onelogger False \
  --mlp_connector_layers 2 \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
