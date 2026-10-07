#!/bin/bash
# 工单 16：ms-swift LoRA 微调 Qwen2.5-VL-3B（修正版）
set -e

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

MODEL="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"
DATASET="./finetune_data/train.jsonl"
OUTPUT_DIR="./output/qwen2.5-vl-3b-lora"

swift sft \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --tuner_type lora \
  --lora_rank 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --max_length 2048 \
  --output_dir "$OUTPUT_DIR" \
  --logging_steps 5 \
  --save_steps 50 \
  --save_total_limit 2 \
  --dataloader_num_workers 2 \
  --gradient_checkpointing true \
  --report_to none
