#!/bin/bash
# 工单十六：启动 LoRA 微调
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/dabaie/code/my_project/.venv/bin/python3}

mkdir -p finetune_output data/logs

echo "=== 工单十六：启动 Qwen2-VL LoRA 微调 ==="
$PYTHON_BIN src/finetune.py \
  --epochs 1 \
  --lr 1e-4 \
  --batch-size 1 \
  --grad-accum 4 \
  --no-4bit \
  2>&1 | tee data/logs/finetune.log

echo "=== 微调结束 ==="
