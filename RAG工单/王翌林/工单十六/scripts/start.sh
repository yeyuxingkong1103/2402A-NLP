#!/bin/bash
# 工单十六：启动脚本（一键运行 数据构建 → 微调 → 评估）
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/dabaie/code/my_project/.venv/bin/python3}
mkdir -p data/logs finetune_output docs/screenshots

echo "[$(date)] 工单十六 启动..."
# 1) 构建/刷新微调数据（如缺失）
if [ ! -f data/train.jsonl ]; then
  echo ">> 构建微调数据"
  $PYTHON_BIN scripts/build_finetune_data.py
fi
# 2) 微调
echo ">> 启动 LoRA 微调（3 epoch）"
$PYTHON_BIN src/finetune.py --epochs 3 --lr 1e-4 --batch-size 1 --grad-accum 4 --no-4bit \
  2>&1 | tee data/logs/finetune.log
# 3) 评估
echo ">> 评估 baseline vs finetuned"
$PYTHON_BIN src/evaluate.py --max-eval 18 2>&1 | tee data/logs/evaluate.log
# 4) 生成图表
$PYTHON_BIN scripts/gen_diagrams.py
echo "[$(date)] 工单十六 完成，结果见 docs/screenshots/"
