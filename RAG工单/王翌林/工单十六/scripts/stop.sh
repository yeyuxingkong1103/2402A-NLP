#!/bin/bash
# 工单十六：停止脚本（结束微调/评估进程）
echo "停止 工单十六 相关进程..."
pkill -f "src/finetune.py" 2>/dev/null && echo "已停止 finetune"
pkill -f "src/evaluate.py" 2>/dev/null && echo "已停止 evaluate"
sleep 1
# 释放 GPU 显存
PYTHON_BIN=${PYTHON_BIN:-/home/dabaie/code/my_project/.venv/bin/python3}
$PYTHON_BIN -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
nvidia-smi --query-gpu=memory.used,memory.free --format=csv 2>/dev/null || true
echo "停止完成"
