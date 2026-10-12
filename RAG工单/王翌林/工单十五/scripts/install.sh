#!/bin/bash
# 工单十五：环境安装脚本（conda + cuda）
set -e
cd "$(dirname "$0")/.."

PYTHON_BIN=${PYTHON_BIN:-/home/dabaie/code/my_project/.venv/bin/python3}

echo "=== 工单十五 环境检查 ==="
if [ ! -x "$PYTHON_BIN" ]; then
  echo "未找到 $PYTHON_BIN，尝试创建 venv..."
  python3 -m venv .venv
  PYTHON_BIN=$(pwd)/.venv/bin/python3
fi

echo "使用 Python: $PYTHON_BIN"
$PYTHON_BIN --version

echo "=== 安装依赖 ==="
$PYTHON_BIN -m pip install --upgrade pip
$PYTHON_BIN -m pip install -r requirements.txt

echo "=== 检查 GPU/CUDA ==="
$PYTHON_BIN -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

echo "=== 检查模型 ==="
for m in bge-m3 bge-reranker-v2-m3 Qwen2-VL-2B-Instruct; do
  if [ -d "/home/dabaie/models/$m" ]; then echo "✓ $m"; else echo "✗ $m 缺失"; fi
done

echo "=== 安装完成 ==="
