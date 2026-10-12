#!/bin/bash
# 工单十五：一键初始化（安装 + 构建知识库）
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/dabaie/code/my_project/.venv/bin/python3}

bash scripts/install.sh
echo "=== 构建知识库 ==="
$PYTHON_BIN scripts/build_kb.py --pdf ./附件/CN100347506C.pdf
echo "=== 初始化完成 ==="
