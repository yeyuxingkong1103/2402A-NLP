#!/usr/bin/env bash
# 启动脚本（对应文档.txt 第 46 行）
set -euo pipefail

# 切到项目根目录
cd "$(dirname "$0")/.."

# 依赖检查
if ! command -v docker >/dev/null 2>&1; then
  echo "警告：未检测到 docker，请确保 Milvus(19530) 与 Redis(6379) 已启动"
fi

# 启动服务（debug 由 config.yaml 的 app.debug 控制）
python main.py
