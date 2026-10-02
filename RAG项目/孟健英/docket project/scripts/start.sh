#!/usr/bin/env bash
# ============================================================
# 日常启动脚本（WSL Ubuntu 环境）
# 作用：拉起 Milvus/Redis 容器 → 启动 Streamlit 应用
# 用法：bash scripts/start.sh
# ============================================================
set -euo pipefail

# 切到项目根目录（脚本在 scripts/ 下）
cd "$(dirname "$0")/.."

echo "==> [1/2] 检查基础设施容器（Milvus / Redis）..."
# 已有同名容器在跑就跳过（避免 compose 用自己声明的版本顶替现有容器、丢数据）
if docker ps --format '{{.Names}}' | grep -q '^milvus-standalone$'; then
  echo "    基础设施已在运行，跳过启动"
else
  docker compose up -d
  sleep 15
fi

echo "==> [2/2] 启动 Streamlit，浏览器访问 http://localhost:8501"
# headless：跳过首次运行的邮箱提示；address 0.0.0.0 方便局域网/宿主机访问
exec streamlit run src/main.py \
  --server.address=0.0.0.0 \
  --server.headless=true
