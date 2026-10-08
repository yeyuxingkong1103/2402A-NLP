#!/usr/bin/env bash
# 工单编号：人工智能NLP-RAG-金融问答系统部署
# scripts/stop_docker_v10.sh —— 工单十 停止容器（保留 named volume rag-v10-data，数据不丢）
# 用法：bash scripts/stop_docker_v10.sh [--prune]   # --prune 连带删除数据卷（慎用）
set -e
cd "$(dirname "$0")/.."

if [ "$1" = "--prune" ]; then
    echo "[v10] 停止并删除容器 + 数据卷 rag-v10-data（数据将丢失）..."
    docker compose down -v
else
    echo "[v10] 停止并删除容器（保留数据卷 rag-v10-data / 网络配置可复用）..."
    docker compose down
fi
echo "[v10] done"
