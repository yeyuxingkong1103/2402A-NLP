#!/usr/bin/env sh
set -eu

echo "[1/4] 检查Docker引擎..."
docker info >/dev/null

echo "[2/4] 构建并启动金融GraphRAG容器..."
docker compose up -d --build

echo "[3/4] 等待健康并运行问答评测..."
python3 container_check.py --base-url http://127.0.0.1:8021

echo "[4/4] 容器状态："
docker compose ps
echo "部署完成：http://服务器IP:8021/"
