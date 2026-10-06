#!/usr/bin/env bash
# 部署脚本（Ubuntu 22.04 / CentOS 兼容）
# 对应文档.txt 第 45 行：检查服务器配置 -> 安装环境 -> 上传代码 -> 解压 -> 安装依赖
set -euo pipefail

echo "===== 0. 检查服务器配置 ====="
if ! command -v python3 >/dev/null 2>&1; then
  echo "未检测到 python3，请先安装：sudo apt-get install -y python3 python3-pip"
  exit 1
fi
echo "python3: $(python3 --version)"

echo "===== 1. 安装系统依赖（redis/milvus 推荐用 Docker 一键拉起） ====="
if ! command -v docker >/dev/null 2>&1; then
  echo "提示：未检测到 docker。Milvus/Redis 推荐使用 Docker 部署："
  echo "  docker run -d --name redis -p 6379:6379 redis:7"
  echo "  docker run -d --name milvus -p 19530:19530 milvusdb/milvus:v3.0.1"
fi

echo "===== 2. 创建并激活 Python 环境 ====="
if command -v conda >/dev/null 2>&1; then
  conda create -n rag_roleplay python=3.10 -y || true
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate rag_roleplay
else
  echo "未检测到 conda，改用系统 python3 并安装 venv"
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "===== 3. 安装依赖 ====="
pip install -U pip
pip install -r requirements.txt

echo "===== 4. 配置 ====="
if [ ! -f config/config.yaml ]; then
  echo "请根据 config/config.yaml 模板填写 llm.api.api_key、milvus.host 等配置"
fi

echo "===== 5. 构建知识库索引 ====="
python scripts/build_index.py || echo "构建索引失败（请先确认 Milvus 已启动）"

echo "===== 6. 启动服务 ====="
bash scripts/start.sh
