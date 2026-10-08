#!/usr/bin/env bash
# ============================================================
# 工单编号：人工智能NLP-RAG-LightRAG优化
# 环境安装脚本（Linux / WSL2 Ubuntu）
# 作用：创建 conda 环境、安装 LightRAG 与全部依赖、准备本地模型
# ============================================================
set -e

echo "========== [1/4] 创建 conda 环境 rag_lightrag =========="
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda create -n rag_lightrag python=3.12 -y || true
conda activate rag_lightrag

echo "========== [2/4] 安装依赖 =========="
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "========== [3/4] 环境变量检查 =========="
: "${deepseek_api_key1:?请先设置 deepseek_api_key1}"
: "${DEEPSEEK_BASE_URL:=https://api.deepseek.com}"
export DEEPSEEK_BASE_URL
echo "LLM: deepseek-chat @ ${DEEPSEEK_BASE_URL}"

echo "========== [4/4] 本地模型检查 =========="
if [ ! -d "/data/models/bge-m3" ]; then
  echo "[提示] 请将 bge-m3 模型放置于 /data/models/bge-m3，"
  echo "       或修改 研发/src/config.py 中 EMBED_MODEL_PATH 指向本地 bge-m3 目录。"
  echo "       下载：huggingface-cli download BAAI/bge-m3 --local-dir /data/models/bge-m3"
fi

echo "安装完成。执行 ./start.sh 开始运行。"
