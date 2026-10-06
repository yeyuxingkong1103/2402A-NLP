#!/usr/bin/env bash
# ============================================================
# 首次部署脚本（WSL Ubuntu / 任意 Linux 环境）
# 流程：环境检查 → 起容器 → 装依赖 → 解析 PDF → 全量入库
# 用法：bash scripts/deploy.sh
# 部署成功后日常启动用：bash scripts/start.sh
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> [0/5] 环境检查"
command -v docker  >/dev/null 2>&1 || { echo "❌ 未安装 Docker，请先安装 Docker Desktop / Docker Engine"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "❌ 未找到 python3，请先安装 Python 3.10+"; exit 1; }
[ -f .env ] || { echo "❌ 未找到 .env，请先执行：cp .env.example .env 并填写 DEEPSEEK_API_KEY"; exit 1; }

echo "==> [1/5] 启动 Milvus / Redis 容器..."
docker compose up -d
echo "    等待基础服务就绪（30 秒）..."
sleep 30

echo "==> [2/5] 安装 Python 依赖..."
python3 -m pip install -r requirements.txt

echo "==> [3/5] 解析指南 PDF（页眉页脚/水印过滤，输出清洗文本）..."
python3 scripts/02_parse_guideline.py

echo "==> [4/5] 切块 + BGE-m3 向量化 + 全量入库 Milvus..."
python3 scripts/03_init_kb.py

echo "==> [5/5] 部署完成 ✅"
echo "    启动应用：bash scripts/start.sh"
echo "    RAG 评测：python3 scripts/05_ragas_eval.py"
