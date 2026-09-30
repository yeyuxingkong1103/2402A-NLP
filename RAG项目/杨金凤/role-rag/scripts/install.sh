#!/usr/bin/env bash
# 一键部署：检查系统/Python/外部依赖，创建 venv 并安装依赖。
# 外部依赖（Redis/MySQL/Milvus）只检查、只提示，不替用户安装（系统服务由用户自行部署）。
set -euo pipefail

cd "$(dirname "$0")/.."   # 切到项目根目录

# 版本比较：ver_ge A B => A >= B
ver_ge() { [ "$(printf '%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]; }

echo "==> 1/7 检查系统（仅支持 Ubuntu 20.04+）"
if [ ! -f /etc/os-release ]; then
  echo "错误：无法读取 /etc/os-release，仅支持 Ubuntu。"; exit 1
fi
. /etc/os-release
if [ "${ID:-}" != "ubuntu" ]; then
  echo "错误：当前系统不是 Ubuntu（ID=${ID:-未知}），本脚本仅支持 Ubuntu。"; exit 1
fi
version_id="${VERSION_ID%\"}"; version_id="${version_id#\"}"   # 去掉可能的引号
if ! ver_ge "$version_id" "20.04"; then
  echo "错误：Ubuntu 版本过低（$version_id），需 >= 20.04。"; exit 1
fi
echo "  [OK] Ubuntu $version_id"

echo "==> 2/7 检查 Python（3.10+）"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "错误：未找到 $PYTHON_BIN，请先安装 Python 3.10+。"; exit 1
fi
py_ver="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if ! ver_ge "$py_ver" "3.10"; then
  echo "错误：Python 版本 $py_ver < 3.10，请升级（如 apt install python3.10 或编译安装）。"; exit 1
fi
echo "  [OK] Python $py_ver（$PYTHON_BIN）"

echo "==> 3/7 检查外部依赖（缺失仅提示，不中断）"
if command -v redis-cli >/dev/null 2>&1 && redis-cli ping >/dev/null 2>&1; then
  echo "  [OK] Redis 已运行"
else
  echo "  [提示] Redis 不可用：sudo apt install redis-server && sudo systemctl enable --now redis-server"
fi
if command -v mysql >/dev/null 2>&1; then
  echo "  [OK] MySQL 客户端已安装"
else
  echo "  [提示] MySQL 未安装：sudo apt install mysql-server"
fi
milvus_uri=$(grep -E '^MILVUS_URI=' .env 2>/dev/null | cut -d= -f2- || true)
milvus_uri="${milvus_uri:-http://127.0.0.1:19530}"
if curl -sf "$milvus_uri/healthz" >/dev/null 2>&1; then
  echo "  [OK] Milvus 已运行（$milvus_uri）"
else
  echo "  [提示] Milvus 不可用（$milvus_uri）：需用 Docker 部署，或 .env 设 USE_MILVUS=false 降级 Chroma"
fi

echo "==> 4/7 创建虚拟环境"
if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
  echo "  已创建 .venv"
else
  echo "  .venv 已存在，跳过"
fi
PIP=".venv/bin/pip"

echo "==> 5/7 安装依赖（CPU 版 torch 优先，避免拉 ~2GB 无用的 CUDA 包）"
"$PIP" install --upgrade pip >/dev/null
if .venv/bin/python -c "import torch" >/dev/null 2>&1; then
  echo "  torch 已安装，跳过"
else
  "$PIP" install torch --index-url https://download.pytorch.org/whl/cpu
fi
"$PIP" install -r requirements.txt
echo "  依赖安装完成"

echo "==> 6/7 检查 .env"
if [ ! -f .env ]; then
  echo "  [提示] 未找到 .env：请 cp .env.example .env 并填写必填项（DEEPSEEK_API_KEY1、MYSQL_URL 等）"
else
  echo "  .env 已存在"
fi

echo "==> 7/7 检查数据目录 data/raw/"
if [ -d data/raw ] && [ -n "$(ls -A data/raw 2>/dev/null)" ]; then
  echo "  data/raw/ 已存在且非空"
else
  mkdir -p data/raw
  echo "  [提示] 已创建 data/raw/，请放入 PDF 文档"
fi

echo ""
echo "安装完成。后续步骤："
echo "  1) 初始化 MySQL：.venv/bin/python init_db.py"
echo "  2) 构建知识库：.venv/bin/python ingest.py"
echo "  3) 启动服务：bash scripts/run.sh"
