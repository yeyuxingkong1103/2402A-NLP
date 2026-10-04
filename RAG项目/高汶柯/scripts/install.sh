#!/bin/bash
# 安装脚本：检查系统、安装基础服务与 Python 依赖
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "=== RAG 多角色扮演系统 安装脚本 ==="
echo "项目目录: $PROJECT_DIR"

# 0. 检查服务器配置
if [ -f /etc/os-release ]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  echo "系统: ${PRETTY_NAME}"
else
  echo "警告：未识别的 Linux 发行版，仅支持 Ubuntu/CentOS"
fi
echo "CPU: $(nproc) 核  MEM: $(free -h | awk '/Mem:/{print $2}')"

# 1. 安装基础服务
if command -v apt >/dev/null 2>&1; then
  sudo apt update -y
  sudo apt install -y python3-pip python3-venv redis-server mysql-server curl
  sudo systemctl enable --now redis-server || true
  sudo systemctl enable --now mysql || true
elif command -v yum >/dev/null 2>&1; then
  sudo yum install -y python3-pip redis mysql-server curl
  sudo systemctl enable --now redis || true
  sudo systemctl enable --now mysqld || true
else
  echo "请手动安装 redis / mysql"
fi

# Milvus：推荐用 Docker Compose 启动
if command -v docker >/dev/null 2>&1; then
  echo "提示：请在 Milvus 目录执行 'docker compose up -d' 启动向量库"
else
  echo "警告：未检测到 Docker，Milvus 需另行安装"
fi

# 2. 创建 Python 环境并安装依赖
cd "$PROJECT_DIR"
if [ ! -d venv ]; then
  python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "如需 OCR / MinerU / 多模态解析：pip install -r requirements-ocr.txt"

# 3. 生成配置
[ -f .env ] || cp .env.example .env
echo "请编辑 .env 填写 LLM_API_KEY 与数据库连接信息"

echo "=== 安装完成，执行 scripts/run.sh 启动系统 ==="
