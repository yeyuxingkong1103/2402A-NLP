#!/usr/bin/env bash
# RAG 角色扮演系统 - 部署安装脚本
# 步骤：0.检查服务器配置 -> 1.检查 Python 环境 -> 2.创建 venv -> 3.安装依赖
set -euo pipefail
cd "$(dirname "$0")/.."

# 0. 检查操作系统
OS="$(. /etc/os-release 2>/dev/null && echo "$ID" || echo "unknown")"
echo "[1/5] 检测操作系统: $OS"
case "$OS" in
  ubuntu|debian) echo "    ✅ Ubuntu/Debian 系列" ;;
  centos|rhel|rocky|almalinux) echo "    ⚠️  CentOS 系列（如缺 python3-venv 请用 yum 安装）" ;;
  *) echo "    ⚠️  未在 $OS 上测试，继续尝试" ;;
esac

# 1. 检查 Python
echo "[2/5] 检查 Python 环境"
if ! command -v python3 &>/dev/null; then
    echo "❌ 未找到 python3。"
    echo "   Ubuntu/Debian: sudo apt install -y python3 python3-venv python3-pip"
    echo "   CentOS/RHEL  : sudo yum install -y python3 python3-pip"
    exit 1
fi
python3 --version
# 建议 3.10~3.12（milvus-lite 等二进制轮子兼容性最好）
python3 -c 'import sys; sys.exit(0 if (3,9) <= sys.version_info < (3,14) else 1)' \
  && echo "    ✅ Python 版本合适" \
  || echo "    ⚠️  Python 版本偏旧或过新（建议 3.10~3.12），部分依赖可能无预编译轮子"

# 2. 创建虚拟环境
if [[ ! -d ".venv" ]]; then
    echo "[3/5] 创建虚拟环境 .venv"
    python3 -m venv .venv || {
        echo "❌ 创建 venv 失败，Ubuntu 请安装 python3-venv：sudo apt install -y python3-venv"
        exit 1
    }
else
    echo "[3/5] 已存在 .venv，跳过创建"
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 3. 安装依赖
echo "[4/5] 安装依赖（requirements.txt）"
pip install --upgrade pip
pip install -r requirements.txt

# 4. 初始化目录与配置
echo "[5/5] 初始化目录与配置"
mkdir -p data/corpus logs
[[ -f .env ]] || cp .env.example .env

echo ""
echo "✅ 安装完成。后续步骤："
echo "  1) 拉取语料: .venv/bin/python scripts/fetch_datasets.py（公开数据集，约 29MB）"
echo "  2) 编辑 .env 配置 LLM/Embedding（Ollama 或在线 API）"
echo "  3) 入库:   .venv/bin/python scripts/seed.py --reset"
echo "  4) 启动:   bash scripts/run.sh"
