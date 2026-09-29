#!/usr/bin/env bash
# 部署 / 安装脚本（Ubuntu / CentOS）
# 用法：bash install.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "================ 0. 检查服务器配置 ================"
OS_ID="$(awk -F= '/^ID=/{gsub(/"/,"",$2);print $2;exit}' /etc/os-release 2>/dev/null || echo unknown)"
echo "系统：$OS_ID  主机：$(hostname)  Python：$(python3 --version 2>/dev/null || echo '未安装')"

if [[ "$OS_ID" == "ubuntu" ]] || [[ "$OS_ID" == "debian" ]]; then
    PKGMGR="apt"
elif [[ "$OS_ID" == "centos" ]] || [[ "$OS_ID" == "rhel" ]] || [[ "$OS_ID" == "fedora" ]]; then
    PKGMGR="yum"
else
    echo "未识别的系统（$OS_ID），按通用 Linux 处理。"
    PKGMGR="apt"
fi

echo "================ 1. 安装系统依赖 ================"
if command -v apt >/dev/null 2>&1; then
    sudo apt update
    sudo apt install -y python3 python3-pip python3-venv unzip git curl
elif command -v yum >/dev/null 2>&1; then
    sudo yum install -y python3 python3-pip unzip git curl
fi

echo "================ 2. 安装 Redis / MySQL / Milvus（按需）==============="
if ! command -v redis-server >/dev/null 2>&1; then
    echo "未检测到 Redis，请手动安装：sudo apt install -y redis-server"
else
    echo "Redis 已安装：$(redis-server --version)"
fi
if ! command -v mysql >/dev/null 2>&1; then
    echo "未检测到 MySQL，请手动安装：sudo apt install -y mysql-server"
else
    echo "MySQL 已安装：$(mysql --version)"
fi
if ! command -v docker >/dev/null 2>&1; then
    echo "未检测到 Docker，Milvus 需通过 Docker 部署。请安装 Docker。"
else
    if ! docker ps --format '{{.Names}}' | grep -q '^milvus$'; then
        echo "启动 Milvus 容器……"
        docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest || true
    else
        echo "Milvus 容器已存在。"
    fi
fi

echo "================ 3. 创建 Conda / venv 环境 ================"
if command -v conda >/dev/null 2>&1; then
    echo "检测到 Conda。"
    if ! conda env list | awk '{print $1}' | grep -q '^roleplay$'; then
        conda create -y -n roleplay python=3.10
    fi
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate roleplay
else
    echo "未检测到 Conda，使用 venv。"
    python3 -m venv .venv
    source .venv/bin/activate
fi

echo "================ 4. 安装 Python 依赖 ================"
python -m pip install --upgrade pip
pip install -r requirements.txt
# MySQL（可选）
read -r -p "是否安装 MySQL 驱动 PyMySQL + cryptography？[y/N] " ans || ans=N
if [[ "$ans" =~ ^[Yy]$ ]]; then
    pip install PyMySQL cryptography
fi

echo "================ 5. 上传 / 拉取代码 ================"
if [[ -d .git ]]; then
    git pull --ff-only || true
else
    echo "如需从远程拉取，请在此处扩展：git clone <repo> ."
fi

echo "================ 6. 配置 .env ================"
if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "已从 .env.example 复制生成 .env，请按需修改。"
else
    echo ".env 已存在，跳过。"
fi

echo "================ 7. 初始化数据库与知识库缓存 ================"
python -c "from database import init_db; init_db()"
python -c "from services import get_retriever; get_retriever()" || true

echo "================ 安装完成 ================"
echo "下一步：bash run.sh  （停止：bash shutdown.sh）"
