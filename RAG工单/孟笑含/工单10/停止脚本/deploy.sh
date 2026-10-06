#!/bin/bash
# 工单编号：人工智能NLP-RAG-金融问答系统部署
# deploy.sh - 手动部署脚本

set -e

echo "=========================================="
echo "金融问答系统 - 手动部署"
echo "=========================================="

PROJECT_DIR=$(cd $(dirname $0) && pwd)
cd $PROJECT_DIR

mkdir -p logs models data

echo "[2/6] 创建虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo "[3/6] 安装依赖..."
pip install -q --upgrade pip
pip install -q -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

echo "[4/6] 设置环境变量..."
export HF_HOME=$PROJECT_DIR/models
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

echo "[5/6] 启动服务..."
nohup python3 app.py > logs/app.log 2>&1 &
APP_PID=$!
echo "服务 PID: $APP_PID"
echo $APP_PID > logs/app.pid

echo "[6/6] 健康检查（等待 30 秒）..."
sleep 30
if curl -s http://localhost:6006/ > /dev/null 2>&1; then
    echo "✅ 服务运行正常：http://localhost:6006"
else
    echo "⚠️ 服务可能未启动，请查看 logs/app.log"
fi

echo "=========================================="
echo "部署完成！访问地址：http://localhost:6006"
echo "=========================================="
