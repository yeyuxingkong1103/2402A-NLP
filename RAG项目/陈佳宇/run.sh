#!/bin/bash
# 启动脚本：启动Milvus、MySQL、Redis、API服务

echo "===== 启动服务 ====="

# 启动基础服务
echo "[1/3] 启动MySQL和Redis..."
sudo service mysql start
sudo service redis-server start

# 检查Milvus（如果用docker部署）
if command -v docker &> /dev/null; then
    echo "[2/3] 检查Milvus容器..."
    if [ ! "$(docker ps -q -f name=milvus)" ]; then
        echo "Milvus未启动，请先启动Milvus容器"
    fi
fi

# 启动API服务
echo "[3/3] 启动FastAPI服务（端口8000）..."
source venv/bin/activate
nohup uvicorn app:app --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &
echo $! > app.pid
echo "API服务已启动，PID: $(cat app.pid)"
echo "接口文档：http://127.0.0.1:8000/docs"
echo "健康检查：http://127.0.0.1:8000/api/health"

echo "===== 启动完成 ====="
