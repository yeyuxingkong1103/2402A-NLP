#!/bin/bash
# 停止脚本：停止API服务

echo "===== 停止服务 ====="

if [ -f app.pid ]; then
    PID=$(cat app.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "API服务已停止（PID: $PID）"
    else
        echo "API服务未在运行"
    fi
    rm -f app.pid
else
    echo "未找到PID文件，尝试杀掉uvicorn进程..."
    pkill -f "uvicorn app:app" || echo "无uvicorn进程"
fi

echo "===== 停止完成 ====="
