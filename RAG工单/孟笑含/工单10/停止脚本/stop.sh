#!/bin/bash
# 工单编号：人工智能NLP-RAG-金融问答系统部署
# stop.sh - 停止服务

if [ -f "logs/app.pid" ]; then
    PID=$(cat logs/app.pid)
    echo "正在停止服务 PID: $PID"
    kill $PID 2>/dev/null || echo "进程已停止"
    rm -f logs/app.pid
    echo "✅ 已停止"
else
    echo "⚠️ 未找到 logs/app.pid"
fi
