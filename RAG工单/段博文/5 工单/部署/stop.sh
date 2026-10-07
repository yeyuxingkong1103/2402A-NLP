#!/bin/bash
# 工单编号：人工智能NLP-RAG-Query理解优化任务
# stop.sh - 停止脚本

echo "=========================================="
echo "  RAG-PDF 问答系统 - 停止服务"
echo "=========================================="

# 查找并停止 uvicorn/main.py 进程
pkill -f "python main.py" || echo "未找到运行中的服务进程"

echo "服务已停止"
