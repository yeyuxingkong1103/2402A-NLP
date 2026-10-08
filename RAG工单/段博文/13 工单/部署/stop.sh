#!/bin/bash
# 工单13：RAG性能瓶颈识别与优化 - 停止脚本

echo "停止 RAG 服务..."
pkill -f "python main.py" 2>/dev/null || true
echo "已停止"
