#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
pkill -f "crossmodal_rag.py" 2>/dev/null && echo "已停止" || echo "无运行进程"
echo "停止操作完成"
