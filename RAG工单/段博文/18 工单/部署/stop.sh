#!/bin/bash
# 文档质检 API 服务停止脚本
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -f "$SCRIPT_DIR/api_server.pid" ]; then
    PID="$(cat "$SCRIPT_DIR/api_server.pid")"
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo "已停止服务 PID: $PID"
    else
        echo "进程 $PID 已不在运行"
    fi
    rm -f "$SCRIPT_DIR/api_server.pid"
else
    echo "未找到 PID 文件，尝试按端口清理..."
    pkill -f "api_server.py" 2>/dev/null && echo "已清理" || echo "无运行中的服务"
fi
