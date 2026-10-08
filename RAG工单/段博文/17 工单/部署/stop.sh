#!/bin/bash
pkill -f "benchmark.py" 2>/dev/null && echo "已停止" || echo "无运行进程"
pkill -f "api_server.py" 2>/dev/null
echo "停止操作完成"
