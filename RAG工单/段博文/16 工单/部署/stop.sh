#!/bin/bash
pkill -f "mock_train.py" 2>/dev/null && echo "已停止" || echo "无运行进程"
echo "停止操作完成"
