#!/usr/bin/env bash
# 结束脚本：停止 API + Streamlit
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

for name in api streamlit; do
    pidfile="run/${name}.pid"
    if [[ -f "$pidfile" ]]; then
        pid="$(cat "$pidfile")"
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" && echo "已停止 $name (pid=$pid)"
        else
            echo "$name 进程不存在（pid=$pid）"
        fi
        rm -f "$pidfile"
    else
        echo "未找到 $name pid 文件，跳过。"
    fi
done

# 兜底：按端口杀
for port in 8000 8501; do
    pids="$(lsof -t -i:"$port" 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
        echo "$port 仍占用，强杀：$pids"
        kill -9 $pids 2>/dev/null || true
    fi
done

echo "停止完成。"
